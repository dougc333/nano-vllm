#!/usr/bin/env python3
"""CPU correctness test: nano-vllm Qwen3MoeForCausalLM vs transformers.

Builds a tiny random Qwen3-MoE model in transformers (eager attention), saves
it as safetensors, loads it into nano-vllm's Qwen3MoeForCausalLM through the
real weight loader, then compares router selection, expert weights, and the
final logits for identical input.

Run with a clean env (the Hermes venv on PYTHONPATH breaks numpy):
    /Users/dc/.pyenv/versions/3.13.14/bin/python -E test_progs/match_qwen3_moe.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── 1. torch + transformers first, with NO fake triton/flash_attn visible ──
# (torch._dynamo/_inductor gracefully skip triton when it's absent, but crash
# on an incomplete fake. nanovllm's plain `import triton` / `from flash_attn
# import ...` is satisfied afterwards via sys.modules shims.)
import torch
torch.compile = lambda fn, **kwargs: fn          # skip torch.compile on CPU

import types
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
dist.init_process_group("gloo", rank=0, world_size=1)

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

# ── 2. Shims for nanovllm's GPU-only imports (kernels never run on CPU) ─────
_flash = types.ModuleType("flash_attn")
_flash.flash_attn_varlen_func = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError())
_flash.flash_attn_with_kvcache = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError())
sys.modules["flash_attn"] = _flash

_lang = types.ModuleType("triton.language")
_lang.dtype = type("dtype", (), {})
_lang.constexpr = type("constexpr", (), {})
_lang.float32 = _lang.dtype()
_lang.float16 = _lang.dtype()
_lang.int32 = _lang.dtype()
_lang.int64 = _lang.dtype()
_triton = types.ModuleType("triton")
_triton.jit = lambda fn: fn
_triton.kernel = lambda fn: fn
_triton.language = _lang
sys.modules["triton.language"] = _lang
sys.modules["triton"] = _triton

from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM as NvQwen3MoeForCausalLM
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context

# ── 2. Tiny Qwen3-MoE config (mirrors Qwen3-30B-A3B structure) ──────────────
VOCAB, H, L, HEADS, KV_HEADS, HEAD_DIM = 128, 64, 2, 8, 2, 16
MOE_INTER, NUM_EXPERTS, TOP_K = 32, 8, 2
T = 16  # sequence length

config = Qwen3MoeConfig(
    vocab_size=VOCAB,
    hidden_size=H,
    intermediate_size=64,          # dense fallback (unused: every layer is MoE)
    num_hidden_layers=L,
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    hidden_act="silu",
    max_position_embeddings=128,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=False,
    tie_word_embeddings=False,     # matches 30B-A3B
    rope_theta=10000.0,
    attention_bias=False,
    attention_dropout=0.0,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    moe_intermediate_size=MOE_INTER,
    num_experts=NUM_EXPERTS,
    num_experts_per_tok=TOP_K,
    norm_topk_prob=True,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    torch_dtype="float32",
)
config._attn_implementation = "eager"

print("=" * 70)
print("  STEP 1: transformers reference model (random weights)")
print("=" * 70)
torch.manual_seed(0)
hf_model = Qwen3MoeForCausalLM(config)
hf_model.eval()
n_params = sum(p.numel() for p in hf_model.parameters())
print(f"  transformers Qwen3MoeForCausalLM: {n_params:,} params")
print(f"  expert tensors: {NUM_EXPERTS} x [{MOE_INTER} -> {H}] (top-{TOP_K} routing)")

SAVE_DIR = "/tmp/nanovllm_moe_test"
os.makedirs(SAVE_DIR, exist_ok=True)
# Write the checkpoint in the REAL hub layout (verified against
# Qwen/Qwen3-30B-A3B): per-expert gate_proj/up_proj/down_proj names.
# (transformers 5.x keeps experts packed as one tensor internally; the hub
# checkpoint and vLLM both use the per-expert layout.)
import re
import safetensors.torch as st

hub_sd = {}
for name, tensor in hf_model.state_dict().items():
    m = re.fullmatch(r"(.*mlp\.experts)\.(gate_up_proj|down_proj)", name)
    if m is None:
        hub_sd[name] = tensor
        continue
    prefix, proj = m.groups()
    if proj == "gate_up_proj":                       # [E, 2I, H] -> per-expert gate/up
        e, two_i, h = tensor.shape
        for e_i in range(e):
            hub_sd[f"{prefix}.{e_i}.gate_proj.weight"] = tensor[e_i, :two_i // 2].contiguous()
            hub_sd[f"{prefix}.{e_i}.up_proj.weight"] = tensor[e_i, two_i // 2:].contiguous()
    else:                                            # [E, H, I] -> per-expert down
        e, h, i = tensor.shape
        for e_i in range(e):
            hub_sd[f"{prefix}.{e_i}.down_proj.weight"] = tensor[e_i].contiguous()
st.save_file(hub_sd, f"{SAVE_DIR}/model.safetensors")
print(f"  saved {len(hub_sd)} tensors (hub layout) to {SAVE_DIR}")

# ── 3. nano-vllm model + real weight loader ─────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2: nano-vllm Qwen3MoeForCausalLM + load_model")
print("=" * 70)
nv_model = NvQwen3MoeForCausalLM(config)
load_model(nv_model, SAVE_DIR)
nv_model.eval()

# ── 4. Spot-check loaded weights (router, experts, qkv) ─────────────────────
print("\n" + "=" * 70)
print("  STEP 3: weight-loading spot checks")
print("=" * 70)
hf_sd = hf_model.state_dict()
exp = nv_model.model.layers[0].mlp.experts

def chk(name, a, b):
    d = (a - b).abs().max().item()
    ok = "OK " if d < 1e-6 else "FAIL"
    print(f"  [{ok}] {name:55s} max|diff| = {d:.3e}")
    return d < 1e-6

all_ok = True
all_ok &= chk("layers.0.mlp.gate.weight (router)",
              nv_model.model.layers[0].mlp.gate.weight, hf_sd["model.layers.0.mlp.gate.weight"])
for i in (0, 3, 7):
    all_ok &= chk(f"experts.{i}.gate_proj -> gate_up slice",
                  exp.gate_up[i, :MOE_INTER],
                  hf_sd["model.layers.0.mlp.experts.gate_up_proj"][i, :MOE_INTER])
    all_ok &= chk(f"experts.{i}.up_proj   -> gate_up slice",
                  exp.gate_up[i, MOE_INTER:],
                  hf_sd["model.layers.0.mlp.experts.gate_up_proj"][i, MOE_INTER:])
    all_ok &= chk(f"experts.{i}.down_proj",
                  exp.down[i], hf_sd["model.layers.0.mlp.experts.down_proj"][i])
qkv = nv_model.model.layers[0].self_attn.qkv_proj.weight
q_size = HEADS * HEAD_DIM
all_ok &= chk("q_proj -> qkv slice", qkv[:q_size], hf_sd["model.layers.0.self_attn.q_proj.weight"])
all_ok &= chk("k_proj -> qkv slice", qkv[q_size:q_size + KV_HEADS * HEAD_DIM],
              hf_sd["model.layers.0.self_attn.k_proj.weight"])
all_ok &= chk("tie_word_embeddings=False: lm_head loaded", nv_model.lm_head.weight, hf_sd["lm_head.weight"])
print(f"\n  -> weight loading: {'ALL OK' if all_ok else 'MISMATCHES FOUND'}")

# ── 5. End-to-end forward: replace flash-attn with eager CPU attention ──────
def eager_attn_forward(self, positions, hidden_states):
    """Plain causal SDPA (single sequence) — same math as transformers eager."""
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    if not self.qkv_bias:
        q = self.q_norm(q)
        k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)
    n_rep = self.num_heads // self.num_kv_heads
    def repeat_kv(x):
        # x: [KV, T, D] -> [KV*n_rep, T, D], heads interleaved like
        # torch.repeat_interleave(x, dim=1) on [B, KV, T, D]
        if n_rep == 1:
            return x
        n_kv, t, d = x.shape
        return x[:, None, :, :].expand(n_kv, n_rep, t, d).reshape(n_kv * n_rep, t, d)
    q, k, v = q.transpose(0, 1), repeat_kv(k.transpose(0, 1)), repeat_kv(v.transpose(0, 1))
    attn = (q @ k.transpose(-1, -2)) * self.scaling
    t = q.size(-2)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(causal, float("-inf"))
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    o = (attn @ v).transpose(0, 1).contiguous()
    return self.o_proj(o.flatten(1, -1))

Qwen3Attention.forward = eager_attn_forward

print("\n" + "=" * 70)
print("  STEP 4: end-to-end logits (same input, both models)")
print("=" * 70)
torch.manual_seed(1)
input_ids = torch.randint(0, VOCAB, (T,))                       # flat tokens, like the runner
positions = torch.arange(T)

with torch.no_grad():
    hf_logits = hf_model(input_ids.unsqueeze(0), use_cache=False).logits   # [1, T, V]

    set_context(True,
                cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, T], dtype=torch.int32),
                max_seqlen_q=T, max_seqlen_k=T,
                slot_mapping=torch.arange(T, dtype=torch.int32),
                block_tables=None)
    hidden = nv_model(input_ids, positions)
    nv_logits = nv_model.compute_logits(hidden)                      # [1, V] (last token)
    reset_context()

print(f"  input_ids: {input_ids.tolist()}")
print(f"  hf logits  (last token): {hf_logits[0, -1, :5].tolist()}")
print(f"  nv logits  (last token): {nv_logits[0, :5].tolist()}")

d_full = (hf_logits[0, -1] - nv_logits[0]).abs()
print(f"\n  max |diff| over {VOCAB} logits: {d_full.max().item():.3e}")
print(f"  mean|diff|:                  {d_full.mean().item():.3e}")

# router agreement: top-k expert picks for layer 0
with torch.no_grad():
    flat = hidden.reshape(-1, H)
    w, idx = nv_model.model.layers[0].mlp.gate(flat)
hf_router_logits = hf_model.model.layers[0].mlp.gate.weight @ flat.T
hf_probs = torch.softmax(hf_router_logits.float(), dim=0)
hf_topk = hf_probs.topk(TOP_K, dim=0).indices.T
print(f"  nv router top-k (token 0): {idx[0].tolist()}")
print(f"  hf router top-k (token 0): {hf_topk[0].tolist()}")
print(f"  router picks identical:    {(idx == hf_topk).all().item()}")

threshold = 1e-4
passed = all_ok and d_full.max().item() < threshold and (idx == hf_topk).all().item()
print("\n" + "=" * 70)
print(f"  RESULT: {'PASS ✅  (nano-vllm MoE matches transformers)' if passed else 'FAIL ❌'}")
print("=" * 70)
sys.exit(0 if passed else 1)
