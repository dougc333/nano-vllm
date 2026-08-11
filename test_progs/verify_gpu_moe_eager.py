#!/usr/bin/env python3
"""GPU eager verification: nano-vllm Qwen3MoeForCausalLM vs transformers.

Identical to match_qwen3_moe.py but runs on the GPU with torch's eager SDPA
attention (no flash-attn required). Proves the MoE forward — cuBLAS linear
layers, per-expert routing, GQA, RoPE, RMSNorm — matches transformers on real
GPU kernels. Use when flash-attn isn't available; complements verify_gpu_moe.py
(which exercises the full engine + flash-attn path).

Run inside the repo on a CUDA machine:
    python test_progs/verify_gpu_moe_eager.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── 1. torch + transformers first (no fake triton/flash_attn visible) ──
import torch
torch.compile = lambda fn, **kwargs: fn

import types
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
dist.init_process_group("gloo", rank=0, world_size=1)

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

# ── 2. Shims for nanovllm's GPU-only imports (kernels never run — eager) ──
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

dev = "cuda"
assert torch.cuda.is_available(), "no CUDA device"

# ── 3. Tiny Qwen3-MoE config (mirrors Qwen3-30B-A3B structure) ─────────────
VOCAB, H, L, HEADS, KV_HEADS, HEAD_DIM = 128, 64, 2, 8, 2, 16
MOE_INTER, NUM_EXPERTS, TOP_K = 32, 8, 2
T = 16

config = Qwen3MoeConfig(
    vocab_size=VOCAB, hidden_size=H, intermediate_size=64, num_hidden_layers=L,
    num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
    hidden_act="silu", max_position_embeddings=128, initializer_range=0.02,
    rms_norm_eps=1e-6, use_cache=False, tie_word_embeddings=False,
    rope_theta=10000.0, attention_bias=False, attention_dropout=0.0,
    decoder_sparse_step=1, mlp_only_layers=[], moe_intermediate_size=MOE_INTER,
    num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K, norm_topk_prob=True,
    output_router_logits=False, router_aux_loss_coef=0.001, torch_dtype="float32",
)
config._attn_implementation = "eager"

print("=" * 70)
print("  STEP 1: transformers reference model (random weights, fp32, GPU)")
print("=" * 70)
torch.manual_seed(0)
hf_model = Qwen3MoeForCausalLM(config).eval().cuda()
print(f"  transformers Qwen3MoeForCausalLM on {dev}")

# Write the checkpoint in the REAL hub layout (per-expert gate/up/down names).
SAVE_DIR = "/tmp/nanovllm_moe_gpu"
os.makedirs(SAVE_DIR, exist_ok=True)
import re
import safetensors.torch as st

hub_sd = {}
for name, tensor in hf_model.cpu().state_dict().items():
    m = re.fullmatch(r"(.*mlp\.experts)\.(gate_up_proj|down_proj)", name)
    if m is None:
        hub_sd[name] = tensor
        continue
    prefix, proj = m.groups()
    if proj == "gate_up_proj":
        e, two_i, h = tensor.shape
        for e_i in range(e):
            hub_sd[f"{prefix}.{e_i}.gate_proj.weight"] = tensor[e_i, :two_i // 2].contiguous()
            hub_sd[f"{prefix}.{e_i}.up_proj.weight"] = tensor[e_i, two_i // 2:].contiguous()
    else:
        e, h, i = tensor.shape
        for e_i in range(e):
            hub_sd[f"{prefix}.{e_i}.down_proj.weight"] = tensor[e_i].contiguous()
st.save_file(hub_sd, f"{SAVE_DIR}/model.safetensors")
hf_model.cuda()

# ── 4. nano-vllm model + real weight loader, on GPU ─────────────────────────
print("=" * 70)
print("  STEP 2: nano-vllm Qwen3MoeForCausalLM + load_model (GPU)")
print("=" * 70)
nv_model = NvQwen3MoeForCausalLM(config)
load_model(nv_model, SAVE_DIR)
nv_model.cuda().eval()

# ── 5. Eager GPU attention (same math as transformers eager SDPA) ───────────
def eager_attn_forward(self, positions, hidden_states):
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
        if n_rep == 1:
            return x
        n_kv, t, d = x.shape
        return x[:, None, :, :].expand(n_kv, n_rep, t, d).reshape(n_kv * n_rep, t, d)

    q, k, v = q.transpose(0, 1), repeat_kv(k.transpose(0, 1)), repeat_kv(v.transpose(0, 1))
    attn = (q @ k.transpose(-1, -2)) * self.scaling
    t = q.size(-2)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    attn = attn.masked_fill(causal, float("-inf"))
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    o = (attn @ v).transpose(0, 1).contiguous()
    return self.o_proj(o.flatten(1, -1))

Qwen3Attention.forward = eager_attn_forward

# ── 6. End-to-end logits on GPU ─────────────────────────────────────────────
print("=" * 70)
print("  STEP 3: end-to-end logits (same input, both models, GPU)")
print("=" * 70)
torch.manual_seed(1)
input_ids = torch.randint(0, VOCAB, (T,), device=dev)
positions = torch.arange(T, device=dev)

with torch.no_grad():
    hf_logits = hf_model(input_ids.unsqueeze(0), use_cache=False).logits
    set_context(True,
                cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32, device=dev),
                cu_seqlens_k=torch.tensor([0, T], dtype=torch.int32, device=dev),
                max_seqlen_q=T, max_seqlen_k=T,
                slot_mapping=torch.arange(T, dtype=torch.int32, device=dev),
                block_tables=None)
    hidden = nv_model(input_ids, positions)
    nv_logits = nv_model.compute_logits(hidden)
    reset_context()

print(f"  input_ids: {input_ids.tolist()}")
print(f"  hf logits  (last token): {hf_logits[0, -1, :5].tolist()}")
print(f"  nv logits  (last token): {nv_logits[0, :5].tolist()}")

d_full = (hf_logits[0, -1] - nv_logits[0]).abs()
print(f"\n  GPU max |diff| over {VOCAB} logits: {d_full.max().item():.3e}")
print(f"  GPU mean|diff|:                  {d_full.mean().item():.3e}")

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
passed = d_full.max().item() < threshold and (idx == hf_topk).all().item()
print("\n" + "=" * 70)
print(f"  RESULT: {'PASS ✅  (nano-vllm MoE matches transformers on GPU)' if passed else 'FAIL ❌'}")
print("=" * 70)
sys.exit(0 if passed else 1)
