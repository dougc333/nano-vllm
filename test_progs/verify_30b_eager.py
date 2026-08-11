#!/usr/bin/env python3
"""Real Qwen3-30B-A3B verification on GPU: nano-vllm (eager SDPA) vs transformers.

Loads the REAL checkpoint through nano-vllm's loader + eager torch SDPA
attention (no flash-attn), runs a short prefill, saves the last-token logits,
frees the GPU, then loads the same checkpoint in transformers and compares —
sequentially, since each needs ~61 GB.

This is the eager analogue of verify_gpu_moe_eager.py, at real scale. It proves
the 48-layer / 128-expert MoE port matches transformers on the actual weights.

Usage (CUDA box, inside the repo, flash-attn NOT required):
    MODEL_DIR=/path/to/Qwen3-30B-A3B python test_progs/verify_30b_eager.py
    # default MODEL_DIR=/tmp/Qwen3-30B-A3B (a symlink to the Drive model is fine)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
torch.compile = lambda fn, **kwargs: fn

import types
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
dist.init_process_group("gloo", rank=0, world_size=1)

# Eager import of Qwen3MoeForCausalLM BEFORE the flash_attn shim: this pulls in
# modeling_qwen3_moe.py (and its flash-attn availability check) while flash_attn
# is still absent from sys.modules. Lazy AutoModelForCausalLM would import it
# AFTER the shim is installed and crash with "flash_attn.__spec__ is None".
from transformers import AutoConfig, Qwen3MoeForCausalLM

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/Qwen3-30B-A3B")
if not os.path.isdir(MODEL_DIR) or not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    sys.exit(f"Model dir not found: {MODEL_DIR} (set MODEL_DIR or symlink it to the Drive model)")
DEV = "cuda"
assert torch.cuda.is_available(), "no CUDA device"

# ── shim flash_attn (not installed); triton only if absent (GPU has real one) ─
_flash = types.ModuleType("flash_attn")
_flash.flash_attn_varlen_func = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError())
_flash.flash_attn_with_kvcache = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError())
sys.modules["flash_attn"] = _flash
try:
    import triton  # noqa: F401  real one present — leave it intact
except ImportError:
    _lang = types.ModuleType("triton.language")
    _lang.dtype = type("dtype", (), {})
    _lang.constexpr = type("constexpr", (), {})
    _lang.float32 = _lang.dtype(); _lang.float16 = _lang.dtype()
    _lang.int32 = _lang.dtype(); _lang.int64 = _lang.dtype()
    _triton = types.ModuleType("triton")
    _triton.jit = lambda fn: fn; _triton.kernel = lambda fn: fn; _triton.language = _lang
    sys.modules["triton.language"] = _lang; sys.modules["triton"] = _triton

from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM as NvModel
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context

config = AutoConfig.from_pretrained(MODEL_DIR)
T = 8
PROMPT = [7, 42, 99, 3, 17, 88, 5, 31]
OUT = "/tmp/nv30b_logits.pt"

free = torch.cuda.mem_get_info(DEV)[0] / 1e9
print(f"GPU free: {free:.0f} GB | model dir: {MODEL_DIR} | dtype: {config.torch_dtype}")
if free < 70:
    print("WARNING: <70GB free — the 61GB model + overhead may OOM.")

# ── eager SDPA attention (single sequence, same math as transformers eager) ──
def eager_attn(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    if not self.qkv_bias:
        q = self.q_norm(q); k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)
    n_rep = self.num_heads // self.num_kv_heads
    def rk(x):
        if n_rep == 1:
            return x
        nk, t, d = x.shape
        return x[:, None, :, :].expand(nk, n_rep, t, d).reshape(nk * n_rep, t, d)
    q, k, v = q.transpose(0, 1), rk(k.transpose(0, 1)), rk(v.transpose(0, 1))
    attn = (q @ k.transpose(-1, -2)) * self.scaling
    t = q.size(-2)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    attn = attn.masked_fill(causal, float("-inf"))
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    o = (attn @ v).transpose(0, 1).contiguous()
    return self.o_proj(o.flatten(1, -1))

Qwen3Attention.forward = eager_attn

# ══ PHASE 1: nano-vllm ══════════════════════════════════════════════════════
print("\n=== nano-vllm: building + loading 30B-A3B (bf16) ===")
torch.set_default_dtype(torch.bfloat16)   # nanovllm builds params in bf16
with torch.device(DEV):
    nv = NvModel(config)
load_model(nv, MODEL_DIR)
nv.cuda().eval()

ids = torch.tensor(PROMPT, device=DEV)
pos = torch.arange(T, device=DEV)
with torch.no_grad():
    set_context(True,
                cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32, device=DEV),
                cu_seqlens_k=torch.tensor([0, T], dtype=torch.int32, device=DEV),
                max_seqlen_q=T, max_seqlen_k=T,
                slot_mapping=torch.arange(T, dtype=torch.int32, device=DEV),
                block_tables=None)
    print("  forward (48 layers x 128 experts, eager)...")
    hidden = nv(ids, pos)
    nv_logits = nv.compute_logits(hidden).float().cpu()      # [1, V]
    reset_context()
    # router info for layer 0 and last, informational
    for li in (0, len(nv.model.layers) - 1):
        w, idx = nv.model.layers[li].mlp.gate(hidden.reshape(-1, hidden.shape[-1]))
        print(f"  nv layer{li} router top-{idx.shape[-1]} (token0): {idx[0].tolist()}")
torch.save(nv_logits, OUT)
print(f"  nv last-token logits saved ({nv_logits.shape})")
del nv, hidden, nv_logits
torch.cuda.empty_cache()
import gc; gc.collect()

# ══ PHASE 2: transformers ═══════════════════════════════════════════════════
print("\n=== transformers: loading 30B-A3B (bf16) ===")
hf = Qwen3MoeForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
                                         device_map="auto", low_cpu_mem_usage=True).eval()
with torch.no_grad():
    hf_logits = hf(torch.tensor(PROMPT, device=DEV).unsqueeze(0), use_cache=False).logits
hf_logits = hf_logits[0, -1].float().cpu()
del hf
torch.cuda.empty_cache()

# ══ COMPARE ═════════════════════════════════════════════════════════════════
nv_logits = torch.load(OUT, map_location="cpu")
d = (hf_logits - nv_logits).abs()
amax = hf_logits.argmax().item() == nv_logits.argmax().item()
print("\n" + "=" * 70)
print(f"  max |diff| over {nv_logits.numel()} logits: {d.max().item():.4f}")
print(f"  mean|diff|:                  {d.mean().item():.4f}")
print(f"  argmax (predicted token) identical: {amax}")
print(f"  nv pred: {nv_logits.argmax().item()}  hf pred: {hf_logits.argmax().item()}")
print("=" * 70)
# bf16 tolerance: same weights, slightly different op ordering -> ~1e-2; argmax
# agreement is the strong signal (generation-identical).
threshold = 0.1
passed = amax and d.max().item() < threshold
print(f"  RESULT: {'PASS ✅ (real 30B-A3B matches transformers on GPU)' if passed else 'FAIL ❌'}")
print("=" * 70)
os.remove(OUT)
sys.exit(0 if passed else 1)
