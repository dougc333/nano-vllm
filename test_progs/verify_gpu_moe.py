#!/usr/bin/env python3
"""GPU verification: nano-vllm Qwen3-MoE end-to-end vs transformers.

Runs the REAL nano-vllm engine (ModelRunner -> flash-attn varlen prefill +
KV-cache decode, scheduler, sampler) on CUDA and compares generated tokens
against a transformers reference with identical weights.

The CPU harness (match_qwen3_moe.py) already proves the math in fp32; this
proves the flash-attn / GQA / engine integration paths that CPU can't reach.

Run on Colab (A100):
    !git clone https://github.com/dougc333/nano-vllm  # or upload the repo
    %cd nano-vllm
    !pip install -q transformers safetensors flash-attn accelerate
    !python test_progs/verify_gpu_moe.py

Optional: verify the REAL Qwen3-30B-A3B checkpoint (needs ~80 GB GPU + ~70 GB
disk; downloads 61 GB). Not run by default:
    QWEN3_MOE_REAL=1 python test_progs/verify_gpu_moe.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

assert torch.cuda.is_available(), "CUDA required — run on a GPU runtime (Colab A100)"
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")

try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.exit("flash-attn not installed. Run: !pip install -q flash-attn (or build: "
             "!pip install flash-attn --no-build-isolation)")

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, AutoTokenizer
import safetensors.torch as st

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

MODEL_DIR = "/tmp/moe_gpu_test"

# ── 1. Build a tiny Qwen3-MoE (bf16, like production) + hub-layout checkpoint ──
VOCAB, H, L, HEADS, KV_HEADS, HEAD_DIM = 128, 64, 2, 8, 2, 16
MOE_INTER, NUM_EXPERTS, TOP_K = 32, 8, 2

config = Qwen3MoeConfig(
    vocab_size=VOCAB, hidden_size=H, intermediate_size=64, num_hidden_layers=L,
    num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
    hidden_act="silu", max_position_embeddings=256, initializer_range=0.02,
    rms_norm_eps=1e-6, use_cache=True, tie_word_embeddings=False,
    rope_theta=10000.0, attention_bias=False, attention_dropout=0.0,
    decoder_sparse_step=1, mlp_only_layers=[], moe_intermediate_size=MOE_INTER,
    num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K, norm_topk_prob=True,
    output_router_logits=False, router_aux_loss_coef=0.001, torch_dtype="bfloat16",
)

print("=" * 70)
print("  STEP 1: tiny Qwen3-MoE reference (random weights, bf16)")
print("=" * 70)
torch.manual_seed(0)
hf_model = Qwen3MoeForCausalLM(config).eval()
print(f"  transformers model: {sum(p.numel() for p in hf_model.parameters()):,} params")

os.makedirs(MODEL_DIR, exist_ok=True)
# transformers 5.x packs experts; write the HUB layout (per-expert names) so
# nano-vllm's real loader path is exercised.
import re
hub_sd = {}
for name, tensor in hf_model.state_dict().items():
    m = re.fullmatch(r"(.*mlp\.experts)\.(gate_up_proj|down_proj)", name)
    if m is None:
        hub_sd[name] = tensor
    elif m.group(2) == "gate_up_proj":
        e, two_i, h = tensor.shape
        for e_i in range(e):
            hub_sd[f"{m.group(1)}.{e_i}.gate_proj.weight"] = tensor[e_i, : two_i // 2].contiguous()
            hub_sd[f"{m.group(1)}.{e_i}.up_proj.weight"] = tensor[e_i, two_i // 2:].contiguous()
    else:
        e, h, i = tensor.shape
        for e_i in range(e):
            hub_sd[f"{m.group(1)}.{e_i}.down_proj.weight"] = tensor[e_i].contiguous()
st.save_file(hub_sd, f"{MODEL_DIR}/model.safetensors")
config.save_pretrained(MODEL_DIR)
# real tokenizer (small download) so the engine can decode; prompts pass as ids
AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(MODEL_DIR)
print(f"  hub-layout checkpoint + config + tokenizer -> {MODEL_DIR}")

# ── 2. nano-vllm engine on GPU ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2: nano-vllm LLM engine (real flash-attn, eager mode)")
print("=" * 70)
# enforce_eager=True: MoE routing uses data-dependent torch.nonzero, which is
# not CUDA-graph-capturable (vLLM needs grouped GEMMs for that). Correctness
# first; graph capture is a known follow-up.
llm = LLM(MODEL_DIR, max_model_len=256, max_num_batched_tokens=4096, enforce_eager=True)
print("  engine up (prefill: flash_attn_varlen_func, decode: flash_attn_with_kvcache)")

# ── 3. Generate with fixed prompt ids, greedy-ish (temperature 1e-9) ────────
PROMPT = [7, 42, 99, 3, 17, 88, 5, 31]          # arbitrary ids in vocab
MAX_NEW = 24
print("\n" + "=" * 70)
print(f"  STEP 3: generate {MAX_NEW} tokens (greedy) — nanovllm")
print("=" * 70)
nv_out = llm.generate(
    [PROMPT],
    SamplingParams(temperature=1e-9, max_tokens=MAX_NEW, ignore_eos=True),
    use_tqdm=False,
)[0]
nv_tokens = nv_out["token_ids"]
print(f"  nanovllm tokens: {nv_tokens}")

# ── 4. transformers reference, same weights, greedy ─────────────────────────
print("\n" + "=" * 70)
print("  STEP 4: transformers reference (greedy)")
print("=" * 70)
with torch.no_grad():
    hf_tokens = hf_model.generate(
        torch.tensor([PROMPT], device="cuda"),
        do_sample=False, max_new_tokens=MAX_NEW, eos_token_id=None,
        use_cache=True,
    )[0, len(PROMPT):].tolist()
print(f"  transformers tokens: {hf_tokens}")

# ── 5. Compare ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
match = nv_tokens == hf_tokens
print(f"  identical token sequences: {match}")
if not match:
    for i, (a, b) in enumerate(zip(nv_tokens, hf_tokens)):
        if a != b:
            print(f"  first divergence at position {i}: nanovllm={a} hf={b}")
            break
print("=" * 70)
print(f"  RESULT: {'PASS ✅  (nano-vllm MoE == transformers on GPU)' if match else 'FAIL ❌'}")
print("=" * 70)

llm.exit()

# ── Optional: real Qwen3-30B-A3B (needs >= 70 GB GPU + ~70 GB disk) ─────────
if os.environ.get("QWEN3_MOE_REAL") == "1":
    print("\n" + "=" * 70)
    print("  STEP 6: REAL Qwen3-30B-A3B (61 GB bf16)")
    print("=" * 70)
    free, total = torch.cuda.mem_get_info()
    if free < 70 * 2**30:
        sys.exit("  not enough GPU memory for the real checkpoint (need >= 70 GB free).")
    from huggingface_hub import snapshot_download
    real_dir = "/tmp/Qwen3-30B-A3B"
    print("  downloading 61 GB (this takes a while)...")
    snapshot_download("Qwen/Qwen3-30B-A3B", local_dir=real_dir)
    # nanovllm: load + short generate
    llm_real = LLM(real_dir, max_model_len=512, enforce_eager=True)
    print("  nanovllm loaded real model. generating 8 tokens...")
    out_real = llm_real.generate(
        [PROMPT], SamplingParams(temperature=1e-9, max_tokens=8, ignore_eos=True), use_tqdm=False)
    print(f"  nanovllm (real 30B-A3B): {out_real[0]['token_ids']}")
    llm_real.exit()
    print("  NOTE: transformers reference skipped (needs another 61 GB — compare on a")
    print("  second run, or eyeball token plausibility + throughput above).")
