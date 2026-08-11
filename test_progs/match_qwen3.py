"""
Match nano-vllm's Qwen3.py to Hugging Face Qwen3 and print layer structure.
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "accelerate"])

import torch
from transformers import Qwen3Config, AutoConfig

MODEL_ID = "Qwen/Qwen3-0.6B"  # smallest Qwen3 — fast to load

# ── 1. Load HF config — this is what nanovllm uses ──
print("=" * 65)
print("  STEP 1: Load HuggingFace Qwen3 config")
print("=" * 65)
config = AutoConfig.from_pretrained(MODEL_ID)
print(f"\n  Model: {MODEL_ID}")
print(f"  architecture: {config.architectures[0]}")
print(f"  model_type:   {config.model_type}")
print(f"  n_layers:     {config.num_hidden_layers}")
print(f"  hidden_size:  {config.hidden_size}")
print(f"  n_heads:      {config.num_attention_heads}")
print(f"  n_kv_heads:   {config.num_key_value_heads}")
print(f"  head_dim:     {getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}")
print(f"  inter_size:   {config.intermediate_size}")
print(f"  vocab_size:   {config.vocab_size}")
print(f"  max_seq_len:  {config.max_position_embeddings}")
print(f"  rope_theta:   {getattr(config, 'rope_theta', '(see rope_scaling)')}")
print(f"  rms_norm_eps: {config.rms_norm_eps}")
print(f"  tie_weights:  {config.tie_word_embeddings}")
print(f"  rope_scaling: {getattr(config, 'rope_scaling', None)}")

# ── 2. Build model on meta device (no weight download) ──
print("\n" + "=" * 65)
print("  STEP 2: Build model on meta device to inspect layer tree")
print("=" * 65)
from transformers import AutoModelForCausalLM
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

def print_tree(module, prefix="", depth=0, max_depth=6):
    if depth > max_depth:
        return
    params = sum(p.numel() for p in module.parameters(recurse=False))
    p_str = f"  [{params:,} params]" if params else ""
    children = list(module.named_children())
    child_str = f"  ({len(children)} children)" if children else ""
    print(f"{prefix}{module.__class__.__name__}{child_str}{p_str}")
    for name, child in children:
        print(f"{prefix}  └─ {name}: ", end="")
        print_tree(child, prefix + "     ", depth + 1, max_depth)

print_tree(model, max_depth=4)

# ── 3. Load actual weights (optional) and print parameter shapes ──
print("\n" + "=" * 65)
print("  STEP 3: HF Qwen3 layers matched to nano-vllm equivalents")
print("=" * 65)
print(f"""
  nano-vllm Qwen3ForCausalLM           ↔  HuggingFace Qwen3ForCausalLM
  ═══════════════════════               ═  ═══════════════════════════
  Qwen3ForCausalLM                      ↔  Qwen3ForCausalLM
    Qwen3Model                          ↔  Qwen3Model
      embed_tokens (VocabParallel)      ↔  embed_tokens (nn.Embedding)
      layers[i] (Qwen3DecoderLayer)     ↔  layers[i] (Qwen3DecoderLayer)
        input_layernorm (RMSNorm)       ↔  input_layernorm (Qwen3RMSNorm)
        self_attn (Qwen3Attention)      ↔  self_attn (Qwen3SdpaAttention)
          qkv_proj (fused QKV linear)   ↔  q_proj + k_proj + v_proj (separate)
          q_norm / k_norm (RMSNorm)     ↔  q_norm / k_norm (if attention_bias=False)
          rotary_emb (RoPE)             ↔  rotary_emb (RoPE)
          attn (PagedAttention)         ↔  sdpa (scaled_dot_product_attention)
          o_proj (RowParallel linear)   ↔  o_proj (nn.Linear)
        mlp (Qwen3MLP)                  ↔  mlp (Qwen3MLP)
          gate_up_proj (fused)          ↔  gate_proj + up_proj (separate)
          SiluAndMul                    ↔  act_fn (SiLU)
          down_proj (RowParallel)       ↔  down_proj (nn.Linear)
        post_attention_layernorm        ↔  post_attention_layernorm
      norm (RMSNorm)                    ↔  norm (Qwen3RMSNorm)
    lm_head (ParallelLMHead)            ↔  lm_head (nn.Linear)
""")

# ── 4. Show the actual parameter shapes for layer 0 ──
print("=" * 65)
print("  STEP 4: Actual parameter shapes (layer 0 example)")
print("=" * 65)
# Load just layer 0 to show shapes
from safetensors import safe_open
import os

# Download weights (just for display purposes — small model)
print("\n  Downloading Qwen3-0.6B to show real shapes...")
model_real = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cpu",
    low_cpu_mem_usage=True
)
print(f"\n  Total parameters: {sum(p.numel() for p in model_real.parameters()):,}")
print(f"  Total size:       {sum(p.numel()*p.element_size() for p in model_real.parameters()) / 1e9:.2f} GB in fp16")

# Print shapes organized like nano-vllm's structure
def fmt_param(name, param):
    return f"  {name:55s} {str(list(param.shape)):25s} {param.numel():>12,}"

print(f"\n  Global weights:")
for n, p in model_real.named_parameters():
    if ".0." not in n and "layers" not in n:
        print(fmt_param(n, p))

print(f"\n  Layer 0 weights:")
for n, p in model_real.named_parameters():
    if "layers.0." in n:
        print(fmt_param(n.replace("layers.0.", ""), p))

print(f"\n  └─ Layers 1..{config.num_hidden_layers-1} have identical shapes")
