
"""
COLAB
load Qwen3-0.6B from Hugging Face, print every layer,
and trace one forward pass through the architecture manually.

Paste into one Colab cell (T4 GPU) or run locally:
    python explore_qwen3.py

Sections:
  1. Load config + tokenizer + model (meta device = no RAM cost for inspection)
  2. Print the full module tree
  3. Print per-layer parameter shapes
  4. Trace activations through one forward pass with hooks
  5. Manual layer-by-layer walk (embedding -> blocks -> norm -> lm_head)
"""

# ============================================================
# 0. Install
# ============================================================
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "transformers", "accelerate", "safetensors"])

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"

# ============================================================
# 1. Config: the architecture blueprint
# ============================================================
config = AutoConfig.from_pretrained(MODEL_ID)
print("=" * 60)
print("ARCHITECTURE CONFIG")
print("=" * 60)
for k in ["model_type", "hidden_size", "intermediate_size",
          "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim",
          "max_position_embeddings", "rope_theta",
          "rms_norm_eps", "vocab_size", "tie_word_embeddings"]:
    print(f"  {k:28s} = {getattr(config, k, '(n/a)')}")

# Derived facts
n_layers = config.num_hidden_layers
n_q = config.num_attention_heads
n_kv = config.num_key_value_heads
d = config.hidden_size
print(f"\n  GQA ratio (q_heads / kv_heads) = {n_q // n_kv}")
print(f"  Total layers = {n_layers}")

# ============================================================
# 2. Module tree (meta device: structure only, zero memory)
# ============================================================
print("\n" + "=" * 60)
print("MODULE TREE")
print("=" * 60)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

def print_tree(module, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    for name, child in module.named_children():
        params = sum(p.numel() for p in child.parameters(recurse=False))
        p_str = f"  [{params:,} params]" if params else ""
        print(f"{prefix}{name}: {child.__class__.__name__}{p_str}")
        print_tree(child, prefix + "  ", depth + 1, max_depth)

print_tree(model)

# ============================================================
# 3. Parameter shapes per layer
# ============================================================
print("\n" + "=" * 60)
print("PARAMETER SHAPES (layer 0 + global)")
print("=" * 60)
shown = 0
for name, p in model.named_parameters():
  print(f"name:{name} p:{p.shape}")
  #if ".0." in name or "model.layers" not in name: