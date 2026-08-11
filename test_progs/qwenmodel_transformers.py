import inspect
import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

print("model:", model)
print("model type:", type(model))
rotary = model.model.rotary_emb

print(rotary)
print(type(rotary))
print(type(rotary).__module__)


rotary_class = type(rotary)
module = inspect.getmodule(rotary_class)

print("\nSource file:")
print(module.__file__)

print("\nQwen3RotaryEmbedding source:")
print(inspect.getsource(rotary_class))

print("\napply_rotary_pos_emb source:")
print(inspect.getsource(module.apply_rotary_pos_emb))