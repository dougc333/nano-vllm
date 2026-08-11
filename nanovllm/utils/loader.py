import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# MoE expert weights: <prefix>.mlp.experts.<idx>.<gate_proj|up_proj|down_proj>.weight
_EXPERT_RE = re.compile(r"(.*mlp\.experts)\.(\d+)\.(\w+)\.weight")
# MoE packed layout (transformers 5.x state_dict): <prefix>.mlp.experts.{gate_up_proj|down_proj}
_EXPERT_PACKED_RE = re.compile(r"(.*mlp\.experts)\.(gate_up_proj|down_proj)$")


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    expert_modules_mapping = getattr(model, "expert_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 1. MoE experts (before packed mapping: names contain gate_proj etc.)
                m = _EXPERT_RE.fullmatch(weight_name)
                if m and m.group(3) in expert_modules_mapping:
                    prefix, expert_idx, proj = m.groups()
                    param = model.get_parameter(f"{prefix}.{expert_modules_mapping[proj]}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name), int(expert_idx), proj)
                    continue
                # 2. packed MoE experts (single [E, ...] tensor per proj)
                m = _EXPERT_PACKED_RE.fullmatch(weight_name)
                if m:
                    param = model.get_parameter(f"{m.group(1)}.{m.group(2)}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    continue
                # 3. fused/packed modules (qkv_proj, gate_up_proj)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 4. plain parameters (norms, o_proj, router gate, lm_head, ...)
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
