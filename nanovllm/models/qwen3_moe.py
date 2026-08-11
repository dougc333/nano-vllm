"""Qwen3 MoE (Qwen3MoeForCausalLM) — sparse mixture-of-experts variant of Qwen3.

Differences vs the dense ``nanovllm.models.qwen3``:
  * The MLP is replaced by a sparse MoE block: a router ("gate") that picks the
    top-k experts per token + ``num_experts`` small MLPs (moe_intermediate_size).
  * Attention, norms, RoPE, embedding/head are identical — ``Qwen3Attention``
    is reused as-is (Qwen3-30B-A3B uses attention_bias=False, so q_norm/k_norm).

HF weight layout (verified against Qwen/Qwen3-30B-A3B):
    mlp.gate.weight                          router, [num_experts, hidden]
    mlp.experts.<i>.gate_proj.weight         per-expert, [moe_intermediate, hidden]
    mlp.experts.<i>.up_proj.weight
    mlp.experts.<i>.down_proj.weight         [hidden, moe_intermediate]

TP note: experts are *replicated* on every rank (no expert parallelism), so
each rank computes the full MoE output for its own token shard with zero
communication. vLLM shards experts across ranks and exchanges tokens with
all2all (expert parallelism) — a memory optimization, not a correctness change.
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP


class Qwen3MoeExperts(nn.Module):
    """All experts stacked into two tensors so the loader can slice by index.

    gate_up: [num_experts, 2 * moe_intermediate_size, hidden_size]  (gate+up fused)
    down:    [num_experts, hidden_size, moe_intermediate_size]
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.gate_up.weight_loader = self.gate_up_weight_loader
        self.down.weight_loader = self.down_weight_loader

    def gate_up_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                              expert_idx: int | None = None, proj: str | None = None):
        if expert_idx is None:
            # packed layout (transformers 5.x state_dict): [E, 2I, H]
            param.data.copy_(loaded_weight)
        elif proj == "gate_proj":
            param.data[expert_idx, : self.intermediate_size] = loaded_weight
        elif proj == "up_proj":
            param.data[expert_idx, self.intermediate_size:] = loaded_weight
        else:
            raise ValueError(f"unexpected proj {proj}")

    def down_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                           expert_idx: int | None = None, proj: str | None = None):
        if expert_idx is None:
            # packed layout (transformers 5.x state_dict): [E, H, I]
            param.data.copy_(loaded_weight)
        else:
            assert proj == "down_proj"
            param.data[expert_idx] = loaded_weight

    def forward(
        self,
        hidden_states: torch.Tensor,      # [T, H]
        topk_idx: torch.Tensor,           # [T, k]
        topk_weights: torch.Tensor,       # [T, k]
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden_states)
        # [num_experts, T, k]: which (token, slot) pairs hit each expert
        expert_mask = F.one_hot(topk_idx, self.num_experts).permute(2, 0, 1)
        for e in range(self.num_experts):
            positions = torch.nonzero(expert_mask[e])              # [n_e, 2] (token, slot)
            if positions.numel() == 0:
                continue
            tok_idx, slot_idx = positions[:, 0], positions[:, 1]
            x = hidden_states[tok_idx]                             # [n_e, H]
            gate_up = F.linear(x, self.gate_up[e])                 # [n_e, 2I]
            gate, up = gate_up.chunk(2, dim=-1)
            x = F.silu(gate) * up                                  # [n_e, I]
            x = F.linear(x, self.down[e])                          # [n_e, H]
            out[tok_idx] += x * topk_weights[tok_idx, slot_idx, None]
        return out


class Qwen3MoeTopKRouter(ReplicatedLinear):
    """Replicated gate: every rank computes identical top-k routing.

    Subclasses ReplicatedLinear so the weight lives at ``mlp.gate.weight``,
    matching the HF checkpoint layout exactly.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config.hidden_size, config.num_experts, bias=False)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = super().forward(hidden_states)               # [T, E]
        router_probs = F.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
        topk_weights, topk_idx = router_probs.topk(self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        return topk_weights, topk_idx


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])           # [T, H]
        topk_weights, topk_idx = self.gate(flat)
        out = self.experts(flat, topk_idx, topk_weights)
        return out.reshape(orig_shape)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # Same sparse schedule as transformers: sparse when the layer is in the
        # sparse step (decoder_sparse_step=1 -> every layer for 30B-A3B), else dense.
        if (layer_idx not in config.mlp_only_layers
                and config.num_experts > 0
                and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,      # the full residual stream
    ) -> torch.Tensor:
        # LLaMA-style residual (transformers Qwen3MoeDecoderLayer): the norms
        # do NOT absorb the residual — add it after each sub-block. (The dense
        # Qwen3 family instead folds residual into the RMSNorm — do not copy
        # that pattern here.)
        h = self.input_layernorm.rms_forward(hidden_states)
        h = self.self_attn(positions, h)
        hidden_states = hidden_states + h
        h = self.post_attention_layernorm.rms_forward(hidden_states)
        h = self.mlp(h)
        return hidden_states + h


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm.rms_forward(hidden_states)


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # dense-MLP fallback layers in mixed sparse/dense configs
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # HF: mlp.experts.<idx>.{gate_proj,up_proj,down_proj}.weight -> stacked params
    expert_modules_mapping = {
        "gate_proj": "gate_up",
        "up_proj": "gate_up",
        "down_proj": "down",
    }

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
