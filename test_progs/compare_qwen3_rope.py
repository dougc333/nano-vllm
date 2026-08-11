#!/usr/bin/env python3
"""Numerically compare Hugging Face Qwen3 RoPE with nano-vLLM RoPE.

The script deliberately uses identical deterministic query/key tensors and
positions for both implementations. It prints representative values before and
after rotation, reports error statistics, asserts numerical equivalence, and
writes a visualization of the output/error vectors.

Run from the nano-vLLM repository:

    python compare_qwen3_rope.py
    python compare_qwen3_rope.py --dtype bfloat16
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)


REPO_ROOT = Path(__file__).resolve().parent
NANO_ROPE_FILE = REPO_ROOT / "nanovllm" / "layers" / "rotary_embedding.py"


def default_qwen_config() -> Path:
    local_checkout = Path.home() / "huggingface/Qwen3-0.6B/config.json"
    if local_checkout.is_file():
        return local_checkout
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    )
    matches = sorted(snapshots.glob("*/config.json"))
    return matches[-1] if matches else local_checkout


def load_nano_rope_module():
    """Load only nano-vLLM's RoPE file, avoiding its CUDA-only package imports."""
    spec = importlib.util.spec_from_file_location("nano_rotary_embedding", NANO_ROPE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {NANO_ROPE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_row(label: str, tensor: torch.Tensor, width: int = 8) -> None:
    values = tensor.detach().float().cpu().flatten()[:width].tolist()
    formatted = " ".join(f"{value:+.8f}" for value in values)
    print(f"{label:<29} {formatted}")


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    error = (reference.float() - candidate.float()).abs()
    reference_f = reference.float()
    candidate_f = candidate.float()
    cosine = torch.nn.functional.cosine_similarity(
        reference_f.flatten(), candidate_f.flatten(), dim=0
    )
    return {
        "max_abs": error.max().item(),
        "mean_abs": error.mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "cosine": cosine.item(),
    }


def print_stats(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name:<8} max_abs={stats['max_abs']:.9e}  "
        f"mean_abs={stats['mean_abs']:.9e}  "
        f"rmse={stats['rmse']:.9e}  cosine={stats['cosine']:.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_qwen_config())
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "rope_comparison.png")
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(
            f"Qwen3 config not found at {args.config}. Pass --config /path/to/config.json"
        )

    config_data = json.loads(args.config.read_text())
    config = Qwen3Config.from_dict(config_data)
    rope_theta = config_data.get("rope_theta")
    if rope_theta is None:
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta", 1_000_000)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cpu")
    nano = load_nano_rope_module()

    # Non-contiguous positions exercise the position lookup more strongly than
    # merely testing [0, 1, 2, 3]. Position zero is retained as a useful control:
    # cos(0)=1 and sin(0)=0, so RoPE must leave that token unchanged.
    positions = torch.tensor([0, 1, 17, 1024], dtype=torch.long, device=device)
    num_tokens = positions.numel()
    q_heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    generator = torch.Generator(device="cpu").manual_seed(20260810)
    q_input = torch.randn(
        num_tokens, q_heads, head_dim, generator=generator, dtype=torch.float32
    ).to(dtype)
    k_input = torch.randn(
        num_tokens, kv_heads, head_dim, generator=generator, dtype=torch.float32
    ).to(dtype)

    # Hugging Face layout: [batch, heads, sequence, head_dim].
    q_hf_layout = q_input.permute(1, 0, 2).unsqueeze(0)
    k_hf_layout = k_input.permute(1, 0, 2).unsqueeze(0)
    position_ids = positions.unsqueeze(0)

    hf_rope = Qwen3RotaryEmbedding(config=config)
    hf_cos, hf_sin = hf_rope(q_hf_layout, position_ids)
    hf_q, hf_k = apply_rotary_pos_emb(
        q_hf_layout, k_hf_layout, hf_cos, hf_sin, unsqueeze_dim=1
    )
    hf_q = hf_q.squeeze(0).permute(1, 0, 2).contiguous()
    hf_k = hf_k.squeeze(0).permute(1, 0, 2).contiguous()

    # nano-vLLM precomputes half-width cos/sin tables. Its apply_rotary_emb
    # function splits each head into two halves, so this is algebraically the
    # same as HF's duplicated full-width cos/sin plus rotate_half().
    nano_rope = nano.RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=rope_theta,
    )
    nano_cos_sin = nano_rope.cos_sin_cache[positions]
    nano_cos, nano_sin = nano_cos_sin.chunk(2, dim=-1)
    nano_q = nano.apply_rotary_emb(q_input, nano_cos, nano_sin)
    nano_k = nano.apply_rotary_emb(k_input, nano_cos, nano_sin)

    # HF duplicates each frequency across the two head halves. Compare the
    # first half to nano-vLLM's compact table.
    cos_stats = error_stats(hf_cos[0, :, : head_dim // 2], nano_cos[:, 0, :])
    sin_stats = error_stats(hf_sin[0, :, : head_dim // 2], nano_sin[:, 0, :])
    q_stats = error_stats(hf_q, nano_q)
    k_stats = error_stats(hf_k, nano_k)

    token_index = 2  # position 17
    head_index = 0
    half = head_dim // 2
    print("\nQwen3-0.6B RoPE differential test")
    print("=" * 78)
    print(f"dtype={args.dtype}, positions={positions.tolist()}, head_dim={head_dim}")
    print(f"q_heads={q_heads}, kv_heads={kv_heads}, rope_theta={rope_theta:g}")
    print(f"Printed example: token index {token_index}, position {positions[token_index].item()}, head {head_index}")
    print("\nFirst eight dimensions (these rotate with dimensions 64..71):")
    tensor_row("Q input [0:8]", q_input[token_index, head_index, :8])
    tensor_row("HF Q after [0:8]", hf_q[token_index, head_index, :8])
    tensor_row("nano Q after [0:8]", nano_q[token_index, head_index, :8])
    tensor_row("abs difference [0:8]", (hf_q - nano_q)[token_index, head_index, :8].abs())
    print("\nPartner dimensions from the second half:")
    tensor_row("Q input [64:72]", q_input[token_index, head_index, half : half + 8])
    tensor_row("HF Q after [64:72]", hf_q[token_index, head_index, half : half + 8])
    tensor_row("nano Q after [64:72]", nano_q[token_index, head_index, half : half + 8])
    tensor_row(
        "abs difference [64:72]",
        (hf_q - nano_q)[token_index, head_index, half : half + 8].abs(),
    )

    print("\nError summary over every tested token and head:")
    print_stats("cos", cos_stats)
    print_stats("sin", sin_stats)
    print_stats("Q", q_stats)
    print_stats("K", k_stats)

    reference = hf_q[token_index, head_index].float().detach().cpu()
    candidate = nano_q[token_index, head_index].float().detach().cpu()
    difference = (reference - candidate).abs()
    dimensions = torch.arange(head_dim)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(dimensions, reference, label="Hugging Face Q after RoPE", linewidth=1.8)
    axes[0].plot(
        dimensions,
        candidate,
        "--",
        label="nano-vLLM Q after RoPE",
        linewidth=1.3,
    )
    axes[0].set_title(
        f"Qwen3-0.6B RoPE output — position {positions[token_index].item()}, head {head_index}"
    )
    axes[0].set_xlabel("Head dimension")
    axes[0].set_ylabel("Rotated value")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].stem(dimensions, difference, basefmt=" ")
    axes[1].set_title("Absolute numerical difference")
    axes[1].set_xlabel("Head dimension")
    axes[1].set_ylabel("|HF − nano-vLLM|")
    axes[1].grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"\nSaved visualization: {args.output}")

    # Float32 paths should agree to a few ULPs. BF16 has coarser rounding, so
    # use a tolerance appropriate to the representation.
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    assert cos_stats["max_abs"] <= tolerance, cos_stats
    assert sin_stats["max_abs"] <= tolerance, sin_stats
    assert q_stats["max_abs"] <= tolerance, q_stats
    assert k_stats["max_abs"] <= tolerance, k_stats
    print(f"PASS: all maximum absolute errors are <= {tolerance:g}")


if __name__ == "__main__":
    main()
