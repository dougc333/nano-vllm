#!/usr/bin/env python3
"""
nano-vllm on CPU — no GPU, no flash-attn, no CUDA graphs.

Uses the REAL Scheduler / BlockManager / Sequence / Config from
/Users/dc/nano-vllm. Only the GPU side is replaced:
  - nanovllm/__init__.py is shimmed (it imports llm.py -> llm_engine ->
    model_runner -> flash_attn, which is CUDA-only).
  - The ModelRunner is replaced by CpuModelRunner, which implements the
    SAME paging math as the real one (model_runner.py prepare_prefill /
    prepare_decode / Attention.store_kvcache) on a tiny random Qwen3
    model in plain torch.

What you can watch live: prefill <-> decode switching (XOR batching),
chunked prefill, prefix-cache block sharing (ref counts), and
preemption-by-recompute when the KV cache runs out of blocks.
"""
import os
import sys
import types

REPO = "/Users/dc/nano-vllm"

# ---- Shim: register the nanovllm package WITHOUT executing __init__.py ----
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(REPO, "nanovllm")]
sys.modules["nanovllm"] = _pkg

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler

torch.manual_seed(0)
H, L, HEADS, KV_HEADS, HEAD_DIM, VOCAB, MLP_H = 32, 2, 4, 2, 8, 64, 64
SCALE = HEAD_DIM ** -0.5


# ============================================================================
# 1. TINY QWEN3 MODEL — paged KV attention in plain torch (CPU)
# ============================================================================
class TinyLayer(nn.Module):
    """One decoder layer: QKV -> paged attention -> MLP, residual stream."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(H, (HEADS + 2 * KV_HEADS) * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, H, bias=False)
        self.gate_up = nn.Linear(H, 2 * MLP_H, bias=False)
        self.down = nn.Linear(MLP_H, H, bias=False)
        self.norm1 = nn.LayerNorm(H, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(H, elementwise_affine=False)


class TinyQwen3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.layers = nn.ModuleList([TinyLayer() for _ in range(L)])
        self.norm_final = nn.LayerNorm(H, elementwise_affine=False)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        self.lm_head.weight = self.embed.weight  # tie, like Qwen3

    def forward(self, x, ctx):
        """x: [T, H] hidden states, ctx: per-step paging context (see runner)."""
        for layer_idx, layer in enumerate(self.layers):
            k_cache = ctx["k_cache"][layer_idx]
            v_cache = ctx["v_cache"][layer_idx]

            h = layer.norm1(x)
            qkv = layer.qkv(h)
            q, k, v = qkv.split([HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
            q = q.view(-1, HEADS, HEAD_DIM)
            k = k.view(-1, KV_HEADS, HEAD_DIM)
            v = v.view(-1, KV_HEADS, HEAD_DIM)

            # --- store k/v into the paged KV cache (Triton store_kvcache equiv) ---
            flat_k = k_cache.view(-1, KV_HEADS * HEAD_DIM)
            flat_v = v_cache.view(-1, KV_HEADS * HEAD_DIM)
            flat_k[ctx["slot_mapping"]] = k.view(-1, KV_HEADS * HEAD_DIM)
            flat_v[ctx["slot_mapping"]] = v.view(-1, KV_HEADS * HEAD_DIM)

            # --- paged attention: per-seq gather via block_table, then matmul ---
            out = torch.empty_like(q)
            for s in range(len(ctx["seqs"])):
                seq = ctx["seqs"][s]
                q0, q1 = int(ctx["cu_seqlens_q"][s]), int(ctx["cu_seqlens_q"][s + 1])
                Tq = q1 - q0
                if ctx["is_prefill"]:
                    Tk = int(ctx["cu_seqlens_k"][s + 1]) - int(ctx["cu_seqlens_k"][s])
                else:  # decode: full context for this seq (context_lens equiv)
                    Tk = ctx["context_lens"][s]

                # gather full KV for this seq from paged blocks (flash_attn block_table equiv)
                k_full, v_full = [], []
                remaining = Tk
                for block_id in seq.block_table:
                    take = min(256, remaining)
                    k_full.append(k_cache[block_id, :take])
                    v_full.append(v_cache[block_id, :take])
                    remaining -= take
                    if remaining == 0:
                        break
                k_full = torch.cat(k_full, 0).repeat_interleave(HEADS // KV_HEADS, dim=1)  # GQA
                v_full = torch.cat(v_full, 0).repeat_interleave(HEADS // KV_HEADS, dim=1)

                scores = torch.einsum("qhd,khd->qhk", q[q0:q1], k_full) * SCALE  # [Tq, H, Tk]
                if ctx["is_prefill"]:
                    # scheduled tokens are the LAST Tq of the Tk context
                    pos = torch.arange(seq.num_cached_tokens, seq.num_cached_tokens + Tq)
                    mask = torch.arange(Tk)[None, :] > pos[:, None]  # future keys -> -inf
                    scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
                attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
                out[q0:q1] = torch.einsum("qhk,khd->qhd", attn, v_full)

            x = x + layer.o_proj(out.flatten(1))
            g = layer.norm2(x)
            g_up = layer.gate_up(g)
            gate, up = g_up.chunk(2, dim=-1)
            x = x + layer.down(F.silu(gate) * up)
        return self.lm_head(self.norm_final(x))


# ============================================================================
# 2. CPU MODEL RUNNER — same interface + paging math as the real ModelRunner
# ============================================================================
class CpuModelRunner:
    def __init__(self, config: Config, num_blocks: int):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.num_blocks = num_blocks
        self.model = TinyQwen3()
        # paged KV cache: [num_layers, num_blocks, block_size, kv_heads, head_dim]
        # (real code: kv_cache[2, num_layers, ...], k_cache = kv_cache[0, layer_id])
        self.k_cache = torch.zeros(L, num_blocks, self.block_size, KV_HEADS, HEAD_DIM)
        self.v_cache = torch.zeros(L, num_blocks, self.block_size, KV_HEADS, HEAD_DIM)

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)

    def run(self, seqs, is_prefill):
        """Mirror of ModelRunner.run: prepare -> forward -> sample -> token ids."""
        if is_prefill:
            input_ids, ctx = self.prepare_prefill(seqs)
        else:
            input_ids, ctx = self.prepare_decode(seqs)

        ctx["seqs"] = seqs
        ctx["is_prefill"] = is_prefill
        ctx["k_cache"], ctx["v_cache"] = self.k_cache, self.v_cache

        logits = self.model(self.model.embed(input_ids), ctx)
        # sample last token of each seq (ParallelLMHead / logits_to_keep equiv)
        if is_prefill:
            last_rows = ctx["cu_seqlens_q"][1:] - 1
            logits = logits[last_rows]
        temps = torch.tensor([seq.temperature for seq in seqs])
        return self.sample(logits, temps).tolist()

    def sample(self, logits, temperatures):
        """Gumbel-max sampling, same trick as the real Sampler."""
        logits = logits.float().div_(temperatures.unsqueeze(1))
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

    # --- prepare_prefill: same slot math as model_runner.py:129 ---
    def prepare_prefill(self, seqs):
        input_ids, cu_q, cu_k, slot_mapping = [], [0], [0], []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq[start:end])
            cu_q.append(cu_q[-1] + (end - start))
            cu_k.append(cu_k[-1] + end)
            if not seq.block_table:  # warmup path, not used here
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                slot_end = seq.block_table[i] * self.block_size + (
                    self.block_size if i != end_block - 1 else end - i * self.block_size
                )
                slot_mapping.extend(range(slot_start, slot_end))
        ctx = {
            "cu_seqlens_q": torch.tensor(cu_q, dtype=torch.int32),
            "cu_seqlens_k": torch.tensor(cu_k, dtype=torch.int32),
            "slot_mapping": torch.tensor(slot_mapping, dtype=torch.int64),
            "context_lens": None,
        }
        return torch.tensor(input_ids), ctx

    # --- prepare_decode: same math as model_runner.py:172 ---
    def prepare_decode(self, seqs):
        input_ids, slot_mapping, context_lens = [], [], []
        for seq in seqs:
            input_ids.append(seq.last_token)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            context_lens.append(len(seq))
        ctx = {
            "cu_seqlens_q": torch.tensor(list(range(len(seqs) + 1)), dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, len(seqs)], dtype=torch.int32),
            "slot_mapping": torch.tensor(slot_mapping, dtype=torch.int64),
            "context_lens": context_lens,
        }
        return torch.tensor(input_ids), ctx


# ============================================================================
# 3. ENGINE LOOP — replicates LLMEngine.step() using the real Scheduler
# ============================================================================
class CpuEngine:
    def __init__(self, model_dir, num_blocks, **cfg_kwargs):
        self.config = Config(model=model_dir, **cfg_kwargs)
        # The real ModelRunner.allocate_kv_cache() sets this after measuring
        # GPU memory; on CPU we fix the budget up front.
        self.config.num_kvcache_blocks = num_blocks
        Sequence.block_size = self.config.kvcache_block_size  # what LLMEngine does
        self.scheduler = Scheduler(self.config)
        self.runner = CpuModelRunner(self.config, num_blocks)

    def add_request(self, prompt_ids, sampling_params):
        self.scheduler.add(Sequence(prompt_ids, sampling_params))

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], is_prefill, True  # stalled: nothing schedulable
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return seqs, is_prefill, False

    def step_visible(self):
        """Like step() but returns BEFORE postprocess so the demo can print
        what was actually scheduled (num_scheduled_tokens is reset after)."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], is_prefill, True
        token_ids = self.runner.run(seqs, is_prefill)
        return seqs, is_prefill, token_ids

    def postprocess(self, seqs, token_ids, is_prefill):
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

    def state_str(self):
        s = self.scheduler
        free = len(s.block_manager.free_block_ids)
        refs = {i: b.ref_count for i, b in enumerate(s.block_manager.blocks) if b.ref_count}
        tables = {q.seq_id: list(q.block_table) for q in s.waiting + s.running}
        return f"free_blocks={free} refs={refs} tables={tables}"


# ============================================================================
# 4. DEMOS
# ============================================================================
def rng_prompt(n, lo=5, hi=VOCAB - 4):
    return [int(t) for t in torch.randint(lo, hi, (n,)).tolist()]


def run_demo(name, engine, requests, inject=None, max_steps=60):
    """requests: [(prompt, sampling_params)] added upfront.
    inject: {step_index: [(prompt, sampling_params)]} added at start of step.
    Returns the step count."""
    print("=" * 72)
    print(f"DEMO: {name}")
    print("=" * 72)
    for p, sp in requests:
        engine.add_request(p, sp)
    inject = inject or {}
    step = 0
    prev_running = set()  # seq ids that were RUNNING after last postprocess
    while not engine.scheduler.is_finished() and step < max_steps:
        for p, sp in inject.get(step, []):
            engine.add_request(p, sp)
        seqs, is_prefill, token_ids = engine.step_visible()
        if isinstance(token_ids, bool):
            print(f"  step {step}: STALLED (no schedulable seqs)")
            break
        phase = "PREFILL" if is_prefill else "decode "
        info = " ".join(
            f"seq{q.seq_id}({q.num_scheduled_tokens}tok,{q.num_cached_tokens}cache)"
            for q in seqs
        )
        # a seq that was RUNNING but is now WAITING with no blocks was preempted
        preempted = [q.seq_id for q in engine.scheduler.waiting
                     if q.seq_id in prev_running and not q.block_table]
        pre_s = f" | PREEMPTED {preempted}" if preempted else ""
        print(f"  step {step:2d} {phase} | {info} | {engine.state_str()}{pre_s}")
        engine.postprocess(seqs, token_ids, is_prefill)
        prev_running = {q.seq_id for q in engine.scheduler.running}
        done = [q for q in seqs if q.is_finished]
        if done:
            for q in done:
                print(f"         DONE seq{q.seq_id}: {q.completion_token_ids}")
        step += 1
    print(f"  finished in {step} steps\n")
    return step


def main():
    model_dir = os.path.join(REPO, "cpu_demo_model")

    # --- A: basic prefill -> decode switching, two concurrent requests -------
    eng = CpuEngine(model_dir, num_blocks=8, max_num_seqs=8, kvcache_block_size=256)
    run_demo(
        "A. Basic: 2 requests, prefill XOR decode, KV blocks",
        eng,
        [(rng_prompt(400), SamplingParams(temperature=0.8, max_tokens=5)),
         (rng_prompt(400), SamplingParams(temperature=0.8, max_tokens=5))],
    )

    # --- B: prefix cache — seq1 arrives after seq0's blocks are hashed ------
    eng = CpuEngine(model_dir, num_blocks=8, max_num_seqs=8, kvcache_block_size=256)
    p0 = rng_prompt(400)
    p1 = p0[:256] + rng_prompt(144)  # shares seq0's first 256 tokens (block 0)
    run_demo(
        "B. Prefix cache: seq1 shares block 0 with seq0 (ref_count=2)",
        eng,
        [(p0, SamplingParams(temperature=0.8, max_tokens=4))],
        inject={1: [(p1, SamplingParams(temperature=0.8, max_tokens=4))]},
    )

    # --- C: preemption — 5 x 512-token requests, only 6 KV blocks ----------
    # 512-token prompts fill 2 blocks each; the first decode crosses a block
    # boundary while the cache is full, forcing preempt + recompute.
    eng = CpuEngine(model_dir, num_blocks=6, max_num_seqs=8, kvcache_block_size=256)
    run_demo(
        "C. Preemption: 5 x 512-token requests, only 6 KV blocks",
        eng,
        [(rng_prompt(512), SamplingParams(temperature=0.8, max_tokens=3)) for _ in range(5)],
    )

    # --- D: chunked prefill — 700-token prompt, 300-token budget ------------
    eng = CpuEngine(model_dir, num_blocks=8, max_num_seqs=8,
                    max_num_batched_tokens=300, kvcache_block_size=256)
    run_demo(
        "D. Chunked prefill: 700-token prompt, 300-token batch budget",
        eng,
        [(rng_prompt(700), SamplingParams(temperature=0.8, max_tokens=4))],
    )


if __name__ == "__main__":
    main()
