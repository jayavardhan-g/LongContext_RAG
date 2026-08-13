"""Cost of the head signal, measured without warm-up contamination.

`run_cascade.py` times each shortlist size in its own block, which reproduces
the Session 3 mistake: the first block pays for cold kernels and reads slower
than identical work later. It showed S=2048 as *faster* than S=1024, which is
not physically possible for a prefill.

Fix is the same one `baselines/run_latency.py` uses -- shuffle the job order so
warm-up is spread across conditions instead of landing entirely on whichever ran
first, and discard the first calls outright.

Reported alongside is the piece the Milestone 2 latency table left out. That
table measures *generation only*, so its headline "15x faster than naive
long-context" silently assumes retrieval is free. Encoding this benchmark's
per-question haystack is not free, and the honest number depends on whether the
corpus is indexed ahead of time:

    per-query corpus (this benchmark as built) -- encoding counts against you
    pre-indexed corpus (a deployed system)     -- encoding is amortized away

Both are reported rather than picking the flattering one.

Usage:
    python -m compress.run_cascade_latency --n 25
"""

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.encoders import BGE_M3, DenseEncoder
from data.budget import fill_budget
from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot
from heads.capture import AttentionCapture
from paths import resolve_model

from .head_scorer import HeadAttentionScorer, load_head_list
from .run_cascade import DEFAULT_HEADS
from .run_fusion_eval import MODEL


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return f"cuda:{min(range(len(out)), key=lambda i: int(out[i]))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/diagnostic/Meta-Llama-3.1-8B-Instruct_scores.npz")
    ap.add_argument("--heads", default=DEFAULT_HEADS)
    ap.add_argument("--head-key", default="head_sum_k5")
    ap.add_argument("--shortlists", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--context-length", type=int, default=32000)
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/cascade")
    args = ap.parse_args()

    device = args.device or pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_hotpot("validation")
    tok = AutoTokenizer.from_pretrained(resolve_model(MODEL))
    pool = DistractorPool(size=args.pool_size, seed=args.seed)
    builder = InstanceBuilder(tok, pool)
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    instances = [builder.build(ds[i], args.context_length, seed=rank) for rank, i in enumerate(idx)]

    z = np.load(args.scores)
    offsets = z["offsets"]
    dense_all = [z["dense"][offsets[i]:offsets[i + 1]] for i in range(len(instances))]

    # Shortlists precomputed so packing cost never lands inside a timed region.
    shortlists = {
        (i, S): fill_budget(inst.chunks, dense_all[i].tolist(), S, builder.title_tokens(inst))
        for i, inst in enumerate(instances)
        for S in args.shortlists
    }

    model = AutoModelForCausalLM.from_pretrained(
        resolve_model(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    head_list = load_head_list(args.heads)

    jobs = [(i, S) for i in range(len(instances)) for S in args.shortlists]
    jobs += [(i, None) for i in range(len(instances))]  # None = full context
    random.Random(args.seed).shuffle(jobs)

    timings: dict[str, list[float]] = defaultdict(list)
    with AttentionCapture(model) as cap:
        scorer = HeadAttentionScorer(model, tok, cap, head_list, (5,))
        for n, (i, S) in enumerate(jobs):
            inst = instances[i]
            chunks = inst.chunks if S is None else shortlists[(i, S)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scorer.score_all(inst.question, chunks)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if n >= args.warmup:
                timings["head_fullctx" if S is None else f"head_S{S}"].append(dt)

    del model
    torch.cuda.empty_cache()

    encoder = DenseEncoder(BGE_M3, device=device)
    enc_jobs = list(range(len(instances))) * 2
    random.Random(args.seed).shuffle(enc_jobs)
    for n, i in enumerate(enc_jobs):
        inst = instances[i]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder.encode([inst.question])
        encoder.encode([c.embed_text for c in inst.chunks])
        torch.cuda.synchronize()
        if n >= args.warmup:
            timings["dense_encode"].append((time.perf_counter() - t0) * 1000)

    stats = {
        k: {"mean": round(float(np.mean(v)), 1), "p50": round(float(np.median(v)), 1), "n": len(v)}
        for k, v in sorted(timings.items())
    }

    # Generation latency is taken from the Milestone 2 table rather than
    # re-measured, so the totals below stay comparable to its published numbers.
    gen_256, gen_naive = 92.0, 1381.0
    enc = stats["dense_encode"]["mean"]
    best_short = min(
        (k for k in stats if k.startswith("head_S")),
        key=lambda k: stats[k]["mean"],
    )
    totals = {
        "naive_long_context": {"per_query": gen_naive, "pre_indexed": gen_naive},
        "dense@256": {"per_query": enc + gen_256, "pre_indexed": gen_256},
        f"cascade@256 ({best_short})": {
            "per_query": enc + stats[best_short]["mean"] + gen_256,
            "pre_indexed": stats[best_short]["mean"] + gen_256,
        },
        "fusion_fullctx@256": {
            "per_query": enc + stats["head_fullctx"]["mean"] + gen_256,
            "pre_indexed": stats["head_fullctx"]["mean"] + gen_256,
        },
    }

    lines = [
        "# Cost of the head signal (shuffled job order, warm-up discarded)", "",
        f"{args.n} questions, unbatched, first {args.warmup} calls discarded.", "",
        "| stage | mean ms | p50 ms | n |", "|---|---|---|---|",
    ]
    for k, v in stats.items():
        lines.append(f"| {k} | {v['mean']:.1f} | {v['p50']:.1f} | {v['n']} |")
    lines += [
        "", "## End-to-end cost per question", "",
        "`per_query` charges the encoder every question, which is what this "
        "benchmark literally does -- each question carries its own haystack. "
        "`pre_indexed` amortizes it away, which is what a deployed system with a "
        "fixed corpus does. Generation is the Milestone 2 measurement "
        f"({gen_256:.0f} ms at a 256-token budget, {gen_naive:.0f} ms naive).", "",
        "| method | per-query ms | vs naive | pre-indexed ms | vs naive |",
        "|---|---|---|---|---|",
    ]
    for name, t in totals.items():
        lines.append(
            f"| {name} | {t['per_query']:.0f} | {gen_naive / t['per_query']:.2f}x | "
            f"{t['pre_indexed']:.0f} | {gen_naive / t['pre_indexed']:.2f}x |"
        )
    lines.append("")

    md = "\n".join(lines)
    (out_dir / "cascade_latency.md").write_text(md)
    with (out_dir / "cascade_latency.json").open("w") as fh:
        json.dump({"stats": stats, "totals": totals, "gen_ms": {"b256": gen_256, "naive": gen_naive}}, fh, indent=2)
    print(md)


if __name__ == "__main__":
    main()
