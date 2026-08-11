"""Measure what compression actually saves at inference time.

    python -m baselines.run_latency --n 25 --context-lengths 4000 16000 32000

Accuracy runs generate in batches, which mixes wall-clock across conditions and
makes per-condition latency meaningless. This measures unbatched, one prompt at
a time, so the numbers correspond to serving a single request.

The first few generations are discarded as warm-up: the first call on a fresh
CUDA context pays autotuning and allocator growth that no later request does.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.budget import fill_budget
from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot, render_context
from paths import resolve_model

from .encoders import BGE_M3, DenseEncoder
from .generate import Generator, build_prompt
from .retrievers import DenseRetriever
from .run_baselines import DEFAULT_MODEL, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--encoder", default=BGE_M3)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[4000, 16000, 32000])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()

    device = args.device or pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_hotpot("validation")
    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    builder = InstanceBuilder(tok, DistractorPool(size=args.pool_size, seed=args.seed))

    import random as _random

    idx = list(range(len(ds)))
    _random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    instances = [
        builder.build(ds[i], ctx, seed=rank)
        for ctx in args.context_lengths
        for rank, i in enumerate(idx)
    ]
    print(f"[data] {len(instances)} instances", flush=True)

    print(f"[score] encoder -> {device}", flush=True)
    encoder = DenseEncoder(args.encoder, device=device)
    dense = DenseRetriever(encoder)
    scores = [dense.score(i.question, i.chunks) for i in instances]
    del encoder, dense
    torch.cuda.empty_cache()

    print(f"[gen] loading {args.model} -> {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    gen = Generator(model, tok)

    jobs = []
    for inst, s in zip(instances, scores):
        titles = builder.title_tokens(inst)
        for budget in args.budgets:
            sel = fill_budget(inst.chunks, s, budget, titles)
            jobs.append((inst.target_tokens, f"dense@{budget}", render_context(sel), inst.question))
        jobs.append(
            (inst.target_tokens, "naive_long_context", render_context(inst.chunks), inst.question)
        )

    # Jobs are built grouped by context length, and whichever group ran first
    # absorbed the residual allocator/autotune warm-up -- a 330-token prompt
    # timed 196 ms in the leading block against 77 ms in a later one. Shuffling
    # spreads any remaining drift evenly over conditions instead of charging it
    # all to one.
    _random.Random(args.seed).shuffle(jobs)

    for ctx, q in [(jobs[0][2], jobs[0][3])] * args.warmup:
        gen.time_one(build_prompt(tok, ctx, q))

    rows = defaultdict(list)
    t0 = time.time()
    for n, (ctx_len, cond, ctx, question) in enumerate(jobs, 1):
        _, elapsed, n_tok = gen.time_one(build_prompt(tok, ctx, question))
        rows[(ctx_len, cond)].append((elapsed, n_tok))
        if n % 50 == 0 or n == len(jobs):
            print(f"[lat] {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    summary = []
    for (ctx_len, cond), vals in sorted(rows.items()):
        lat = sorted(v[0] for v in vals)
        summary.append(
            {
                "context_length": ctx_len,
                "condition": cond,
                "n": len(vals),
                "prompt_tokens": round(sum(v[1] for v in vals) / len(vals), 1),
                "latency_ms_mean": round(1000 * sum(lat) / len(lat), 1),
                "latency_ms_p50": round(1000 * lat[len(lat) // 2], 1),
            }
        )

    baseline = {
        r["context_length"]: r["latency_ms_mean"]
        for r in summary
        if r["condition"] == "naive_long_context"
    }
    for r in summary:
        r["speedup_vs_naive"] = round(baseline[r["context_length"]] / r["latency_ms_mean"], 2)

    tag = args.model.split("/")[-1]
    with (out_dir / f"{tag}_latency.json").open("w") as fh:
        json.dump({"model": args.model, "n": args.n, "summary": summary}, fh, indent=2)

    lines = [
        "# Latency vs naive long-context",
        "",
        f"{args.n} questions per context length, unbatched, greedy, "
        f"{gen.max_new_tokens} new tokens, {args.warmup} warm-up calls discarded.",
        "",
        "| context | condition | prompt tok | latency ms (mean) | p50 | speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['context_length']} | {r['condition']} | {r['prompt_tokens']:.0f} | "
            f"{r['latency_ms_mean']:.0f} | {r['latency_ms_p50']:.0f} | "
            f"{r['speedup_vs_naive']:.2f}x |"
        )
    md = "\n".join(lines)
    (out_dir / f"{tag}_latency.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
