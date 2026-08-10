"""Identify retrieval heads by sweeping needle-in-a-haystack probes.

Usage:
    python -m heads.run_head_id --context-lengths 2000 4000 8000 --n-depths 10
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .capture import AttentionCapture
from .haystack import DEFAULT_HAYSTACK_DIR, PromptBuilder
from .needles import NEEDLES
from .scorer import RetrievalScorer

DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def pick_device() -> str:
    """Choose the GPU with the most free memory; this box shares GPU 0."""
    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--haystack-dir", default=str(DEFAULT_HAYSTACK_DIR))
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--n-depths", type=int, default=10)
    ap.add_argument("--needles", nargs="+", default=None, help="needle ids; default all")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap instances (smoke test)")
    ap.add_argument("--out", default="results/head_id")
    args = ap.parse_args()

    device = args.device or pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]

    print(f"[load] {args.model} -> {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    builder = PromptBuilder(tok, args.haystack_dir)
    needles = [n for n in NEEDLES if args.needles is None or n.id in args.needles]
    depths = [round(i / max(args.n_depths - 1, 1), 4) for i in range(args.n_depths)]

    instances = [
        builder.build(n, ctx, d)
        for ctx in args.context_lengths
        for d in depths
        for n in needles
    ]
    if args.limit:
        instances = instances[: args.limit]
    print(
        f"[plan] {len(instances)} instances = {len(args.context_lengths)} lengths "
        f"x {len(depths)} depths x {len(needles)} needles",
        flush=True,
    )

    results = []
    t0 = time.time()
    with AttentionCapture(model) as cap:
        scorer = RetrievalScorer(model, tok, cap, max_new_tokens=args.max_new_tokens)
        for i, inst in enumerate(instances, 1):
            results.append(scorer.score_instance(inst))
            if i % 10 == 0 or i == len(instances):
                acc = sum(r.answer_found for r in results) / len(results)
                print(
                    f"[{i}/{len(instances)}] {time.time() - t0:.0f}s "
                    f"needle-recall={acc:.2f}",
                    flush=True,
                )

    n_layers, n_heads = len(results[0].scores), len(results[0].scores[0])
    agg = _aggregate(results, n_layers, n_heads, args.threshold)

    with (out_dir / f"{tag}_instances.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    with (out_dir / f"{tag}_head_scores.json").open("w") as f:
        json.dump(
            {
                "model": args.model,
                "n_instances": len(results),
                "n_layers": n_layers,
                "n_heads": n_heads,
                "threshold": args.threshold,
                "context_lengths": args.context_lengths,
                "depths": depths,
                "needle_ids": [n.id for n in needles],
                **agg,
            },
            f,
            indent=2,
        )

    print(f"\n[done] {time.time() - t0:.0f}s -> {out_dir}/{tag}_head_scores.json")
    print(f"[heads] {len(agg['retrieval_heads'])} heads above {args.threshold}")
    print("\nTop 20 heads by mean retrieval score:")
    for h in agg["ranked"][:20]:
        print(f"  L{h['layer']:2d}H{h['head']:2d}  {h['score']:.3f}")


def _aggregate(results, n_layers, n_heads, threshold):
    total = [[0.0] * n_heads for _ in range(n_layers)]
    hits = [[0] * n_heads for _ in range(n_layers)]
    for r in results:
        for l in range(n_layers):
            for h in range(n_heads):
                total[l][h] += r.scores[l][h]
                hits[l][h] += r.hit_steps[l][h]
    n = len(results)
    mean = [[total[l][h] / n for h in range(n_heads)] for l in range(n_layers)]

    # Copy-precision, micro-averaged: of every decode step across the sweep where
    # the emitted token was in the needle, how often did this head's argmax point
    # at it? Unlike |g_h|/|k| this does not depend on how much of the needle the
    # answer happens to cover, so needles whose answer is a short span are not
    # penalised. Micro rather than macro so an instance with one copy step cannot
    # swing the average.
    copyable = sum(r.copyable_steps for r in results)
    precision = [
        [(hits[l][h] / copyable if copyable else 0.0) for h in range(n_heads)]
        for l in range(n_layers)
    ]

    def _precision_over(subset):
        denom = sum(r.copyable_steps for r in subset)
        acc = [[0] * n_heads for _ in range(n_layers)]
        for r in subset:
            for l in range(n_layers):
                for h in range(n_heads):
                    acc[l][h] += r.hit_steps[l][h]
        return [
            [(acc[l][h] / denom if denom else 0.0) for h in range(n_heads)]
            for l in range(n_layers)
        ]

    by_kind = defaultdict(list)
    by_needle = defaultdict(list)
    for r in results:
        by_kind[r.needle_kind].append(r)
        by_needle[r.needle_id].append(r)
    kind_means = {k: _precision_over(rs) for k, rs in by_kind.items()}
    # Reported per needle as well as per kind: a "kind" with a single needle is
    # confounded with that needle's identity, which is exactly how the
    # needle-length artifact first masqueraded as a generalization gap.
    needle_means = {k: _precision_over(rs) for k, rs in by_needle.items()}

    # Ranked by copy-precision (the headline metric); wu_score carried alongside
    # so the published formulation stays visible for comparison.
    ranked = sorted(
        (
            {
                "layer": l,
                "head": h,
                "score": precision[l][h],
                "wu_score": mean[l][h],
            }
            for l in range(n_layers)
            for h in range(n_heads)
        ),
        key=lambda d: -d["score"],
    )
    return {
        "metric": "copy_precision",
        "mean_scores": precision,
        "wu_mean_scores": mean,
        "mean_scores_by_needle_kind": kind_means,
        "mean_scores_by_needle": needle_means,
        "copyable_steps_total": copyable,
        "ranked": ranked,
        "retrieval_heads": [r for r in ranked if r["score"] >= threshold],
        "needle_recall": sum(r.answer_found for r in results) / n,
    }


if __name__ == "__main__":
    main()
