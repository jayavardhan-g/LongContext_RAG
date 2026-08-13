"""Placebo controls for the dense+head fusion result.

Fusing dense with the head signal improves evidence selection. Before that can
be attributed to the head signal carrying model-internal information, the cheap
explanations have to be ruled out: the same lift would appear if the head signal
were standing in for sentence length, or if *any* weakly-correlated second
signal helped simply by breaking ties in dense's ranking.

So the same fusion rule is applied to placebos with the same weight:

    length   -- head_sum accumulates mass over tokens, so it is length-biased by
                construction; if gold sentences are longer than distractors, a
                pure length prior would reproduce the lift for free.
    random   -- tests whether any tie-break helps.
    bm25     -- a real but *external* second signal, so it separates "a
                decorrelated signal helps" from "a model-internal one helps".

    python -m compress.analyze_fusion
"""

import argparse
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot
from paths import resolve_model

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
RECALL_KS = (1, 5, 10, 20)


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd else np.zeros_like(x)


def recall_at_k(scores: np.ndarray, gold: np.ndarray, k: int) -> float:
    n_gold = int(gold.sum())
    if n_gold == 0:
        return 0.0
    top = np.argsort(-scores)[:k]
    return float(gold[top].sum()) / n_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/diagnostic/Meta-Llama-3.1-8B-Instruct_scores.npz")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--context-length", type=int, default=32000)
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", type=float, default=0.25, help="weight on the second signal")
    ap.add_argument("--head-key", default="head_sum_k5")
    ap.add_argument("--out", default="results/diagnostic/fusion_controls.md")
    args = ap.parse_args()

    z = np.load(args.scores)
    offsets = z["offsets"]
    gold_all = z["is_gold"]

    # Sentence lengths are not in the npz; rebuilding the instances is the only
    # way to get them, and it is tokenizer-only work (no GPU).
    ds = load_hotpot("validation")
    tok = AutoTokenizer.from_pretrained(resolve_model(MODEL))
    pool = DistractorPool(size=args.pool_size, seed=args.seed)
    builder = InstanceBuilder(tok, pool)
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    lengths = np.concatenate([
        np.array([c.n_tokens for c in builder.build(ds[i], args.context_length, seed=rank).chunks],
                 dtype=np.float32)
        for rank, i in enumerate(idx)
    ])
    if len(lengths) != len(gold_all):
        raise RuntimeError(
            f"rebuilt {len(lengths)} chunks but the score dump has {len(gold_all)} -- "
            "instances do not match; check --n / --pool-size / --seed"
        )

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(len(gold_all)).astype(np.float32)

    w = args.weight
    signals = {
        "dense (alone)": lambda d, s: d,
        args.head_key + " (alone)": lambda d, s: s["head"],
        f"dense + head (w={w})": lambda d, s: (1 - w) * zscore(d) + w * zscore(s["head"]),
        f"dense + length (w={w})": lambda d, s: (1 - w) * zscore(d) + w * zscore(s["length"]),
        f"dense + random (w={w})": lambda d, s: (1 - w) * zscore(d) + w * zscore(s["random"]),
        f"dense + bm25 (w={w})": lambda d, s: (1 - w) * zscore(d) + w * zscore(s["bm25"]),
    }

    totals = {name: {k: 0.0 for k in RECALL_KS} for name in signals}
    n_inst = len(offsets) - 1
    gold_len_ratio = []
    for i in range(n_inst):
        a, b = offsets[i], offsets[i + 1]
        g = gold_all[a:b]
        parts = {
            "head": z[args.head_key][a:b],
            "length": lengths[a:b],
            "random": noise[a:b],
            "bm25": z["bm25"][a:b],
        }
        d = z["dense"][a:b]
        if g.sum():
            gold_len_ratio.append(lengths[a:b][g].mean() / max(lengths[a:b].mean(), 1e-9))
        for name, fn in signals.items():
            s = fn(d, parts)
            for k in RECALL_KS:
                totals[name][k] += recall_at_k(s, g, k)

    lines = [
        "# Fusion controls",
        "",
        f"{n_inst} instances at {args.context_length} tokens, second-signal weight {w}.",
        "",
        f"Gold sentences are {np.mean(gold_len_ratio):.2f}x the mean sentence length, "
        "so a length prior is not a priori useless -- which is why it is tested.",
        "",
        "| ranking signal | " + " | ".join(f"R@{k}" for k in RECALL_KS) + " |",
        "|---" * (len(RECALL_KS) + 1) + "|",
    ]
    for name in signals:
        vals = " | ".join(f"{totals[name][k] / n_inst:.3f}" for k in RECALL_KS)
        lines.append(f"| {name} | {vals} |")
    md = "\n".join(lines) + "\n"

    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
