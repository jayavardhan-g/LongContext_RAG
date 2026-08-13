"""Two checks the headline table cannot answer on its own.

**Is the head signal winning by finding the answer rather than the evidence?**
Attention is read at the step that generates the first answer token, so the head
signal is pointed at whatever the model is about to copy. That is a real
retrieval capability, but it means head-selected context can contain the answer
string while missing the annotated supporting facts -- which would explain
`head_sum_k5` scoring F1 0.66 on 0.44 full support. Annotated-evidence recall
would then be the wrong yardstick for it, and saying so matters more than the
number does.

**Are the differences real at n=300?** F1 on the dev and test halves disagrees
about the ranking of the top three methods, and the same method differs by ~0.09
F1 between halves -- so the halves differ in difficulty by more than the methods
differ from each other. Methods are scored on identical questions, so a paired
bootstrap over questions is the right test and is far tighter than the split.

    python -m compress.analyze_eval
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from data.budget import fill_budget
from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot, render_context
from eval.metrics import normalize_answer
from paths import resolve_model

from .run_fusion_eval import MODEL, fuse


def paired_bootstrap(a: list[float], b: list[float], n_boot: int, seed: int) -> tuple[float, float, float, float]:
    """Mean difference a-b with a percentile CI, resampling questions in pairs.

    Paired because both methods answer the same questions: the question-to-
    question variance is large here (the two halves differ by ~0.09 F1 on the
    same method) and pairing removes it from the comparison entirely.
    """
    rng = np.random.default_rng(seed)
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    diff = arr_a - arr_b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided p for H0: mean difference is zero.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/diagnostic/fusion_eval.jsonl")
    ap.add_argument("--scores", default="results/diagnostic/Meta-Llama-3.1-8B-Instruct_scores.npz")
    ap.add_argument("--head-key", default="head_sum_k5")
    ap.add_argument("--weight", type=float, default=0.20)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--context-length", type=int, default=32000)
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/diagnostic/eval_analysis.md")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.eval)]

    # ---- answer containment ------------------------------------------------
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
    signals = {
        "dense": lambda sc: sc["dense"],
        "reranker": lambda sc: sc["reranker"],
        args.head_key: lambda sc: sc[args.head_key],
        f"fusion_w{args.weight:.2f}": lambda sc: fuse(sc["dense"], sc[args.head_key], args.weight),
    }

    contains = defaultdict(list)
    for i, inst in enumerate(instances):
        sc = {k: z[k][offsets[i]:offsets[i + 1]] for k in ("dense", "reranker", "bm25", args.head_key)}
        titles = builder.title_tokens(inst)
        gold_norm = normalize_answer(inst.answer)
        for name, fn in signals.items():
            sel = fill_budget(inst.chunks, fn(sc).tolist(), args.budget, titles)
            ctx = normalize_answer(render_context(sel))
            # yes/no answers are not spans and cannot be "contained"; excluded so
            # they do not inflate every method equally.
            if gold_norm not in ("yes", "no"):
                contains[name].append(float(gold_norm in ctx))

    # ---- paired bootstrap --------------------------------------------------
    at_budget = [r for r in records if r["budget"] == args.budget]
    by_method = defaultdict(dict)
    for r in at_budget:
        by_method[r["method"]][r["qid"]] = r["f1"]
    qids = sorted(set.intersection(*[set(v) for v in by_method.values()]))
    f1s = {m: [by_method[m][q] for q in qids] for m in by_method}

    head, fusion = args.head_key, f"fusion_w{args.weight:.2f}"
    pairs = [
        (fusion, "dense"), (head, "dense"), ("reranker", "dense"),
        (fusion, "reranker"), (fusion, head), (head, "reranker"),
    ]

    lines = [
        "# Post-hoc analysis of the fused-retrieval evaluation", "",
        f"Budget {args.budget}, {len(qids)} questions, {args.n_boot} bootstrap resamples.", "",
        "## Does the selected context contain the answer string?", "",
        "Span answers only (yes/no excluded). This is the yardstick the head "
        "signal is actually optimising, as opposed to annotated supporting facts.", "",
        "| method | answer present in context |", "|---|---|",
    ]
    for name in signals:
        v = contains[name]
        lines.append(f"| {name} | {sum(v) / len(v):.3f} |")

    lines += [
        "", f"## Paired bootstrap on F1 (budget {args.budget})", "",
        "| comparison | mean F1 diff | 95% CI | p |", "|---|---|---|---|",
    ]
    for a, b in pairs:
        if a not in f1s or b not in f1s:
            continue
        d, lo, hi, p = paired_bootstrap(f1s[a], f1s[b], args.n_boot, args.seed)
        sig = "" if lo <= 0 <= hi else " **"
        lines.append(f"| {a} vs {b} | {d:+.4f}{sig} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} |")
    lines += ["", "`**` marks a CI that excludes zero.", ""]

    md = "\n".join(lines)
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
