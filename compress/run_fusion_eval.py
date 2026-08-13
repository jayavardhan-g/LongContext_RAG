"""Does the fusion's retrieval gain survive into QA accuracy?

It is not automatic. Milestone 2 found that pushing dense retrieval's evidence
coverage from 0.61 to 0.96 full support *lowered* F1 from 0.61 to 0.57 -- extra
context costs more than the extra evidence is worth. So a better ranking has to
be shown to produce better answers; it cannot be assumed.

Two things this run does that the diagnostic did not:

  * **Held-out weight selection.** The fusion weight is swept on the first half
    of the questions using retrieval metrics only, then frozen and reported on
    the second half. The diagnostic picked 0.25 from two candidates on the same
    data it reported, which is worth exactly as much as any other number chosen
    after seeing the answer.
  * **Generation at matched budgets**, through the same greedy packer and the
    same generator as every Milestone 2 baseline, so the F1 is comparable to the
    published tables rather than to a re-tuned pipeline.

Chunk scores are reused from the diagnostic's `--dump-scores` npz: the head
signal costs a full-context forward pass per instance and nothing about it
changes here.

Usage:
    python -m compress.run_fusion_eval --budgets 256 512 1024
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

from data.budget import fill_budget
from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot, render_context
from eval.metrics import exact_match, f1, full_support, gold_recall
from paths import resolve_model

from baselines.generate import Generator, build_prompt

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
WEIGHT_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return f"cuda:{min(range(len(out)), key=lambda i: int(out[i]))}"


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd else np.zeros_like(x)


def fuse(dense: np.ndarray, head: np.ndarray, w: float) -> np.ndarray:
    return (1 - w) * zscore(dense) + w * zscore(head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/diagnostic/Meta-Llama-3.1-8B-Instruct_scores.npz")
    ap.add_argument("--head-key", default="head_sum_k5")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--context-length", type=int, default=32000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--select-budget", type=int, default=256,
                    help="budget whose full-support decides the weight on dev")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/diagnostic")
    args = ap.parse_args()

    device = args.device or pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    ds = load_hotpot("validation")
    tok = AutoTokenizer.from_pretrained(resolve_model(MODEL))
    pool = DistractorPool(size=args.pool_size, seed=args.seed)
    builder = InstanceBuilder(tok, pool)
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    instances = [
        builder.build(ds[i], args.context_length, seed=rank) for rank, i in enumerate(idx)
    ]

    z = np.load(args.scores)
    offsets = z["offsets"]
    if len(offsets) - 1 != len(instances):
        raise RuntimeError(
            f"score dump has {len(offsets) - 1} instances, rebuilt {len(instances)}"
        )
    per_inst = [
        {k: z[k][offsets[i]:offsets[i + 1]] for k in ("dense", "reranker", "bm25", args.head_key)}
        for i in range(len(instances))
    ]
    for inst, sc in zip(instances, per_inst):
        if len(sc["dense"]) != len(inst.chunks):
            raise RuntimeError(f"{inst.qid}: score length {len(sc['dense'])} != {len(inst.chunks)} chunks")

    # Dev/test split over questions. The weight is chosen on dev and never
    # re-touched, so the test half is a clean estimate of the fused method.
    half = len(instances) // 2
    dev, test = set(range(half)), set(range(half, len(instances)))

    print(f"[weight] sweeping on {len(dev)} dev questions, "
          f"full_support@{args.select_budget}", flush=True)
    dev_scores = {}
    for w in WEIGHT_GRID:
        vals = []
        for i in sorted(dev):
            inst, sc = instances[i], per_inst[i]
            s = fuse(sc["dense"], sc[args.head_key], w).tolist()
            sel = fill_budget(inst.chunks, s, args.select_budget, builder.title_tokens(inst))
            vals.append(full_support(sel, inst.chunks))
        dev_scores[w] = sum(vals) / len(vals)
        print(f"[weight]   w={w:.2f}  full_support={dev_scores[w]:.4f}", flush=True)
    best_w = max(dev_scores, key=lambda w: dev_scores[w])
    print(f"[weight] chosen w={best_w:.2f} (dev only)", flush=True)

    methods = {
        "dense": lambda sc: sc["dense"].tolist(),
        "reranker": lambda sc: sc["reranker"].tolist(),
        args.head_key: lambda sc: sc[args.head_key].tolist(),
        f"fusion_w{best_w:.2f}": lambda sc: fuse(sc["dense"], sc[args.head_key], best_w).tolist(),
    }

    records, prompts = [], []
    for i, (inst, sc) in enumerate(zip(instances, per_inst)):
        titles = builder.title_tokens(inst)
        for name, fn in methods.items():
            s = fn(sc)
            for budget in args.budgets:
                sel = fill_budget(inst.chunks, s, budget, titles)
                ctx = render_context(sel)
                records.append({
                    "qid": inst.qid,
                    "split": "dev" if i in dev else "test",
                    "gold": inst.answer,
                    "method": name,
                    "budget": budget,
                    "used_tokens": len(tok(ctx, add_special_tokens=False).input_ids),
                    "gold_recall": gold_recall(sel, inst.chunks),
                    "full_support": full_support(sel, inst.chunks),
                })
                prompts.append(build_prompt(tok, ctx, inst.question))

    print(f"[gen] {len(prompts)} conditions", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolve_model(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    gen = Generator(model, tok, max_new_tokens=args.max_new_tokens)
    t0 = time.time()
    answers = gen.generate(prompts, batch_size=args.batch_size)
    print(f"[gen] done in {time.time() - t0:.0f}s", flush=True)

    for rec, pred in zip(records, answers):
        rec["pred"] = pred
        rec["em"] = exact_match(pred, rec["gold"])
        rec["f1"] = f1(pred, rec["gold"])

    with (out_dir / "fusion_eval.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    summary = _summarize(records)
    with (out_dir / "fusion_eval_summary.json").open("w") as fh:
        json.dump({
            "model": MODEL,
            "head_key": args.head_key,
            "weight_grid": list(WEIGHT_GRID),
            "dev_full_support": {str(k): round(v, 4) for k, v in dev_scores.items()},
            "chosen_weight": best_w,
            "n_questions": args.n,
            "context_length": args.context_length,
            "budgets": args.budgets,
            "runtime_s": round(time.time() - t_start, 1),
            "summary": summary,
        }, fh, indent=2)

    md = _markdown(summary, dev_scores, best_w, args)
    (out_dir / "fusion_eval.md").write_text(md)
    print("\n" + md, flush=True)
    print(f"[done] {time.time() - t_start:.0f}s", flush=True)


def _summarize(records) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["split"], r["method"], r["budget"])].append(r)
        groups[("all", r["method"], r["budget"])].append(r)
    out = []
    for (split, method, budget), rs in sorted(groups.items()):
        n = len(rs)
        out.append({
            "split": split, "method": method, "budget": budget, "n": n,
            "used_tokens": round(sum(r["used_tokens"] for r in rs) / n, 1),
            "gold_recall": round(sum(r["gold_recall"] for r in rs) / n, 4),
            "full_support": round(sum(r["full_support"] for r in rs) / n, 4),
            "em": round(sum(r["em"] for r in rs) / n, 4),
            "f1": round(sum(r["f1"] for r in rs) / n, 4),
        })
    return out


def _markdown(summary, dev_scores, best_w, args) -> str:
    lines = [
        "# Fused retrieval -- QA accuracy at matched budgets",
        f"\n{args.n} questions at {args.context_length} tokens, "
        f"head signal `{args.head_key}`.\n",
        "Weight swept on the dev half by full support, then frozen. "
        f"Dev curve: {', '.join(f'{w:.2f}:{v:.3f}' for w, v in dev_scores.items())}. "
        f"**Chosen w = {best_w:.2f}.**\n",
        "`test` is the headline -- the weight never saw it.\n",
    ]
    for split in ("test", "dev", "all"):
        rows = [r for r in summary if r["split"] == split]
        if not rows:
            continue
        lines += [
            f"## {split} (n = {rows[0]['n']} per condition)", "",
            "| method | budget | used tok | gold recall | full support | EM | F1 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (r["budget"], -r["f1"])):
            lines.append(
                f"| {r['method']} | {r['budget']} | {r['used_tokens']:.0f} | "
                f"{r['gold_recall']:.3f} | {r['full_support']:.3f} | "
                f"{r['em']:.3f} | {r['f1']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
