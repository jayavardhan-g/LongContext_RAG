"""Milestone 2 driver: dense-retrieval RAG and naive long-context baselines.

Usage:
    python -m baselines.run_baselines --n 200 --context-lengths 4000 16000 \
        --budgets 256 512 1024 2048

Runs in two phases so the retriever and the generator never sit on the GPU at
once, and so chunk scoring -- which does not depend on the token budget -- is
paid for once per instance rather than once per budget.
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
from eval.metrics import exact_match, f1, full_support, gold_recall, recall_at_k
from paths import resolve_model

from .encoders import BGE_M3, DenseEncoder
from .generate import Generator, build_prompt
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    DocumentOrderRetriever,
    OracleRetriever,
    RandomRetriever,
    rrf,
)

DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
RECALL_KS = (1, 2, 5, 10, 20, 50)


def pick_device() -> str:
    """Freest GPU by nvidia-smi -- avoids initialising a CUDA context on all 8."""
    if not torch.cuda.is_available():
        return "cpu"
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return f"cuda:{min(range(len(out)), key=lambda i: int(out[i]))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--encoder", default=BGE_M3)
    ap.add_argument("--n", type=int, default=200, help="questions to evaluate")
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[4000, 16000])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--gold-depth", type=float, default=None, help="None = random")
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--skip-naive", action="store_true", help="omit full-context runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()

    device = args.device or pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[data] loading HotpotQA distractor + pool ({args.pool_size} train ex)", flush=True)
    ds = load_hotpot("validation")
    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    pool = DistractorPool(size=args.pool_size, seed=args.seed)
    builder = InstanceBuilder(tok, pool)
    print(f"[data] pool has {len(pool)} unique paragraphs", flush=True)

    idx = list(range(len(ds)))
    import random as _random

    _random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    instances = []
    for ctx_len in args.context_lengths:
        for rank, i in enumerate(idx):
            instances.append(
                builder.build(ds[i], ctx_len, gold_depth=args.gold_depth, seed=rank)
            )
    print(
        f"[data] {len(instances)} instances "
        f"({args.n} questions x {len(args.context_lengths)} lengths)",
        flush=True,
    )
    lens = defaultdict(list)
    for inst in instances:
        lens[inst.target_tokens].append(inst.context_tokens)
    for k in sorted(lens):
        v = lens[k]
        print(
            f"       target {k:6d} -> actual mean {sum(v) / len(v):7.0f} "
            f"(min {min(v)}, max {max(v)})",
            flush=True,
        )

    # ---- Phase 1: score chunks -------------------------------------------
    print(f"[score] loading encoder {args.encoder} -> {device}", flush=True)
    encoder = DenseEncoder(args.encoder, device=device)
    retrievers = [
        DocumentOrderRetriever(),
        BM25Retriever(),
        DenseRetriever(encoder),
        RandomRetriever(seed=args.seed),
        OracleRetriever(),
    ]

    scores: list[dict[str, list[float]]] = []
    t0 = time.time()
    for n, inst in enumerate(instances, 1):
        s = {r.name: r.score(inst.question, inst.chunks) for r in retrievers}
        s["hybrid"] = rrf([s["dense"], s["bm25"]])
        scores.append(s)
        if n % 25 == 0 or n == len(instances):
            print(f"[score] {n}/{len(instances)}  {time.time() - t0:.0f}s", flush=True)

    del encoder, retrievers
    torch.cuda.empty_cache()

    # ---- Assemble every (instance, method, budget) condition --------------
    records, prompts = [], []
    for inst, sc in zip(instances, scores):
        titles = builder.title_tokens(inst)
        full_ctx = render_context(inst.chunks)
        # Measured on the rendered string, not summed from chunk counts: BPE
        # merges across sentence boundaries, so the sum drifts ~5% low and would
        # overstate every compression ratio computed against it.
        full_tokens = len(tok(full_ctx, add_special_tokens=False).input_ids)

        def record(method, budget, sel, ctx, ranking):
            n_ctx = len(tok(ctx, add_special_tokens=False).input_ids)
            rec = {
                "qid": inst.qid,
                "question": inst.question,
                "gold": inst.answer,
                "qtype": inst.qtype,
                "level": inst.level,
                "context_length": inst.target_tokens,
                "full_context_tokens": full_tokens,
                "method": method,
                "budget": budget,
                "used_tokens": n_ctx,
                "n_selected": len(sel),
                "n_chunks": len(inst.chunks),
                "gold_recall": gold_recall(sel, inst.chunks),
                "full_support": full_support(sel, inst.chunks),
            }
            if ranking is not None:
                rec.update(
                    {f"recall@{k}": recall_at_k(ranking, inst.chunks, k) for k in RECALL_KS}
                )
            records.append(rec)
            prompts.append(build_prompt(tok, ctx, inst.question))

        for method, s in sc.items():
            for budget in args.budgets:
                sel = fill_budget(inst.chunks, s, budget, titles)
                record(method, budget, sel, render_context(sel), s)

        if not args.skip_naive:
            # Budget keyed to the target length rather than this instance's own
            # token count, so every naive run at a given length lands in one
            # group instead of one group per instance.
            # Recall@k is left off: naive ranks nothing, it takes everything.
            record(
                "naive_long_context",
                inst.target_tokens,
                inst.chunks,
                full_ctx,
                None,
            )

    print(f"[gen] {len(prompts)} conditions to generate", flush=True)

    # ---- Phase 2: generate -----------------------------------------------
    print(f"[gen] loading {args.model} -> {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
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

    tag = args.model.split("/")[-1]
    raw_path = out_dir / f"{tag}_baselines.jsonl"
    with raw_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    summary = _summarize(records)
    with (out_dir / f"{tag}_baselines_summary.json").open("w") as fh:
        json.dump(
            {
                "model": args.model,
                "encoder": args.encoder,
                "n_questions": args.n,
                "context_lengths": args.context_lengths,
                "budgets": args.budgets,
                "gold_depth": args.gold_depth,
                "seed": args.seed,
                "runtime_s": round(time.time() - t_start, 1),
                "summary": summary,
            },
            fh,
            indent=2,
        )

    md = _markdown(summary, args)
    (out_dir / f"{tag}_baselines.md").write_text(md)
    print("\n" + md)
    print(f"[done] {time.time() - t_start:.0f}s -> {raw_path}")


def _summarize(records) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["context_length"], r["method"], r["budget"])].append(r)

    out = []
    for (ctx_len, method, budget), rs in sorted(groups.items()):
        n = len(rs)
        out.append(
            {
                "context_length": ctx_len,
                "method": method,
                "budget": budget,
                "n": n,
                "used_tokens": round(sum(r["used_tokens"] for r in rs) / n, 1),
                "compression": round(
                    1 - sum(r["used_tokens"] for r in rs)
                    / sum(r["full_context_tokens"] for r in rs),
                    4,
                ),
                "gold_recall": round(sum(r["gold_recall"] for r in rs) / n, 4),
                "full_support": round(sum(r["full_support"] for r in rs) / n, 4),
                "em": round(sum(r["em"] for r in rs) / n, 4),
                "f1": round(sum(r["f1"] for r in rs) / n, 4),
                **{
                    f"recall@{k}": round(sum(r[f"recall@{k}"] for r in rs) / n, 4)
                    for k in RECALL_KS
                    if f"recall@{k}" in rs[0]
                },
            }
        )
    return out


def _markdown(summary, args) -> str:
    lines = [
        "# Milestone 2 -- baselines on pooled-distractor HotpotQA",
        "",
        f"{args.n} questions, seed {args.seed}, gold depth "
        f"{'random' if args.gold_depth is None else args.gold_depth}.",
        "",
        "`compression` is 1 - used/full, so higher means a smaller context.",
        "`full_support` is the fraction of questions where *every* gold sentence",
        "survived selection -- the metric that matters for multi-hop.",
        "",
    ]
    by_len = defaultdict(list)
    for row in summary:
        by_len[row["context_length"]].append(row)

    for ctx_len in sorted(by_len):
        lines += [
            f"## Context length {ctx_len}",
            "",
            "| method | budget | used tok | compression | gold recall | full support | EM | F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(by_len[ctx_len], key=lambda r: (r["method"], r["budget"])):
            lines.append(
                f"| {r['method']} | {r['budget']} | {r['used_tokens']:.0f} | "
                f"{r['compression']:.3f} | {r['gold_recall']:.3f} | "
                f"{r['full_support']:.3f} | {r['em']:.3f} | {r['f1']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
