"""Does the retrieval-head signal rank evidence better than embedding similarity?

Scoring only -- no generation. The question this answers is whether the head
signal is worth building a mechanism around, and that is decided entirely by
where it ranks the gold sentences. Generating answers as well would multiply the
cost by an order of magnitude to confirm something the ranking already settles.

Two numbers decide it:

  * **Ranking quality vs dense and vs a cross-encoder**, at matched budget.
    Recall@k, plus gold recall / full support / gold-token precision after the
    shared greedy packer has spent a 256-token budget. Precision is reported
    because Milestone 2 found that raising evidence coverage past ~0.78 full
    support *lowers* F1 -- so the target is a cleaner 256 tokens, not more gold.

  * **Whether the signal survives the questions the model gets wrong.** The head
    signal needs a forward pass over the full context to exist, so if it points
    at the gold only when the model already answered correctly from that same
    context, it is circular and cannot beat naive long-context on accuracy. The
    run splits every metric by the naive long-context outcome from Milestone 2
    to test exactly that.

Usage:
    python -m compress.run_diagnostic --n 300 --context-lengths 32000 \
        --pool-size 8000 --out results/diagnostic
"""

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.encoders import BGE_M3, DenseEncoder
from baselines.rerankers import BGE_RERANKER, CrossEncoderReranker
from baselines.retrievers import BM25Retriever, DenseRetriever, OracleRetriever, rrf
from data.budget import fill_budget
from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot
from eval.metrics import full_support, gold_recall, recall_at_k
from heads.capture import AttentionCapture
from paths import resolve_model

from .head_scorer import HeadAttentionScorer, load_head_list

DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_HEADS = "results/head_id/Meta-Llama-3.1-8B-Instruct_retrieval_heads.json"
DEFAULT_NAIVE = "results/baselines/Meta-Llama-3.1-8B-Instruct_baselines.jsonl"
RECALL_KS = (1, 2, 5, 10, 20, 50)


def pick_device() -> str:
    """Freest GPU by nvidia-smi -- avoids initialising a CUDA context on all 8."""
    if not torch.cuda.is_available():
        return "cpu"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return f"cuda:{min(range(len(out)), key=lambda i: int(out[i]))}"


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks, ties shared -- the rank transform Spearman is defined on."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, implemented here to avoid a scipy dependency."""
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def zscore(xs: list[float]) -> list[float]:
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5
    return [(x - m) / sd for x in xs] if sd else [0.0] * n


def add_fusions(sc: dict[str, list[float]], head_key: str) -> None:
    """Fuse the head signal with dense two different ways.

    Two families, because a null result from one fusion rule is a fact about the
    rule as much as about the signals. RRF is scale-free (M2's argument: cosine
    and unbounded BM25 are not comparable); the z-scored sum keeps score
    magnitude, which RRF throws away. `rrf_dense_reranker` is the control -- it
    says whether fusion helps *at all* here when both inputs are strong.
    """
    if head_key not in sc:
        return
    sc["fuse_rrf_dense_head"] = rrf([sc["dense"], sc[head_key]])
    zd, zh = zscore(sc["dense"]), zscore(sc[head_key])
    sc["fuse_z_dense_head"] = [0.5 * d + 0.5 * h for d, h in zip(zd, zh)]
    sc["fuse_z_dense_head_25"] = [0.75 * d + 0.25 * h for d, h in zip(zd, zh)]
    if "reranker" in sc:
        sc["fuse_rrf_dense_reranker"] = rrf([sc["dense"], sc["reranker"]])


def gold_token_precision(selected) -> float:
    """Share of the selected tokens that are gold.

    The complement of this is what the generator has to read past to find the
    evidence, and Milestone 2 showed that is what actually costs F1.
    """
    total = sum(c.n_tokens for c in selected)
    if total == 0:
        return 0.0
    return sum(c.n_tokens for c in selected if c.is_gold) / total


def load_naive_outcomes(path: Path) -> dict[tuple[str, int], float]:
    """(qid, context_length) -> F1 of the naive long-context condition."""
    if not path.is_file():
        return {}
    out: dict[tuple[str, int], float] = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("method") == "naive_long_context":
                out[(r["qid"], r["context_length"])] = r.get("f1", 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--encoder", default=BGE_M3)
    ap.add_argument("--reranker", default=BGE_RERANKER)
    ap.add_argument("--heads", default=DEFAULT_HEADS)
    ap.add_argument("--naive-from", default=DEFAULT_NAIVE)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[32000])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--gold-depth", type=float, default=None)
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-reranker", action="store_true")
    ap.add_argument("--fuse-with", default="head_sum_k5", help="head variant to fuse with dense")
    ap.add_argument("--dump-scores", action="store_true", help="write raw per-chunk scores")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/diagnostic")
    args = ap.parse_args()

    device = args.device or pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"[dev] {device}", flush=True)

    ds = load_hotpot("validation")
    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    pool = DistractorPool(size=args.pool_size, seed=args.seed)
    builder = InstanceBuilder(tok, pool)

    # Same shuffle and same per-instance seed as run_baselines, so instances are
    # identical and the numbers here line up with the Milestone 2 tables.
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    instances = [
        builder.build(ds[i], ctx_len, gold_depth=args.gold_depth, seed=rank)
        for ctx_len in args.context_lengths
        for rank, i in enumerate(idx)
    ]
    print(f"[data] {len(instances)} instances, mean {sum(i.context_tokens for i in instances) / len(instances):.0f} ctx tokens", flush=True)

    scores: list[dict[str, list[float]]] = [dict() for _ in instances]

    # ---- Phase 1: retrieval baselines ------------------------------------
    print(f"[score] encoder {args.encoder}", flush=True)
    encoder = DenseEncoder(args.encoder, device=device)
    retrievers = [BM25Retriever(), DenseRetriever(encoder), OracleRetriever()]
    if not args.skip_reranker:
        retrievers.append(CrossEncoderReranker(args.reranker, device=device))

    t0 = time.time()
    for n, (inst, sc) in enumerate(zip(instances, scores), 1):
        for r in retrievers:
            sc[r.name] = r.score(inst.question, inst.chunks)
        if n % 25 == 0 or n == len(instances):
            print(f"[score] {n}/{len(instances)}  {time.time() - t0:.0f}s", flush=True)

    del encoder, retrievers
    torch.cuda.empty_cache()

    # ---- Phase 2: head signal --------------------------------------------
    print(f"[heads] loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    head_list = load_head_list(args.heads)
    print(f"[heads] top-{max(args.k_values)} of {len(head_list)}: "
          f"{['L%dH%d' % h for h in head_list[:5]]}", flush=True)

    t0 = time.time()
    with AttentionCapture(model) as cap:
        scorer = HeadAttentionScorer(model, tok, cap, head_list, tuple(args.k_values))
        for n, (inst, sc) in enumerate(zip(instances, scores), 1):
            sc.update(scorer.score_all(inst.question, inst.chunks))
            if n % 25 == 0 or n == len(instances):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"[heads] {n}/{len(instances)}  {time.time() - t0:.0f}s "
                      f"eta {(len(instances) - n) / max(rate, 1e-9):.0f}s", flush=True)
    variant_names = scorer.variant_names()
    del model, scorer
    torch.cuda.empty_cache()

    # ---- Fusion + signal correlation --------------------------------------
    for sc in scores:
        add_fusions(sc, args.fuse_with)

    # Is the head signal actually decorrelated from dense, or merely noisy? The
    # split-by-naive-outcome result only hints at this; the rank correlation is
    # what distinguishes "orthogonal signal" from "less sensitive to anything".
    pairs = [("dense", args.fuse_with), ("reranker", args.fuse_with), ("dense", "reranker")]
    correlations = {}
    for a, b in pairs:
        vals = [spearman(sc[a], sc[b]) for sc in scores if a in sc and b in sc]
        if vals:
            correlations[f"{a}~{b}"] = round(sum(vals) / len(vals), 4)
    # Restricted to gold chunks: two signals can agree on the mass of irrelevant
    # sentences while disagreeing on the few that matter.
    gold_corr = []
    for inst, sc in zip(instances, scores):
        gi = [i for i, c in enumerate(inst.chunks) if c.is_gold]
        if len(gi) > 1 and args.fuse_with in sc:
            gold_corr.append(
                spearman([sc["dense"][i] for i in gi], [sc[args.fuse_with][i] for i in gi])
            )
    if gold_corr:
        correlations[f"dense~{args.fuse_with}(gold only)"] = round(
            sum(gold_corr) / len(gold_corr), 4
        )
    print(f"[corr] {correlations}", flush=True)

    if args.dump_scores:
        import numpy as np

        keep = ["dense", "reranker", "bm25", args.fuse_with]
        flat = {k: [] for k in keep}
        is_gold, offsets = [], [0]
        for inst, sc in zip(instances, scores):
            for k in keep:
                flat[k] += sc.get(k, [0.0] * len(inst.chunks))
            is_gold += [c.is_gold for c in inst.chunks]
            offsets.append(len(is_gold))
        np.savez_compressed(
            out_dir / f"{args.model.split('/')[-1]}_scores.npz",
            is_gold=np.array(is_gold, dtype=bool),
            offsets=np.array(offsets, dtype=np.int64),
            **{k: np.array(v, dtype=np.float32) for k, v in flat.items()},
        )

    # ---- Metrics ----------------------------------------------------------
    naive = load_naive_outcomes(Path(args.naive_from))
    if not naive:
        print(f"[warn] no naive outcomes at {args.naive_from}; skipping the split", flush=True)

    records = []
    for inst, sc in zip(instances, scores):
        titles = builder.title_tokens(inst)
        naive_f1 = naive.get((inst.qid, inst.target_tokens))
        for method, s in sc.items():
            rec = {
                "qid": inst.qid,
                "context_length": inst.target_tokens,
                "method": method,
                "n_chunks": len(inst.chunks),
                "n_gold": inst.n_gold,
                "naive_f1": naive_f1,
                **{f"recall@{k}": recall_at_k(s, inst.chunks, k) for k in RECALL_KS},
            }
            for budget in args.budgets:
                sel = fill_budget(inst.chunks, s, budget, titles)
                rec[f"gold_recall@{budget}"] = gold_recall(sel, inst.chunks)
                rec[f"full_support@{budget}"] = full_support(sel, inst.chunks)
                rec[f"precision@{budget}"] = gold_token_precision(sel)
            records.append(rec)

    tag = args.model.split("/")[-1]
    raw = out_dir / f"{tag}_diagnostic.jsonl"
    with raw.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    summary = _summarize(records, args.budgets)
    payload = {
        "model": args.model,
        "encoder": args.encoder,
        "reranker": None if args.skip_reranker else args.reranker,
        "heads": args.heads,
        "n_questions": args.n,
        "context_lengths": args.context_lengths,
        "budgets": args.budgets,
        "k_values": args.k_values,
        "seed": args.seed,
        "runtime_s": round(time.time() - t_start, 1),
        "correlations": correlations,
        "summary": summary,
    }
    with (out_dir / f"{tag}_diagnostic_summary.json").open("w") as fh:
        json.dump(payload, fh, indent=2)

    md = _markdown(summary, records, args, variant_names, correlations)
    (out_dir / f"{tag}_diagnostic.md").write_text(md)
    print("\n" + md, flush=True)
    print(f"[done] {time.time() - t_start:.0f}s -> {raw}", flush=True)


def _mean(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _summarize(records, budgets) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["context_length"], r["method"])].append(r)

    keys = [f"recall@{k}" for k in RECALL_KS]
    for b in budgets:
        keys += [f"gold_recall@{b}", f"full_support@{b}", f"precision@{b}"]

    out = []
    for (ctx_len, method), rs in sorted(groups.items()):
        row = {"context_length": ctx_len, "method": method, "n": len(rs)}
        row.update({k: round(_mean(rs, k), 4) for k in keys})

        # The circularity test: does the ranking hold up on the questions the
        # model failed to answer from the very context the signal is read from?
        scored = [r for r in rs if r["naive_f1"] is not None]
        for label, subset in (
            ("naive_ok", [r for r in scored if r["naive_f1"] >= 0.5]),
            ("naive_bad", [r for r in scored if r["naive_f1"] < 0.5]),
        ):
            row[f"n_{label}"] = len(subset)
            if subset:
                row[f"recall@5_{label}"] = round(_mean(subset, "recall@5"), 4)
                row[f"full_support@{budgets[0]}_{label}"] = round(
                    _mean(subset, f"full_support@{budgets[0]}"), 4
                )
        out.append(row)
    return out


def _markdown(summary, records, args, variant_names, correlations=None) -> str:
    b0 = args.budgets[0]
    lines = [
        "# Head-signal diagnostic (scoring only, no generation)",
        "",
        f"{args.n} questions, seed {args.seed}, heads from `{args.heads}`.",
        "",
        "`precision@N` is the share of selected tokens that are gold -- the "
        "quantity Milestone 2 found to matter more than recall.",
        "",
    ]
    if correlations:
        lines += [
            "## Signal correlation (mean Spearman over instances)",
            "",
            "Distinguishes an orthogonal signal from one that is merely less "
            "sensitive to everything.",
            "",
            "| pair | rho |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v:+.3f} |" for k, v in correlations.items()]
        lines.append("")

    by_len = defaultdict(list)
    for row in summary:
        by_len[row["context_length"]].append(row)

    fusions = [
        "fuse_rrf_dense_head",
        "fuse_z_dense_head",
        "fuse_z_dense_head_25",
        "fuse_rrf_dense_reranker",
    ]
    order = ["oracle", "dense", "reranker", "bm25"] + fusions + variant_names
    rank = {m: i for i, m in enumerate(order)}

    for ctx_len in sorted(by_len):
        rows = sorted(by_len[ctx_len], key=lambda r: rank.get(r["method"], 999))
        lines += [
            f"## Context length {ctx_len}",
            "",
            f"| method | R@1 | R@5 | R@10 | gold recall@{b0} | full support@{b0} | precision@{b0} |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['method']} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | "
                f"{r['recall@10']:.3f} | {r[f'gold_recall@{b0}']:.3f} | "
                f"{r[f'full_support@{b0}']:.3f} | {r[f'precision@{b0}']:.3f} |"
            )
        lines += [
            "",
            "### Split by the naive long-context outcome",
            "",
            "If a method's ranking collapses in the `naive_bad` column it is "
            "circular: it finds the evidence only when the model already used it.",
            "",
            f"| method | R@5 (naive ok) | R@5 (naive bad) | full support@{b0} (ok) | full support@{b0} (bad) |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            ok = r.get("recall@5_naive_ok")
            bad = r.get("recall@5_naive_bad")
            if ok is None and bad is None:
                continue
            fs_ok = r.get(f"full_support@{b0}_naive_ok")
            fs_bad = r.get(f"full_support@{b0}_naive_bad")
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            lines.append(
                f"| {r['method']} | {fmt(ok)} | {fmt(bad)} | {fmt(fs_ok)} | {fmt(fs_bad)} |"
            )
        n_ok = rows[0].get("n_naive_ok", 0) if rows else 0
        n_bad = rows[0].get("n_naive_bad", 0) if rows else 0
        lines += ["", f"n = {n_ok} naive-correct, {n_bad} naive-wrong.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
