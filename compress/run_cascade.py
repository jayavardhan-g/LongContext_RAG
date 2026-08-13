"""The cascade: make the head signal cheap enough to be worth having.

The accuracy result costs more than it saves. Reading the head signal needs a
forward pass over the whole 32k context -- the exact cost compression exists to
avoid -- so "fusion beats dense by +0.063 F1" is an accuracy claim with a
latency regression hiding behind it.

The cascade fixes the cost without changing the signal:

    dense scores every sentence          (bi-encoder, no LLM)
      -> keep the top S tokens           (shortlist, S << context)
      -> head pass over the shortlist    (prefill S, not 32k)
      -> fuse, then spend the 256 budget

Prefill is the dominant term, so shrinking it from 32k to S is close to an S/32k
speedup on the signal. What it risks is recall: the head signal can only rerank
what dense already retrieved, so the shortlist's full support is a hard ceiling
on the cascade. Milestone 2 puts dense's full support at 0.957 by 2048 tokens,
which is why the sweep starts there and not lower.

Latency is measured in the same run, unbatched, including the cost of computing
the signal -- otherwise the efficiency claim is not honest.

Usage:
    python -m compress.run_cascade --shortlists 1024 2048 4096 --budget 256
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
from baselines.generate import Generator, build_prompt
from data.budget import fill_budget
from data.hotpot import DistractorPool, InstanceBuilder, load_hotpot, render_context
from eval.metrics import exact_match, f1, full_support, gold_recall
from heads.capture import AttentionCapture
from paths import resolve_model

from .head_scorer import HeadAttentionScorer, load_head_list
from .run_fusion_eval import MODEL, fuse

DEFAULT_HEADS = "results/head_id/Meta-Llama-3.1-8B-Instruct_retrieval_heads.json"


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
    ap.add_argument("--weight", type=float, default=0.20)
    ap.add_argument("--shortlists", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--context-length", type=int, default=32000)
    ap.add_argument("--pool-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--latency-n", type=int, default=25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/cascade")
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
    instances = [builder.build(ds[i], args.context_length, seed=rank) for rank, i in enumerate(idx)]

    # Dense scores are reused: the bi-encoder pass is unchanged by the cascade,
    # and re-running it would only add variance to the comparison.
    z = np.load(args.scores)
    offsets = z["offsets"]
    dense_all = [z["dense"][offsets[i]:offsets[i + 1]] for i in range(len(instances))]
    head_full = [z[args.head_key][offsets[i]:offsets[i + 1]] for i in range(len(instances))]

    print(f"[load] {MODEL} -> {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolve_model(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    head_list = load_head_list(args.heads)

    selections: dict[tuple[str, int], list] = {}
    head_pass_ms: dict[int, list[float]] = defaultdict(list)

    with AttentionCapture(model) as cap:
        scorer = HeadAttentionScorer(model, tok, cap, head_list, (5,))
        for S in args.shortlists:
            t0 = time.time()
            for i, inst in enumerate(instances):
                titles = builder.title_tokens(inst)
                short = fill_budget(inst.chunks, dense_all[i].tolist(), S, titles)

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                hs = scorer.score_all(inst.question, short)[args.head_key]
                torch.cuda.synchronize()
                head_pass_ms[S].append((time.perf_counter() - t1) * 1000)

                pos = {(c.para_idx, c.sent_idx): j for j, c in enumerate(inst.chunks)}
                ds_short = np.array(
                    [dense_all[i][pos[(c.para_idx, c.sent_idx)]] for c in short], dtype=np.float32
                )
                fused = fuse(ds_short, np.asarray(hs, dtype=np.float32), args.weight).tolist()
                selections[(f"cascade_S{S}", i)] = fill_budget(short, fused, args.budget, titles)
                selections[(f"cascade_S{S}_denseonly", i)] = fill_budget(
                    short, ds_short.tolist(), args.budget, titles
                )
                if (i + 1) % 50 == 0:
                    print(f"[cascade S={S}] {i + 1}/{len(instances)} {time.time() - t0:.0f}s", flush=True)

        # Full-context head pass, timed on a subset, as the cost reference.
        full_ms = []
        for inst in instances[: args.latency_n]:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            scorer.score_all(inst.question, inst.chunks)
            torch.cuda.synchronize()
            full_ms.append((time.perf_counter() - t1) * 1000)

    # Reference conditions built from the already-computed full-context scores.
    for i, inst in enumerate(instances):
        titles = builder.title_tokens(inst)
        selections[("dense", i)] = fill_budget(inst.chunks, dense_all[i].tolist(), args.budget, titles)
        fused_full = fuse(dense_all[i], head_full[i], args.weight).tolist()
        selections[("fusion_fullctx", i)] = fill_budget(inst.chunks, fused_full, args.budget, titles)

    records, prompts = [], []
    for (method, i), sel in sorted(selections.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        inst = instances[i]
        ctx = render_context(sel)
        records.append({
            "qid": inst.qid, "gold": inst.answer, "method": method, "budget": args.budget,
            "used_tokens": len(tok(ctx, add_special_tokens=False).input_ids),
            "gold_recall": gold_recall(sel, inst.chunks),
            "full_support": full_support(sel, inst.chunks),
        })
        prompts.append(build_prompt(tok, ctx, inst.question))

    print(f"[gen] {len(prompts)} conditions", flush=True)
    gen = Generator(model, tok, max_new_tokens=args.max_new_tokens)
    answers = gen.generate(prompts, batch_size=args.batch_size)
    for rec, pred in zip(records, answers):
        rec["pred"] = pred
        rec["em"] = exact_match(pred, rec["gold"])
        rec["f1"] = f1(pred, rec["gold"])

    # ---- dense encoder latency, so the cascade's total cost is complete ----
    del model
    torch.cuda.empty_cache()
    encoder = DenseEncoder(BGE_M3, device=device)
    enc_ms = []
    for inst in instances[: args.latency_n]:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        encoder.encode([inst.question])
        encoder.encode([c.embed_text for c in inst.chunks])
        torch.cuda.synchronize()
        enc_ms.append((time.perf_counter() - t1) * 1000)

    latency = {
        "dense_encode_ms": round(float(np.mean(enc_ms)), 1),
        "head_pass_fullctx_ms": round(float(np.mean(full_ms)), 1),
        **{f"head_pass_S{S}_ms": round(float(np.mean(v)), 1) for S, v in head_pass_ms.items()},
    }
    print(f"[latency] {latency}", flush=True)

    with (out_dir / "cascade.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    summary = _summarize(records)
    with (out_dir / "cascade_summary.json").open("w") as fh:
        json.dump({
            "model": MODEL, "head_key": args.head_key, "weight": args.weight,
            "shortlists": args.shortlists, "budget": args.budget, "n_questions": args.n,
            "context_length": args.context_length, "latency_ms": latency,
            "runtime_s": round(time.time() - t_start, 1), "summary": summary,
        }, fh, indent=2)

    md = _markdown(summary, latency, args)
    (out_dir / "cascade.md").write_text(md)
    print("\n" + md, flush=True)
    print(f"[done] {time.time() - t_start:.0f}s", flush=True)


def _summarize(records) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[r["method"]].append(r)
    out = []
    for method, rs in sorted(groups.items()):
        n = len(rs)
        out.append({
            "method": method, "n": n,
            "used_tokens": round(sum(r["used_tokens"] for r in rs) / n, 1),
            "gold_recall": round(sum(r["gold_recall"] for r in rs) / n, 4),
            "full_support": round(sum(r["full_support"] for r in rs) / n, 4),
            "em": round(sum(r["em"] for r in rs) / n, 4),
            "f1": round(sum(r["f1"] for r in rs) / n, 4),
        })
    return out


def _markdown(summary, latency, args) -> str:
    lines = [
        "# Cascade: dense shortlist -> head rescore", "",
        f"{args.n} questions at {args.context_length} tokens, final budget "
        f"{args.budget}, fusion weight {args.weight}.", "",
        "| method | used tok | gold recall | full support | EM | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(summary, key=lambda r: -r["f1"]):
        lines.append(
            f"| {r['method']} | {r['used_tokens']:.0f} | {r['gold_recall']:.3f} | "
            f"{r['full_support']:.3f} | {r['em']:.3f} | {r['f1']:.3f} |"
        )
    lines += [
        "", "## Cost of producing the signal (unbatched, per question)", "",
        "| stage | ms |", "|---|---|",
    ]
    for k, v in latency.items():
        lines.append(f"| {k} | {v:.1f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
