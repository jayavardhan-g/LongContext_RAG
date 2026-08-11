"""Causally verify the head list: mask the identified heads and see if retrieval dies.

This is the verification that matters for this model, because Wu et al. publish
no Llama-3.x head list to compare against. Positional agreement with another
checkpoint would be weak evidence anyway; whether the heads *do the job the label
claims* is the actual question, and it is the same evidence the original paper
uses to call them retrieval heads.

The control is what makes it a test: masking any k heads costs the model some
capacity, so a recall drop alone proves nothing. Random heads of the same count
are masked as a baseline, and the identified heads must hurt far more.

Usage:
    python -m heads.run_ablation --heads results/head_id/<tag>_retrieval_heads.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ablate import HeadMasker, greedy_answer
from .haystack import DEFAULT_HAYSTACK_DIR, PromptBuilder
from .needles import NEEDLES
from .run_head_id import DEFAULT_MODEL, pick_device
from .scorer import _normalize


def build_conditions(ranked, n_layers, n_heads, seeds, sizes):
    """Identified-head conditions plus random-head controls of matched size."""
    top = [(h["layer"], h["head"]) for h in ranked]
    top_set = set(top)
    pool = [
        (l, h)
        for l in range(n_layers)
        for h in range(n_heads)
        if (l, h) not in top_set
    ]

    conditions = [("baseline", [])]
    for k in sizes:
        if k <= len(top):
            conditions.append((f"top-{k} retrieval heads", top[:k]))
    for k in sizes:
        for s in seeds:
            rng = random.Random(s)
            conditions.append((f"random-{k} heads (seed {s})", rng.sample(pool, k)))
    return conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--heads", required=True, help="retrieval_heads.json from the sweep")
    ap.add_argument("--haystack-dir", default=str(DEFAULT_HAYSTACK_DIR))
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[2000, 4000])
    ap.add_argument("--n-depths", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/ablation")
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

    ranked = json.loads(Path(args.heads).read_text())
    builder = PromptBuilder(tok, args.haystack_dir)
    depths = [round(i / max(args.n_depths - 1, 1), 4) for i in range(args.n_depths)]
    instances = [
        builder.build(n, ctx, d)
        for ctx in args.context_lengths
        for d in depths
        for n in NEEDLES
    ]

    masker = HeadMasker(model)
    conditions = build_conditions(
        ranked,
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        args.seeds,
        args.sizes,
    )
    print(
        f"[plan] {len(conditions)} conditions x {len(instances)} instances",
        flush=True,
    )

    results = []
    t0 = time.time()
    for name, heads in conditions:
        correct = 0
        for inst in instances:
            with masker.mask(heads):
                text = greedy_answer(model, tok, inst.input_ids, args.max_new_tokens)
            correct += _normalize(inst.needle.answer) in _normalize(text)
        recall = correct / len(instances)
        results.append({"condition": name, "n_masked": len(heads), "recall": recall,
                        "heads": [list(h) for h in heads]})
        print(f"  {name:32s} recall={recall:.3f}   ({time.time() - t0:.0f}s)", flush=True)

    report = _report(results, instances, args)
    (out_dir / f"{tag}_ablation.json").write_text(json.dumps(results, indent=2))
    (out_dir / f"{tag}_ablation.md").write_text(report)
    print("\n" + report)


def _report(results, instances, args):
    base = next(r["recall"] for r in results if r["condition"] == "baseline")
    lines = [
        "# Head ablation — causal check on the identified retrieval heads",
        "",
        f"- {len(instances)} needle-in-a-haystack instances per condition "
        f"({len(args.context_lengths)} context lengths x {args.n_depths} depths x "
        f"{len(instances) // (len(args.context_lengths) * args.n_depths)} needles)",
        "- metric: needle recall (did the answer appear in the generation)",
        "- heads are zeroed at the input to `o_proj`, removing only that head's "
        "contribution to the residual stream",
        "",
        "| condition | heads masked | recall | drop vs baseline |",
        "|---|---|---|---|",
    ]
    for r in results:
        drop = base - r["recall"]
        drop_s = "—" if r["condition"] == "baseline" else f"{drop:+.3f}"
        lines.append(
            f"| {r['condition']} | {r['n_masked']} | {r['recall']:.3f} | {drop_s} |"
        )

    # Compare each identified-head condition against random controls of equal size.
    lines += ["", "## Identified vs random, at matched mask size", "",
              "| heads masked | identified | random (mean) | gap |", "|---|---|---|---|"]
    for k in args.sizes:
        ident = [r for r in results if r["condition"] == f"top-{k} retrieval heads"]
        rand = [r for r in results if r["condition"].startswith(f"random-{k} ")]
        if not ident or not rand:
            continue
        rmean = sum(r["recall"] for r in rand) / len(rand)
        lines.append(
            f"| {k} | {ident[0]['recall']:.3f} | {rmean:.3f} | "
            f"**{rmean - ident[0]['recall']:+.3f}** |"
        )
    lines += [
        "",
        "A large gap means the identified heads are doing work that randomly "
        "chosen heads are not — which is what earns them the name. A small gap "
        "would mean the head list is not causally special, and the compression "
        "premise of this project would need rethinking.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
