"""Turn a head-score sweep into the documented head list and its figures.

Usage:
    python -m heads.analyze --scores results/head_id/<tag>_head_scores.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Retrieval score is a magnitude, so the ramp is single-hue light->dark.
SEQUENTIAL = "Blues"
METRIC_LABEL = "copy-precision"
INK = "#1a1a1a"
MUTED = "#6b7280"


def heatmap(mean: np.ndarray, out_path: Path, title: str, threshold: float):
    n_layers, n_heads = mean.shape
    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(mean, aspect="auto", cmap=SEQUENTIAL, vmin=0.0, origin="lower")

    # Ring the heads that clear the threshold; color alone should not carry it.
    ys, xs = np.where(mean >= threshold)
    ax.scatter(xs, ys, s=26, facecolors="none", edgecolors="#b45309", linewidths=1.2,
               label=f"retrieval head (>= {threshold})")

    ax.set_xlabel("head", color=INK)
    ax.set_ylabel("layer", color=INK)
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.set_xticks(range(0, n_heads, 4))
    ax.set_yticks(range(0, n_layers, 4))
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if len(xs):
        ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(METRIC_LABEL, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def kind_consistency(data, out_path: Path, top_heads, threshold: float):
    """Do the top heads hold up across needle types, or are they surface artifacts?"""
    kinds = sorted(data["mean_scores_by_needle_kind"])
    if len(kinds) < 2:
        return None
    labels = [f"L{h['layer']}H{h['head']}" for h in top_heads]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(labels))
    width = 0.8 / len(kinds)
    # Single hue, stepped light->dark: the kinds are ordered facets of one measure.
    shades = plt.get_cmap(SEQUENTIAL)(np.linspace(0.35, 0.85, len(kinds)))
    for i, kind in enumerate(kinds):
        m = np.array(data["mean_scores_by_needle_kind"][kind])
        vals = [m[h["layer"], h["head"]] for h in top_heads]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
               label=kind.replace("_", " "), color=shades[i], edgecolor="white", linewidth=0.6)
    ax.axhline(threshold, color="#b45309", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", color=INK, fontsize=8)
    ax.set_ylabel(METRIC_LABEL, color=INK)
    ax.set_title("Do the top heads survive a change of needle type?", color=INK, fontsize=12, pad=10)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=len(kinds), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return kinds


def answered_only(instances_path, shape):
    """Recompute mean scores over just the instances the model actually answered.

    A head that looks weak only because the model failed to retrieve at all is a
    different situation from a head that stays quiet while retrieval succeeds;
    the whole-sweep average cannot tell those apart.
    """
    rows = [json.loads(l) for l in Path(instances_path).read_text().splitlines() if l.strip()]
    answered = [r for r in rows if r["answer_found"]]
    if not answered:
        return None
    # Must be the same metric as the headline table, micro-averaged the same
    # way -- comparing an answered-only Wu score against an all-instance
    # copy-precision would be meaningless.
    acc = np.zeros(shape)
    denom = 0
    for r in answered:
        acc += np.array(r["hit_steps"])
        denom += r.get("copyable_steps", 0)
    if not denom:
        return None
    return acc / denom, len(answered), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--instances", default=None,
                    help="per-instance jsonl; enables the answered-only recompute")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    path = Path(args.scores)
    data = json.loads(path.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = path.stem.replace("_head_scores", "")
    mean = np.array(data["mean_scores"])
    threshold = data["threshold"]

    heatmap(
        mean,
        out_dir / f"{tag}_head_heatmap.png",
        f"Copy-precision per attention head — {data['model'].split('/')[-1]}\n"
        f"{data['n_instances']} needle-in-a-haystack instances",
        threshold,
    )
    top = data["ranked"][: args.top_n]
    kinds = kind_consistency(data, out_dir / f"{tag}_kind_consistency.png", top, threshold)
    answered = answered_only(args.instances, mean.shape) if args.instances else None

    lines = [
        f"# Retrieval heads — {data['model']}",
        "",
        f"- instances: {data['n_instances']} "
        f"({len(data['context_lengths'])} context lengths x {len(data['depths'])} depths "
        f"x {len(data['needle_ids'])} needles)",
        f"- context lengths: {data['context_lengths']}",
        f"- needle recall (model actually answered): {data['needle_recall']:.3f}",
        f"- threshold: {threshold}",
        f"- heads above threshold: {len(data['retrieval_heads'])} "
        f"of {data['n_layers'] * data['n_heads']}",
        "",
        f"## Top {args.top_n} heads",
        "",
        "Ranked by **copy-precision**: of the decode steps where the emitted "
        "token was in the needle, the fraction where this head's attention "
        "argmax pointed at it. Wu et al.'s `|g_h|/|k|` is shown alongside — it "
        "divides by the whole needle, so it under-rates heads on needles whose "
        "answer covers only a short span.",
        "",
        "| rank | layer | head | copy-precision | Wu et al. score |",
        "|---|---|---|---|---|",
    ]
    for i, h in enumerate(top, 1):
        wu = h.get("wu_score")
        wu_s = f"{wu:.4f}" if wu is not None else "-"
        lines.append(
            f"| {i} | {h['layer']} | {h['head']} | {h['score']:.4f} | {wu_s} |"
        )

    if answered:
        am, n_hit, n_all = answered
        heatmap(am, out_dir / f"{tag}_head_heatmap_answered.png",
                "Copy-precision, answered instances only "
                f"({n_hit}/{n_all})", threshold)
        order = np.argsort(am, axis=None)[::-1][: args.top_n]
        lines += ["", f"## Answered-only recompute ({n_hit}/{n_all} instances)", "",
                  "Ranking over just the instances where the model produced the "
                  "answer, so heads are not penalised for retrieval failures.", "",
                  "| layer | head | answered-only | all-instance |", "|---|---|---|---|"]
        for idx in order:
            l, h = int(idx // am.shape[1]), int(idx % am.shape[1])
            lines.append(f"| {l} | {h} | {am[l, h]:.4f} | {mean[l, h]:.4f} |")
        top_all = {(h["layer"], h["head"]) for h in top}
        top_ans = {(int(i // am.shape[1]), int(i % am.shape[1])) for i in order}
        lines += ["", f"Overlap of the two top-{args.top_n} lists: "
                      f"**{len(top_all & top_ans)}/{args.top_n}**."]

    for key, title in (("mean_scores_by_needle_kind", "needle kind"),
                       ("mean_scores_by_needle", "individual needle")):
        groups = sorted(data.get(key, {}))
        if len(groups) < 2:
            continue
        lines += ["", f"## Copy-precision by {title} (top heads)", "",
                  "| layer | head | " + " | ".join(g.replace('_', ' ') for g in groups) + " |",
                  "|---" * (2 + len(groups)) + "|"]
        for h in top:
            row = [f"{np.array(data[key][g])[h['layer'], h['head']]:.3f}" for g in groups]
            lines.append(f"| {h['layer']} | {h['head']} | " + " | ".join(row) + " |")

    (out_dir / f"{tag}_head_list.md").write_text("\n".join(lines) + "\n")
    json.dump(
        [{"layer": h["layer"], "head": h["head"], "score": h["score"]}
         for h in data["retrieval_heads"]],
        (out_dir / f"{tag}_retrieval_heads.json").open("w"),
        indent=2,
    )
    print("\n".join(lines[:14]))
    print(f"\n[wrote] {out_dir}/{tag}_head_list.md, _head_heatmap.png, _retrieval_heads.json")


if __name__ == "__main__":
    main()
