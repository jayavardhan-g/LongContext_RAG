"""Accuracy-vs-compression figures.

    python -m eval.plot --summary results/baselines/<tag>_baselines_summary.json

Three competing retrievers get colour; the oracle ceiling, the naive
long-context point and the truncation floor are references, drawn in neutral
grey and separated by dash pattern rather than hue.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Slots 1-3 of the reference categorical palette, the subset documented as
# clearing the all-pairs CVD and normal-vision floors in both modes.
SERIES = {
    "dense": ("#2a78d6", "Dense (bge-m3)"),
    "bm25": ("#eb6834", "BM25"),
    "hybrid": ("#1baf7a", "Hybrid (RRF)"),
}
REFERENCE = {
    "oracle": ("#52514e", (0, (5, 2)), "Oracle"),
    "truncate": ("#8f8e8a", (0, (1, 2)), "Truncation"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd6"


def _style(ax, xs):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlim(xs[0] / 1.15, xs[-1] * 1.45)


def plot_metric(summary, metric, ylabel, title, out_path):
    lengths = sorted({r["context_length"] for r in summary})
    # Extra width per panel leaves room for the end-of-line direct labels.
    fig, axes = plt.subplots(
        1, len(lengths), figsize=(5.6 * len(lengths), 4.4), sharey=True
    )
    if len(lengths) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fcfcfb")

    for ax, ctx_len in zip(axes, lengths):
        rows = [r for r in summary if r["context_length"] == ctx_len]
        budgets = sorted({r["budget"] for r in rows if r["method"] != "naive_long_context"})

        ends = []
        for i, (method, (color, label)) in enumerate(SERIES.items()):
            pts = sorted(
                (r["budget"], r[metric]) for r in rows if r["method"] == method
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            # Series that coincide exactly would hide each other entirely; a
            # small dash offset keeps both visible without changing the values.
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label,
                    dashes=(1, 0) if i == 0 else (6, 1.5 * i), zorder=3 + i)
            ends.append([ys[-1], xs[-1], color, label])

        for method, (color, dash, label) in REFERENCE.items():
            pts = sorted((r["budget"], r[metric]) for r in rows if r["method"] == method)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=1.6, linestyle=dash, label=label, zorder=2)

        naive = [r for r in rows if r["method"] == "naive_long_context"]
        if naive:
            y = naive[0][metric]
            ax.axhline(y, color=INK, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
            # Sits just under its own rule with a surface-coloured backing: the
            # naive line often runs straight through the retriever curves.
            ax.annotate(
                f"naive long-context ({naive[0]['used_tokens']:,.0f} tok)",
                (budgets[0], y), color=INK, fontsize=8, va="top", ha="left",
                xytext=(0, -3), textcoords="offset points", zorder=5,
                bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.0),
            )

        # Push coincident end-labels apart so they stay readable when two
        # methods land on the same score.
        span = max(
            (r[metric] for r in rows), default=1.0
        ) - min((r[metric] for r in rows), default=0.0)
        gap = max(span, 0.1) * 0.06
        ends.sort()
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j - 1][0] < gap:
                ends[j][0] = ends[j - 1][0] + gap
        for y, x, color, label in ends:
            ax.annotate(f"  {label}", (x, y), color=color, fontsize=8,
                        va="center", ha="left", zorder=6,
                        annotation_clip=False)

        compression = {
            r["budget"]: r["compression"] for r in rows if r["method"] == "dense"
        }
        _style(ax, budgets)
        ax.set_xticklabels(
            [f"{b}\n{compression.get(b, 0):.0%}" for b in budgets], color=MUTED, fontsize=8
        )
        ax.set_title(f"context {ctx_len:,} tokens", color=INK, fontsize=11, pad=10)
        ax.set_xlabel("context budget (tokens) / compression", color=MUTED, fontsize=9)

    axes[0].set_ylabel(ylabel, color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=INK, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=160, facecolor="#fcfcfb", bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=None, help="output dir; defaults beside summary")
    args = ap.parse_args()

    blob = json.loads(Path(args.summary).read_text())
    summary = blob["summary"]
    out_dir = Path(args.out) if args.out else Path(args.summary).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(summary, "f1", "answer F1", "Answer F1 at matched context budgets",
                out_dir / "accuracy_vs_compression.png")
    plot_metric(summary, "full_support", "questions with all gold sentences",
                "Gold-evidence retention at matched context budgets",
                out_dir / "support_vs_compression.png")
    plot_metric(summary, "em", "answer EM", "Answer EM at matched context budgets",
                out_dir / "em_vs_compression.png")


if __name__ == "__main__":
    main()
