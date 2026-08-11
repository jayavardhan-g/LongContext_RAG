"""Compare our head scores against Wu et al.'s published head_score/*.json.

Caveat that governs how to read the output: Wu et al. publish no head list for
Llama-3 or Llama-3.1. The closest Llama artifact is `llama-2-7b-80k`, which
shares the 32x32 layer/head grid but is a different checkpoint entirely --
different pretraining, different tokenizer, and MHA rather than GQA. Retrieval
heads are reported to be model-specific, so *low* positional overlap here is the
expected result and is not evidence that our implementation is wrong.

What this script can legitimately check is distributional: does our sweep
reproduce the *shape* the paper describes -- a sparse set of heads carrying the
copying, rather than the ability spread across the layer/head grid?

Functional verification of our own head list lives in `heads/run_ablation.py`.

Usage:
    python -m heads.compare_published --scores results/head_id/<tag>_head_scores.json
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

PUBLISHED_BASE = (
    "https://raw.githubusercontent.com/nightdessert/Retrieval_Head/main/head_score/"
)
AVAILABLE = [
    "llama-2-7b-80k",
    "llama-2-13b-64k",
    "Mistral-7B-v0.2-hf",
    "Mixtral-8x7B-v0.1",
    "Qwen1.5-14B",
    "Qwen1.5-14B-Chat",
    "Yi-6B-200K",
]


def load_published(name: str, cache_dir: Path) -> dict[tuple[int, int], float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.json"
    if not path.exists():
        urllib.request.urlretrieve(PUBLISHED_BASE + name + ".json", path)
    raw = json.loads(path.read_text())
    # each value is the per-instance score list; the head's score is its mean
    return {
        tuple(int(x) for x in k.split("-")): float(np.mean(v)) for k, v in raw.items()
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--published", default="llama-2-7b-80k", choices=AVAILABLE)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cache-dir", default="/mnt/warm_storage/jayavardhan/tools/lcrag_data/published_heads")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.scores).read_text())
    # Compare like with like: our faithful reimplementation of |g_h|/|k|, not the
    # copy-precision variant, or model and metric differences get confounded.
    ours = np.array(data.get("wu_mean_scores", data["mean_scores"]))
    pub_map = load_published(args.published, Path(args.cache_dir))
    pl = max(l for l, _ in pub_map) + 1
    ph = max(h for _, h in pub_map) + 1

    lines = [
        f"# Comparison against published head scores — {args.published}",
        "",
        f"- ours: `{data['model']}`, {ours.shape[0]}x{ours.shape[1]} grid, "
        f"metric `|g_h|/|k|` (Wu et al. formulation)",
        f"- published: `{args.published}`, {pl}x{ph} grid",
        "",
        "**Wu et al. publish no head list for Llama-3 or Llama-3.1.** This is a "
        "cross-model comparison against a different checkpoint, not a "
        "reproduction check. Retrieval heads are reported to be model-specific, "
        "so low positional overlap is expected and is not evidence of a bug.",
        "",
    ]

    ours_n = int((ours >= 0.1).sum())
    pub_n = sum(1 for s in pub_map.values() if s >= 0.1)
    lines += [
        "## Distributional shape (the part that is comparable)",
        "",
        "| | heads >= 0.1 | of | share | max score |",
        "|---|---|---|---|---|",
        f"| ours | {ours_n} | {ours.size} | {ours_n / ours.size:.1%} | {ours.max():.3f} |",
        f"| {args.published} | {pub_n} | {len(pub_map)} | {pub_n / len(pub_map):.1%} | "
        f"{max(pub_map.values()):.3f} |",
        "",
    ]

    if (pl, ph) == ours.shape:
        pub = np.zeros_like(ours)
        for (l, h), s in pub_map.items():
            pub[l, h] = s
        ours_top = {
            (int(i // ours.shape[1]), int(i % ours.shape[1]))
            for i in np.argsort(ours, axis=None)[::-1][: args.top_k]
        }
        pub_top = {
            (int(i // pub.shape[1]), int(i % pub.shape[1]))
            for i in np.argsort(pub, axis=None)[::-1][: args.top_k]
        }
        rho = spearman(ours.ravel(), pub.ravel())
        shared = sorted(ours_top & pub_top)
        lines += [
            "## Positional overlap (expected to be low — different model)",
            "",
            f"- top-{args.top_k} overlap: **{len(shared)}/{args.top_k}**"
            + (f" — {', '.join(f'L{l}H{h}' for l, h in shared)}" if shared else ""),
            f"- Spearman rho over all {ours.size} heads: **{rho:+.3f}**",
            "",
        ]

    lines += [
        "## Reading",
        "",
        "Agreement on *shape* (a low-single-digit percentage of heads carrying the "
        "copying) supports the implementation. Disagreement on *which* heads is "
        "uninformative here — it is what the published work predicts across "
        "different checkpoints. The head list for this model is verified "
        "functionally by ablation instead; see `results/ablation/`.",
    ]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
