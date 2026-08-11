# Comparison against published head scores — llama-2-7b-80k

- ours: `NousResearch/Meta-Llama-3.1-8B-Instruct`, 32x32 grid, metric `|g_h|/|k|` (Wu et al. formulation)
- published: `llama-2-7b-80k`, 32x32 grid

**Wu et al. publish no head list for Llama-3 or Llama-3.1.** This is a cross-model comparison against a different checkpoint, not a reproduction check. Retrieval heads are reported to be model-specific, so low positional overlap is expected and is not evidence of a bug.

## Distributional shape (the part that is comparable)

| | heads >= 0.1 | of | share | max score |
|---|---|---|---|---|
| ours | 21 | 1024 | 2.1% | 0.725 |
| llama-2-7b-80k | 33 | 1024 | 3.2% | 0.943 |

## Positional overlap (expected to be low — different model)

- top-20 overlap: **1/20** — L16H1
- Spearman rho over all 1024 heads: **+0.015**

## Reading

Agreement on *shape* (a low-single-digit percentage of heads carrying the copying) supports the implementation. Disagreement on *which* heads is uninformative here — it is what the published work predicts across different checkpoints. The head list for this model is verified functionally by ablation instead; see `results/ablation/`.
