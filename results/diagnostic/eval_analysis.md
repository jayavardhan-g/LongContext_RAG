# Post-hoc analysis of the fused-retrieval evaluation

Budget 256, 300 questions, 10000 bootstrap resamples.

## Does the selected context contain the answer string?

Span answers only (yes/no excluded). This is the yardstick the head signal is actually optimising, as opposed to annotated supporting facts.

| method | answer present in context |
|---|---|
| dense | 0.796 |
| reranker | 0.842 |
| head_sum_k5 | 0.863 |
| fusion_w0.20 | 0.898 |

## Paired bootstrap on F1 (budget 256)

| comparison | mean F1 diff | 95% CI | p |
|---|---|---|---|
| fusion_w0.20 vs dense | +0.0632 ** | [+0.0296, +0.0990] | 0.0000 |
| head_sum_k5 vs dense | +0.0553 ** | [+0.0108, +0.1017] | 0.0158 |
| reranker vs dense | +0.0344 | [-0.0050, +0.0747] | 0.0906 |
| fusion_w0.20 vs reranker | +0.0289 | [-0.0094, +0.0685] | 0.1434 |
| fusion_w0.20 vs head_sum_k5 | +0.0079 | [-0.0285, +0.0463] | 0.6902 |
| head_sum_k5 vs reranker | +0.0209 | [-0.0231, +0.0654] | 0.3482 |

`**` marks a CI that excludes zero.
