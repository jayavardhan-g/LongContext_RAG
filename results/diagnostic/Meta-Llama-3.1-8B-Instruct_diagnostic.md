# Head-signal diagnostic (scoring only, no generation)

300 questions, seed 0, heads from `results/head_id/Meta-Llama-3.1-8B-Instruct_retrieval_heads.json`.

`precision@N` is the share of selected tokens that are gold -- the quantity Milestone 2 found to matter more than recall.

## Signal correlation (mean Spearman over instances)

Distinguishes an orthogonal signal from one that is merely less sensitive to everything.

| pair | rho |
|---|---|
| dense~head_sum_k5 | +0.213 |
| reranker~head_sum_k5 | +0.354 |
| dense~reranker | +0.432 |
| dense~head_sum_k5(gold only) | +0.072 |

## Context length 32000

| method | R@1 | R@5 | R@10 | gold recall@256 | full support@256 | precision@256 |
|---|---|---|---|---|---|---|
| oracle | 0.433 | 0.999 | 1.000 | 0.999 | 0.997 | 0.397 |
| dense | 0.349 | 0.770 | 0.869 | 0.818 | 0.610 | 0.321 |
| reranker | 0.390 | 0.836 | 0.910 | 0.869 | 0.697 | 0.340 |
| bm25 | 0.306 | 0.616 | 0.743 | 0.661 | 0.373 | 0.255 |
| fuse_rrf_dense_head | 0.358 | 0.807 | 0.891 | 0.845 | 0.677 | 0.334 |
| fuse_z_dense_head | 0.331 | 0.822 | 0.929 | 0.868 | 0.733 | 0.342 |
| fuse_z_dense_head_25 | 0.346 | 0.844 | 0.935 | 0.882 | 0.747 | 0.348 |
| fuse_rrf_dense_reranker | 0.380 | 0.832 | 0.914 | 0.870 | 0.703 | 0.340 |
| head_sum_k1 | 0.044 | 0.128 | 0.206 | 0.136 | 0.013 | 0.062 |
| head_sum_k5 | 0.319 | 0.672 | 0.787 | 0.712 | 0.437 | 0.284 |
| head_sum_k10 | 0.329 | 0.673 | 0.768 | 0.703 | 0.417 | 0.284 |
| head_sum_k25 | 0.314 | 0.660 | 0.766 | 0.689 | 0.393 | 0.281 |
| head_mean_k1 | 0.026 | 0.110 | 0.182 | 0.151 | 0.030 | 0.059 |
| head_mean_k5 | 0.305 | 0.670 | 0.764 | 0.712 | 0.427 | 0.288 |
| head_mean_k10 | 0.298 | 0.641 | 0.749 | 0.698 | 0.413 | 0.289 |
| head_mean_k25 | 0.293 | 0.631 | 0.731 | 0.693 | 0.407 | 0.287 |
| head_max_k1 | 0.021 | 0.095 | 0.158 | 0.109 | 0.013 | 0.046 |
| head_max_k5 | 0.306 | 0.635 | 0.740 | 0.663 | 0.363 | 0.264 |
| head_max_k10 | 0.310 | 0.638 | 0.749 | 0.667 | 0.373 | 0.269 |
| head_max_k25 | 0.297 | 0.638 | 0.743 | 0.669 | 0.370 | 0.270 |

### Split by the naive long-context outcome

If a method's ranking collapses in the `naive_bad` column it is circular: it finds the evidence only when the model already used it.

| method | R@5 (naive ok) | R@5 (naive bad) | full support@256 (ok) | full support@256 (bad) |
|---|---|---|---|---|
| oracle | 1.000 | 0.999 | 1.000 | 0.993 |
| dense | 0.842 | 0.697 | 0.733 | 0.487 |
| reranker | 0.902 | 0.771 | 0.813 | 0.580 |
| bm25 | 0.698 | 0.534 | 0.507 | 0.240 |
| fuse_rrf_dense_head | 0.833 | 0.781 | 0.720 | 0.633 |
| fuse_z_dense_head | 0.849 | 0.794 | 0.773 | 0.693 |
| fuse_z_dense_head_25 | 0.883 | 0.806 | 0.833 | 0.660 |
| fuse_rrf_dense_reranker | 0.899 | 0.765 | 0.840 | 0.567 |
| head_sum_k1 | 0.136 | 0.119 | 0.007 | 0.020 |
| head_sum_k5 | 0.658 | 0.685 | 0.400 | 0.473 |
| head_sum_k10 | 0.650 | 0.695 | 0.373 | 0.460 |
| head_sum_k25 | 0.639 | 0.681 | 0.367 | 0.420 |
| head_mean_k1 | 0.116 | 0.104 | 0.027 | 0.033 |
| head_mean_k5 | 0.666 | 0.674 | 0.413 | 0.440 |
| head_mean_k10 | 0.638 | 0.645 | 0.380 | 0.447 |
| head_mean_k25 | 0.614 | 0.648 | 0.380 | 0.433 |
| head_max_k1 | 0.105 | 0.085 | 0.007 | 0.020 |
| head_max_k5 | 0.632 | 0.638 | 0.340 | 0.387 |
| head_max_k10 | 0.630 | 0.647 | 0.333 | 0.413 |
| head_max_k25 | 0.614 | 0.661 | 0.307 | 0.433 |

n = 150 naive-correct, 150 naive-wrong.
