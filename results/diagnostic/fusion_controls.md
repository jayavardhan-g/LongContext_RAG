# Fusion controls

300 instances at 32000 tokens, second-signal weight 0.25.

Gold sentences are 1.18x the mean sentence length, so a length prior is not a priori useless -- which is why it is tested.

| ranking signal | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| dense (alone) | 0.349 | 0.770 | 0.869 | 0.937 |
| head_sum_k5 (alone) | 0.319 | 0.672 | 0.787 | 0.854 |
| dense + head (w=0.25) | 0.346 | 0.844 | 0.935 | 0.970 |
| dense + length (w=0.25) | 0.332 | 0.736 | 0.856 | 0.930 |
| dense + random (w=0.25) | 0.341 | 0.750 | 0.863 | 0.927 |
| dense + bm25 (w=0.25) | 0.357 | 0.761 | 0.857 | 0.925 |
