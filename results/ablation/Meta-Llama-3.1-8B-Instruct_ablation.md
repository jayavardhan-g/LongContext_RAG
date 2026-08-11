# Head ablation — causal check on the identified retrieval heads

- 80 needle-in-a-haystack instances per condition (2 context lengths x 5 depths x 8 needles)
- metric: needle recall (did the answer appear in the generation)
- heads are zeroed at the input to `o_proj`, removing only that head's contribution to the residual stream

| condition | heads masked | recall | drop vs baseline |
|---|---|---|---|
| baseline | 0 | 0.975 | — |
| top-1 retrieval heads | 1 | 0.975 | +0.000 |
| top-5 retrieval heads | 5 | 0.688 | +0.287 |
| top-10 retrieval heads | 10 | 0.625 | +0.350 |
| random-1 heads (seed 0) | 1 | 0.975 | +0.000 |
| random-1 heads (seed 1) | 1 | 0.975 | +0.000 |
| random-1 heads (seed 2) | 1 | 0.975 | +0.000 |
| random-5 heads (seed 0) | 5 | 0.975 | +0.000 |
| random-5 heads (seed 1) | 5 | 0.975 | +0.000 |
| random-5 heads (seed 2) | 5 | 0.975 | +0.000 |
| random-10 heads (seed 0) | 10 | 0.950 | +0.025 |
| random-10 heads (seed 1) | 10 | 0.975 | +0.000 |
| random-10 heads (seed 2) | 10 | 0.963 | +0.012 |

## Identified vs random, at matched mask size

| heads masked | identified | random (mean) | gap |
|---|---|---|---|
| 1 | 0.975 | 0.975 | **+0.000** |
| 5 | 0.688 | 0.975 | **+0.287** |
| 10 | 0.625 | 0.962 | **+0.337** |

A large gap means the identified heads are doing work that randomly chosen heads are not — which is what earns them the name. A small gap would mean the head list is not causally special, and the compression premise of this project would need rethinking.
