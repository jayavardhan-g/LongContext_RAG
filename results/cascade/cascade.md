# Cascade: dense shortlist -> head rescore

300 questions at 32000 tokens, final budget 256, fusion weight 0.2.

| method | used tok | gold recall | full support | EM | F1 |
|---|---|---|---|---|---|
| fusion_fullctx | 241 | 0.886 | 0.757 | 0.527 | 0.670 |
| cascade_S1024 | 239 | 0.864 | 0.700 | 0.517 | 0.661 |
| cascade_S4096 | 240 | 0.879 | 0.730 | 0.517 | 0.661 |
| cascade_S2048 | 239 | 0.869 | 0.710 | 0.510 | 0.652 |
| cascade_S1024_denseonly | 239 | 0.818 | 0.610 | 0.470 | 0.608 |
| dense | 241 | 0.818 | 0.610 | 0.470 | 0.607 |
| cascade_S4096_denseonly | 240 | 0.818 | 0.610 | 0.470 | 0.606 |
| cascade_S2048_denseonly | 239 | 0.818 | 0.610 | 0.467 | 0.605 |

## Cost of producing the signal (unbatched, per question)

| stage | ms |
|---|---|
| dense_encode_ms | 594.9 |
| head_pass_fullctx_ms | 1453.4 |
| head_pass_S1024_ms | 154.3 |
| head_pass_S2048_ms | 87.8 |
| head_pass_S4096_ms | 153.8 |
