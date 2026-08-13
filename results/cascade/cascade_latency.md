# Cost of the head signal (shuffled job order, warm-up discarded)

25 questions, unbatched, first 10 calls discarded.

| stage | mean ms | p50 ms | n |
|---|---|---|---|
| dense_encode | 396.7 | 347.9 | 40 |
| head_S1024 | 105.0 | 119.4 | 23 |
| head_S2048 | 126.3 | 104.3 | 22 |
| head_S4096 | 210.9 | 218.7 | 22 |
| head_fullctx | 1497.1 | 1500.5 | 23 |

## End-to-end cost per question

`per_query` charges the encoder every question, which is what this benchmark literally does -- each question carries its own haystack. `pre_indexed` amortizes it away, which is what a deployed system with a fixed corpus does. Generation is the Milestone 2 measurement (92 ms at a 256-token budget, 1381 ms naive).

| method | per-query ms | vs naive | pre-indexed ms | vs naive |
|---|---|---|---|---|
| naive_long_context | 1381 | 1.00x | 1381 | 1.00x |
| dense@256 | 489 | 2.83x | 92 | 15.01x |
| cascade@256 (head_S1024) | 594 | 2.33x | 197 | 7.01x |
| fusion_fullctx@256 | 1986 | 0.70x | 1589 | 0.87x |
