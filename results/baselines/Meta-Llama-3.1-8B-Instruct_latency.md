# Latency vs naive long-context

25 questions per context length, unbatched, greedy, 32 new tokens, 10 warm-up calls discarded.

| context | condition | prompt tok | latency ms (mean) | p50 | speedup |
|---|---|---|---|---|---|
| 4000 | dense@1024 | 1067 | 123 | 86 | 1.91x |
| 4000 | dense@2048 | 2051 | 183 | 148 | 1.28x |
| 4000 | dense@256 | 328 | 108 | 70 | 2.19x |
| 4000 | dense@512 | 575 | 120 | 92 | 1.96x |
| 4000 | naive_long_context | 3928 | 235 | 206 | 1.00x |
| 16000 | dense@1024 | 1067 | 132 | 100 | 4.89x |
| 16000 | dense@2048 | 2050 | 132 | 112 | 4.88x |
| 16000 | dense@256 | 329 | 110 | 82 | 5.84x |
| 16000 | dense@512 | 576 | 117 | 90 | 5.53x |
| 16000 | naive_long_context | 15462 | 644 | 620 | 1.00x |
| 32000 | dense@1024 | 1068 | 134 | 86 | 10.34x |
| 32000 | dense@2048 | 2049 | 155 | 122 | 8.91x |
| 32000 | dense@256 | 330 | 92 | 68 | 14.97x |
| 32000 | dense@512 | 575 | 93 | 74 | 14.92x |
| 32000 | naive_long_context | 30834 | 1381 | 1367 | 1.00x |