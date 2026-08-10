# Retrieval-Head-Guided Efficient Long-Context RAG

Standard RAG decides what to put in the context window with an *external* signal:
embedding similarity between the query and candidate chunks. This project uses an
*internal* one — the attention of the specific heads that empirically do the
copy-from-context work inside the generator itself — to decide what to keep and
how far to compress a long context.

The headline question: **at a matched context budget, does selecting context with
the model's own retrieval heads beat selecting it with a dense retriever?**

## Method

Three systems are compared at matched context budgets on multi-hop QA:

| System | Context selection signal |
|---|---|
| Naive long-context | none — everything in the window |
| Dense-retrieval RAG | external embedding similarity |
| Retrieval-head-guided (ours) | internal attention from identified retrieval heads |

## Stage 1 — Retrieval-head identification

Reimplements the retrieval score from Wu et al., *Retrieval Head Mechanistically
Explains Long-Context Factuality*.

A needle sentence is planted at a known depth in a long distractor context (Paul
Graham essays) and the model is asked a question only the needle answers. At each
decode step, a head scores a hit if its attention **argmax** lands on a needle
token that is exactly the token being generated — the head caught copying.

The headline metric is **copy-precision**: of the decode steps where the emitted
token was in the needle, the fraction where the head's argmax pointed at it.
Wu et al.'s `|g_h| / |k|` is reported alongside it.

### Why not the published normalizer as headline

`|g_h| / |k|` divides by the *whole needle*, but the model only copies the
*answer span*. On a 26-token needle whose answer is ~10 tokens, a head copying
perfectly cannot score above 0.371. In the first sweep this made retrieval heads
look like they collapsed on rare proper nouns (`L15H30`: 0.87 on natural facts
vs 0.28 on a rare name) — a textbook "heads don't generalize" result that was
purely an artifact. That needle's recall was 0.97: the model retrieved it
perfectly.

Dividing each needle's observed score by its achievable ceiling gave a ratio of
0.76–0.89 across all eight needles — uniform once the ceiling was accounted for.
Under copy-precision, which is invariant to answer-span length, `L15H30` scores
0.800–0.918 on every needle.

Scores are broken out per individual needle as well as per needle kind: a kind
holding a single needle is confounded with that needle's identity, which is how
the artifact disguised itself.

### Result

**25 of 1024 heads score above 0.1**, and the signal is sparse and concentrated:

| rank | head | copy-precision | Wu et al. score |
|---|---|---|---|
| 1 | L15H30 | 0.865 | 0.725 |
| 2 | L8H1 | 0.430 | 0.362 |
| 3 | L16H1 | 0.376 | 0.319 |
| 4 | L24H27 | 0.372 | 0.308 |
| 5 | L16H20 | 0.338 | 0.282 |

240 instances (3 context lengths × 10 depths × 8 needles), needle recall 0.988,
154 s on one H200. Recomputing over only the 237 answered instances leaves the
top-20 unchanged (overlap 20/20), so the list is not an artifact of failures.

### Why the score is computed at decode time only

`output_attentions` was removed in transformers 5.x, and attention weights are
only produced by the eager path — SDPA and flash never materialize the matrix.
Running the model eager throughout is not viable: an eager prefill allocates a
`[heads, T, T]` matrix per layer, roughly 4 GB per layer at T=8k, growing
quadratically.

The score only ever reads attention at *decode* steps, where the query length is
1 and the matrix is `[heads, 1, T]` — about 1 MB per layer. Since
`LlamaAttention.forward` re-reads `config._attn_implementation` on every call,
the pipeline prefills under SDPA and flips to eager only for the single-token
steps, capturing weights with a forward hook. The final prompt token is fed
inside the decode loop rather than the prefill, so even the attention producing
the *first* generated token is captured at `q_len=1`.

## Layout

```
heads/       retrieval-head identification (needles, haystack, capture, scoring)
baselines/   dense-retrieval RAG, naive long-context
compress/    retrieval-head-guided context compression
eval/        Recall@k, QA accuracy/F1
results/
```

## Setup

```bash
conda create -p /mnt/warm_storage/jayavardhan/tools/conda-envs/lcrag python=3.11
/mnt/warm_storage/jayavardhan/tools/conda-envs/lcrag/bin/pip install \
    torch transformers accelerate datasets sentencepiece protobuf \
    numpy pandas matplotlib tqdm huggingface_hub hf_transfer
```

Run head identification:

```bash
python -m heads.run_head_id --context-lengths 2000 4000 8000 --n-depths 10
```

Model: `NousResearch/Meta-Llama-3.1-8B-Instruct` (an ungated mirror of
`meta-llama/Llama-3.1-8B-Instruct`; identical weights, no access request needed).
