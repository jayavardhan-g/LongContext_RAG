"""Scoring sentences by what the model's own retrieval heads attend to.

The signal is read at a *decode* step, not from the prefill: the query length is
1 there, so the attention matrix is [heads, 1, T] instead of [heads, T, T]. The
step used is the one that produces the first answer token, which is the moment
the model is deciding what to say and therefore the moment its retrieval heads
are pointed at whatever it is about to copy.

Three choices in here are real and each is a measured variant rather than an
assertion, because none of them is obviously right a priori:

  * **What to normalize over.** Raw attention is dominated by the BOS sink and
    by the instruction/question tokens, none of which are selectable. Each
    head's attention is renormalized over the union of the sentence spans, so a
    score reads as "share of this head's context-directed attention".
  * **How to aggregate a span.** `sum` rewards long sentences simply for having
    more tokens to accumulate mass on; `mean` divides that out but then rewards
    a short sentence with one incidental spike; `max` takes the peak token.
  * **How many heads.** Averaged over per-head normalized scores so each head
    counts equally -- a diffuse head would otherwise contribute most of the mass
    under a plain sum. Session 2 showed L15H30 alone is causally dispensable, so
    k=1 is reported only as the control it deserves to be.
"""

import torch
from transformers import DynamicCache

from baselines.retrievers import Retriever
from data.hotpot import Chunk

from .spans import SpanPromptBuilder

AGGREGATIONS = ("sum", "mean", "max")


def load_head_list(path: str) -> list[tuple[int, int]]:
    """Read the ranked retrieval heads produced by Milestone 1."""
    import json

    with open(path) as fh:
        heads = json.load(fh)
    return [(h["layer"], h["head"]) for h in heads]


class HeadAttentionScorer:
    """Produces every (aggregation, k) variant from a single forward pass.

    One prefill is by far the dominant cost, and the variants differ only in how
    the captured attention is reduced, so computing them separately would pay
    that cost once per variant for no reason.
    """

    def __init__(
        self,
        model,
        tokenizer,
        capture,
        heads: list[tuple[int, int]],
        k_values: tuple[int, ...] = (1, 5, 10, 25),
    ):
        self.model = model
        self.tok = tokenizer
        self.capture = capture
        self.builder = SpanPromptBuilder(tokenizer)
        self.k_values = tuple(sorted(k_values))
        self.heads = heads[: max(self.k_values)]
        if len(self.heads) < max(self.k_values):
            raise ValueError(
                f"need {max(self.k_values)} heads but the list has {len(self.heads)}"
            )
        self.device = next(model.parameters()).device
        self._l_idx = torch.tensor([l for l, _ in self.heads], device=self.device)
        self._h_idx = torch.tensor([h for _, h in self.heads], device=self.device)

    def variant_names(self) -> list[str]:
        return [f"head_{agg}_k{k}" for agg in AGGREGATIONS for k in self.k_values]

    @torch.no_grad()
    def attention_for(self, question: str, chunks: list[Chunk]) -> tuple[torch.Tensor, list[tuple[int, int]], int]:
        """Attention over the prompt from the first-answer-token decode step."""
        sp = self.builder.build(question, chunks)
        ids = torch.tensor([sp.input_ids], device=self.device)

        cache = DynamicCache()
        cfg = self.model.config
        previous = cfg._attn_implementation

        # Prefill under SDPA; only the final token runs eager, where the captured
        # matrix is [heads, 1, T] rather than the [heads, T, T] an eager prefill
        # would allocate (~4 GB per layer at 32k).
        cfg._attn_implementation = "sdpa"
        self.model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)
        cfg._attn_implementation = previous

        with self.capture.recording():
            self.model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True)
        attn = self.capture.stack()  # [layers, heads, keys]
        return attn, sp.spans, sp.n_tokens

    @torch.no_grad()
    def score_all(self, question: str, chunks: list[Chunk]) -> dict[str, list[float]]:
        attn, spans, _ = self.attention_for(question, chunks)
        selected = attn[self._l_idx, self._h_idx].float()  # [K, T]
        del attn

        n_chunks = len(chunks)
        seg = torch.full((selected.shape[1],), -1, dtype=torch.long, device=self.device)
        lengths = torch.ones(n_chunks, device=self.device)
        for i, (s, e) in enumerate(spans):
            if e > s:
                seg[s:e] = i
                lengths[i] = e - s

        valid = seg >= 0
        seg_v = seg[valid]
        p = selected[:, valid]
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

        K = p.shape[0]
        sums = torch.zeros(K, n_chunks, device=self.device).index_add_(1, seg_v, p)
        maxes = torch.zeros(K, n_chunks, device=self.device).scatter_reduce(
            1, seg_v.unsqueeze(0).expand(K, -1), p, reduce="amax", include_self=True
        )
        per_head = {"sum": sums, "mean": sums / lengths, "max": maxes}

        out: dict[str, list[float]] = {}
        for agg, mat in per_head.items():
            for k in self.k_values:
                out[f"head_{agg}_k{k}"] = mat[:k].mean(dim=0).cpu().tolist()
        return out


class HeadRetriever(Retriever):
    """Adapter so a chosen head variant plugs into the Milestone 2 pipeline."""

    def __init__(self, scorer: HeadAttentionScorer, aggregation: str = "sum", k: int = 5):
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.scorer = scorer
        self.key = f"head_{aggregation}_k{k}"
        self.name = self.key

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        return self.scorer.score_all(question, chunks)[self.key]
