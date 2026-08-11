"""Selecting chunks under a token budget.

Every method in this project -- dense, BM25, naive truncation, and later the
retrieval-head-guided compressor -- differs only in the *scores* it assigns to
chunks. They all spend their budget through this one function, so a difference
in results is a difference in the retrieval signal and not in how greedily each
implementation happened to pack its context.

Budget is defined as tokens of retrieved context handed to the generator,
excluding the question and the prompt scaffolding.
"""

from .hotpot import Chunk


def para_overhead(title_tokens: int) -> int:
    """Cost of introducing a paragraph: its title header and surrounding newline."""
    return title_tokens + 2


def fill_budget(
    chunks: list[Chunk],
    scores: list[float],
    budget: int,
    title_tokens: dict[str, int],
) -> list[Chunk]:
    """Greedily take the highest-scoring chunks that still fit in `budget`.

    Skips over a chunk that does not fit and keeps going rather than stopping at
    the first overflow: sentence lengths range from 5 to ~286 tokens, so halting
    on the first oversized sentence would leave large budgets badly underspent
    and would penalise whichever method happened to rank a long sentence highly.
    """
    if budget <= 0:
        return []

    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    selected: list[Chunk] = []
    seen_paras: set[int] = set()
    used = 0

    for i in order:
        c = chunks[i]
        cost = c.n_tokens + 1
        if c.para_idx not in seen_paras:
            cost += para_overhead(title_tokens[c.title])
        if used + cost > budget:
            continue
        selected.append(c)
        seen_paras.add(c.para_idx)
        used += cost

    return selected
