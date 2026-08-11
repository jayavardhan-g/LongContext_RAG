"""Chunk scorers. A retriever's whole job here is to score chunks for a question.

Exact search is used rather than a FAISS index: each instance has its own
haystack of a few hundred to a few thousand sentences, so a single matmul is
both exact and faster than building a per-instance ANN index. An index only
earns its cost against one large shared corpus (the fullwiki setting), which is
not the setting chosen here.
"""

import random
import re

import torch
from rank_bm25 import BM25Okapi

from data.hotpot import Chunk

from .encoders import DenseEncoder

_TOKEN_RE = re.compile(r"\w+")


class Retriever:
    name = "base"

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    """bge-m3 cosine similarity between the question and each title-prefixed sentence."""

    name = "dense"

    def __init__(self, encoder: DenseEncoder):
        self.enc = encoder

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        q = self.enc.encode([question])
        d = self.enc.encode([c.embed_text for c in chunks])
        return (d @ q[0]).tolist()


class BM25Retriever(Retriever):
    """Lexical baseline. Multi-hop questions are entity-heavy, which is exactly
    where BM25 stays competitive with a strong dense retriever."""

    name = "bm25"

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        corpus = [_TOKEN_RE.findall(c.embed_text.lower()) for c in chunks]
        bm25 = BM25Okapi(corpus)
        return bm25.get_scores(_TOKEN_RE.findall(question.lower())).tolist()


class DocumentOrderRetriever(Retriever):
    """Naive truncation: keep the front of the context until the budget runs out.

    This is what "just feed the model as much as fits" does, and it is the
    reference point that says how much of any gain is retrieval rather than
    simply having fewer tokens.
    """

    name = "truncate"

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        return [-i for i in range(len(chunks))]


class OracleRetriever(Retriever):
    """Ceiling: gold sentences first, everything else in document order.

    Not a baseline to beat -- it is the upper bound that says how much of the
    remaining error is retrieval failure versus the generator failing on a
    context that already contains the answer.
    """

    name = "oracle"

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        return [1e6 - i if c.is_gold else -i for i, c in enumerate(chunks)]


class RandomRetriever(Retriever):
    """Floor, seeded per question so it is reproducible."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        rng = random.Random(f"{self.seed}:{question}")
        return [rng.random() for _ in chunks]


def rrf(score_lists: list[list[float]], k: int = 60) -> list[float]:
    """Reciprocal-rank fusion.

    Rank fusion rather than a weighted sum of scores because the two signals
    live on incomparable scales (cosine in [-1,1] versus unbounded BM25), so a
    weighted sum would need a tuning constant per dataset to be fair to either
    side. RRF needs none.

    Takes precomputed score lists rather than retrievers so the expensive dense
    pass is paid once per instance, not once for `dense` and again for `hybrid`.
    """
    n = len(score_lists[0])
    out = [0.0] * n
    for s in score_lists:
        for rank, i in enumerate(sorted(range(n), key=lambda i: -s[i])):
            out[i] += 1.0 / (k + rank + 1)
    return out
