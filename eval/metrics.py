"""Answer and retrieval metrics.

`normalize_answer`, `exact_match` and `f1` reproduce the official HotpotQA
evaluation script so numbers here are comparable to published ones.
"""

import re
import string
from collections import Counter

from data.hotpot import Chunk


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace (HotpotQA official)."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p_norm, g_norm = normalize_answer(pred), normalize_answer(gold)

    # Official script special-case: yes/no/noanswer only ever match exactly, so a
    # predicted "yes" cannot earn partial credit against a span answer.
    if p_norm in ("yes", "no", "noanswer") or g_norm in ("yes", "no", "noanswer"):
        return float(p_norm == g_norm)

    p_toks, g_toks = p_norm.split(), g_norm.split()
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def gold_recall(selected: list[Chunk], all_chunks: list[Chunk]) -> float:
    """Fraction of gold sentences present in the selected context."""
    n_gold = sum(c.is_gold for c in all_chunks)
    if n_gold == 0:
        return 0.0
    return sum(c.is_gold for c in selected) / n_gold


def full_support(selected: list[Chunk], all_chunks: list[Chunk]) -> float:
    """1.0 only if *every* gold sentence made it in.

    The stricter companion to gold_recall, and the one that matters for
    multi-hop: a question needing two facts is not answerable from one of them,
    so a mean recall of 0.5 can mean "half the answer for everyone" rather than
    "fully answerable for half".
    """
    n_gold = sum(c.is_gold for c in all_chunks)
    if n_gold == 0:
        return 0.0
    return float(sum(c.is_gold for c in selected) == n_gold)


def recall_at_k(scores: list[float], chunks: list[Chunk], k: int) -> float:
    """Classic Recall@k over chunks, independent of the token budget."""
    n_gold = sum(c.is_gold for c in chunks)
    if n_gold == 0:
        return 0.0
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return sum(chunks[i].is_gold for i in top) / n_gold
