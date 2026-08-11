"""Pooled-distractor HotpotQA: a long-context benchmark with exact gold labels.

HotpotQA's distractor setting gives 10 paragraphs, ~1.3k Llama tokens. That is
far too short to study long-context compression -- Llama-3.1 has a 128k window,
so nothing is lost in the middle at 1.3k and every method ties. This module
keeps each question's 2 gold paragraphs and its 8 hard TF-IDF distractors, then
pads with distractor paragraphs pooled from *other* questions until the context
reaches a target token length.

That buys three things the native setting cannot give:
  - a context-length knob (4k / 8k / 16k / 32k) with everything else held fixed;
  - exact sentence-level gold labels, so Recall is measured, not estimated;
  - controlled gold *depth*, which is the lost-in-the-middle probe.

The cost is that the haystack is semi-synthetic. Pooled paragraphs are excluded
by title whenever the title appears anywhere in the source question's own 10
paragraphs, which prevents re-inserting gold content at an unlabelled position.
A pooled paragraph under a different title could still happen to mention the
answer; that residual risk is inherent to any distractor pool (fullwiki
included) and is not corrected for here.
"""

import random
from dataclasses import dataclass, field
from functools import lru_cache

from datasets import load_dataset

HOTPOT_ID = "hotpotqa/hotpot_qa"


@dataclass
class Chunk:
    """One sentence -- the unit every retriever selects over.

    Gold supporting facts in HotpotQA are annotated per sentence, so making the
    sentence the retrieval unit lets Recall be scored directly against the
    annotation with no span mapping in between.
    """

    text: str
    title: str
    para_idx: int
    sent_idx: int
    is_gold: bool
    n_tokens: int = 0

    @property
    def embed_text(self) -> str:
        """Title-prefixed form used for scoring.

        A bare HotpotQA sentence often drops its subject ("He was born in..."),
        which is unrecoverable for a retriever. The title restores it.
        """
        return f"{self.title}: {self.text}"


@dataclass
class LongQAInstance:
    qid: str
    question: str
    answer: str
    qtype: str
    level: str
    chunks: list[Chunk]
    target_tokens: int
    gold_depth: float | None
    n_native_paras: int
    n_pooled_paras: int
    meta: dict = field(default_factory=dict)

    @property
    def gold_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks if c.is_gold]

    @property
    def n_gold(self) -> int:
        return sum(c.is_gold for c in self.chunks)

    @property
    def context_tokens(self) -> int:
        return sum(c.n_tokens for c in self.chunks)


def _clean(s: str) -> str:
    return s.strip()


def _paragraphs(example) -> list[tuple[str, list[str]]]:
    ctx = example["context"]
    return [
        (t, [_clean(s) for s in sents if _clean(s)])
        for t, sents in zip(ctx["title"], ctx["sentences"])
    ]


def _gold_keys(example) -> set[tuple[str, int]]:
    sf = example["supporting_facts"]
    return set(zip(sf["title"], sf["sent_id"]))


@lru_cache(maxsize=2)
def load_hotpot(split: str = "validation"):
    return load_dataset(HOTPOT_ID, "distractor")[split]


class DistractorPool:
    """Deduplicated pool of paragraphs used to pad contexts to length.

    Built from the train split so that padding never pulls another *evaluation*
    question's gold paragraph into the haystack.
    """

    def __init__(self, size: int = 20000, split: str = "train", seed: int = 0):
        ds = load_hotpot(split)
        n = min(size, len(ds))
        by_title: dict[str, list[str]] = {}
        for i in range(n):
            for title, sents in _paragraphs(ds[i]):
                if title not in by_title and sents:
                    by_title[title] = sents
        self.titles = sorted(by_title)
        self.by_title = by_title
        self._seed = seed

    def __len__(self) -> int:
        return len(self.titles)

    def sample_order(self, seed: int) -> list[str]:
        order = list(self.titles)
        random.Random(self._seed + seed).shuffle(order)
        return order


class InstanceBuilder:
    """Builds long-context instances at a target token length."""

    def __init__(self, tokenizer, pool: DistractorPool):
        self.tok = tokenizer
        self.pool = pool
        self._tok_cache: dict[str, int] = {}

    def _count(self, texts: list[str]) -> list[int]:
        """Token counts, memoised -- the pool is reused across every instance."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._tok_cache]
        if missing:
            enc = self.tok(missing, add_special_tokens=False).input_ids
            for t, ids in zip(missing, enc):
                self._tok_cache[t] = len(ids)
        return [self._tok_cache[t] for t in texts]

    def title_tokens(self, inst: "LongQAInstance") -> dict[str, int]:
        """Token count per title, for the budget accounting in data.budget."""
        titles = list({c.title for c in inst.chunks})
        return dict(zip(titles, self._count(titles)))

    def _para_tokens(self, title: str, sents: list[str]) -> int:
        # Header plus a newline per sentence, matching render_context below.
        return sum(self._count(sents)) + self._count([title])[0] + len(sents) + 2

    def build(
        self,
        example,
        target_tokens: int,
        gold_depth: float | None = None,
        seed: int = 0,
    ) -> LongQAInstance:
        native = _paragraphs(example)
        gold_keys = _gold_keys(example)
        gold_titles = {t for t, _ in gold_keys}
        native_titles = {t for t, _ in native}

        gold_paras = [(t, s) for t, s in native if t in gold_titles]
        other_paras = [(t, s) for t, s in native if t not in gold_titles]

        used = sum(self._para_tokens(t, s) for t, s in native)
        pooled: list[tuple[str, list[str]]] = []
        for title in self.pool.sample_order(seed):
            if used >= target_tokens:
                break
            if title in native_titles:
                continue
            sents = self.pool.by_title[title]
            cost = self._para_tokens(title, sents)
            if used + cost > target_tokens and pooled:
                continue  # try a shorter paragraph rather than overshooting
            pooled.append((title, sents))
            used += cost

        # Distractors are shuffled together so pooled and native padding are
        # indistinguishable by position; gold is then placed deliberately.
        rng = random.Random(seed)
        distractors = other_paras + pooled
        rng.shuffle(distractors)

        if gold_depth is None:
            slots = sorted(rng.sample(range(len(distractors) + 1), len(gold_paras)))
        else:
            start = int(round(gold_depth * len(distractors)))
            slots = [min(start + i, len(distractors)) for i in range(len(gold_paras))]

        ordered: list[tuple[str, list[str]]] = []
        gi = 0
        for pos in range(len(distractors) + 1):
            while gi < len(slots) and slots[gi] == pos:
                ordered.append(gold_paras[gi])
                gi += 1
            if pos < len(distractors):
                ordered.append(distractors[pos])

        chunks: list[Chunk] = []
        for p_idx, (title, sents) in enumerate(ordered):
            counts = self._count(sents)
            for s_idx, (sent, n_tok) in enumerate(zip(sents, counts)):
                chunks.append(
                    Chunk(
                        text=sent,
                        title=title,
                        para_idx=p_idx,
                        sent_idx=s_idx,
                        is_gold=(title, s_idx) in gold_keys,
                        n_tokens=n_tok,
                    )
                )

        inst = LongQAInstance(
            qid=example["id"],
            question=example["question"],
            answer=example["answer"],
            qtype=example.get("type", ""),
            level=example.get("level", ""),
            chunks=chunks,
            target_tokens=target_tokens,
            gold_depth=gold_depth,
            n_native_paras=len(native),
            n_pooled_paras=len(pooled),
        )
        if inst.n_gold != len(gold_keys):
            raise RuntimeError(
                f"{inst.qid}: {inst.n_gold} gold chunks marked but annotation has "
                f"{len(gold_keys)} -- title/sent_id mismatch"
            )
        return inst


def render_context(chunks: list[Chunk]) -> str:
    """Render selected chunks back into titled paragraphs, in document order.

    Selection is per sentence but the model reads paragraphs, so contiguous
    sentences from one paragraph are regrouped under a single title header.
    Every method renders through this same function, so none of them gains or
    loses tokens to formatting differences.
    """
    parts: list[str] = []
    current: int | None = None
    for c in sorted(chunks, key=lambda c: (c.para_idx, c.sent_idx)):
        if c.para_idx != current:
            parts.append(f"\n{c.title}\n" if parts else f"{c.title}\n")
            current = c.para_idx
        parts.append(f"{c.text}\n")
    return "".join(parts).strip()
