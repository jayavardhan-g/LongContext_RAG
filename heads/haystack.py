"""Haystack construction: long distractor context with a needle at a known depth.

Prompts are assembled from separately-tokenized segments rather than by
tokenizing one big string. That costs a little fidelity at segment boundaries
but buys an exact, unambiguous token span for the needle, which the retrieval
score depends on -- a span located by substring search after the fact would be
off by a token whenever BPE merges across the boundary.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .needles import Needle

DEFAULT_HAYSTACK_DIR = Path(
    "/mnt/warm_storage/jayavardhan/tools/lcrag_data/pg_essays"
)

SENTINEL = "<<<LCRAG_CTX>>>"

INSTRUCTION = (
    "You are given a long document. Read it carefully and answer the question "
    "at the end using only information stated in the document."
)


@lru_cache(maxsize=4)
def load_haystack_text(haystack_dir: str = str(DEFAULT_HAYSTACK_DIR)) -> str:
    files = sorted(Path(haystack_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt haystack files under {haystack_dir}")
    parts = [f.read_text(encoding="utf-8", errors="ignore").strip() for f in files]
    return "\n\n".join(parts)


@dataclass
class NIAHInstance:
    input_ids: list[int]
    needle_start: int
    needle_end: int
    needle: Needle
    context_tokens: int
    depth: float

    @property
    def needle_len(self) -> int:
        return self.needle_end - self.needle_start


class PromptBuilder:
    """Builds NIAH prompts with exact needle token spans for one tokenizer."""

    def __init__(self, tokenizer, haystack_dir: str = str(DEFAULT_HAYSTACK_DIR)):
        self.tok = tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if SENTINEL not in rendered:
            raise RuntimeError("chat template dropped the sentinel")
        self.chat_head, self.chat_tail = rendered.split(SENTINEL)

        # The chat template emits BOS as literal text, so every segment is
        # tokenized without special tokens to avoid a duplicate BOS.
        text = load_haystack_text(haystack_dir)
        self.haystack_ids = tokenizer(text, add_special_tokens=False).input_ids

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def build(self, needle: Needle, context_tokens: int, depth: float) -> NIAHInstance:
        """Insert `needle` at `depth` (0=start, 1=end) of a context_tokens haystack."""
        if not 0.0 <= depth <= 1.0:
            raise ValueError(f"depth must be in [0, 1], got {depth}")
        if context_tokens > len(self.haystack_ids):
            raise ValueError(
                f"requested {context_tokens} haystack tokens but corpus has "
                f"{len(self.haystack_ids)}"
            )

        body = self.haystack_ids[:context_tokens]
        split = int(round(depth * len(body)))
        prefix_text = self.tok.decode(body[:split])
        suffix_text = self.tok.decode(body[split:])

        question_block = (
            f"\n\nQuestion: {needle.question}\n"
            "Answer briefly, using the exact wording from the document."
        )

        head_ids = self._ids(f"{self.chat_head}{INSTRUCTION}\n\n{prefix_text}\n")
        needle_ids = self._ids(needle.text.strip())
        tail_ids = self._ids(f"\n{suffix_text}{question_block}{self.chat_tail}")

        return NIAHInstance(
            input_ids=head_ids + needle_ids + tail_ids,
            needle_start=len(head_ids),
            needle_end=len(head_ids) + len(needle_ids),
            needle=needle,
            context_tokens=context_tokens,
            depth=depth,
        )
