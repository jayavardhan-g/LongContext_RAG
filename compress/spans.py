"""Full-context prompts with an exact token span per sentence.

The head signal is attention mass landing on a sentence, so every sentence needs
an unambiguous token range inside the prompt. Segments are tokenized separately
and concatenated rather than tokenizing one big string -- the same trade the
NIAH builder makes in `heads/haystack.py`. Locating spans by substring search
after the fact would be off by a token wherever BPE merges across a sentence
boundary, and those boundaries are exactly what is being measured.

The small tokenization drift this introduces never reaches a reported number:
this prompt is only ever used to *score* chunks. Generation always goes through
`baselines.generate.build_prompt` on `render_context`, so the matched-budget
comparison against the baselines is unaffected.
"""

from dataclasses import dataclass

from baselines.generate import INSTRUCTION
from data.hotpot import Chunk

SENTINEL = "<<<LCRAG_CTX>>>"


@dataclass
class SpannedPrompt:
    input_ids: list[int]
    spans: list[tuple[int, int]]  # one (start, end) per chunk, in `chunks` order

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


class SpanPromptBuilder:
    """Builds the full-context prompt and reports where each sentence landed."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if SENTINEL not in rendered:
            raise RuntimeError("chat template dropped the sentinel")
        self.chat_head, self.chat_tail = rendered.split(SENTINEL)

    def _ids(self, text: str) -> list[int]:
        # The chat template emits BOS as literal text, so no segment may add one.
        return self.tok(text, add_special_tokens=False).input_ids

    def build(self, question: str, chunks: list[Chunk]) -> SpannedPrompt:
        order = sorted(range(len(chunks)), key=lambda i: (chunks[i].para_idx, chunks[i].sent_idx))

        # Mirrors data.hotpot.render_context exactly, but keeps the piece
        # boundaries so each sentence keeps its own segment.
        pieces: list[str] = []
        owner: list[int | None] = []
        current: int | None = None
        for i in order:
            c = chunks[i]
            if c.para_idx != current:
                pieces.append(f"\n{c.title}\n" if pieces else f"{c.title}\n")
                owner.append(None)
                current = c.para_idx
            pieces.append(f"{c.text}\n")
            owner.append(i)
        if pieces:
            pieces[-1] = pieces[-1].rstrip("\n")  # render_context strips the tail

        head = f"{self.chat_head}Context:\n"
        tail = f"\n\nQuestion: {question}\n\n{INSTRUCTION}{self.chat_tail}"

        input_ids = self._ids(head)
        spans: list[tuple[int, int]] = [(0, 0)] * len(chunks)
        for piece, who in zip(pieces, owner):
            ids = self._ids(piece)
            if who is not None:
                spans[who] = (len(input_ids), len(input_ids) + len(ids))
            input_ids += ids
        input_ids += self._ids(tail)

        return SpannedPrompt(input_ids=input_ids, spans=spans)
