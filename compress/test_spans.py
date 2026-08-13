"""Checks that each sentence's reported token span really is that sentence.

The head signal is attention mass summed over a span, so a span that is off by a
token silently mixes a neighbouring sentence into the score. That failure is
invisible downstream -- the numbers still look like numbers -- which is why it
gets its own test.

    python -m compress.test_spans
"""

import sys

from transformers import AutoTokenizer

from baselines.generate import build_prompt
from data.hotpot import Chunk, render_context
from paths import resolve_model

from .spans import SpanPromptBuilder

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

PASS, FAIL = "[PASS]", "[FAIL]"
_failures = 0


def check(cond, msg):
    global _failures
    print(f"{PASS if cond else FAIL} {msg}", flush=True)
    if not cond:
        _failures += 1


def make_chunks() -> list[Chunk]:
    """Two paragraphs, deliberately including the shapes that break naive spans."""
    specs = [
        ("Alan Turing", 0, [
            "Alan Turing was a British mathematician.",
            "He worked at Bletchley Park during the war.",
        ]),
        ("Bletchley Park", 1, [
            "Bletchley Park is an estate in Milton Keynes.",
            "It housed the Government Code and Cypher School.",
            "Turing, Welchman and others worked there.",
        ]),
    ]
    chunks = []
    for title, p_idx, sents in specs:
        for s_idx, text in enumerate(sents):
            chunks.append(
                Chunk(text=text, title=title, para_idx=p_idx, sent_idx=s_idx,
                      is_gold=(p_idx == 0 and s_idx == 1))
            )
    return chunks


def main():
    tok = AutoTokenizer.from_pretrained(resolve_model(MODEL))
    builder = SpanPromptBuilder(tok)
    chunks = make_chunks()
    question = "Where did Alan Turing work during the war?"

    sp = builder.build(question, chunks)

    check(len(sp.spans) == len(chunks), f"one span per chunk ({len(sp.spans)})")

    # Every span decodes back to its own sentence. Leading whitespace is allowed
    # because the sentence segment carries the newline that separates it.
    exact = 0
    for c, (s, e) in zip(chunks, sp.spans):
        decoded = tok.decode(sp.input_ids[s:e]).strip()
        if decoded == c.text:
            exact += 1
        else:
            print(f"       span mismatch: {decoded!r} != {c.text!r}")
    check(exact == len(chunks), f"every span decodes to its sentence ({exact}/{len(chunks)})")

    starts = [s for s, _ in sp.spans]
    check(starts == sorted(starts), "spans are in document order")
    check(
        all(sp.spans[i][1] <= sp.spans[i + 1][0] for i in range(len(sp.spans) - 1)),
        "spans do not overlap",
    )
    check(all(e > s for s, e in sp.spans), "no empty spans")

    # The assembled prompt must be the same text the generator would see, or the
    # signal is being read off a different context than the one being scored.
    reference = build_prompt(tok, render_context(chunks), question)
    rebuilt = tok.decode(sp.input_ids)
    check(rebuilt == reference, "assembled prompt matches build_prompt(render_context(...))")
    if rebuilt != reference:
        print(f"       rebuilt : {rebuilt[-200:]!r}")
        print(f"       expected: {reference[-200:]!r}")

    # A gold sentence must be locatable by span alone -- this is what the scorer
    # relies on when it sums attention over is_gold chunks.
    gold = [i for i, c in enumerate(chunks) if c.is_gold]
    s, e = sp.spans[gold[0]]
    check(
        "Bletchley" in tok.decode(sp.input_ids[s:e]),
        "gold span contains the gold content",
    )

    # Spans must sit inside the context region, never in the question or the
    # chat scaffolding, or renormalizing over "the context" would be wrong.
    covered = tok.decode(sp.input_ids[: max(e for _, e in sp.spans)])
    check(question not in covered, "spans stop before the question block")
    check("Context:" not in tok.decode(sp.input_ids[sp.spans[0][0] :][: sp.spans[0][1] - sp.spans[0][0]]),
          "first span excludes the 'Context:' header")

    print()
    if _failures:
        print(f"{_failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
