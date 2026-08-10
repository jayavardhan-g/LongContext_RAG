"""Needle definitions for the needle-in-a-haystack probes used to score heads.

The retrieval score only registers when the model *copies* tokens out of the
needle, so every needle here carries a distinctive span that cannot be answered
from parametric knowledge. The set deliberately mixes four kinds of needle
(natural fact, random digits, rare proper nouns, invented terms) so that the
resulting head list is not an artifact of one surface form.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Needle:
    id: str
    kind: str
    text: str
    question: str
    answer: str


NEEDLES: list[Needle] = [
    Needle(
        id="sf_sandwich",
        kind="natural_fact",
        text=(
            "The best thing to do in San Francisco is eat a sandwich and sit in "
            "Dolores Park on a sunny day."
        ),
        question="What is the best thing to do in San Francisco?",
        answer="eat a sandwich and sit in Dolores Park on a sunny day",
    ),
    Needle(
        id="magic_osaka",
        kind="random_digits",
        text="The special magic Osaka number is 7492013.",
        question="What is the special magic Osaka number?",
        answer="7492013",
    ),
    Needle(
        id="magic_helsinki",
        kind="random_digits",
        text="The special magic Helsinki number is 3856147.",
        question="What is the special magic Helsinki number?",
        answer="3856147",
    ),
    Needle(
        id="glass_bridge",
        kind="natural_fact",
        text=(
            "The Kestrel Glass Bridge was completed in 1987 and spans the "
            "Marrowfen Gorge in northern Aldberry."
        ),
        question="When was the Kestrel Glass Bridge completed and what does it span?",
        answer="completed in 1987 and spans the Marrowfen Gorge",
    ),
    Needle(
        id="rare_name",
        kind="rare_proper_noun",
        text=(
            "The chief archivist of the Vantroven Institute is Dr. Ilsabet "
            "Quorrigan, who joined in 2011."
        ),
        question="Who is the chief archivist of the Vantroven Institute?",
        answer="Dr. Ilsabet Quorrigan",
    ),
    Needle(
        id="invented_term",
        kind="invented_term",
        text=(
            "A pelluvian cascade is the term used for a chain of three or more "
            "overlapping tidal reversals."
        ),
        question="What is a pelluvian cascade?",
        answer="a chain of three or more overlapping tidal reversals",
    ),
    Needle(
        id="recipe_code",
        kind="random_digits",
        text="The access code for the Thornbury vault is QX-4471-ZM.",
        question="What is the access code for the Thornbury vault?",
        answer="QX-4471-ZM",
    ),
    Needle(
        id="animal_fact",
        kind="natural_fact",
        text=(
            "The saltmarsh nightjar migrates each autumn from the Brindle Flats "
            "to the Cape of Ashen Sands."
        ),
        question="Where does the saltmarsh nightjar migrate to each autumn?",
        answer="the Cape of Ashen Sands",
    ),
]

NEEDLES_BY_ID = {n.id: n for n in NEEDLES}
