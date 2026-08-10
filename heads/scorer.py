"""Retrieval score (Wu et al., "Retrieval Head Mechanistically Explains
Long-Context Factuality").

A head scores a hit at a decode step when its attention argmax lands on a needle
token that is exactly the token being generated -- i.e. the head is caught in the
act of copying that token out of the context. The head's retrieval score on an
instance is the fraction of needle tokens it copied this way.

Two counts are kept because the paper's |g_h| / |k| is ambiguous about repeats:
`score` uses distinct needle positions (set semantics, the reading we report),
`hit_steps` counts every qualifying decode step. They differ only when a head
re-copies the same position, which is rare but not never.
"""

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from .haystack import NIAHInstance


@dataclass
class InstanceResult:
    needle_id: str
    needle_kind: str
    context_tokens: int
    depth: float
    needle_len: int
    generation: str
    answer_found: bool
    scores: list[list[float]] = field(default_factory=list)  # [layers][heads]
    hit_steps: list[list[int]] = field(default_factory=list)
    # Decode steps that were a copy *opportunity*: the emitted token appears in
    # the needle, so a retrieval head had something to copy. Denominator of the
    # copy-precision score.
    copyable_steps: int = 0


class RetrievalScorer:
    def __init__(self, model, tokenizer, capture, max_new_tokens: int = 32):
        self.model = model
        self.tok = tokenizer
        self.capture = capture
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device
        eos = tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        # Llama-3 instruct ends turns with <|eot_id|>, which is not eos_token_id
        # in every checkpoint's config.
        for name in ("<|eot_id|>", "<|end_of_text|>"):
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid is not None and tid >= 0:
                self.eos_ids.add(tid)

    @torch.no_grad()
    def score_instance(self, inst: NIAHInstance) -> InstanceResult:
        ids = torch.tensor([inst.input_ids], device=self.device)
        prompt_ids = ids[0]
        prompt_len = prompt_ids.numel()
        ns, ne, klen = inst.needle_start, inst.needle_end, inst.needle_len

        n_layers = self.capture.n_layers
        n_heads = self.model.config.num_attention_heads

        cache = DynamicCache()
        cfg = self.model.config
        previous_impl = cfg._attn_implementation

        # Prefill everything but the final token under SDPA; the final token is
        # fed inside the decode loop so that the attention producing the first
        # generated token is captured at q_len=1 rather than in a T x T prefill.
        cfg._attn_implementation = "sdpa"
        self.model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)
        cfg._attn_implementation = previous_impl

        covered = torch.zeros((n_layers, n_heads, klen), dtype=torch.bool, device=self.device)
        hit_steps = torch.zeros((n_layers, n_heads), dtype=torch.int32, device=self.device)

        needle_tokens = set(prompt_ids[ns:ne].tolist())
        copyable_steps = 0

        cur = ids[:, -1:]
        generated: list[int] = []
        for _ in range(self.max_new_tokens):
            with self.capture.recording():
                out = self.model(input_ids=cur, past_key_values=cache, use_cache=True)
            attn = self.capture.stack()  # [layers, heads, keys]
            nxt = int(out.logits[0, -1].argmax())
            if nxt in self.eos_ids:
                break
            generated.append(nxt)
            if nxt in needle_tokens:
                copyable_steps += 1

            j, hit = copy_hits(attn, prompt_ids, nxt, ns, ne)
            hit_steps += hit.to(torch.int32)
            rel = (j - ns).clamp(0, klen - 1)
            covered |= torch.zeros_like(covered).scatter_(
                2, rel.unsqueeze(-1), hit.unsqueeze(-1)
            )

            cur = torch.tensor([[nxt]], device=self.device)

        text = self.tok.decode(generated, skip_special_tokens=True)
        scores = (covered.sum(-1).float() / klen).cpu().tolist()
        return InstanceResult(
            needle_id=inst.needle.id,
            needle_kind=inst.needle.kind,
            context_tokens=inst.context_tokens,
            depth=inst.depth,
            needle_len=klen,
            generation=text,
            answer_found=_normalize(inst.needle.answer) in _normalize(text),
            scores=scores,
            hit_steps=hit_steps.cpu().tolist(),
            copyable_steps=copyable_steps,
        )


def copy_hits(
    attn: torch.Tensor,
    prompt_ids: torch.Tensor,
    generated_token: int,
    needle_start: int,
    needle_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which heads were caught copying `generated_token` out of the needle?

    A hit needs both halves of the rule: the head's attention argmax lands
    *inside* the needle span, and the token sitting at that position *is* the
    token being emitted. Attending to the needle while generating something else
    is not retrieval, and emitting a needle token while attending elsewhere is
    not retrieval either.

    Returns (argmax_index, hit_mask), both [layers, heads].
    """
    j = attn.argmax(dim=-1)
    in_span = (j >= needle_start) & (j < needle_end)
    # Positions past the prompt are freshly generated tokens; they can never be
    # in the needle span, so clamping only keeps the gather in bounds.
    tok_at_j = prompt_ids[j.clamp(max=prompt_ids.numel() - 1)]
    return j, in_span & (tok_at_j == generated_token)


def _normalize(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .,:;!?\"'")
