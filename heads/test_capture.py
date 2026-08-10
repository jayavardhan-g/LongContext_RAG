"""Plumbing checks for the attention-capture path, on a tiny random Llama.

Weights are random, so the scores are meaningless -- what is under test is the
mechanism: that the eager flip actually happens, that a hook fires for every
layer, that the captured rows are real attention distributions over exactly the
right number of keys, and that a hand-planted copy is detected as a hit.

Run offline:  HF_HUB_OFFLINE=1 python -m heads.test_capture
"""

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from .capture import AttentionCapture
from .haystack import PromptBuilder
from .needles import NEEDLES
from .scorer import RetrievalScorer

TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_tiny(vocab_size: int, device: str):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA, like the 8B model
        max_position_embeddings=4096,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(device).eval()


def hit_rule_checks(device):
    """Plant known attention patterns and assert the copy rule fires correctly."""
    from .scorer import copy_hits

    # context of 10 tokens, needle occupies [4, 7)
    prompt = torch.tensor([10, 11, 12, 13, 90, 91, 92, 14, 15, 16], device=device)
    ns, ne = 4, 7
    attn = torch.zeros((1, 4, 10), device=device)
    attn[0, 0, 5] = 1.0  # argmax inside needle, at the token being generated
    attn[0, 1, 5] = 1.0  # argmax inside needle, but wrong token generated
    attn[0, 2, 9] = 1.0  # argmax outside needle, on an equal-valued token
    attn[0, 3, 0] = 1.0  # attention sink

    prompt[9] = 91  # make the outside-span token identical to the emitted one

    _, hit = copy_hits(attn, prompt, generated_token=91, needle_start=ns, needle_end=ne)
    out = []
    out.append(("hit: argmax in needle AND token matches", bool(hit[0, 0])))
    out.append(("no hit: right position, wrong emitted token",
                not bool(copy_hits(attn, prompt, 999, ns, ne)[1][0, 1])))
    out.append(("no hit: matching token but argmax outside needle", not bool(hit[0, 2])))
    out.append(("no hit: attention sink at position 0", not bool(hit[0, 3])))
    return out


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
        device = f"cuda:{max(free)[1]}"

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = build_tiny(len(tok), device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    checks = []

    # 1. hooks fire on every layer, and only under recording()
    with AttentionCapture(model) as cap:
        ids = torch.randint(0, 1000, (1, 16), device=device)
        model(input_ids=ids)
        checks.append(("no capture outside recording()", all(b is None for b in cap.buffer)))

        with cap.recording():
            checks.append(("impl flipped to eager", model.config._attn_implementation == "eager"))
            model(input_ids=ids)
        attn = cap.stack()
        checks.append(("impl restored after", model.config._attn_implementation == "sdpa"))
        checks.append((f"stack shape == [{n_layers},{n_heads},16]",
                       tuple(attn.shape) == (n_layers, n_heads, 16)))
        checks.append(("rows sum to 1", torch.allclose(
            attn.float().sum(-1), torch.ones(n_layers, n_heads, device=device), atol=2e-2)))

        # 2. a decode step attends over prompt + generated tokens
        from transformers import DynamicCache
        cache = DynamicCache()
        prompt = torch.randint(0, 1000, (1, 40), device=device)
        model.config._attn_implementation = "sdpa"
        model(input_ids=prompt[:, :-1], past_key_values=cache, use_cache=True)
        model.config._attn_implementation = "sdpa"
        with cap.recording():
            model(input_ids=prompt[:, -1:], past_key_values=cache, use_cache=True)
        a = cap.stack()
        checks.append((f"decode-step keys == 40 (q_len=1)", a.shape[-1] == 40))
        checks.append(("decode stack is 1 query row", a.ndim == 3))

        # 3. end-to-end scorer runs and produces in-range scores
        builder = PromptBuilder(tok)
        inst = builder.build(NEEDLES[1], 600, 0.5)
        span = tok.decode(inst.input_ids[inst.needle_start:inst.needle_end])
        checks.append(("needle span exact", span.strip() == NEEDLES[1].text.strip()))

        scorer = RetrievalScorer(model, tok, cap, max_new_tokens=8)
        res = scorer.score_instance(inst)
        s = torch.tensor(res.scores)
        checks.append((f"scores shape [{n_layers},{n_heads}]", tuple(s.shape) == (n_layers, n_heads)))
        checks.append(("scores in [0,1]", bool((s >= 0).all() and (s <= 1).all())))

    checks.extend(hit_rule_checks(device))

    ok = all(p for _, p in checks)
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
