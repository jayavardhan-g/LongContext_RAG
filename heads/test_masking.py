"""Verify the masking mechanism itself before trusting any ablation result.

A null ablation result has two very different explanations -- genuine redundancy,
or a mask that never fired. These checks separate them:

1. masking a head must change the logits at all;
2. masking more heads must change them more;
3. masking an entire layer's heads must be catastrophic.

If (3) is not catastrophic, the hook is not doing what it claims and every
ablation number is meaningless.

Run:  HF_HUB_OFFLINE=1 python -m heads.test_masking
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ablate import HeadMasker
from .run_head_id import DEFAULT_MODEL, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    prompt = "The capital of France is Paris. The capital of Japan is"
    ids = torch.tensor([tok(prompt, add_special_tokens=True).input_ids], device=device)
    masker = HeadMasker(model)

    with torch.no_grad():
        clean = model(ids).logits[0, -1].float()

    def delta(heads):
        with torch.no_grad(), masker.mask(heads):
            out = model(ids).logits[0, -1].float()
        return float((out - clean).abs().max()), tok.decode(int(out.argmax()))

    checks = []
    d1, tok1 = delta([(15, 30)])
    checks.append((f"masking L15H30 changes logits (max|d|={d1:.3f})", d1 > 0))

    d5, _ = delta([(15, 30), (8, 1), (16, 1), (24, 27), (16, 20)])
    checks.append((f"masking 5 heads changes more than 1 ({d5:.3f} > {d1:.3f})", d5 > d1))

    layer_heads = [(15, h) for h in range(model.config.num_attention_heads)]
    dL, tokL = delta(layer_heads)
    checks.append((f"masking all of layer 15 is largest ({dL:.3f})", dL > d5))

    all_heads = [
        (l, h)
        for l in range(model.config.num_hidden_layers)
        for h in range(model.config.num_attention_heads)
    ]
    dA, tokA = delta(all_heads)
    checks.append((f"masking every head destroys the model ({dA:.3f})", dA > dL))

    clean_tok = tok.decode(int(clean.argmax()))
    checks.append((f"clean prediction is sensible (got {clean_tok!r})",
                   "Tokyo" in clean_tok or "tokyo" in clean_tok.lower()))
    checks.append((f"all-heads-masked prediction differs (got {tokA!r})",
                   tokA != clean_tok))

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(ok for _, ok in checks)
    print(f"\n{'MASKING VERIFIED' if passed else 'MASKING BROKEN — ablation numbers are meaningless'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
