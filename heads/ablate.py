"""Head masking, for causally testing whether identified heads do the retrieving.

Zeroing happens on the *input* to `o_proj`, which is the concatenation of the
per-head attention outputs. Slice `[h*head_dim : (h+1)*head_dim]` is head h's
contribution, so zeroing it removes that head from the residual stream while
leaving every other head untouched. Masking the attention weights instead would
renormalize the softmax and perturb the surviving heads.
"""

from contextlib import contextmanager

import torch


class HeadMasker:
    """Zero the output of specific (layer, head) pairs."""

    def __init__(self, model):
        self.model = model
        self.head_dim = getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        )
        self.layers = self._find_layers(model)

    @staticmethod
    def _find_layers(model):
        obj = model
        for attr in ("model", "layers"):
            obj = getattr(obj, attr)
        return list(obj)

    @contextmanager
    def mask(self, heads: list[tuple[int, int]]):
        """Zero the given (layer, head) pairs for the duration of the block."""
        by_layer: dict[int, list[int]] = {}
        for layer, head in heads:
            by_layer.setdefault(layer, []).append(head)

        handles = []
        for layer_idx, head_ids in by_layer.items():
            o_proj = self.layers[layer_idx].self_attn.o_proj
            handles.append(o_proj.register_forward_pre_hook(self._make_hook(head_ids)))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    def _make_hook(self, head_ids: list[int]):
        hd = self.head_dim

        def pre_hook(_module, args):
            x = args[0].clone()
            for h in head_ids:
                x[..., h * hd : (h + 1) * hd] = 0
            return (x,) + tuple(args[1:])

        return pre_hook


@torch.no_grad()
def greedy_answer(model, tokenizer, input_ids: list[int], max_new_tokens: int = 32) -> str:
    """Plain greedy decode. No attention capture, so this runs under fast SDPA."""
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
