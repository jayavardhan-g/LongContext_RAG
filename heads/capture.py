"""Per-head attention capture at decode steps.

transformers 5.x removed `output_attentions`; attention weights now surface only
when the eager attention path runs, since SDPA/flash never materialize the
matrix. Running the whole model eager is not an option here -- an eager prefill
allocates a [heads, T, T] matrix per layer, which is ~4 GB per layer at T=8k and
grows quadratically.

What makes this cheap is that the retrieval score is only ever read at *decode*
steps, where the query length is 1 and the attention matrix is [heads, 1, T].
`LlamaAttention.forward` re-reads `config._attn_implementation` on every call, so
we prefill under SDPA and flip to eager only for the single-token steps.
"""

from contextlib import contextmanager

import torch


class AttentionCapture:
    """Grabs attention weights from every self-attention module of a decoder."""

    def __init__(self, model):
        self.model = model
        self.layers = self._find_layers(model)
        self.n_layers = len(self.layers)
        self.buffer: list[torch.Tensor | None] = [None] * self.n_layers
        self._handles = []
        self._enabled = False

    @staticmethod
    def _find_layers(model):
        for path in ("model.layers", "model.model.layers", "transformer.h"):
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return list(obj)
        raise RuntimeError("could not locate decoder layers on this model")

    def __enter__(self):
        for idx, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer {idx} has no self_attn module")
            self._handles.append(
                attn.register_forward_hook(self._make_hook(idx))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.buffer = [None] * self.n_layers
        return False

    def _make_hook(self, idx):
        def hook(_module, _args, output):
            if not self._enabled:
                return
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            # Detach immediately: keeping the graph alive across 32 layers x
            # many decode steps is what turns this from megabytes into OOM.
            self.buffer[idx] = weights.detach() if weights is not None else None

        return hook

    @contextmanager
    def recording(self):
        """Enable eager attention + hook recording for the enclosed forward pass."""
        cfg = self.model.config
        previous = cfg._attn_implementation
        cfg._attn_implementation = "eager"
        self._enabled = True
        self.buffer = [None] * self.n_layers
        try:
            yield self
        finally:
            self._enabled = False
            cfg._attn_implementation = previous

    def stack(self) -> torch.Tensor:
        """Return captured attention as [layers, heads, keys] for a 1-token query."""
        missing = [i for i, b in enumerate(self.buffer) if b is None]
        if missing:
            raise RuntimeError(
                "no attention weights captured for layers "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}; the eager path "
                "did not run -- check that config._attn_implementation was honored"
            )
        # each entry is [batch, heads, q_len, keys] with batch == q_len == 1
        out = torch.cat([b[:, :, -1, :] for b in self.buffer], dim=0)
        return out
