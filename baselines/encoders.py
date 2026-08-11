"""Dense encoder for the retrieval baseline.

bge-m3 is run through plain `transformers` rather than `sentence-transformers`:
its dense head is exactly CLS pooling followed by L2 normalisation (confirmed
against the checkpoint's own `1_Pooling/config.json`), so the wrapper would add
a dependency without adding behaviour.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from paths import resolve_model

BGE_M3 = "BAAI/bge-m3"


class DenseEncoder:
    """CLS-pooled, L2-normalised sentence embeddings."""

    def __init__(
        self,
        model_name: str = BGE_M3,
        device: str = "cuda",
        max_length: int = 512,
        dtype: torch.dtype = torch.float16,
    ):
        path = resolve_model(model_name)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(hidden, dim=-1).float().cpu())
        return torch.cat(out) if out else torch.empty(0)
