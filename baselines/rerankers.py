"""Cross-encoder reranking baseline.

bge-m3 is a bi-encoder: the question and the sentence are embedded separately
and never see each other, so relevance is a dot product between two summaries.
The head signal, by contrast, has full cross-attention between the question and
every sentence. Beating a bi-encoder with a cross-attention method is a soft
win, and "did you compare against a cross-encoder?" is the first question that
result invites -- so the cross-encoder is measured here rather than argued about.

bge-reranker-v2-m3 is the matched choice: same family and same training data as
the bge-m3 bi-encoder, which keeps the comparison about the architecture rather
than about who had the better corpus.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.hotpot import Chunk
from paths import resolve_model

from .retrievers import Retriever

BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker(Retriever):
    """Scores each (question, sentence) pair jointly through one encoder."""

    name = "reranker"

    def __init__(
        self,
        model_name: str = BGE_RERANKER,
        device: str = "cuda",
        max_length: int = 320,
        batch_size: int = 256,
        dtype: torch.dtype = torch.float16,
    ):
        path = resolve_model(model_name)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, question: str, chunks: list[Chunk]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            enc = self.tok(
                [question] * len(batch),
                [c.embed_text for c in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            out += logits[:, 0].float().cpu().tolist()
        return out
