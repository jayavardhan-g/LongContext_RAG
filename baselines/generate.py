"""Answer generation with Llama-3.1-8B-Instruct.

The same generator serves every method, so the only thing that varies across
conditions is which chunks made it into the context.
"""

import time

import torch

INSTRUCTION = (
    "Answer the question using only the context above.\n"
    "Reply with the shortest possible answer -- a name, a date, or a short "
    "phrase copied from the context. If it is a yes/no question, reply with "
    "exactly yes or no. Do not explain and do not write a sentence."
)


def build_prompt(tokenizer, context: str, question: str) -> str:
    user = f"Context:\n{context}\n\nQuestion: {question}\n\n{INSTRUCTION}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_answer(text: str) -> str:
    """Trim the model's reply to the answer span.

    Even under an explicit instruction the model sometimes prefixes "Answer:"
    or appends a sentence; both are formatting, not a wrong answer, and letting
    them through would understate EM for every method equally.
    """
    ans = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "answer:", "A:"):
        if ans.startswith(prefix):
            ans = ans[len(prefix) :].strip()
    return ans.strip().strip('"').rstrip(".").strip()


class Generator:
    def __init__(self, model, tokenizer, max_new_tokens: int = 32):
        self.model = model
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _batches(self, order, prompts, batch_size, max_batch_chars):
        """Group sorted prompts into batches bounded by both count and total size.

        A fixed batch size is fine at 512-token contexts and OOMs at 32k, so the
        batch also yields early once its padded footprint would exceed
        `max_batch_chars`. Since `order` is sorted by length, the bound binds
        only on the long batches.
        """
        batch: list[int] = []
        for i in order:
            longest = max(len(prompts[j]) for j in batch + [i])
            if batch and (
                len(batch) >= batch_size or longest * (len(batch) + 1) > max_batch_chars
            ):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        batch_size: int = 8,
        max_batch_chars: int = 400_000,
    ) -> list[str]:
        """Greedy decoding, batched.

        Prompts are sorted by length before batching so a 32k context does not
        pad a batch of 512-token ones out to its own length, then restored to
        the caller's order.
        """
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out: dict[int, str] = {}

        side = self.tok.padding_side
        self.tok.padding_side = "left"
        t0, last_report = time.time(), 0
        try:
            for idx in self._batches(order, prompts, batch_size, max_batch_chars):
                enc = self.tok(
                    [prompts[j] for j in idx],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.model.device)
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
                new = gen[:, enc.input_ids.shape[1] :]
                for j, seq in zip(idx, new):
                    out[j] = clean_answer(
                        self.tok.decode(seq, skip_special_tokens=True)
                    )
                # Compared against a running marker rather than `% 500`, which
                # silently stops firing once long contexts shrink the batch
                # below the step and no count lands in the window.
                if len(out) - last_report >= 500 or len(out) == len(prompts):
                    last_report = len(out)
                    done, total = len(out), len(prompts)
                    rate = done / max(time.time() - t0, 1e-6)
                    print(
                        f"[gen] {done}/{total}  {time.time() - t0:.0f}s  "
                        f"eta {(total - done) / max(rate, 1e-9):.0f}s",
                        flush=True,
                    )
        finally:
            self.tok.padding_side = side

        return [out[i] for i in range(len(prompts))]

    @torch.inference_mode()
    def time_one(self, prompt: str) -> tuple[str, float, int]:
        """Unbatched latency for a single prompt.

        Latency is measured here and not during the batched accuracy run,
        because batching mixes wall-clock across conditions and would make the
        compute-saved claim meaningless.
        """
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        text = self.tok.decode(gen[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
        return clean_answer(text), elapsed, enc.input_ids.shape[1]
