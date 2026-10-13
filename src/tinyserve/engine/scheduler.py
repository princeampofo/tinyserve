"""Continuous batching: decide, at every step, which sequences run.

Static batching fixes a batch and runs it until its longest request is done.
Here the batch changes every step instead:

    step():
        1. admit:  waiting requests take free cache slots (first come, first served)
                   and are prefilled
        2. decode: every running sequence gets one new token, in one batched model call
        3. retire: finished sequences leave and free their slots for step 1 of the
                   next step

A request that finishes early frees its slot straight away, and a newly
arrived request starts at the next step instead of waiting for a whole batch
to drain.
"""

from collections import deque
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.request import Request, append_token
from tinyserve.engine.sampler import make_generator, sample
from tinyserve.model.llama import LlamaConfig, LlamaWeights, forward


@dataclass
class Sequence:
    """A request that is running: it owns a cache slot and its own RNG."""

    request: Request
    slot: int  # which row of the KV cache holds this sequence's keys and values
    generator: torch.Generator
    num_cached: int = 0  # how many tokens' K/V are in the slot so far


class Scheduler:
    def __init__(
        self,
        weights: LlamaWeights,
        config: LlamaConfig,
        max_num_seqs: int,
        max_model_len: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.weights = weights
        self.config = config
        self.tokenizer = tokenizer
        self.device = weights.embed.device
        self.max_model_len = max_model_len
        # The contiguous cache reserves max_model_len tokens for every slot up
        # front, whether a sequence ends up using 20 tokens or all of them.
        # This is the memory a paged cache will stop wasting.
        self.cache = KVCache.allocate(
            config, max_num_seqs, max_model_len, self.device, weights.embed.dtype
        )
        self.free_slots = list(range(max_num_seqs))
        self.waiting: deque[Request] = deque()
        self.running: list[Sequence] = []

    def add(self, request: Request) -> None:
        if len(request.prompt_ids) + request.params.max_tokens > self.max_model_len:
            raise ValueError("prompt + max_tokens does not fit in a cache slot")
        self.waiting.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one scheduling step. Returns the requests that finished during it."""
        dropped = self._admit()
        self._decode()
        return dropped + self._retire()

    def _admit(self) -> list[Request]:
        """FCFS: admit waiting requests in arrival order while slots are free.
        Returns requests that were aborted while waiting."""
        dropped = []
        while self.waiting and self.free_slots:
            request = self.waiting.popleft()
            if request.finished:
                dropped.append(request)
                continue
            seq = Sequence(
                request, self.free_slots.pop(), make_generator(request.params.seed, self.device)
            )
            self._prefill(seq)
            self.running.append(seq)
        return dropped

    def _prefill(self, seq: Sequence) -> None:
        """Run the prompt through the model into the sequence's slot and sample its first token.

        Prefills run one prompt at a time, before the step's decode. While a
        long prompt is being prefilled, every running sequence waits for it:
        that pause shows up as a spike in their inter-token latency.
        """
        prompt = torch.tensor([seq.request.prompt_ids], device=self.device)
        logits = forward(self.weights, self.config, prompt, 0, self.cache, [seq.slot])
        seq.num_cached = prompt.shape[1]
        token = sample(logits[:, -1], [seq.request.params], [seq.generator])
        append_token(seq.request, int(token.item()), self.config, self.tokenizer)

    def _decode(self) -> None:
        """One new token for every running sequence, in a single batched model call."""
        active = [s for s in self.running if not s.request.finished]
        if not active:
            return
        input_ids = torch.tensor([[s.request.output_ids[-1]] for s in active], device=self.device)
        logits = forward(
            self.weights,
            self.config,
            input_ids,
            [s.num_cached for s in active],
            self.cache,
            [s.slot for s in active],
        )
        tokens = sample(
            logits[:, -1], [s.request.params for s in active], [s.generator for s in active]
        )
        for seq, token in zip(active, tokens.tolist()):
            seq.num_cached += 1
            append_token(seq.request, token, self.config, self.tokenizer)

    def _retire(self) -> list[Request]:
        """Remove finished sequences (including aborted ones) and free their slots."""
        finished = [s for s in self.running if s.request.finished]
        self.running = [s for s in self.running if not s.request.finished]
        for seq in finished:
            self.free_slots.append(seq.slot)
        return [s.request for s in finished]
