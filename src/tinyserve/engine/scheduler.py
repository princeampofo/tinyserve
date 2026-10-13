"""Continuous batching over a block-based KV cache, with admission and preemption.

Static batching fixes a batch and runs it until its longest request is done.
Here the batch changes every step instead:

    step():
        1. admit:   waiting requests join (first come, first served) while the
                    cache has blocks for their prompt, and are prefilled
        2. grow:    every running sequence about to start a new block gets one;
                    if none are free, preempt the most recently admitted sequence
        3. decode:  every running sequence gets one new token, in one batched model call
        4. retire:  finished sequences leave and free their blocks

A request that finishes early frees its memory straight away, and a newly
arrived request starts at the next step instead of waiting for a whole batch
to drain.

With block_size=16 (paged), a sequence holds only the blocks it has filled,
so many more sequences fit in the same memory. Running out of blocks becomes
possible, which is what admission and preemption handle. With
block_size=max_model_len (contiguous), each sequence reserves its whole
maximum length in one block at admission and never needs another.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.block_allocator import BlockAllocator
from tinyserve.engine.kv_cache import KVCache, blocks_for, bytes_per_token
from tinyserve.engine.request import Request, append_token
from tinyserve.engine.sampler import sample
from tinyserve.model.llama import LlamaConfig, LlamaWeights, forward

log = logging.getLogger(__name__)


@dataclass(eq=False)
class Sequence:
    """A request that is running: it owns KV cache blocks."""

    request: Request
    block_table: list[int] = field(default_factory=list)  # block ids holding its tokens, in order
    num_cached: int = 0  # how many tokens' K/V are in the cache so far


class Scheduler:
    def __init__(
        self,
        weights: LlamaWeights,
        config: LlamaConfig,
        max_num_seqs: int,
        max_model_len: int,
        num_blocks: int,
        block_size: int = 16,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        if num_blocks * block_size < max_model_len:
            raise ValueError("the KV cache must fit at least one max_model_len sequence")
        self.weights = weights
        self.config = config
        self.tokenizer = tokenizer
        self.device = weights.embed.device
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.cache = KVCache.allocate(
            config, num_blocks, block_size, self.device, weights.embed.dtype
        )
        self.allocator = BlockAllocator(num_blocks)
        # Admission keeps this many blocks spare, so sequences that are already
        # running have room to grow for a while before anyone must be preempted.
        self.watermark_blocks = int(0.01 * num_blocks)
        self.waiting: deque[Request] = deque()
        self.running: list[Sequence] = []  # in admission order
        self.num_preemptions = 0
        self.peak_running = 0

    @classmethod
    def with_memory_budget(
        cls,
        weights: LlamaWeights,
        config: LlamaConfig,
        kv_cache: Literal["paged", "contiguous"],
        kv_cache_gb: float,
        max_num_seqs: int,
        max_model_len: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> "Scheduler":
        """Size the cache from a memory budget (MPS can't report free GPU memory the
        way CUDA does, so the budget is a config value).

        paged: the budget becomes as many 16-token blocks as fit.
        contiguous: the budget becomes as many max_model_len-token slots as fit,
        so it caps how many sequences can run at once.
        """
        budget_tokens = int(kv_cache_gb * 2**30) // bytes_per_token(config, weights.embed.dtype)
        if kv_cache == "paged":
            block_size = 16
            num_blocks = budget_tokens // block_size
        else:
            block_size = max_model_len
            num_blocks = min(max_num_seqs, budget_tokens // max_model_len)
            max_num_seqs = num_blocks
        log.info(
            "%s KV cache: %.2f GB = %d tokens = %d blocks of %d tokens",
            kv_cache,
            kv_cache_gb,
            budget_tokens,
            num_blocks,
            block_size,
        )
        return cls(weights, config, max_num_seqs, max_model_len, num_blocks, block_size, tokenizer)

    def add(self, request: Request) -> None:
        if len(request.prompt_ids) + request.params.max_tokens > self.max_model_len:
            raise ValueError("prompt + max_tokens is longer than max_model_len")
        self.waiting.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one scheduling step. Returns the requests that finished during it."""
        dropped = self._admit()
        self._grow()
        self._decode()
        return dropped + self._retire()

    def _admit(self) -> list[Request]:
        """FCFS: admit waiting requests in arrival order while there is room.
        Returns requests that were aborted while waiting."""
        dropped = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            request = self.waiting[0]
            if request.finished:
                dropped.append(self.waiting.popleft())
                continue
            # Every token so far needs a cache slot, plus one for the token the
            # first decode step will write. A preempted request comes back with
            # its generated tokens too, which are recomputed along with the prompt.
            num_tokens = len(request.prompt_ids) + len(request.output_ids) + 1
            needed = blocks_for(num_tokens, self.block_size)
            reserve = self.watermark_blocks if self.running else 0  # nobody to protect when idle
            if needed > self.allocator.num_free - reserve:
                break  # FCFS: don't let smaller requests behind it jump the queue
            self.waiting.popleft()
            seq = Sequence(request, [self.allocator.allocate() for _ in range(needed)])
            self._prefill(seq)
            self.running.append(seq)
        self.peak_running = max(self.peak_running, len(self.running))
        return dropped

    def _prefill(self, seq: Sequence) -> None:
        """Run the sequence's tokens through the model into its blocks and sample the next token.

        For a new request that's the prompt. For a preempted one it's the prompt
        plus everything it had generated: the K/V were thrown away, so they are
        recomputed in one prefill (much faster than the decode steps that first
        produced them).

        Prefills run one at a time, before the step's decode. While a long
        prompt is being prefilled, every running sequence waits for it.
        """
        request = seq.request
        tokens = torch.tensor([request.prompt_ids + request.output_ids], device=self.device)
        logits = forward(self.weights, self.config, tokens, 0, self.cache, [seq.block_table])
        seq.num_cached = tokens.shape[1]
        token = sample(logits[:, -1], [request.params], [request.rng(self.device)])
        append_token(request, int(token.item()), self.config, self.tokenizer)

    def _grow(self) -> None:
        """Make sure every running sequence has a block for the token it writes this step.

        A sequence needs a new block when its next position starts one (every 16
        tokens). If the pool is empty, preempt: evict the most recently admitted
        sequence, free its blocks, and put it at the front of the waiting queue
        to be recomputed later. The newest sequence has the least work to redo,
        and the oldest ones, closest to finishing, keep going.
        """
        for seq in list(self.running):
            if seq not in self.running or seq.request.finished:
                continue  # preempted earlier in this loop, or done
            if seq.num_cached < len(seq.block_table) * self.block_size:
                continue  # its current block still has room
            while self.allocator.num_free == 0:
                self._preempt(self.running[-1])
            if seq in self.running:  # it may have been the one preempted
                seq.block_table.append(self.allocator.allocate())

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        for block in seq.block_table:
            self.allocator.free(block)
        self.waiting.appendleft(seq.request)
        self.num_preemptions += 1
        log.info(
            "preempted request %s (%d tokens, %d blocks freed); %d running",
            seq.request.id,
            seq.num_cached,
            len(seq.block_table),
            len(self.running),
        )

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
            [s.block_table for s in active],
        )
        tokens = sample(
            logits[:, -1],
            [s.request.params for s in active],
            [s.request.rng(self.device) for s in active],
        )
        for seq, token in zip(active, tokens.tolist()):
            seq.num_cached += 1
            append_token(seq.request, token, self.config, self.tokenizer)

    def _retire(self) -> list[Request]:
        """Remove finished sequences (including aborted ones) and free their blocks."""
        finished = [s for s in self.running if s.request.finished]
        self.running = [s for s in self.running if not s.request.finished]
        for seq in finished:
            for block in seq.block_table:
                self.allocator.free(block)
        return [s.request for s in finished]
