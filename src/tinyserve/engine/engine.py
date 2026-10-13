"""Text generation: run the model, sample a token, append it, repeat.

Generation has two phases:

1. Prefill: run the whole prompt through the model in one call. This fills the
   KV cache with every prompt token's K and V and yields the first new token.
2. Decode: feed back the newest token alone, one model call per token, until
   an EOS token, a stop string, or max_tokens.

`generate()` does this for one prompt. `run_static_batch()` does it for a batch
of requests at once. `Engine` serves the HTTP server from a background thread,
with either static batching or continuous batching (`scheduler.py`).
"""

import logging
import queue
import threading
import time
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.request import Request, append_token
from tinyserve.engine.sampler import SamplingParams, make_generator, sample
from tinyserve.engine.scheduler import Scheduler
from tinyserve.model.llama import LlamaConfig, LlamaWeights, forward

log = logging.getLogger(__name__)


@torch.inference_mode()
def generate(
    weights: LlamaWeights,
    config: LlamaConfig,
    prompt_ids: list[int],
    params: SamplingParams,
    use_kv_cache: bool = True,
) -> list[int]:
    """Generate up to params.max_tokens tokens for one prompt. Returns only the new tokens
    (including the EOS token, if one ended the sequence).

    use_kv_cache=False is the simple version kept for comparison: every step
    reruns the model over the entire sequence so far. Its output is identical;
    it just redoes all the work for earlier tokens at every step.
    """
    device = weights.embed.device
    generator = make_generator(params.seed, device)
    output: list[int] = []

    cache = None
    if use_kv_cache:
        max_len = len(prompt_ids) + params.max_tokens
        cache = KVCache.allocate(config, 1, max_len, device, weights.embed.dtype)

    input_ids = torch.tensor([prompt_ids], device=device)  # prefill: the whole prompt
    start_pos = 0
    while len(output) < params.max_tokens:
        logits = forward(weights, config, input_ids, start_pos, cache)
        next_token = int(sample(logits[:, -1], [params], [generator]).item())
        output.append(next_token)
        if next_token in config.eos_token_ids:
            break

        if cache is not None:
            # Decode: only the new token goes through the model; the cache
            # already holds K and V for everything before it.
            start_pos += input_ids.shape[1]
            input_ids = torch.tensor([[next_token]], device=device)
        else:
            input_ids = torch.tensor([prompt_ids + output], device=device)

    return output


@torch.inference_mode()
def run_static_batch(
    weights: LlamaWeights,
    config: LlamaConfig,
    requests: list[Request],
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> None:
    """Static batching: prefill every request, then decode them all together until
    the last one finishes. Results are written into each Request.

    Batching pays off because decode is memory-bound: each step has to read all
    the weights anyway, and one read can serve every row of the batch. The catch
    is that a request which finishes early keeps its row, and keeps costing a
    row of compute every step, until the slowest request in the batch is done.
    New requests can't join until then either. Continuous batching fixes both.
    """
    device = weights.embed.device
    # Every row may run for as many steps as the longest request, so each row
    # needs room for the longest prompt plus the largest max_tokens.
    max_len = max(len(r.prompt_ids) for r in requests) + max(r.params.max_tokens for r in requests)
    cache = KVCache.allocate(config, len(requests), max_len, device, weights.embed.dtype)
    params = [r.params for r in requests]
    generators = [make_generator(r.params.seed, device) for r in requests]

    # Prefill one prompt at a time, each into its own cache row. Batching
    # prompts of different lengths would need padding, and prefill is already
    # compute-bound, so batching it would gain little.
    last_logits = torch.cat(
        [
            forward(weights, config, torch.tensor([r.prompt_ids], device=device), 0, cache, [i])[
                :, -1
            ]
            for i, r in enumerate(requests)
        ]
    )
    lengths = [len(r.prompt_ids) for r in requests]

    while True:
        next_tokens = sample(last_logits, params, generators).tolist()
        for request, token in zip(requests, next_tokens):
            if not request.finished:
                append_token(request, token, config, tokenizer)
        if all(r.finished for r in requests):
            return

        # Decode one token for every row, finished rows included: static
        # batching has no way to hand their slot to anyone else.
        input_ids = torch.tensor([[t] for t in next_tokens], device=device)
        last_logits = forward(weights, config, input_ids, lengths, cache)[:, -1]
        lengths = [n + 1 for n in lengths]


class Engine:
    """Owns the model and runs requests on a background thread.

    The model calls block for milliseconds to seconds, so they can't run on the
    web server's event loop: it would stop accepting and answering requests.
    The server puts requests on a queue; this thread takes them off, runs them,
    and reports back through each request's on_finish.

    batching="continuous" (the default) runs the scheduler step by step, letting
    requests join and leave the batch at every step. batching="static" is the
    simpler baseline: collect up to max_batch_size requests, run them to completion.
    """

    def __init__(
        self,
        weights: LlamaWeights,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_batch_size: int = 8,
        batch_wait_s: float = 0.05,
        batching: Literal["continuous", "static"] = "continuous",
        max_model_len: int = 4096,
    ) -> None:
        self.weights = weights
        self.config = config
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_s
        self.batching = batching
        self.max_model_len = max_model_len
        self.queue: queue.Queue[Request] = queue.Queue()

    def submit(self, request: Request) -> None:
        self.queue.put(request)

    def start(self) -> None:
        target = self._continuous_loop if self.batching == "continuous" else self._static_loop
        threading.Thread(target=target, name="engine", daemon=True).start()

    def _finish(self, requests: list[Request]) -> None:
        for request in requests:
            if request.on_finish is not None:
                request.on_finish(request)

    def _continuous_loop(self) -> None:
        scheduler = Scheduler(
            self.weights, self.config, self.max_batch_size, self.max_model_len, self.tokenizer
        )
        while True:
            # Block only when idle; otherwise pick up new arrivals without waiting
            # and go straight back to stepping the running sequences.
            if not scheduler.has_work():
                scheduler.add(self.queue.get())
            while not self.queue.empty():
                scheduler.add(self.queue.get_nowait())
            try:
                finished = scheduler.step()
            except Exception:
                log.exception("scheduler step failed; failing all in-flight requests")
                finished = [s.request for s in scheduler.running] + list(scheduler.waiting)
                for request in finished:
                    request.finish_reason = request.finish_reason or "error"
                scheduler = Scheduler(
                    self.weights,
                    self.config,
                    self.max_batch_size,
                    self.max_model_len,
                    self.tokenizer,
                )
            self._finish(finished)

    def _next_batch(self) -> list[Request]:
        """Block until a request arrives, then wait up to batch_wait_s for more to
        fill the batch. Waiting longer makes bigger batches (more throughput) but
        delays the first request (worse latency)."""
        batch = [self.queue.get()]
        deadline = time.monotonic() + self.batch_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self.queue.get(timeout=remaining))
            except queue.Empty:
                break
        return [r for r in batch if not r.finished]  # skip requests aborted while queued

    def _static_loop(self) -> None:
        while True:
            batch = self._next_batch()
            if not batch:
                continue
            log.info("running a batch of %d requests", len(batch))
            try:
                run_static_batch(self.weights, self.config, batch, self.tokenizer)
            except Exception:
                log.exception("batch of %d requests failed", len(batch))
                for request in batch:
                    if not request.finished:
                        request.finish_reason = "error"
            self._finish(batch)
