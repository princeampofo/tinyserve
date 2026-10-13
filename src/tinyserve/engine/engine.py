"""Text generation: run the model, sample a token, append it, repeat.

Generation has two phases:

1. Prefill: run the whole prompt through the model in one call. This fills the
   KV cache with every prompt token's K and V and yields the first new token.
2. Decode: feed back the newest token alone, one model call per token, until
   an EOS token, a stop string, or max_tokens.

`generate()` does this for one prompt. `run_static_batch()` does it for a batch
of requests at once, and `Engine` runs batches on a background thread for the
HTTP server.
"""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.sampler import SamplingParams, make_generator, sample
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


@dataclass
class Request:
    """One generation request as the engine sees it: token ids in, token ids out."""

    prompt_ids: list[int]
    params: SamplingParams
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    output_ids: list[int] = field(default_factory=list)
    # "stop" (EOS token or stop string), "length" (hit max_tokens), "abort"
    # (the caller gave up), "error"; None while still running.
    finish_reason: str | None = None
    # Called on the engine thread once the request is done.
    on_finish: Callable[["Request"], None] | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


def append_token(
    request: Request,
    token: int,
    config: LlamaConfig,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Record a newly generated token and decide whether the request is done."""
    request.output_ids.append(token)
    if token in config.eos_token_ids:
        request.finish_reason = "stop"
    elif request.params.stop and tokenizer is not None:
        # Stop strings are text, and one can span several tokens, so check the decoded output.
        text = tokenizer.decode(request.output_ids)
        if any(s in text for s in request.params.stop):
            request.finish_reason = "stop"
    if not request.finished and len(request.output_ids) >= request.params.max_tokens:
        request.finish_reason = "length"


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
            forward(weights, config, torch.tensor([r.prompt_ids], device=device), 0, cache.row(i))[
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
    """Owns the model and runs batches on a background thread.

    The model calls block for milliseconds to seconds, so they can't run on the
    web server's event loop: it would stop accepting and answering requests.
    The server puts requests on a queue, and this thread takes them off in
    batches, runs them, and reports back through each request's on_finish.
    """

    def __init__(
        self,
        weights: LlamaWeights,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_batch_size: int = 8,
        batch_wait_s: float = 0.05,
    ) -> None:
        self.weights = weights
        self.config = config
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_s
        self.queue: queue.Queue[Request] = queue.Queue()

    def submit(self, request: Request) -> None:
        self.queue.put(request)

    def start(self) -> None:
        threading.Thread(target=self._loop, name="engine", daemon=True).start()

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

    def _loop(self) -> None:
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
            for request in batch:
                if request.on_finish is not None:
                    request.on_finish(request)
