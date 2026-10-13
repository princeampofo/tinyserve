"""Offline throughput: submit a fixed set of requests at once and report output tokens/sec.

    uv run python benchmarks/batch_throughput.py --batching static --batch-sizes 1 4 8 16
    uv run python benchmarks/batch_throughput.py --batching continuous --batch-sizes 16

Every request asks for a different number of output tokens (EOS is ignored so
the lengths are exact), which shows what static batching wastes: a batch runs
until its longest request is done, and finished rows keep using compute.
Continuous batching hands a finished row's slot to the next request instead.

All requests arrive at t=0, so "mean_finish_s" (the average time until a
request is complete) is how long a request waits for its answer on average.
"""

import argparse
import platform
import random
import time

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tinyserve.engine.engine import run_static_batch
from tinyserve.engine.request import Request
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Scheduler
from tinyserve.model.llama import LlamaConfig, LlamaWeights
from tinyserve.model.weights import DEFAULT_MODEL, load_model

TOPICS = ["a lighthouse", "the ocean", "a city at night", "a robot", "a garden", "the moon"]


def make_requests(
    tokenizer: PreTrainedTokenizerBase, n: int, min_out: int, max_out: int
) -> list[Request]:
    rng = random.Random(0)
    requests = []
    for i in range(n):
        prompt = f"Write a story about {TOPICS[i % len(TOPICS)]}." + " Make it vivid." * (i % 4)
        max_tokens = rng.randint(min_out, max_out)
        requests.append(Request(tokenizer(prompt).input_ids, SamplingParams(max_tokens, 0.0)))
    return requests


def run_static(
    weights: LlamaWeights, config: LlamaConfig, requests: list[Request], batch_size: int
) -> list[float]:
    """Run the requests in fixed batches, one after another. Returns each request's finish time."""
    start = time.perf_counter()
    finish_times = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i : i + batch_size]
        run_static_batch(weights, config, batch)
        finish_times += [time.perf_counter() - start] * len(batch)
    return finish_times


def run_continuous(
    weights: LlamaWeights, config: LlamaConfig, requests: list[Request], max_num_seqs: int
) -> list[float]:
    """Run the requests through the scheduler. Returns each request's finish time."""
    max_len = max(len(r.prompt_ids) + r.params.max_tokens for r in requests)
    scheduler = Scheduler(weights, config, max_num_seqs, max_len)
    for request in requests:
        scheduler.add(request)
    start = time.perf_counter()
    finish_times = []
    while scheduler.has_work():
        finish_times += [time.perf_counter() - start] * len(scheduler.step())
    return finish_times


def wasted_row_steps(requests: list[Request], batch_size: int) -> float:
    """Static batching only: each batch takes as many decode steps as its longest
    request, and rows that finished earlier spend the remaining steps on useless work."""
    batches = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
    useful = sum(r.params.max_tokens for r in requests)
    row_steps = sum(len(b) * max(r.params.max_tokens for r in b) for b in batches)
    return 1 - useful / row_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batching", choices=["static", "continuous"], default="static")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    config, weights = load_model(args.model, args.device, dtype)
    config.eos_token_ids = []  # always generate exactly max_tokens
    print(f"model={args.model} device={args.device} dtype={dtype} machine={platform.platform()}")

    warmup = make_requests(tokenizer, 2, 4, 4)
    run_static(weights, config, warmup, 2)
    run_continuous(weights, config, make_requests(tokenizer, 2, 4, 4), 2)

    print(
        "batching,batch_size,requests,output_tokens,seconds,tokens_per_second,mean_finish_s,wasted_row_steps"
    )
    for batch_size in args.batch_sizes:
        requests = make_requests(tokenizer, args.num_requests, args.min_tokens, args.max_tokens)
        if args.batching == "static":
            finish_times = run_static(weights, config, requests, batch_size)
            waste = wasted_row_steps(requests, batch_size)
        else:
            finish_times = run_continuous(weights, config, requests, batch_size)
            waste = 0.0  # finished sequences leave the batch at once
        elapsed = max(finish_times)
        tokens = sum(len(r.output_ids) for r in requests)
        mean_finish = sum(finish_times) / len(finish_times)
        print(
            f"{args.batching},{batch_size},{len(requests)},{tokens},{elapsed:.1f},"
            f"{tokens / elapsed:.1f},{mean_finish:.1f},{waste:.0%}"
        )


if __name__ == "__main__":
    main()
