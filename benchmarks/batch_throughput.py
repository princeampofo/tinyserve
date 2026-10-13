"""Offline throughput: run a fixed set of requests in batches and report output tokens/sec.

    uv run python benchmarks/batch_throughput.py --batch-sizes 1 4 8 16

Every request asks for a different number of output tokens (EOS is ignored so
the lengths are exact), which shows what static batching wastes: a batch runs
until its longest request is done, and finished rows keep using compute.
"""

import argparse
import platform
import random
import time

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tinyserve.engine.engine import Request, run_static_batch
from tinyserve.engine.sampler import SamplingParams
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
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
    run_static_batch(weights, config, warmup)

    print("batch_size,requests,output_tokens,seconds,tokens_per_second,wasted_row_steps")
    for batch_size in args.batch_sizes:
        requests = make_requests(tokenizer, args.num_requests, args.min_tokens, args.max_tokens)
        batches = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        # Each batch takes as many decode steps as its longest request; any row
        # that finished earlier spends the remaining steps doing useless work.
        useful = sum(r.params.max_tokens for r in requests)
        row_steps = sum(len(b) * max(r.params.max_tokens for r in b) for b in batches)

        start = time.perf_counter()
        for batch in batches:
            run_static_batch(weights, config, batch)
        elapsed = time.perf_counter() - start

        tokens = sum(len(r.output_ids) for r in requests)
        print(
            f"{batch_size},{len(requests)},{tokens},{elapsed:.1f},{tokens / elapsed:.1f},"
            f"{1 - useful / row_steps:.0%}"
        )


if __name__ == "__main__":
    main()
