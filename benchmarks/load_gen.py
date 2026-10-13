"""Load generator: streaming requests with Poisson arrivals against an OpenAI-compatible server.

    uv run python benchmarks/load_gen.py --rate 2 --duration 120
    uv run python benchmarks/load_gen.py --rate 4 --duration 60 --output results.csv

Requests arrive like independent users would: the gap between arrivals is
random (exponential) with mean 1/rate, so load comes in bursts and lulls rather
than a perfectly even drip. Prompt and output lengths are drawn from a fixed
synthetic distribution (seeded, so every run sends the same workload).

Latency is measured on the client, from the moment a request is sent:
    TTFT   until the first streamed text arrives
    ITL    between consecutive streamed chunks
    E2E    until the stream ends
"""

import argparse
import asyncio
import csv
import random
import statistics
import time
from dataclasses import dataclass, field

from openai import APIError, AsyncOpenAI

WORDS = [
    "the",
    "a",
    "model",
    "server",
    "token",
    "cache",
    "batch",
    "request",
    "memory",
    "latency",
    "queue",
    "block",
    "stream",
]


@dataclass
class Result:
    sent_at: float  # seconds since the run started
    prompt_words: int
    max_tokens: int
    ttft: float | None = None
    itls: list[float] = field(default_factory=list)
    e2e: float | None = None
    output_tokens: int = 0
    error: str = ""


def make_workload(n: int, min_tokens: int, max_tokens: int, seed: int) -> list[tuple[str, int]]:
    """(prompt, max_tokens) pairs: prompts of 20-200 words, outputs uniform in [min, max] tokens."""
    rng = random.Random(seed)
    workload = []
    for _ in range(n):
        prompt = " ".join(rng.choice(WORDS) for _ in range(rng.randint(20, 200)))
        workload.append((prompt, rng.randint(min_tokens, max_tokens)))
    return workload


async def send(client: AsyncOpenAI, model: str, prompt: str, result: Result) -> None:
    start = time.perf_counter()
    last = start
    try:
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=result.max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},  # exact output lengths make runs comparable
        )
        async for chunk in stream:
            now = time.perf_counter()
            if chunk.usage is not None:
                result.output_tokens = chunk.usage.completion_tokens
            if not chunk.choices or not chunk.choices[0].text:
                continue
            if result.ttft is None:
                result.ttft = now - start
            else:
                result.itls.append(now - last)
            last = now
        result.e2e = time.perf_counter() - start
    except APIError as e:  # record and keep going: one failed request shouldn't end the run
        result.error = repr(e)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


async def run(args: argparse.Namespace) -> list[Result]:
    client = AsyncOpenAI(base_url=f"{args.url}/v1", api_key="unused", timeout=600, max_retries=0)
    model = (await client.models.list()).data[0].id
    # Enough requests to cover the duration at this rate, with room to spare.
    workload = make_workload(
        int(args.rate * args.duration * 2) + 10, args.min_tokens, args.max_tokens, args.seed
    )
    arrivals = random.Random(args.seed + 1)

    results: list[Result] = []
    tasks = []
    start = time.perf_counter()
    for prompt, max_tokens in workload:
        now = time.perf_counter() - start
        if now >= args.duration:
            break
        result = Result(sent_at=now, prompt_words=len(prompt.split()), max_tokens=max_tokens)
        results.append(result)
        tasks.append(asyncio.create_task(send(client, model, prompt, result)))
        await asyncio.sleep(arrivals.expovariate(args.rate))
    print(f"sent {len(tasks)} requests in {args.duration}s; waiting for the last ones to finish")
    await asyncio.gather(*tasks)
    return results


def summarize(results: list[Result], elapsed: float) -> None:
    ok = [r for r in results if not r.error and r.e2e is not None]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    itls = [x for r in ok for x in r.itls]
    tokens = sum(r.output_tokens for r in ok)
    print(f"requests: {len(ok)} ok, {len(results) - len(ok)} failed, in {elapsed:.0f}s")
    print(f"output throughput: {tokens / elapsed:.1f} tokens/s")
    print(
        f"TTFT  p50 {percentile(ttfts, 50) * 1000:.0f} ms  p99 {percentile(ttfts, 99) * 1000:.0f} ms"
    )
    print(
        f"ITL   p50 {percentile(itls, 50) * 1000:.0f} ms  p99 {percentile(itls, 99) * 1000:.0f} ms"
    )
    if ok:
        print(f"E2E   mean {statistics.mean(r.e2e or 0 for r in ok):.1f} s")


def write_csv(results: list[Result], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sent_at",
                "prompt_words",
                "max_tokens",
                "output_tokens",
                "ttft_s",
                "mean_itl_s",
                "e2e_s",
                "error",
            ]
        )
        for r in results:
            mean_itl = statistics.mean(r.itls) if r.itls else ""
            writer.writerow(
                [
                    f"{r.sent_at:.3f}",
                    r.prompt_words,
                    r.max_tokens,
                    r.output_tokens,
                    r.ttft,
                    mean_itl,
                    r.e2e,
                    r.error,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=2.0, help="mean requests per second")
    parser.add_argument("--duration", type=float, default=60, help="seconds to keep sending")
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write one CSV row per request here")
    args = parser.parse_args()

    start = time.perf_counter()
    results = asyncio.run(run(args))
    summarize(results, time.perf_counter() - start)
    if args.output:
        write_csv(results, args.output)


if __name__ == "__main__":
    main()
