"""Load generator: streaming chat requests with Poisson arrivals against an OpenAI-compatible server.

    uv run python benchmarks/load_gen.py --rate 2 --duration 60
    uv run python benchmarks/load_gen.py --rate 1 --warmup 15 --duration 75 --out-dir results/run1

Requests arrive like independent users would: the gap between arrivals is
random (exponential) with mean 1/rate, so load comes in bursts and lulls rather
than a perfectly even drip.

Workload (synthetic, seeded, the same for every server): each request asks for
an essay on one of a few topics, padded with 0-180 words of context, so prompts
are roughly 50-300 tokens; max_tokens is uniform in [min_tokens, max_tokens].
Servers that support `ignore_eos` (tinyserve, the HF baseline) generate exactly
max_tokens; others may stop earlier, which the results report as fewer tokens.

A run has three phases:
    warmup    requests are sent, nothing is measured (fills the batch, warms kernels)
    measure   throughput = tokens received during this window / its length;
              latency stats come from requests *sent* during this window
    drain     sending stops; wait up to drain_timeout for the measured requests
              to finish, then cancel the rest. A request that never got its
              first token reports TTFT as a lower bound (cancel time - send
              time) and is counted as censored, so overload can't hide its tail.

Latency is measured on the client: TTFT until the first streamed text, ITL
between consecutive streamed chunks (one chunk per token on every server here).
"""

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from openai import APIError, AsyncOpenAI

TOPICS = [
    "the history of lighthouses",
    "how a city water system works",
    "the life cycle of stars",
    "why the ocean is salty",
    "how bread rises",
    "the invention of the printing press",
    "how bees communicate",
    "the physics of bicycles",
]
FILLER = (
    "Consider the people involved, the tools they used, the problems they faced, "
    "the mistakes that were made along the way, and what changed as a result."
)


@dataclass
class Request:
    sent_at: float  # seconds since the run started
    max_tokens: int
    ttft: float | None = None
    ttft_censored: bool = False  # cancelled before the first token: ttft is a lower bound
    chunk_times: list[float] = field(default_factory=list)  # run-relative, one per token
    e2e: float | None = None
    error: str = ""


def make_workload(n: int, min_tokens: int, max_tokens: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    words = FILLER.split()
    workload = []
    for _ in range(n):
        filler = " ".join(rng.choice(words) for _ in range(rng.randint(0, 180)))
        prompt = f"Write a detailed essay about {rng.choice(TOPICS)}. {filler}"
        workload.append((prompt, rng.randint(min_tokens, max_tokens)))
    return workload


async def send(client: AsyncOpenAI, model: str, prompt: str, request: Request, t0: float) -> None:
    start = time.perf_counter()
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=request.max_tokens,
            temperature=0,
            stream=True,
            extra_body={"ignore_eos": True},  # exact lengths where the server supports it
        )
        async for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            now = time.perf_counter()
            if request.ttft is None:
                request.ttft = now - start
            request.chunk_times.append(now - t0)
        request.e2e = time.perf_counter() - start
    except APIError as e:  # record and keep going: one failed request shouldn't end the run
        request.error = repr(e)
    except asyncio.CancelledError:  # still running when the drain timed out
        if request.ttft is None:
            request.ttft = time.perf_counter() - start
            request.ttft_censored = True
        raise


async def poll_metrics(url: str, t0: float, trace: list[dict[str, float]]) -> None:
    """Sample the server's /metrics once a second (tinyserve only; others have none)."""
    wanted = ["running_requests", "waiting_requests", "kv_blocks_used_ratio", "preemptions_total"]
    async with httpx.AsyncClient() as http:
        while True:
            try:
                response = await http.get(f"{url}/metrics", timeout=2)
            except httpx.HTTPError:
                return
            sample = {"t": time.perf_counter() - t0}
            for line in response.text.splitlines():
                name, _, value = line.partition(" ")
                if name in wanted:
                    sample[name] = float(value)
            if len(sample) == 1:
                return  # this server has no tinyserve metrics
            trace.append(sample)
            await asyncio.sleep(1)


async def run_load(
    url: str,
    rate: float,
    warmup: float,
    duration: float,
    drain_timeout: float,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> tuple[list[Request], list[dict[str, float]]]:
    client = AsyncOpenAI(base_url=f"{url}/v1", api_key="unused", timeout=None, max_retries=0)
    model = (await client.models.list()).data[0].id
    n = int(rate * (warmup + duration) * 2) + 20  # more than will be needed
    workload = make_workload(n, min_tokens, max_tokens, seed)
    arrivals = random.Random(seed + 1)

    t0 = time.perf_counter()
    trace: list[dict[str, float]] = []
    poller = asyncio.create_task(poll_metrics(url, t0, trace))
    requests: list[Request] = []
    tasks: list[asyncio.Task[None]] = []
    for prompt, tokens in workload:
        now = time.perf_counter() - t0
        if now >= warmup + duration:
            break
        request = Request(sent_at=now, max_tokens=tokens)
        requests.append(request)
        tasks.append(asyncio.create_task(send(client, model, prompt, request, t0)))
        await asyncio.sleep(arrivals.expovariate(rate))

    measured = [t for t, r in zip(tasks, requests) if r.sent_at >= warmup]
    if measured:
        await asyncio.wait(measured, timeout=drain_timeout)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    poller.cancel()
    await client.close()
    return requests, trace


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def summarize(
    requests: list[Request], trace: list[dict[str, float]], warmup: float, duration: float
) -> dict[str, float]:
    end = warmup + duration
    window_tokens = sum(warmup <= t < end for r in requests for t in r.chunk_times)
    measured = [r for r in requests if warmup <= r.sent_at < end]
    ttfts = [r.ttft for r in measured if r.ttft is not None]
    itls = [b - a for r in measured for a, b in zip(r.chunk_times, r.chunk_times[1:])]
    e2es = [r.e2e for r in measured if r.e2e is not None]
    window = [s for s in trace if warmup <= s["t"] < end]
    preemptions = float("nan")
    if len(window) > 1 and "preemptions_total" in window[0]:
        preemptions = window[-1]["preemptions_total"] - window[0]["preemptions_total"]
    return {
        "output_tokens_per_s": window_tokens / duration,
        "requests_sent": len(measured),
        "requests_completed": len(e2es),
        "requests_failed": sum(bool(r.error) for r in measured),
        "ttft_censored": sum(r.ttft_censored for r in measured),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p99_s": percentile(ttfts, 99),
        "itl_p50_s": percentile(itls, 50),
        "itl_p99_s": percentile(itls, 99),
        "e2e_p50_s": percentile(e2es, 50),
        "mean_output_tokens": (
            statistics.mean(len(r.chunk_times) for r in measured) if measured else 0.0
        ),
        "peak_running": max((s.get("running_requests", 0) for s in window), default=float("nan")),
        "peak_kv_used": max(
            (s.get("kv_blocks_used_ratio", 0) for s in window), default=float("nan")
        ),
        "preemptions": preemptions,
    }


def write_outputs(
    out_dir: Path,
    requests: list[Request],
    trace: list[dict[str, float]],
    summary: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out_dir / "requests.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sent_at", "max_tokens", "output_tokens", "ttft_s", "ttft_censored", "e2e_s", "error"]
        )
        for r in requests:
            writer.writerow(
                [f"{r.sent_at:.3f}", r.max_tokens, len(r.chunk_times), r.ttft, r.ttft_censored,
                 r.e2e, r.error]
            )  # fmt: skip
    # Tokens received in each second of the run: shows warmup, steady state, and any drift.
    last = int(max((t for r in requests for t in r.chunk_times), default=0))
    per_second = [0] * (last + 1)
    for r in requests:
        for t in r.chunk_times:
            per_second[int(t)] += 1
    with open(out_dir / "tokens_per_second.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["second", "tokens"])
        writer.writerows(enumerate(per_second))
    if trace:
        with open(out_dir / "server_trace.csv", "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=list(trace[0]))
            dict_writer.writeheader()
            dict_writer.writerows(trace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=2.0, help="mean requests per second")
    parser.add_argument("--warmup", type=float, default=0, help="seconds sent but not measured")
    parser.add_argument("--duration", type=float, default=60, help="seconds measured")
    parser.add_argument("--drain-timeout", type=float, default=60)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, help="write summary.json and per-request CSVs here")
    args = parser.parse_args()

    requests, trace = asyncio.run(
        run_load(
            args.url,
            args.rate,
            args.warmup,
            args.duration,
            args.drain_timeout,
            args.min_tokens,
            args.max_tokens,
            args.seed,
        )
    )
    summary = summarize(requests, trace, args.warmup, args.duration)
    for key, value in summary.items():
        print(f"{key:>20}: {value:.3f}")
    if args.out_dir:
        write_outputs(args.out_dir, requests, trace, summary)


if __name__ == "__main__":
    main()
