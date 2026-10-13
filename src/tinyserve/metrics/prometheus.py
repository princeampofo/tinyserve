"""Prometheus metrics: what the server is doing, scraped from /metrics.

Latency is recorded per token, on the engine thread, at the moment a token is
appended to its request: that is when a streaming client can first see it.

    time-to-first-token (TTFT)   arrival -> first token: queueing + prefill
    inter-token latency (ITL)    gap between consecutive tokens of one request:
                                 one decode step, plus any prefill or preemption
                                 that stalled it
    end-to-end latency           arrival -> finished

Histograms keep counts per latency bucket, so Prometheus can compute p50/p99
over any time window (histogram_quantile) without the server storing every sample.
"""

import time

from prometheus_client import Counter, Gauge, Histogram

# Buckets span from fast decode steps (a few ms) to requests stuck in a queue for
# minutes. A percentile can't be reported above the largest bucket, so an
# overloaded server's TTFT needs the long tail: with 120 s as the top bucket, a
# real p99 of 337 s showed up on the dashboard as a flat line at 120 s.
LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 60, 120, 180, 300, 600)  # fmt: skip

TTFT = Histogram(
    "ttft_seconds",
    "Time from request arrival to its first generated token",
    buckets=LATENCY_BUCKETS,
)
INTER_TOKEN_LATENCY = Histogram(
    "inter_token_latency_seconds",
    "Time between consecutive generated tokens of one request",
    buckets=LATENCY_BUCKETS,
)
E2E_LATENCY = Histogram(
    "e2e_latency_seconds", "Time from request arrival until it finished", buckets=LATENCY_BUCKETS
)
GENERATED_TOKENS = Counter("generated_tokens", "Tokens generated across all requests")
REQUESTS = Counter("requests", "Finished requests, by how they finished", ["status"])
WAITING_REQUESTS = Gauge("waiting_requests", "Requests queued, not yet running")
RUNNING_REQUESTS = Gauge("running_requests", "Requests currently being generated")
KV_BLOCKS_USED_RATIO = Gauge("kv_blocks_used_ratio", "Fraction of KV cache blocks in use")
PREEMPTIONS = Counter("preemptions", "Sequences evicted from the KV cache to be recomputed")

# prometheus_client appends "_total" to counter names on export, so these appear
# as generated_tokens_total, requests_total{status=...} and preemptions_total.


def record_token(arrival_time: float, previous_token_time: float | None, now: float) -> None:
    GENERATED_TOKENS.inc()
    if previous_token_time is None:
        TTFT.observe(now - arrival_time)
    else:
        INTER_TOKEN_LATENCY.observe(now - previous_token_time)


def record_finish(arrival_time: float, status: str) -> None:
    E2E_LATENCY.observe(time.monotonic() - arrival_time)
    REQUESTS.labels(status=status).inc()


def record_scheduler_state(waiting: int, running: int, kv_blocks_used_ratio: float) -> None:
    WAITING_REQUESTS.set(waiting)
    RUNNING_REQUESTS.set(running)
    KV_BLOCKS_USED_RATIO.set(kv_blocks_used_ratio)
