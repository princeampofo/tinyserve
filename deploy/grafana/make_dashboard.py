"""Generates dashboards/tinyserve.json. Edit this file, not the JSON:

uv run python deploy/grafana/make_dashboard.py
"""

import json
from pathlib import Path
from typing import Any

RATE = "[30s]"  # window for rates and percentiles: smooth, but still reacts within seconds


def quantiles(metric: str) -> list[dict[str, str]]:
    return [
        {
            "expr": f"histogram_quantile({q}, sum by (le) (rate({metric}_bucket{RATE})))",
            "legendFormat": f"p{int(q * 100)}",
        }
        for q in (0.5, 0.99)
    ]


def panel(
    title: str,
    targets: list[dict[str, str]],
    unit: str,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    kind: str = "timeseries",
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": kind,
        "title": title,
        "description": description,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit, "min": 0}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
        "targets": [{"refId": chr(65 + i), **t} for i, t in enumerate(targets)],
    }


def expr(e: str, legend: str = "") -> dict[str, str]:
    return {"expr": e, "legendFormat": legend}


THROUGHPUT = f"sum(rate(generated_tokens_total{RATE}))"

panels = [
    # Top row: the numbers to glance at.
    panel("Output tokens/s", [expr(THROUGHPUT)], "short", 0, 0, 6, 4, "stat"),
    panel("Running", [expr("running_requests")], "short", 6, 0, 6, 4, "stat"),
    panel("Waiting", [expr("waiting_requests")], "short", 12, 0, 6, 4, "stat"),
    panel("KV cache used", [expr("kv_blocks_used_ratio")], "percentunit", 18, 0, 6, 4, "stat"),
    # Throughput and load.
    panel(
        "Output throughput",
        [expr(THROUGHPUT, "tokens/s")],
        "short",
        0,
        4,
        description="Generated tokens per second, across all requests.",
    ),
    panel(
        "Requests in the engine",
        [expr("running_requests", "running"), expr("waiting_requests", "waiting")],
        "short",
        12,
        4,
        description="Running: being decoded. Waiting: queued for admission.",
    ),
    # Latency.
    panel(
        "Time to first token",
        quantiles("ttft_seconds"),
        "s",
        0,
        12,
        description="Arrival to first token: queueing plus prefill.",
    ),
    panel(
        "Inter-token latency",
        quantiles("inter_token_latency_seconds"),
        "s",
        12,
        12,
        description="Gap between tokens of one request: a decode step, plus any prefill or preemption that stalled it.",
    ),
    panel("End-to-end latency", quantiles("e2e_latency_seconds"), "s", 0, 20),
    panel(
        "Finished requests/s",
        [expr(f"sum by (status) (rate(requests_total{RATE}))", "{{status}}")],
        "reqps",
        12,
        20,
        description="stop: EOS or stop string. length: hit max_tokens. abort: client left or timed out.",
    ),
    # Memory.
    panel("KV cache blocks used", [expr("kv_blocks_used_ratio", "used")], "percentunit", 0, 28),
    panel(
        "Preemptions/s",
        [expr(f"rate(preemptions_total{RATE})", "preemptions")],
        "short",
        12,
        28,
        description="Sequences evicted from the KV cache to be recomputed later.",
    ),
]
panels[10]["fieldConfig"]["defaults"]["max"] = 1  # the KV cache can't be more than 100% used
for stat in panels[:4]:
    # One neutral color: Grafana's default thresholds would paint throughput red above 80.
    stat["fieldConfig"]["defaults"]["color"] = {"mode": "fixed", "fixedColor": "blue"}

dashboard = {
    "uid": "tinyserve",
    "title": "tinyserve",
    "tags": ["llm", "inference"],
    "timezone": "browser",
    "schemaVersion": 39,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "panels": panels,
}

path = Path(__file__).parent / "dashboards" / "tinyserve.json"
path.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {path}")
