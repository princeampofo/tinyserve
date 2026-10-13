"""Plots and a summary table from a run_matrix.py results directory.

    uv run python benchmarks/plot.py benchmarks/results/3b
    uv run python benchmarks/plot.py benchmarks/results/3b --trace benchmarks/results/thermal

Every point is the median of its runs, with a bar from the min to the max run.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "A": "A  HF generate(), one at a time",
    "B": "B  static batching",
    "C": "C  continuous, contiguous KV",
    "D": "D  continuous, paged KV",
    "E": "E  mlx-lm server",
}
# Categorical slots in a fixed order, validated for colorblind separation; the
# lighter ones are below 3:1 contrast, so every line also gets a direct label
# and its own marker shape.
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D", "E": "v"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MEAN_OUTPUT_TOKENS = 144  # max_tokens is uniform in 32..256


def style(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "request rate (req/s)") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load(results_dir: Path) -> dict[tuple[str, float], list[dict[str, float]]]:
    runs: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    with open(results_dir / "results.csv") as f:
        for row in csv.DictReader(f):
            values = {k: float(v) for k, v in row.items() if k not in ("config", "started")}
            runs[(row["config"], values["rate"])].append(values)
    return runs


def stat(runs: list[dict[str, float]], key: str) -> tuple[float, float, float]:
    values = [r[key] for r in runs if not math.isnan(r[key])]
    if not values:
        return math.nan, math.nan, math.nan
    return statistics.median(values), min(values), max(values)


def series(
    runs: dict[tuple[str, float], list[dict[str, float]]], config: str, key: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    rates = sorted(rate for c, rate in runs if c == config)
    med, lo, hi = zip(*(stat(runs[(config, r)], key) for r in rates)) if rates else ((), (), ())
    return rates, list(med), list(lo), list(hi)


def plot_line(
    ax: plt.Axes,
    config: str,
    x: list[float],
    med: list[float],
    lo: list[float],
    hi: list[float],
    label_end: bool = True,
) -> None:
    color = COLORS[config]
    ax.errorbar(
        x, med, yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]],
        color=color, marker=MARKERS[config], markersize=6, linewidth=2, capsize=3,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[config],
    )  # fmt: skip
    if (
        not label_end
    ):  # where end labels would collide, the shared legend + marker shape identify it
        return
    ax.annotate(
        config, (x[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=9, color=INK, fontweight="bold",
    )  # fmt: skip


def saturation_rate(rates: list[float], throughput: list[float]) -> float | None:
    """The first rate where the server delivers less than 90% of the offered tokens/s."""
    for rate, tokens in zip(rates, throughput):
        if tokens < 0.9 * rate * MEAN_OUTPUT_TOKENS:
            return rate
    return None


def plot_throughput(
    runs: dict[tuple[str, float], list[dict[str, float]]], configs: list[str], out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style(ax, "Output throughput vs. offered load", "output tokens/s")
    all_rates = sorted({rate for _, rate in runs})
    offered = [r * MEAN_OUTPUT_TOKENS for r in all_rates]
    ax.plot(
        all_rates,
        offered,
        color=MUTED,
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        label="offered load (every token delivered)",
    )
    for config in configs:
        rates, med, lo, hi = series(runs, config, "output_tokens_per_s")
        plot_line(ax, config, rates, med, lo, hi)
        sat = saturation_rate(rates, med)
        if sat is not None:
            y = med[rates.index(sat)]
            ax.plot(
                [sat],
                [y],
                marker="o",
                markersize=14,
                markerfacecolor="none",
                markeredgecolor=COLORS[config],
                markeredgewidth=1.5,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_rates, [f"{r:g}" for r in all_rates])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.text(0.99, 0.02, "circle = saturation: <90% of offered tokens delivered", transform=ax.transAxes,
            ha="right", fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_latency(
    runs: dict[tuple[str, float], list[dict[str, float]]], configs: list[str], out: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    panels = [
        ("ttft_p50_s", "Time to first token, p50", "seconds (log)"),
        ("ttft_p99_s", "Time to first token, p99", "seconds (log)"),
        ("itl_p50_s", "Inter-token latency, p50", "milliseconds"),
        ("itl_p99_s", "Inter-token latency, p99", "milliseconds"),
    ]
    all_rates = sorted({rate for _, rate in runs})
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        style(ax, title, ylabel)
        scale = 1000 if key.startswith("itl") else 1
        for config in configs:
            rates, med, lo, hi = series(runs, config, key)
            plot_line(
                ax,
                config,
                rates,
                [m * scale for m in med],
                [v * scale for v in lo],
                [v * scale for v in hi],
                label_end=False,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_rates, [f"{r:g}" for r in all_rates])
        if key.startswith("ttft"):
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_kv_trace(results_dir: Path, rate: float, run: int, out: Path) -> None:
    """Contiguous (C) vs. paged (D) at one rate: how many sequences ran, and how full the cache was."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, facecolor=SURFACE)
    style(top, f"Sequences running at {rate:g} req/s", "running", xlabel="")
    style(bottom, "KV cache blocks in use", "fraction of the pool", xlabel="seconds into the run")
    for config in ("C", "D"):
        path = results_dir / "runs" / f"{config}_{rate}_{run}" / "server_trace.csv"
        if not path.exists():
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        t = [float(r["t"]) for r in rows]
        for ax, key in ((top, "running_requests"), (bottom, "kv_blocks_used_ratio")):
            y = [float(r[key]) for r in rows]
            ax.plot(t, y, color=COLORS[config], linewidth=2, label=NAMES[config])
            ax.annotate(config, (t[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK, fontweight="bold")  # fmt: skip
    bottom.set_ylim(0, 1.05)
    top.set_ylim(bottom=0)
    top.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_thermal(trace_dir: Path, out: Path) -> None:
    with open(trace_dir / "tokens_per_second.csv") as f:
        rows = [(int(r["second"]), int(r["tokens"])) for r in csv.DictReader(f)]
    window = 15
    t = [s for s, _ in rows][window:]
    smoothed = [
        statistics.mean(tok for _, tok in rows[i - window : i]) for i in range(window, len(rows))
    ]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(
        ax,
        "One long run: throughput over time (15 s rolling mean)",
        "output tokens/s",
        "seconds into the run",
    )
    ax.plot([x / 60 for x in t], smoothed, color=COLORS["D"], linewidth=2)
    ax.set_xlabel("minutes into the run", color=MUTED, fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fmt(
    rs: list[dict[str, float]], key: str, scale: float = 1, unit: str = "s", digits: int = 2
) -> str:
    med = stat(rs, key)[0]
    return "–" if math.isnan(med) else f"{med * scale:.{digits}f} {unit}".strip()


def summary_table(runs: dict[tuple[str, float], list[dict[str, float]]], configs: list[str]) -> str:
    """Markdown: one row per (config, rate) with medians; '>=' marks TTFTs that hit the drain cutoff."""
    lines = [
        "| Config | Rate (req/s) | Output tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | Peak running | Preemptions |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for config in configs:
        for rate in sorted(r for c, r in runs if c == config):
            rs = runs[(config, rate)]
            censored = any(r["ttft_censored"] > 0 for r in rs)
            ge = "≥ " if censored else ""

            lines.append(
                f"| {config} | {rate:g} | {fmt(rs, 'output_tokens_per_s', unit='', digits=1)} "
                f"| {ge}{fmt(rs, 'ttft_p50_s')} | {ge}{fmt(rs, 'ttft_p99_s')} "
                f"| {fmt(rs, 'itl_p50_s', 1000, 'ms', 0)} | {fmt(rs, 'itl_p99_s', 1000, 'ms', 0)} "
                f"| {fmt(rs, 'peak_running', unit='', digits=0)} | {fmt(rs, 'preemptions', unit='', digits=0)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--trace", type=Path, help="a load_gen --out-dir from one long run")
    args = parser.parse_args()

    runs = load(args.results_dir)
    configs = [c for c in NAMES if any(c == k for k, _ in runs)]
    plot_throughput(runs, configs, args.results_dir / "throughput_vs_rate.png")
    plot_latency(runs, configs, args.results_dir / "latency_vs_rate.png")
    top_rate = max(rate for _, rate in runs)
    plot_kv_trace(
        args.results_dir, top_rate, 0, args.results_dir / "kv_cache_contiguous_vs_paged.png"
    )
    if args.trace:
        plot_thermal(args.trace, args.results_dir / "thermal_trace.png")
    table = summary_table(runs, configs)
    (args.results_dir / "summary.md").write_text(table)
    metadata = json.loads((args.results_dir / "metadata.json").read_text())
    print(table)
    print(json.dumps(metadata["hardware"], indent=2))


if __name__ == "__main__":
    main()
