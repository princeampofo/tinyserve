"""Run the benchmark matrix: every server config x every request rate x several runs.

    uv run python benchmarks/run_matrix.py --out benchmarks/results/3b
    uv run python benchmarks/run_matrix.py --configs C D --rates 0.5 1 --runs 1 --cooldown 5 \\
        --warmup 5 --duration 20 --out /tmp/smoke

Configs (all on this machine, same model, same workload):
    A  HuggingFace generate(), one request at a time         (benchmarks/hf_server.py)
    B  tinyserve, static batching
    C  tinyserve, continuous batching, contiguous KV cache
    D  tinyserve, continuous batching, paged KV cache
    E  mlx-lm's server (Apple's MLX framework): the Mac reference ceiling

Protocol, for a fanless laptop that throttles under sustained load:
  - each (config, rate) point is one run: warmup, a fixed measurement window, a
    bounded drain, then a cooldown so the next run starts on a cool chip
  - the config order rotates every pass, and the rate order alternates, so no
    config always runs hot or always runs first
  - every point is repeated `--runs` times; plot.py reports the median and min-max
Results: results.csv (one row per run), metadata.json (hardware, commands,
protocol), and runs/<config>_<rate>_<run>/ with per-request CSVs and traces.
"""

import argparse
import asyncio
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import httpx
from load_gen import run_load, summarize, write_outputs

PORT = 8123
URL = f"http://127.0.0.1:{PORT}"


def server_command(config: str, model: str, kv_cache_gb: float) -> list[str]:
    tinyserve = ["uv", "run", "python", "-m", "tinyserve", "--model", model, "--port", str(PORT)]
    continuous = ["--kv-cache-gb", str(kv_cache_gb), "--max-model-len", "2048", "--max-batch-size", "64"]  # fmt: skip
    return {
        "A": ["uv", "run", "python", "benchmarks/hf_server.py", "--model", model, "--port", str(PORT)],
        "B": [*tinyserve, "--batching", "static", "--max-batch-size", "16"],
        "C": [*tinyserve, *continuous, "--kv-cache", "contiguous"],
        "D": [*tinyserve, *continuous, "--kv-cache", "paged"],
        "E": ["uvx", "--from", "mlx-lm==0.32.0", "mlx_lm.server", "--model", model,
              "--port", str(PORT), "--decode-concurrency", "64", "--log-level", "WARNING"],
    }[config]  # fmt: skip


def start_server(command: list[str], log_path: Path) -> subprocess.Popen[bytes]:
    with open(log_path, "ab") as log:  # the child keeps its own handle to the file
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600  # first start may download or convert weights
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            if httpx.get(f"{URL}/v1/models", timeout=2).status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(1)
    process.kill()
    raise RuntimeError(f"server did not become ready; see {log_path}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def hardware() -> dict[str, str]:
    def sh(*cmd: str) -> str:
        return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()

    return {
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": str(int(sh("sysctl", "-n", "hw.memsize")) // 2**30),
        "model": sh("sysctl", "-n", "hw.model"),
        "os": platform.platform(),
        "power": sh("pmset", "-g", "batt").splitlines()[0],
        "low_power_mode": sh("sh", "-c", "pmset -g | grep lowpowermode").split()[-1],
        "git_commit": sh("git", "rev-parse", "--short", "HEAD"),
        "uncommitted_changes": str(
            bool(sh("git", "status", "--porcelain", "--untracked-files=no"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--kv-cache-gb", type=float, default=2.0)
    parser.add_argument("--warmup", type=float, default=15)
    parser.add_argument("--duration", type=float, default=75)
    parser.add_argument("--drain-timeout", type=float, default=30)
    parser.add_argument("--cooldown", type=float, default=120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "hardware": hardware(),
        "protocol": {k: v for k, v in vars(args).items() if k != "out"},
        "commands": {
            c: " ".join(server_command(c, args.model, args.kv_cache_gb)) for c in args.configs
        },
        "workload": "load_gen.py: chat 'Write a detailed essay about <topic>' + 0-180 filler words "
        "(~50-300 prompt tokens), max_tokens uniform 32-256, ignore_eos where supported, "
        "temperature 0, Poisson arrivals",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = args.out / "metadata.json"
    if metadata_path.exists():  # a second invocation into the same directory adds to it
        previous = json.loads(metadata_path.read_text())
        metadata["commands"] = previous["commands"] | metadata["commands"]
        metadata["protocol"] = previous["protocol"] | {"extra": vars(args) | {"out": str(args.out)}}
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    results_path = args.out / "results.csv"
    write_header = not results_path.exists()

    for run in range(args.runs):
        configs = args.configs[run % len(args.configs) :] + args.configs[: run % len(args.configs)]
        rates = args.rates if run % 2 == 0 else list(reversed(args.rates))
        for config in configs:
            command = server_command(config, args.model, args.kv_cache_gb)
            print(f"[run {run + 1}/{args.runs}] config {config}: starting server", flush=True)
            server = start_server(command, args.out / f"server_{config}.log")
            try:
                for rate in rates:
                    started = time.strftime("%H:%M:%S")
                    requests, trace = asyncio.run(
                        run_load(URL, rate, args.warmup, args.duration, args.drain_timeout,
                                 32, 256, seed=run)
                    )  # fmt: skip
                    summary = summarize(requests, trace, args.warmup, args.duration)
                    write_outputs(
                        args.out / "runs" / f"{config}_{rate}_{run}", requests, trace, summary
                    )
                    row = {
                        "config": config,
                        "rate": rate,
                        "run": run,
                        "started": started,
                        **summary,
                    }
                    with open(results_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        if write_header:
                            writer.writeheader()
                            write_header = False
                        writer.writerow(row)
                    print(
                        f"  {config} rate={rate}: {summary['output_tokens_per_s']:.1f} tok/s, "
                        f"TTFT p50 {summary['ttft_p50_s']:.2f}s, ITL p50 {summary['itl_p50_s'] * 1000:.0f}ms; "
                        f"cooling down {args.cooldown:.0f}s",
                        flush=True,
                    )
                    time.sleep(args.cooldown)
            finally:
                stop_server(server)
    print(f"done: {results_path}")


if __name__ == "__main__":
    main()
