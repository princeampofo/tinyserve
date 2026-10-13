"""Generate text for one prompt and report decode speed.

uv run python benchmarks/single_request.py "Explain what a KV cache is."
uv run python benchmarks/single_request.py --no-kv-cache --max-tokens 256
"""

import argparse
import platform
import time

import torch
from transformers import AutoTokenizer

from tinyserve.engine.engine import generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.model.weights import DEFAULT_MODEL, load_model

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Write a short story about a lighthouse.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=DTYPES)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", help="always generate max-tokens")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config, weights = load_model(args.model, args.device, DTYPES[args.dtype])
    if args.ignore_eos:
        config.eos_token_ids = []
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False).input_ids
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, seed=0)
    use_kv_cache = not args.no_kv_cache

    # Warmup: the first MPS calls compile kernels and would skew the timing.
    generate(weights, config, prompt_ids, SamplingParams(max_tokens=4, temperature=0))

    start = time.perf_counter()
    output = generate(weights, config, prompt_ids, params, use_kv_cache)
    elapsed = time.perf_counter() - start

    print(tokenizer.decode(output, skip_special_tokens=True))
    print("-" * 60)
    print(f"model={args.model} device={args.device} dtype={args.dtype} kv_cache={use_kv_cache}")
    print(f"machine={platform.machine()} {platform.platform()}")
    print(f"prompt_tokens={len(prompt_ids)} generated_tokens={len(output)} seconds={elapsed:.2f}")
    print(f"tokens_per_second={len(output) / elapsed:.1f}")


if __name__ == "__main__":
    main()
