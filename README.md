# tinyserve

An LLM inference server built from scratch in PyTorch, serving Llama 3.2.

## Setup

```bash
uv sync
hf auth login   # Llama 3.2 is gated: request access at huggingface.co/meta-llama/Llama-3.2-1B-Instruct first
```

## Generate

```bash
uv run python benchmarks/single_request.py "Explain what a KV cache is."
```

## Tests

```bash
uv run pytest            # fast: tiny random-weight Llama vs HuggingFace
uv run pytest -m slow    # real Llama 3.2 1B vs HuggingFace (downloads ~2.5GB)
```
