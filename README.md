# tinyserve

An LLM inference server built from scratch in PyTorch, serving Llama 3.2.

**Status:** the Llama forward pass is done and matches HuggingFace logits (bit-identical on CPU, within 1e-5 on Apple MPS). Serving features come next.

## Setup

```bash
uv sync
```

## Tests

```bash
uv run pytest            # fast: tiny random-weight Llama vs HuggingFace
uv run pytest -m slow    # real Llama 3.2 1B vs HuggingFace (downloads ~2.5GB)
```
