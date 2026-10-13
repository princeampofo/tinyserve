# tinyserve

An LLM inference server built from scratch in PyTorch, serving Llama 3.2.

## Setup

```bash
uv sync
hf auth login   # Llama 3.2 is gated: request access at huggingface.co/meta-llama/Llama-3.2-1B-Instruct first
```

## Serve

```bash
uv run python -m tinyserve            # OpenAI-compatible API on localhost:8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(model="llama", messages=[{"role": "user", "content": "Hi!"}])
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
