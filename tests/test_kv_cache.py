"""Running with a KV cache gives the same logits as recomputing the whole sequence."""

import torch
from conftest import Model

from tinyserve.engine.kv_cache import KVCache
from tinyserve.model.llama import forward


def test_decoding_one_token_at_a_time_matches_full_forward(tiny: Model, device: str) -> None:
    tokens = torch.randint(0, tiny.config.vocab_size, (1, 24), device=device)
    cache = KVCache.allocate(tiny.config, 1, 32, device, torch.float32)

    with torch.no_grad():
        expected = forward(tiny.weights, tiny.config, tokens)
        prefill = forward(tiny.weights, tiny.config, tokens[:, :16], 0, cache)
        decoded = [
            forward(tiny.weights, tiny.config, tokens[:, i : i + 1], i, cache)
            for i in range(16, 24)
        ]

    actual = torch.cat([prefill, *decoded], dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_prefilling_in_two_chunks_matches_one_chunk(tiny: Model, device: str) -> None:
    """The second chunk's queries must see the first chunk's cached keys, but not
    later tokens of their own chunk (causal masking aligned to the end)."""
    tokens = torch.randint(0, tiny.config.vocab_size, (1, 20), device=device)
    cache = KVCache.allocate(tiny.config, 1, 20, device, torch.float32)

    with torch.no_grad():
        expected = forward(tiny.weights, tiny.config, tokens)
        first = forward(tiny.weights, tiny.config, tokens[:, :12], 0, cache)
        second = forward(tiny.weights, tiny.config, tokens[:, 12:], 12, cache)

    torch.testing.assert_close(torch.cat([first, second], dim=1), expected, atol=1e-5, rtol=1e-5)
