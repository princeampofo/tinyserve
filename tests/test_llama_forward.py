"""Our hand-written Llama forward pass produces the same logits as HuggingFace."""

import pytest
import torch
from conftest import PROMPTS, Model, tiny_hf_model
from transformers import PreTrainedTokenizerBase

from tinyserve.model.llama import LlamaConfig, forward


def test_config_parses_llama3_rope_scaling() -> None:
    config = LlamaConfig.from_hf_dict(tiny_hf_model().config.to_dict())
    assert config.head_dim == 16
    assert config.num_kv_heads == 2
    assert config.rope_theta == 500000.0
    assert config.rope_scaling is not None and config.rope_scaling["factor"] == 32.0


def test_tiny_model_logits_match_huggingface(tiny: Model, device: str) -> None:
    input_ids = torch.randint(0, tiny.config.vocab_size, (3, 40), device=device)
    with torch.no_grad():
        expected = tiny.hf(input_ids).logits
        actual = forward(tiny.weights, tiny.config, input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_tiny_model_logits_match_huggingface_at_large_positions(tiny: Model, device: str) -> None:
    """Positions past the original 8K context exercise the llama3 RoPE scaling."""
    input_ids = torch.randint(0, tiny.config.vocab_size, (1, 16), device=device)
    positions = torch.arange(12000, 12016, device=device)[None]
    with torch.no_grad():
        expected = tiny.hf(input_ids, position_ids=positions).logits
        actual = forward(tiny.weights, tiny.config, input_ids, start_pos=12000)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_llama_3_2_1b_logits_match_huggingface(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase, device: str
) -> None:
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            expected = llama_1b.hf(input_ids).logits
            actual = forward(llama_1b.weights, llama_1b.config, input_ids)

        # fp32 with different op ordering: logits (range ~±30) agree to ~1e-4.
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
