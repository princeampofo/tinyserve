"""Our hand-written Llama forward pass produces the same logits as HuggingFace."""

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

from tinyserve.model.llama import LlamaConfig, forward
from tinyserve.model.weights import DEFAULT_MODEL, load_model, weights_from_state_dict


def tiny_hf_model(device: str = "cpu") -> LlamaForCausalLM:
    """A 2-layer random-weight Llama with the same shape rules as Llama 3.2:
    GQA (4 query heads per KV head), llama3 RoPE scaling, tied embeddings."""
    torch.manual_seed(0)
    hf_config = HFLlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=16384,
        rms_norm_eps=1e-5,
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        tie_word_embeddings=True,
    )
    with torch.device(device):
        return LlamaForCausalLM(hf_config).eval()


def test_config_parses_llama3_rope_scaling() -> None:
    hf = tiny_hf_model()
    config = LlamaConfig.from_hf_dict(hf.config.to_dict())
    assert config.head_dim == 16
    assert config.num_kv_heads == 2
    assert config.rope_theta == 500000.0
    assert config.rope_scaling is not None and config.rope_scaling["factor"] == 32.0


def test_tiny_model_logits_match_huggingface(device: str) -> None:
    hf = tiny_hf_model(device)
    config = LlamaConfig.from_hf_dict(hf.config.to_dict())
    weights = weights_from_state_dict(config, hf.state_dict(), device)

    input_ids = torch.randint(0, config.vocab_size, (3, 40), device=device)
    with torch.no_grad():
        expected = hf(input_ids).logits
        actual = forward(weights, config, input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_tiny_model_logits_match_huggingface_at_large_positions(device: str) -> None:
    """Positions past the original 8K context exercise the llama3 RoPE scaling."""
    hf = tiny_hf_model(device)
    config = LlamaConfig.from_hf_dict(hf.config.to_dict())
    weights = weights_from_state_dict(config, hf.state_dict(), device)

    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    positions = torch.arange(12000, 12016, device=device)[None]
    with torch.no_grad():
        expected = hf(input_ids, position_ids=positions).logits
        actual = forward(weights, config, input_ids, positions)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Continuous batching improves LLM serving throughput because",
]


@pytest.mark.slow
def test_llama_3_2_1b_logits_match_huggingface(device: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    hf = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float32).eval()
    hf.to(torch.device(device))
    config, weights = load_model(DEFAULT_MODEL, device, torch.float32)

    try:
        for prompt in PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                expected = hf(input_ids).logits
                actual = forward(weights, config, input_ids)

            # fp32 with different op ordering: logits (range ~±30) agree to ~1e-4.
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    finally:
        del hf, weights
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
