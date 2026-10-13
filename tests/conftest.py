import gc
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
)
from transformers import LlamaConfig as HFLlamaConfig

from tinyserve.model.llama import LlamaConfig, LlamaWeights
from tinyserve.model.weights import DEFAULT_MODEL, load_model, weights_from_state_dict

# Parity tests run in fp32 on every device available. fp16 can flip near-tie
# tokens between devices, so it is reserved for performance runs.
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Continuous batching improves LLM serving throughput because",
]


@pytest.fixture(scope="module", params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@dataclass
class Model:
    """The same weights loaded twice: HuggingFace's model (the reference) and ours."""

    hf: LlamaForCausalLM
    config: LlamaConfig
    weights: LlamaWeights


def tiny_hf_model(device: str = "cpu", vocab_size: int = 512) -> LlamaForCausalLM:
    """A 2-layer random-weight Llama with the same shape rules as Llama 3.2:
    GQA (4 query heads per KV head), llama3 RoPE scaling, tied embeddings."""
    torch.manual_seed(0)
    hf_config = HFLlamaConfig(
        vocab_size=vocab_size,
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


@pytest.fixture
def tiny(device: str) -> Model:
    hf = tiny_hf_model(device)
    config = LlamaConfig.from_hf_dict(hf.config.to_dict())
    return Model(hf, config, weights_from_state_dict(config, hf.state_dict(), device))


@pytest.fixture(scope="module")
def llama_1b(device: str) -> Iterator[Model]:
    """Real Llama 3.2 1B Instruct in fp32, shared by the tests in one module."""
    hf = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float32).eval()
    hf.to(torch.device(device))
    config, weights = load_model(DEFAULT_MODEL, device, torch.float32)
    yield Model(hf, config, weights)

    del hf, weights
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(DEFAULT_MODEL)


def hf_greedy(model: Model, prompt_ids: list[int], max_tokens: int) -> list[int]:
    input_ids = torch.tensor([prompt_ids], device=model.hf.device)
    out = model.hf.generate(  # type: ignore[misc]  # transformers annotates `self` twice
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=model.config.eos_token_ids,
        pad_token_id=model.config.eos_token_ids[0],
    )
    return out[0, len(prompt_ids) :].tolist()
