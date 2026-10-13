"""Greedy generation produces exactly the same tokens as HuggingFace generate()."""

import pytest
import torch
from conftest import PROMPTS, Model, hf_greedy
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.engine import generate
from tinyserve.engine.sampler import SamplingParams


@pytest.mark.parametrize("use_kv_cache", [True, False])
def test_tiny_model_greedy_output_matches_huggingface(
    tiny: Model, device: str, use_kv_cache: bool
) -> None:
    torch.manual_seed(1)
    prompt_ids = torch.randint(0, tiny.config.vocab_size, (10,)).tolist()
    params = SamplingParams(max_tokens=30, temperature=0)

    expected = hf_greedy(tiny, prompt_ids, 30)
    actual = generate(tiny.weights, tiny.config, prompt_ids, params, use_kv_cache)

    assert actual == expected


def test_generation_stops_at_eos_token(tiny: Model) -> None:
    prompt_ids = [1, 2, 3]
    first = generate(tiny.weights, tiny.config, prompt_ids, SamplingParams(10, temperature=0))
    # Pretend the model's last greedy token is an EOS token: generation should
    # now stop right after that token first appears.
    eos = first[-1]
    tiny.config.eos_token_ids = [eos]

    output = generate(tiny.weights, tiny.config, prompt_ids, SamplingParams(10, temperature=0))

    assert output == first[: first.index(eos) + 1]


@pytest.mark.slow
def test_llama_3_2_1b_greedy_output_matches_huggingface(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase
) -> None:
    for prompt in PROMPTS:
        prompt_ids = tokenizer(prompt).input_ids
        params = SamplingParams(max_tokens=40, temperature=0)

        expected = hf_greedy(llama_1b, prompt_ids, 40)
        actual = generate(llama_1b.weights, llama_1b.config, prompt_ids, params)

        assert actual == expected, f"{tokenizer.decode(actual)!r} != {tokenizer.decode(expected)!r}"
