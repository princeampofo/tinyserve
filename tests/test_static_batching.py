"""Running requests as a static batch gives each one exactly the output it gets alone."""

import pytest
import torch
from conftest import PROMPTS, Model, hf_greedy
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.engine import generate, run_static_batch
from tinyserve.engine.request import Request
from tinyserve.engine.sampler import SamplingParams


def random_prompts(lengths: list[int], vocab_size: int) -> list[list[int]]:
    torch.manual_seed(2)
    return [torch.randint(0, vocab_size, (n,)).tolist() for n in lengths]


def test_greedy_batch_matches_huggingface_for_every_prompt(tiny: Model) -> None:
    """Prompts of different lengths and different max_tokens share one batch."""
    prompts = random_prompts([5, 17, 9, 30], tiny.config.vocab_size)
    max_tokens = [20, 8, 25, 12]
    requests = [
        Request(p, SamplingParams(max_tokens=m, temperature=0)) for p, m in zip(prompts, max_tokens)
    ]

    run_static_batch(tiny.weights, tiny.config, requests)

    for request, prompt, m in zip(requests, prompts, max_tokens):
        assert request.output_ids == hf_greedy(tiny, prompt, m)
        assert request.finish_reason == "length"


def test_seeded_sampling_does_not_depend_on_the_rest_of_the_batch(tiny: Model) -> None:
    prompt = random_prompts([12], tiny.config.vocab_size)[0]
    params = SamplingParams(max_tokens=20, temperature=1.0, top_p=0.9, seed=42)
    alone = generate(tiny.weights, tiny.config, prompt, params)

    others = random_prompts([4, 25], tiny.config.vocab_size)
    requests = [
        Request(others[0], SamplingParams(max_tokens=30, temperature=0.7)),
        Request(prompt, params),
        Request(others[1], SamplingParams(max_tokens=5, temperature=0)),
    ]
    run_static_batch(tiny.weights, tiny.config, requests)

    assert requests[1].output_ids == alone


def test_a_request_hitting_eos_stops_while_others_continue(tiny: Model) -> None:
    prompts = random_prompts([6, 6], tiny.config.vocab_size)
    greedy = SamplingParams(max_tokens=15, temperature=0)
    first_alone = generate(tiny.weights, tiny.config, prompts[0], greedy)
    eos = first_alone[4]
    tiny.config.eos_token_ids = [eos]
    expected_first = first_alone[: first_alone.index(eos) + 1]

    requests = [Request(p, greedy) for p in prompts]
    run_static_batch(tiny.weights, tiny.config, requests)

    assert requests[0].output_ids == expected_first
    assert requests[0].finish_reason == "stop"
    assert requests[1].output_ids == generate(tiny.weights, tiny.config, prompts[1], greedy)


@pytest.mark.slow
def test_llama_3_2_1b_greedy_batch_matches_huggingface(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompt_ids = [tokenizer(p).input_ids for p in PROMPTS]
    requests = [Request(p, SamplingParams(max_tokens=40, temperature=0)) for p in prompt_ids]

    run_static_batch(llama_1b.weights, llama_1b.config, requests)

    for request, prompt in zip(requests, prompt_ids):
        assert request.output_ids == hf_greedy(llama_1b, prompt, 40), tokenizer.decode(
            request.output_ids
        )
