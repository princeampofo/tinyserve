"""Continuous batching gives every request exactly the output it gets alone,
however requests join and leave the running batch."""

import pytest
import torch
from conftest import PROMPTS, Model, hf_greedy
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.engine import generate
from tinyserve.engine.request import Request
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Scheduler


def random_prompts(lengths: list[int], vocab_size: int) -> list[list[int]]:
    torch.manual_seed(3)
    return [torch.randint(0, vocab_size, (n,)).tolist() for n in lengths]


def run_until_done(scheduler: Scheduler) -> list[Request]:
    finished = []
    while scheduler.has_work():
        finished += scheduler.step()
    return finished


def test_more_requests_than_slots_all_match_huggingface(tiny: Model) -> None:
    """With 2 slots and 5 requests, later requests take over slots as earlier ones finish."""
    prompts = random_prompts([5, 17, 9, 30, 3], tiny.config.vocab_size)
    max_tokens = [20, 4, 11, 7, 15]
    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=2, max_model_len=64)
    requests = [
        Request(p, SamplingParams(max_tokens=m, temperature=0)) for p, m in zip(prompts, max_tokens)
    ]
    for request in requests:
        scheduler.add(request)

    run_until_done(scheduler)

    for request, prompt, m in zip(requests, prompts, max_tokens):
        assert request.output_ids == hf_greedy(tiny, prompt, m)


def test_a_request_joining_mid_flight_gets_the_same_output(tiny: Model) -> None:
    first, second = random_prompts([8, 12], tiny.config.vocab_size)
    greedy = SamplingParams(max_tokens=20, temperature=0)
    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=4, max_model_len=64)
    a, b = Request(first, greedy), Request(second, greedy)

    scheduler.add(a)
    for _ in range(6):
        scheduler.step()
    scheduler.add(b)  # joins while `a` is 7 tokens in
    run_until_done(scheduler)

    assert a.output_ids == hf_greedy(tiny, first, 20)
    assert b.output_ids == hf_greedy(tiny, second, 20)


def test_a_finished_request_leaves_at_the_step_it_finishes(tiny: Model) -> None:
    short, long = random_prompts([4, 4], tiny.config.vocab_size)
    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=2, max_model_len=64)
    scheduler.add(Request(short, SamplingParams(max_tokens=3, temperature=0)))
    scheduler.add(Request(long, SamplingParams(max_tokens=30, temperature=0)))

    finished_per_step = [len(scheduler.step()) for _ in range(3)]

    # Step 1 prefills (token 1) and decodes (token 2); step 2 produces token 3.
    assert finished_per_step == [0, 1, 0]
    assert len(scheduler.running) == 1 and len(scheduler.free_slots) == 1


def test_seeded_sampling_does_not_depend_on_the_rest_of_the_batch(tiny: Model) -> None:
    prompt, *others = random_prompts([12, 4, 25], tiny.config.vocab_size)
    params = SamplingParams(max_tokens=20, temperature=1.0, top_p=0.9, seed=42)
    alone = generate(tiny.weights, tiny.config, prompt, params)

    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=3, max_model_len=64)
    request = Request(prompt, params)
    scheduler.add(Request(others[0], SamplingParams(max_tokens=30, temperature=0.7)))
    scheduler.step()
    scheduler.add(request)
    scheduler.add(Request(others[1], SamplingParams(max_tokens=5, temperature=0)))
    run_until_done(scheduler)

    assert request.output_ids == alone


def test_all_slots_are_free_after_every_request_finishes(tiny: Model) -> None:
    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=3, max_model_len=64)
    for prompt in random_prompts([5, 6, 7, 8, 9, 10, 11], tiny.config.vocab_size):
        scheduler.add(Request(prompt, SamplingParams(max_tokens=6, temperature=0)))

    finished = run_until_done(scheduler)

    assert len(finished) == 7
    assert sorted(scheduler.free_slots) == [0, 1, 2]


def test_an_aborted_request_frees_its_slot(tiny: Model) -> None:
    prompts = random_prompts([5, 5], tiny.config.vocab_size)
    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=1, max_model_len=64)
    running = Request(prompts[0], SamplingParams(max_tokens=50, temperature=0))
    waiting = Request(prompts[1], SamplingParams(max_tokens=5, temperature=0))
    scheduler.add(running)
    scheduler.add(waiting)
    scheduler.step()

    running.finish_reason = "abort"  # what the server does when a request times out
    assert scheduler.step() == [running]
    run_until_done(scheduler)

    assert len(running.output_ids) < 50
    assert waiting.output_ids == hf_greedy(tiny, prompts[1], 5)
    assert scheduler.free_slots == [0]


def test_a_request_too_long_for_a_slot_is_rejected(tiny: Model) -> None:
    scheduler = Scheduler(tiny.weights, tiny.config, max_num_seqs=1, max_model_len=16)
    with pytest.raises(ValueError):
        scheduler.add(Request(list(range(10)), SamplingParams(max_tokens=10)))


@pytest.mark.slow
def test_llama_3_2_1b_continuous_batching_matches_huggingface(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompt_ids = [tokenizer(p).input_ids for p in PROMPTS]
    max_tokens = [40, 15, 30]
    scheduler = Scheduler(llama_1b.weights, llama_1b.config, max_num_seqs=2, max_model_len=128)
    requests = [
        Request(p, SamplingParams(max_tokens=m, temperature=0))
        for p, m in zip(prompt_ids, max_tokens)
    ]
    for request in requests:
        scheduler.add(request)

    run_until_done(scheduler)

    for request, prompt, m in zip(requests, prompt_ids, max_tokens):
        assert request.output_ids == hf_greedy(llama_1b, prompt, m), tokenizer.decode(
            request.output_ids
        )
