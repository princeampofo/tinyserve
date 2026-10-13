"""Continuous batching gives every request exactly the output it gets alone,
however requests join and leave the running batch, with either cache layout."""

import pytest
import torch
from conftest import PROMPTS, Model, hf_greedy
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.engine import generate
from tinyserve.engine.request import Request
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Scheduler


@pytest.fixture(params=["paged", "contiguous"])
def kv_cache(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def make_scheduler(model: Model, kv_cache: str, max_num_seqs: int, max_model_len: int) -> Scheduler:
    """A scheduler with plenty of memory: these tests are about batching, not running out."""
    if kv_cache == "contiguous":
        num_blocks, block_size = max_num_seqs, max_model_len
    else:
        num_blocks, block_size = max_num_seqs * max_model_len // 16, 16
    return Scheduler(
        model.weights, model.config, max_num_seqs, max_model_len, num_blocks, block_size
    )


def random_prompts(lengths: list[int], vocab_size: int) -> list[list[int]]:
    torch.manual_seed(3)
    return [torch.randint(0, vocab_size, (n,)).tolist() for n in lengths]


def run_until_done(scheduler: Scheduler) -> list[Request]:
    finished = []
    while scheduler.has_work():
        finished += scheduler.step()
    return finished


def test_more_requests_than_fit_at_once_all_match_huggingface(tiny: Model, kv_cache: str) -> None:
    """With room for 2 sequences and 5 requests, later requests start as earlier ones finish."""
    prompts = random_prompts([5, 17, 9, 30, 3], tiny.config.vocab_size)
    max_tokens = [20, 4, 11, 7, 15]
    scheduler = make_scheduler(tiny, kv_cache, 2, 64)
    requests = [
        Request(p, SamplingParams(max_tokens=m, temperature=0)) for p, m in zip(prompts, max_tokens)
    ]
    for request in requests:
        scheduler.add(request)

    run_until_done(scheduler)

    for request, prompt, m in zip(requests, prompts, max_tokens):
        assert request.output_ids == hf_greedy(tiny, prompt, m)


def test_a_request_joining_mid_flight_gets_the_same_output(tiny: Model, kv_cache: str) -> None:
    first, second = random_prompts([8, 12], tiny.config.vocab_size)
    greedy = SamplingParams(max_tokens=20, temperature=0)
    scheduler = make_scheduler(tiny, kv_cache, 4, 64)
    a, b = Request(first, greedy), Request(second, greedy)

    scheduler.add(a)
    for _ in range(6):
        scheduler.step()
    scheduler.add(b)  # joins while `a` is 7 tokens in
    run_until_done(scheduler)

    assert a.output_ids == hf_greedy(tiny, first, 20)
    assert b.output_ids == hf_greedy(tiny, second, 20)


def test_a_finished_request_leaves_at_the_step_it_finishes(tiny: Model, kv_cache: str) -> None:
    short, long = random_prompts([4, 4], tiny.config.vocab_size)
    scheduler = make_scheduler(tiny, kv_cache, 2, 64)
    scheduler.add(Request(short, SamplingParams(max_tokens=3, temperature=0)))
    scheduler.add(Request(long, SamplingParams(max_tokens=30, temperature=0)))

    finished_per_step = [len(scheduler.step()) for _ in range(3)]

    # Step 1 prefills (token 1) and decodes (token 2); step 2 produces token 3.
    assert finished_per_step == [0, 1, 0]
    assert len(scheduler.running) == 1


def test_seeded_sampling_does_not_depend_on_the_rest_of_the_batch(
    tiny: Model, kv_cache: str
) -> None:
    prompt, *others = random_prompts([12, 4, 25], tiny.config.vocab_size)
    params = SamplingParams(max_tokens=20, temperature=1.0, top_p=0.9, seed=42)
    alone = generate(tiny.weights, tiny.config, prompt, params)

    scheduler = make_scheduler(tiny, kv_cache, 3, 64)
    request = Request(prompt, params)
    scheduler.add(Request(others[0], SamplingParams(max_tokens=30, temperature=0.7)))
    scheduler.step()
    scheduler.add(request)
    scheduler.add(Request(others[1], SamplingParams(max_tokens=5, temperature=0)))
    run_until_done(scheduler)

    assert request.output_ids == alone


def test_every_block_is_free_after_every_request_finishes(tiny: Model, kv_cache: str) -> None:
    scheduler = make_scheduler(tiny, kv_cache, 3, 64)
    for prompt in random_prompts([5, 6, 7, 8, 9, 10, 11], tiny.config.vocab_size):
        scheduler.add(Request(prompt, SamplingParams(max_tokens=6, temperature=0)))

    finished = run_until_done(scheduler)

    assert len(finished) == 7
    assert scheduler.allocator.num_free == scheduler.allocator.num_blocks


def test_an_aborted_request_frees_its_blocks(tiny: Model, kv_cache: str) -> None:
    prompts = random_prompts([5, 5], tiny.config.vocab_size)
    scheduler = make_scheduler(tiny, kv_cache, 1, 64)
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
    assert scheduler.allocator.num_free == scheduler.allocator.num_blocks


def test_a_request_longer_than_max_model_len_is_rejected(tiny: Model, kv_cache: str) -> None:
    scheduler = make_scheduler(tiny, kv_cache, 1, 16)
    with pytest.raises(ValueError):
        scheduler.add(Request(list(range(10)), SamplingParams(max_tokens=10)))


@pytest.mark.slow
def test_llama_3_2_1b_continuous_batching_matches_huggingface(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase, kv_cache: str
) -> None:
    prompt_ids = [tokenizer(p).input_ids for p in PROMPTS]
    max_tokens = [40, 15, 30]
    scheduler = make_scheduler(llama_1b, kv_cache, 2, 128)
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
