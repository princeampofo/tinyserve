"""The paged KV cache: tokens scattered across blocks give the same results, sequences
hold only the blocks they need, and running out of blocks is handled by
admission control and preemption without changing anyone's output."""

import pytest
import torch
from conftest import PROMPTS, Model, hf_greedy
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.engine import generate
from tinyserve.engine.kv_cache import KVCache, token_indices
from tinyserve.engine.request import Request
from tinyserve.engine.sampler import SamplingParams
from tinyserve.engine.scheduler import Scheduler
from tinyserve.model.llama import forward


def random_prompts(lengths: list[int], vocab_size: int) -> list[list[int]]:
    torch.manual_seed(4)
    return [torch.randint(0, vocab_size, (n,)).tolist() for n in lengths]


def run_until_done(scheduler: Scheduler) -> None:
    while scheduler.has_work():
        scheduler.step()


def test_token_positions_map_through_the_block_table() -> None:
    write, read = token_indices([[7, 2]], torch.tensor([[20]]), kv_len=18, block_size=16)

    assert write.tolist() == [[2 * 16 + 4]]  # position 20: 2nd block (2), offset 4
    assert read[0, :3].tolist() == [112, 113, 114]  # positions 0-2: block 7
    assert read[0, 16:].tolist() == [32, 33]  # positions 16-17: block 2


def test_tokens_in_scattered_blocks_give_the_same_logits(tiny: Model, device: str) -> None:
    tokens = torch.randint(0, tiny.config.vocab_size, (1, 40), device=device)
    cache = KVCache.allocate(tiny.config, 10, 16, device, torch.float32)
    table = [[5, 2, 9]]  # 48 slots, deliberately out of order

    with torch.no_grad():
        expected = forward(tiny.weights, tiny.config, tokens)
        prefill = forward(tiny.weights, tiny.config, tokens[:, :30], 0, cache, table)
        decoded = [
            forward(tiny.weights, tiny.config, tokens[:, i : i + 1], i, cache, table)
            for i in range(30, 40)
        ]

    torch.testing.assert_close(torch.cat([prefill, *decoded], 1), expected, atol=1e-5, rtol=1e-5)


def test_a_sequence_holds_only_the_blocks_it_has_filled(tiny: Model) -> None:
    scheduler = Scheduler(tiny.weights, tiny.config, 4, 128, num_blocks=32, block_size=16)
    scheduler.add(Request(list(range(20)), SamplingParams(max_tokens=50, temperature=0)))

    scheduler.step()  # prefill 20 tokens + decode 1: 21 tokens cached -> 2 blocks
    assert len(scheduler.running[0].block_table) == 2
    for _ in range(11):
        scheduler.step()  # 32 tokens cached, about to write the 33rd
    assert len(scheduler.running[0].block_table) == 2
    scheduler.step()
    assert len(scheduler.running[0].block_table) == 3


def test_requests_wait_when_there_are_not_enough_free_blocks(tiny: Model) -> None:
    # 8 blocks; each 40-token prompt needs 3 to be admitted, so only 2 fit at once.
    scheduler = Scheduler(tiny.weights, tiny.config, 10, 64, num_blocks=8, block_size=16)
    for prompt in random_prompts([40, 40, 40], tiny.config.vocab_size):
        scheduler.add(Request(prompt, SamplingParams(max_tokens=10, temperature=0)))

    scheduler.step()

    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1


def test_preempted_sequences_finish_with_identical_output(tiny: Model) -> None:
    """6 blocks (96 tokens) for 3 sequences that each grow to 50 tokens (4 blocks):
    they can't all fit, so some are preempted and recomputed along the way."""
    prompts = random_prompts([10, 12, 9], tiny.config.vocab_size)
    scheduler = Scheduler(tiny.weights, tiny.config, 3, 64, num_blocks=6, block_size=16)
    requests = [Request(p, SamplingParams(max_tokens=40, temperature=0)) for p in prompts]
    for request in requests:
        scheduler.add(request)

    run_until_done(scheduler)

    assert scheduler.num_preemptions > 0
    for request, prompt in zip(requests, prompts):
        assert request.output_ids == hf_greedy(tiny, prompt, 40)
    assert scheduler.allocator.num_free == 6


def test_preempted_seeded_sampling_continues_the_same_random_stream(tiny: Model) -> None:
    prompts = random_prompts([10, 12, 9], tiny.config.vocab_size)
    params = [SamplingParams(max_tokens=40, temperature=1.0, seed=i) for i in range(3)]
    alone = [generate(tiny.weights, tiny.config, p, sp) for p, sp in zip(prompts, params)]

    scheduler = Scheduler(tiny.weights, tiny.config, 3, 64, num_blocks=6, block_size=16)
    requests = [Request(p, sp) for p, sp in zip(prompts, params)]
    for request in requests:
        scheduler.add(request)
    run_until_done(scheduler)

    assert scheduler.num_preemptions > 0
    assert [r.output_ids for r in requests] == alone


def test_the_most_recently_admitted_sequence_is_preempted_first(tiny: Model) -> None:
    """Each 15-token prompt fits in 1 block (with room for the first decoded token),
    so all 3 are admitted, using 3 of the 4 blocks. When they reach token 17 they
    each need a 2nd block: the oldest takes the last free one, and the newest is
    preempted so the middle one can grow."""
    prompts = random_prompts([15, 15, 15], tiny.config.vocab_size)
    scheduler = Scheduler(tiny.weights, tiny.config, 3, 48, num_blocks=4, block_size=16)
    requests = [Request(p, SamplingParams(max_tokens=30, temperature=0)) for p in prompts]
    for request in requests:
        scheduler.add(request)

    while scheduler.num_preemptions == 0:
        scheduler.step()

    assert [s.request for s in scheduler.running] == requests[:2]
    assert list(scheduler.waiting) == [requests[2]]
    assert len(requests[2].output_ids) > 0  # it keeps what it generated, to recompute later


def test_paged_fits_more_sequences_than_contiguous_in_the_same_memory(tiny: Model) -> None:
    """The same 512-token budget: contiguous reserves max_model_len=128 tokens per
    sequence (4 sequences); paged gives each 30-token sequence just 2 blocks."""
    prompts = random_prompts([12] * 16, tiny.config.vocab_size)

    def peak_running(scheduler: Scheduler) -> int:
        for prompt in prompts:
            scheduler.add(Request(prompt, SamplingParams(max_tokens=18, temperature=0)))
        run_until_done(scheduler)
        return scheduler.peak_running

    contiguous = Scheduler(tiny.weights, tiny.config, 16, 128, num_blocks=4, block_size=128)
    paged = Scheduler(tiny.weights, tiny.config, 16, 128, num_blocks=32, block_size=16)

    assert peak_running(contiguous) == 4
    assert peak_running(paged) == 16


@pytest.mark.slow
def test_llama_3_2_1b_preempted_sequences_match_huggingface(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompt_ids = [tokenizer(p).input_ids for p in PROMPTS]
    # 7 blocks (112 tokens) for 3 sequences that each grow past 48 tokens (4 blocks).
    scheduler = Scheduler(llama_1b.weights, llama_1b.config, 3, 64, num_blocks=7, block_size=16)
    requests = [Request(p, SamplingParams(max_tokens=40, temperature=0)) for p in prompt_ids]
    for request in requests:
        scheduler.add(request)

    run_until_done(scheduler)

    assert scheduler.num_preemptions > 0
    for request, prompt in zip(requests, prompt_ids):
        assert request.output_ids == hf_greedy(llama_1b, prompt, 40), tokenizer.decode(
            request.output_ids
        )
