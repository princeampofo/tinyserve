"""The block allocator hands out every block exactly once and gets every block back."""

import random

import pytest

from tinyserve.engine.block_allocator import BlockAllocator


def test_allocates_every_block_once_then_runs_out() -> None:
    allocator = BlockAllocator(8)
    blocks = [allocator.allocate() for _ in range(8)]

    assert sorted(blocks) == list(range(8))
    with pytest.raises(RuntimeError):
        allocator.allocate()


def test_freeing_a_block_twice_is_an_error() -> None:
    allocator = BlockAllocator(4)
    block = allocator.allocate()
    allocator.free(block)

    with pytest.raises(RuntimeError):
        allocator.free(block)


def test_freeing_a_block_that_was_never_allocated_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        BlockAllocator(4).free(2)


def test_no_leaks_after_10k_random_operations() -> None:
    rng = random.Random(0)
    allocator = BlockAllocator(64)
    owned: set[int] = set()

    for _ in range(10_000):
        if owned and (allocator.num_free == 0 or rng.random() < 0.5):
            block = rng.choice(sorted(owned))
            owned.remove(block)
            allocator.free(block)
        else:
            block = allocator.allocate()
            assert block not in owned  # never handed out twice
            owned.add(block)
        assert allocator.num_free + len(owned) == 64

    for block in owned:
        allocator.free(block)
    assert allocator.num_free == 64
    assert sorted(allocator.allocate() for _ in range(64)) == list(range(64))
