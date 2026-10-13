"""The sampler picks tokens the way each request's settings say it should."""

import torch

from tinyserve.engine.sampler import SamplingParams, make_generator, sample

# A 6-token vocabulary with clearly ranked probabilities: 0.4, 0.3, 0.15, 0.1, 0.04, 0.01.
LOGITS = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.04, 0.01]).log()


def draw(params: SamplingParams, n: int, seed: int = 0) -> list[int]:
    """Sample n tokens from LOGITS with one generator, as n decode steps would."""
    g = make_generator(seed, "cpu")
    return [int(sample(LOGITS[None], [params], [g]).item()) for _ in range(n)]


def test_temperature_zero_is_greedy() -> None:
    assert set(draw(SamplingParams(temperature=0), 50)) == {0}


def test_same_seed_gives_same_tokens() -> None:
    params = SamplingParams(temperature=1.0)
    assert draw(params, 50, seed=7) == draw(params, 50, seed=7)


def test_different_seeds_give_different_tokens() -> None:
    params = SamplingParams(temperature=1.0)
    assert draw(params, 50, seed=1) != draw(params, 50, seed=2)


def test_top_k_only_samples_the_k_most_likely_tokens() -> None:
    assert set(draw(SamplingParams(temperature=1.0, top_k=2), 500)) == {0, 1}


def test_top_p_only_samples_the_smallest_set_covering_p() -> None:
    # 0.4 + 0.3 = 0.7 < 0.8, so token 2 is needed too; 0.4 + 0.3 + 0.15 = 0.85 covers it.
    assert set(draw(SamplingParams(temperature=1.0, top_p=0.8), 500)) == {0, 1, 2}


def test_tiny_top_p_still_keeps_the_most_likely_token() -> None:
    assert set(draw(SamplingParams(temperature=1.0, top_p=0.01), 50)) == {0}


def test_sampled_frequencies_follow_the_probabilities() -> None:
    tokens = torch.tensor(draw(SamplingParams(temperature=1.0), 4000))
    freqs = torch.bincount(tokens, minlength=6).float() / len(tokens)
    torch.testing.assert_close(freqs, LOGITS.exp(), atol=0.03, rtol=0)


def test_greedy_and_sampled_rows_mix_in_one_batch() -> None:
    logits = torch.stack([LOGITS, LOGITS.flip(0)])  # row 1's most likely token is 5
    params = [SamplingParams(temperature=1.0, top_k=1), SamplingParams(temperature=0)]
    generators = [make_generator(0, "cpu"), make_generator(1, "cpu")]

    assert sample(logits, params, generators).tolist() == [0, 5]
