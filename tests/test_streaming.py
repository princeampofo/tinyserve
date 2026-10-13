"""Streamed text deltas add up to exactly the final text, and never show the
client a broken character or any part of a stop string."""

from transformers import PreTrainedTokenizerBase

from tinyserve.api.streaming import TextStream, output_text


def stream_token_by_token(
    tokenizer: PreTrainedTokenizerBase, ids: list[int], stop: list[str]
) -> list[str]:
    stream = TextStream(tokenizer, stop)
    deltas = [stream.delta(ids[:n], finished=False) for n in range(1, len(ids))]
    return deltas + [stream.delta(ids, finished=True)]


def test_deltas_add_up_to_the_final_text(tokenizer: PreTrainedTokenizerBase) -> None:
    ids = tokenizer(
        "The quick brown fox jumps over the lazy dog.", add_special_tokens=False
    ).input_ids

    deltas = stream_token_by_token(tokenizer, ids, stop=[])

    assert "".join(deltas) == output_text(tokenizer, ids, [])
    assert sum(1 for d in deltas if d) > 5  # sent as it went, not all at the end


def test_a_character_split_across_tokens_is_sent_whole(tokenizer: PreTrainedTokenizerBase) -> None:
    ids = tokenizer("Nice 🦙 llama", add_special_tokens=False).input_ids
    assert any("�" in tokenizer.decode(ids[:n]) for n in range(1, len(ids)))  # it does split

    deltas = stream_token_by_token(tokenizer, ids, stop=[])

    assert all("�" not in d for d in deltas)
    assert "".join(deltas) == "Nice 🦙 llama"


def test_text_that_could_start_a_stop_string_is_held_back(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    ids = tokenizer("Answer: 42\n\nQuestion: next", add_special_tokens=False).input_ids

    deltas = stream_token_by_token(tokenizer, ids, stop=["\n\nQuestion"])

    assert all("\n" not in d for d in deltas)
    assert "".join(deltas) == "Answer: 42"


def test_held_back_text_is_released_when_it_turns_out_not_to_be_a_stop(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    ids = tokenizer("a#b##c", add_special_tokens=False).input_ids

    deltas = stream_token_by_token(tokenizer, ids, stop=["###"])

    assert "".join(deltas) == "a#b##c"
