"""The official OpenAI client works against the server unchanged."""

import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
import pytest
import uvicorn
from conftest import Model, tiny_hf_model
from transformers import PreTrainedTokenizerBase

from tinyserve.api.server import create_app
from tinyserve.engine.engine import Engine, generate
from tinyserve.engine.sampler import SamplingParams
from tinyserve.model.llama import LlamaConfig, LlamaWeights
from tinyserve.model.weights import weights_from_state_dict


def serve(
    weights: LlamaWeights, config: LlamaConfig, tokenizer: PreTrainedTokenizerBase
) -> Iterator[str]:
    """Run the real server on a free local port in a background thread; yield its base URL."""
    engine = Engine(weights, config, tokenizer, max_batch_size=4, batch_wait_s=0.01)
    engine.start()
    app = create_app(engine, tokenizer, "tinyserve-test", max_model_len=512)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join()


@pytest.fixture(scope="module")
def tiny_with_real_vocab(tokenizer: PreTrainedTokenizerBase) -> Model:
    """The tiny random model, widened to Llama's 128K vocabulary so the real
    tokenizer's ids fit. Its text is gibberish, but every code path is real."""
    hf = tiny_hf_model(vocab_size=len(tokenizer))
    config = LlamaConfig.from_hf_dict(hf.config.to_dict())
    config.eos_token_ids = [128009]  # <|eot_id|>
    return Model(hf, config, weights_from_state_dict(config, hf.state_dict()))


@pytest.fixture(scope="module")
def base_url(tiny_with_real_vocab: Model, tokenizer: PreTrainedTokenizerBase) -> Iterator[str]:
    yield from serve(tiny_with_real_vocab.weights, tiny_with_real_vocab.config, tokenizer)


@pytest.fixture
def client(base_url: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{base_url}/v1", api_key="unused", max_retries=0)


def test_health_reports_ok(base_url: str) -> None:
    assert httpx.get(f"{base_url}/health").json() == {"status": "ok"}


def test_models_lists_the_served_model(client: openai.OpenAI) -> None:
    assert [m.id for m in client.models.list()] == ["tinyserve-test"]


def test_completion_returns_the_engines_greedy_output(
    client: openai.OpenAI, tiny_with_real_vocab: Model, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompt = "The capital of France is"
    expected_ids = generate(
        tiny_with_real_vocab.weights,
        tiny_with_real_vocab.config,
        tokenizer(prompt).input_ids,
        SamplingParams(max_tokens=8, temperature=0),
    )

    response = client.completions.create(
        model="tinyserve-test", prompt=prompt, max_tokens=8, temperature=0
    )

    assert response.choices[0].text == tokenizer.decode(expected_ids, skip_special_tokens=True)
    assert response.choices[0].finish_reason == "length"
    assert response.usage is not None and response.usage.completion_tokens == 8


def test_completion_with_several_prompts_returns_one_choice_each(client: openai.OpenAI) -> None:
    response = client.completions.create(
        model="tinyserve-test", prompt=["one", "two", "three"], max_tokens=3, temperature=0
    )
    assert [c.index for c in response.choices] == [0, 1, 2]


def test_chat_completion_returns_an_assistant_message(client: openai.OpenAI) -> None:
    response = client.chat.completions.create(
        model="tinyserve-test",
        messages=[{"role": "user", "content": "Hi!"}],
        max_tokens=5,
        temperature=0,
    )
    assert response.choices[0].message.role == "assistant"
    assert isinstance(response.choices[0].message.content, str)
    assert response.usage is not None and response.usage.completion_tokens == 5


def test_stop_string_ends_the_output_before_the_stop_string(client: openai.OpenAI) -> None:
    full = client.completions.create(
        model="tinyserve-test", prompt="Hello", max_tokens=12, temperature=0
    )
    text = full.choices[0].text
    stop = text[len(text) // 2 : len(text) // 2 + 2]  # some text from the middle of the output

    response = client.completions.create(
        model="tinyserve-test", prompt="Hello", max_tokens=12, temperature=0, stop=[stop]
    )

    assert response.choices[0].text == text[: text.find(stop)]
    assert response.choices[0].finish_reason == "stop"


def test_seeded_sampling_is_reproducible(client: openai.OpenAI) -> None:
    def ask() -> str:
        response = client.completions.create(
            model="tinyserve-test", prompt="Once", max_tokens=10, temperature=1.0, seed=3
        )
        return response.choices[0].text

    assert ask() == ask()


def test_concurrent_requests_all_complete(client: openai.OpenAI) -> None:
    def ask(i: int) -> str | None:
        response = client.completions.create(
            model="tinyserve-test", prompt=f"Request {i}", max_tokens=6, temperature=0
        )
        return response.choices[0].finish_reason

    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(ask, range(8))) == ["length"] * 8


def test_streaming_is_rejected_for_now(client: openai.OpenAI) -> None:
    with pytest.raises(openai.BadRequestError):
        client.completions.create(model="tinyserve-test", prompt="Hi", stream=True)


def test_prompt_longer_than_the_context_is_rejected(client: openai.OpenAI) -> None:
    with pytest.raises(openai.BadRequestError):
        client.completions.create(model="tinyserve-test", prompt="word " * 600, max_tokens=10)


@pytest.mark.slow
def test_llama_3_2_1b_answers_a_question_through_the_openai_client(
    llama_1b: Model, tokenizer: PreTrainedTokenizerBase
) -> None:
    for base_url in serve(llama_1b.weights, llama_1b.config, tokenizer):
        client = openai.OpenAI(base_url=f"{base_url}/v1", api_key="unused")
        response = client.chat.completions.create(
            model="tinyserve-test",
            messages=[{"role": "user", "content": "What is the capital of France? One word."}],
            max_tokens=10,
            temperature=0,
        )
        assert "Paris" in (response.choices[0].message.content or "")
        assert response.choices[0].finish_reason == "stop"
