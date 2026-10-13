"""OpenAI-compatible HTTP API in front of the engine.

Each HTTP request is turned into token ids and handed to the engine thread.
The event loop stays free while the engine works, so many requests can be in
flight at once, which is what lets the engine batch them.

Without streaming, the handler awaits the finished request. With stream=true,
the engine signals after every token and the handler sends the new text as a
Server-Sent Event; if the client disconnects, the request is aborted so the
engine stops generating for it and frees its KV cache blocks.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from transformers import PreTrainedTokenizerBase

from tinyserve.api.schemas import (
    AssistantMessage,
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    SamplingFields,
    Usage,
)
from tinyserve.api.streaming import TextStream, output_text, sse
from tinyserve.engine.engine import Engine
from tinyserve.engine.request import Request
from tinyserve.engine.sampler import SamplingParams

# Builds one streamed chunk from (new text or None, finish_reason or None).
ChunkBuilder = Callable[[str | None, str | None], dict[str, Any]]


def create_app(
    engine: Engine,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    max_model_len: int = 4096,
    request_timeout_s: float = 600.0,
) -> FastAPI:
    app = FastAPI(title="tinyserve")

    def check_supported(body: SamplingFields) -> None:
        if body.n != 1:
            raise HTTPException(400, "only n=1 is supported")

    def sampling_params(body: SamplingFields, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            seed=body.seed,
            stop=body.stop_list(),
            ignore_eos=body.ignore_eos,
        )

    def submit(
        prompt_ids: list[int],
        params: SamplingParams,
        notify: Callable[[Request], None],
        notify_every_token: bool,
    ) -> Request:
        """Hand a request to the engine. `notify` runs on this event loop (not the
        engine thread) when it finishes, and after every token if asked."""
        if len(prompt_ids) + params.max_tokens > max_model_len:
            raise HTTPException(
                400,
                f"prompt ({len(prompt_ids)} tokens) + max_tokens ({params.max_tokens}) "
                f"exceeds the {max_model_len}-token context limit",
            )
        loop = asyncio.get_running_loop()

        def on_engine_thread(request: Request) -> None:
            # Asyncio objects may only be touched from their own loop's thread.
            loop.call_soon_threadsafe(notify, request)

        request = Request(
            prompt_ids,
            params,
            on_token=on_engine_thread if notify_every_token else None,
            on_finish=on_engine_thread,
        )
        engine.submit(request)
        return request

    async def run(prompt_ids: list[int], params: SamplingParams) -> Request:
        """Submit and wait for the finished request."""
        done: asyncio.Future[Request] = asyncio.get_running_loop().create_future()

        def resolve(request: Request) -> None:
            if not done.done():  # it is cancelled if we already timed out
                done.set_result(request)

        request = submit(prompt_ids, params, resolve, notify_every_token=False)
        try:
            await asyncio.wait_for(done, request_timeout_s)
        except TimeoutError:
            request.finish_reason = "abort"  # the engine drops it at its next step
            raise HTTPException(504, "request timed out") from None
        if request.finish_reason == "error":
            raise HTTPException(500, "generation failed")
        return request

    def stream(
        prompt_ids: list[int],
        params: SamplingParams,
        chunk: ChunkBuilder,
        include_usage: bool,
        first_chunk: dict[str, Any] | None = None,
    ) -> StreamingResponse:
        updates: asyncio.Queue[Request] = asyncio.Queue()
        request = submit(prompt_ids, params, updates.put_nowait, notify_every_token=True)

        async def events() -> AsyncIterator[str]:
            text = TextStream(tokenizer, params.stop)
            try:
                if first_chunk is not None:
                    yield sse(first_chunk)
                while True:
                    try:
                        await asyncio.wait_for(updates.get(), request_timeout_s)
                    except TimeoutError:
                        request.finish_reason = "abort"
                    finished = request.finished
                    delta = text.delta(list(request.output_ids), finished)
                    if delta:
                        yield sse(chunk(delta, None))
                    if finished:
                        break
                yield sse(chunk(None, request.finish_reason))
                if include_usage:
                    final = chunk(None, None) | {
                        "choices": [],
                        "usage": usage(request).model_dump(),
                    }
                    yield sse(final)
                yield sse("[DONE]")
            finally:
                # Reached early only if the client disconnected (the response
                # task is cancelled): stop generating and free the request's blocks.
                if not request.finished:
                    request.finish_reason = "abort"

        return StreamingResponse(events(), media_type="text/event-stream")

    def usage(*requests: Request) -> Usage:
        prompt_tokens = sum(len(r.prompt_ids) for r in requests)
        completion_tokens = sum(len(r.output_ids) for r in requests)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        # The app is only created after the model has loaded, so serving at all means ready.
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "tinyserve", "created": 0}],
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest) -> CompletionResponse | StreamingResponse:
        check_supported(body)
        prompts = [body.prompt] if isinstance(body.prompt, str) else body.prompt
        params = sampling_params(body, body.max_tokens)
        response_id, created = f"cmpl-{time.time_ns():x}", int(time.time())

        if body.stream:
            if len(prompts) != 1:
                raise HTTPException(400, "streaming supports a single prompt")

            def chunk(text: str | None, finish_reason: str | None) -> dict[str, Any]:
                choice = {
                    "index": 0,
                    "text": text or "",
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
                return {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [choice],
                }

            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            return stream(tokenizer(prompts[0]).input_ids, params, chunk, include_usage)

        # Submit every prompt before awaiting any, so they can share a batch.
        requests = await asyncio.gather(*(run(tokenizer(p).input_ids, params) for p in prompts))
        return CompletionResponse(
            id=response_id,
            created=created,
            model=model_name,
            choices=[
                CompletionChoice(
                    index=i,
                    text=output_text(tokenizer, r.output_ids, r.params.stop),
                    finish_reason=str(r.finish_reason),
                )
                for i, r in enumerate(requests)
            ],
            usage=usage(*requests),
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        check_supported(body)
        # The chat template renders the conversation as one string with Llama's
        # special header tokens; it already starts with <|begin_of_text|>.
        prompt = tokenizer.apply_chat_template(
            [m.model_dump() for m in body.messages], add_generation_prompt=True, tokenize=False
        )
        prompt_ids: list[int] = tokenizer(str(prompt), add_special_tokens=False).input_ids
        max_tokens = (
            body.max_completion_tokens or body.max_tokens or max_model_len - len(prompt_ids)
        )
        params = sampling_params(body, max_tokens)
        response_id, created = f"chatcmpl-{time.time_ns():x}", int(time.time())

        if body.stream:

            def chunk(text: str | None, finish_reason: str | None) -> dict[str, Any]:
                delta = {"content": text} if text is not None else {}
                return {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }

            # OpenAI's first chat chunk announces the role, with no text yet.
            first = chunk("", None)
            first["choices"][0]["delta"] = {"role": "assistant", "content": ""}
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            return stream(prompt_ids, params, chunk, include_usage, first_chunk=first)

        request = await run(prompt_ids, params)
        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=AssistantMessage(
                        content=output_text(tokenizer, request.output_ids, request.params.stop)
                    ),
                    finish_reason=str(request.finish_reason),
                )
            ],
            usage=usage(request),
        )

    return app
