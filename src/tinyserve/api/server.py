"""OpenAI-compatible HTTP API in front of the engine.

Each HTTP request is turned into token ids, handed to the engine thread, and
awaited. The event loop stays free while the engine works, so many requests can
be waiting at once, which is what lets the engine batch them.
"""

import asyncio
import time

from fastapi import FastAPI, HTTPException
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
from tinyserve.engine.engine import Engine, Request
from tinyserve.engine.sampler import SamplingParams


def create_app(
    engine: Engine,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    max_model_len: int = 4096,
    request_timeout_s: float = 600.0,
) -> FastAPI:
    app = FastAPI(title="tinyserve")

    def check_supported(body: SamplingFields) -> None:
        if body.stream:
            raise HTTPException(400, "stream=true is not supported yet")
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
        )

    async def run(prompt_ids: list[int], params: SamplingParams) -> Request:
        """Submit to the engine and wait for the result without blocking the event loop."""
        if len(prompt_ids) + params.max_tokens > max_model_len:
            raise HTTPException(
                400,
                f"prompt ({len(prompt_ids)} tokens) + max_tokens ({params.max_tokens}) "
                f"exceeds the {max_model_len}-token context limit",
            )
        loop = asyncio.get_running_loop()
        done: asyncio.Future[Request] = loop.create_future()

        def resolve(request: Request) -> None:
            if not done.done():  # it is cancelled if we already timed out
                done.set_result(request)

        def on_finish(request: Request) -> None:
            # Runs on the engine thread; futures must be resolved on their own loop.
            loop.call_soon_threadsafe(resolve, request)

        request = Request(prompt_ids, params, on_finish=on_finish)
        engine.submit(request)
        try:
            await asyncio.wait_for(done, request_timeout_s)
        except TimeoutError:
            request.finish_reason = "abort"  # the engine drops it at its next step
            raise HTTPException(504, "request timed out") from None
        if request.finish_reason == "error":
            raise HTTPException(500, "generation failed")
        return request

    def output_text(request: Request) -> str:
        """Decoded output, cut before the first stop string (OpenAI doesn't return it)."""
        text = str(tokenizer.decode(request.output_ids, skip_special_tokens=True))
        cut = min((i for s in request.params.stop if (i := text.find(s)) != -1), default=None)
        return text if cut is None else text[:cut]

    @app.get("/health")
    def health() -> dict[str, str]:
        # The app is only created after the model has loaded, so serving at all means ready.
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "tinyserve", "created": 0}],
        }

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest) -> CompletionResponse:
        check_supported(body)
        prompts = [body.prompt] if isinstance(body.prompt, str) else body.prompt
        params = sampling_params(body, body.max_tokens)
        # Submit every prompt before awaiting any, so they can share a batch.
        requests = await asyncio.gather(*(run(tokenizer(p).input_ids, params) for p in prompts))
        prompt_tokens = sum(len(r.prompt_ids) for r in requests)
        completion_tokens = sum(len(r.output_ids) for r in requests)
        return CompletionResponse(
            id=f"cmpl-{requests[0].id}",
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(index=i, text=output_text(r), finish_reason=str(r.finish_reason))
                for i, r in enumerate(requests)
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest) -> ChatCompletionResponse:
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
        request = await run(prompt_ids, sampling_params(body, max_tokens))
        return ChatCompletionResponse(
            id=f"chatcmpl-{request.id}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=AssistantMessage(content=output_text(request)),
                    finish_reason=str(request.finish_reason),
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(request.output_ids),
                total_tokens=len(prompt_ids) + len(request.output_ids),
            ),
        )

    return app
