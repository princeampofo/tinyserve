"""Baseline A: HuggingFace generate(), one request at a time, behind the same streaming chat API.

    uv run python benchmarks/hf_server.py --port 8000

This is what serving looks like without an engine: requests wait in a queue,
and each one gets the GPU to itself from prefill to its last token. It exists
only so the benchmark can push identical traffic through it and compare.
Only what the load generator uses is implemented: streaming
/v1/chat/completions, /v1/models and /health.
"""

import argparse
import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from transformers.generation.streamers import BaseStreamer

from tinyserve.api.streaming import TextStream, sse
from tinyserve.model.weights import DEFAULT_MODEL


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[dict[str, str]]
    max_tokens: int = 256
    ignore_eos: bool = False


@dataclass
class Job:
    prompt_ids: list[int]
    max_tokens: int
    ignore_eos: bool
    on_tokens: Callable[[list[int] | None], object]  # None = finished
    aborted: threading.Event = field(default_factory=threading.Event)


class Streamer(BaseStreamer):
    """Receives token ids from generate() as they are produced."""

    def __init__(self, on_tokens: Callable[[list[int] | None], object]) -> None:
        self.on_tokens = on_tokens
        self.prompt_seen = False

    def put(self, value: torch.Tensor) -> None:
        if not self.prompt_seen:  # generate() first passes the prompt itself
            self.prompt_seen = True
            return
        self.on_tokens(value.flatten().tolist())

    def end(self) -> None:
        self.on_tokens(None)


class StopWhenAborted(StoppingCriteria):
    def __init__(self, aborted: threading.Event) -> None:
        self.aborted = aborted

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return torch.full((input_ids.shape[0],), self.aborted.is_set(), device=input_ids.device)


def worker(model: Any, jobs: "queue.Queue[Job]") -> None:
    while True:
        job = jobs.get()
        if job.aborted.is_set():
            job.on_tokens(None)
            continue
        input_ids = torch.tensor([job.prompt_ids], device=model.device)
        with torch.inference_mode():
            model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=job.max_tokens,
                min_new_tokens=job.max_tokens if job.ignore_eos else None,
                do_sample=False,
                temperature=None,
                top_p=None,
                streamer=Streamer(job.on_tokens),
                stopping_criteria=[StopWhenAborted(job.aborted)],
                pad_token_id=model.generation_config.eos_token_id[0],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    dtype = torch.float32 if args.device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    model.to(torch.device(args.device))
    jobs: queue.Queue[Job] = queue.Queue()
    threading.Thread(target=worker, args=(model, jobs), daemon=True).start()

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": args.model, "object": "model", "created": 0, "owned_by": "hf"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest) -> StreamingResponse:
        prompt = tokenizer.apply_chat_template(
            body.messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids: list[int] = tokenizer(str(prompt), add_special_tokens=False).input_ids
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue[list[int] | None] = asyncio.Queue()
        job = Job(
            prompt_ids,
            body.max_tokens,
            body.ignore_eos,
            lambda tokens: loop.call_soon_threadsafe(updates.put_nowait, tokens),
        )
        jobs.put(job)
        created = int(time.time())

        def chunk(text: str | None, finish_reason: str | None) -> dict[str, Any]:
            delta = {"content": text} if text is not None else {}
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            return {
                "id": "hf",
                "object": "chat.completion.chunk",
                "created": created,
                "model": args.model,
                "choices": [choice],
            }

        async def events() -> AsyncIterator[str]:
            output: list[int] = []
            text = TextStream(tokenizer, [])
            try:
                while (tokens := await updates.get()) is not None:
                    output += tokens
                    if delta := text.delta(output, finished=False):
                        yield sse(chunk(delta, None))
                if delta := text.delta(output, finished=True):
                    yield sse(chunk(delta, None))
                yield sse(chunk(None, "length" if len(output) >= body.max_tokens else "stop"))
                yield sse("[DONE]")
            finally:
                job.aborted.set()  # a no-op if it finished; stops generate() if the client left

        return StreamingResponse(events(), media_type="text/event-stream")

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
