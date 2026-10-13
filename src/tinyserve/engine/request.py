"""A generation request as the engine sees it, and the rule for when it is done."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

from tinyserve.engine.sampler import SamplingParams, make_generator
from tinyserve.model.llama import LlamaConfig


@dataclass(eq=False)  # two requests are equal only if they are the same object
class Request:
    """Token ids in, token ids out."""

    prompt_ids: list[int]
    params: SamplingParams
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    output_ids: list[int] = field(default_factory=list)
    # "stop" (EOS token or stop string), "length" (hit max_tokens), "abort"
    # (the caller gave up), "error"; None while still running.
    finish_reason: str | None = None
    # Called on the engine thread once the request is done.
    on_finish: Callable[["Request"], None] | None = None
    # The request's own random stream, created when it is first scheduled. It
    # lives on the request (not the running sequence) so that a preempted
    # request continues the same stream when it is recomputed.
    generator: torch.Generator | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def rng(self, device: str | torch.device) -> torch.Generator:
        if self.generator is None:
            self.generator = make_generator(self.params.seed, device)
        return self.generator


def append_token(
    request: Request,
    token: int,
    config: LlamaConfig,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Record a newly generated token and decide whether the request is done."""
    request.output_ids.append(token)
    if token in config.eos_token_ids:
        request.finish_reason = "stop"
    elif request.params.stop and tokenizer is not None:
        # Stop strings are text, and one can span several tokens, so check the decoded output.
        text = tokenizer.decode(request.output_ids)
        if any(s in text for s in request.params.stop):
            request.finish_reason = "stop"
    if not request.finished and len(request.output_ids) >= request.params.max_tokens:
        request.finish_reason = "length"
