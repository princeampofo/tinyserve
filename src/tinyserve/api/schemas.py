"""Request and response bodies for the OpenAI-compatible API.

Only the fields this server acts on are declared. Other OpenAI fields
(presence_penalty, logit_bias, ...) are accepted and ignored, so existing
clients work unchanged.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StreamOptions(BaseModel):
    include_usage: bool = False  # send a last chunk with token counts


class SamplingFields(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = ""
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    # Not in OpenAI's API; clients send these through extra_body.
    top_k: int = Field(0, ge=0)
    ignore_eos: bool = False  # generate exactly max_tokens, for load testing
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else self.stop


class CompletionRequest(SamplingFields):
    prompt: str | list[str]
    max_tokens: int = Field(16, ge=1)  # OpenAI's default for /v1/completions


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(SamplingFields):
    messages: list[ChatMessage]
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)  # newer name for max_tokens


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: None = None
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int
    message: AssistantMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage
