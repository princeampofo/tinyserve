"""Streaming: turning a growing list of token ids into text the client can safely see.

Tokens and text don't line up one to one, so a streamed delta can't just be
"decode the newest token":
  - One character can span several tokens. Llama's byte-level tokenizer splits
    an emoji into byte tokens, and decoding only some of them gives the
    replacement character U+FFFD until the rest arrive.
  - A stop string can span several tokens too. If stop=["###"] and the output
    so far ends in "#", that "#" may turn out to be the start of "###", which
    must never reach the client.
So we decode the whole output each time, hold back any tail that could still
change, and send only the new part. Concatenating every delta gives exactly
the same text as the non-streaming response.
"""

import json
from typing import Any

from transformers import PreTrainedTokenizerBase


def output_text(tokenizer: PreTrainedTokenizerBase, output_ids: list[int], stop: list[str]) -> str:
    """The finished output: decoded, and cut before the first stop string (OpenAI doesn't return it)."""
    text = str(tokenizer.decode(output_ids, skip_special_tokens=True))
    cut = min((i for s in stop if (i := text.find(s)) != -1), default=None)
    return text if cut is None else text[:cut]


class TextStream:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop: list[str]) -> None:
        self.tokenizer = tokenizer
        self.stop = stop
        self.sent = 0  # characters of the output already sent

    def delta(self, output_ids: list[int], finished: bool) -> str:
        """The text that became safe to send since the last call."""
        text = output_text(self.tokenizer, output_ids, self.stop)
        if not finished:
            text = text[: max(self.sent, len(text) - self._unsettled_tail(text))]
        delta = text[self.sent :]
        self.sent = len(text)
        return delta

    def _unsettled_tail(self, text: str) -> int:
        """How many characters at the end of `text` might still change or be cut."""
        if text.endswith("�"):  # a character still missing some of its bytes
            return len(text) - len(text.rstrip("�"))
        # The longest ending of the text that is the start of some stop string.
        longest = max((len(s) - 1 for s in self.stop), default=0)
        for n in range(min(longest, len(text)), 0, -1):
            if any(s.startswith(text[-n:]) for s in self.stop):
                return n
        return 0


def sse(data: dict[str, Any] | str) -> str:
    """One Server-Sent Event. OpenAI streams `data: {json}` events and ends with `data: [DONE]`."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n"
