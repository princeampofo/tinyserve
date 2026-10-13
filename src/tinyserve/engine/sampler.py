"""Turn logits into the next token: greedy, temperature, top-k and top-p.

Everything is done for the whole batch at once (one row per sequence), with
each row's own settings, so a batch can mix greedy and sampled requests.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0  # 0 means greedy: always pick the highest-scoring token
    top_k: int = 0  # keep only the k most likely tokens; 0 means no limit
    top_p: float = 1.0  # keep the smallest set of tokens whose probabilities sum to top_p
    seed: int | None = None  # same seed + same prompt -> same output
    stop: list[str] = field(default_factory=list)  # end the output when any of these appears


def make_generator(seed: int | None, device: str | torch.device) -> torch.Generator:
    """Each request owns its random stream, so its output doesn't depend on
    what else happened to be in the batch."""
    g = torch.Generator(device=device)
    if seed is None:
        g.seed()
    else:
        g.manual_seed(seed)
    return g


def sample(
    logits: torch.Tensor,
    params: list[SamplingParams],
    generators: list[torch.Generator],
) -> torch.Tensor:
    """logits: [batch, vocab] for the last position of each sequence -> next tokens: [batch]."""
    device = logits.device
    logits = logits.float()
    greedy_tokens = logits.argmax(dim=-1)
    if all(p.temperature == 0 for p in params):
        return greedy_tokens

    # Temperature divides the logits: < 1 sharpens the distribution, > 1 flattens it.
    # (Greedy rows get temperature 1 here; their sampled token is discarded below.)
    temperature = torch.tensor([p.temperature or 1.0 for p in params], device=device)
    logits = logits / temperature[:, None]

    # Sort each row from most to least likely, so top-k and top-p become
    # "keep a prefix of this sorted list".
    sorted_logits, sorted_ids = logits.sort(dim=-1, descending=True)
    rank = torch.arange(logits.shape[-1], device=device)[None, :]  # [1, vocab]

    top_k = torch.tensor([p.top_k or logits.shape[-1] for p in params], device=device)
    drop = rank >= top_k[:, None]

    # top-p: drop a token if the tokens ranked above it already cover top_p of
    # the probability mass. The top token is never dropped.
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs
    top_p = torch.tensor([p.top_p for p in params], device=device)
    drop |= mass_before >= top_p[:, None]

    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    probs = sorted_logits.softmax(dim=-1)

    # Draw one token per row with that row's own generator. Dividing the
    # probabilities by Exponential(1) noise and taking the argmax picks token i
    # with probability probs[i] (the "exponential race" trick), which is the same
    # as torch.multinomial but lets each row use its own generator.
    noise = torch.stack([torch.empty_like(probs[0]).exponential_(generator=g) for g in generators])
    choice = (probs / noise).argmax(dim=-1)  # index into the sorted order
    sampled_tokens = sorted_ids.gather(-1, choice[:, None]).squeeze(-1)

    is_greedy = torch.tensor([p.temperature == 0 for p in params], device=device)
    return torch.where(is_greedy, greedy_tokens, sampled_tokens)
