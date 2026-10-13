"""Text generation: run the model, sample a token, append it, repeat.

Generation has two phases:

1. Prefill: run the whole prompt through the model in one call. This fills the
   KV cache with every prompt token's K and V and yields the first new token.
2. Decode: feed back the newest token alone, one model call per token, until
   an EOS token or max_tokens.
"""

import torch

from tinyserve.engine.kv_cache import KVCache
from tinyserve.engine.sampler import SamplingParams, make_generator, sample
from tinyserve.model.llama import LlamaConfig, LlamaWeights, forward


@torch.inference_mode()
def generate(
    weights: LlamaWeights,
    config: LlamaConfig,
    prompt_ids: list[int],
    params: SamplingParams,
    use_kv_cache: bool = True,
) -> list[int]:
    """Generate up to params.max_tokens tokens for one prompt. Returns only the new tokens
    (including the EOS token, if one ended the sequence).

    use_kv_cache=False is the simple version kept for comparison: every step
    reruns the model over the entire sequence so far. Its output is identical;
    it just redoes all the work for earlier tokens at every step.
    """
    device = weights.embed.device
    generator = make_generator(params.seed, device)
    output: list[int] = []

    cache = None
    if use_kv_cache:
        max_len = len(prompt_ids) + params.max_tokens
        cache = KVCache.allocate(config, 1, max_len, device, weights.embed.dtype)

    input_ids = torch.tensor([prompt_ids], device=device)  # prefill: the whole prompt
    start_pos = 0
    while len(output) < params.max_tokens:
        logits = forward(weights, config, input_ids, start_pos, cache)
        next_token = int(sample(logits[:, -1], [params], [generator]).item())
        output.append(next_token)
        if next_token in config.eos_token_ids:
            break

        if cache is not None:
            # Decode: only the new token goes through the model; the cache
            # already holds K and V for everything before it.
            start_pos += input_ids.shape[1]
            input_ids = torch.tensor([[next_token]], device=device)
        else:
            input_ids = torch.tensor([prompt_ids + output], device=device)

    return output
