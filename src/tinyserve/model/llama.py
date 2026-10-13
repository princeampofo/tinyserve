"""A Llama 3 forward pass written from scratch in plain PyTorch.

Read `forward()` at the bottom first: it is the whole model in about fifteen
lines. Everything above it is one building block of that loop:

    tokens -> embed -> for each layer: [norm -> attention -> add, norm -> MLP -> add]
           -> final norm -> lm_head -> logits

Weights live in plain dataclasses rather than nn.Modules. A forward pass is
just a function of (weights, input), and writing it that way keeps every
tensor operation visible, which is what later lets us swap in a KV cache and
paged attention without fighting a framework.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from tinyserve.engine.kv_cache import KVCache
from tinyserve.model.attention import causal_attention


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    # Llama 3.1+ "llama3" RoPE scaling. None means plain RoPE.
    rope_scaling: dict[str, Any] | None
    tie_word_embeddings: bool
    # Generating any of these ends the sequence. Llama 3.2 Instruct has three:
    # <|end_of_text|>, <|eom_id|> and <|eot_id|> (end of a chat turn).
    eos_token_ids: list[int]

    @classmethod
    def from_hf_dict(
        cls, d: dict[str, Any], generation_config: dict[str, Any] | None = None
    ) -> "LlamaConfig":
        """Build from a HuggingFace config.json dict (and optionally generation_config.json,
        whose EOS list is the one HF generate() stops on).

        Hub config.json files use `rope_theta` + `rope_scaling`; transformers 5
        merges both into `rope_parameters`. Accept either.
        """
        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        rope_theta = rope.get("rope_theta", d.get("rope_theta", 10000.0))
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        eos = (generation_config or {}).get("eos_token_id", d.get("eos_token_id"))
        eos_token_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            num_kv_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=rope_theta,
            rope_scaling=rope if rope_type == "llama3" else None,
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            eos_token_ids=eos_token_ids,
        )


@dataclass
class LayerWeights:
    attn_norm: torch.Tensor  # [hidden]
    wq: torch.Tensor  # [num_heads * head_dim, hidden]
    wk: torch.Tensor  # [num_kv_heads * head_dim, hidden]
    wv: torch.Tensor  # [num_kv_heads * head_dim, hidden]
    wo: torch.Tensor  # [hidden, num_heads * head_dim]
    mlp_norm: torch.Tensor  # [hidden]
    w_gate: torch.Tensor  # [intermediate, hidden]
    w_up: torch.Tensor  # [intermediate, hidden]
    w_down: torch.Tensor  # [hidden, intermediate]


@dataclass
class LlamaWeights:
    embed: torch.Tensor  # [vocab, hidden]
    layers: list[LayerWeights]
    final_norm: torch.Tensor  # [hidden]
    lm_head: torch.Tensor  # [vocab, hidden]; the same tensor as `embed` when tied


# ---------------------------------------------------------------- building blocks


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: scale each vector to unit root-mean-square, then by a learned weight.

    Unlike LayerNorm it does not subtract the mean, which saves work and works
    just as well. The statistic is computed in fp32 because squaring fp16
    activations can overflow; HF does the same, and matching it is needed for
    logit parity.
    """
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def rope_inv_freq(config: LlamaConfig, device: torch.device) -> torch.Tensor:
    """The rotation speed of each pair of dimensions in a head: [head_dim // 2].

    Pair i rotates by angle `position * theta^(-2i/d)`. Early pairs spin fast
    and encode fine-grained, local position; late pairs spin slowly and encode
    coarse, long-range position.

    Llama 3 trained at 8K context and was extended to 128K by slowing down the
    slow pairs ("llama3" scaling): pairs whose wavelength is longer than the
    original context are slowed by `factor`, fast pairs are untouched, and the
    ones in between are blended smoothly.
    """
    d = config.head_dim
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64, device=device).float() / d)
    )
    s = config.rope_scaling
    if s is None:
        return inv_freq

    factor = s["factor"]
    low_freq_factor = s["low_freq_factor"]
    high_freq_factor = s["high_freq_factor"]
    old_context = s["original_max_position_embeddings"]
    low_freq_wavelen = old_context / low_freq_factor
    high_freq_wavelen = old_context / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    blended = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, blended, scaled)


def rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of every rotation angle, for each position: [batch, seq, head_dim].

    Positions are passed in explicitly instead of assumed to be 0..n-1, because
    during decode the single new token sits at position n, not 0.
    Angles are computed in fp32: positions up to 128K times a frequency need the
    precision.
    """
    angles = positions[..., None].float() * inv_freq  # [batch, seq, head_dim // 2]
    angles = torch.cat([angles, angles], dim=-1)  # one angle per dim, matching rotate_half
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k by its position: x [batch, heads, seq, head_dim].

    Dimension i is paired with dimension i + head_dim/2 (the "rotate half"
    layout HF checkpoints use), and each pair is rotated like a 2D point:
        (a, b) -> (a cos - b sin, b cos + a sin)
    Because q and k are both rotated, their dot product depends only on the
    distance between the two tokens, which is how attention sees relative position.
    """
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    cos, sin = cos[:, None], sin[:, None]  # broadcast over heads
    return x * cos + rotated * sin


def attention_block(
    x: torch.Tensor,
    w: LayerWeights,
    config: LlamaConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
) -> torch.Tensor:
    """Project to q/k/v, rotate q/k by position, attend, project back. x: [batch, seq, hidden].

    With a cache, the new tokens' K and V are written into it at
    [start_pos, start_pos + seq), and attention reads every cached token from 0
    up to the newest one. Without a cache, the tokens attend only to each other.
    """
    batch, seq, _ = x.shape
    # [batch, seq, heads * head_dim] -> [batch, heads, seq, head_dim]
    q = (x @ w.wq.T).view(batch, seq, config.num_heads, config.head_dim).transpose(1, 2)
    k = (x @ w.wk.T).view(batch, seq, config.num_kv_heads, config.head_dim).transpose(1, 2)
    v = (x @ w.wv.T).view(batch, seq, config.num_kv_heads, config.head_dim).transpose(1, 2)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if k_cache is not None and v_cache is not None:
        end = start_pos + seq
        k_cache[:, :, start_pos:end] = k
        v_cache[:, :, start_pos:end] = v
        k, v = k_cache[:, :, :end], v_cache[:, :, :end]

    out = causal_attention(q, k, v)  # [batch, heads, seq, head_dim]
    out = out.transpose(1, 2).reshape(batch, seq, config.num_heads * config.head_dim)
    return out @ w.wo.T


def mlp_block(x: torch.Tensor, w: LayerWeights) -> torch.Tensor:
    """SwiGLU MLP: down( silu(gate(x)) * up(x) ).

    The gate decides, per hidden unit, how much of `up(x)` passes through. It
    expands to 4x the hidden size (2048 -> 8192 for 1B) and back.
    """
    return (F.silu(x @ w.w_gate.T) * (x @ w.w_up.T)) @ w.w_down.T


# ---------------------------------------------------------------- the model


def forward(
    weights: LlamaWeights,
    config: LlamaConfig,
    input_ids: torch.Tensor,
    start_pos: int = 0,
    cache: KVCache | None = None,
) -> torch.Tensor:
    """Run the model on new tokens. input_ids: [batch, seq] -> logits: [batch, seq, vocab].

    The tokens sit at positions start_pos .. start_pos + seq - 1. Prefill passes
    the whole prompt with start_pos=0; each decode step passes one token with
    start_pos = number of tokens already in the cache.
    """
    seq = input_ids.shape[1]
    positions = torch.arange(start_pos, start_pos + seq, device=input_ids.device)[None]

    x = weights.embed[input_ids]  # [batch, seq, hidden]
    cos, sin = rope_cos_sin(positions, rope_inv_freq(config, x.device), x.dtype)

    for i, layer in enumerate(weights.layers):
        k_cache = cache.k[i] if cache is not None else None
        v_cache = cache.v[i] if cache is not None else None
        # Pre-norm residual blocks: normalize the input to each block, then add
        # its output back onto the unnormalized "residual stream" x.
        h = rms_norm(x, layer.attn_norm, config.rms_norm_eps)
        x = x + attention_block(h, layer, config, cos, sin, start_pos, k_cache, v_cache)
        x = x + mlp_block(rms_norm(x, layer.mlp_norm, config.rms_norm_eps), layer)

    x = rms_norm(x, weights.final_norm, config.rms_norm_eps)
    return x @ weights.lm_head.T
