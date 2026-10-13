"""The attention math itself: softmax(Q K^T / sqrt(d)) V.

This is kept in its own file because it is the piece that changes as the engine
grows. The keys and values come out of the KV cache: today a contiguous one,
later a paged one. The math below stays the same.
"""

import torch
import torch.nn.functional as F


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_index: torch.Tensor
) -> torch.Tensor:
    """Causal grouped-query attention for a batch of sequences of different lengths.

    Shapes:
        q: [batch, num_heads, q_len, head_dim]          the new tokens
        k, v: [batch, num_kv_heads, kv_len, head_dim]   keys/values the new tokens may read
        q_index: [batch, q_len]  where each query's own token sits along k/v's kv_len axis
    Returns:
        [batch, num_heads, q_len, head_dim]

    Query i of row b may attend to keys 0..q_index[b, i]: itself and everything
    before it. That one rule covers every case:
      - prefill of a whole prompt: q_index = 0..n-1, the usual triangle
      - decode: one query per row at that row's current length
      - a batch whose rows have different lengths: keys past a row's own length
        hold another (longer) sequence's slot space or zeros, and are masked out

    Llama 3.2 uses grouped-query attention (GQA): 32 query heads share 8 KV heads,
    so each KV head serves a group of 4 query heads. Storing 8 instead of 32 KV
    heads makes the KV cache 4x smaller, which is what limits how many sequences
    we can serve at once. To do the math we repeat each KV head once per query
    head in its group, so the shapes line up for a normal attention call.
    """
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)

    key_index = torch.arange(k.shape[2], device=q.device)
    mask = key_index <= q_index[:, :, None]  # [batch, q_len, kv_len], True = may attend
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])  # same for all heads
