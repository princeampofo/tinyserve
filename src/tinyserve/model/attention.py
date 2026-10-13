"""The attention math itself: softmax(Q K^T / sqrt(d)) V.

This is kept in its own file because it is the piece that changes as the engine
grows. Here the keys and values are just the prompt's own K and V. With a
KV cache they are read out of the cache instead (first a contiguous one, later
a paged one), but the math below stays the same.
"""

import torch
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal grouped-query attention.

    Shapes:
        q: [batch, num_heads, seq_len, head_dim]
        k, v: [batch, num_kv_heads, seq_len, head_dim]
    Returns:
        [batch, num_heads, seq_len, head_dim]

    Llama 3.2 uses grouped-query attention (GQA): 32 query heads share 8 KV heads,
    so each KV head serves a group of 4 query heads. Storing 8 instead of 32 KV
    heads makes the KV cache 4x smaller, which is what limits how many sequences
    we can serve at once. To do the math we repeat each KV head once per query
    head in its group, so the shapes line up for a normal attention call.
    """
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    # is_causal=True masks out future positions: token i only attends to tokens 0..i.
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
