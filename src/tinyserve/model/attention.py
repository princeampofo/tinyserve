"""The attention math itself: softmax(Q K^T / sqrt(d)) V.

This is kept in its own file because it is the piece that changes as the engine
grows. The keys and values come out of the KV cache: today a contiguous one,
later a paged one. The math below stays the same.
"""

import torch
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal grouped-query attention where the queries are the *last* tokens of the sequence.

    Shapes:
        q: [batch, num_heads, q_len, head_dim]       the new tokens
        k, v: [batch, num_kv_heads, kv_len, head_dim]  every token so far, new ones included
    Returns:
        [batch, num_heads, q_len, head_dim]

    Prefill has q_len == kv_len (the whole prompt at once). Decode has q_len == 1:
    one new token attending to everything cached before it plus itself.

    Llama 3.2 uses grouped-query attention (GQA): 32 query heads share 8 KV heads,
    so each KV head serves a group of 4 query heads. Storing 8 instead of 32 KV
    heads makes the KV cache 4x smaller, which is what limits how many sequences
    we can serve at once. To do the math we repeat each KV head once per query
    head in its group, so the shapes line up for a normal attention call.
    """
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)

    q_len, kv_len = q.shape[2], k.shape[2]
    if q_len == 1:
        # A single newest token may look at every token, so nothing to mask.
        return F.scaled_dot_product_attention(q, k, v)
    if q_len == kv_len:
        # Token i may look at tokens 0..i.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # General case: query i sits at absolute position (kv_len - q_len + i).
    # (SDPA's is_causal aligns to the top-left corner, which would be wrong here.)
    query_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
    key_pos = torch.arange(kv_len, device=q.device)
    mask = key_pos[None, :] <= query_pos[:, None]  # [q_len, kv_len], True = may attend
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
