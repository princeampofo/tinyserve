"""A contiguous KV cache: one preallocated K and V tensor per layer.

Why cache at all: to generate token n, attention needs the keys and values of
tokens 0..n-1. Those never change once computed (a token's K and V depend only
on the tokens before it), so instead of recomputing them for the whole sequence
at every step, we compute each token's K and V once and store them here. Each
decode step then runs the model on just one new token.

"Contiguous" means each sequence gets one slab of memory sized for `max_len`
tokens up front, whether it ends up using 10 tokens or all of them. That is
simple, but it wastes memory on short sequences, which is the problem a paged
cache solves later.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # llama.py imports this file, so import its config only for type hints
    from tinyserve.model.llama import LlamaConfig


@dataclass
class KVCache:
    k: list[torch.Tensor]  # one per layer: [batch, num_kv_heads, max_len, head_dim]
    v: list[torch.Tensor]

    @classmethod
    def allocate(
        cls,
        config: "LlamaConfig",
        batch: int,
        max_len: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        shape = (batch, config.num_kv_heads, max_len, config.head_dim)
        return cls(
            k=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_layers)],
            v=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_layers)],
        )

    @property
    def max_len(self) -> int:
        return self.k[0].shape[2]

    def row(self, i: int) -> "KVCache":
        """The cache of one sequence in the batch, as a view: writes go to the full cache."""
        return KVCache(k=[k[i : i + 1] for k in self.k], v=[v[i : i + 1] for v in self.v])
