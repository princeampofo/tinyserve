"""The KV cache: one preallocated pool of fixed-size blocks per layer.

Why cache at all: to generate token n, attention needs the keys and values of
tokens 0..n-1. Those never change once computed (a token's K and V depend only
on the tokens before it), so instead of recomputing them for the whole sequence
at every step, we compute each token's K and V once and store them here. Each
decode step then runs the model on just one new token.

Memory is split into blocks of `block_size` tokens. Each sequence has a block
table: the list of block ids holding its tokens in order. Token at position p
lives in block `block_table[p // block_size]`, at offset `p % block_size`.

The same layout covers both cache designs:
  - paged (block_size=16): a sequence takes a new 16-token block only when it
    needs one, so memory tracks what sequences actually use.
  - contiguous (block_size=max_model_len): each sequence gets one block big
    enough for the longest possible sequence, reserved up front. Simple, but a
    request that stops after 50 tokens still holds max_model_len tokens' worth.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # llama.py imports this file, so import its config only for type hints
    from tinyserve.model.llama import LlamaConfig


@dataclass
class KVCache:
    k: list[torch.Tensor]  # one per layer: [num_blocks, block_size, num_kv_heads, head_dim]
    v: list[torch.Tensor]

    @classmethod
    def allocate(
        cls,
        config: "LlamaConfig",
        num_blocks: int,
        block_size: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        shape = (num_blocks, block_size, config.num_kv_heads, config.head_dim)
        return cls(
            k=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_layers)],
            v=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_layers)],
        )

    @property
    def num_blocks(self) -> int:
        return self.k[0].shape[0]

    @property
    def block_size(self) -> int:
        return self.k[0].shape[1]


def bytes_per_token(config: "LlamaConfig", dtype: torch.dtype) -> int:
    """K and V, for every layer and KV head: 2 x 16 x 8 x 64 x 2 bytes = 32 KB for Llama 3.2 1B in fp16."""
    element_size = torch.empty(0, dtype=dtype).element_size()
    return 2 * config.num_layers * config.num_kv_heads * config.head_dim * element_size


def blocks_for(num_tokens: int, block_size: int) -> int:
    return -(-num_tokens // block_size)  # ceiling division


def token_indices(
    block_tables: list[list[int]], positions: torch.Tensor, kv_len: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate (sequence, position) into an index into the cache viewed as one flat
    list of token slots, [num_blocks * block_size]. Computed once per forward call
    and shared by every layer.

    Returns:
        write: [batch, seq]     where each new token's K/V goes
        read:  [batch, kv_len]  where tokens 0..kv_len-1 of each sequence live

    Example with block_size=16 and block table [7, 2]: position 20 is in the
    sequence's 2nd block (20 // 16 = 1 -> block 2) at offset 4, so flat index 2 * 16 + 4 = 36.
    """
    batch = positions.shape[0]
    # Pad the tables to the same width so they fit in one tensor. Padding points
    # at block 0; shorter sequences only reach it for positions past their own
    # length, which the attention mask hides.
    width = max(len(t) for t in block_tables)
    table = torch.tensor(
        [t + [0] * (width - len(t)) for t in block_tables], device=positions.device
    )

    def lookup(pos: torch.Tensor) -> torch.Tensor:
        return table.gather(1, pos // block_size) * block_size + pos % block_size

    all_positions = torch.arange(kv_len, device=positions.device).expand(batch, kv_len)
    return lookup(positions), lookup(all_positions)
