"""Hands out KV cache blocks from a fixed pool.

The paged KV cache is one big preallocated tensor split into equal blocks of
16 tokens. A sequence owns a list of block ids (its "block table") and gets one
more block each time it grows past a multiple of 16 tokens. This allocator
only tracks which block ids are free; it never touches the tensor itself.
"""


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        # A stack of free ids: popping and pushing at the end are both O(1).
        self._free = list(range(num_blocks))
        # Lets free() catch double frees in O(1) instead of searching the stack.
        self._is_free = [True] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("no free KV cache blocks")
        block = self._free.pop()
        self._is_free[block] = False
        return block

    def free(self, block: int) -> None:
        if self._is_free[block]:
            raise RuntimeError(f"block {block} freed twice")
        self._is_free[block] = True
        self._free.append(block)
