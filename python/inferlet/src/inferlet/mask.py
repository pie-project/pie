"""
Packed-bitmask logit-mask semantics — port of `crates/inferlet/src/mask.rs`,
byte-identical to the engine's `0x65 MaskApply` op.

A logit mask is one bit per vocabulary token, packed into ``ceil(vocab/32)``
``u32`` words: bit ``1`` = allowed. Token ``j``'s bit is word ``j >> 5``, bit
``j & 31``. The host grammar matcher hands such a mask out; ``unpack_mask``
turns it into the ``[vocab] bool`` cell a `masked_argmax` epilogue reads.
"""

from __future__ import annotations

from typing import Sequence


def mask_words(vocab: int) -> int:
    """Number of ``u32`` words a packed mask for ``vocab`` tokens occupies."""
    return -(-vocab // 32)


def all_allowed(vocab: int) -> list[int]:
    """An all-allowed packed mask (every bit ``1``) — the identity under the
    word-wise AND that composes two constraints."""
    return [0xFFFF_FFFF] * mask_words(vocab)


def bit_allowed(mask: Sequence[int], j: int) -> bool:
    """Whether token ``j`` is allowed. Tokens past the mask's word coverage
    read as disallowed — a model's output vocabulary is routinely padded
    above its tokenizer's, and those slots decode to no token at all."""
    word = j >> 5
    return word < len(mask) and (mask[word] >> (j & 31)) & 1 == 1


def pack_allowed(vocab: int, allowed: Sequence[int]) -> list[int]:
    """Pack an allowed-token id list into a packed bitmask; ids ``>= vocab``
    are ignored."""
    mask = [0] * mask_words(vocab)
    for token in allowed:
        if 0 <= token < vocab:
            mask[token >> 5] |= 1 << (token & 31)
    return mask


def unpack_mask(packed: Sequence[int], vocab: int) -> list[bool]:
    """Expand a packed mask into one ``bool`` per token. An empty ``packed``
    means the constraint is inactive, so everything is allowed."""
    if not packed:
        return [True] * vocab
    return [bit_allowed(packed, j) for j in range(vocab)]


def apply_mask_argmax(logits: Sequence[float], mask: Sequence[int]) -> int:
    """Argmax over ``logits`` with the packed mask applied (a disallowed token
    is ``-inf``; ties go to the lowest index; all-disallowed returns 0)."""
    best_idx = 0
    best_val = float("-inf")
    for j, logit in enumerate(logits):
        v = logit if bit_allowed(mask, j) else float("-inf")
        if v > best_val:
            best_val = v
            best_idx = j
    return best_idx
