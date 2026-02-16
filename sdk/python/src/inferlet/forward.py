"""
Forward pass wrapper for ``pie:core/inference``.

Low-level access to model inference. Most users should prefer
``Context.generate()`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wit_world.imports import inference as _inf

from ._async import await_future

if TYPE_CHECKING:
    from .model import Model
    from .sampler import Sampler


class ForwardPass:
    """Low-level forward pass over a model.

    Usage::

        fp = ForwardPass(model)
        fp.context(ctx)
        fp.input_tokens(tokens, positions)
        fp.sampler([last_idx], Sampler.greedy())
        output = fp.execute()
    """

    __slots__ = ("_handle",)

    def __init__(self, model: Model) -> None:
        self._handle = _inf.ForwardPass(model._handle)

    def context(self, ctx) -> None:
        """Bind a context (KV cache) to this forward pass."""
        self._handle.context(ctx._handle)

    def input_tokens(self, tokens: list[int], positions: list[int]) -> None:
        """Set the input token IDs and their absolute positions."""
        self._handle.input_tokens(tokens, positions)

    def speculative_tokens(self, tokens: list[int], positions: list[int]) -> None:
        """Set speculative (draft) tokens and their positions."""
        self._handle.input_speculative_tokens(tokens, positions)

    def enable_speculative_output(self, flag: bool = True) -> None:
        """Enable/disable speculative output (enabled by default)."""
        self._handle.output_speculative_tokens(flag)

    def attention_mask(self, mask: list[list[int]]) -> None:
        """Set a custom BRLE attention mask (default: causal)."""
        self._handle.attention_mask(mask)

    def logit_mask(self, mask: list[int]) -> None:
        """Set a BRLE-encoded logit mask for constrained generation."""
        self._handle.logit_mask(mask)

    def sampler(self, indices: list[int], sampler: Sampler) -> None:
        """Set which token positions to sample from and the sampler to use."""
        self._handle.sampler(indices, sampler._variant)

    def adapter(self, adapter) -> None:
        """Attach an adapter (e.g. LoRA) for this pass."""
        self._handle.adapter(adapter._handle)

    async def execute(self):
        """Execute the forward pass.

        Returns the ``Output`` variant from the WIT bindings.
        """
        future = self._handle.execute()
        return await await_future(future, "Forward pass execution failed")
