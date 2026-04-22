"""
ForwardPass class for low-level model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wit_world.imports import forward as _forward
from .async_utils import await_future

if TYPE_CHECKING:
    from .model import Queue


@dataclass
class ForwardPassResult:
    """Result of a forward pass execution."""

    tokens: list[int] | None
    distributions: list[tuple[list[int], list[float]]] | None


class ForwardPass:
    """
    Low-level forward pass for advanced model inference.

    Provides fine-grained control over input tokens, KV cache,
    attention masks, and output sampling.
    """

    def __init__(self, inner: _forward.ForwardPass, queue: "Queue") -> None:
        self._inner = inner
        self._queue = queue

    def input_tokens(self, tokens: list[int], positions: list[int]) -> None:
        """
        Set input tokens with their positions.

        Args:
            tokens: Token IDs to process
            positions: Position indices for each token
        """
        _forward.input_tokens(self._inner, tokens, positions)

    def input_embeddings(self, emb_ptrs: list[int], positions: list[int]) -> None:
        """
        Set input embeddings with their positions.

        Args:
            emb_ptrs: Pointers to embedding resources
            positions: Position indices for each embedding
        """
        _forward.input_embeddings(self._inner, emb_ptrs, positions)

    def kv_cache(
        self,
        kv_page_ptrs: list[int],
        last_kv_page_len: int,
        actual_pages: int | None = None,
    ) -> None:
        """
        Set the KV cache pages for the forward pass.

        Args:
            kv_page_ptrs: Pointers to KV cache pages
            last_kv_page_len: Length of the last KV page
            actual_pages: Number of pages that currently contain data. Defaults
                to len(kv_page_ptrs). Supply a smaller value when pre-allocating
                trailing pages (see Rust SDK Context::flush).
        """
        if actual_pages is None:
            actual_pages = len(kv_page_ptrs)
        _forward.kv_cache(self._inner, kv_page_ptrs, last_kv_page_len, actual_pages)

    def attention_mask(self, mask: list[list[int]]) -> None:
        """
        Set the attention mask for the forward pass.

        Args:
            mask: 2D attention mask (list of lists)
        """
        _forward.attention_mask(self._inner, mask)

    def output_tokens(self, indices: list[int], temperature: float) -> None:
        """
        Request token sampling at specific indices.

        Args:
            indices: Indices at which to sample tokens
            temperature: Sampling temperature
        """
        _forward.output_tokens(self._inner, indices, temperature)

    def output_tokens_top_p(
        self, indices: list[int], temperature: float, top_p: float
    ) -> None:
        """
        Request token sampling with top-p (nucleus) sampling.

        Args:
            indices: Indices at which to sample tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        _forward.output_tokens_top_p(self._inner, indices, temperature, top_p)

    def output_tokens_top_k(
        self, indices: list[int], temperature: float, top_k: int
    ) -> None:
        """
        Request token sampling with top-k sampling.

        Args:
            indices: Indices at which to sample tokens
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
        """
        _forward.output_tokens_top_k(self._inner, indices, temperature, top_k)

    def output_tokens_top_k_top_p(
        self, indices: list[int], temperature: float, top_k: int, top_p: float
    ) -> None:
        """
        Request token sampling with combined top-k and top-p.

        Args:
            indices: Indices at which to sample tokens
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            top_p: Nucleus sampling threshold
        """
        _forward.output_tokens_top_k_top_p(
            self._inner, indices, temperature, top_k, top_p
        )

    def output_tokens_min_p(
        self, indices: list[int], temperature: float, min_p: float
    ) -> None:
        """
        Request token sampling with min-p threshold.

        Args:
            indices: Indices at which to sample tokens
            temperature: Sampling temperature
            min_p: Minimum probability threshold
        """
        _forward.output_tokens_min_p(self._inner, indices, temperature, min_p)

    def output_distributions(
        self, indices: list[int], temperature: float, top_k: int | None = None
    ) -> None:
        """
        Request logit distributions at specific indices.

        Args:
            indices: Indices at which to get distributions
            temperature: Temperature for softmax
            top_k: Optional top-k limit for distributions
        """
        _forward.output_distributions(self._inner, indices, temperature, top_k)

    def output_embeddings(self, emb_ptrs: list[int], indices: list[int]) -> None:
        """
        Request output embeddings at specific indices.

        Args:
            emb_ptrs: Pointers to embedding resources to fill
            indices: Indices at which to extract embeddings
        """
        _forward.output_embeddings(self._inner, emb_ptrs, indices)

    def set_adapter(self, adapter_ptr: int) -> None:
        """
        Set the adapter for this forward pass (LoRA inference).

        Args:
            adapter_ptr: Pointer to the adapter resource
        """
        from wit_world.imports import inferlet_adapter_common as _adapter

        _adapter.set_adapter(self._inner, adapter_ptr)

    def set_adapter_seed(self, seed: int) -> None:
        """
        Set the random seed for adapter perturbation (ZO optimization).

        Args:
            seed: Random seed for deterministic perturbation
        """
        from wit_world.imports import inferlet_zo_evolve as _zo

        _zo.set_adapter_seed(self._inner, seed)

    def execute(self) -> ForwardPassResult:
        """
        Execute the forward pass and wait for results.

        Returns:
            ForwardPassResult containing tokens and/or distributions
        """
        result_resource = self._inner.execute()

        # When no output is requested (e.g., flush()), the result may be None.
        # This is valid - return empty result like JS SDK does.
        if result_resource is None:
            return ForwardPassResult(tokens=None, distributions=None)

        # Wait for result to be ready
        pollable = result_resource.pollable()
        pollable.block()

        tokens = result_resource.get_tokens()
        distributions = result_resource.get_distributions()

        return ForwardPassResult(
            tokens=list(tokens) if tokens else None,
            distributions=list(distributions) if distributions else None,
        )

    def __enter__(self) -> "ForwardPass":
        return self

    def __exit__(self, *args: object) -> None:
        pass  # Resources managed by WASM runtime
