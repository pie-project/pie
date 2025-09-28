"""
Python Backend Handler

This module provides the FlashInfer-based handler for the Python backend.
It instantiates the common Handler class with FlashInfer operations.
"""

from common import Handler, ModelInfo
from backend_ops import FlashInferOps
import torch


class PythonHandler(Handler):
    """Python backend handler using FlashInfer operations."""

    def __init__(
        self,
        model,
        model_info: ModelInfo,
        kv_page_size: int,
        max_dist_size: int,
        max_num_kv_pages: int,
        max_num_embeds: int,
        max_batch_tokens: int,
        max_num_adapters: int,
        max_adapter_rank: int,
        dtype: torch.dtype,
        device: str,
    ):
        """Initialize Python handler with FlashInfer operations."""

        # Create FlashInfer ops instance
        flashinfer_ops = FlashInferOps()

        # Initialize parent with FlashInfer ops
        super().__init__(
            model=model,
            model_info=model_info,
            ops=flashinfer_ops,
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_kv_pages=max_num_kv_pages,
            max_num_embeds=max_num_embeds,
            max_batch_tokens=max_batch_tokens,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            dtype=dtype,
            device=device,
        )

        print("✅ PythonHandler initialized with FlashInfer backend")
        print(f"   FlashInfer available: {flashinfer_ops.available}")

    def upload_handler(self, reqs):
        """Handle adapter upload requests."""
        _ = reqs  # Parameter not currently used
        raise NotImplementedError("upload_handler not yet implemented")

    def download_handler(self, reqs):
        """Handle adapter download requests."""
        _ = reqs  # Parameter not currently used
        raise NotImplementedError("download_handler not yet implemented")


# For backwards compatibility, create Handler as an alias to PythonHandler
Handler = PythonHandler
