"""
Python Backend Handler

This module provides the FlashInfer-based handler for the Python backend.
It instantiates the common Handler class with FlashInfer operations.
"""

from common import Handler
from backend_ops import FlashInferOps


class PythonHandler(Handler):
    """Python backend handler using FlashInfer operations."""

    def __init__(
        self,
        config: dict,
    ):
        """Initialize Python handler with FlashInfer operations."""

        # Create FlashInfer ops instance
        flashinfer_ops = FlashInferOps()

        # Initialize parent with FlashInfer ops
        super().__init__(
            config=config,
            ops=flashinfer_ops,
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
