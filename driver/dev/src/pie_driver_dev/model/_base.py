"""Base class shared by dense-transformer ForwardPass implementations.

`DenseForwardPass` owns the boilerplate that every dense decoder
architecture (llama3, qwen2/3, gemma2/3, mistral3, olmo3, phi3, …) was
copy-pasting:

- Workspace buffer + the standard flashinfer wrappers (`wrapper_decode`,
  `wrapper_append`)
- Default `embed_tokens`, `embed_inputs`, `sample`

Subclasses must implement `_run_layers(...)` (the per-layer for-loop
calling their architecture-specific `attention` and `mlp`) and
`lm_head(...)`. Subclasses may override `__init__` (calling
`super().__init__(...)` first) to add architecture-specific state such as
RoPE caches or rejection of unsupported configs.

This class targets dense single-attention-type architectures. Models with
heterogeneous per-layer attention (e.g. gemma4 with sliding+full) or MoE
(mixtral, gpt_oss) need bespoke handling and don't inherit from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

import pie_kernels as ops

from . import ModelConfig, common
from ..adapter import AdapterSubpass
from ..config import RuntimeConfig
from ..schema import WeightStore

if TYPE_CHECKING:
    import torch.distributed as dist


class DenseForwardPass(ABC):
    """ForwardPass base for dense-transformer architectures."""

    KV_LAYOUT: str = "NHD"
    WORKSPACE_BYTES: int = 128 * 1024 * 1024
    DECODE_WRAPPER_KWARGS: dict = {}

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
    ):
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights
        self.tp_size = runtime_config.tensor_parallel_size
        self.tp_rank = runtime_config.rank % self.tp_size

        device = runtime_config.device
        self.workspace_buffer = torch.zeros(
            self.WORKSPACE_BYTES, dtype=torch.uint8, device=device,
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, self.KV_LAYOUT, **self.DECODE_WRAPPER_KWARGS,
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, self.KV_LAYOUT,
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Default: standard embedding lookup. Models that scale embeddings
        (Gemma family) override this."""
        return F.embedding(token_ids, self.weights.get("embed_token"))

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        ids = torch.as_tensor(
            batch_metadata["token_ids"],
            device=self.runtime_config.device,
            dtype=torch.int32,
        )
        return self.embed_tokens(ids)

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict[str, Any]) -> dict[str, Any]:
        return common.sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=lambda x: self.lm_head(x),
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

    @abstractmethod
    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper: Any,
    ) -> torch.Tensor:
        """Iterate `range(num_layers)`, calling the architecture's
        `attention` and `mlp` per layer. Models without adapter integration
        accept and ignore `adapter_subpass`."""

    @abstractmethod
    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden_states to vocab logits (after the final norm)."""
