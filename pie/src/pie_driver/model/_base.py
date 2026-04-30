"""Base class shared by dense-transformer ForwardPass implementations.

`CudaGraphForwardPass` owns the boilerplate that every dense decoder
architecture (llama3, qwen2/3, gemma2/3, mistral3, olmo3, phi3, …) was
copy-pasting:

- Workspace buffer + the standard flashinfer wrappers
  (`wrapper_decode`, `wrapper_append`, per-bin `cuda_graph_wrappers`,
  `wrapper_decode_fallback`)
- Shared static buffers used as graph capture inputs:
  `shared_static_{hidden, indptr, last_len, position_ids,
  batch_indices, batch_positions}` and `shared_kv_indices_buffer`
- The decode batch-size bin scheme (`[1,2,4,8,16] + range(24, limit+1, 16)`)
- Helpers: `_get_bin`, `warmup_cuda_graphs`, `_run_layers_graphed`,
  `embed_tokens`, `embed_inputs`, `sample`
- The standard `transform()` dispatch shape: decode-mode picks the
  graphed path when `use_cuda_graphs and adapter_subpass is None`,
  otherwise eager decode; `single_token_inference_mode=False` always
  uses the prefill wrapper

Subclasses must implement `_run_layers(...)` (the per-layer for-loop
calling their architecture-specific `attention` and `mlp`) and
`lm_head(...)`. Subclasses may override `__init__` (calling
`super().__init__(...)` first) to add architecture-specific state such as
RoPE caches, alternative wrappers, or rejection of unsupported configs.

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


def _default_decode_bins(limit: int) -> list[int]:
    """Decode batch-size bin set: powers-of-two up to 16, then steps of 16."""
    bins = [b for b in (1, 2, 4, 8, 16) if b <= limit]
    if limit > 16:
        bins.extend(range(24, limit + 1, 16))
        if bins[-1] < limit:
            bins.append(limit)
    return sorted(set(bins))


class CudaGraphForwardPass(ABC):
    """ForwardPass base for dense-transformer architectures with the
    standard decode-time CUDA graph capture pattern. See module docstring
    for the contract."""

    # Subclasses can override these if they need a different layout / size.
    KV_LAYOUT: str = "NHD"
    WORKSPACE_BYTES: int = 128 * 1024 * 1024
    # Extra kwargs forwarded to every `BatchDecodeWithPagedKVCacheWrapper`
    # constructed by this base (the per-bin graph wrappers, the always-on
    # `wrapper_decode`, and the eager `wrapper_decode_fallback`). Override
    # in subclasses that want e.g. ``use_tensor_cores=True``.
    DECODE_WRAPPER_KWARGS: dict = {}

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
        compute_process_group: "dist.ProcessGroup | None" = None,
    ):
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights
        self.compute_process_group = compute_process_group
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

        # --- CUDA graph state --------------------------------------------
        self.use_cuda_graphs = runtime_config.use_cuda_graphs

        limit = runtime_config.max_batch_size or 512
        self.cuda_graph_bins = _default_decode_bins(limit)
        max_bin = self.cuda_graph_bins[-1]

        # Shared static buffers, sized to the largest bin.
        self.shared_static_hidden = torch.zeros(
            (max_bin, model_config.dim_hidden),
            dtype=runtime_config.activation_dtype, device=device,
        )
        self.shared_static_indptr = torch.zeros(
            max_bin + 1, dtype=torch.int32, device=device,
        )
        self.shared_static_last_len = torch.zeros(
            max_bin, dtype=torch.int32, device=device,
        )
        self.shared_static_position_ids = torch.zeros(
            max_bin, dtype=torch.int32, device=device,
        )
        self.shared_static_batch_indices = torch.zeros(
            max_bin, dtype=torch.int32, device=device,
        )
        self.shared_static_batch_positions = torch.zeros(
            max_bin, dtype=torch.int32, device=device,
        )

        # Per-bin (indptr_view, last_len_view) pairs into the shared buffers.
        self.cuda_graph_aux_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # The +1 page reserved by the engine becomes our padding scratch.
        self.scratch_page_idx = runtime_config.max_num_kv_pages

        max_num_pages = runtime_config.max_num_kv_pages + 1
        self.shared_kv_indices_buffer = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device,
        )

        # Per-bin graph-aware decode wrappers.
        self.cuda_graph_wrappers: dict[int, Any] = {}
        if self.use_cuda_graphs:
            for b in self.cuda_graph_bins:
                indptr_view = self.shared_static_indptr[: b + 1]
                last_len_view = self.shared_static_last_len[:b]
                self.cuda_graph_aux_buffers[b] = (indptr_view, last_len_view)
                self.cuda_graph_wrappers[b] = ops.BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    self.KV_LAYOUT,
                    use_cuda_graph=True,
                    paged_kv_indptr_buffer=indptr_view,
                    paged_kv_indices_buffer=self.shared_kv_indices_buffer,
                    paged_kv_last_page_len_buffer=last_len_view,
                    **self.DECODE_WRAPPER_KWARGS,
                )

        # bin_size → (graph,) for replay (populated by `warmup_cuda_graphs`).
        self.cuda_graph_img: dict[int, tuple] = {}
        # Eager fall-back wrapper for batches that exceed the largest bin.
        self.wrapper_decode_fallback = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, self.KV_LAYOUT, **self.DECODE_WRAPPER_KWARGS,
        )

    # =====================================================================
    # CUDA graph capture / replay
    # =====================================================================

    def _get_bin(self, batch_size: int) -> int | None:
        """Smallest bin >= batch_size, or None if the batch exceeds the cap."""
        for b in self.cuda_graph_bins:
            if b >= batch_size:
                return b
        return None

    def warmup_cuda_graphs(self, kv_cache_at_layer: list[torch.Tensor]):
        """Capture a graph per bin so we don't pay the capture cost on the
        first few real iterations. No-op when graphs are disabled."""
        if not self.use_cuda_graphs:
            return

        from tqdm import tqdm

        cfg = self.model_config
        device = self.runtime_config.device
        page_size = self.runtime_config.kv_page_size
        local_q = cfg.num_q_heads // self.tp_size
        local_kv = cfg.num_kv_heads // self.tp_size

        print(f"Warmup: Capturing CUDA graphs for bins {self.cuda_graph_bins}...")

        for b in tqdm(self.cuda_graph_bins, desc="CUDA Graphs"):
            indptr_view, last_len_view = self.cuda_graph_aux_buffers[b]
            hidden_view = self.shared_static_hidden[:b]

            # Plan with a uniform-1-page-per-request layout; subsequent
            # replays will overwrite these views with real values.
            indptr_view.copy_(torch.arange(b + 1, dtype=torch.int32, device=device))
            last_len_view.fill_(1)

            wrapper = self.cuda_graph_wrappers[b]
            wrapper.plan(
                indptr=indptr_view,
                indices=self.shared_kv_indices_buffer,
                last_page_len=last_len_view,
                num_qo_heads=local_q,
                num_kv_heads=local_kv,
                head_dim=cfg.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.runtime_config.activation_dtype,
            )

            # Capture KV writes to the scratch page so we never touch real
            # context state.
            self.shared_kv_indices_buffer[:b].copy_(
                torch.full((b,), self.scratch_page_idx, device=device, dtype=torch.int32)
            )

            pos_view = self.shared_static_position_ids[:b]
            pos_view.zero_()
            batch_indices_view = self.shared_static_batch_indices[:b]
            batch_indices_view.copy_(torch.arange(b, device=device, dtype=torch.int32))
            batch_pos_view = self.shared_static_batch_positions[:b]
            batch_pos_view.zero_()

            graph = torch.cuda.CUDAGraph()
            # Eager warmup so first-call JITs happen before capture.
            self._run_layers(
                hidden_states=hidden_view,
                position_ids=pos_view,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=self.shared_kv_indices_buffer,
                kv_page_indptr=indptr_view,
                kv_last_page_lens=last_len_view,
                batch_indices=batch_indices_view,
                batch_positions=batch_pos_view,
                wrapper=wrapper,
                adapter_subpass=None,
            )
            with torch.cuda.graph(graph):
                self._run_layers(
                    hidden_states=hidden_view,
                    position_ids=pos_view,
                    kv_cache_at_layer=kv_cache_at_layer,
                    kv_page_indices=self.shared_kv_indices_buffer,
                    kv_page_indptr=indptr_view,
                    kv_last_page_lens=last_len_view,
                    batch_indices=batch_indices_view,
                    batch_positions=batch_pos_view,
                    wrapper=wrapper,
                    adapter_subpass=None,
                )
            self.cuda_graph_img[b] = (graph,)

        torch.cuda.synchronize()
        print("Warmup complete.")

    def _run_layers_graphed(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        total_pages_cpu: int = 0,
    ) -> torch.Tensor:
        """Replay the captured graph for the smallest bin >= batch_size,
        padding inputs as needed. Falls back to the eager decode path when
        the batch exceeds the largest bin."""
        cfg = self.model_config
        batch_size = hidden_states.shape[0]
        bin_size = self._get_bin(batch_size)

        if bin_size is None or bin_size not in self.cuda_graph_img:
            wrapper = self.wrapper_decode_fallback
            page_size = self.runtime_config.kv_page_size
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=cfg.num_q_heads // self.tp_size,
                num_kv_heads=cfg.num_kv_heads // self.tp_size,
                head_dim=cfg.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=hidden_states.dtype,
            )
            return self._run_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                wrapper=wrapper,
                adapter_subpass=None,
            )

        # Stage live inputs into the static buffers the graph reads from.
        self.shared_static_hidden[:batch_size].copy_(hidden_states)
        indptr_view = self.cuda_graph_aux_buffers[bin_size][0]
        indptr_view[: batch_size + 1].copy_(kv_page_indptr)
        last_len_view = self.cuda_graph_aux_buffers[bin_size][1]
        last_len_view[:batch_size].copy_(kv_last_page_lens)

        num_indices = kv_page_indices.numel()
        self.shared_kv_indices_buffer[:num_indices].copy_(kv_page_indices)

        self.shared_static_position_ids[:batch_size].copy_(position_ids)
        self.shared_static_batch_indices[:batch_size].copy_(batch_indices)
        self.shared_static_batch_positions[:batch_size].copy_(batch_positions)

        # Pad the slack rows so flashinfer doesn't sample / write to KV
        # pages belonging to other contexts when batch_size < bin_size.
        if batch_size < bin_size:
            remainder = bin_size - batch_size
            padding = torch.arange(
                1, remainder + 1, device=self.runtime_config.device, dtype=torch.int32,
            )
            padding.add_(total_pages_cpu)
            indptr_view[batch_size + 1 :].copy_(padding)
            self.shared_kv_indices_buffer[num_indices:].fill_(self.scratch_page_idx)
            last_len_view[batch_size:].fill_(1)
            self.shared_static_position_ids[batch_size:].zero_()
            self.shared_static_batch_indices[batch_size:].zero_()
            self.shared_static_batch_positions[batch_size:].zero_()

        graph = self.cuda_graph_img[bin_size][0]
        graph.replay()
        return self.shared_static_hidden[:batch_size].clone()

    # =====================================================================
    # Standard transforms (subclasses can override)
    # =====================================================================

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

    # =====================================================================
    # Subclass contract
    # =====================================================================

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
