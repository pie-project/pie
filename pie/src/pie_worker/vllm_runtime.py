"""vLLM-backed runtime for Pie.

Replaces Pie's native Runtime when backend="vllm". Implements the same
RPC interface (handshake_rpc, fire_batch, query_rpc, etc.) so Pie's
Rust server and IPC protocol work unchanged.

Architecture:
  Pie Rust IPC -> PieVllmRuntime.fire_batch() -> PieVllmBatchTranslator
  -> vLLM Worker.execute_model() -> PieVllmSamplingBridge -> response

Request lifecycle optimisation:
  First fire_batch for a sequence → NewRequestData (creates in InputBatch)
  Subsequent fire_batch calls    → CachedRequestData (updates existing state)
  Sequence exits batch           → finished_req_ids (removes from InputBatch)

This avoids the per-step overhead of adding/removing from vLLM's InputBatch
on every fire_batch call for continuing sequences.

Custom attention masks:
  BRLE-encoded masks from Pie's flattened_masks field are decoded and passed
  to FlashInfer via pie_custom_mask on the model runner. This enables
  mask_tokens(), mask_token_range(), and custom attention patterns (e.g.,
  attention-sink, windowed attention, jacobi decoding). Only applies during
  multi-token (prefill) steps; single-token decode is always causal.

Hybrid models (GDN/Mamba + attention, e.g., Qwen3.5):
  Custom attention masks apply ONLY to FlashInfer attention layers (~25% of
  layers in Qwen3.5). GDN/linear attention layers (~75%) process tokens via
  recurrent state updates — masking is inapplicable since the recurrence is
  causal by construction. Both backend types ignore the causal and custom_mask
  fields on CommonAttentionMetadata; FlashInfer derives its own causal flag
  from whether custom_mask is present.

  KV cache operations on hybrid models:
    - KV export/import (prefix_caching): Only attention-layer KV is exported.
      GDN state is managed by vLLM's MambaCacheManager and is NOT exported.
      On import, GDN layers recompute from scratch — functionally correct
      but slower than on pure-attention models.
    - Fork/shared prefix (prefix_tree): Attention KV pages are shared via
      PIE's page table duplication. GDN state is restored from block-boundary
      checkpoints (mamba_cache_mode="align") or recomputed ("none").
    - Speculative decoding: Each speculative path updates GDN state in-place.
      State cannot be partially rolled back. With mamba_cache_mode="align",
      vLLM checkpoints state at block boundaries for restoration.

Known limitations vs Pie's native runtime:
  - CMA-ES adapters: Not supported. set_adapter_seed(), initialize_adapter(),
    update_adapter() raise NotImplementedError. Use standard HF LoRA via
    upload_adapter_rpc() instead.
  - Input embeddings / multimodal: embed_image_rpc() preprocesses and stores
    image data, but full fire_batch integration requires Rust-side changes
    to pass input_embed_ptrs through BatchedForwardPassRequest.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RuntimeConfig
from .hf_chat_utils import CHATML_FALLBACK, sanitize_chat_template, get_stop_tokens_from_hf
from .vllm_batch_translator import PieVllmBatchTranslator
from .vllm_response_packager import ResponsePackager


@dataclass
class DecodedBatchArrays:
    """Decoded numpy arrays from a fire_batch kwargs dict."""
    token_ids: np.ndarray        # (total_tokens,) int64
    qo_indptr: np.ndarray        # (num_requests+1,) int32
    kv_page_indices: np.ndarray  # (total_pages,) int32
    kv_page_indptr: np.ndarray   # (num_requests+1,) int32
    kv_last_page_lens: np.ndarray  # (num_requests,) int32
    num_requests: int
    tokens_per_req: list[int]
    blocks_per_req: list[list[int]]
    seq_lens: np.ndarray         # (num_requests,) int32
    sampling_params_list: list[dict[str, Any]]
    adapter_indices: list[int | None]
    request_ids: list[str] | None = None
    is_new: list[bool] | None = None
from .vllm_sampling_bridge import PieVllmSamplingBridge
from .vllm_sequence_tracker import SequenceTracker
from .vllm_capture import get_capture as _get_capture

# vLLM is installed as a pip package (pip install vllm==0.17.0) with Pie
# integration patches applied to the installed site-packages.  No source
# checkout or sys.path manipulation is needed.

# ---------------------------------------------------------------------------
# Prometheus metrics for benchmarking (exposed on PIE_METRICS_PORT, default 8000)
#
# Emits the same vllm:* metrics as standalone vLLM so benchmarks can compare
# both backends using identical metric names.  Per-step metrics are observed
# in fire_batch(); per-request metrics are observed when a request finishes
# (tracked via _REQUEST_TIMING dict).
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Histogram, start_http_server as _prom_start_http

    # Per-step metrics (observed every fire_batch call, Pie-specific)
    _PIE_METRICS = {
        "model_forward": Histogram(
            "pie_vllm_model_forward_seconds",
            "Model forward pass time per batch",
        ),
        "batch_total": Histogram(
            "pie_vllm_batch_total_seconds",
            "Total fire_batch time per batch",
        ),
        "translation_overhead": Histogram(
            "pie_vllm_translation_overhead_seconds",
            "Translation overhead per batch",
        ),
    }

    # Per-request vllm:* metrics are provided by the shared monkey-patch
    # in _pie_step_metrics.py (applied to GPUWorker.execute_model via
    # scripts/patch-vllm-metrics.sh). That patch runs on BOTH standalone
    # vLLM and Pie, ensuring identical profiling points.

    _PROM_AVAILABLE = True
except ImportError:
    _PIE_METRICS = {}
    _PROM_AVAILABLE = False


class PieVllmRuntime:
    """vLLM-backed runtime implementing Pie's RPC interface.

    Drop-in replacement for pie_worker.runtime.Runtime when backend="vllm".
    Loads models via vLLM's GPUWorker and executes forward passes via
    GPUModelRunner.

    Args:
        config: Pie RuntimeConfig with backend="vllm".
        group_id: Data parallel group ID (for multi-GPU).
        process_groups: torch.distributed process groups from Pie.
        compute_process_groups: TP-only process groups.
        group_topology: List of rank lists per group.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        group_id: int = 0,
        process_groups: dict | None = None,
        compute_process_groups: dict | None = None,
        group_topology: list[list[int]] | None = None,
        tp_queue: "multiprocessing.Queue | None" = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.group_id = group_id
        self.kv_page_size = config.kv_page_size

        # Inter-process queue for TP leader→follower signaling.
        # Replaces NCCL broadcast_object_list to avoid timeout when the
        # engine is idle (NCCL has a 600s default timeout that kills both
        # workers if no collective runs within that window).
        self._tp_queue: "multiprocessing.Queue | None" = tp_queue

        # Bridge components (pure Python, no GPU needed)
        self.translator = PieVllmBatchTranslator()
        self.sampling_bridge = PieVllmSamplingBridge()

        # vLLM worker state -- populated by _init_vllm_worker()
        self.vllm_worker = None
        self.vllm_config = None
        self.num_gpu_blocks: int = 0
        self.num_kv_cache_groups: int = 1  # single KV group for standard models

        # Sequence lifecycle tracker (token history, active requests,
        # batch counter, req ID bookkeeping).  Re-created after
        # _init_vllm_worker() once num_kv_cache_groups is known.
        self._seq_tracker = SequenceTracker()

        # HF tokenizer for server-side chat template rendering.
        # Loaded lazily during handshake_rpc (needs snapshot_dir).
        self._hf_tokenizer = None

        # Embedding storage for multimodal support.
        # Key: embed pointer index → tensor of embeddings.
        # Populated by embed_image_rpc(), consumed during fire_batch
        # when input_embeddings are referenced.
        self._embed_storage: dict[int, torch.Tensor] = {}

        # Eagerly initialise the vLLM GPU worker (model load + KV cache
        # profiling).  This mirrors native Runtime, which does all heavy init
        # in __init__ so the worker signals ready only after GPU resources are
        # allocated.  The Rust IPC handshake has a short timeout, so init must
        # complete before the worker connects.
        self._init_vllm_worker()

        # Re-create tracker now that num_kv_cache_groups is known.
        self._seq_tracker = SequenceTracker(
            num_kv_cache_groups=self.num_kv_cache_groups,
        )

        # LoRA adapter registry for HF LoRA support.
        # Key: adapter pointer index → (name, path) pair.
        # Populated by upload_adapter_rpc(), consumed during fire_batch
        # when adapter_indices reference a registered adapter.
        self._adapter_registry: dict[int, tuple[str, str]] = {}
        self._adapter_counter: int = 0

        # Start Prometheus metrics server for benchmarking
        if _PROM_AVAILABLE:
            metrics_port = int(os.environ.get("PIE_METRICS_PORT", "8000"))
            try:
                _prom_start_http(metrics_port)
                print(f"[metrics] Prometheus server started on :{metrics_port}",
                      file=sys.stderr, flush=True)
            except OSError as e:
                print(f"[metrics] Could not start Prometheus server on :{metrics_port}: {e}",
                      file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Backward-compatible accessors (delegate to _seq_tracker)
    #
    # Tests and legacy code may access these attributes directly.
    # The _ensure_tracker() helper lazily creates the tracker when
    # __init__ was bypassed (e.g. tests using __new__).
    # ------------------------------------------------------------------

    def _ensure_tracker(self) -> SequenceTracker:
        """Return _seq_tracker, creating it if not yet initialised."""
        try:
            return self.__dict__["_seq_tracker"]
        except KeyError:
            tracker = SequenceTracker()
            self.__dict__["_seq_tracker"] = tracker
            return tracker

    @property
    def _batch_counter(self) -> int:
        return self._ensure_tracker().batch_counter

    @_batch_counter.setter
    def _batch_counter(self, value: int) -> None:
        self._ensure_tracker().batch_counter = value

    @property
    def _token_history(self) -> dict:
        return self._ensure_tracker().token_history

    @_token_history.setter
    def _token_history(self, value: dict) -> None:
        self._ensure_tracker()._token_history = value

    @property
    def _active_requests(self) -> dict:
        return self._ensure_tracker().active_requests

    @_active_requests.setter
    def _active_requests(self, value: dict) -> None:
        self._ensure_tracker()._active_requests = value

    @property
    def _last_batch_req_ids(self) -> list[str]:
        return self._ensure_tracker().last_batch_req_ids

    @_last_batch_req_ids.setter
    def _last_batch_req_ids(self, value: list[str]) -> None:
        self._ensure_tracker().last_batch_req_ids = value

    @property
    def _all_issued_req_ids(self) -> set[str]:
        return self._ensure_tracker().all_issued_req_ids

    @_all_issued_req_ids.setter
    def _all_issued_req_ids(self, value: set[str]) -> None:
        self._ensure_tracker()._all_issued_req_ids = value

    # ------------------------------------------------------------------
    # Worker Initialisation
    # ------------------------------------------------------------------

    def _init_vllm_worker(self) -> None:
        """Full 8-step vLLM GPUWorker initialisation.

        Steps:
          1. Build VllmConfig from Pie config
          2. Create Worker
          3. init_device()
          4. load_model()
          5. determine_available_memory()
          6. get_kv_cache_spec()
          7. Compute KV cache config and allocate
          8. compile_or_warm_up_model()
        """
        from .vllm_config_bridge import PieToVllmConfigBridge
        from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_configs,
            generate_scheduler_kv_cache_config,
        )

        # Step 1: Build vLLM config
        self.vllm_config = PieToVllmConfigBridge.build_vllm_config(self.config)

        # Pie's bridge is synchronous (calls execute_model then immediately
        # processes the result). vLLM nightly enables async scheduling by
        # default, which splits execute_model/sample_tokens into two phases
        # with GPU-side token caching that conflicts with pie_manages_tokens.
        # Force synchronous scheduling to avoid these issues.
        self.vllm_config.scheduler_config.async_scheduling = False

        # Step 2: Create worker
        tp_rank = self.config.rank
        tp_size = self.config.tensor_parallel_size

        # For TP>1, use shared master_port from manager.py so all workers
        # in the TP group can find each other for NCCL init.
        # For TP=1 (master_port=0), use a random port (backward compat).
        if getattr(self.config, 'master_port', 0) > 0:
            port = self.config.master_port
        else:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        self.vllm_worker = VllmGPUWorker(
            vllm_config=self.vllm_config,
            local_rank=tp_rank,
            rank=tp_rank,
            distributed_init_method=f"tcp://localhost:{port}",
            is_driver_worker=(tp_rank == 0),
        )

        # vLLM nightly requires set_current_vllm_config() for all worker
        # operations (init_device, load_model, KV cache init, warmup).
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(self.vllm_config):
            # Step 3: Init device (sets CUDA, inits distributed, creates model runner)
            self.vllm_worker.init_device()
            self._device = self.vllm_worker.device

            # Step 4: Load model weights
            self.vllm_worker.load_model()

            # Step 5: Profile available GPU memory
            available_memory: int = self.vllm_worker.determine_available_memory()

            # Step 6: Get KV cache spec
            kv_cache_spec = self.vllm_worker.get_kv_cache_spec()

            # Step 7: Compute KV cache config and allocate
            kv_cache_configs = get_kv_cache_configs(
                self.vllm_config,
                [kv_cache_spec],       # single-worker list
                [available_memory],    # single-worker list
            )

            scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                kv_cache_configs
            )
            self.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
            self.num_kv_cache_groups = len(scheduler_kv_cache_config.kv_cache_groups)

            # Upgrade mamba_cache_mode to "align" for hybrid models now that
            # we authoritatively know num_kv_cache_groups.  This is deferred
            # from build_vllm_config() because the bridge cannot know the
            # model architecture before vLLM loads it.  "align" enables
            # block-boundary state checkpointing for prefix caching and
            # speculative token rollback on GDN/Mamba layers.
            if self.num_kv_cache_groups > 1:
                self.vllm_config.cache_config.mamba_cache_mode = "align"
                print(
                    f"[INFO] Hybrid model detected ({self.num_kv_cache_groups} "
                    f"KV cache groups): mamba_cache_mode='align'",
                    file=sys.stderr, flush=True,
                )

            # For hybrid models (e.g., Qwen3.5 with DeltaNet + attention),
            # vLLM overrides block_size (e.g., 544 instead of 16). The bridge
            # must use the actual block size so the Rust scheduler allocates
            # the correct number of blocks per sequence.
            actual_block_size = self.vllm_config.cache_config.block_size
            if actual_block_size != self.kv_page_size:
                if os.environ.get("PIE_VLLM_DEBUG"):
                    print(
                        f"[DEBUG] vLLM overrode block_size: "
                        f"{self.kv_page_size} -> {actual_block_size} "
                        f"(hybrid model)", flush=True,
                    )
                self.kv_page_size = actual_block_size

            # Update vllm_config cache settings
            self.vllm_config.cache_config.num_gpu_blocks = self.num_gpu_blocks
            self.vllm_config.cache_config.num_cpu_blocks = 0

            # Allocate KV caches on the worker
            self.vllm_worker.initialize_from_config(kv_cache_configs[0])

            # Step 8: Warmup / compile
            self.vllm_worker.compile_or_warm_up_model()

        # Step 9: Wire Pie integration hooks on the model runner.
        # These patches (in gpu_model_runner.py) add attributes that are
        # called/checked during model execution:
        #   - pie_logit_hook: captures logits for distribution mode (type 0)
        #   - pie_manages_tokens: forces token_ids_cpu as source of truth
        #     instead of vLLM's cached prev_sampled_token_ids (GPU), which
        #     may be wrong when Pie controls token selection (flush,
        #     distribution mode, fork)
        #   - pie_hidden_state_hook: captures hidden states for output embeds
        model_runner = self.vllm_worker.model_runner
        if hasattr(model_runner, "pie_logit_hook"):
            model_runner.pie_logit_hook = self.sampling_bridge.logit_hook
        if hasattr(model_runner, "pie_manages_tokens"):
            model_runner.pie_manages_tokens = True

        # Hidden state capture for output embeddings.
        self._captured_sample_hidden_states = None
        self._captured_full_hidden_states = None

        def _hidden_state_hook(sample_hs, full_hs):
            self._captured_sample_hidden_states = sample_hs
            self._captured_full_hidden_states = full_hs

        if hasattr(model_runner, "pie_hidden_state_hook"):
            model_runner.pie_hidden_state_hook = _hidden_state_hook

    # ------------------------------------------------------------------
    # SchedulerOutput Construction (delegated to SequenceTracker)
    # ------------------------------------------------------------------

    def _build_scheduler_output(
        self,
        batch_id: int,
        token_ids: np.ndarray,
        qo_indptr: np.ndarray,
        tokens_per_req: list[int],
        blocks_per_req: list[list[int]],
        seq_lens: np.ndarray,
        sampling_params_list: list[dict[str, Any]],
        adapter_indices: list[int | None] | None = None,
        request_ids: list[str] | None = None,
        is_new: list[bool] | None = None,
    ):
        """Delegate to SequenceTracker.build_scheduler_output()."""
        assert request_ids is not None, "request_ids is required"
        assert is_new is not None, "is_new is required"
        return self._seq_tracker.build_scheduler_output(
            batch_id=batch_id,
            token_ids=token_ids,
            qo_indptr=qo_indptr,
            tokens_per_req=tokens_per_req,
            blocks_per_req=blocks_per_req,
            seq_lens=seq_lens,
            sampling_params_list=sampling_params_list,
            adapter_indices=adapter_indices,
            adapter_registry=self._adapter_registry,
            request_ids=request_ids,
            is_new=is_new,
        )

    # ------------------------------------------------------------------
    # Error Recovery (delegated to SequenceTracker)
    # ------------------------------------------------------------------

    def _recover_from_failed_batch(self, scheduler_output: Any) -> None:
        """Delegate to SequenceTracker.recover_from_failed_batch()."""
        self._seq_tracker.recover_from_failed_batch(
            scheduler_output,
            tp_queue=self._tp_queue,
            vllm_worker=self.vllm_worker,
            vllm_config=self.vllm_config,
        )

    # ------------------------------------------------------------------
    # Response Packaging
    # ------------------------------------------------------------------

    def _package_response(
        self,
        model_output: Any,
        pie_batch: dict[str, Any],
        num_requests: int,
        *,
        logits_expanded: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert vLLM ModelRunnerOutput to Pie's ForwardPassResponse format.

        Delegates to ResponsePackager (instantiated per call so tests that
        bypass __init__ and assign sampling_bridge directly still work).

        Args:
            model_output: ModelRunnerOutput from vLLM.
            pie_batch: Original Pie batch kwargs (for sampler metadata).
            num_requests: Number of requests in the batch.
            logits_expanded: If True, logits were already expanded for
                multi-position and should not be re-sliced.

        Returns:
            List of result dicts matching Pie's ForwardPassResponse format.
        """
        packager = ResponsePackager(self.sampling_bridge)
        return packager.package_response(
            model_output,
            pie_batch,
            num_requests,
            self._last_batch_req_ids,
            logits_expanded=logits_expanded,
        )

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _normalize_model_name(hf_repo: str) -> str:
        """Normalize HF repo to Pie-style model name.

        Strips the org prefix and lowercases:
            'Qwen/Qwen3-0.6B' -> 'qwen3-0.6b'
        """
        name = hf_repo.split("/")[-1] if "/" in hf_repo else hf_repo
        return name.lower()

    # ==================================================================
    # RPC Interface (matches pie_worker.runtime.Runtime)
    # ==================================================================

    def handshake_rpc(self, **kwargs: Any) -> dict:
        """Handle handshake RPC from Rust.

        Returns model metadata and resource capacity so Pie's Rust scheduler
        can allocate KV pages.  The response format matches the native
        runtime's handshake_rpc exactly.

        The vLLM worker is initialised eagerly in __init__, so num_gpu_blocks
        and other resource counts are already populated by the time this runs.
        """
        from . import hf_utils

        snapshot_dir = hf_utils.get_hf_snapshot_dir(self.config.hf_repo)
        tokenizer_info = hf_utils.load_hf_tokenizer(snapshot_dir)

        # Load HF tokenizer for server-side chat template rendering.
        if self._hf_tokenizer is None:
            from transformers import AutoTokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.config.hf_repo,
                trust_remote_code=getattr(self.config, 'trust_remote_code', False),
            )

        # Get stop tokens from HF tokenizer + GenerationConfig
        prompt_stop_tokens = get_stop_tokens_from_hf(self._hf_tokenizer, snapshot_dir)

        # Raw HF chat template (before sanitization) for in-process rendering.
        # The Rust runtime decides compatibility by attempting to render at startup.
        raw_chat_template = getattr(self._hf_tokenizer, "chat_template", "") or ""

        return {
            "version": "1.0.0",
            "model_name": self._normalize_model_name(self.config.hf_repo),
            "model_traits": [],
            "model_description": f"vLLM-backed {self.config.hf_repo}",
            "prompt_template": sanitize_chat_template(raw_chat_template),
            "prompt_template_type": "jinja2" if raw_chat_template else "none",
            "prompt_stop_tokens": prompt_stop_tokens,
            "kv_page_size": self.kv_page_size,
            "max_batch_tokens": self.config.max_batch_tokens or 10240,
            "max_batch_size": self.config.max_batch_size or 128,
            "resources": {
                # Block 0 is vLLM's null sentinel, so usable blocks are
                # 1..num_gpu_blocks-1.  Each Pie page expands to G vLLM
                # blocks, so max Pie pages = (num_gpu_blocks - 1) // G.
                0: (self.num_gpu_blocks - 1) // self.num_kv_cache_groups,  # KV pages
                1: self.config.max_num_embeds, # Embed slots
                2: self.config.max_num_adapters, # Adapter slots
            },
            "tokenizer_num_vocab": tokenizer_info["num_vocab"],
            "tokenizer_merge_table": tokenizer_info["merge_table"],
            "tokenizer_special_tokens": tokenizer_info["special_tokens"],
            "tokenizer_split_regex": tokenizer_info["split_regex"],
            "tokenizer_escape_non_printable": tokenizer_info["escape_non_printable"],
            "tokenizer_sentencepiece_space": tokenizer_info["sentencepiece_space"],
            # Raw HF chat template for in-process minijinja rendering.
            # The Rust runtime validates compatibility at startup by rendering
            # a synthetic message and comparing with Python RPC output.
            "chat_template": raw_chat_template,
            # Hybrid model capabilities — informational for inferlet adaptation.
            # has_ssm_layers is True for models with GDN/Mamba + attention layers
            # (e.g., Qwen3.5). Custom masks apply only to attention layers.
            "model_capabilities": {
                "has_ssm_layers": self.num_kv_cache_groups > 1,
                "kv_export_covers_all_layers": self.num_kv_cache_groups <= 1,
            },
        }

    def query_rpc(self, **kwargs: Any) -> dict:
        """Handle query RPC."""
        query = kwargs.get("query", "")
        value = "unknown query"
        if query == "ping":
            value = "pong"
        return {"value": value}

    def format_chat_rpc(self, **kwargs: Any) -> dict:
        """Render chat messages using HF tokenizer's chat template.

        Server-side rendering eliminates the need for local template engines
        (minijinja, @huggingface/jinja) in SDK inferlets and ensures the
        template always matches the model.

        Args (via kwargs):
            messages: list of dicts with 'role', 'content', and optional fields
                      (tool_calls, tool_call_id, etc.) in OpenAI format.
            add_generation_prompt: bool, whether to append generation prompt.

        Returns:
            dict with 'token_ids': list of int token IDs.
        """
        import json

        # IPC sends "messages_json" (Rust struct field name);
        # direct calls may use "messages".
        messages = kwargs.get("messages_json", kwargs.get("messages"))
        if messages is None:
            raise ValueError("format_chat_rpc requires 'messages_json' or 'messages'")
        # messages may arrive as a JSON string (from WIT) or list (direct call)
        if isinstance(messages, str):
            messages = json.loads(messages)
        add_generation_prompt = kwargs.get("add_generation_prompt", True)

        # Support wrapped payload: {"messages": [...], "chat_template_kwargs": {...}}
        # This allows the inferlet to pass enable_thinking and other kwargs without
        # changing the WIT interface. Backward-compatible: plain list still works.
        explicit_kwargs: dict[str, Any] = {}
        if isinstance(messages, dict):
            explicit_kwargs = messages.get("chat_template_kwargs") or {}
            # Unwrap nested enable_thinking from ChatTemplateKwargs struct
            if isinstance(explicit_kwargs, dict) and "enable_thinking" in explicit_kwargs:
                pass  # already flat
            messages = messages.get("messages", [])

        # Extract tools for tool calling support
        tools_json = kwargs.get("tools_json")
        if tools_json:
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
        else:
            tools = None

        if self._hf_tokenizer is None:
            raise RuntimeError(
                "HF tokenizer not loaded. Call handshake_rpc first."
            )

        # Parse /no_think and /think suffixes from the last user message.
        # Qwen3.5 models require enable_thinking=False passed via
        # chat_template_kwargs rather than inline prompt suffixes.
        chat_template_kwargs: dict[str, Any] = {}
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    stripped = content.rstrip()
                    if stripped.endswith(" /no_think") or stripped == "/no_think":
                        msg["content"] = stripped[: -len("/no_think")].rstrip()
                        chat_template_kwargs["enable_thinking"] = False
                    elif stripped.endswith(" /think") or stripped == "/think":
                        msg["content"] = stripped[: -len("/think")].rstrip()
                        chat_template_kwargs["enable_thinking"] = True
                break

        # Merge explicit kwargs from wrapped payload (inferlet-provided).
        # Explicit kwargs take precedence over /think suffix parsing.
        if explicit_kwargs:
            chat_template_kwargs.update(explicit_kwargs)

        if tools:
            chat_template_kwargs["tools"] = tools

        result = self._hf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **chat_template_kwargs,
        )
        # transformers>=5 returns BatchEncoding (dict-like); extract input_ids.
        if hasattr(result, "input_ids"):
            token_ids = list(result.input_ids)
        else:
            token_ids = list(result)

        # Strip leading BOS token — the SDK manages sequence-level BOS via
        # begin_of_sequence state. Matches old behavior where local rendering
        # never emitted BOS (bos_token was not in template context).
        bos_id = getattr(self._hf_tokenizer, "bos_token_id", None)
        if bos_id is not None and token_ids and token_ids[0] == bos_id:
            token_ids = token_ids[1:]

        return {"token_ids": token_ids}

    def _decode_brle_mask(
        self,
        kwargs: dict[str, Any],
        qo_indptr: np.ndarray,
        tokens_per_req: np.ndarray,
        seq_lens: np.ndarray,
    ) -> torch.Tensor | None:
        """Decode BRLE masks from Pie's batch fields into a FlashInfer custom_mask.

        Reuses the same decode_brle_batch function as Pie's native runtime.
        The resulting flat boolean tensor matches FlashInfer's custom_mask
        format: sum(q_len[i] * kv_len[i]) elements in row-major order.

        Returns None if masks are trivially causal (no custom masking needed).
        """
        decode = PieVllmBatchTranslator.decode_binary_array

        flattened_masks_u32 = decode(
            kwargs["flattened_masks"], np.uint32
        ).astype(np.int32)
        mask_indptr = decode(kwargs["mask_indptr"], np.uint32).astype(np.int32)

        # If masks are empty, no custom masking
        if len(flattened_masks_u32) == 0:
            return None

        num_requests = len(tokens_per_req)
        total_tokens = int(np.sum(tokens_per_req))

        # Compute position IDs (same formula as batching.py)
        context_lens = seq_lens - tokens_per_req
        global_indices = np.arange(total_tokens, dtype=np.int32)
        request_starts = np.repeat(qo_indptr[:-1], tokens_per_req)
        request_contexts = np.repeat(context_lens, tokens_per_req)
        position_ids = global_indices - request_starts + request_contexts

        # Compute cumulative bit offsets per token
        all_seq_lens = np.repeat(seq_lens, tokens_per_req)
        token_acc_seq_lens = np.zeros(total_tokens + 1, dtype=np.int32)
        np.cumsum(all_seq_lens, out=token_acc_seq_lens[1:])

        # Decode BRLE → flat boolean array
        from .batching import decode_brle_batch
        mask_np = decode_brle_batch(
            flattened_masks_u32, mask_indptr, position_ids, token_acc_seq_lens
        )

        # Check if mask is trivially causal (all True up to position+1)
        # by checking if the total True count equals sum(position_id + 1).
        # If so, skip custom mask (FlashInfer's causal mode is faster).
        expected_true_count = int(np.sum(position_ids + 1))
        actual_true_count = int(np.sum(mask_np))
        if actual_true_count == expected_true_count:
            return None

        return torch.as_tensor(mask_np, device=self._device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Decomposed fire_batch helpers (for step-level scheduler)
    # ------------------------------------------------------------------

    @staticmethod
    def decode_batch_arrays(kwargs: dict, kv_page_size: int) -> DecodedBatchArrays:
        """Decode raw bytes from fire_batch kwargs into numpy arrays.

        Pure function — no runtime state needed.  Used by both fire_batch
        (backward compat) and the step-level scheduler in manager.py.
        """
        decode = PieVllmBatchTranslator.decode_binary_array

        token_ids = decode(kwargs["token_ids"], np.uint32).astype(np.int64)
        qo_indptr = decode(kwargs["qo_indptr"], np.uint32).astype(np.int32)
        kv_page_indices = decode(kwargs["kv_page_indices"], np.uint32).astype(np.int32)
        kv_page_indptr = decode(kwargs["kv_page_indptr"], np.uint32).astype(np.int32)
        kv_last_page_lens = decode(kwargs["kv_last_page_lens"], np.uint32).astype(np.int32)

        num_requests = len(qo_indptr) - 1
        tokens_per_req = PieVllmBatchTranslator.tokens_per_request(qo_indptr)
        blocks_per_req = PieVllmBatchTranslator.pages_to_blocks(
            kv_page_indices, kv_page_indptr
        )
        seq_lens = PieVllmBatchTranslator.compute_seq_lens(
            kv_page_indptr, kv_last_page_lens, kv_page_size
        )
        sampling_params_list = PieVllmBatchTranslator.extract_sampling_params(
            kwargs, num_requests
        )
        adapter_indices = kwargs.get("adapter_indices", [None] * num_requests)
        request_ids = kwargs.get("request_ids", None)
        is_new = kwargs.get("is_new", None)

        return DecodedBatchArrays(
            token_ids=token_ids,
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            num_requests=num_requests,
            tokens_per_req=tokens_per_req,
            blocks_per_req=blocks_per_req,
            seq_lens=seq_lens,
            sampling_params_list=sampling_params_list,
            adapter_indices=adapter_indices,
            request_ids=request_ids,
            is_new=is_new,
        )

    def prepare_step(
        self,
        arrays: DecodedBatchArrays,
        kwargs: dict,
    ) -> Any:
        """Build SchedulerOutput and configure model runner for one GPU step.

        Handles: batch counter, _build_scheduler_output, BRLE mask decode,
        multi-position logits, TP queue.  Returns the SchedulerOutput.
        """
        batch_id = self._batch_counter
        self._batch_counter += 1

        if os.environ.get("PIE_VLLM_DEBUG"):
            print(f"[DEBUG] prepare_step #{batch_id}: "
                  f"num_reqs={arrays.num_requests}, "
                  f"tokens_per_req={arrays.tokens_per_req}, "
                  f"seq_lens={arrays.seq_lens.tolist()[:10]}",
                  file=sys.stderr, flush=True)

        scheduler_output = self._build_scheduler_output(
            batch_id=batch_id,
            token_ids=arrays.token_ids,
            qo_indptr=arrays.qo_indptr,
            tokens_per_req=arrays.tokens_per_req,
            blocks_per_req=arrays.blocks_per_req,
            seq_lens=arrays.seq_lens,
            sampling_params_list=arrays.sampling_params_list,
            adapter_indices=arrays.adapter_indices,
            request_ids=arrays.request_ids,
            is_new=arrays.is_new,
        )

        # BRLE masks (only for multi-token prefill steps)
        single_token_mode = kwargs.get("single_token_mode", True)
        if not single_token_mode and "flattened_masks" in kwargs:
            custom_mask = self._decode_brle_mask(
                kwargs, arrays.qo_indptr, arrays.tokens_per_req, arrays.seq_lens
            )
            if custom_mask is not None:
                self.vllm_worker.model_runner.pie_custom_mask = custom_mask

        # Multi-position logits
        decode = PieVllmBatchTranslator.decode_binary_array
        request_num_samplers = decode(
            kwargs.get("request_num_samplers", b""), np.uint32
        ).astype(np.int32)
        total_slots = int(np.sum(request_num_samplers))
        num_generate = int(np.sum(request_num_samplers > 0))
        if total_slots > num_generate:
            total_tokens = int(np.sum(arrays.tokens_per_req))
            self.vllm_worker.model_runner.pie_logits_indices = torch.arange(
                total_tokens, device=self._device
            )
        else:
            self.vllm_worker.model_runner.pie_logits_indices = None

        # TP queue (for non-leader workers)
        if self._tp_queue is not None:
            mr = self.vllm_worker.model_runner
            scheduler_output._pie_custom_mask = getattr(mr, "pie_custom_mask", None)
            scheduler_output._pie_logits_indices = getattr(mr, "pie_logits_indices", None)
            self._tp_queue.put(scheduler_output)

        return scheduler_output

    def execute_step(self, scheduler_output: Any) -> Any:
        """Run one GPU step.  Returns ModelRunnerOutput.

        Handles set_current_vllm_config, execute_model, async output unwrap.
        On error, calls _recover_from_failed_batch and re-raises.
        """
        from vllm.config import set_current_vllm_config

        # Wireshark-mode capture: dump SchedulerOutput before execute_model
        _cap = os.environ.get("PIE_CAPTURE")
        if _cap:
            from pie_worker.vllm_capture import capture_scheduler_output
            capture_scheduler_output(scheduler_output, f"batch-{self._batch_counter-1}", _cap)

        try:
            with set_current_vllm_config(self.vllm_config):
                result = self.vllm_worker.execute_model(scheduler_output)

                from vllm.v1.outputs import AsyncModelRunnerOutput
                if isinstance(result, AsyncModelRunnerOutput):
                    result = result.get_output()
                elif result is None:
                    result = self.vllm_worker.sample_tokens(None)
                if hasattr(result, 'get_output'):
                    result = result.get_output()

                if _cap:
                    from pie_worker.vllm_capture import capture_model_output
                    capture_model_output(result, self._seq_tracker.last_batch_req_ids,
                                        f"batch-{self._batch_counter-1}", _cap)

        except Exception as e:
            import traceback
            print(f"[ERROR] execute_step failed: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self.vllm_worker.model_runner.pie_logits_indices = None
            self.vllm_worker.model_runner.pie_custom_mask = None
            self._recover_from_failed_batch(scheduler_output)
            raise

        return result

    def extract_sampled_tokens(self, model_output: Any) -> list[int]:
        """Extract per-request sampled token IDs from ModelRunnerOutput.

        Returns tokens in the order of _last_batch_req_ids (set by
        prepare_step → _build_scheduler_output).

        Handles both scalar (single-step) and list (multi-step) sampled values.
        """
        sampled = model_output.sampled_token_ids
        req_id_to_idx = model_output.req_id_to_index
        tokens = []
        for req_id in self._last_batch_req_ids:
            idx = req_id_to_idx.get(req_id)
            if idx is not None:
                val = sampled[idx]
                # vLLM may return a list per request (e.g., multi-step);
                # take the last token (most recent decode step).
                if isinstance(val, (list, tuple)):
                    tokens.append(int(val[-1]) if val else 0)
                else:
                    tokens.append(int(val))
            else:
                tokens.append(0)  # flush request — no token
        return tokens

    @torch.inference_mode()
    def fire_batch(self, _drain_fn=None, _merge_fn=None, _counts_fn=None,
                   **kwargs: Any) -> dict:
        """Execute a batched forward pass via vLLM's Worker.

        When max_decode_steps > 1, loops internally for multi-step decode.
        Between steps, calls _drain_fn to get new requests from the
        side-channel, merges them via _merge_fn, and processes the combined
        batch. This enables continuous batching at the Python level.

        Args:
            _drain_fn: Callable returning list[(req_id, msgpack_bytes)] of
                new fire_batch requests from the side-channel. None to disable.
            _merge_fn: merge_fire_batch_kwargs from batch_merger.
            _counts_fn: compute_batch_generate_counts from batch_merger.
            **kwargs: Batch fields from Rust IPC (msgpack-decoded).

        This is the critical path:
          1. Decode batch arrays from msgpack bytes
          2. Build per-request data
          3. Construct SchedulerOutput
          4. Call worker.execute_model()
          5. Handle two-phase: if None, call worker.sample_tokens(None)
          6. Package response
          7. Track request IDs for next call's finished_req_ids

        Args:
            **kwargs: Batch fields from Rust IPC (msgpack-decoded).

        Returns:
            Dict with 'results' list matching ForwardPassResponse format.
        """
        t_start = time.perf_counter()

        # --- 0. Process explicit freed_block_ids from Rust ResourceManager ---
        freed_raw = kwargs.pop("freed_block_ids", None)
        if freed_raw is not None and len(freed_raw) > 0:
            freed_ids = set(np.frombuffer(freed_raw, dtype=np.uint32).tolist())
            self._seq_tracker.finish_by_block_ids(freed_ids)

        # --- 1. Decode batch metadata ---
        arrays = self.decode_batch_arrays(kwargs, self.kv_page_size)

        t_translate = time.perf_counter()

        num_requests = arrays.num_requests

        # --- 2-3. Build SchedulerOutput + configure model runner ---
        try:
            scheduler_output = self.prepare_step(arrays, kwargs)
            if os.environ.get("PIE_MS_DEBUG"):
                _so = scheduler_output
                _nr = _so.scheduled_new_reqs
                _cr = _so.scheduled_cached_reqs
                print(f"[MS-DEBUG] step=0 new_reqs={len(_nr)} "
                      f"cached_req_ids={getattr(_cr, 'req_ids', [])} "
                      f"finished={_so.finished_req_ids} "
                      f"num_sched_tokens={_so.num_scheduled_tokens}",
                      file=sys.stderr, flush=True)
                for _nri, _nrd in enumerate(_nr):
                    print(f"  NEW[{_nri}] req_id={_nrd.req_id} "
                          f"prompt_len={len(_nrd.prompt_token_ids)} "
                          f"num_computed={_nrd.num_computed_tokens} "
                          f"blocks={[b[:5] for b in _nrd.block_ids]}...",
                          file=sys.stderr, flush=True)
        except Exception as e:
            import traceback
            print(f"[ERROR] prepare_step failed: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

        # --- Capture input (step 0) ---
        _cap = _get_capture()
        _cap_bid = self._batch_counter - 1  # batch_id from prepare_step
        if _cap.enabled:
            _cap.capture_step(
                batch_id=_cap_bid, step_index=0,
                scheduler_output=scheduler_output,
                tracker=self._seq_tracker, arrays=arrays,
            )

        # --- 4. Execute model ---
        t_model_start = time.perf_counter()
        result = self.execute_step(scheduler_output)
        t_model_end = time.perf_counter()

        # --- 5b. Expand logits for multi-position requests ---
        # When request_num_samplers[i] > 1 (cacheback/speculative), compute
        # logits at additional positions from full_hidden_states. Must happen
        # before _package_response consumes captured logits.
        logits_expanded = self._expand_logits_for_multi_position(
            kwargs, arrays.qo_indptr, num_requests
        )

        # --- 6. Package response ---
        try:
            results = self._package_response(
                result, kwargs, num_requests,
                logits_expanded=logits_expanded,
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] _package_response failed: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

        # --- Capture output (step 0) ---
        if _cap.enabled:
            _cap.capture_output(
                batch_id=_cap_bid, step_index=0,
                results=results, model_output=result,
            )

        # --- 7. Store output embeddings (if requested) ---
        output_embed_ptrs = kwargs.get("output_embed_ptrs", [])
        output_embed_indices = kwargs.get("output_embed_indices", [])
        if self._captured_sample_hidden_states is not None and output_embed_ptrs:
            self._store_output_embeddings(
                output_embed_ptrs, output_embed_indices,
                self._captured_sample_hidden_states,
                self._captured_full_hidden_states,
                arrays.qo_indptr, num_requests,
            )
        self._captured_sample_hidden_states = None
        self._captured_full_hidden_states = None

        # --- Multi-step decode loop ---
        max_steps = kwargs.get("max_decode_steps", 1)
        if os.environ.get("PIE_TRACE") and max_steps > 1:
            print(f"[PIE-TRACE] {time.clock_gettime_ns(time.CLOCK_MONOTONIC)} py.fb_multistep {self._batch_counter} max_decode_steps={max_steps}",
                  file=sys.stderr, flush=True)
        if max_steps > 1:
            all_step_results = [results]  # results from step 0
            _orig_num_reqs = len(results)  # track original request count
            _all_merged_req_ids = []  # (req_id, split_result) for merged requests
            _all_merged_counts = []
            for step_i in range(1, max_steps):
                # Extract sampled tokens from previous step
                prev_tokens = []
                for r in results:
                    toks = r.get("tokens", [])
                    if len(toks) > 0:
                        prev_tokens.append(toks[0])
                if not prev_tokens:
                    break  # no tokens (all flush or EOS)

                # Build continuation by updating kwargs and re-decoding.
                # This uses the SAME path as a fresh fire_batch, ensuring
                # arrays are built identically to what Rust would provide.
                new_kv_lens = arrays.kv_last_page_lens.copy()
                new_kv_lens += 1
                # Handle page boundary: wrap kv_last_page_lens when crossing.
                # Pages are pre-allocated by the SDK (extra_kv_tokens), so the
                # next page exists. If no pre-allocation, stop instead.
                if self.kv_page_size > 0:
                    _crossing = new_kv_lens > self.kv_page_size
                    if np.any(_crossing):
                        # Check if the next page exists for each crossing request
                        _can_cross = True
                        for _ri in range(len(new_kv_lens)):
                            if _crossing[_ri]:
                                _pages_needed = int(arrays.kv_page_indptr[_ri + 1] - arrays.kv_page_indptr[_ri])
                                _pages_used = (int(arrays.seq_lens[_ri]) + self.kv_page_size - 1) // self.kv_page_size
                                if _pages_used >= _pages_needed:
                                    _can_cross = False  # no more pages
                                    break
                        if not _can_cross:
                            break
                        new_kv_lens[_crossing] = 1  # wrap to start of next page
                kwargs["token_ids"] = np.array(prev_tokens, dtype=np.uint32).tobytes()
                kwargs["qo_indptr"] = np.arange(len(prev_tokens) + 1, dtype=np.uint32).tobytes()
                kwargs["kv_last_page_lens"] = new_kv_lens.astype(np.uint32).tobytes()
                kwargs["single_token_mode"] = True
                kwargs["flattened_masks"] = b""
                kwargs["mask_indptr"] = np.zeros(len(prev_tokens) + 1, dtype=np.uint32).tobytes()
                kwargs["output_token_indptr"] = np.arange(len(prev_tokens) + 1, dtype=np.uint32).tobytes()
                # Continuing requests are no longer new after first fire
                if "is_new" in kwargs and kwargs["is_new"]:
                    kwargs["is_new"] = [False] * len(kwargs["is_new"])
                # Re-decode kwargs through the standard path
                arrays = self.decode_batch_arrays(kwargs, self.kv_page_size)
                num_requests = arrays.num_requests

                # Between-step merge: drain side-channel for new requests
                # that arrived while GPU was busy, merge into current batch.
                _merged_req_ids = None
                _merged_counts = None
                if _drain_fn and _merge_fn:
                    import msgpack as _msgpack
                    _new_pending = _drain_fn()
                    if _new_pending:
                        _new_items = [(_rid, _msgpack.unpackb(_raw))
                                      for _rid, _raw in _new_pending]
                        _new_batches = [_kw for _, _kw in _new_items]
                        _merged_req_ids = [_rid for _rid, _ in _new_items]
                        if _counts_fn:
                            _merged_counts = _counts_fn(_new_batches)
                        _all_batches = [kwargs] + _new_batches
                        kwargs = _merge_fn(_all_batches)
                        # Process freed_block_ids from merged batches BEFORE
                        # prepare_step, so the tracker finishes sequences whose
                        # blocks are about to be reused by the new requests.
                        _merged_freed = kwargs.pop("freed_block_ids", None)
                        if _merged_freed is not None and len(_merged_freed) > 0:
                            _freed_set = set(np.frombuffer(
                                _merged_freed, dtype=np.uint32
                            ).tolist())
                            self._seq_tracker.finish_by_block_ids(_freed_set)
                        arrays = self.decode_batch_arrays(kwargs, self.kv_page_size)
                        num_requests = arrays.num_requests
                        _all_merged_req_ids.extend(_merged_req_ids)
                        if _merged_counts:
                            _all_merged_counts.extend(_merged_counts)
                        else:
                            _all_merged_counts.extend([1] * len(_merged_req_ids))
                        if os.environ.get("PIE_TRACE"):
                            print(f"[PIE-TRACE] {time.clock_gettime_ns(time.CLOCK_MONOTONIC)} py.ms_merge {self._batch_counter} step={step_i} new_reqs={len(_new_pending)}",
                                  file=sys.stderr, flush=True)

                if os.environ.get("PIE_TRACE"):
                    print(f"[PIE-TRACE] {time.clock_gettime_ns(time.CLOCK_MONOTONIC)} py.ms_step {self._batch_counter} step={step_i} reqs={arrays.num_requests}",
                          file=sys.stderr, flush=True)

                try:
                    scheduler_output = self.prepare_step(arrays, kwargs)
                    if _cap.enabled:
                        _cap.capture_step(
                            batch_id=_cap_bid, step_index=step_i,
                            scheduler_output=scheduler_output,
                            tracker=self._seq_tracker, arrays=arrays,
                            is_merged=bool(_merged_req_ids),
                            merged_req_count=len(_merged_req_ids) if _merged_req_ids else 0,
                        )
                    result = self.execute_step(scheduler_output)
                    logits_expanded = self._expand_logits_for_multi_position(
                        kwargs, arrays.qo_indptr, arrays.num_requests
                    )
                    results = self._package_response(
                        result, kwargs, arrays.num_requests,
                        logits_expanded=logits_expanded,
                    )
                    if _cap.enabled:
                        _cap.capture_output(
                            batch_id=_cap_bid, step_index=step_i,
                            results=results, model_output=result,
                        )
                except Exception as e:
                    print(f"[MS-ERROR] step {step_i}: {e}", file=sys.stderr, flush=True)
                    import traceback; traceback.print_exc(file=sys.stderr)
                    break

                self._captured_sample_hidden_states = None
                self._captured_full_hidden_states = None
                all_step_results.append(results)

            # Combine all steps: accumulate tokens for ORIGINAL requests only.
            # Merged (new) requests get their tokens from the steps they joined.
            if len(all_step_results) > 1:
                # Original requests: always at indices 0.._orig_num_reqs-1
                combined = []
                for req_idx in range(_orig_num_reqs):
                    all_tokens, all_dists = [], []
                    for sr in all_step_results:
                        if req_idx < len(sr):
                            all_tokens.extend(sr[req_idx].get("tokens", []))
                            all_dists.extend(sr[req_idx].get("dists", []))
                    combined.append({"tokens": all_tokens, "dists": all_dists})
                results = combined

                if os.environ.get("PIE_TRACE"):
                    print(f"[PIE-TRACE] {time.clock_gettime_ns(time.CLOCK_MONOTONIC)} py.ms_done {self._batch_counter} steps={len(all_step_results)} merged={len(_all_merged_req_ids)}",
                          file=sys.stderr, flush=True)

        t_end = time.perf_counter()

        metrics = {
            "model_forward_ms": (t_model_end - t_model_start) * 1000,
            "translation_overhead_ms": (t_translate - t_start) * 1000,
            "total_ms": (t_end - t_start) * 1000,
            "batch_size": num_requests,
            "batch_tokens": int(np.sum(arrays.tokens_per_req)),
        }

        # Update Prometheus histograms (Pie-specific per-step metrics)
        if _PIE_METRICS:
            _PIE_METRICS["model_forward"].observe(t_model_end - t_model_start)
            _PIE_METRICS["batch_total"].observe(t_end - t_start)
            _PIE_METRICS["translation_overhead"].observe(t_translate - t_start)

        # Per-step prefill/decode metrics come from the shared monkey-patch
        # in _pie_step_metrics.py (applied to GPUWorker.execute_model).
        # That patch classifies steps as prefill or decode and emits:
        #   pie_step_prefill_seconds, pie_step_decode_seconds, pie_step_total_seconds
        # Both standalone vLLM and Pie emit from the same code point.

        if os.environ.get("PIE_VLLM_TIMING"):
            # Use CLOCK_MONOTONIC for cross-process correlation with Rust IPC timing
            wall_end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            wall_start_ns = int(t_start * 1e9)
            print(f"[TIMING] batch#{self._batch_counter - 1}: "
                  f"total={metrics['total_ms']:.1f}ms "
                  f"model={metrics['model_forward_ms']:.1f}ms "
                  f"xlate={metrics['translation_overhead_ms']:.2f}ms "
                  f"pkg={metrics['total_ms'] - metrics['model_forward_ms'] - metrics['translation_overhead_ms']:.2f}ms "
                  f"reqs={num_requests} toks={metrics['batch_tokens']} "
                  f"ws={wall_start_ns} we={wall_end_ns}",
                  file=sys.stderr, flush=True)

        ret = {"results": results, "metrics": metrics}
        # If requests were merged during multi-step, split their responses.
        # If merge failed, return raw tuples for manager.py to re-process.
        if max_steps > 1 and '_all_merged_req_ids' in dir() and _all_merged_req_ids:
            # Check if items are merged (int req_ids from side-channel) or
            # raw (tuple from failed merge).  Side-channel request IDs are
            # integers (int.from_bytes), NOT strings.
            _merged_resps = []
            _queued_raw = []
            for item in _all_merged_req_ids:
                if isinstance(item, int):
                    _merged_resps.append(item)
                else:
                    _queued_raw.append(item)
            # For successfully merged requests: their results are in the
            # combined results at indices >= _orig_num_reqs
            if _merged_resps:
                _split_results = []
                _offset = _orig_num_reqs
                for _mr_idx, _rid in enumerate(_merged_resps):
                    _cnt = _all_merged_counts[_mr_idx] if _mr_idx < len(_all_merged_counts) else 1
                    # Build per-request result entries (Rust expects one per
                    # generate request in the batch).
                    _mr_results = []
                    for _ri in range(_offset, _offset + _cnt):
                        _ri_tokens, _ri_dists = [], []
                        for sr in all_step_results:
                            if _ri < len(sr):
                                _ri_tokens.extend(sr[_ri].get("tokens", []))
                                _ri_dists.extend(sr[_ri].get("dists", []))
                        _mr_results.append({"tokens": _ri_tokens, "dists": _ri_dists})
                    _split_results.append({
                        "req_id": _rid,
                        "result": {"results": _mr_results, "metrics": {}},
                    })
                    _offset += _cnt
                ret["_merged_responses"] = _split_results
            if _queued_raw:
                ret["_queued_requests"] = _queued_raw
        return ret

    def _store_output_embeddings(
        self,
        output_embed_ptrs: list[list[int]],
        output_embed_indices: list[list[int]],
        sample_hidden_states: torch.Tensor,
        full_hidden_states: torch.Tensor | None,
        qo_indptr: np.ndarray,
        num_requests: int,
    ) -> None:
        """Store hidden states at embed pointer slots for later retrieval.

        Two modes:
          - Single-position (decode): output_embed_indices[i] is empty →
            use sample_hidden_states[i] (one hidden state per request).
          - Multi-position (prefill): output_embed_indices[i] has explicit
            positions → use full_hidden_states[qo_indptr[i] + local_idx].

        Args:
            output_embed_ptrs: Per-request list of embed pointer IDs to store at.
            output_embed_indices: Per-request list of token positions to capture.
            sample_hidden_states: (num_positions, hidden_dim) from sample positions.
            full_hidden_states: (num_tokens, hidden_dim) full sequence states.
            qo_indptr: CSR pointers for query tokens per request.
            num_requests: Number of requests in the batch.
        """
        for i in range(min(num_requests, len(output_embed_ptrs))):
            ptrs = output_embed_ptrs[i]
            if not ptrs:
                continue

            indices = output_embed_indices[i] if i < len(output_embed_indices) else []

            if not indices:
                # Single-position fast path: use sample_hidden_states[i]
                if i < sample_hidden_states.shape[0]:
                    hs = sample_hidden_states[i].detach().cpu()
                    for ptr in ptrs:
                        self._embed_storage[ptr] = hs
            else:
                # Multi-position: use full_hidden_states at specific positions
                if full_hidden_states is None:
                    continue
                base = int(qo_indptr[i]) if i < len(qo_indptr) - 1 else 0
                for j, ptr in enumerate(ptrs):
                    if j < len(indices):
                        local_idx = indices[j]
                        global_idx = base + local_idx
                        if global_idx < full_hidden_states.shape[0]:
                            self._embed_storage[ptr] = (
                                full_hidden_states[global_idx].detach().cpu()
                            )

    def _expand_logits_for_multi_position(
        self,
        kwargs: dict[str, Any],
        qo_indptr: np.ndarray,
        num_requests: int,
    ) -> bool:
        """Expand captured logits for multi-position requests.

        Delegates to ResponsePackager. See
        ResponsePackager.expand_logits_for_multi_position for details.

        Returns:
            True if logits were expanded (caller must skip re-slicing),
            False if no expansion was needed.
        """
        packager = ResponsePackager(self.sampling_bridge)
        return packager.expand_logits_for_multi_position(
            kwargs,
            qo_indptr,
            num_requests,
            captured_full_hidden_states=self._captured_full_hidden_states,
            vllm_worker=self.vllm_worker,
            vllm_config=self.vllm_config,
        )

    # ==================================================================
    # Unsupported RPCs (stubs for interface compatibility)
    # ==================================================================

    def embed_image_rpc(self, **kwargs: Any) -> dict:
        """Preprocess an image and store it for multimodal forward passes.

        Decodes the image blob, stores it at the requested pointer slots,
        and returns the number of embedding slots used.  The stored data
        will be attached as ``mm_features`` on ``NewRequestData`` once the
        Rust batcher passes ``input_embed_ptrs`` through the batch path.

        Note: Full multimodal integration in ``fire_batch`` requires Rust-side
        changes to include ``input_embed_ptrs`` in ``BatchedForwardPassRequest``.
        Currently, only the image preprocessing and storage step works.

        Args:
            **kwargs: Must include:
                embed_ptrs: list[int] — pointer slots to store data at.
                image_blob: bytes — raw image data (JPEG/PNG).
                position_offset: int — position offset for the embeddings.

        Returns:
            Dict with 'num_embed_tokens' (= len(embed_ptrs)).
        """
        embed_ptrs = kwargs["embed_ptrs"]
        image_blob = kwargs["image_blob"]
        position_offset = kwargs.get("position_offset", 0)

        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_blob))

        # Store preprocessed image data at each pointer slot.
        # When Rust-side batching includes input_embed_ptrs, the bridge
        # will attach these as vLLM mm_features in _build_scheduler_output().
        for ptr in embed_ptrs:
            self._embed_storage[ptr] = {
                "image": image,
                "position_offset": position_offset,
            }

        return {"num_embed_tokens": len(embed_ptrs)}

    def initialize_adapter_rpc(self, **kwargs: Any) -> None:
        """CMA-ES adapter init — not supported with vLLM backend.

        CMA-ES / Zero-Order adapters use Pie's native runtime with custom
        rand_mv noise perturbation.  This is fundamentally incompatible
        with vLLM's LoRA approach.
        """
        raise NotImplementedError(
            "CMA-ES initialize_adapter not supported with vLLM backend. "
            "Use upload_adapter_rpc for standard HF LoRA adapters."
        )

    def update_adapter_rpc(self, **kwargs: Any) -> None:
        """CMA-ES adapter update — not supported with vLLM backend."""
        raise NotImplementedError(
            "CMA-ES update_adapter not supported with vLLM backend"
        )

    def upload_adapter_rpc(self, **kwargs: Any) -> dict:
        """Register a HF LoRA adapter for per-request use.

        Writes the adapter data to disk in HF LoRA format and registers
        it in the adapter registry.  Subsequent ``fire_batch`` calls with
        matching ``adapter_indices`` will attach a ``LoRARequest`` to the
        vLLM ``NewRequestData``.

        Args:
            **kwargs: Must include:
                adapter_ptr: int — Pie adapter pointer ID.
                name: str — Human-readable adapter name.
                adapter_data: bytes — Serialized HF LoRA weights.

        Returns:
            Dict with 'adapter_ptr' confirming registration.
        """
        adapter_ptr = kwargs["adapter_ptr"]
        name = kwargs["name"]
        adapter_data = kwargs["adapter_data"]

        # Write adapter to disk under adapter_path (sanitize name)
        base = Path(self.config.adapter_path).resolve()
        adapter_dir = (base / name).resolve()
        if not str(adapter_dir).startswith(str(base) + "/") and adapter_dir != base:
            raise ValueError(f"Invalid adapter name (path traversal): {name!r}")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if isinstance(adapter_data, (bytes, bytearray)):
            adapter_file.write_bytes(adapter_data)
        else:
            adapter_file.write_bytes(bytes(adapter_data))

        # Register in adapter registry
        self._adapter_registry[adapter_ptr] = (name, str(adapter_dir))

        return {"adapter_ptr": adapter_ptr}

    def download_adapter_rpc(self, **kwargs: Any) -> dict:
        """Download a previously uploaded adapter's data.

        Args:
            **kwargs: Must include:
                adapter_ptr: int — Pie adapter pointer ID.
                name: str — Adapter name.

        Returns:
            Dict with 'adapter_data' bytes.
        """
        adapter_ptr = kwargs["adapter_ptr"]

        if adapter_ptr not in self._adapter_registry:
            raise ValueError(f"Adapter {adapter_ptr} not registered")

        name, path = self._adapter_registry[adapter_ptr]
        adapter_file = Path(path) / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

        return {"adapter_data": adapter_file.read_bytes()}

    def shutdown(self) -> None:
        """Clean up vLLM worker resources.

        For TP>1, sends None sentinel via queue to signal non-leader
        workers to exit their worker_loop before releasing the vLLM worker.
        """
        if self._tp_queue is not None and self.config.rank == 0:
            try:
                self._tp_queue.put(None)
            except Exception:
                pass  # Best-effort shutdown signal
        if self.vllm_worker is not None:
            self.vllm_worker = None

    def worker_loop(self) -> None:
        """Non-leader worker loop for TP followers.

        In Pie's multi-GPU architecture, non-leader ranks wait for commands
        from the group leader. For vLLM TP>1, non-leaders must call
        execute_model() simultaneously with the leader to participate in
        NCCL collective operations (allreduce, broadcast) within the
        model forward pass.

        The leader sends the SchedulerOutput via multiprocessing.Queue
        before each execute_model() call.  Non-leaders receive it here
        and call execute_model() to join the NCCL collectives.

        Using a Queue instead of NCCL broadcast_object_list avoids the
        NCCL 600s idle timeout that kills both workers when the engine
        has no requests for >10 minutes.
        """
        import signal
        from vllm.config import set_current_vllm_config

        if self._tp_queue is None:
            raise RuntimeError("worker_loop requires a tp_queue for TP signaling")

        shutdown_requested = False

        def _handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

        while not shutdown_requested:
            # Wait for SchedulerOutput from leader via queue.
            # Leader sends None as shutdown signal.
            try:
                scheduler_output = self._tp_queue.get()
            except Exception:
                # Queue closed or process shutting down — exit cleanly
                break

            if scheduler_output is None:
                break  # Shutdown signal from leader

            # Sync custom mask and logits_indices from leader so both
            # ranks build the same CommonAttentionMetadata and compute
            # logits at the same positions (required for NCCL sync).
            _mask = getattr(scheduler_output, "_pie_custom_mask", None)
            self.vllm_worker.model_runner.pie_custom_mask = _mask
            _logits_idx = getattr(scheduler_output, "_pie_logits_indices", None)
            self.vllm_worker.model_runner.pie_logits_indices = _logits_idx

            # Execute model step — participates in NCCL collectives
            # alongside the leader's execute_model() call.
            try:
                with set_current_vllm_config(self.vllm_config):
                    result = self.vllm_worker.execute_model(scheduler_output)
                    # Mirror the leader's two-phase async scheduling pattern:
                    # - AsyncModelRunnerOutput: call get_output() to drain it
                    # - None: call sample_tokens() to complete the step
                    # Without this, the next execute_model() call fails with
                    # "sample_tokens() must be called after execute_model()".
                    from vllm.v1.outputs import AsyncModelRunnerOutput
                    if isinstance(result, AsyncModelRunnerOutput):
                        result.get_output()
                    elif result is None:
                        self.vllm_worker.sample_tokens(None)
            except Exception as e:
                # Clear per-batch state so the next batch doesn't reuse
                # stale mask/indices from this failed batch.
                self.vllm_worker.model_runner.pie_logits_indices = None
                self.vllm_worker.model_runner.pie_custom_mask = None
                # Log but continue — leader may send a recovery batch next.
                print(f"[WARN] Non-leader worker execute_model error: {e}",
                      file=sys.stderr, flush=True)
