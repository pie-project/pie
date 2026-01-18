"""
Runtime orchestrator for PIE backend.

This module provides the main Runtime class that orchestrates inference
by delegating to specialized components:
- config.py: RuntimeConfig
- batching.py: Batch
- loader.py: ModelLoader
"""

from __future__ import annotations

import random

import numpy as np
from . import utils
import torch
import torch.distributed as dist

from .config import RuntimeConfig
from .batching import Batch
from .loader import ModelLoader
from .adapter import AdapterSubpass, CmaesAdapter
from .model import llama3, qwen2, qwen3, gemma2, gemma3, common
from .model.chat_templates import (
    Llama3Template,
    Qwen2_5Template,
    Qwen3Template,
    GPTOSSTemplate,
    Gemma2Template,
    Gemma3Template,
    ChatTemplate,
)

# gpt_oss requires CUDA-only features, only import on CUDA platforms
if torch.cuda.is_available():
    from .model import gpt_oss
from . import message
from . import hf_utils
from . import telemetry

import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

# Re-export RuntimeConfig for backward compatibility
__all__ = ["Runtime", "RuntimeConfig"]


class StepTiming(NamedTuple):
    """Timing data for a single fire_batch step."""

    # Top-level stages
    build_batch: float
    get_inputs: float
    get_sampling_meta: float
    broadcast: float
    inference: float
    create_responses: float
    total: float
    # build_batch breakdown
    decode_u32: float
    mask_loop: float
    brle_decode: float
    sampler_loop: float


@dataclass
class LatencyStats:
    """Tracks latency statistics for fire_batch stages.

    When profiling is enabled, emits spans to OpenTelemetry.
    When disabled, this is a no-op.
    """

    enabled: bool = False
    step_count: int = 0

    def record_span(self, timing: StepTiming, traceparent: str | None = None):
        """Record timing as an OpenTelemetry span.

        Args:
            timing: Timing data for the step.
            traceparent: Optional W3C traceparent string for cross-language propagation.
        """
        self.step_count += 1

        if not self.enabled:
            return

        # Create a span with all timing attributes (in milliseconds)
        # Use traceparent if provided (from Rust) to link traces across languages
        with telemetry.start_span_with_traceparent(
            "py.fire_batch",
            traceparent,
            step=self.step_count,
            total_ms=timing.total * 1000,
            build_batch_ms=timing.build_batch * 1000,
            decode_u32_ms=timing.decode_u32 * 1000,
            mask_loop_ms=timing.mask_loop * 1000,
            brle_decode_ms=timing.brle_decode * 1000,
            sampler_loop_ms=timing.sampler_loop * 1000,
            get_inputs_ms=timing.get_inputs * 1000,
            get_sampling_meta_ms=timing.get_sampling_meta * 1000,
            broadcast_ms=timing.broadcast * 1000,
            inference_ms=timing.inference * 1000,
            create_responses_ms=timing.create_responses * 1000,
        ) as span:
            pass  # span is closed automatically


class Runtime:
    """
    Main runtime orchestrator for PIE inference.

    This class has been refactored to focus on protocol handling,
    delegating loading to ModelLoader and batch management to BatchBuilder.
    """

    config: RuntimeConfig

    # Model components (renamed from forward_pass to engine to avoid collision)
    engine: llama3.ForwardPass
    model_config: llama3.ModelConfig
    kv_cache_at_layer: list[torch.Tensor]

    # Adapter state
    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]
    adapters: dict

    # Logging
    log_queue: object

    def _log(self, msg: str, level: str = "INFO") -> None:
        """Log a message to the queue if available."""
        if self.log_queue is not None:
            self.log_queue.put({"message": msg, "level": level})

    def __init__(
        self,
        config: RuntimeConfig,
        log_queue: object = None,
        group_id: int = 0,
        result_queue: object = None,
        result_queues: list = None,
        process_groups: dict = None,
        compute_process_groups: dict = None,
        group_topology: list = None,
    ):
        """
        Initialize the runtime.

        Args:
            config: Runtime configuration
            log_queue: Optional queue for sending logs back to controller
            group_id: Data Parallel Execution Group ID (Default/Local group)
            result_queue: Optional queue for results (single worker usage)
            result_queues: List of queues for all groups (Rank 0 only)
            process_groups: Dict of {group_id: ProcessGroup} for all groups (Rank 0 only)
            compute_process_groups: Dict of {group_id: ProcessGroup} for TP sync
            group_topology: List of groups [[rank0, rank1], [rank2, rank3], ...]
        """
        self.config = config
        self.log_queue = log_queue
        self.group_id = group_id
        self.result_queue = result_queue
        self.result_queues = result_queues if result_queues is not None else []
        self.process_groups = process_groups if process_groups is not None else {}
        self.compute_process_groups = (
            compute_process_groups if compute_process_groups is not None else {}
        )
        self.group_topology = group_topology if group_topology is not None else []
        self.adapters = {}

        # Determine if this rank is the group leader (lowest rank in its group)
        self.is_group_leader = False
        if self.group_topology and 0 <= group_id < len(self.group_topology):
            my_group = self.group_topology[group_id]
            if my_group and config.rank == min(my_group):
                self.is_group_leader = True

        # Initialize telemetry (only on rank 0 to avoid duplicate spans)
        if config.rank == 0:
            telemetry.init_telemetry(
                enabled=config.telemetry_enabled,
                service_name=config.telemetry_service_name,
                endpoint=config.telemetry_endpoint,
            )

        self._latency_stats = LatencyStats(enabled=config.telemetry_enabled)

        # Initialize seeds
        self._log(f"Initializing with random seed: {config.random_seed}", "DEBUG")

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        # DUMMY MODE: Skip weight loading and use dummy forward pass
        if config.dummy_mode:
            self._init_dummy_mode()
            return

        # Load model weights using ModelLoader
        loader = ModelLoader(config, log_queue=log_queue)

        # print(f"[DEBUG R{config.rank}] Runtime: Loading weights...")
        self._log("Loading model weights", "DEBUG")

        weights, normalized_arch, self.info = loader.load()

        # print(f"[DEBUG R{config.rank}] Runtime: Weights loaded")
        # Store snapshot_dir for tokenizer loading
        self.snapshot_dir = loader.snapshot_dir

        self._log("Loaded model weights", "DEBUG")

        # Store architecture type
        self.type = self.info["architecture"]["type"]
        # print(f"[DEBUG R{config.rank}] Runtime: Creating engine for {self.type}...")

        # Create model-specific components based on architecture
        match self.type:
            case "llama3" | "l4ma":
                # Create model config
                self.model_config = llama3.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST (needed for ForwardPass)
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = llama3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                    compute_process_group=self.compute_process_groups.get(
                        self.group_id
                    ),
                )
                # Create adapter cache
                self.adapter_at_layer = llama3.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = llama3.create_kv_cache(
                    self.model_config, config
                )

                # Warmup CUDA graphs
                self.engine.warmup_cuda_graphs(self.kv_cache_at_layer)

            case "qwen2":
                # Create model config
                self.model_config = qwen2.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = qwen2.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                    compute_process_group=self.compute_process_groups.get(
                        self.group_id
                    ),
                )
                # Create adapter cache
                self.adapter_at_layer = qwen2.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = qwen2.create_kv_cache(
                    self.model_config, config
                )

            case "qwen3":
                # Create model config
                self.model_config = qwen3.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = qwen3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                    compute_process_group=self.compute_process_groups.get(
                        self.group_id
                    ),
                )

                # Create adapter cache
                self.adapter_at_layer = qwen3.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = qwen3.create_kv_cache(
                    self.model_config, config
                )

            case "gptoss":
                # gpt_oss requires CUDA-only features
                if not torch.cuda.is_available():
                    raise ValueError(
                        "GPT-OSS model requires CUDA. Apple Silicon is not supported. "
                        "Please use llama3, qwen2, or qwen3 models instead."
                    )

                # Create model config
                self.model_config = gpt_oss.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = gpt_oss.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                    compute_process_group=compute_process_group,
                )
                # Create adapter cache
                self.adapter_at_layer = gpt_oss.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = gpt_oss.create_kv_cache(
                    self.model_config, config
                )

            case "gemma2":
                # Create model config
                self.model_config = gemma2.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = gemma2.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                    compute_process_group=self.compute_process_groups.get(
                        self.group_id
                    ),
                )
                # Create adapter cache
                self.adapter_at_layer = gemma2.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = gemma2.create_kv_cache(
                    self.model_config, config
                )

            case "gemma3":
                # Create model config
                self.model_config = gemma3.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = gemma3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                    compute_process_group=self.compute_process_groups.get(
                        self.group_id
                    ),
                )
                # Create adapter cache
                self.adapter_at_layer = gemma3.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = gemma3.create_kv_cache(
                    self.model_config, config
                )

            case _:
                raise ValueError(f"Unsupported architecture type: {self.type}")

    def _init_dummy_mode(self) -> None:
        """
        Initialize dummy mode - no GPU weight loading.

        Creates a DummyForwardPass that returns random tokens instead of
        running actual inference. Useful for testing scheduling logic
        and benchmarking throughput without GPU overhead.
        """
        from .model.dummy import (
            DummyModelConfig,
            DummyForwardPass,
            create_kv_cache,
            create_adapter_cache,
        )

        self._log("Initializing in DUMMY MODE - no GPU weights will be loaded", "INFO")

        self.type = "dummy"
        self.model_config = DummyModelConfig()
        self.config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
            self.config
        )

        self.engine = DummyForwardPass(self.model_config, self.config)
        self.kv_cache_at_layer = create_kv_cache(self.model_config, self.config)
        self.adapter_at_layer = create_adapter_cache(self.model_config, self.config)

        # Load tokenizer from HuggingFace (doesn't require GPU)
        # This is needed for the Rust tokenizer to work properly
        try:
            self.snapshot_dir = hf_utils.get_hf_snapshot_dir(self.config.hf_repo)
            self._log(f"Loaded tokenizer from {self.config.hf_repo}", "DEBUG")
        except Exception as e:
            self._log(f"Could not load tokenizer: {e}. Using empty tokenizer.", "WARN")
            self.snapshot_dir = None

        self.info = {
            "architecture": {"type": "dummy"},
            "vocab_size": self.model_config.vocab_size,
        }

        self._log("Dummy mode initialization complete", "INFO")

    # ========================================================================
    # Metadata Accessors
    # ========================================================================

    def get_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "name": self.config.hf_repo,
            "description": "",
            "version": "1.0.0",
        }

    def get_chat_template(self) -> dict:
        """Get chat template configuration based on model type."""
        template: ChatTemplate | None = None

        if self.type == "llama3" or self.type == "l4ma" or self.type == "dummy":
            # Use Llama3Template for dummy mode too (since default hf_repo is Llama)
            template = Llama3Template
        elif self.type == "qwen2":
            template = Qwen2_5Template
        elif self.type == "qwen3":
            template = Qwen3Template
        elif self.type == "gptoss":
            template = GPTOSSTemplate
        elif self.type == "gemma2":
            template = Gemma2Template
        elif self.type == "gemma3":
            template = Gemma3Template

        if template:
            return {
                "template_type": template.template_type,
                "template_content": template.template,
                "stop_tokens": template.stop_tokens,
            }

        return {
            "template_type": "none",
            "template_content": "",
            "stop_tokens": [],
        }

    def get_tokenizer(self) -> dict:
        """Get tokenizer configuration with merge table from HuggingFace."""
        if self.snapshot_dir is None:
            return {
                "type": "bpe",
                "num_vocab": self.engine.weights.get("embed_token").shape[0],
                "merge_table": {},
                "split_regex": "",
                "special_tokens": {},
                "escape_non_printable": False,
                "sentencepiece_space": False,
            }

        # Load tokenizer info from HuggingFace
        tokenizer_info = hf_utils.load_hf_tokenizer(self.snapshot_dir)

        return {
            "type": tokenizer_info.get("type", "bpe"),
            "num_vocab": tokenizer_info.get(
                "num_vocab", self.engine.weights.get("embed_token").shape[0]
            ),
            "merge_table": tokenizer_info.get("merge_table", {}),
            "split_regex": tokenizer_info.get("split_regex", ""),
            "special_tokens": tokenizer_info.get("special_tokens", {}),
            "escape_non_printable": tokenizer_info.get("escape_non_printable", False),
            "sentencepiece_space": tokenizer_info.get("sentencepiece_space", False),
        }

    # ========================================================================
    # Service Protocol Implementation
    # ========================================================================

    def handshake(self, req: message.HandshakeRequest) -> message.HandshakeResponse:
        """Handle handshake request returning model and tokenizer info."""
        metadata = self.get_metadata()
        template = self.get_chat_template()
        tokenizer = self.get_tokenizer()

        return message.HandshakeResponse(
            version=metadata.get("version", "1.0.0"),
            model_name=metadata["name"],
            model_traits=[],
            model_description=metadata.get("description", ""),
            prompt_template=template["template_content"],
            prompt_template_type=template["template_type"],
            prompt_stop_tokens=template["stop_tokens"],
            kv_page_size=self.config.kv_page_size,
            max_batch_tokens=self.config.max_batch_tokens or 10240,
            max_batch_size=self.config.max_batch_size or 128,
            resources={
                0: self.config.max_num_kv_pages or 0,
                1: self.config.max_num_embeds,
                2: self.config.max_num_adapters,
            },
            tokenizer_num_vocab=tokenizer["num_vocab"],
            tokenizer_merge_table=tokenizer["merge_table"],
            tokenizer_special_tokens=tokenizer["special_tokens"],
            tokenizer_split_regex=tokenizer["split_regex"],
            tokenizer_escape_non_printable=tokenizer["escape_non_printable"],
            tokenizer_sentencepiece_space=tokenizer["sentencepiece_space"],
        )

    def query(self, req: message.QueryRequest) -> message.QueryResponse:
        """Handle query request."""
        value = "unknown query"
        match req.query:
            case "ping":
                value = "pong"
        return message.QueryResponse(value=value)

    def embed_image(self, req: message.EmbedImageRequest) -> None:
        """Handle image embedding requests."""
        # TODO: implement image embedding
        pass

    def initialize_adapter(self, req: message.InitializeAdapterRequest) -> None:
        """Initialize adapter functionality."""
        args = {
            "adapter_ptr": req.adapter_ptr,
            "rank": req.rank,
            "alpha": req.alpha,
            "population_size": req.population_size,
            "mu_fraction": req.mu_fraction,
            "initial_sigma": req.initial_sigma,
        }

        if self.config.world_size > 1:
            msg = {"type": "INIT_ADAPTER", "kwargs": args}
            my_pg = self.compute_process_groups.get(self.group_id)
            leader_global_rank = (
                self.group_topology[self.group_id][0] if self.group_topology else 0
            )
            utils.broadcast_struct(
                msg, src=leader_global_rank, device=self.config.device, group=my_pg
            )

        self._initialize_adapter(**args)

    def update_adapter(self, req: message.UpdateAdapterRequest) -> None:
        """Update adapter parameters."""
        args = {
            "adapter_ptr": req.adapter_ptr,
            "scores": req.scores,
            "seeds": req.seeds,
            "max_sigma": req.max_sigma,
        }

        if self.config.world_size > 1:
            msg = {"type": "UPDATE_ADAPTER", "kwargs": args}
            my_pg = self.compute_process_groups.get(self.group_id)
            leader_global_rank = (
                self.group_topology[self.group_id][0] if self.group_topology else 0
            )
            utils.broadcast_struct(
                msg, src=leader_global_rank, device=self.config.device, group=my_pg
            )

        self._update_adapter(**args)

    def upload_adapter(self, req: message.UploadAdapterRequest) -> None:
        """Upload adapter weights."""
        data = req.adapter_data
        if isinstance(data, list):
            data = bytes(data)

        args = {"adapter_ptr": req.adapter_ptr, "name": req.name, "data": data}

        if self.config.world_size > 1:
            msg = {"type": "UPLOAD_ADAPTER", "kwargs": args}
            my_pg = self.compute_process_groups.get(self.group_id)
            leader_global_rank = (
                self.group_topology[self.group_id][0] if self.group_topology else 0
            )
            utils.broadcast_struct(
                msg, src=leader_global_rank, device=self.config.device, group=my_pg
            )

        self._upload_adapter(**args)

    def _upload_adapter(self, adapter_ptr: int, name: str, data: bytes) -> None:
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                # For multi-GPU, append rank to filename to ensure each rank loads its own shard
                # Current CmaesAdapter implementation expects rank-specific checkpoints
                # For multi-GPU, append rank to filename to ensure each rank loads its own shard
                # Current CmaesAdapter implementation expects rank-specific checkpoints
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"

                adapter.upload(name, data)

    def download_adapter(self, req: message.DownloadAdapterRequest) -> None:
        """Download adapter weights."""
        args = {"adapter_ptr": req.adapter_ptr, "name": req.name}

        if self.config.world_size > 1:
            msg = {"type": "DOWNLOAD_ADAPTER", "kwargs": args}
            my_pg = self.compute_process_groups.get(self.group_id)
            leader_global_rank = (
                self.group_topology[self.group_id][0] if self.group_topology else 0
            )
            utils.broadcast_struct(
                msg, src=leader_global_rank, device=self.config.device, group=my_pg
            )

        self._download_adapter(**args)

    def _download_adapter(self, adapter_ptr: int, name: str) -> bytes:
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                # For multi-GPU, append rank to filename to save rank-specific shards
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"

                return adapter.download(name)
        return b""

    # ========================================================================
    # Internal Adapter Methods
    # ========================================================================

    @torch.inference_mode()
    def _initialize_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ):
        cfg = self.model_config

        # Check if adapter limits are exceeded
        if adapter_ptr >= self.config.max_num_adapters:
            raise ValueError(
                f"Adapter pointer {adapter_ptr} exceeds max_num_adapters {self.config.max_num_adapters}"
            )
        # print parameters
        # print(f"Initializing adapter {adapter_ptr} with rank {rank}, alpha {alpha}, population size {population_size}, mu fraction {mu_fraction}, initial sigma {initial_sigma}")
        # Calculate local shard sizes for distributed adapters
        tp_size = self.config.tensor_parallel_size
        gpu_rank = self.config.rank % tp_size

        local_num_q_heads = cfg.num_q_heads // tp_size
        local_num_kv_heads = cfg.num_kv_heads // tp_size

        # Local output features (sharded up-projection)
        local_out_features = [
            cfg.dim_head * local_num_q_heads,
            cfg.dim_head * local_num_kv_heads,
            cfg.dim_head * local_num_kv_heads,
        ]

        # Initialize adapter
        self.adapters[adapter_ptr] = CmaesAdapter(
            adapter_id=adapter_ptr,
            adapter_at_layer=self.adapter_at_layer,
            rank=rank,
            alpha=alpha,
            in_features=cfg.dim_hidden,
            out_features=local_out_features,
            num_layers=cfg.num_layers,
            population_size=population_size,
            mu_fraction=mu_fraction,
            initial_sigma=initial_sigma,
            min_sigma=1e-7,
            min_var=1e-8,
            max_var=1e4,
            device=self.config.device,
            dtype=self.config.activation_dtype,
            gpu_rank=gpu_rank,
            world_size=tp_size,
            adapter_path=self.config.adapter_path,
        )

    @torch.inference_mode()
    def _update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        # Synchronize before adapter update to ensure any pending inference is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.config.device)

        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                adapter.update(scores, seeds, max_sigma)

        # Synchronize after to ensure adapter update is complete before next inference
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.config.device)

    # ========================================================================
    # Batch Execution
    # ========================================================================

    @torch.inference_mode()
    def worker_loop(self):
        """
        Worker loop for ranks > 0.
        Waits for control messages from rank 0 and executes commands.
        """
        import torch.distributed as dist
        import signal

        # Re-enable SIGTERM handling so this worker can be terminated
        # by the parent process during shutdown
        shutdown_requested = False

        def sigterm_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True

        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigterm_handler)

        device = self.config.device

        while not shutdown_requested:
            # Receive control message
            try:
                # Use compute_process_groups for TP-only broadcast
                # src must be the leader's GLOBAL rank, not group-local rank
                my_pg = self.compute_process_groups.get(self.group_id)
                leader_global_rank = self.group_topology[self.group_id][0]
                msg = utils.broadcast_struct(
                    None, src=leader_global_rank, device=device, group=my_pg
                )
            except Exception:
                break

            if shutdown_requested:
                break

            if msg == "STOP":
                break

            if isinstance(msg, dict):
                msg_type = msg.get("type")

                if msg_type == "STEP":
                    # Execute inference step (tensors already broadcast from Rank 0)
                    inputs = msg["inputs"]
                    sampling_metadata = msg["sampling_metadata"]
                    try:
                        result = self._run_step(inputs, sampling_metadata)

                        # If I'm the group leader of a secondary group, push result to result_queue
                        if (
                            self.is_group_leader
                            and self.group_id > 0
                            and self.result_queue is not None
                        ):
                            self.result_queue.put(result)
                    except Exception as e:
                        # Log error but continue - don't let one error hang the whole system
                        print(f"Worker {self.config.rank} _run_step error: {e}")
                        # Sync CUDA to clear any pending operations
                        torch.cuda.synchronize()

                        # Push error result to avoid deadlock
                        if (
                            self.is_group_leader
                            and self.group_id > 0
                            and self.result_queue is not None
                        ):
                            self.result_queue.put(None)

                elif msg_type == "STEP_RAW":
                    # FAST PATH for DP: Receive raw kwargs, build tensors locally
                    # This avoids synchronous dist.broadcast() AND Batch building on Rank 0
                    from .batching import Batch

                    kwargs = msg["kwargs"]
                    try:
                        # Build batch and tensors locally
                        batch = Batch(
                            kwargs,
                            self.config.kv_page_size,
                            self.config.max_dist_size,
                            self.adapters,
                        )
                        device = self.config.device
                        inputs = batch.get_model_inputs(device)
                        sampling_metadata = batch.get_sampling_metadata(
                            device, self.config.activation_dtype
                        )

                        # Execute inference
                        sampling_results = self._run_step(inputs, sampling_metadata)

                        # Package responses locally (critical: moves this work to worker)
                        responses = batch.create_responses(sampling_results)
                        results = [
                            {"tokens": resp.tokens, "dists": resp.dists}
                            for resp in responses
                        ]

                        # Return fully packaged response dict
                        if self.result_queue is not None:
                            self.result_queue.put(
                                {
                                    "results": results,
                                    "batch_size": len(responses),
                                }
                            )
                    except Exception as e:
                        print(f"Worker {self.config.rank} STEP_RAW error: {e}")
                        torch.cuda.synchronize()
                        if self.result_queue is not None:
                            self.result_queue.put(None)

                elif msg_type == "INIT_ADAPTER":
                    # Initialize adapter
                    kwargs = msg["kwargs"]
                    self._initialize_adapter(**kwargs)

                elif msg_type == "UPDATE_ADAPTER":
                    # Update adapter
                    kwargs = msg["kwargs"]
                    self._update_adapter(**kwargs)

                elif msg_type == "UPLOAD_ADAPTER":
                    # Upload adapter
                    kwargs = msg["kwargs"]
                    self._upload_adapter(**kwargs)

                elif msg_type == "DOWNLOAD_ADAPTER":
                    # Download adapter
                    kwargs = msg["kwargs"]
                    self._download_adapter(**kwargs)

            # Other message types can be added here

    def _run_step(self, inputs: dict, sampling_metadata: dict) -> list:
        """
        Execute a single inference step (Embed -> Transform -> Sample).

        Returns:
            Sampling results (only valid on Rank 0 usually, but we return whatever comes out)
        """
        # TP Barrier: Only sync if TP degree > 1
        if self.config.world_size > 1:
            # If we have compute process groups, use them for TP sync
            if self.compute_process_groups:
                my_compute_pg = self.compute_process_groups.get(self.group_id)
                # Only barrier if TP degree > 1 (group size > 1)
                if my_compute_pg and dist.get_world_size(group=my_compute_pg) > 1:
                    dist.barrier(group=my_compute_pg)
            else:
                # Fallback to global barrier (legacy behavior)

                dist.barrier()

        # 2. Embed inputs
        input_embeds = self.engine.embed_inputs(inputs)

        # 3. Use raw indices/seeds to create local AdapterSubpass (to avoid device mismatch)
        adapter_subpass = None
        if inputs.get("adapter_indices"):
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=inputs["adapter_indices"],
                adapter_extras=self.adapters,
                rand_seeds=inputs["adapter_seeds"],
                qo_indptr=inputs["qo_indptr"],
            )

        # 4. Run transformer forward pass
        hidden_states = self.engine.transform(
            input_embeds=input_embeds,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            custom_mask=inputs["custom_mask"],
            single_token_inference_mode=inputs["single_token_inference_mode"],
            adapter_subpass=adapter_subpass,
            total_pages_cpu=inputs.get("total_pages_cpu", 0),
        )

        # 4. Sampling Pass
        # Pass hidden_states (replicated or sharded? Transform returns replicated in current llama3.py)
        sampling_results = self.engine.sample(hidden_states, sampling_metadata)

        return sampling_results

    # ========================================================================
    # RPC Method Wrappers
    # ========================================================================

    def handshake_rpc(self, **kwargs) -> dict:
        """Handle handshake RPC."""
        req = message.HandshakeRequest(**kwargs)
        resp = self.handshake(req)
        return {
            "version": resp.version,
            "model_name": resp.model_name,
            "model_traits": resp.model_traits,
            "model_description": resp.model_description,
            "prompt_template": resp.prompt_template,
            "prompt_template_type": resp.prompt_template_type,
            "prompt_stop_tokens": resp.prompt_stop_tokens,
            "kv_page_size": resp.kv_page_size,
            "max_batch_tokens": resp.max_batch_tokens,
            "max_batch_size": resp.max_batch_size,
            "resources": resp.resources,
            "tokenizer_num_vocab": resp.tokenizer_num_vocab,
            "tokenizer_merge_table": resp.tokenizer_merge_table,
            "tokenizer_special_tokens": resp.tokenizer_special_tokens,
            "tokenizer_split_regex": resp.tokenizer_split_regex,
            "tokenizer_escape_non_printable": resp.tokenizer_escape_non_printable,
            "tokenizer_sentencepiece_space": resp.tokenizer_sentencepiece_space,
        }

    def query_rpc(self, **kwargs) -> dict:
        """Handle query RPC."""
        req = message.QueryRequest(**kwargs)
        resp = self.query(req)
        return {"value": resp.value}

    def fire_batch(self, **kwargs) -> dict:
        """
        Execute a pre-batched forward pass from Rust via RPC.

        This receives already-batched data from Rust and:
        1. Decodes attention masks from BRLE
        2. Creates tensors on GPU
        3. Executes inference
        4. Returns results

        Args:
            **kwargs: BatchedForwardPassRequest fields

        Returns:
            Dictionary with 'results' list of ForwardPassResponse dicts
        """
        from .batching import Batch

        t_start = time.perf_counter()

        # Use group_id from kwargs if present (from Rust), else default to local
        target_group_id = kwargs.get("group_id", self.group_id)

        # Check if this is a remote DP group (only relevant for central coordinator mode)
        # In symmetric IPC mode, each group leader handles its own group, so this is always False
        is_remote_dp_group = (
            target_group_id != 0
            and self.group_topology
            and target_group_id < len(self.group_topology)
            and 0 not in self.group_topology[target_group_id]
        )

        # SYMMETRIC IPC MODE: If target_group_id == self.group_id, we ARE the group leader
        # and should execute locally. The Rust server already routed the request to us.
        # This check supersedes the rank != 0 check for the symmetric architecture.
        if target_group_id == self.group_id:
            is_remote_dp_group = False

        # PROFILING: Track request distribution and timings
        if not hasattr(self, "_dp_stats"):
            self._dp_stats = {
                "group_counts": {},
                "step_raw_times": [],
                "local_times": [],
                "last_report": time.perf_counter(),
            }
        stats = self._dp_stats
        stats["group_counts"][target_group_id] = (
            stats["group_counts"].get(target_group_id, 0) + 1
        )

        if is_remote_dp_group and self.config.world_size > 1:
            # FAST PATH: Send raw kwargs to group leader, let them build tensors AND package responses
            # This eliminates ALL Batch processing on Rank 0 for remote DP groups
            t0 = time.perf_counter()
            msg = {"type": "STEP_RAW", "kwargs": kwargs}
            utils._control_channel.send(msg, destination_group=target_group_id)
            t_broadcast = time.perf_counter() - t0

            # Wait for fully packaged result from worker
            t0 = time.perf_counter()
            packaged_result = None
            if self.result_queues and target_group_id < len(self.result_queues):
                try:
                    packaged_result = self.result_queues[target_group_id].get(
                        timeout=300
                    )
                    if packaged_result is None:
                        print(
                            f"[Warning] Group {target_group_id} returned error result"
                        )
                except Exception as e:
                    print(
                        f"[Error] Timeout waiting for group {target_group_id} result: {e}"
                    )
            t_inference = time.perf_counter() - t0

            # Use pre-packaged results directly (no Batch building!)
            if (
                packaged_result
                and isinstance(packaged_result, dict)
                and "results" in packaged_result
            ):
                results = packaged_result["results"]
            else:
                results = []

            # Dummy timing values (no local batch processing happened)
            t_build_batch = 0.0
            t_get_inputs = 0.0
            t_get_sampling_meta = 0.0
            t_create_responses = 0.0
            build_timing = {
                "decode_u32": 0.0,
                "mask_loop": 0.0,
                "brle_decode": 0.0,
                "sampler_loop": 0.0,
            }

            t_total = time.perf_counter() - t_start

            # PROFILING: Track timing per path and report periodically
            stats["step_raw_times"].append(t_total)

            # Report every 10 seconds
            now = time.perf_counter()
            if now - stats["last_report"] > 10.0:
                stats["last_report"] = now
                total_reqs = sum(stats["group_counts"].values())
                avg_raw = sum(stats["step_raw_times"]) / max(
                    len(stats["step_raw_times"]), 1
                )
                avg_local = sum(stats["local_times"]) / max(
                    len(stats["local_times"]), 1
                )
                print(
                    f"[PROFILING] Groups: {stats['group_counts']} | "
                    f"Total: {total_reqs} | "
                    f"STEP_RAW avg: {avg_raw*1000:.1f}ms ({len(stats['step_raw_times'])}) | "
                    f"Local avg: {avg_local*1000:.1f}ms ({len(stats['local_times'])})"
                )
                # Reset for next window
                stats["step_raw_times"] = []
                stats["local_times"] = []

            # Record latency stats
            self._latency_stats.record_span(
                StepTiming(
                    build_batch=t_build_batch,
                    get_inputs=t_get_inputs,
                    get_sampling_meta=t_get_sampling_meta,
                    broadcast=t_broadcast,
                    inference=t_inference,
                    create_responses=t_create_responses,
                    total=t_total,
                    decode_u32=build_timing["decode_u32"],
                    mask_loop=build_timing["mask_loop"],
                    brle_decode=build_timing["brle_decode"],
                    sampler_loop=build_timing["sampler_loop"],
                ),
            )

            return {"results": results}

        else:
            # LOCAL PATH: Build tensors and execute (or broadcast for TP groups)
            t0 = time.perf_counter()
            batch = Batch(
                kwargs,
                self.config.kv_page_size,
                self.config.max_dist_size,
                self.adapters,
            )
            build_timing = batch.timing
            t_build_batch = time.perf_counter() - t0

            device = self.config.device
            t0 = time.perf_counter()
            inputs = batch.get_model_inputs(device)
            t_get_inputs = time.perf_counter() - t0

            t0 = time.perf_counter()
            sampling_metadata = batch.get_sampling_metadata(
                device, self.config.activation_dtype
            )
            t_get_sampling_meta = time.perf_counter() - t0

            # Broadcast to workers if multi-GPU TP group
            # Skip broadcast if:
            # - Single process mode (world_size == 1)
            # - IPC mode (rank != 0) - we're already the group leader processing locally
            t0 = time.perf_counter()
            should_broadcast = (
                self.config.world_size > 1
                and self.config.rank == 0  # Only group leader (tp_rank 0) broadcasts
                and target_group_id == self.group_id  # Only for MY TP group
            )
            if should_broadcast:
                msg = {
                    "type": "STEP",
                    "inputs": inputs,
                    "sampling_metadata": sampling_metadata,
                }
                # Use compute_process_groups for TP-only broadcast
                # src must be the leader's GLOBAL rank, not group-local rank
                target_pg = self.compute_process_groups.get(target_group_id)
                leader_global_rank = self.group_topology[target_group_id][0]
                utils.broadcast_struct(
                    msg,
                    src=leader_global_rank,
                    device=device,
                    group=target_pg,
                    group_id=target_group_id,
                )
            t_broadcast = time.perf_counter() - t0

            # Execute inference locally
            t0 = time.perf_counter()
            if target_group_id == self.group_id:
                sampling_results = self._run_step(inputs, sampling_metadata)
            else:
                # TP group with Rank 0: wait for result
                if self.result_queues and target_group_id < len(self.result_queues):
                    try:
                        sampling_results = self.result_queues[target_group_id].get(
                            timeout=300
                        )
                        if sampling_results is None:
                            print(
                                f"[Warning] Group {target_group_id} returned error result"
                            )
                            sampling_results = []
                    except Exception as e:
                        print(
                            f"[Error] Timeout waiting for group {target_group_id} result: {e}"
                        )
                        sampling_results = []
                else:
                    print(
                        f"[Warning] No result_queue for group {target_group_id}, executing locally"
                    )
                    sampling_results = self._run_step(inputs, sampling_metadata)
            t_inference = time.perf_counter() - t0

        # Package responses
        t0 = time.perf_counter()
        responses = batch.create_responses(sampling_results)

        # Convert to serializable format
        results = []
        for resp in responses:
            results.append(
                {
                    "tokens": resp.tokens,
                    "dists": resp.dists,
                }
            )
        t_create_responses = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        # PROFILING: Track timing per path and report periodically
        if is_remote_dp_group:
            stats["step_raw_times"].append(t_total)
        else:
            stats["local_times"].append(t_total)

        # Report every 10 seconds (simple summary only)
        now = time.perf_counter()
        if now - stats["last_report"] > 10.0:
            stats["last_report"] = now
            # Reset for next window
            stats["step_raw_times"] = []
            stats["local_times"] = []

        # Record latency stats
        self._latency_stats.record_span(
            StepTiming(
                build_batch=t_build_batch,
                get_inputs=t_get_inputs,
                get_sampling_meta=t_get_sampling_meta,
                broadcast=t_broadcast,
                inference=t_inference,
                create_responses=t_create_responses,
                total=t_total,
                decode_u32=build_timing["decode_u32"],
                mask_loop=build_timing["mask_loop"],
                brle_decode=build_timing["brle_decode"],
                sampler_loop=build_timing["sampler_loop"],
            ),
            traceparent=kwargs.get("trace_context"),  # Cross-language propagation
        )

        return {"results": results}

    def embed_image_rpc(self, **kwargs) -> None:
        """Handle embed_image RPC."""
        req = message.EmbedImageRequest(**kwargs)
        self.embed_image(req)

    def initialize_adapter_rpc(self, **kwargs) -> None:
        """Handle initialize_adapter RPC."""
        req = message.InitializeAdapterRequest(**kwargs)
        self.initialize_adapter(req)

    def update_adapter_rpc(self, **kwargs) -> None:
        """Handle update_adapter RPC."""
        req = message.UpdateAdapterRequest(**kwargs)
        self.update_adapter(req)

    def upload_adapter_rpc(self, **kwargs) -> None:
        """Handle upload_adapter RPC."""
        req = message.UploadAdapterRequest(**kwargs)
        self.upload_adapter(req)

    def download_adapter_rpc(self, **kwargs) -> None:
        """Handle download_adapter RPC."""
        req = message.DownloadAdapterRequest(**kwargs)
        self.download_adapter(req)

    def shutdown(self):
        """
        Cleanup runtime resources.

        For Rank 0 in multi-GPU setup, this broadcasts the 'STOP' signal to workers.
        """
        if self.config.world_size > 1 and self.config.rank == 0:
            print("Broadcasting STOP signal to workers...")
            utils.broadcast_struct("STOP", src=0, device=self.config.device)

        if dist.is_initialized():
            dist.destroy_process_group()
