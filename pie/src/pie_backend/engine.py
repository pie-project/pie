"""
Inference engine for PIE.

Pure computation: load a model, run inference (embed → transform → sample),
manage adapters. No distributed coordination, no RPC, no batching.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from .config import RuntimeConfig
from .loader import ModelLoader
from .adapter import AdapterSubpass, CmaesAdapter
from . import model as model_registry
from . import hf_utils
from . import telemetry


class Engine:
    """Inference engine. Loads a model and runs forward passes.

    This class owns the model components (forward pass, KV cache, adapters)
    and provides the core inference step. It has no knowledge of distributed
    coordination, RPC, or batching — those belong in worker.py.
    """

    config: RuntimeConfig

    # Model components
    forward_pass: object  # e.g., llama3.ForwardPass
    model_config: object  # e.g., llama3.ModelConfig
    kv_cache_at_layer: list[torch.Tensor]

    # CPU swap pool (pinned host memory, mirrors GPU KV cache layout)
    kv_cache_at_layer_host: list[torch.Tensor]  # [pool_size, 2, page_size, kv_heads, dim_head] per layer
    swap_pool_size: int  # total host slots (Rust manages allocation)

    # Adapter state
    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]
    adapters: dict

    # Model info
    arch_type: str
    info: dict
    snapshot_dir: str | None

    def __init__(
        self,
        config: RuntimeConfig,
        model_config,
        forward_pass,
        kv_cache_at_layer: list,
        adapter_at_layer: list,
        arch_type: str,
        info: dict,
        snapshot_dir: str | None = None,
        kv_cache_at_layer_host: list | None = None,
        swap_pool_size: int = 0,
    ):
        self.config = config
        self.model_config = model_config
        self.forward_pass = forward_pass
        self.kv_cache_at_layer = kv_cache_at_layer
        self.kv_cache_at_layer_host = kv_cache_at_layer_host or []
        self.swap_pool_size = swap_pool_size
        self.adapter_at_layer = adapter_at_layer
        self.arch_type = arch_type
        self.info = info
        self.snapshot_dir = snapshot_dir
        self.adapters = {}

    @classmethod
    def load(
        cls,
        config: RuntimeConfig,
        log_queue: object = None,
        compute_process_group=None,
    ) -> "Engine":
        """Load a model from config and return an Engine.

        Args:
            config: Runtime configuration
            log_queue: Optional queue for sending logs back to controller
            compute_process_group: Optional process group for tensor parallelism
        """

        def _log(msg: str, level: str = "INFO"):
            if log_queue is not None:
                log_queue.put({"message": msg, "level": level})

        # Initialize telemetry (only on rank 0 to avoid duplicate spans)
        if config.rank == 0:
            telemetry.init_telemetry(
                enabled=config.telemetry_enabled,
                service_name=config.telemetry_service_name,
                endpoint=config.telemetry_endpoint,
            )

        # Initialize seeds
        _log(f"Initializing with random seed: {config.random_seed}", "DEBUG")
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        # DUMMY MODE: Skip weight loading and use dummy forward pass
        if config.dummy_mode:
            return cls._load_dummy(config, log_queue)

        # Load model weights using ModelLoader
        loader = ModelLoader(config, log_queue=log_queue)
        _log("Loading model weights", "DEBUG")
        weights, normalized_arch, info = loader.load()
        snapshot_dir = loader.snapshot_dir
        _log("Loaded model weights", "DEBUG")

        # Look up architecture module via registry
        arch_type = info["architecture"]["type"]
        mod = model_registry.get_module(arch_type)

        # Create model-specific components
        model_config = mod.ModelConfig.from_dict(normalized_arch)
        config.max_num_kv_pages = model_config.eval_max_num_kv_pages(config)

        forward_pass = mod.ForwardPass(
            model_config,
            config,
            weights,
            compute_process_group=compute_process_group,
        )

        adapter_at_layer = mod.create_adapter_cache(model_config, config)
        kv_cache_at_layer = mod.create_kv_cache(model_config, config)

        # Compact weight memory layout for GPU locality (MPS TLB optimization).
        # Must run AFTER KV cache allocation: the CPU roundtrip inside
        # compact_weights() frees all GPU weights and re-uploads them into
        # the address space above the KV cache, giving contiguous layout.
        if hasattr(forward_pass, "compact_weights"):
            forward_pass.compact_weights()

        # Warmup CUDA graphs if supported
        if hasattr(forward_pass, "warmup_cuda_graphs"):
            forward_pass.warmup_cuda_graphs(kv_cache_at_layer)

        # Compact weight memory layout for GPU locality (MPS TLB optimization)
        if hasattr(forward_pass, "compact_weights"):
            forward_pass.compact_weights()

        # Allocate CPU swap pool (pinned host memory)
        host_kv, pool_size = cls._create_host_kv_cache(
            kv_cache_at_layer, config.swap_budget_bytes,
        )

        return cls(
            config=config,
            model_config=model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_at_layer=adapter_at_layer,
            arch_type=arch_type,
            info=info,
            snapshot_dir=snapshot_dir,
            kv_cache_at_layer_host=host_kv,
            swap_pool_size=pool_size,
        )

    @classmethod
    def _load_dummy(cls, config: RuntimeConfig, log_queue: object = None) -> "Engine":
        """Load dummy mode — no GPU weights."""
        from .model.dummy import (
            DummyModelConfig,
            DummyForwardPass,
            create_kv_cache,
            create_adapter_cache,
        )

        def _log(msg: str, level: str = "INFO"):
            if log_queue is not None:
                log_queue.put({"message": msg, "level": level})

        _log("Initializing in DUMMY MODE - no GPU weights will be loaded", "INFO")

        model_config = DummyModelConfig()
        config.max_num_kv_pages = model_config.eval_max_num_kv_pages(config)

        forward_pass = DummyForwardPass(model_config, config)
        kv_cache_at_layer = create_kv_cache(model_config, config)
        adapter_at_layer = create_adapter_cache(model_config, config)

        # Load tokenizer from HuggingFace (doesn't require GPU)
        snapshot_dir = None
        try:
            snapshot_dir = hf_utils.get_hf_snapshot_dir(config.hf_repo)
            _log(f"Loaded tokenizer from {config.hf_repo}", "DEBUG")
        except Exception as e:
            _log(f"Could not load tokenizer: {e}. Using empty tokenizer.", "WARN")

        info = {
            "architecture": {"type": "dummy"},
            "vocab_size": model_config.vocab_size,
        }

        _log("Dummy mode initialization complete", "INFO")

        host_kv, pool_size = cls._create_host_kv_cache(
            kv_cache_at_layer, config.swap_budget_bytes,
        )

        return cls(
            config=config,
            model_config=model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_at_layer=adapter_at_layer,
            arch_type="dummy",
            info=info,
            snapshot_dir=snapshot_dir,
            kv_cache_at_layer_host=host_kv,
            swap_pool_size=pool_size,
        )

    # ========================================================================
    # CPU Swap Pool
    # ========================================================================

    @staticmethod
    def _create_host_kv_cache(
        gpu_kv: list[torch.Tensor],
        swap_budget_bytes: int,
    ) -> tuple[list[torch.Tensor], int]:
        """Allocate pinned CPU tensors mirroring GPU KV cache layout.

        Returns (host_tensors, pool_size). If budget is 0 or there are no
        GPU KV layers, returns ([], 0).
        """
        if swap_budget_bytes <= 0 or not gpu_kv:
            return [], 0

        # gpu_kv[layer] shape: [max_pages+1, 2, page_size, kv_heads, dim_head]
        num_layers = len(gpu_kv)
        _, two, page_size, kv_heads, dim_head = gpu_kv[0].shape
        dtype = gpu_kv[0].dtype
        bytes_per_element = gpu_kv[0].element_size()

        # Per-page bytes across ALL layers
        per_page_bytes = num_layers * two * page_size * kv_heads * dim_head * bytes_per_element
        pool_size = swap_budget_bytes // per_page_bytes

        if pool_size == 0:
            return [], 0

        host_kv = [
            torch.zeros(
                (pool_size, two, page_size, kv_heads, dim_head),
                dtype=dtype,
                device="cpu",
            ).pin_memory()
            for _ in range(num_layers)
        ]

        return host_kv, pool_size

    # ========================================================================
    # Inference
    # ========================================================================

    @torch.inference_mode()
    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> list:
        """Execute a single inference step (Embed → Transform → Sample).

        This is the core forward pass. It does NOT handle batching, TP barriers,
        or distributed broadcast — those are handled by the worker.

        Args:
            inputs: Model inputs dict (token_ids, position_ids, kv_page_indices, etc.)
            sampling_metadata: Sampling configuration dict

        Returns:
            Sampling results list
        """
        # Embed inputs
        input_embeds = self.forward_pass.embed_inputs(inputs)

        # Create AdapterSubpass if adapters are active
        adapter_subpass = None
        if inputs.get("adapter_indices"):
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=inputs["adapter_indices"],
                adapter_extras=self.adapters,
                rand_seeds=inputs["adapter_seeds"],
                qo_indptr=inputs["qo_indptr"],
            )

        # Run transformer forward pass
        hidden_states = self.forward_pass.transform(
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

        # Sampling pass
        sampling_results = self.forward_pass.sample(hidden_states, sampling_metadata)

        return sampling_results

    # ========================================================================
    # Adapter Management
    # ========================================================================

    @torch.inference_mode()
    def init_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ):
        """Initialize an adapter."""
        cfg = self.model_config

        if adapter_ptr >= self.config.max_num_adapters:
            raise ValueError(
                f"Adapter pointer {adapter_ptr} exceeds max_num_adapters {self.config.max_num_adapters}"
            )

        # Calculate local shard sizes for distributed adapters
        tp_size = self.config.tensor_parallel_size
        gpu_rank = self.config.rank % tp_size

        local_num_q_heads = cfg.num_q_heads // tp_size
        local_num_kv_heads = cfg.num_kv_heads // tp_size

        local_out_features = [
            cfg.dim_head * local_num_q_heads,
            cfg.dim_head * local_num_kv_heads,
            cfg.dim_head * local_num_kv_heads,
        ]

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
    def update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        """Update adapter parameters."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.config.device)

        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                adapter.update(scores, seeds, max_sigma)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.config.device)

    def load_adapter(self, adapter_ptr: int, name: str, data: bytes) -> None:
        """Load adapter weights from file."""
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"
                adapter.upload(name, data)

    def save_adapter(self, adapter_ptr: int, name: str) -> bytes:
        """Save adapter weights to file."""
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"
                return adapter.download(name)
        return b""

    # ========================================================================
    # Metadata
    # ========================================================================

    def query(self, query: str) -> str:
        """Handle a simple query."""
        match query:
            case "ping":
                return "pong"
            case _:
                return "unknown query"
