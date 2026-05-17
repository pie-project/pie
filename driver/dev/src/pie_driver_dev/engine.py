"""
Inference engine for PIE.

Pure computation: load a model, run inference (embed → transform → sample),
manage adapters. No distributed coordination, no RPC, no batching.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pie_driver_dev.config import NativeRuntimeConfig as RuntimeConfig
from ._bridge.batching import Batch
from .loader import ModelLoader
from .adapter import AdapterSubpass, CmaesAdapter
from . import batch_tensors
from . import model as model_registry
from . import hf_utils
from ._bridge import telemetry


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
    ) -> "Engine":
        """Load a model from config and return an Engine.

        Args:
            config: Runtime configuration
            log_queue: Optional queue for sending logs back to controller

        TP collectives use torch.distributed's default process group, which
        in pie is the per-replica TP group brought up by `_init_distributed`.
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
        config.total_pages = model_config.eval_total_pages(config)

        forward_pass = mod.ForwardPass(
            model_config,
            config,
            weights,
        )

        adapter_at_layer = mod.create_adapter_cache(model_config, config)
        kv_cache_at_layer = mod.create_kv_cache(model_config, config)

        # Compact weight memory layout for GPU locality (MPS TLB optimization).
        # Must run AFTER KV cache allocation: the CPU roundtrip inside
        # compact_weights() frees all GPU weights and re-uploads them into
        # the address space above the KV cache, giving contiguous layout.
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

        # Load the real model's config.json (if available) to pick up the
        # actual vocab size. Grammar-constrained decoding builds logit masks
        # sized to the tokenizer's vocab, so a mismatched vocab here would
        # make the Python BRLE decoder truncate/misalign the mask.
        snapshot_dir = None
        hf_vocab_size: int | None = None
        try:
            snapshot_dir = hf_utils.get_hf_snapshot_dir(config.hf_repo)
            _log(f"Loaded tokenizer from {config.hf_repo}", "DEBUG")
            import json as _json
            from pathlib import Path as _Path
            cfg_path = _Path(snapshot_dir) / "config.json"
            if cfg_path.is_file():
                hf_cfg = _json.loads(cfg_path.read_text())
                hf_vocab_size = hf_cfg.get("vocab_size")
        except Exception as e:
            _log(f"Could not load tokenizer: {e}. Using empty tokenizer.", "WARN")

        model_config = DummyModelConfig()
        if hf_vocab_size is not None:
            model_config.vocab_size = int(hf_vocab_size)
        config.total_pages = model_config.eval_total_pages(config)

        forward_pass = DummyForwardPass(model_config, config)
        kv_cache_at_layer = create_kv_cache(model_config, config)
        adapter_at_layer = create_adapter_cache(model_config, config)

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

        # pin_memory() is CUDA-only; on MPS we use regular CPU tensors
        use_pinned = torch.cuda.is_available()
        host_kv = []
        for _ in range(num_layers):
            t = torch.zeros(
                (pool_size, two, page_size, kv_heads, dim_head),
                dtype=dtype,
                device="cpu",
            )
            if use_pinned:
                t = t.pin_memory()
            host_kv.append(t)

        return host_kv, pool_size

    # ========================================================================
    # Inference
    # ========================================================================

    # ========================================================================
    # KV swap — page copies. Bridge stays torch-free; the index_copy_ calls
    # live here next to the engine that owns the GPU and pinned-host KV
    # tensors.
    # ========================================================================

    def kv_copy_d2h(self, phys_ids: list[int], slots: list[int]) -> None:
        gpu_kv = self.kv_cache_at_layer
        host_kv = self.kv_cache_at_layer_host
        max_gpu = gpu_kv[0].shape[0]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for p in phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"swap_out: GPU phys_id {p} out of bounds [0, {max_gpu})")
        for s in slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"swap_out: CPU slot {s} out of bounds [0, {max_cpu})")
        src = torch.tensor(phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        dst = torch.tensor(slots, dtype=torch.long)
        for layer_idx in range(len(gpu_kv)):
            host_kv[layer_idx].index_copy_(0, dst, gpu_kv[layer_idx][src].cpu())

    def kv_copy_h2d(self, phys_ids: list[int], slots: list[int]) -> None:
        gpu_kv = self.kv_cache_at_layer
        host_kv = self.kv_cache_at_layer_host
        max_gpu = gpu_kv[0].shape[0]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for p in phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"swap_in: GPU phys_id {p} out of bounds [0, {max_gpu})")
        for s in slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"swap_in: CPU slot {s} out of bounds [0, {max_cpu})")
        dst = torch.tensor(phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        src = torch.tensor(slots, dtype=torch.long)
        for layer_idx in range(len(gpu_kv)):
            gpu_kv[layer_idx].index_copy_(0, dst, host_kv[layer_idx][src].to(gpu_kv[layer_idx].device))

    def kv_copy_d2d(self, src_phys_ids: list[int], dst_phys_ids: list[int]) -> None:
        gpu_kv = self.kv_cache_at_layer
        max_gpu = gpu_kv[0].shape[0]
        for p in src_phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"copy_d2d: src phys_id {p} out of bounds [0, {max_gpu})")
        for p in dst_phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"copy_d2d: dst phys_id {p} out of bounds [0, {max_gpu})")
        src = torch.tensor(src_phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        dst = torch.tensor(dst_phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        for layer_idx in range(len(gpu_kv)):
            gpu_kv[layer_idx].index_copy_(0, dst, gpu_kv[layer_idx][src])

    def kv_copy_h2h(self, src_slots: list[int], dst_slots: list[int]) -> None:
        host_kv = self.kv_cache_at_layer_host
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for s in src_slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"copy_h2h: src slot {s} out of bounds [0, {max_cpu})")
        for s in dst_slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"copy_h2h: dst slot {s} out of bounds [0, {max_cpu})")
        src = torch.tensor(src_slots, dtype=torch.long)
        dst = torch.tensor(dst_slots, dtype=torch.long)
        for layer_idx in range(len(host_kv)):
            host_kv[layer_idx].index_copy_(0, dst, host_kv[layer_idx][src])

    @torch.inference_mode()
    def build_model_inputs(self, batch: Batch) -> dict:
        """Materialize a Batch into the dict of torch tensors `fire_batch` expects."""
        if batch.has_speculative_inputs:
            return batch_tensors.build_spec_expanded_model_inputs(
                batch, self.config.device
            )
        return batch_tensors.build_model_inputs(batch, self.config.device)

    def build_sampling_metadata(self, batch: Batch) -> dict:
        """Materialize a Batch into the sampling-metadata dict `fire_batch` expects."""
        if batch.has_speculative_inputs:
            return batch_tensors.build_spec_expanded_sampling_metadata(
                batch, self.config.device, self.config.activation_dtype
            )
        return batch_tensors.build_sampling_metadata(
            batch, self.config.device, self.config.activation_dtype
        )

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

        # Bounds-check kv_page_indices/indptr/last_page_lens. The caller
        # (batch_tensors.build_model_inputs) is expected to attach pre-computed CPU
        # min/max in `inputs` so we don't have to issue 6 CUDA syncs every
        # forward pass. Falls back to GPU `.item()` if the precomputed
        # values are absent (legacy paths).
        kv_idx = inputs["kv_page_indices"]
        max_pages = self.kv_cache_at_layer[0].shape[0]
        page_size = self.kv_cache_at_layer[0].shape[2]
        if kv_idx.numel() > 0:
            kv_max = inputs.get("kv_page_indices_max")
            kv_min = inputs.get("kv_page_indices_min")
            if kv_max is None or kv_min is None:
                kv_max = kv_idx.max().item()
                kv_min = kv_idx.min().item()
            if kv_max >= max_pages or kv_min < 0:
                raise ValueError(
                    f"fire_batch: kv_page_indices out of bounds: "
                    f"min={kv_min} max={kv_max} max_pages={max_pages}"
                )

        kv_indptr = inputs["kv_page_indptr"]
        if kv_indptr.numel() > 0:
            indptr_last = inputs.get("kv_page_indptr_last")
            if indptr_last is None:
                indptr_last = kv_indptr[-1].item()
            if indptr_last > kv_idx.numel():
                raise ValueError(
                    f"fire_batch: kv_page_indptr last={indptr_last} exceeds "
                    f"kv_page_indices length={kv_idx.numel()}"
                )

        kv_last = inputs["kv_last_page_lens"]
        if kv_last.numel() > 0:
            last_max = inputs.get("kv_last_page_lens_max")
            last_min = inputs.get("kv_last_page_lens_min")
            if last_max is None or last_min is None:
                last_max = kv_last.max().item()
                last_min = kv_last.min().item()
            if last_max > page_size or last_min < 1:
                raise ValueError(
                    f"fire_batch: kv_last_page_lens out of range: "
                    f"min={last_min} max={last_max} page_size={page_size}"
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

    def capabilities(self):
        """Report this driver's resolved capacities up to pie's runtime.

        See `pie_driver_dev._bridge.capabilities.DriverCapabilities` for the contract.
        """
        from ._bridge.capabilities import DriverCapabilities

        hf_cfg = self.info.get("hf_config", {}) if isinstance(self.info, dict) else {}
        # max_model_len: prefer max_position_embeddings; some configs use
        # max_sequence_length instead.
        max_model_len = int(
            hf_cfg.get("max_position_embeddings")
            or hf_cfg.get("max_sequence_length")
            or 0
        )
        vocab_size = int(
            getattr(self.model_config, "num_vocabs", None)
            or getattr(self.model_config, "vocab_size", 0)
        )
        total_pages = int(self.config.total_pages or 0)
        max_forward_tokens = max(
            1, min(10240, total_pages * int(self.config.kv_page_size))
        )
        max_forward_requests = max(1, min(512, max_forward_tokens))
        unconstrained = (1 << 32) - 1
        return DriverCapabilities(
            total_pages=total_pages,
            kv_page_size=int(self.config.kv_page_size),
            swap_pool_size=int(self.swap_pool_size),
            max_forward_tokens=max_forward_tokens,
            max_forward_requests=max_forward_requests,
            max_page_refs=total_pages,
            max_logit_rows=unconstrained,
            max_prob_rows=unconstrained,
            max_custom_mask_bytes=unconstrained,
            max_sampler_rows=unconstrained,
            max_logprob_labels=unconstrained,
            arch_name=self.arch_type,
            vocab_size=vocab_size,
            max_model_len=max_model_len,
            activation_dtype=str(self.config.activation_dtype).removeprefix("torch."),
            # Native uses pie's BRLE mask path on every layer's flashinfer
            # call — fully respects user-supplied attention masks.
            supports_user_attention_mask=True,
            # Native owns the CMA-ES adapter math (rand_mv noise injection
            # at Q/K/V down/up). The vllm/sglang bridges raise.
            supports_adapters=True,
            snapshot_dir=str(self.snapshot_dir) if self.snapshot_dir else "",
        )
