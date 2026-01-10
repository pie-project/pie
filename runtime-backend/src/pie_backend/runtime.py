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

from .config import RuntimeConfig
from .batching import Batch
from .loader import ModelLoader
from .adapter import AdapterSubpass, CmaesAdapter
from .model import llama3, qwen2, qwen3, common
from .model.chat_templates import (
    Llama3Template,
    Qwen2_5Template,
    Qwen3Template,
    GPTOSSTemplate,
    ChatTemplate,
)

# gpt_oss requires CUDA-only features, only import on CUDA platforms
if torch.cuda.is_available():
    from .model import gpt_oss
from . import message
from . import hf_utils

# Re-export RuntimeConfig for backward compatibility
__all__ = ["Runtime", "RuntimeConfig"]


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

    def __init__(self, config: RuntimeConfig, log_queue: object):
        """
        Initialize the runtime.

        Args:
            config: Runtime configuration
            log_queue: Queue for sending logs back to controller
        """
        self.config = config
        self.log_queue = log_queue
        self.adapters = {}

        # Initialize seeds
        msg = f"Initializing with random seed: {config.random_seed}"
        self.log_queue.put({"message": msg, "level": "DEBUG"})

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        # Load model weights using ModelLoader
        loader = ModelLoader(config, log_queue=log_queue)

        msg = "Loading model weights"
        self.log_queue.put({"message": msg, "level": "DEBUG"})

        weights, normalized_arch, self.info = loader.load()

        # Store snapshot_dir for tokenizer loading
        self.snapshot_dir = loader.snapshot_dir

        msg = "Loaded model weights"
        self.log_queue.put({"message": msg, "level": "DEBUG"})

        # Store architecture type
        self.type = self.info["architecture"]["type"]

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
                )
                # Create adapter cache
                self.adapter_at_layer = gpt_oss.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = gpt_oss.create_kv_cache(
                    self.model_config, config
                )

            case _:
                raise ValueError(f"Unsupported architecture type: {self.type}")

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

        if self.type == "llama3" or self.type == "l4ma":
            template = Llama3Template
        elif self.type == "qwen2":
            template = Qwen2_5Template
        elif self.type == "qwen3":
            template = Qwen3Template
        elif self.type == "gptoss":
            template = GPTOSSTemplate

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
            utils.broadcast_struct(msg, src=0, device=self.config.device)

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
            utils.broadcast_struct(msg, src=0, device=self.config.device)

        self._update_adapter(**args)

    def upload_adapter(self, req: message.UploadAdapterRequest) -> None:
        """Upload adapter weights."""
        data = req.adapter_data
        if isinstance(data, list):
            data = bytes(data)

        args = {"adapter_ptr": req.adapter_ptr, "name": req.name, "data": data}

        if self.config.world_size > 1:
            msg = {"type": "UPLOAD_ADAPTER", "kwargs": args}
            utils.broadcast_struct(msg, src=0, device=self.config.device)

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

    def download_adapter(
        self, req: message.DownloadAdapterRequest
    ) -> message.DownloadAdapterResponse:
        """Download adapter weights."""
        args = {"adapter_ptr": req.adapter_ptr, "name": req.name}

        if self.config.world_size > 1:
            msg = {"type": "DOWNLOAD_ADAPTER", "kwargs": args}
            utils.broadcast_struct(msg, src=0, device=self.config.device)

        data = self._download_adapter(**args)
        return message.DownloadAdapterResponse(adapter_data=data)

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
        world_size = self.config.world_size
        gpu_rank = self.config.rank

        local_num_q_heads = cfg.num_q_heads // world_size
        local_num_kv_heads = cfg.num_kv_heads // world_size

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
            world_size=world_size,
        )

    @torch.inference_mode()
    def _update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                adapter.update(scores, seeds, max_sigma)

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
                msg = utils.broadcast_struct(None, src=0, device=device)
            except Exception:
                break

            if shutdown_requested:
                break

            if msg == "STOP":
                break

            if isinstance(msg, dict):
                msg_type = msg.get("type")

                if msg_type == "STEP":
                    # Execute inference step
                    inputs = msg["inputs"]
                    sampling_metadata = msg["sampling_metadata"]
                    try:
                        self._run_step(inputs, sampling_metadata)
                    except Exception as e:
                        # Log error but continue - don't let one error hang the whole system
                        print(f"Worker {self.config.rank} _run_step error: {e}")
                        # Sync CUDA to clear any pending operations
                        torch.cuda.synchronize()

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
        if self.config.world_size > 1:
            torch.distributed.barrier()

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
    # Pycrust RPC Method Wrappers
    # ========================================================================

    def handshake_rpc(self, **kwargs) -> dict:
        """Handle handshake via pycrust RPC."""
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
        }

    def query_rpc(self, **kwargs) -> dict:
        """Handle query via pycrust RPC."""
        req = message.QueryRequest(**kwargs)
        resp = self.query(req)
        return {"value": resp.value}

    @torch.inference_mode()
    def fire_batch(self, **kwargs) -> dict:
        """
        Execute a pre-batched forward pass from Rust via pycrust RPC.

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
        from .batching import Batch, _decode_brle

        # Build internal Batch object from pre-batched data
        batch = self._build_batch_from_request(kwargs)

        # Get model inputs and sampling metadata
        device = self.config.device
        inputs = batch.get_model_inputs(device)
        sampling_metadata = batch.get_sampling_metadata(
            device, self.config.activation_dtype
        )

        # Broadcast to workers if multi-GPU
        if self.config.world_size > 1:
            msg = {
                "type": "STEP",
                "inputs": inputs,
                "sampling_metadata": sampling_metadata,
            }
            utils.broadcast_struct(msg, src=0, device=device)

        # Execute inference
        sampling_results = self._run_step(inputs, sampling_metadata)

        # Package responses
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

        return {"results": results}

    def _build_batch_from_request(self, args: dict) -> Batch:
        """
        Convert BatchedForwardPassRequest dict to internal Batch object.

        Args:
            args: Dictionary with batched request fields

        Returns:
            Batch object ready for inference
        """
        from .batching import Batch, _decode_brle

        batch = Batch()

        # Direct assignments (already concatenated by Rust)
        batch.token_ids = list(args["token_ids"])
        batch.position_ids = list(args["position_ids"])
        batch.kv_page_indices = list(args["kv_page_indices"])
        batch.kv_page_indptr = list(args["kv_page_indptr"])
        batch.kv_last_page_lens = list(args["kv_last_page_lens"])
        batch.qo_indptr = list(args["qo_indptr"])
        batch.single_token_mode = args["single_token_mode"]
        batch.total_tokens = len(args["token_ids"])

        # Process per-request data
        masks = args["masks"]
        adapter_indices = args["adapter_indices"]
        adapter_seeds = args["adapter_seeds"]
        output_token_indices = args["output_token_indices"]
        output_token_samplers = args["output_token_samplers"]
        output_embed_ptrs = args["output_embed_ptrs"]
        output_embed_indices = args["output_embed_indices"]

        num_requests = len(masks)
        token_offset = 0

        for i in range(num_requests):
            # Calculate tokens for this request
            req_token_count = args["qo_indptr"][i + 1] - args["qo_indptr"][i]

            # Calculate sequence length from KV pages
            kv_start = args["kv_page_indptr"][i]
            kv_end = args["kv_page_indptr"][i + 1]
            num_pages = kv_end - kv_start
            kv_last_len = args["kv_last_page_lens"][i]

            if num_pages >= 1:
                seq_len = self.config.kv_page_size * (num_pages - 1) + kv_last_len
            else:
                seq_len = kv_last_len

            context_len = seq_len - req_token_count

            # Decode BRLE masks
            req_masks = masks[i]
            attention_mask = np.zeros((req_token_count, seq_len), dtype=np.bool_)
            for j, brle in enumerate(req_masks):
                decoded = _decode_brle(brle)
                expected_len = context_len + j + 1
                if len(decoded) >= expected_len:
                    attention_mask[j, :expected_len] = decoded[:expected_len]

            batch.attention_masks.append(attention_mask.flatten())

            # Handle adapters
            adapter_idx = adapter_indices[i]
            if adapter_idx is not None and adapter_idx in self.adapters:
                seed = adapter_seeds[i] if adapter_seeds[i] is not None else 0
                batch.adapter_seeds.extend([seed] * req_token_count)
                batch.adapter_indices.append(adapter_idx)
                batch.adapter_subpass_needed = True

            # Handle output indices (adjust for batch offset)
            for idx in output_token_indices[i]:
                batch.indices_for_logits.append(idx + token_offset)

            # Handle samplers
            for sampler_config in output_token_samplers[i]:
                params = {}
                sampler_idx = sampler_config.get("sampler", 1)
                batch.sampler_types.append(sampler_idx)

                if sampler_idx == 0:
                    params["top_k"] = min(
                        sampler_config.get("top_k", self.config.max_dist_size),
                        self.config.max_dist_size,
                    )
                else:
                    params["top_k"] = sampler_config.get("top_k", 0)
                    params["top_p"] = sampler_config.get("top_p", 1.0)
                    params["min_p"] = sampler_config.get("min_p", 0.0)

                params["temperature"] = sampler_config.get("temperature", 1.0)
                batch.sampler_params.append(params)

            # Handle embed outputs
            for idx, ptr in zip(output_embed_indices[i], output_embed_ptrs[i]):
                batch.indices_for_embed_storage.append(idx + token_offset)
                batch.embed_storage_pointers.append(ptr)

            # Create dummy request for response packaging
            dummy_req = message.ForwardPassRequest(
                input_tokens=[],
                input_token_positions=[],
                input_embed_ptrs=[],
                input_embed_positions=[],
                adapter=adapter_indices[i],
                adapter_seed=adapter_seeds[i],
                mask=[],
                output_token_indices=output_token_indices[i],
                output_token_samplers=output_token_samplers[i],
            )
            batch.requests.append(dummy_req)

            token_offset += req_token_count

        return batch

    def embed_image_rpc(self, **kwargs) -> None:
        """Handle embed_image via pycrust RPC."""
        req = message.EmbedImageRequest(**kwargs)
        self.embed_image(req)

    def initialize_adapter_rpc(self, **kwargs) -> None:
        """Handle initialize_adapter via pycrust RPC."""
        req = message.InitializeAdapterRequest(**kwargs)
        self.initialize_adapter(req)

    def update_adapter_rpc(self, **kwargs) -> None:
        """Handle update_adapter via pycrust RPC."""
        req = message.UpdateAdapterRequest(**kwargs)
        self.update_adapter(req)

    def upload_adapter_rpc(self, **kwargs) -> None:
        """Handle upload_adapter via pycrust RPC."""
        req = message.UploadAdapterRequest(**kwargs)
        self.upload_adapter(req)

    def download_adapter_rpc(self, **kwargs) -> bytes:
        """Handle download_adapter via pycrust RPC."""
        req = message.DownloadAdapterRequest(**kwargs)
        resp = self.download_adapter(req)
        return resp.adapter_data

    def shutdown(self):
        """
        Cleanup runtime resources.

        For Rank 0 in multi-GPU setup, this broadcasts the 'STOP' signal to workers.
        """
        if self.config.world_size > 1 and self.config.rank == 0:
            print("Broadcasting STOP signal to workers...")
            utils.broadcast_struct("STOP", src=0, device=self.config.device)
