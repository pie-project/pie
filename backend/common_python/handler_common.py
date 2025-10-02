"""TODO: Add module docstring."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

import message

# Safe import of adapter functionality
from adapter_import_utils import ensure_adapter_available

from debug_utils import is_tensor_debug_enabled, checkpoint_validation


class Handler:
    """TODO: Add class docstring."""

    max_num_kv_pages: int
    max_num_embeds: int
    max_num_adapters: int
    max_adapter_rank: int

    def __init__(
        self,
        config: dict,
        ops,  # Backend operations (FlashInferOps, MetalOps, etc.)
    ):
        """TODO: Add method docstring."""
        self.adapters = {}
        self.ops = ops  # backend operations

        # Put imports here to avoid circular import
        # pylint: disable=import-outside-toplevel
        from model_loader import load_model, load_model_info
        from model_factory import create_model_and_fusion_map

        self.model_info = load_model_info(config)
        self.kv_page_size = config["kv_page_size"]
        self.max_dist_size = config["max_dist_size"]
        self.max_num_embeds = config["max_num_embeds"]
        self.max_batch_tokens = config["max_batch_tokens"]
        self.max_num_adapters = config["max_num_adapters"]
        self.max_adapter_rank = config["max_adapter_rank"]
        self.dtype = getattr(torch, config["dtype"])
        self.device = config["device"]
        self.logits_dtype = getattr(torch, config["dtype"])

        # If `gpu_mem_headroom` is set by the user, then we will cap the KV
        # cache size so that there is some percentage of GPU memory left over
        # after loading the model and allocating the KV cache.
        if "gpu_mem_headroom" in config:
            adaptive_kv_cache_size = True
        else:
            adaptive_kv_cache_size = False

        # If the KV cache size is fixed, then we allocate the KV cache,
        # which are big contiguous tensors, before loading the model,
        # since during model loading, many temporary tensors are created,
        # which may lead to fragmentation and out-of-memory errors if big
        # tensors are not allocated up front.
        if not adaptive_kv_cache_size:
            self.max_num_kv_pages = config["max_num_kv_pages"]
            self.kv_cache_at_layer = [
                torch.zeros(
                    (
                        self.max_num_kv_pages,
                        2,
                        self.kv_page_size,
                        self.model_info.architecture.num_key_value_heads,
                        self.model_info.architecture.head_size,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.model_info.architecture.num_layers)
            ]

        self.embeds = torch.empty(
            (self.max_num_embeds, self.model_info.architecture.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        self.max_num_adapters,
                        self.max_adapter_rank * 3,
                        self.model_info.architecture.hidden_size,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.zeros(
                    (
                        self.max_num_adapters,
                        self.model_info.architecture.head_size
                        * (
                            self.model_info.architecture.num_query_heads
                            + self.model_info.architecture.num_key_value_heads * 2
                        ),
                        self.max_adapter_rank,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                ),
            )
            for _ in range(self.model_info.architecture.num_layers)
        ]

        self.lm = load_model(
            config,
            self.model_info,
            create_model_and_fusion_map,
        )

        # If `gpu_mem_headroom` is set by the user, then we have to allocate the KV
        # cache at the end and dynamically calculate the number of KV pages based on
        # the available GPU memory.
        if adaptive_kv_cache_size:
            # Calculate the available GPU memory for the KV cache after accounting for
            # the reserved GPU memory specified through `gpu_mem_headroom`.
            free_gpu_mem_bytes, total_gpu_mem_bytes = torch.cuda.mem_get_info(
                self.device
            )
            used_gpu_mem_bytes = total_gpu_mem_bytes - free_gpu_mem_bytes
            reserved_gpu_mem_percentage = config["gpu_mem_headroom"]
            useable_gpu_mem_bytes = total_gpu_mem_bytes * (
                1 - (reserved_gpu_mem_percentage / 100)
            )
            available_kv_cache_bytes = useable_gpu_mem_bytes - used_gpu_mem_bytes

            if available_kv_cache_bytes <= 0:
                raise ValueError(
                    "Not enough GPU memory available to allocate the KV cache. "
                    "Please decrease 'gpu_mem_headroom'."
                )

            # Calculate the number of KV pages based on the available GPU memory.
            self.max_num_kv_pages = int(
                available_kv_cache_bytes
                / (
                    self.kv_page_size
                    * 2
                    * self.model_info.architecture.num_key_value_heads
                    * self.model_info.architecture.head_size
                    * self.model_info.architecture.num_layers
                    * self.dtype.itemsize
                )
            )

            # If the user also specified "max_num_kv_pages", then we will use the
            # smaller of the two values.
            if "max_num_kv_pages" in config:
                if self.max_num_kv_pages > config["max_num_kv_pages"]:
                    self.max_num_kv_pages = config["max_num_kv_pages"]
                elif self.max_num_kv_pages < config["max_num_kv_pages"]:
                    print(
                        f"'max_num_kv_pages' is reduced to {self.max_num_kv_pages} "
                        "to respect 'gpu_mem_headroom'."
                    )

            self.kv_cache_at_layer = [
                torch.zeros(
                    (
                        self.max_num_kv_pages,
                        2,
                        self.kv_page_size,
                        self.model_info.architecture.num_key_value_heads,
                        self.model_info.architecture.head_size,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.model_info.architecture.num_layers)
            ]

        self.inter_fill_time = time.time()

    def handshake(
        self, reqs: list[message.HandshakeRequest]
    ) -> list[message.HandshakeResponse]:
        """Handle handshake requests."""
        resps = []
        for _ in reqs:
            # Request details not currently used

            resp = message.HandshakeResponse(
                version=self.model_info.version,
                model_name=self.model_info.name,
                model_traits=["todo"],
                model_description=self.model_info.description,
                prompt_template=self.model_info.template_content,
                prompt_template_type=self.model_info.template_type,
                prompt_stop_tokens=self.model_info.stop_tokens,
                kv_page_size=self.kv_page_size,
                max_batch_tokens=self.max_batch_tokens,
                resources={
                    0: self.max_num_kv_pages,
                    1: self.max_num_embeds,
                    2: self.max_num_adapters,
                },
                tokenizer_num_vocab=self.model_info.tokenizer.num_vocab,
                tokenizer_merge_table=self.model_info.tokenizer.merge_table,
                tokenizer_special_tokens=self.model_info.tokenizer.special_tokens,
                tokenizer_split_regex=self.model_info.tokenizer.split_regex,
                tokenizer_escape_non_printable=self.model_info.tokenizer.escape_non_printable,
            )
            resps.append(resp)
        return resps

    def query(self, reqs: list[message.QueryRequest]) -> list[message.QueryResponse]:
        """Handle query requests."""
        resps = []
        for req in reqs:
            value = "unknown query"
            match req.query:
                case "ping":
                    value = "pong"
            resp = message.QueryResponse(value=value)
            resps.append(resp)
        return resps

    def embed_image(self, reqs: list[message.EmbedImageRequest]):
        """
        Embeds images into the specified embed pointers.
        """
        for req in reqs:
            if len(req.embed_ptrs) > self.max_num_embeds:
                raise ValueError(
                    f"Number of embed pointers {len(req.embed_ptrs)} exceeds "
                    f"maximum {self.max_num_embeds}."
                )

            image_tensor = self.ops.decode_image(
                req.image_blob, dtype=self.dtype, device=self.device
            )
            image_embed = self.lm.model.embed_image(image_tensor)

            for i, ptr in enumerate(req.embed_ptrs):
                if ptr < 0 or ptr >= self.max_num_embeds:
                    raise ValueError(
                        f"Embed pointer {ptr} out of range [0, {self.max_num_embeds})."
                    )
                self.embeds[ptr].copy_(image_embed[i], non_blocking=True)

    @torch.inference_mode()
    def initialize_adapter(self, reqs: list[message.InitializeAdapterRequest]):
        """Initialize adapter functionality."""
        # cfg = self.lm.config  # Available if needed

        for _ in reqs:
            pass  # Request details not currently used

    @torch.inference_mode()
    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]):
        """Update adapter functionality."""
        for req in reqs:
            if req.adapter_ptr in self.adapters:
                pass

    @torch.inference_mode()
    @checkpoint_validation(
        checkpoint_name="handler_forward_pass",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def forward_pass(self, reqs: list[message.ForwardPassRequest]):
        """
        Processes a batch of forward pass requests through the language model.
        """
        # Sort requests by adapter to optimize the adapter subpass.
        reqs = sorted(reqs, key=lambda o: (o.adapter is None, o.adapter))

        # 1. Consolidate and process all requests into a single batch.
        batch = ForwardPassBatch(self)
        for req in reqs:
            batch.add_request(req)

        # 2. Finalize the batch to get model inputs as tensors.
        model_inputs = batch.finalize()

        # 3. Run the forward pass through the model.
        with _device_context(self.device):
            output_embeds = self.lm.model.forward(
                kv_cache_at_layer=self.kv_cache_at_layer, **model_inputs
            )

            # 4. Package the model outputs into response messages.
            responses = batch.package_responses(output_embeds)

        return responses

    def heartbeat(
        self, reqs: list[message.HeartbeatRequest]
    ) -> list[message.HeartbeatResponse]:
        """Handle heartbeat requests to keep the connection alive."""
        resps = []
        for _ in reqs:
            resps.append(message.HeartbeatResponse())
        return resps

    def upload_handler(self, reqs: list[message.UploadAdapterRequest]):
        """Handle adapter upload requests."""
        _ = reqs  # Parameter not currently used
        raise NotImplementedError("upload_handler not yet implemented")

    def download_handler(
        self, reqs: list[message.DownloadAdapterRequest]
    ) -> list[message.DownloadAdapterResponse]:
        """Handle adapter download requests."""
        _ = reqs  # Parameter not currently used
        raise NotImplementedError("download_handler not yet implemented")


@contextmanager
def _device_context(device: str):
    """Context manager that activates the appropriate device when possible."""
    if not device:
        yield
        return

    if device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(device):
            yield
        return

    # For CPU/MPS/Metal we rely on tensors already being on the target device.
    with nullcontext():
        yield


def _decode_brle(brle_buffer: list[int]) -> np.ndarray:
    """
    Decodes a Binary Run-Length Encoded buffer into a boolean numpy array.
    The format assumes alternating runs of False and True, starting with False.
    """
    if not brle_buffer:
        return np.array([], dtype=bool)

    total_size = sum(brle_buffer)
    if total_size == 0:
        return np.array([], dtype=bool)

    decoded_array = np.empty(total_size, dtype=bool)
    current_pos = 0
    value = True  # In attention masking, True means attend.
    for run_len in brle_buffer:
        if run_len > 0:
            decoded_array[current_pos : current_pos + run_len] = value
        current_pos += run_len
        value = not value  # Flip value for the next run
    return decoded_array


class ForwardPassBatch:
    """Consolidates and processes a batch of forward pass requests."""

    # Static constant for the maximum top_k value for distributions.
    TOP_K_MAX_BOUND = 1024

    def __init__(self, handler: Handler):
        """Initializes the batch processor."""
        self._handler = handler
        self.logits_dtype = getattr(handler, "logits_dtype", handler.dtype)
        self._original_reqs: list[message.ForwardPassRequest] = []

        # Inputs for the model
        self.adapter_indices: list[int] = []
        self.seeds: list[int] = []
        self.kv_page_indices: list[int] = []
        self.kv_page_indptr: list[int] = [0]
        self.kv_last_page_lengths: list[int] = []
        self.qo_indptr: list[int] = [0]
        self.attention_masks: list[np.ndarray] = []
        self.batch_token_ids: list[int] = []
        self.batch_position_ids: list[int] = []

        # Tracking state
        self.total_tokens_in_batch: int = 0
        self.single_token_inference_mode: bool = True
        self.adapter_subpass_needed: bool = False

        # Output mapping for all logit-based operations (dists and sampling)
        self.indices_for_logits: list[int] = []
        self.indices_for_embed_storage: list[int] = []
        self.embed_storage_pointers: list[int] = []

        # Sampler type and consolidated parameters
        self.sampler_type: list[int] = []
        self.sampler_params: list[dict] = []

    def add_request(self, req: message.ForwardPassRequest):
        """Processes and adds a single request to the batch."""
        self._original_reqs.append(req)

        # Handle adapter information
        if req.adapter is not None and req.adapter in self._handler.adapters:
            seed = req.adapter_seed if req.adapter_seed is not None else 0
            self.seeds.extend([seed] * len(req.input_tokens))
            self.adapter_indices.append(req.adapter)
            self.adapter_subpass_needed = True

        # Handle KV cache pages
        kv_page_ptrs = req.kv_page_ptrs or []
        self.kv_page_indices.extend(kv_page_ptrs)
        self.kv_page_indptr.append(len(self.kv_page_indices))
        self.kv_last_page_lengths.append(req.kv_page_last_len or 0)

        # Handle output mappings for embeddings that need to be stored
        output_embed_indices = req.output_embed_indices or []
        output_embed_ptrs = req.output_embed_ptrs or []
        if len(output_embed_indices) != len(output_embed_ptrs):
            raise ValueError(
                f"Mismatch between output_embed_indices length ({len(output_embed_indices)}) "
                f"and output_embed_ptrs length ({len(output_embed_ptrs)})"
            )
        for token_idx, storage_ptr in zip(output_embed_indices, output_embed_ptrs):
            self.indices_for_embed_storage.append(
                token_idx + self.total_tokens_in_batch
            )
            self.embed_storage_pointers.append(storage_ptr)

        # Handle output mappings for tokens requiring logits.
        output_token_indices = req.output_token_indices or []
        for token_idx in output_token_indices:
            self.indices_for_logits.append(token_idx + self.total_tokens_in_batch)

        # Extract sampler configurations.
        # sampler_idx=0 is for distributions, existing samplers are shifted by +1.
        output_token_samplers = req.output_token_samplers or []
        for sampler_config in output_token_samplers:
            params = {}
            sampler_idx = sampler_config["sampler"]
            self.sampler_type.append(sampler_idx)

            if sampler_idx == 0:
                params["top_k"] = min(
                    sampler_config.get("top_k", self._handler.max_dist_size),
                    self._handler.max_dist_size,
                )
            else:
                params["top_k"] = sampler_config.get("top_k", 0)
                params["top_p"] = sampler_config.get("top_p", 1.0)
                params["min_p"] = sampler_config.get("min_p", 0.0)

            params["temperature"] = sampler_config.get("temperature", 1.0)
            self.sampler_params.append(params)

        # Handle input tokens and positions
        self.batch_token_ids.extend(req.input_tokens)
        self.batch_position_ids.extend(req.input_token_positions)
        self.total_tokens_in_batch += len(req.input_tokens)
        self.qo_indptr.append(self.total_tokens_in_batch)

        if len(req.input_tokens) > 1:
            self.single_token_inference_mode = False

        attention_mask = self._generate_mask_for_request(req)
        self.attention_masks.append(attention_mask)

    def _generate_mask_for_request(self, req: message.ForwardPassRequest) -> np.ndarray:
        """Generates the custom attention mask for a single request."""
        if len(req.mask) != len(req.input_tokens):
            raise ValueError(
                f"Mismatch between number of masks ({len(req.mask)}) and "
                f"input tokens ({len(req.input_tokens)})."
            )

        kv_page_ptrs = req.kv_page_ptrs or []
        kv_page_last_len = req.kv_page_last_len or 0

        # Ensure we have at least one page for proper computation
        if len(kv_page_ptrs) >= 1:
            sequence_length = (
                self._handler.kv_page_size * (len(kv_page_ptrs) - 1) + kv_page_last_len
            )
        else:
            sequence_length = kv_page_last_len

        # Validate sequence_length is sufficient for input tokens
        input_token_count = len(req.input_tokens)
        if sequence_length < input_token_count:
            raise ValueError(
                f"Insufficient sequence length ({sequence_length}) for input tokens "
                f"({input_token_count}). Sequence length must be at least equal to "
                f"the number of input tokens."
            )

        context_length = sequence_length - input_token_count

        request_attention_mask = np.zeros(
            (len(req.input_tokens), sequence_length), dtype=np.bool_
        )
        for i, brle_buffer in enumerate(req.mask):
            decoded_mask = _decode_brle(brle_buffer)
            expected_len = context_length + i + 1
            if len(decoded_mask) != expected_len:
                raise ValueError(
                    f"Decoded mask for token {i} has length {len(decoded_mask)}, "
                    f"but expected {expected_len}"
                )
            request_attention_mask[i, :expected_len] = decoded_mask

        return request_attention_mask.flatten()

    def finalize(self) -> dict:
        """Finalizes batch preparation, creating tensors and the adapter subpass."""
        device = self._handler.device

        adapter_subpass = None
        if self.adapter_subpass_needed:
            adapter_subpass_class = ensure_adapter_available()
            seeds_tensor = torch.as_tensor(self.seeds, device=device, dtype=torch.long)
            adapter_subpass = adapter_subpass_class(
                adapter_at_layer=self._handler.adapter_at_layer,
                adapter_indices=self.adapter_indices,
                adapter_extras=self._handler.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=self.qo_indptr,
            )

        batched_attention_mask = (
            np.concatenate(self.attention_masks)
            if self.attention_masks
            else np.array([], dtype=np.bool_)
        )
        token_ids_tensor = torch.as_tensor(
            self.batch_token_ids, device=device, dtype=torch.int32
        )
        input_embeds = self._handler.lm.model.embed_tokens(token_ids_tensor)

        if input_embeds.numel() and is_tensor_debug_enabled():
            embed_min, embed_max = input_embeds.aminmax()
            has_nan = torch.isnan(input_embeds).any().item()
            has_inf = torch.isinf(input_embeds).any().item()
            print(
                "[MetalTensorDebug] stage=input_embeds",
                "dtype=",
                input_embeds.dtype,
                "device=",
                input_embeds.device,
                "min=",
                float(embed_min),
                "max=",
                float(embed_max),
                "has_nan=",
                bool(has_nan),
                "has_inf=",
                bool(has_inf),
            )

        return {
            "input_embeds": input_embeds,
            "position_ids": torch.as_tensor(
                self.batch_position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lengths, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": self.single_token_inference_mode,
            "adapter_subpass": adapter_subpass,
        }

    def package_responses(
        self, output_embeds: torch.Tensor
    ) -> list[message.ForwardPassResponse]:
        """Packages the model outputs into responses for each original request."""
        # Handle storing specified embeddings
        if self.indices_for_embed_storage:
            embeddings_to_store = output_embeds[self.indices_for_embed_storage]
            for i, ptr in enumerate(self.embed_storage_pointers):
                self._handler.embeds[ptr].copy_(
                    embeddings_to_store[i], non_blocking=True
                )

        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[])
                for _ in self._original_reqs
            ]

        # Calculate logits for all required tokens (both dists and samples)
        logits_input = output_embeds[self.indices_for_logits]
        if logits_input.dtype != self.logits_dtype:
            logits_input = logits_input.to(self.logits_dtype)

        logits = self._handler.lm.lm_head(logits_input)

        # Promote logits to handler dtype for numerically stable softmax on Metal/MPS
        if logits.dtype != self.logits_dtype:
            logits = logits.to(dtype=self.logits_dtype)

        # Apply temperature scaling to all logits
        temperatures = torch.tensor(
            [p["temperature"] for p in self.sampler_params],
            device=self._handler.device,
            dtype=self.logits_dtype,
        ).unsqueeze(1)
        scaled_logits = logits / torch.clamp(temperatures, min=1e-6)

        # We compute probabilities for the entire batch of logit requests
        probs = torch.softmax(scaled_logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities produced by LM head")

        # Group requests by sampler type for efficient batch processing
        sampler_groups = {}
        for i, sampler_idx in enumerate(self.sampler_type):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        num_logit_requests = len(self.indices_for_logits)
        # Initialize result containers. Using lists of Nones helps place results correctly.
        final_dists: list[tuple[list[int], list[float]] | None] = [
            None
        ] * num_logit_requests
        final_tokens_tensor = torch.empty(
            num_logit_requests, dtype=torch.long, device=self._handler.device
        )

        for sampler_idx, indices in sampler_groups.items():
            if not indices:
                continue

            indices_tensor = torch.tensor(
                indices, device=self._handler.device, dtype=torch.long
            )
            group_probs = probs.index_select(0, indices_tensor)

            # Handle distributions (sampler_idx=0)
            if sampler_idx == 0:
                group_top_k = [self.sampler_params[i]["top_k"] for i in indices]
                max_k = max(group_top_k) if group_top_k else 0
                if max_k > 0:
                    topk_vals, topk_inds = torch.topk(group_probs, k=max_k, sorted=True)
                    for i, original_idx in enumerate(indices):
                        k = self.sampler_params[original_idx]["top_k"]
                        ids = topk_inds[i, :k].tolist()
                        vals = topk_vals[i, :k].tolist()
                        final_dists[original_idx] = (ids, vals)

            # Handle sampling operations (sampler_idx > 0)
            else:
                sampled = None
                if sampler_idx == 1:  # Old 0: sampling_from_probs
                    sampled = self._handler.ops.sampling_from_probs(group_probs)
                elif sampler_idx == 2:  # Old 1: top_p_sampling_from_probs
                    top_p_vals = torch.tensor(
                        [self.sampler_params[i]["top_p"] for i in indices],
                        device=self._handler.device,
                        dtype=self._handler.dtype,
                    )
                    sampled = self._handler.ops.top_p_sampling_from_probs(
                        group_probs, top_p=top_p_vals
                    )
                elif sampler_idx == 3:  # Old 2: top_k_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [self.sampler_params[i]["top_k"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.long,
                    )
                    sampled = self._handler.ops.top_k_sampling_from_probs(
                        group_probs, top_k=top_k_vals
                    )
                elif sampler_idx == 4:  # Old 3: min_p_sampling_from_probs
                    min_p_vals = torch.tensor(
                        [self.sampler_params[i]["min_p"] for i in indices],
                        device=self._handler.device,
                        dtype=self._handler.dtype,
                    )
                    sampled = self._handler.ops.min_p_sampling_from_probs(
                        group_probs, min_p=min_p_vals
                    )
                elif sampler_idx == 5:  # Old 4: top_k_top_p_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [self.sampler_params[i]["top_k"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.long,
                    )
                    top_p_vals = torch.tensor(
                        [self.sampler_params[i]["top_p"] for i in indices],
                        device=self._handler.device,
                        dtype=self._handler.dtype,
                    )
                    sampled = self._handler.ops.top_k_top_p_sampling_from_probs(
                        group_probs, top_k=top_k_vals, top_p=top_p_vals
                    )
                else:
                    raise ValueError(f"Unknown sampler index: {sampler_idx}")

                # Place sampled tokens into the main tensor at their original batch positions
                # Ensure sampled tokens have the correct dtype (torch.long for token indices)
                if sampled.dtype != torch.long:
                    sampled = sampled.to(torch.long)
                final_tokens_tensor.scatter_(0, indices_tensor, sampled)

        # Distribute batched results back to individual responses
        responses = []
        cursor = 0
        for req in self._original_reqs:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists = []
            request_tokens = []

            # Iterate through the slice of results belonging to this request
            for i in range(cursor, cursor + num_outputs):
                if self.sampler_type[i] == 0:  # This was a distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:  # This was a sampling request
                    request_tokens.append(final_tokens_tensor[i].item())

            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses
