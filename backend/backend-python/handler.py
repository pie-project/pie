# pylint: disable-all
# type: ignore
# Ignoring checks for pylint and pyright since we are actively working on this file

"""TODO: Add module docstring."""

from __future__ import annotations

import time

import numpy as np
import torch

import message
from adapter import AdapterSubpass
import flashinfer as ops

from config.common import ModelInfo


class Handler:
    """TODO: Add class docstring."""

    max_num_kv_pages: int
    max_num_embeds: int
    max_num_adapters: int
    max_adapter_rank: int

    def __init__(
        self,
        model,
        model_info: ModelInfo,
        kv_page_size: int,
        max_dist_size: int,
        max_num_kv_pages: int,
        max_num_embeds: int,
        max_num_adapters: int,
        max_adapter_rank: int,
        dtype: torch.dtype,
        device: str,
    ):
        """TODO: Add method docstring."""
        self.adapters = {}

        self.lm = model
        self.model_info = model_info
        self.kv_page_size = kv_page_size
        self.max_dist_size = max_dist_size
        self.max_num_kv_pages = max_num_kv_pages
        self.max_num_embeds = max_num_embeds
        self.max_num_adapters = max_num_adapters
        self.max_adapter_rank = max_adapter_rank
        self.dtype = dtype
        self.device = device

        self.kv_cache_at_layer = [
            torch.zeros(
                (
                    max_num_kv_pages,
                    2,
                    kv_page_size,
                    self.lm.config.num_key_value_heads,
                    self.lm.config.head_size,
                ),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.lm.config.num_layers)
        ]

        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        max_num_adapters,
                        self.lm.config.hidden_size,
                        max_adapter_rank * 3,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    (
                        max_num_adapters,
                        max_adapter_rank,
                        self.lm.config.head_size
                        * (
                            self.lm.config.num_query_heads
                            + self.lm.config.num_key_value_heads * 2
                        ),
                    ),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(self.lm.config.num_layers)
        ]

        self.embeds = torch.empty(
            (max_num_embeds, self.lm.config.hidden_size), device=device, dtype=dtype
        )

        self.inter_fill_time = time.time()

    def handshake(
        self, reqs: list[message.HandshakeRequest]
    ) -> list[message.HandshakeResponse]:
        resps = []
        for req in reqs:
            # req.version

            resp = message.HandshakeResponse(
                version=self.model_info.version,
                model_name=self.model_info.name,
                model_traits=["todo"],
                model_description=self.model_info.description,
                prompt_template=self.model_info.template_content,
                prompt_template_type=self.model_info.template_type,
                prompt_stop_tokens=self.model_info.stop_tokens,
                kv_page_size=self.kv_page_size,
                resources={
                    0: self.max_num_kv_pages,
                    1: self.max_num_embeds,
                    2: self.max_num_adapters,
                },
                tokenizer_merge_table=self.model_info.tokenizer.merge_table,
                tokenizer_special_tokens=self.model_info.tokenizer.special_tokens,
                tokenizer_split_regex=self.model_info.tokenizer.split_regex,
                tokenizer_escape_non_printable=self.model_info.tokenizer.escape_non_printable,
            )
            resps.append(resp)
        return resps

    def heartbeat(
        self, reqs: list[message.HeartbeatRequest]
    ) -> list[message.HeartbeatResponse]:
        resps = []
        for req in reqs:
            resp = message.HeartbeatResponse()
            resps.append(resp)
        return resps

    def query(self, reqs: list[message.QueryRequest]) -> list[message.QueryResponse]:
        resps = []
        for req in reqs:
            value = "unknown query"
            match req.query:
                case "ping":
                    value = "pong"
            resp = message.QueryResponse(value=value)
            resps.append(resp)
        return resp

    def embed_image(self, reqs: list[message.EmbedImageRequest]):
        """
        Embeds images into the specified embed pointers.
        """
        for req in reqs:
            if len(req.embed_ptrs) > self.max_num_embeds:
                raise ValueError(
                    f"Number of embed pointers {len(req.embed_ptrs)} exceeds maximum {self.max_num_embeds}."
                )

            image_tensor = ops.image.decode_image(
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

        cfg = self.lm.config

        for req in reqs:
            self.adapters[req.adapter_ptr] = CmaesAdapter(
                adapter_id=req.adapter_ptr,
                adapter_at_layer=self.adapter_at_layer,
                rank=req.rank,
                alpha=req.alpha,
                in_features=cfg.hidden_size,
                out_features=[
                    cfg.head_size * cfg.num_query_heads,
                    cfg.head_size * cfg.num_key_value_heads,
                    cfg.head_size * cfg.num_key_value_heads,
                ],
                num_layers=cfg.num_layers,
                population_size=req.population_size,
                mu_fraction=req.mu_fraction,
                initial_sigma=req.initial_sigma,
                min_sigma=1e-7,
                min_var=1e-8,
                max_var=1e4,
                device=self.device,
                dtype=self.dtype,
            )

    @torch.inference_mode()
    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]):

        for req in reqs:
            if req.adapter_ptr in self.adapters:
                adapter = self.adapters[req.adapter_ptr]
                if isinstance(adapter, CmaesAdapter):
                    adapter.update(req.scores, req.seeds, req.max_sigma)

    @torch.inference_mode()
    def upload_adapter(self, reqs: list[message.UploadAdapterRequest]):
        for req in reqs:
            if req.adapter_ptr in self.adapters:
                adapter = self.adapters[req.adapter_ptr]
                if isinstance(adapter, CmaesAdapter):
                    adapter.upload(req.adapter_data)

    @torch.inference_mode()
    def download_adapter(
        self, reqs: list[message.DownloadAdapterRequest]
    ) -> list[message.DownloadAdapterResponse]:
        resps = []
        for req in reqs:
            if req.adapter_ptr in self.adapters:
                adapter = self.adapters[req.adapter_ptr]
                if isinstance(adapter, CmaesAdapter):
                    data = adapter.download()
                    resp = message.DownloadAdapterResponse(adapter_data=data)
                    resps.append(resp)

        return resps

    @torch.inference_mode()
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
        with torch.cuda.device(self.device):
            output_embeds = self.lm.model.forward(
                kv_cache_at_layer=self.kv_cache_at_layer, **model_inputs
            )

            # 4. Package the model outputs into response messages.
            responses = batch.package_responses(output_embeds)

        return responses


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
        self.kv_page_indices.extend(req.kv_page_ptrs)
        self.kv_page_indptr.append(len(self.kv_page_indices))
        self.kv_last_page_lengths.append(req.kv_page_last_len)

        # Handle output mappings for embeddings that need to be stored
        for token_idx, storage_ptr in zip(
            req.output_embed_indices, req.output_embed_ptrs
        ):
            self.indices_for_embed_storage.append(
                token_idx + self.total_tokens_in_batch
            )
            self.embed_storage_pointers.append(storage_ptr)

        # Handle output mappings for tokens requiring logits.
        for token_idx in req.output_token_indices:
            self.indices_for_logits.append(token_idx + self.total_tokens_in_batch)

        # Extract _sampler configurations.
        # sampler_idx=0 is for distributions, existing samplers are shifted by +1.
        for sampler_config in req.output_token_samplers:
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

        sequence_length = (
            self._handler.kv_page_size * (len(req.kv_page_ptrs) - 1)
            + req.kv_page_last_len
        )
        context_length = sequence_length - len(req.input_tokens)

        request_attention_mask = np.zeros(
            (len(req.input_tokens), sequence_length), dtype=np.bool_
        )
        for i, brle_buffer in enumerate(req.mask):
            decoded_mask = _decode_brle(brle_buffer)
            expected_len = context_length + i + 1
            if len(decoded_mask) != expected_len:
                raise ValueError(
                    f"Decoded mask for token {i} has length {len(decoded_mask)}, but expected {expected_len}"
                )
            request_attention_mask[i, :expected_len] = decoded_mask

        return request_attention_mask.flatten()

    def finalize(self) -> dict:
        """Finalizes batch preparation, creating tensors and the adapter subpass."""
        device = self._handler.device
        # print('seeds', self.seeds)
        # print('adapter_indices', self.adapter_indices)
        # print('qo_indptr', self.qo_indptr)
        # print('kv_page_indices', self.kv_page_indices)
        # print('kv_page_indptr', self.kv_page_indptr)
        # print('kv_last_page_lengths', self.kv_last_page_lengths)
        # #print('attention_masks', self.attention_masks)
        # print('batch_token_ids', self.batch_token_ids)
        # print('batch_position_ids', self.batch_position_ids)
        # print('single_token_inference_mode', self.single_token_inference_mode)
        adapter_subpass = None
        if self.adapter_subpass_needed:
            seeds_tensor = torch.as_tensor(self.seeds, device=device, dtype=torch.long)
            adapter_subpass = AdapterSubpass(
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
        logits = self._handler.lm.lm_head(output_embeds[self.indices_for_logits])

        # Apply temperature scaling to all logits
        temperatures = torch.tensor(
            [p["temperature"] for p in self.sampler_params],
            device=self._handler.device,
            dtype=self._handler.dtype,
        ).unsqueeze(1)
        scaled_logits = logits / torch.clamp(temperatures, min=1e-6)

        # We compute probabilities for the entire batch of logit requests
        probs = torch.softmax(scaled_logits, dim=-1)

        # Group requests by _sampler type for efficient batch processing
        sampler_groups = {}
        for i, sampler_idx in enumerate(self.sampler_type):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        num_logit_requests = len(self.indices_for_logits)
        # Initialize result containers. Using lists of Nones helps place results correctly.
        final_dists = [None] * num_logit_requests
        final_tokens_tensor = torch.empty(
            num_logit_requests, dtype=torch.int32, device=self._handler.device
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
                    sampled = ops.sampling.sampling_from_probs(group_probs)
                elif sampler_idx == 2:  # Old 1: top_p_sampling_from_probs
                    top_p_vals = torch.tensor(
                        [self.sampler_params[i]["top_p"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.float32,
                    )
                    sampled = ops.sampling.top_p_sampling_from_probs(
                        group_probs, top_p=top_p_vals
                    )
                elif sampler_idx == 3:  # Old 2: top_k_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [self.sampler_params[i]["top_k"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.int32,
                    )
                    sampled = ops.sampling.top_k_sampling_from_probs(
                        group_probs, top_k=top_k_vals
                    )
                elif sampler_idx == 4:  # Old 3: min_p_sampling_from_probs
                    min_p_vals = torch.tensor(
                        [self.sampler_params[i]["min_p"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.float32,
                    )
                    sampled = ops.sampling.min_p_sampling_from_probs(
                        group_probs, min_p=min_p_vals
                    )
                elif sampler_idx == 5:  # Old 4: top_k_top_p_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [self.sampler_params[i]["top_k"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.int32,
                    )
                    top_p_vals = torch.tensor(
                        [self.sampler_params[i]["top_p"] for i in indices],
                        device=self._handler.device,
                        dtype=torch.float32,
                    )
                    sampled = ops.sampling.top_k_top_p_sampling_from_probs(
                        group_probs, top_k=top_k_vals, top_p=top_p_vals
                    )
                else:
                    raise ValueError(f"Unknown _sampler index: {sampler_idx}")

                # Place sampled tokens into the main tensor at their original batch positions
                final_tokens_tensor.scatter_(0, indices_tensor, sampled)

        # Distribute batched results back to individual responses
        responses = []
        cursor = 0
        for req in self._original_reqs:
            num_outputs = len(req.output_token_indices)
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
