"""Adapter that exposes a vllm model under pie_driver's ForwardPass contract.

`pie_driver_dev.engine.Engine` calls three methods on its `forward_pass` object:
`embed_inputs(inputs) -> hidden`, `transform(...) -> hidden`, and
`sample(hidden, sampling_metadata) -> list`. We satisfy that contract here
while delegating the actual compute to vllm.

Custom attention masks: this bridge runs vLLM's standard causal attention
and silently drops any user-supplied mask. The driver advertises that
behavior via `DriverCapabilities.supports_user_attention_mask=False`;
inferlets that need non-causal patterns must run on the `native` driver.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from ._bridge.config import RuntimeConfig
from .common import TOKEN_SAMPLING_TYPES, host_tokens, sample_common

from .attn_metadata import build_common_metadata


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


class VllmForwardPass:
    """Thin shim around a vllm model.

    `embed_inputs` runs the model's input-embedding layer.

    `transform` builds vllm-style attention metadata from pie's batch dicts,
    enters `set_forward_context`, and runs the vllm model's forward.

    `sample` reuses pie_driver's `sample_common`, calling the vllm model's
    `compute_logits` as the LM head.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        vllm_config: Any,
        attn_backend: Any,
        runtime_config: RuntimeConfig,
        model_config: Any,
        cg_dispatcher: Any | None = None,
    ):
        self.model = model
        self.vllm_config = vllm_config
        self.runtime_config = runtime_config
        self.model_config = model_config
        self.device = torch.device(runtime_config.device)
        # `runtime_config.activation_dtype` is a string (bridge stays
        # torch-free); resolve once here so the sampling hot path doesn't
        # do a getattr per call.
        self.activation_dtype = getattr(torch, runtime_config.activation_dtype)
        self._profile_enabled = _env_flag("PIE_VLLM_PROFILE")
        # Match standalone vLLM's text-only path: keep embedding inside the
        # compiled/cudagraphed model forward. The package __init__ isolates
        # the default AOT cache root for this call surface.
        self.use_input_ids_forward = _env_flag("PIE_VLLM_INPUT_IDS_FORWARD", True)
        self._last_transform_profile: dict[str, float] = {}
        self._last_sample_profile: dict[str, float] = {}

        # Lazy: built on first transform() call so we know the resolved
        # backend (set during model construction inside set_current_vllm_config).
        self._builder = None
        self._kv_spec = None
        self._layer_names: list[str] = []

        # vllm's CUDA-graph dispatcher; None if cudagraph capture is disabled
        # (enforce_eager=True / --no-cuda-graphs). When set, transform() asks
        # the dispatcher for the runtime mode + padded shape and forwards
        # those into vllm's set_forward_context — vllm's CUDAGraphWrapper
        # then either replays the captured graph or transparently falls
        # through to eager. Construction lives in VllmEngine.load.
        self.cg_dispatcher = cg_dispatcher

        # Persistent input buffers for cudagraph replay. Captured graphs
        # bake in tensor *addresses*, not values — pie must copy_ each
        # fire's data into the same buffers and pass slices off them.
        # Sized to the largest captured shape; `None` until set up by the
        # engine's capture warmup (which knows max_cudagraph_capture_size).
        self._buf_input_embeds: torch.Tensor | None = None
        self._buf_input_ids: torch.Tensor | None = None
        self._buf_positions: torch.Tensor | None = None
        self._buf_max_n: int = 0

    def setup_cg_buffers(self, max_n: int, hidden_size: int) -> None:
        """Allocate the persistent input_embeds / positions buffers used as
        the input slice during both capture and replay. Idempotent — safe
        to call multiple times with the same max_n."""
        if (
            self._buf_positions is not None
            and self._buf_max_n >= max_n
            and (self.use_input_ids_forward or self._buf_input_embeds is not None)
        ):
            return
        self._buf_max_n = max_n
        if not self.use_input_ids_forward:
            self._buf_input_embeds = torch.zeros(
                max_n, hidden_size,
                dtype=self.activation_dtype,
                device=self.device,
            )
        self._buf_input_ids = torch.zeros(max_n, dtype=torch.long, device=self.device)
        self._buf_positions = torch.zeros(
            max_n, dtype=torch.int64, device=self.device,
        )

    def _ensure_metadata_builder(self) -> None:
        """Construct the per-backend AttentionMetadataBuilder once.

        We call this lazily (not in __init__) because vllm's `current_vllm_config`
        context must be active and the model fully constructed before
        `get_kv_cache_spec` is sound.
        """
        if self._builder is not None:
            return

        from ._vllm_compat import (
            AttentionLayerBase,
            extract_layer_index,
            set_current_vllm_config,
        )

        fc = self.vllm_config.compilation_config.static_forward_context
        attn_layers = [
            (name, layer) for name, layer in fc.items()
            if isinstance(layer, AttentionLayerBase)
        ]
        # Sort by extracted index so layer ordering is stable & matches
        # pie's swap RPC indexing.
        attn_layers.sort(key=lambda x: extract_layer_index(x[0]))

        self._layer_names = [name for name, _ in attn_layers]
        first_layer = attn_layers[0][1]
        backend = first_layer.attn_backend

        with set_current_vllm_config(self.vllm_config):
            self._kv_spec = first_layer.get_kv_cache_spec(self.vllm_config)
            builder_cls = backend.get_builder_cls()
            self._builder = builder_cls(
                kv_cache_spec=self._kv_spec,
                layer_names=self._layer_names,
                vllm_config=self.vllm_config,
                device=self.device,
            )

    def embed_inputs(self, inputs: dict) -> torch.Tensor:
        """Run the model's input-embedding layer."""
        token_ids = inputs["token_ids"].to(self.device, non_blocking=True)
        return self.model.embed_input_ids(token_ids)

    def transform(
        self,
        *,
        input_embeds: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr_cpu=None,
        kv_page_indices_cpu=None,
        kv_page_indptr_cpu=None,
        kv_last_page_lens_cpu=None,
    ) -> torch.Tensor:
        """Run the model's transformer trunk inside `set_forward_context`."""
        from ._vllm_compat import CUDAGraphMode, BatchDescriptor, set_forward_context

        t0 = time.perf_counter()
        self._ensure_metadata_builder()
        t_builder = time.perf_counter()

        page_size = self._kv_spec.block_size

        # vllm's RoPE kernel expects int64 positions; pie uses int32.
        positions = position_ids.to(self.device, dtype=torch.int64, non_blocking=True)
        t_positions = time.perf_counter()

        common = build_common_metadata(
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_page_indices_cpu=kv_page_indices_cpu,
            kv_page_indptr_cpu=kv_page_indptr_cpu,
            kv_last_page_lens_cpu=kv_last_page_lens_cpu,
            page_size=page_size,
            device=self.device,
            position_ids=positions,
        )
        t_common = time.perf_counter()
        common_positions = getattr(common, "positions", None)

        # Ask vllm whether a captured graph applies to this num_tokens. If
        # mode is NONE the wrapper falls through to eager and we run the
        # model unmodified. Otherwise we route through the persistent
        # buffers — captured graphs replay against fixed input *addresses*,
        # so pie must copy_ each fire's data into the same buffers it
        # passed at capture time and slice off them.
        actual_n = common.num_actual_tokens
        actual_reqs = common.num_reqs
        uniform_decode = False
        if self.cg_dispatcher is not None:
            uniform_decode_query_len = getattr(
                self.cg_dispatcher, "uniform_decode_query_len", 1
            )
            uniform_decode = (
                common.max_query_len == uniform_decode_query_len
                and actual_n == common.num_reqs * uniform_decode_query_len
            )
            cg_mode, batch_desc = self.cg_dispatcher.dispatch(
                num_tokens=actual_n,
                uniform_decode=uniform_decode,
                has_lora=False,
                invalid_modes={CUDAGraphMode.FULL},
            )
        else:
            cg_mode, batch_desc = CUDAGraphMode.NONE, BatchDescriptor(actual_n)
        t_dispatch = time.perf_counter()

        # Backend-specific metadata. `common_prefix_len=0` disables cascade
        # attention (we don't use it from pie's side).
        backend_metadata = self._builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
        )
        t_backend = time.perf_counter()

        # Same slot_mapping for every layer (no cross-attention or shared KV).
        slot_mapping_dict = {
            name: common.slot_mapping for name in self._layer_names
        }

        graph_input_copy_bytes = 0
        graph_input_zero_bytes = 0
        graph_position_copy_bytes = 0
        graph_position_zero_bytes = 0
        graph_slot_mapping_pad_bytes = 0
        graph_slot_mapping_cat_bytes = 0

        forward_n = batch_desc.num_tokens
        if cg_mode != CUDAGraphMode.NONE:
            assert self._buf_positions is not None, (
                "cg_dispatcher returned non-NONE mode but persistent buffers "
                "are not initialized — engine.load must call setup_cg_buffers."
            )
            if self.use_input_ids_forward:
                assert input_ids is not None and self._buf_input_ids is not None
                graph_input_copy_bytes += self._leading_dim_bytes(
                    input_ids, actual_n
                )
                self._buf_input_ids[:actual_n].copy_(input_ids, non_blocking=True)
                if forward_n > actual_n:
                    graph_input_zero_bytes += self._leading_dim_bytes(
                        self._buf_input_ids, forward_n - actual_n
                    )
                    self._buf_input_ids[actual_n:forward_n].zero_()
                input_ids_in = self._buf_input_ids[:forward_n]
                input_embeds_in = None
            else:
                assert input_embeds is not None and self._buf_input_embeds is not None
                graph_input_copy_bytes += self._leading_dim_bytes(
                    input_embeds, actual_n
                )
                self._buf_input_embeds[:actual_n].copy_(
                    input_embeds, non_blocking=True
                )
                if forward_n > actual_n:
                    graph_input_zero_bytes += self._leading_dim_bytes(
                        self._buf_input_embeds, forward_n - actual_n
                    )
                    self._buf_input_embeds[actual_n:forward_n].zero_()
                input_ids_in = None
                input_embeds_in = self._buf_input_embeds[:forward_n]
            positions_for_forward = (
                common_positions if common_positions is not None else positions
            )
            if positions_for_forward.shape[0] >= forward_n:
                graph_position_copy_bytes += self._leading_dim_bytes(
                    positions_for_forward, forward_n
                )
                self._buf_positions[:forward_n].copy_(
                    positions_for_forward[:forward_n], non_blocking=True
                )
            else:
                graph_position_copy_bytes += self._leading_dim_bytes(
                    positions, actual_n
                )
                self._buf_positions[:actual_n].copy_(positions, non_blocking=True)
                if forward_n > actual_n:
                    graph_position_zero_bytes += self._leading_dim_bytes(
                        self._buf_positions, forward_n - actual_n
                    )
                    self._buf_positions[actual_n:forward_n].zero_()
            positions_in = self._buf_positions[:forward_n]

            if forward_n > common.slot_mapping.shape[0]:
                slot_pad_tokens = forward_n - common.slot_mapping.shape[0]
                graph_slot_mapping_pad_bytes = (
                    slot_pad_tokens
                    * common.slot_mapping.element_size()
                    * len(self._layer_names)
                )
                graph_slot_mapping_cat_bytes = (
                    forward_n
                    * common.slot_mapping.element_size()
                    * len(self._layer_names)
                )
                slot_mapping_dict = self._pad_slot_mapping(
                    slot_mapping_dict, forward_n
                )
        else:
            input_ids_in = input_ids if self.use_input_ids_forward else None
            input_embeds_in = input_embeds
            if self.use_input_ids_forward:
                input_embeds_in = None
            else:
                assert input_embeds_in is not None
            positions_in = common_positions if common_positions is not None else positions
        t_graph_inputs = time.perf_counter()

        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=forward_n,
            slot_mapping=slot_mapping_dict,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=batch_desc,
        ):
            if self.use_input_ids_forward:
                hidden_states = self.model.forward(
                    input_ids=input_ids_in,
                    positions=positions_in,
                )
            else:
                hidden_states = self.model.forward(
                    input_ids=None,
                    positions=positions_in,
                    inputs_embeds=input_embeds_in,
                )
        t_model = time.perf_counter()

        if forward_n > actual_n:
            hidden_states = hidden_states[:actual_n]
        if self._profile_enabled:
            forward_reqs = getattr(batch_desc, "num_reqs", None)
            if forward_reqs is None:
                forward_reqs = actual_reqs
            self._last_transform_profile = {
                "ensure_builder": t_builder - t0,
                "common_metadata": t_common - t_builder,
                "backend_metadata": t_backend - t_dispatch,
                "positions": t_positions - t_builder,
                "dispatch": t_dispatch - t_common,
                "graph_inputs": t_graph_inputs - t_backend,
                "model_forward": t_model - t_graph_inputs,
                "uniform_decode_ratio": float(uniform_decode),
                "cg_full_ratio": float(cg_mode == CUDAGraphMode.FULL),
                "cg_piecewise_ratio": float(cg_mode == CUDAGraphMode.PIECEWISE),
                "cg_none_ratio": float(cg_mode == CUDAGraphMode.NONE),
                "input_ids_forward_ratio": float(self.use_input_ids_forward),
                "actual_tokens": float(actual_n),
                "forward_tokens": float(forward_n),
                "pad_tokens": float(max(forward_n - actual_n, 0)),
                "actual_requests": float(actual_reqs),
                "forward_requests": float(forward_reqs),
                "graph_input_copy_bytes": float(graph_input_copy_bytes),
                "graph_input_zero_bytes": float(graph_input_zero_bytes),
                "graph_position_copy_bytes": float(graph_position_copy_bytes),
                "graph_position_zero_bytes": float(graph_position_zero_bytes),
                "graph_slot_mapping_pad_bytes": float(graph_slot_mapping_pad_bytes),
                "graph_slot_mapping_cat_bytes": float(graph_slot_mapping_cat_bytes),
                "graph_total_copy_bytes": float(
                    graph_input_copy_bytes
                    + graph_position_copy_bytes
                    + graph_slot_mapping_cat_bytes
                ),
            }
        return hidden_states

    @staticmethod
    def _leading_dim_bytes(tensor: torch.Tensor, rows: int) -> int:
        if rows <= 0 or tensor.ndim == 0:
            return 0
        leading = max(int(tensor.shape[0]), 1)
        row_elems = int(tensor.numel()) // leading
        return int(rows) * row_elems * tensor.element_size()

    def _pad_slot_mapping(
        self, slot_mapping_dict: dict, target_len: int,
    ) -> dict:
        """Pad each layer's slot_mapping tensor up to target_len with -1
        (vllm's PAD_SLOT_ID). Attention-write kernels skip slot_id == -1
        so the trailing padded tokens compute-then-discard their KV."""
        out = {}
        padded_cache = {}
        for name, sm in slot_mapping_dict.items():
            if sm.shape[0] >= target_len:
                out[name] = sm
                continue
            cache_key = (id(sm), target_len)
            padded = padded_cache.get(cache_key)
            if padded is None:
                pad_rows = target_len - sm.shape[0]
                pad = torch.full(
                    (pad_rows,), -1, dtype=sm.dtype, device=sm.device,
                )
                padded = torch.cat([sm, pad], dim=0)
                padded_cache[cache_key] = padded
            out[name] = padded
        return out

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        """Sample via pie_driver's sample_common; vllm's LM head is the lm_head_fn."""
        t0 = time.perf_counter()
        greedy_top_tokens = self._try_vllm_greedy_top_tokens(
            hidden_states, sampling_metadata
        )
        if greedy_top_tokens is not None:
            if self._profile_enabled:
                self._last_sample_profile = {
                    "total": time.perf_counter() - t0,
                    "vllm_greedy_top_tokens_ratio": 1.0,
                }
            return greedy_top_tokens
        out = sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.activation_dtype,
        )
        if self._profile_enabled:
            self._last_sample_profile = {
                "total": time.perf_counter() - t0,
                "vllm_greedy_top_tokens_ratio": 0.0,
                **sampling_metadata.get("_sample_profile", {}),
            }
        return out

    def _try_vllm_greedy_top_tokens(
        self, hidden_states: torch.Tensor, sampling_metadata: dict
    ) -> dict | None:
        """Use vLLM's logits-processor top-token path for pure greedy batches."""
        indices_for_logits = sampling_metadata.get("indices_for_logits")
        if not indices_for_logits:
            return None
        if not sampling_metadata.get("all_temperatures_greedy", False):
            return None
        sampler_groups = sampling_metadata.get("sampler_groups") or {}
        if not sampler_groups or not all(
            sampler_idx in TOKEN_SAMPLING_TYPES for sampler_idx in sampler_groups
        ):
            return None
        if sampling_metadata.get("sampling_masks") is not None:
            return None

        logits_processor = getattr(self.model, "logits_processor", None)
        lm_head = getattr(self.model, "lm_head", None)
        get_top_tokens = getattr(logits_processor, "get_top_tokens", None)
        if get_top_tokens is None or lm_head is None:
            return None

        if sampling_metadata.get("all_logits_in_order", False):
            logits_input = hidden_states
        else:
            logits_input = hidden_states[indices_for_logits]
        token_tensor = get_top_tokens(lm_head, logits_input)
        tokens = host_tokens(token_tensor)
        n = len(indices_for_logits)
        return {
            "tokens": tokens,
            "dists": [None] * n,
            "logits": [None] * n,
            "logprobs": [None] * n,
            "entropies": [None] * n,
            "nan_indices": [],
        }
