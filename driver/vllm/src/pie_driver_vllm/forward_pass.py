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
from .hybrid_blocks import select_kernel_block_size

from .attn_metadata import CommonMetadataWorkspace, build_common_metadata


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
        self._gpu_profile_enabled = _env_flag("PIE_VLLM_GPU_PROFILE")
        # Match standalone vLLM's text-only path: keep embedding inside the
        # compiled/cudagraphed model forward. The package __init__ isolates
        # the default AOT cache root for this call surface.
        self.use_input_ids_forward = _env_flag("PIE_VLLM_INPUT_IDS_FORWARD", True)
        self._last_transform_profile: dict[str, float] = {}
        self._last_sample_profile: dict[str, float] = {}
        self._greedy_top_token_ops: tuple[Any, Any] | bool | None = None

        # Lazy: built on first transform() call so we know the resolved
        # backends/specs (set during model construction inside
        # set_current_vllm_config). Hybrid attention+Mamba models need one
        # metadata object per layer name; pure attention models also accept
        # the dict form, so we use it uniformly.
        self._builder = None
        self._kv_spec = None
        self._page_size: int | None = None
        # vLLM's BlockPool reserves physical block 0 as NULL/padding. Pie's
        # scheduler hands out zero-based real page ids, so map scheduler page
        # N to vLLM physical block N+1 in all vLLM metadata.
        self._kv_block_id_offset = 1
        self._layer_names: list[str] = []
        self._metadata_groups: list[dict[str, Any]] | None = None
        self._full_metadata_workspaces: dict[int, CommonMetadataWorkspace] = {}

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

    def _get_full_metadata_workspace(
        self,
        *,
        kernel_page_size: int,
        max_blocks_per_req: int,
    ) -> CommonMetadataWorkspace:
        max_reqs = int(self.vllm_config.scheduler_config.max_num_seqs)
        max_tokens = int(
            self.vllm_config.compilation_config.max_cudagraph_capture_size
            or max_reqs
        )
        key = int(kernel_page_size)
        workspace = self._full_metadata_workspaces.get(key)
        if workspace is None or (
            workspace.max_reqs < max_reqs
            or workspace.max_tokens < max_tokens
            or workspace.max_blocks_per_req < max_blocks_per_req
        ):
            workspace = CommonMetadataWorkspace(
                max_reqs=max_reqs,
                max_tokens=max_tokens,
                max_blocks_per_req=max_blocks_per_req,
                device=self.device,
            )
            self._full_metadata_workspaces[key] = workspace
        return workspace

    def _build_common_metadata(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr_cpu=None,
        kv_page_indices_cpu=None,
        kv_page_indptr_cpu=None,
        kv_last_page_lens_cpu=None,
        page_size: int,
        kernel_page_size: int,
        position_ids: torch.Tensor | None = None,
        full_cg: bool = False,
        batch_desc: Any | None = None,
        for_cudagraph_capture: bool = False,
    ):
        pad_num_reqs = None
        pad_num_tokens = None
        workspace = None
        max_seq_len_override = None
        if full_cg:
            if batch_desc is None:
                raise RuntimeError("FULL cudagraph metadata requires batch_desc")
            pad_num_tokens = int(batch_desc.num_tokens)
            pad_num_reqs = int(batch_desc.num_reqs or pad_num_tokens)
            blocks_per_page = int(page_size) // int(kernel_page_size)
            max_model_len = int(self.vllm_config.model_config.max_model_len)
            max_manager_pages = (max_model_len + int(page_size) - 1) // int(page_size)
            max_blocks_per_req = max(1, max_manager_pages * blocks_per_page)
            workspace = self._get_full_metadata_workspace(
                kernel_page_size=int(kernel_page_size),
                max_blocks_per_req=max_blocks_per_req,
            )
            if for_cudagraph_capture:
                max_seq_len_override = max_model_len
        return build_common_metadata(
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_page_indices_cpu=kv_page_indices_cpu,
            kv_page_indptr_cpu=kv_page_indptr_cpu,
            kv_last_page_lens_cpu=kv_last_page_lens_cpu,
            page_size=page_size,
            kernel_page_size=kernel_page_size,
            device=self.device,
            position_ids=position_ids,
            pad_num_reqs=pad_num_reqs,
            pad_num_tokens=pad_num_tokens,
            workspace=workspace,
            block_id_offset=self._kv_block_id_offset,
            max_seq_len_override=max_seq_len_override,
            force_decode=bool(full_cg and for_cudagraph_capture),
        )

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

    def _call_model(self, *, input_ids, positions, input_embeds):
        if self.use_input_ids_forward:
            return self.model(input_ids=input_ids, positions=positions)
        return self.model(
            input_ids=None,
            positions=positions,
            inputs_embeds=input_embeds,
        )

    def _call_model_with_stream_sync(
        self,
        *,
        input_ids,
        positions,
        input_embeds,
        use_vllm_stream: bool,
    ):
        """Run vLLM CUDA-graph calls on vLLM's stream with explicit ordering.

        vLLM captures/replays graphs on `vllm.utils.current_stream()`, which is
        a dedicated non-default CUDA stream. Pie prepares input/metadata tensors
        on PyTorch's current stream and samples on that stream afterwards, so
        graph replay needs bidirectional waits to avoid reading stale inputs or
        sampling stale outputs.
        """
        if not use_vllm_stream or self.device.type != "cuda":
            return self._call_model(
                input_ids=input_ids,
                positions=positions,
                input_embeds=input_embeds,
            )

        from vllm.utils.torch_utils import current_stream as vllm_current_stream

        producer_stream = torch.cuda.current_stream(self.device)
        graph_stream = vllm_current_stream()
        graph_stream.wait_stream(producer_stream)
        with torch.cuda.stream(graph_stream):
            hidden_states = self._call_model(
                input_ids=input_ids,
                positions=positions,
                input_embeds=input_embeds,
            )
        producer_stream.wait_stream(graph_stream)
        return hidden_states

    def _ensure_metadata_builder(self) -> None:
        """Construct the per-backend AttentionMetadataBuilder once.

        We call this lazily (not in __init__) because vllm's `current_vllm_config`
        context must be active and the model fully constructed before
        `get_kv_cache_spec` is sound.
        """
        if self._metadata_groups is not None:
            return

        from ._vllm_compat import (
            AttentionLayerBase,
            extract_layer_index,
            set_current_vllm_config,
        )
        from vllm.v1.kv_cache_interface import AttentionSpec

        fc = self.vllm_config.compilation_config.static_forward_context
        attn_layers = [
            (name, layer) for name, layer in fc.items()
            if isinstance(layer, AttentionLayerBase)
        ]
        # Sort by extracted index so layer ordering is stable & matches
        # pie's swap RPC indexing.
        attn_layers.sort(key=lambda x: extract_layer_index(x[0]))

        groups: list[dict[str, Any]] = []
        with set_current_vllm_config(self.vllm_config):
            for name, layer in attn_layers:
                spec = layer.get_kv_cache_spec(self.vllm_config)
                if spec is None:
                    continue
                backend = getattr(layer, "attn_backend", None)
                if backend is None:
                    backend = layer.get_attn_backend()
                builder_cls = backend.get_builder_cls()
                kernel_block_size = select_kernel_block_size(
                    spec,
                    [backend],
                    self.vllm_config,
                )
                builder_spec = (
                    spec.copy_with_new_block_size(kernel_block_size)
                    if isinstance(spec, AttentionSpec)
                    else spec
                )
                # Pie's scheduler currently exposes one KV-page space keyed by
                # cache_config.block_size. Mamba "none" mode consumes a single
                # running-state block from that table, while attention expands
                # the same manager block into smaller kernel blocks when needed.
                metadata_kernel_block_size = (
                    kernel_block_size
                    if isinstance(spec, AttentionSpec)
                    else int(self.vllm_config.cache_config.block_size)
                )

                group = None
                for candidate in groups:
                    if (
                        candidate["backend"] is backend
                        and candidate["builder_cls"] is builder_cls
                        and candidate["builder_spec"] == builder_spec
                        and candidate["metadata_kernel_block_size"]
                        == metadata_kernel_block_size
                    ):
                        group = candidate
                        break
                if group is None:
                    group = {
                        "backend": backend,
                        "builder_cls": builder_cls,
                        "spec": spec,
                        "builder_spec": builder_spec,
                        "kernel_block_size": kernel_block_size,
                        "metadata_kernel_block_size": metadata_kernel_block_size,
                        "layer_names": [],
                        "builder": None,
                    }
                    groups.append(group)
                group["layer_names"].append(name)

            for group in groups:
                group["builder"] = group["builder_cls"](
                    kv_cache_spec=group["builder_spec"],
                    layer_names=group["layer_names"],
                    vllm_config=self.vllm_config,
                    device=self.device,
                )

        if not groups:
            raise RuntimeError("No vLLM metadata builders could be initialized.")

        self._metadata_groups = groups
        self._layer_names = [
            name for group in groups for name in group["layer_names"]
        ]
        self._kv_spec = groups[0]["builder_spec"]
        self._builder = groups[0]["builder"]
        self._page_size = int(
            getattr(self.vllm_config.cache_config, "block_size", None)
            or self._kv_spec.block_size
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

        assert self._page_size is not None
        page_size = self._page_size

        # vllm's RoPE kernel expects int64 positions; pie uses int32.
        positions = position_ids.to(self.device, dtype=torch.int64, non_blocking=True)
        t_positions = time.perf_counter()

        common_cache = {}
        dispatch_common = self._build_common_metadata(
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_page_indices_cpu=kv_page_indices_cpu,
            kv_page_indptr_cpu=kv_page_indptr_cpu,
            kv_last_page_lens_cpu=kv_last_page_lens_cpu,
            page_size=page_size,
            kernel_page_size=page_size,
            position_ids=positions,
        )
        common_cache[page_size] = dispatch_common
        t_common = time.perf_counter()
        common_positions = getattr(dispatch_common, "positions", None)

        # Ask vllm whether a captured graph applies to this num_tokens. If
        # mode is NONE the wrapper falls through to eager and we run the
        # model unmodified. Otherwise we route through the persistent
        # buffers — captured graphs replay against fixed input *addresses*,
        # so pie must copy_ each fire's data into the same buffers it
        # passed at capture time and slice off them.
        actual_n = dispatch_common.num_actual_tokens
        actual_reqs = dispatch_common.num_reqs
        uniform_decode = False
        if self.cg_dispatcher is not None:
            uniform_decode_query_len = getattr(
                self.cg_dispatcher, "uniform_decode_query_len", 1
            )
            uniform_decode = (
                dispatch_common.max_query_len == uniform_decode_query_len
                and actual_n == dispatch_common.num_reqs * uniform_decode_query_len
            )
            invalid_modes = set()
            if _env_flag("PIE_VLLM_DISABLE_FULL_CG"):
                invalid_modes.add(CUDAGraphMode.FULL)
            cg_mode, batch_desc = self.cg_dispatcher.dispatch(
                num_tokens=actual_n,
                uniform_decode=uniform_decode,
                has_lora=False,
                invalid_modes=invalid_modes,
            )
        else:
            cg_mode, batch_desc = CUDAGraphMode.NONE, BatchDescriptor(actual_n)
        t_dispatch = time.perf_counter()
        full_cg = cg_mode == CUDAGraphMode.FULL
        if full_cg:
            common_cache = {}

        # Backend-specific metadata. `common_prefix_len=0` disables cascade
        # attention (we don't use it from pie's side). Hybrid attention+Mamba
        # models require a dict keyed by layer name because attention and Mamba
        # layers consume different metadata classes.
        assert self._metadata_groups is not None
        backend_metadata = {}
        slot_mapping_dict = {}
        for group in self._metadata_groups:
            metadata_kernel_block_size = int(group["metadata_kernel_block_size"])
            group_common = common_cache.get(metadata_kernel_block_size)
            if group_common is None:
                group_common = self._build_common_metadata(
                    qo_indptr=qo_indptr,
                    kv_page_indices=kv_page_indices,
                    kv_page_indptr=kv_page_indptr,
                    kv_last_page_lens=kv_last_page_lens,
                    qo_indptr_cpu=qo_indptr_cpu,
                    kv_page_indices_cpu=kv_page_indices_cpu,
                    kv_page_indptr_cpu=kv_page_indptr_cpu,
                    kv_last_page_lens_cpu=kv_last_page_lens_cpu,
                    page_size=page_size,
                    kernel_page_size=metadata_kernel_block_size,
                    position_ids=positions,
                    full_cg=full_cg,
                    batch_desc=batch_desc,
                )
                common_cache[metadata_kernel_block_size] = group_common
            metadata = group["builder"].build(
                common_prefix_len=0,
                common_attn_metadata=group_common,
            )
            for name in group["layer_names"]:
                backend_metadata[name] = metadata
                slot_mapping_dict[name] = group_common.slot_mapping
        t_backend = time.perf_counter()

        common = common_cache.get(page_size, dispatch_common)
        common_positions = getattr(common, "positions", None)

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

        gpu_model_events = None
        if self._gpu_profile_enabled and self.device.type == "cuda":
            gpu_model_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            gpu_model_events[0].record(torch.cuda.current_stream(self.device))
        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=forward_n,
            slot_mapping=slot_mapping_dict,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=batch_desc,
        ):
            hidden_states = self._call_model_with_stream_sync(
                input_ids=input_ids_in,
                positions=positions_in,
                input_embeds=input_embeds_in,
                use_vllm_stream=cg_mode != CUDAGraphMode.NONE,
            )
        if gpu_model_events is not None:
            gpu_model_events[1].record(torch.cuda.current_stream(self.device))
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
            if gpu_model_events is not None:
                gpu_model_events[1].synchronize()
                self._last_transform_profile["gpu_model_forward"] = (
                    gpu_model_events[0].elapsed_time(gpu_model_events[1]) / 1000.0
                )
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
                    **sampling_metadata.get("_sample_profile", {}),
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
        token_tensor = self.greedy_top_token_tensor(hidden_states, sampling_metadata)
        if token_tensor is None:
            return None
        tokens = host_tokens(token_tensor)
        n = len(sampling_metadata["indices_for_logits"])
        return {
            "tokens": tokens,
            "dists": [None] * n,
            "logits": [None] * n,
            "logprobs": [None] * n,
            "entropies": [None] * n,
            "nan_indices": [],
        }

    def greedy_top_token_tensor(
        self, hidden_states: torch.Tensor, sampling_metadata: dict
    ) -> torch.Tensor | None:
        """Return a GPU tensor of greedy top tokens when the batch is eligible."""
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

        ops = self._resolve_greedy_top_token_ops()
        if ops is None:
            return None
        logits_processor, lm_head = ops
        get_top_tokens = logits_processor.get_top_tokens

        if sampling_metadata.get("all_logits_in_order", False):
            logits_input = hidden_states
        else:
            logits_input = hidden_states[indices_for_logits]
        try:
            gpu_events = None
            if self._gpu_profile_enabled and logits_input.is_cuda:
                gpu_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                gpu_events[0].record()
            token_tensor = get_top_tokens(lm_head, logits_input)
            if gpu_events is not None:
                gpu_events[1].record()
                gpu_events[1].synchronize()
                sampling_metadata["_sample_profile"] = {
                    **sampling_metadata.get("_sample_profile", {}),
                    "gpu_greedy_top_tokens": (
                        gpu_events[0].elapsed_time(gpu_events[1]) / 1000.0
                    ),
                }
        except Exception as exc:
            # Some trust-remote-code wrappers expose HF-style lm_head modules
            # that are not compatible with vLLM's vocab-parallel argmax.
            self._greedy_top_token_ops = False
            if os.environ.get("PIE_VLLM_DEBUG_GREEDY_TOP"):
                print(f"[pie-vllm] disabling greedy top-token path: {exc}", flush=True)
            return None
        return token_tensor

    def _resolve_greedy_top_token_ops(self) -> tuple[Any, Any] | None:
        cached = self._greedy_top_token_ops
        if cached is False:
            return None
        if cached is not None:
            return cached

        modules: list[Any] = []
        seen: set[int] = set()

        def add_module(obj: Any | None) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            modules.append(obj)

        add_module(self.model)
        for root in list(modules):
            for attr in (
                "language_model",
                "model",
                "base_model",
                "llm",
                "module",
            ):
                add_module(getattr(root, attr, None))

        for module in modules:
            logits_processor = getattr(module, "logits_processor", None)
            lm_head = getattr(module, "lm_head", None)
            get_top_tokens = getattr(logits_processor, "get_top_tokens", None)
            if get_top_tokens is not None and lm_head is not None:
                self._greedy_top_token_ops = (logits_processor, lm_head)
                if os.environ.get("PIE_VLLM_DEBUG_GREEDY_TOP"):
                    print(
                        "[pie-vllm] using greedy top-token path from "
                        f"{module.__class__.__name__}",
                        flush=True,
                    )
                return self._greedy_top_token_ops

        self._greedy_top_token_ops = False
        return None
