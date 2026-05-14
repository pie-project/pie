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

from typing import Any

import torch

from ._bridge.config import RuntimeConfig
from .common import sample_common

from .attn_metadata import build_common_metadata


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
        self._buf_positions: torch.Tensor | None = None
        self._buf_max_n: int = 0

    def setup_cg_buffers(self, max_n: int, hidden_size: int) -> None:
        """Allocate the persistent input_embeds / positions buffers used as
        the input slice during both capture and replay. Idempotent — safe
        to call multiple times with the same max_n."""
        if self._buf_input_embeds is not None and self._buf_max_n >= max_n:
            return
        self._buf_max_n = max_n
        self._buf_input_embeds = torch.zeros(
            max_n, hidden_size,
            dtype=self.activation_dtype,
            device=self.device,
        )
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
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model's transformer trunk inside `set_forward_context`."""
        from ._vllm_compat import CUDAGraphMode, BatchDescriptor, set_forward_context

        self._ensure_metadata_builder()

        page_size = self._kv_spec.block_size

        common = build_common_metadata(
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            page_size=page_size,
            device=self.device,
        )

        # Backend-specific metadata. `common_prefix_len=0` disables cascade
        # attention (we don't use it from pie's side).
        backend_metadata = self._builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
        )

        # Same slot_mapping for every layer (no cross-attention or shared KV).
        slot_mapping_dict = {
            name: common.slot_mapping for name in self._layer_names
        }

        # vllm's RoPE kernel expects int64 positions; pie uses int32.
        positions = position_ids.to(self.device, dtype=torch.int64, non_blocking=True)

        # Ask vllm whether a captured graph applies to this num_tokens. If
        # mode is NONE the wrapper falls through to eager and we run the
        # model unmodified. Otherwise we route through the persistent
        # buffers — captured graphs replay against fixed input *addresses*,
        # so pie must copy_ each fire's data into the same buffers it
        # passed at capture time and slice off them.
        actual_n = common.num_actual_tokens
        if self.cg_dispatcher is not None:
            cg_mode, batch_desc = self.cg_dispatcher.dispatch(
                num_tokens=actual_n,
                uniform_decode=False,
                has_lora=False,
            )
        else:
            cg_mode, batch_desc = CUDAGraphMode.NONE, BatchDescriptor(actual_n)

        forward_n = batch_desc.num_tokens
        if cg_mode != CUDAGraphMode.NONE:
            assert self._buf_input_embeds is not None, (
                "cg_dispatcher returned non-NONE mode but persistent buffers "
                "are not initialized — engine.load must call setup_cg_buffers."
            )
            self._buf_input_embeds[:actual_n].copy_(input_embeds, non_blocking=True)
            if forward_n > actual_n:
                self._buf_input_embeds[actual_n:forward_n].zero_()
            self._buf_positions[:actual_n].copy_(positions, non_blocking=True)
            if forward_n > actual_n:
                self._buf_positions[actual_n:forward_n].zero_()
            input_embeds_in = self._buf_input_embeds[:forward_n]
            positions_in = self._buf_positions[:forward_n]

            if forward_n > actual_n:
                slot_mapping_dict = self._pad_slot_mapping(
                    slot_mapping_dict, forward_n
                )
        else:
            input_embeds_in = input_embeds
            positions_in = positions

        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=forward_n,
            slot_mapping=slot_mapping_dict,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=batch_desc,
        ):
            hidden_states = self.model.forward(
                input_ids=None,
                positions=positions_in,
                inputs_embeds=input_embeds_in,
            )

        if forward_n > actual_n:
            hidden_states = hidden_states[:actual_n]
        return hidden_states

    @staticmethod
    def _pad_to(t: torch.Tensor, target_len: int) -> torch.Tensor:
        """Right-pad a 1-D or 2-D tensor's leading dim with zeros up to
        `target_len`. Caller has already checked target_len > t.shape[0]."""
        pad_rows = target_len - t.shape[0]
        if t.ndim == 1:
            pad = torch.zeros(pad_rows, dtype=t.dtype, device=t.device)
        else:
            pad = torch.zeros(
                (pad_rows, *t.shape[1:]), dtype=t.dtype, device=t.device,
            )
        return torch.cat([t, pad], dim=0)

    def _pad_slot_mapping(
        self, slot_mapping_dict: dict, target_len: int,
    ) -> dict:
        """Pad each layer's slot_mapping tensor up to target_len with -1
        (vllm's PAD_SLOT_ID). Attention-write kernels skip slot_id == -1
        so the trailing padded tokens compute-then-discard their KV."""
        out = {}
        for name, sm in slot_mapping_dict.items():
            if sm.shape[0] >= target_len:
                out[name] = sm
                continue
            pad_rows = target_len - sm.shape[0]
            pad = torch.full(
                (pad_rows,), -1, dtype=sm.dtype, device=sm.device,
            )
            out[name] = torch.cat([sm, pad], dim=0)
        return out

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        """Sample via pie_driver's sample_common; vllm's LM head is the lm_head_fn."""
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.activation_dtype,
        )
