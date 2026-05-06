"""Adapter that exposes a vllm model under pie_driver's ForwardPass contract.

`pie_driver.engine.Engine` calls three methods on its `forward_pass` object:
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

from pie_driver.config import RuntimeConfig
from pie_driver.model.common import sample_common

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
    ):
        self.model = model
        self.vllm_config = vllm_config
        self.runtime_config = runtime_config
        self.model_config = model_config
        self.device = torch.device(runtime_config.device)

        # Lazy: built on first transform() call so we know the resolved
        # backend (set during model construction inside set_current_vllm_config).
        self._builder = None
        self._kv_spec = None
        self._layer_names: list[str] = []

        # Cudagraph dispatch state (Lever 5, ticket #100). Set by
        # `VllmEngine._capture_cudagraphs()` after capture completes;
        # `transform()` consults it to thread cudagraph_runtime_mode +
        # batch_descriptor through set_forward_context. None means the
        # dispatcher hasn't been initialized — transform falls through to
        # the eager path.
        self._cg_dispatcher = None
        # Persistent inputs_embeds + positions buffers for cudagraph replay.
        # Captured graphs alias data_ptrs at capture time, so every replay
        # must read from the same persistent buffers; transform() pads
        # decode batches up to a captured size and copies inputs in-place.
        # Sized to max_cudagraph_capture_size at capture time.
        self._cg_inputs_embeds_buf: torch.Tensor | None = None
        self._cg_positions_buf: torch.Tensor | None = None
        # `slot_mapping` indexes the KV cache write target per query token; the
        # KV-append op reads the tensor directly from forward_context, so it
        # MUST live at the same data_ptr across capture and replay. (Backend
        # attention metadata — paged_kv_indptr, kv_indices, block_table — is
        # routed through the FlashInfer wrapper's plan(), which copies into
        # FlashInfer's own persistent cudagraph buffers, so those are
        # transparent to us. slot_mapping has no such wrapper.)
        self._cg_slot_mapping_buf: torch.Tensor | None = None
        self._cg_query_len = 1  # 1 + num_speculative_tokens; set at capture.

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
        single_token_mode: bool = False,
    ) -> torch.Tensor:
        """Run the model's transformer trunk inside `set_forward_context`.

        For Lever 5 cudagraph dispatch (ticket #100): if the dispatcher has
        been initialized AND the batch is uniform-decode, we route through
        a captured FULL graph by padding inputs into persistent buffers and
        threading `cudagraph_runtime_mode` + `batch_descriptor` into
        set_forward_context. Mixed and prefill batches always run eager.
        """
        from ._vllm_compat import (
            CUDAGraphMode,
            BatchDescriptor,
            set_forward_context,
        )

        self._ensure_metadata_builder()

        page_size = self._kv_spec.block_size
        num_tokens = int(input_embeds.shape[0])

        # Decide cudagraph dispatch. We dispatch only on uniform-decode
        # batches at query_len=1, gated on the existing pie convention
        # `single_token_mode` (set by the Rust runtime in shmem_schema.py
        # and consumed by phi3/mistral3/olmo3 model files for the same
        # purpose). This is a pure host-side bool; no GPU sync. Spec-
        # decode uniform batches (query_len > 1) currently fall through
        # to eager — separate follow-up to extend pie's runtime flag for
        # spec-aware uniform detection.
        cudagraph_runtime_mode = CUDAGraphMode.NONE
        batch_descriptor: BatchDescriptor | None = None
        padded_tokens = num_tokens
        if (
            self._cg_dispatcher is not None
            and self._cg_dispatcher.cudagraph_mode != CUDAGraphMode.NONE
            and single_token_mode
            and self._cg_query_len == 1
        ):
            cudagraph_runtime_mode, batch_descriptor = (
                self._cg_dispatcher.dispatch(
                    num_tokens=num_tokens,
                    uniform_decode=True,
                )
            )
            if cudagraph_runtime_mode != CUDAGraphMode.NONE:
                padded_tokens = batch_descriptor.num_tokens

        # Build attention metadata. Pad to `padded_tokens` when replaying so
        # the captured graph's persistent FlashInfer buffers are populated
        # to the captured shape. Each pad slot is one dummy decode request
        # (qlen=1) pointing at page 0 (pie's warmup scratch); the per-layer
        # slot_mapping is then patched to -1 for those rows so the KV write
        # kernel skips them and no real cache state is touched.
        if padded_tokens != num_tokens:
            pad_reqs = padded_tokens - num_tokens  # one query token per pad
            qpad = torch.arange(
                1, pad_reqs + 1, dtype=qo_indptr.dtype, device=qo_indptr.device
            ) + qo_indptr[-1]
            qo_indptr_eff = torch.cat([qo_indptr, qpad])

            zeros_pages = torch.zeros(
                pad_reqs,
                dtype=kv_page_indices.dtype,
                device=kv_page_indices.device,
            )
            kv_page_indices_eff = torch.cat([kv_page_indices, zeros_pages])

            ppad = torch.arange(
                1,
                pad_reqs + 1,
                dtype=kv_page_indptr.dtype,
                device=kv_page_indptr.device,
            ) + kv_page_indptr[-1]
            kv_page_indptr_eff = torch.cat([kv_page_indptr, ppad])

            ones_pad = torch.ones(
                pad_reqs,
                dtype=kv_last_page_lens.dtype,
                device=kv_last_page_lens.device,
            )
            kv_last_page_lens_eff = torch.cat([kv_last_page_lens, ones_pad])
        else:
            qo_indptr_eff = qo_indptr
            kv_page_indices_eff = kv_page_indices
            kv_page_indptr_eff = kv_page_indptr
            kv_last_page_lens_eff = kv_last_page_lens

        common = build_common_metadata(
            qo_indptr=qo_indptr_eff,
            kv_page_indices=kv_page_indices_eff,
            kv_page_indptr=kv_page_indptr_eff,
            kv_last_page_lens=kv_last_page_lens_eff,
            page_size=page_size,
            device=self.device,
        )

        if padded_tokens != num_tokens:
            # Mark padded query tokens as no-write so KV cache stays clean.
            common.slot_mapping[num_tokens:padded_tokens] = -1

        # When replaying a captured graph, the KV-append op (and any other op
        # that reads slot_mapping out of forward_context) was captured against
        # the data_ptr() of `common.slot_mapping` at capture time. Per-call
        # tensors land at fresh addresses, so the captured graph reads stale
        # / freed memory and silently corrupts KV writes — symptom is
        # progressively-degenerate decode output (e.g. token "Instructions"
        # repeated). Substitute the persistent buffer in `common` BEFORE
        # `builder.build` so any backend that snapshots `common.slot_mapping`
        # into its own AttentionMetadata struct also picks up the persistent
        # reference.
        if cudagraph_runtime_mode != CUDAGraphMode.NONE:
            assert self._cg_slot_mapping_buf is not None
            sm_buf = self._cg_slot_mapping_buf
            sm_buf[:padded_tokens].copy_(
                common.slot_mapping[:padded_tokens], non_blocking=True
            )
            common.slot_mapping = sm_buf[:padded_tokens]

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

        # Route through persistent buffers when replaying a captured graph
        # so input data_ptrs match capture-time addresses. For eager (NONE)
        # we use the per-call tensors as before.
        if cudagraph_runtime_mode != CUDAGraphMode.NONE:
            assert self._cg_inputs_embeds_buf is not None, (
                "cudagraph buffers not allocated; _capture_cudagraphs() must "
                "run before any FULL replay path"
            )
            embed_buf = self._cg_inputs_embeds_buf
            pos_buf = self._cg_positions_buf
            embed_buf[:num_tokens].copy_(input_embeds, non_blocking=True)
            pos_buf[:num_tokens].copy_(positions, non_blocking=True)
            # Pad slots beyond num_tokens are unused by attention (slot_mapping
            # contains -1 for those rows in build_common_metadata via padded
            # qo_indptr) but the position/embed values must be valid for
            # other ops that index into them — zero is safe.
            if padded_tokens > num_tokens:
                embed_buf[num_tokens:padded_tokens].zero_()
                pos_buf[num_tokens:padded_tokens].zero_()
            model_inputs_embeds = embed_buf[:padded_tokens]
            model_positions = pos_buf[:padded_tokens]
        else:
            model_inputs_embeds = input_embeds
            model_positions = positions

        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=padded_tokens,
            slot_mapping=slot_mapping_dict,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            # `self.model(...)` (not `.forward(...)`) so a CUDAGraphWrapper
            # installed at FULL mode by `_capture_cudagraphs` sees the
            # forward_context and dispatches capture/replay.
            hidden_states = self.model(
                input_ids=None,
                positions=model_positions,
                inputs_embeds=model_inputs_embeds,
            )

        if padded_tokens != num_tokens:
            # Trim padding rows from the captured graph's output before
            # downstream sampling/logits.
            hidden_states = hidden_states[:num_tokens]

        return hidden_states

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        """Sample via pie_driver's sample_common; vllm's LM head is the lm_head_fn."""
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.runtime_config.activation_dtype,
        )
