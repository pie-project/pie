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

import os
from typing import Any

import torch

from pie_driver.config import RuntimeConfig
from pie_driver.model.common import (
    NEEDS_PROBS_TYPES,
    LOGPROB_TYPES,
    RAW_LOGITS_TYPE,
    sample_common,
    scaled_softmax,
)
from pie_kernels.sampling import top_p_sampling_from_probs

from .attn_metadata import build_common_metadata


# Sampler type id for TopP (matches Rust `Sampler::type_id()`); defined as
# a module constant so the sample fast-path eligibility check is a host-side
# integer compare with no GPU sync.
_SAMPLER_IDX_TOP_P = 3


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
        mamba_layers: list[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
        is_hybrid: bool = False,
        mamba_allocator: Any = None,
    ):
        self.model = model
        self.vllm_config = vllm_config
        self.runtime_config = runtime_config
        self.model_config = model_config
        self.device = torch.device(runtime_config.device)

        # Hybrid (Transformer + Mamba) support. State slots are now
        # ContextId-keyed (#108 phase 5a) via `mamba_allocator`; see
        # `mamba_state.py` for the rationale.
        self._is_hybrid = is_hybrid
        # `mamba_layer_names`: layer prefixes for which `transform()` must
        # publish a `GDNAttentionMetadata` entry into the per-layer
        # forward-context dict. Order matches `engine.mamba_layers`.
        self._mamba_layer_names: list[str] = (
            [name for name, _, _ in (mamba_layers or [])]
        )
        # Per-request slot allocator. Required on hybrid; ignored on
        # pure-attention (transform skips the GDN metadata branch).
        self._mamba_allocator = mamba_allocator

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

        # Captured sample fast-path: one cuda graph per discrete capture
        # size wraps compute_logits + scaled_softmax + top_p_sampling.
        # Replay from sample() when the batch is uniform TopP with no
        # logprobs/dists. Disabled by default; PIE_SAMPLE_GRAPH=1 to enable.
        self._cg_sample_enabled = False
        self._cg_sample_graphs: dict[int, "torch.cuda.CUDAGraph"] = {}
        self._cg_sample_hidden_buf: torch.Tensor | None = None
        self._cg_sample_top_p_buf: torch.Tensor | None = None
        self._cg_sample_temperatures_buf: torch.Tensor | None = None
        self._cg_sample_tokens_buf: torch.Tensor | None = None
        # Whether the captured sample graph fired on the most recent batch.
        # Surfaced via gpu_timings for bench-side probing.
        self._sample_fastpath_used_last = False

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
            get_mamba_base,
            set_current_vllm_config,
        )

        # vllm 0.20.0 has `MambaBase(AttentionLayerBase)`, so a plain
        # AttentionLayerBase isinstance check captures GDN/Mamba layers too.
        # We need only the genuine attention layers here (FlashInfer
        # metadata is built from their kv_spec).
        MambaBase = get_mamba_base()
        fc = self.vllm_config.compilation_config.static_forward_context
        attn_layers = [
            (name, layer) for name, layer in fc.items()
            if isinstance(layer, AttentionLayerBase)
            and not isinstance(layer, MambaBase)
        ]
        # Sort by extracted index so layer ordering is stable & matches
        # pie's swap RPC indexing.
        attn_layers.sort(key=lambda x: extract_layer_index(x[0]))

        self._layer_names = [name for name, _ in attn_layers]
        if not attn_layers:
            # Pure-mamba models — not in pie's smoke-test target. The
            # FlashInfer builder isn't constructed, but `transform()` still
            # needs to publish GDN metadata, which only consumes
            # `CommonAttentionMetadata` (we build it directly below). Set
            # `_kv_spec` to None and let the GDN-only path handle it.
            self._kv_spec = None
            self._builder = None
            return
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
        context_ids: list[int] | None = None,
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

        # On hybrid models, `_kv_spec` is the attention spec; the mamba
        # layers don't share its block_size. Use the attention page_size
        # for FlashInfer-side metadata; GDN metadata is built independently.
        page_size = self._kv_spec.block_size if self._kv_spec is not None else 0
        num_tokens = int(input_embeds.shape[0])

        # Decide cudagraph dispatch. We dispatch only on uniform-decode
        # batches at query_len=1, gated on the existing pie convention
        # `single_token_mode` (set by the Rust runtime in shmem_schema.py
        # and consumed by phi3/mistral3/olmo3 model files for the same
        # purpose). This is a pure host-side bool; no GPU sync.
        #
        # Hybrid models: cudagraph capture is disabled at engine init for
        # ticket #107 (the GDN backend has its own UNIFORM_BATCH cudagraph
        # path that conflicts with pie's FULL_DECODE_ONLY capture
        # machinery). The dispatcher is None on hybrid, so this gate
        # naturally falls through to eager.
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
            page_size=page_size if page_size > 0 else 1,
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

        # Backend-specific attention metadata. `common_prefix_len=0`
        # disables cascade attention. Returns one dataclass shared by all
        # attention layers (FlashInfer reads `common.block_table_tensor`
        # etc. directly out of forward_context, so we only need one).
        if self._builder is not None:
            backend_metadata = self._builder.build(
                common_prefix_len=0,
                common_attn_metadata=common,
            )
        else:
            backend_metadata = None

        # Per-layer attn_metadata dict. vllm's hybrid path (qwen3-next /
        # qwen3.5-moe) reads `forward_context.attn_metadata[layer.prefix]`
        # — Attention layers expect their own FlashInfer struct, GDN
        # layers expect a `GDNAttentionMetadata`. We always publish a
        # dict (uniform shape on hybrid and pure-attention) so model code
        # doesn't branch on the metadata type.
        per_layer_metadata: dict[str, Any] = {}
        if backend_metadata is not None:
            for name in self._layer_names:
                per_layer_metadata[name] = backend_metadata

        if self._is_hybrid and self._mamba_layer_names:
            from .gdn_metadata import build_gdn_metadata

            if self._mamba_allocator is None:
                raise RuntimeError(
                    "Hybrid model loaded without a mamba state-slot "
                    "allocator. Engine.load() must construct a "
                    "MambaSlotAllocator and pass it to VllmForwardPass "
                    "(#108 phase 5a)."
                )

            # Per-row state slots from the allocator. ContextId comes
            # from the runtime via `BatchedForwardPassRequest.context_ids`
            # and is plumbed through `Engine.fire_batch -> transform`.
            # Pure synthetic warmup calls (e.g. `_warmup_compile`) skip
            # the GDN metadata branch implicitly: warmup happens on
            # pure-attention models or via a separate code path.
            if context_ids is None:
                raise RuntimeError(
                    "Hybrid transform requires context_ids; got None. "
                    "Caller must thread `context_ids` from the batch "
                    "(see Batch.get_model_inputs in pie_driver/batching.py)."
                )

            slot_ids = self._mamba_allocator.slots_for(context_ids)

            # Reuse `common`'s computed seq_lens / num_computed_tokens to
            # avoid re-walking pie's CSR. `common.compute_num_computed_tokens()`
            # caches; second access is free.
            num_computed_tokens = common.compute_num_computed_tokens()
            gdn = build_gdn_metadata(
                qo_indptr=qo_indptr_eff,
                seq_lens=common.seq_lens,
                num_computed_tokens=num_computed_tokens,
                num_actual_tokens=padded_tokens,
                device=self.device,
                state_slot_ids=slot_ids,
            )
            for name in self._mamba_layer_names:
                per_layer_metadata[name] = gdn

        # Same slot_mapping for every attention layer (no cross-attention
        # or shared KV). Mamba layers don't read slot_mapping; they get
        # their state slots via `non_spec_state_indices_tensor` in the
        # GDN metadata.
        slot_mapping_dict = (
            {name: common.slot_mapping for name in self._layer_names}
            if self._layer_names else {}
        )

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
            attn_metadata=per_layer_metadata,
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
        """Sample via pie_driver's sample_common; vllm's LM head is the lm_head_fn.

        Fast path: when captured sample graphs are available and the
        batch shape passes the eligibility gate (all TopP samplers, no
        logprobs/dists/raw-logits/sampling_masks, batch size in
        `_cg_sample_graphs`), replay the captured graph instead of
        running compute_logits + softmax + flashinfer eagerly.
        Eligibility is host-only, so the slow-path fall-through is free.
        """
        self._sample_fastpath_used_last = False
        if self._cg_sample_enabled and self._sample_fastpath_eligible(
            hidden_states, sampling_metadata
        ):
            tokens = self._sample_fastpath_replay(hidden_states, sampling_metadata)
            if tokens is not None:
                self._sample_fastpath_used_last = True
                num = len(sampling_metadata["indices_for_logits"])
                return {
                    "tokens": tokens,
                    "dists": [None] * num,
                    "logits": [None] * num,
                    "logprobs": [None] * num,
                    "entropies": [None] * num,
                    "nan_indices": [],
                }

        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.runtime_config.activation_dtype,
        )

    # ------------------------------------------------------------------
    # Sample-stage cudagraph capture + replay
    # ------------------------------------------------------------------

    def _sample_fastpath_eligible(
        self, hidden_states: torch.Tensor, sampling_metadata: dict
    ) -> bool:
        """Host-side gate: is this batch shaped right for the captured graph?

        All checks must be CPU-only (no `.item()` / `.tolist()` on device
        tensors) so the eligibility decision adds no GPU sync to the slow
        path.

        Conditions:
          - sample graphs were captured (`_cg_sample_graphs` non-empty);
          - exactly one sampler type, and that type is TopP (idx=3);
          - logits requested for every row of `hidden_states` (no skipping);
          - no `sampling_masks`, no logprobs/dists/raw-logits requests;
          - batch size matches one of the captured graph sizes.
        """
        if not self._cg_sample_graphs:
            return False
        if sampling_metadata.get("sampling_masks") is not None:
            return False

        sampler_groups = sampling_metadata.get("sampler_groups", {})
        # One sampler type total. `dict.keys()` over a small dict is host-side.
        non_empty = [k for k, v in sampler_groups.items() if v]
        if len(non_empty) != 1 or non_empty[0] != _SAMPLER_IDX_TOP_P:
            return False
        # No raw-logits / logprobs / entropy / dist requests anywhere.
        if RAW_LOGITS_TYPE in sampler_groups and sampler_groups[RAW_LOGITS_TYPE]:
            return False
        if any(t in sampler_groups and sampler_groups[t] for t in LOGPROB_TYPES):
            return False
        if 0 in sampler_groups and sampler_groups[0]:
            return False  # Distribution sampler

        indices = sampling_metadata.get("indices_for_logits", [])
        if not indices:
            return False
        n = hidden_states.shape[0]
        # We capture graphs sized to `padded_tokens` from transform(). When
        # the captured trunk path was used, num_logit_requests usually equals
        # the padded transform size. For the eager-trunk + captured-sample
        # combo (PIE_CUDA_GRAPHS=0 PIE_SAMPLE_GRAPH=1), n is the un-padded
        # request count — also fine if it happens to match a capture size.
        if len(indices) != n:
            return False
        if n not in self._cg_sample_graphs:
            return False

        # Pre-built sampler param tensors must exist (sample_common builds
        # them once per fire_batch in pie_driver.batching).
        if sampling_metadata.get("top_p") is None:
            return False
        if sampling_metadata.get("temperatures") is None:
            return False
        return True

    def _sample_fastpath_replay(
        self, hidden_states: torch.Tensor, sampling_metadata: dict
    ) -> list[int] | None:
        """Replay the captured sample graph for `n = hidden_states.shape[0]`.

        Copies inputs into the persistent buffers the graph was captured
        against, replays, then returns the sampled token IDs as a list of
        ints. The trailing `.tolist()` forces a CPU sync — same shape as
        the slow path's final sync, so total per-batch sync count is
        unchanged.
        """
        n = hidden_states.shape[0]
        graph = self._cg_sample_graphs.get(n)
        if graph is None:
            return None
        hbuf = self._cg_sample_hidden_buf
        pbuf = self._cg_sample_top_p_buf
        tbuf = self._cg_sample_temperatures_buf
        out = self._cg_sample_tokens_buf
        if hbuf is None or pbuf is None or tbuf is None or out is None:
            return None

        top_p = sampling_metadata["top_p"]
        temps = sampling_metadata["temperatures"]
        # Pre-built tensors live on device already (pie's batching.py builds
        # them via torch.as_tensor(..., device=device)). If the upstream ever
        # changes, the .to() below is a no-op when already on-device.
        if top_p.device != self.device:
            top_p = top_p.to(self.device, non_blocking=True)
        if temps.device != self.device:
            temps = temps.to(self.device, non_blocking=True)

        # `indices_for_logits` selects the rows of hidden_states the slow
        # path would have gathered before compute_logits. Captured graphs
        # always operate on the FIRST `n` rows of the persistent buffer, so
        # we must apply that gather *before* the copy. For the common case
        # where every row is a logit request (decode-only batch) the gather
        # is a no-op — `indices_for_logits == range(n)`. We assert that
        # under DEBUG, but skip the check on the hot path.
        indices = sampling_metadata["indices_for_logits"]
        if list(indices) == list(range(n)):
            hidden_in = hidden_states
        else:
            idx_t = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            hidden_in = hidden_states.index_select(0, idx_t)

        # Per-row clamp: scaled_softmax inside the captured graph also clamps
        # at `greedy_threshold=1e-5`, so we only need to forward the raw
        # temperatures here. dtype must match the captured graph (float32).
        hbuf[:n].copy_(hidden_in, non_blocking=True)
        pbuf[:n].copy_(top_p, non_blocking=True)
        tbuf[:n].copy_(temps.unsqueeze(1) if temps.ndim == 1 else temps,
                       non_blocking=True)

        graph.replay()

        return out[:n].tolist()
