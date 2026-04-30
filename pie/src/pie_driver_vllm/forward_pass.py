"""Adapter that exposes a vllm model under pie_driver's ForwardPass contract.

`pie_driver.engine.Engine` calls three methods on its `forward_pass` object:
`embed_inputs(inputs) -> hidden`, `transform(...) -> hidden`, and
`sample(hidden, sampling_metadata) -> list`. We satisfy that contract here
while delegating the actual compute to vllm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pie_driver.config import RuntimeConfig
from pie_driver.model.common import sample_common

from .attn_metadata import build_common_metadata


@dataclass(frozen=True)
class AttentionShape:
    """Shape parameters for one attention layer. Frozen so the same value
    can be used as a dict key (we set-compare to detect uniformity).

    Plain LLMs have a single shape across all layers. Mixed-attention
    architectures (Gemma3 sliding+full, Qwen3-Next hybrid) carry different
    shapes per layer; we capture per-layer to make that visible at the
    type level even though we currently assert uniformity.
    """
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int


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
        # Per-layer attention shapes captured at install. Plain LLMs have
        # one shape across the dict; mixed-attention models carry per-
        # layer differences. Today we assert uniformity in transform()
        # because the FlashInfer plan is one-shape-per-batch; per-layer
        # plans are a future expansion when a real mixed model lands.
        self._layer_shapes: dict[str, AttentionShape] = {}
        # True when the installed mask-aware impl declares it consumes the
        # FlashInfer prefill wrapper. Set during _ensure_metadata_builder.
        self._impl_uses_flashinfer_wrapper: bool = False
        # Mask plumbing: lazily allocated FlashInfer wrapper, planned once
        # per batch in transform() when the FlashInfer fast path is active.
        self._mask_wrapper: Any | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

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

        # Per-layer attention shapes. `num_heads`/`num_kv_heads` are
        # post-TP; `head_size` is uniform across Q/K for non-MLA archs.
        self._layer_shapes = {
            name: AttentionShape(
                num_qo_heads=int(layer.num_heads),
                num_kv_heads=int(layer.num_kv_heads),
                head_dim=int(layer.head_size),
            )
            for name, layer in attn_layers
        }

        # The installed mask strategy flags whether it needs the FlashInfer
        # prefill wrapper pre-planned this batch. We read from the first
        # layer's strategy; mixed-impl models would need a per-layer flag,
        # out of scope today.
        from .mask_strategies import first_attention_strategy
        strategy = first_attention_strategy(self.vllm_config)
        self._impl_uses_flashinfer_wrapper = bool(
            getattr(strategy, "pie_uses_flashinfer_wrapper", False)
        )

        with set_current_vllm_config(self.vllm_config):
            self._kv_spec = first_layer.get_kv_cache_spec(self.vllm_config)
            builder_cls = backend.get_builder_cls()
            self._builder = builder_cls(
                kv_cache_spec=self._kv_spec,
                layer_names=self._layer_names,
                vllm_config=self.vllm_config,
                device=self.device,
            )

    def _uniform_shape_or_fail(self) -> AttentionShape:
        """Return the model's attention shape, asserting all layers match.

        The FlashInfer prefill wrapper is planned once per batch with a
        single (num_qo_heads, num_kv_heads, head_dim) tuple. If a model
        mixes shapes across layers (Gemma3 sliding+full, Qwen3-Next
        hybrid attention), per-layer plans are needed — fail loudly here
        so the bug surfaces as a refusal, not as silently-wrong logits.
        """
        shapes = set(self._layer_shapes.values())
        if len(shapes) == 1:
            return next(iter(shapes))
        raise NotImplementedError(
            "pie_driver_vllm: model has mixed attention shapes across "
            f"layers ({len(shapes)} distinct shapes). FlashInfer plan() is "
            "currently one-shape-per-batch; per-layer planning is not yet "
            f"implemented. Shapes seen: {sorted(shapes, key=str)}."
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

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
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
    ) -> torch.Tensor:
        """Run the model's transformer trunk inside `set_forward_context`."""
        from ._vllm_compat import set_forward_context

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

        # Mask plumbing: only when there's something to apply. The mask-
        # aware impl proxies (installed at engine load — see
        # `mask_strategies.install_mask_strategies`) read `pie_attn_extras`
        # from additional_kwargs. Absent key is the zero-overhead signal:
        # the proxy delegates straight to the wrapped impl's forward.
        pie_attn_extras = None
        if custom_mask is not None:
            from .mask_compute import (
                PieAttnExtras, make_flashinfer_wrapper, plan_flashinfer_wrapper,
            )

            pie_attn_extras = PieAttnExtras.build(
                custom_mask=custom_mask,
                query_start_loc=common.query_start_loc,
                seq_lens=common.seq_lens,
                block_table=common.block_table_tensor,
                page_size=page_size,
                device=self.device,
            )

            # Pre-plan FlashInfer wrapper ONCE for this batch IF the active
            # strategy uses it. Native pie follows the same shape: plan()
            # per batch, run() per layer. The plan is one-shape-per-batch,
            # so we assert per-layer shape uniformity here — when a real
            # mixed-attention model lands, this is the spot to grow into
            # per-layer plans.
            if self._impl_uses_flashinfer_wrapper:
                shape = self._uniform_shape_or_fail()
                if self._mask_wrapper is None:
                    self._mask_wrapper = make_flashinfer_wrapper(
                        128 * 1024 * 1024, self.device,
                    )
                plan_flashinfer_wrapper(
                    self._mask_wrapper,
                    extras=pie_attn_extras,
                    kv_page_indices=kv_page_indices,
                    kv_page_indptr=kv_page_indptr,
                    kv_last_page_lens=kv_last_page_lens,
                    num_qo_heads=shape.num_qo_heads,
                    num_kv_heads=shape.num_kv_heads,
                    head_dim_qk=shape.head_dim,
                    q_data_type=input_embeds.dtype,
                )
                pie_attn_extras.flashinfer_wrapper = self._mask_wrapper

        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=common.num_actual_tokens,
            slot_mapping=slot_mapping_dict,
        ):
            # `set_forward_context` takes its `additional_kwargs` from the
            # platform hook, not from a kwarg, so inject ours after entering.
            if pie_attn_extras is not None:
                from ._vllm_compat import get_forward_context

                get_forward_context().additional_kwargs["pie_attn_extras"] = pie_attn_extras

            hidden_states = self.model.forward(
                input_ids=None,
                positions=positions,
                inputs_embeds=input_embeds,
            )

        return hidden_states

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        """Sample via pie_driver's sample_common; vllm's LM head is the lm_head_fn."""
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.runtime_config.activation_dtype,
        )
