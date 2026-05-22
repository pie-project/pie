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
from .common import TOKEN_SAMPLING_TYPES, sample_common

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

        # vllm's CUDA-graph dispatcher; None if cudagraph capture is disabled.
        # When set, transform() asks the dispatcher for the runtime mode +
        # padded shape and forwards those into vllm's set_forward_context.
        # vllm's CUDAGraphWrapper then either replays the captured graph or
        # transparently falls through to eager. Construction lives in
        # VllmEngine.load.
        self.cg_dispatcher = cg_dispatcher

        # Persistent input buffers for cudagraph replay. Captured graphs bake
        # in tensor *addresses*, not values — pie must copy_ each fire's data
        # into the same buffers and pass slices off them.
        self._buf_input_ids: torch.Tensor | None = None
        self._buf_positions: torch.Tensor | None = None
        self._buf_max_n: int = 0

    def setup_cg_buffers(self, max_n: int) -> None:
        """Allocate persistent input buffers for capture/replay."""
        if self._buf_input_ids is not None and self._buf_max_n >= max_n:
            return
        self._buf_max_n = max_n
        self._buf_input_ids = torch.zeros(
            max_n, dtype=torch.long, device=self.device,
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
        token_ids: torch.Tensor,
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

        token_ids = token_ids.to(self.device, dtype=torch.long, non_blocking=True)

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
            assert self._buf_input_ids is not None, (
                "cg_dispatcher returned non-NONE mode but persistent buffers "
                "are not initialized — engine.load must call setup_cg_buffers."
            )
            assert self._buf_positions is not None
            self._buf_input_ids[:actual_n].copy_(token_ids, non_blocking=True)
            if forward_n > actual_n:
                self._buf_input_ids[actual_n:forward_n].zero_()
            self._buf_positions[:actual_n].copy_(positions, non_blocking=True)
            if forward_n > actual_n:
                self._buf_positions[actual_n:forward_n].zero_()
            input_ids_in = self._buf_input_ids[:forward_n]
            input_embeds_in = None
            positions_in = self._buf_positions[:forward_n]

            if forward_n > actual_n:
                slot_mapping_dict = self._pad_slot_mapping(
                    slot_mapping_dict, forward_n
                )
        else:
            input_ids_in = token_ids
            input_embeds_in = None
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
                input_ids=input_ids_in,
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
        greedy = self._try_greedy_sample(hidden_states, sampling_metadata)
        if greedy is not None:
            return greedy
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.activation_dtype,
        )

    def _try_greedy_sample(
        self, hidden_states: torch.Tensor, sampling_metadata: dict
    ) -> dict | None:
        """Fast path for temperature-0 token sampling.

        vLLM's LogitsProcessor can reduce vocab-parallel argmax by gathering
        only (max value, token id) pairs across TP ranks. The generic Pie
        sampler must materialize full logits/probs, which is much slower for
        TP decode and unnecessary for deterministic sampling.
        """
        indices_for_logits = sampling_metadata.get("indices_for_logits")
        if not indices_for_logits:
            return {
                "tokens": [],
                "dists": [],
                "logits": [],
                "logprobs": [],
                "entropies": [],
            }

        sampler_groups = sampling_metadata["sampler_groups"]
        if any(k not in TOKEN_SAMPLING_TYPES for k in sampler_groups):
            return None
        if sampling_metadata.get("sampling_masks") is not None:
            return None

        temperatures = sampling_metadata["temperatures"]
        if bool(torch.any(temperatures > 1e-5).item()):
            return None

        rows = torch.nan_to_num(hidden_states[indices_for_logits])
        tokens = self._vocab_parallel_top_tokens(rows)
        token_list = tokens.tolist()
        n = len(indices_for_logits)
        return {
            "tokens": token_list,
            "dists": [None] * n,
            "logits": [None] * n,
            "logprobs": [None] * n,
            "entropies": [None] * n,
            "nan_indices": [],
        }

    def _vocab_parallel_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return greedy token ids without materializing full TP logits.

        vLLM exposes `LogitsProcessor.get_top_tokens` for this, but that helper
        uses vLLM's custom collective wrapper for the final tiny all-gather. Pie
        already disables NCCL P2P on SYS-level GPU pairs, so use PyTorch's
        process-group collective directly here and keep the payload to
        `(max_value, token_id)` per sampled row.
        """
        import torch.distributed as dist
        from vllm.distributed import get_tp_group

        lm_head = self.model.lm_head
        logits_processor = self.model.logits_processor

        logits = lm_head.quant_method.apply(lm_head, hidden_states)
        if logits_processor.soft_cap is not None:
            logits = (
                torch.tanh(logits / logits_processor.soft_cap)
                * logits_processor.soft_cap
            )
        if logits_processor.scale != 1.0:
            logits = logits * logits_processor.scale

        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        tp_group = get_tp_group()
        tp_size = tp_group.world_size
        if tp_size == 1:
            tokens = global_indices
        else:
            local_pair = torch.stack(
                [local_max_vals.float(), global_indices.float()], dim=-1
            ).contiguous()
            gathered_flat = torch.empty(
                (tp_size * local_pair.shape[0], local_pair.shape[1]),
                dtype=local_pair.dtype,
                device=local_pair.device,
            )
            dist.all_gather_into_tensor(
                gathered_flat, local_pair, group=tp_group.device_group
            )
            gathered = gathered_flat.view(tp_size, hidden_states.shape[0], 2)
            gathered = gathered.transpose(0, 1)
            max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
            tokens = tokens.squeeze(-1).to(torch.int64)

        return tokens
