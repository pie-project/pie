"""Adapter that exposes an SGLang ModelRunner under pie_driver's ForwardPass contract.

`pie_driver.engine.Engine` calls `embed_inputs / transform / sample` on
this object. We delegate compute to SGLang's ModelRunner.

Pie samples at multiple per-request positions (best-of-n, distribution
mode, multi-step). SGLang's default `LogitsProcessor` only gathers the
last extend-token's logits, so we replace it with `_HiddenCapture`, which
stashes per-token hidden states into a stable buffer. `sample()` then
runs pie's per-output gather + LM head + sampler via `sample_common`.
"""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from pie_driver.config import RuntimeConfig
from pie_driver.model.common import sample_common

from .forward_batch import build_sglang_forward_batch
from .mask_hooks import make_mask_strategy


class _HiddenCapture(torch.nn.Module):
    """Drop-in replacement for `runner.model.logits_processor`.

    Inherits from `nn.Module` so sglang's `model.logits_processor = ...`
    assignment passes its child-module typecheck. The forward signature
    matches sglang's LogitsProcessor: `(input_ids, hidden_states, lm_head,
    forward_batch, aux)`.

    Cuda-graph safe: copies into a stable pre-allocated `captured_buffer`
    via `copy_(hidden_states)`. Source and dest pointers are both fixed
    across replays, so the captured memcpy is correct on every replay.
    (Plain attribute assignment isn't replay-safe — the assignment itself
    isn't recorded by graph capture.)

    `next_token_logits` is a `(max_tokens, 1)` stand-in because sglang's
    `CudaGraphRunner.replay()` post-slices it. Pie ignores the values.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.captured_buffer = torch.empty(
            (max_tokens, hidden_size), dtype=dtype, device=device,
        )
        self._fake_logits = torch.empty((max_tokens, 1), dtype=dtype, device=device)

    def forward(self, input_ids, hidden_states, lm_head, forward_batch, aux=None):
        n = hidden_states.shape[0]
        self.captured_buffer[:n].copy_(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=self._fake_logits[:n],
            hidden_states=self.captured_buffer[:n],
        )


class SGLangForwardPass:
    """Thin shim around an SGLang ModelRunner.

    Contract:
      - `embed_inputs(inputs)`: passthrough — sglang owns input embedding.
      - `transform(...)`: build a `ForwardBatch`, hand pie's mask to the
        strategy, run `runner.forward()`, return per-token hidden states.
      - `sample(hidden, sampling_metadata)`: gather pie's requested
        indices, apply the LM head, run pie's sampler.
    """

    def __init__(
        self,
        *,
        runner: Any,
        runtime_config: RuntimeConfig,
        page_size: int,
    ):
        self.runner = runner
        self.runtime_config = runtime_config
        self.page_size = page_size
        self.device = torch.device(runtime_config.device)

        # One-shot install: replace the LogitsProcessor and patch the
        # attention backend. Buffer sized for the worst case across both
        # eager EXTEND (chunked_prefill_size tokens) and captured DECODE
        # (max_running_requests) so one stable pointer serves both paths.
        max_tokens = max(
            int(getattr(runner.server_args, "chunked_prefill_size", 0) or 0),
            int(runner.max_running_requests),
            8192,
        )
        self._capture = _HiddenCapture(
            max_tokens=max_tokens,
            hidden_size=int(runner.model_config.hidden_size),
            dtype=runtime_config.activation_dtype,
            device=self.device,
        )
        runner.model.logits_processor = self._capture
        self._mask_strategy = make_mask_strategy(runner.attn_backend)

        # ParallelLMHead with TP > 1 needs an all-gather; for v1 we only
        # support TP=1 and the matmul-against-weight path matches sglang's
        # `_compute_lm_head` (layers/logits_processor.py:891-913).
        self._lm_head_module = runner.model.lm_head

        # Set by `Engine.load` when adapter mode is enabled; stays None
        # otherwise so `transform()` skips the slot writes.
        self.adapter_subpass_slot = None

    # ------------------------------------------------------------------
    # Pie contract
    # ------------------------------------------------------------------

    def embed_inputs(self, inputs: dict) -> dict:
        return inputs  # sglang's forward() owns input embedding.

    def transform(
        self,
        *,
        input_embeds: dict,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        single_token_inference_mode: bool,
        custom_mask: torch.Tensor | None = None,
        adapter_subpass=None,
    ) -> torch.Tensor:
        fb, meta = build_sglang_forward_batch(
            runner=self.runner,
            inputs={
                "token_ids": input_embeds["token_ids"],
                "position_ids": position_ids,
                "qo_indptr": qo_indptr,
                "kv_page_indices": kv_page_indices,
                "kv_page_indptr": kv_page_indptr,
                "kv_last_page_lens": kv_last_page_lens,
                "single_token_inference_mode": single_token_inference_mode,
            },
            page_size=self.page_size,
            device=self.device,
        )

        # DECODE applies an implicit causal mask in-kernel, so pie's
        # synthesized causal in `flattened_masks` is redundant there. For
        # EXTEND we hand the mask to the per-backend strategy (FlashInfer
        # via `apply_to_forward_batch`; Triton/Flex/TorchNative through
        # their hooked `init_forward_metadata`).
        if custom_mask is not None and fb.forward_mode != ForwardMode.DECODE:
            self._mask_strategy.set(custom_mask, meta.mask_indptr)
            self._mask_strategy.apply_to_forward_batch(fb)

        # CMA-ES adapter: hand the per-batch subpass to the QKV wrappers
        # via the slot they hold a reference to. No-op when adapter mode
        # is disabled (slot is None) or when this batch has no adapter
        # tokens (subpass is None).
        slot = self.adapter_subpass_slot
        if slot is not None:
            slot.current = adapter_subpass

        try:
            self.runner.forward(fb)
        finally:
            self._mask_strategy.clear()
            if slot is not None:
                slot.current = None

        n_tokens = int(fb.input_ids.shape[0])
        return self._capture.captured_buffer[:n_tokens]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _lm_head_fn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the LM head. Mirrors sglang's `_compute_lm_head` (TP=1)."""
        lm_head = self._lm_head_module
        if hasattr(lm_head, "weight"):
            return torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)
        return lm_head(hidden_states)  # LoRA-wrapped / GGUF fallback.

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        if not sampling_metadata or sampling_metadata.get("indices_for_logits") is None:
            return {"tokens": [], "dists": [], "spec_tokens": [], "spec_positions": []}
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self._lm_head_fn,
            device=self.device,
            dtype=self.runtime_config.activation_dtype,
        )
