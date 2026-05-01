"""Adapter that exposes an SGLang ModelRunner under pie_driver's ForwardPass contract.

`pie_driver.engine.Engine` calls three methods on its `forward_pass` object:
`embed_inputs(inputs) -> hidden`, `transform(...) -> hidden`, and
`sample(hidden, sampling_metadata) -> dict`. We delegate the actual compute
to SGLang's ModelRunner.

Pie supports multiple sampling positions per request (best-of-n, distribution
mode in prefill, multi-step parallel generation). SGLang's default
`LogitsProcessor` gathers logits to *one* position per request — the last
extend-token. To preserve pie's contract we replace the model's
LogitsProcessor with `_HiddenCapture`, which stashes the full per-token
hidden state tensor; `sample()` then applies pie's per-output
`indices_for_logits` gather + the LM head + sampling itself via
`pie_driver.model.common.sample_common`.

Custom attention masks: this bridge runs SGLang's standard causal attention
and silently drops any user-supplied mask. The driver advertises that
behavior via `DriverCapabilities.supports_user_attention_mask=False`;
inferlets that need non-causal patterns must run on the `native` driver.
"""

from __future__ import annotations

from typing import Any

import torch

from pie_driver.config import RuntimeConfig
from pie_driver.model.common import sample_common

from .forward_batch import build_sglang_forward_batch


class _HiddenCapture(torch.nn.Module):
    """Drop-in replacement for `runner.model.logits_processor`.

    SGLang models assign their LogitsProcessor as a `nn.Module` child, so
    we inherit from `torch.nn.Module` to satisfy `__setattr__`'s typecheck.
    Models call us as `self.logits_processor(input_ids, hidden_states,
    lm_head, forward_batch, aux=None)` — we stash the full per-token
    hidden_states for `transform()` to read and return a sentinel
    `LogitsProcessorOutput` (the caller doesn't read its fields).
    """

    def __init__(self):
        super().__init__()
        self.captured: torch.Tensor | None = None

    def forward(self, input_ids, hidden_states, lm_head, forward_batch, aux=None):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        self.captured = hidden_states
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)


class SGLangForwardPass:
    """Thin shim around an SGLang ModelRunner.

    Contract:
      - `embed_inputs(inputs)`: passthrough — sglang owns input embedding.
      - `transform(...)`: build a `ForwardBatch`, run `runner.forward()`,
        return per-token hidden states.
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

        # One-shot install: hidden-state capture replaces the LogitsProcessor.
        # User-supplied attention masks are silently dropped — the driver runs
        # causal-only attention via sglang's stock backend.
        self._capture = _HiddenCapture()
        runner.model.logits_processor = self._capture

        # ParallelLMHead with TP > 1 needs an all-gather; for v1 we only
        # support TP=1 and the matmul-against-weight path matches sglang's
        # `_compute_lm_head` (layers/logits_processor.py:891-913).
        self._lm_head_module = runner.model.lm_head

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
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
    ) -> torch.Tensor:
        fb = build_sglang_forward_batch(
            runner=self.runner,
            inputs={
                "token_ids": input_embeds["token_ids"],
                "position_ids": position_ids,
                "qo_indptr": qo_indptr,
                "kv_page_indices": kv_page_indices,
                "kv_page_indptr": kv_page_indptr,
                "kv_last_page_lens": kv_last_page_lens,
            },
            page_size=self.page_size,
            device=self.device,
        )

        self._capture.captured = None
        self.runner.forward(fb)

        if self._capture.captured is None:
            raise RuntimeError(
                "pie_driver_sgl: hidden states were not captured. The "
                "model didn't call its logits_processor (unexpected forward path)."
            )
        return self._capture.captured

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
