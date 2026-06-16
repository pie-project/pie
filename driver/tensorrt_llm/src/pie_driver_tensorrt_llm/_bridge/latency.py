"""Latency tracking for fire_batch stages."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
from typing import NamedTuple

from . import telemetry


class StepTiming(NamedTuple):
    """Timing data for a single fire_batch step."""

    # Top-level stages
    build_batch: float
    get_inputs: float
    get_sampling_meta: float
    broadcast: float
    inference: float
    create_responses: float
    total: float
    # build_batch breakdown
    decode_u32: float
    mask_loop: float
    brle_decode: float
    sampler_loop: float


@dataclass
class LatencyStats:
    """Tracks latency statistics for fire_batch stages.

    When profiling is enabled, emits spans to OpenTelemetry.
    When disabled, this is a no-op.
    """

    enabled: bool = False
    step_count: int = 0
    summary_enabled: bool = False
    total_build_batch: float = 0.0
    total_get_inputs: float = 0.0
    total_get_sampling_meta: float = 0.0
    total_broadcast: float = 0.0
    total_inference: float = 0.0
    total_create_responses: float = 0.0
    total_total: float = 0.0
    total_decode_u32: float = 0.0
    total_mask_loop: float = 0.0
    total_brle_decode: float = 0.0
    total_sampler_loop: float = 0.0
    summary_every: int = 0

    def __post_init__(self):
        self.summary_enabled = os.environ.get("PIE_TRTLLM_LATENCY_SUMMARY") == "1"
        self.summary_every = int(os.environ.get("PIE_TRTLLM_LATENCY_EVERY", "0") or 0)
        if self.summary_enabled:
            atexit.register(self.print_summary)

    def record_span(self, timing: StepTiming, traceparent: str | None = None):
        """Record timing as an OpenTelemetry span.

        Args:
            timing: Timing data for the step.
            traceparent: Optional W3C traceparent string for cross-language propagation.
        """
        self.step_count += 1
        self.total_build_batch += timing.build_batch
        self.total_get_inputs += timing.get_inputs
        self.total_get_sampling_meta += timing.get_sampling_meta
        self.total_broadcast += timing.broadcast
        self.total_inference += timing.inference
        self.total_create_responses += timing.create_responses
        self.total_total += timing.total
        self.total_decode_u32 += timing.decode_u32
        self.total_mask_loop += timing.mask_loop
        self.total_brle_decode += timing.brle_decode
        self.total_sampler_loop += timing.sampler_loop
        if self.summary_enabled and self.summary_every > 0:
            if self.step_count % self.summary_every == 0:
                self.print_summary()

        if not self.enabled:
            return

        # Create a span with all timing attributes (in milliseconds)
        # Use traceparent if provided (from Rust) to link traces across languages
        with telemetry.start_span_with_traceparent(
            "py.fire_batch",
            traceparent,
            step=self.step_count,
            total_ms=timing.total * 1000,
            build_batch_ms=timing.build_batch * 1000,
            decode_u32_ms=timing.decode_u32 * 1000,
            mask_loop_ms=timing.mask_loop * 1000,
            brle_decode_ms=timing.brle_decode * 1000,
            sampler_loop_ms=timing.sampler_loop * 1000,
            get_inputs_ms=timing.get_inputs * 1000,
            get_sampling_meta_ms=timing.get_sampling_meta * 1000,
            broadcast_ms=timing.broadcast * 1000,
            inference_ms=timing.inference * 1000,
            create_responses_ms=timing.create_responses * 1000,
        ) as span:
            pass  # span is closed automatically

    def print_summary(self):
        if not self.summary_enabled or self.step_count <= 0:
            return
        steps = float(self.step_count)

        def ms(value: float) -> float:
            return value * 1000.0

        print(
            "[pie-trtllm-latency] "
            f"steps={self.step_count} "
            f"total_ms={ms(self.total_total):.3f} "
            f"inference_ms={ms(self.total_inference):.3f} "
            f"create_responses_ms={ms(self.total_create_responses):.3f} "
            f"build_batch_ms={ms(self.total_build_batch):.3f} "
            f"decode_u32_ms={ms(self.total_decode_u32):.3f} "
            f"mask_loop_ms={ms(self.total_mask_loop):.3f} "
            f"brle_decode_ms={ms(self.total_brle_decode):.3f} "
            f"sampler_loop_ms={ms(self.total_sampler_loop):.3f} "
            f"avg_total_ms={ms(self.total_total / steps):.3f} "
            f"avg_inference_ms={ms(self.total_inference / steps):.3f}",
            flush=True,
        )
