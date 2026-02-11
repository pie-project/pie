"""Latency tracking for fire_batch stages.

Extracted from backend.py to keep the Engine focused on inference.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def record_span(self, timing: StepTiming, traceparent: str | None = None):
        """Record timing as an OpenTelemetry span.

        Args:
            timing: Timing data for the step.
            traceparent: Optional W3C traceparent string for cross-language propagation.
        """
        self.step_count += 1

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
