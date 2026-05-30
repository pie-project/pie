"""Latency tracking for fire_batch stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import os

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
    driver_profile: dict[str, float] | None = None
    response_profile: dict[str, float] | None = None


@dataclass
class LatencyStats:
    """Tracks latency statistics for fire_batch stages.

    When profiling is enabled, emits spans to OpenTelemetry.
    When disabled, this is a no-op.
    """

    enabled: bool = False
    step_count: int = 0
    profile_enabled: bool = bool(os.environ.get("PIE_VLLM_PROFILE"))
    profile_interval: int = int(os.environ.get("PIE_VLLM_PROFILE_INTERVAL", "256"))
    profile_each: bool = bool(os.environ.get("PIE_VLLM_PROFILE_EACH"))
    _sums: dict[str, float] | None = None

    def record_span(self, timing: StepTiming, traceparent: str | None = None):
        """Record timing as an OpenTelemetry span.

        Args:
            timing: Timing data for the step.
            traceparent: Optional W3C traceparent string for cross-language propagation.
        """
        self.step_count += 1
        if self.profile_enabled:
            if self._sums is None:
                self._sums = {}
            base = {
                "total": timing.total,
                "build_batch": timing.build_batch,
                "get_inputs": timing.get_inputs,
                "get_sampling_meta": timing.get_sampling_meta,
                "broadcast": timing.broadcast,
                "inference": timing.inference,
                "create_responses": timing.create_responses,
                "decode_u32": timing.decode_u32,
                "mask_loop": timing.mask_loop,
                "brle_decode": timing.brle_decode,
                "sampler_loop": timing.sampler_loop,
            }
            if timing.driver_profile:
                base.update({f"driver_{k}": v for k, v in timing.driver_profile.items()})
            if timing.response_profile:
                base.update({f"response_{k}": v for k, v in timing.response_profile.items()})
            if self.profile_each:
                self._print_profile_values(base, prefix=f"[pie-vllm-step] step={self.step_count}")
            for k, v in base.items():
                self._sums[k] = self._sums.get(k, 0.0) + float(v)
            if self.step_count % self.profile_interval == 0:
                self._print_profile()

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
        ):
            pass  # span recorded; closed automatically on block exit

    def _print_profile(self) -> None:
        if not self._sums or self.step_count == 0:
            return
        n = self.step_count
        values = {k: v / n for k, v in self._sums.items()}
        self._print_profile_values(values, prefix=f"[pie-vllm-profile] steps={n}")

    def _print_profile_values(self, values: dict[str, float], *, prefix: str) -> None:
        keys = sorted(values)
        parts = []
        for k in keys:
            avg = values[k]
            if k.endswith("_ratio"):
                parts.append(f"{k}={avg * 100.0:.1f}%")
            elif k.endswith("_bytes"):
                if avg >= 1024 * 1024:
                    parts.append(f"{k}={avg / (1024 * 1024):.3f}MiB")
                elif avg >= 1024:
                    parts.append(f"{k}={avg / 1024:.3f}KiB")
                else:
                    parts.append(f"{k}={avg:.0f}B")
            elif (
                k.endswith("_tokens")
                or k.endswith("_requests")
                or k.endswith("_count")
            ):
                parts.append(f"{k}={avg:.1f}")
            else:
                parts.append(f"{k}={avg * 1000.0:.3f}ms")
        print(f"{prefix} " + " ".join(parts), flush=True)
