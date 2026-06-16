"""
OpenTelemetry telemetry module for PIE worker.

This module provides tracing functionality that can be enabled/disabled
via the [telemetry] section in the pie configuration.

When enabled: Spans are batched and exported asynchronously to an OTLP endpoint.
When disabled: NoOpTracerProvider is used for zero overhead.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.trace import Tracer, Span, Status, StatusCode

# Global state
_tracer: Tracer | None = None
_initialized: bool = False


def init_telemetry(
    enabled: bool,
    service_name: str = "pie",
    endpoint: str = "localhost:4317",
) -> None:
    """
    Initialize telemetry.

    Args:
        enabled: If True, set up real tracing with OTLP export.
                 If False, use NoOpTracerProvider for zero overhead.
        service_name: Service name to appear in Jaeger.
        endpoint: OTLP gRPC endpoint (e.g., "localhost:4317").
    """
    global _tracer, _initialized

    if _initialized:
        return

    if enabled:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create OTLP exporter (async via BatchSpanProcessor)
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=True,  # No TLS for local development
        )

        # BatchSpanProcessor batches spans and exports in background thread
        # Default: max 512 spans per batch, 5s schedule delay
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
    else:
        # NoOpTracerProvider - zero overhead
        from opentelemetry.trace import NoOpTracerProvider

        trace.set_tracer_provider(NoOpTracerProvider())
        _tracer = trace.get_tracer(__name__)

    _initialized = True


def get_tracer() -> Tracer:
    """
    Get the global tracer.

    Returns:
        The tracer (either real or no-op depending on initialization).
    """
    global _tracer

    if _tracer is None:
        # Not initialized - return no-op tracer
        _tracer = trace.get_tracer(__name__)

    return _tracer


def shutdown_telemetry() -> None:
    """Gracefully shutdown telemetry, flushing any pending spans."""
    global _initialized

    if _initialized:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        _initialized = False


@contextmanager
def create_span(name: str, **attributes) -> Generator[Span, None, None]:
    """
    Create a span as a context manager.

    Args:
        name: Span name (e.g., "pie.fire_batch").
        **attributes: Additional attributes to set on the span.

    Yields:
        The created span.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def record_timing(span: Span, name: str, duration_ms: float) -> None:
    """
    Record a timing as a span attribute.

    Args:
        span: The span to record on.
        name: Timing name (e.g., "build_batch_ms").
        duration_ms: Duration in milliseconds.
    """
    span.set_attribute(name, duration_ms)


@contextmanager
def start_span_with_traceparent(
    name: str, traceparent: str | None, **attributes
) -> Generator[Span, None, None]:
    """
    Start a span with an optional traceparent as parent context.

    This enables cross-language trace propagation by extracting
    the parent context from a W3C traceparent string.

    Args:
        name: Span name.
        traceparent: W3C traceparent string (e.g., "00-trace_id-span_id-flags").
        **attributes: Additional attributes to set on the span.

    Yields:
        The created span.
    """
    tracer = get_tracer()

    # Extract parent context from traceparent if provided
    context = None
    if traceparent:
        try:
            from opentelemetry.propagate import extract

            # Create carrier dict with traceparent header
            carrier = {"traceparent": traceparent}
            context = extract(carrier)
        except Exception:
            # Fall back to no parent context if extraction fails
            pass

    # Start span with optional parent context
    with tracer.start_as_current_span(name, context=context) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
