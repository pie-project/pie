//! OpenTelemetry telemetry module for Pie runtime.
//!
//! This module provides OTLP tracing and metrics export to SigNoz.
//! Telemetry is controlled via the `[telemetry]` section in pie config.

use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::TracerProvider;
use opentelemetry_sdk::{runtime, Resource};
use std::sync::OnceLock;

/// Global metrics holder
static METRICS: OnceLock<Metrics> = OnceLock::new();

/// All metric instruments for the Pie runtime
pub struct Metrics {
    // Scheduler metrics
    pub scheduler_arrival_rate: Gauge<f64>,
    pub scheduler_estimated_latency_ms: Gauge<f64>,
    pub scheduler_batch_wait_time_ms: Histogram<f64>,
    pub scheduler_fire_decisions: Counter<u64>,

    // Resource metrics
    pub kv_pages_allocated: Gauge<u64>,
    pub kv_pages_available: Gauge<u64>,
    pub kv_pages_oom_kills: Counter<u64>,
    pub instances_active: Gauge<u64>,

    // FFI metrics
    pub ffi_serialize_us: Histogram<u64>,
    pub ffi_queue_push_us: Histogram<u64>,
    pub ffi_response_wait_us: Histogram<u64>,
    pub ffi_deserialize_us: Histogram<u64>,
    pub ffi_total_us: Histogram<u64>,
}

impl Metrics {
    fn new(meter: &Meter) -> Self {
        Self {
            // Scheduler metrics
            scheduler_arrival_rate: meter
                .f64_gauge("scheduler.arrival_rate")
                .with_description("Requests per second (EMA-based)")
                .with_unit("req/s")
                .build(),
            scheduler_estimated_latency_ms: meter
                .f64_gauge("scheduler.estimated_latency_ms")
                .with_description("Predicted batch latency")
                .with_unit("ms")
                .build(),
            scheduler_batch_wait_time_ms: meter
                .f64_histogram("scheduler.batch_wait_time_ms")
                .with_description("Time waiting for batch to fill")
                .with_unit("ms")
                .build(),
            scheduler_fire_decisions: meter
                .u64_counter("scheduler.fire_decisions")
                .with_description("Fire now vs wait decisions")
                .build(),

            // Resource metrics
            kv_pages_allocated: meter
                .u64_gauge("kv_pages.allocated")
                .with_description("Currently allocated KV pages")
                .build(),
            kv_pages_available: meter
                .u64_gauge("kv_pages.available")
                .with_description("Free KV pages")
                .build(),
            kv_pages_oom_kills: meter
                .u64_counter("kv_pages.oom_kills")
                .with_description("OOM preemption events")
                .build(),
            instances_active: meter
                .u64_gauge("instances.active")
                .with_description("Active inference instances")
                .build(),

            // FFI metrics
            ffi_serialize_us: meter
                .u64_histogram("ffi.serialize_us")
                .with_description("Rust→Python serialization time")
                .with_unit("us")
                .build(),
            ffi_queue_push_us: meter
                .u64_histogram("ffi.queue_push_us")
                .with_description("Lock-free queue push time")
                .with_unit("us")
                .build(),
            ffi_response_wait_us: meter
                .u64_histogram("ffi.response_wait_us")
                .with_description("Waiting for Python response")
                .with_unit("us")
                .build(),
            ffi_deserialize_us: meter
                .u64_histogram("ffi.deserialize_us")
                .with_description("Python→Rust deserialization time")
                .with_unit("us")
                .build(),
            ffi_total_us: meter
                .u64_histogram("ffi.total_us")
                .with_description("E2E FFI call latency")
                .with_unit("us")
                .build(),
        }
    }
}

/// Get a reference to the global metrics.
/// Returns None if metrics have not been initialized.
pub fn metrics() -> Option<&'static Metrics> {
    METRICS.get()
}

/// Initialize OpenTelemetry OTLP tracing and metrics.
///
/// Returns an optional tracing-opentelemetry layer that can be added to the subscriber.
/// Caller is responsible for only calling this when telemetry is enabled.
pub fn init_otel_layer<S>(
    endpoint: &str,
    service_name: &str,
) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    eprintln!("[Telemetry] Initializing OTLP export to: {endpoint}");

    // Build resource with service name
    let resource = Resource::new(vec![KeyValue::new(
        "service.name",
        service_name.to_string(),
    )]);

    // === Initialize Metrics ===
    let metrics_exporter = match opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Failed to create OTLP metrics exporter: {}, metrics disabled",
                err
            );
            // Continue without metrics
            return init_tracing_only(endpoint, resource);
        }
    };

    let reader = PeriodicReader::builder(metrics_exporter, runtime::Tokio).build();
    
    let meter_provider = SdkMeterProvider::builder()
        .with_resource(resource.clone())
        .with_reader(reader)
        .build();

    // Create and store global metrics
    let meter = meter_provider.meter("pie-runtime");
    let _ = METRICS.set(Metrics::new(&meter));

    // Set global meter provider
    opentelemetry::global::set_meter_provider(meter_provider);

    // === Initialize Tracing ===
    let span_exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Failed to create OTLP span exporter: {}, tracing disabled",
                err
            );
            return None;
        }
    };

    let provider = TracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(span_exporter, runtime::Tokio)
        .build();

    let tracer = provider.tracer("pie-runtime");
    opentelemetry::global::set_tracer_provider(provider);

    eprintln!("[Telemetry] OTLP tracing and metrics initialized successfully");

    Some(tracing_opentelemetry::layer().with_tracer(tracer))
}

/// Initialize tracing only (when metrics fails to init)
fn init_tracing_only<S>(
    endpoint: &str,
    resource: Resource,
) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    let span_exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Failed to create OTLP span exporter: {}, telemetry disabled",
                err
            );
            return None;
        }
    };

    let provider = TracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(span_exporter, runtime::Tokio)
        .build();

    let tracer = provider.tracer("pie-runtime");
    opentelemetry::global::set_tracer_provider(provider);

    Some(tracing_opentelemetry::layer().with_tracer(tracer))
}

/// Shutdown OpenTelemetry, flushing any pending spans and metrics.
pub fn shutdown() {
    if METRICS.get().is_some() {
        opentelemetry::global::shutdown_tracer_provider();
        // Note: Meter provider shutdown happens automatically on drop
    }
}
