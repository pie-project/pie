use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::sync::OnceLock;

static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

pub fn init_otel_layer<S>(
    endpoint: &str,
    service_name: &str,
) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    eprintln!("[Telemetry] Initializing OTLP export to: {endpoint}");

    let resource = Resource::builder_empty()
        .with_attribute(KeyValue::new("service.name", service_name.to_string()))
        .build();

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
            return init_tracing_only(endpoint, resource);
        }
    };

    let reader = PeriodicReader::builder(metrics_exporter).build();

    let meter_provider = SdkMeterProvider::builder()
        .with_resource(resource.clone())
        .with_reader(reader)
        .build();

    let _ = METER_PROVIDER.set(meter_provider.clone());
    opentelemetry::global::set_meter_provider(meter_provider);

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

    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(span_exporter)
        .build();

    let tracer = provider.tracer("pie-runtime");
    let _ = TRACER_PROVIDER.set(provider.clone());
    opentelemetry::global::set_tracer_provider(provider);

    eprintln!("[Telemetry] OTLP tracing and metrics initialized successfully");

    Some(tracing_opentelemetry::layer().with_tracer(tracer))
}

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

    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(span_exporter)
        .build();

    let tracer = provider.tracer("pie-runtime");
    let _ = TRACER_PROVIDER.set(provider.clone());
    opentelemetry::global::set_tracer_provider(provider);

    Some(tracing_opentelemetry::layer().with_tracer(tracer))
}
