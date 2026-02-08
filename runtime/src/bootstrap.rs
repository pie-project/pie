use anyhow::{Context, Result};
// use ring::rand::{SecureRandom, SystemRandom};

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::adapter;
use crate::auth;
use crate::context;
use crate::device;
use crate::inference;

use crate::kvcache::PageStore;
use crate::program;
use crate::runtime;
use crate::server;
use crate::telemetry;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub auth: AuthConfig,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log_dir: Option<PathBuf>,
    pub registry_url: String,
    pub telemetry: TelemetryConfig,
    pub models: Vec<ModelConfig>,
}


#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub chat_template: String,
    pub stop_tokens: Vec<String>,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub devices: Vec<DeviceConfig>,
}

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub hostname: String,
    pub total_pages: usize,
    pub max_batch_tokens: usize,
    pub max_batch_size: usize,
    /// Maximum number of batches in flight per device (default: 3).
    pub max_in_flight_batches: usize,
    /// RPC request timeout in seconds (default: 30).
    pub request_timeout_secs: u64,
    /// Maximum wait time before forcing a batch fire, in ms (default: 50).
    pub max_wait_ms: u64,
    /// Minimum batch size before considering throughput optimization (default: 8).
    pub min_batch_for_optimization: usize,
}

#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub service_name: String,
}

#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub enabled: bool,
    pub authorized_users_dir: PathBuf,
}

pub async fn bootstrap(
    config: Config,
) -> Result<String> {
    // Initialize tracing with file logging if log_dir is specified
    init_tracing(&config.log_dir, config.verbose, &config.telemetry)?;

    // Ensure the cache directory exists
    fs::create_dir_all(&config.cache_dir).with_context(|| {
        let err_msg = format!(
            "Setup failure: could not create cache dir at {:?}",
            &config.cache_dir
        );
        tracing::error!(error = %err_msg);
        err_msg
    })?;

    // Create the Wasmtime engine (shared between runtime and server)
    let mut wasm_config = wasmtime::Config::default();
    wasm_config.async_support(true);

    // TODO: Adjust settings later: https://docs.wasmtime.dev/api/wasmtime/struct.PoolingAllocationConfig.html
    // let mut pooling_config = PoolingAllocationConfig::default();
    // wasm_config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling_config));
    
    let wasm_engine = wasmtime::Engine::new(&wasm_config).unwrap();


    // Spawn the auth actor
    auth::spawn(
        config.auth.enabled,
        &config.auth.authorized_users_dir,
    );

    program::spawn(
        wasm_engine.clone(),
        config.registry_url.clone(),
        config.cache_dir.clone(),
    );

    runtime::spawn(wasm_engine);
    server::spawn(&config.host, config.port);

    // Spawn device services (flat across all models)
    let mut device_indices_per_model: Vec<Vec<usize>> = Vec::new();
    for model_config in &config.models {
        let mut model_device_ids = Vec::new();
        for (i, dev_cfg) in model_config.devices.iter().enumerate() {
            let idx = device::spawn(i as u8, dev_cfg);
            model_device_ids.push(idx);
        }
        device_indices_per_model.push(model_device_ids);
    }

    // Spawn per-model services: context, adapter, inference
    for (model_idx, model_config) in config.models.iter().enumerate() {
        let page_store = Arc::new(RwLock::new(PageStore::new(model_config.kv_page_size, &model_config.devices)));

        // Spawn services with shared page store
        context::spawn(page_store.clone());
        inference::spawn(page_store, &model_config.devices, &device_indices_per_model[model_idx]);
        adapter::spawn(&model_config.devices);
    }

    Ok(auth::get_internal_auth_token().await?)
}

/// Initialize the tracing subscriber with optional file logging and OTLP export.
fn init_tracing(
    log_dir: &Option<PathBuf>,
    verbose: bool,
    telemetry_config: &TelemetryConfig,
) -> Result<()> {
    use tracing_subscriber::fmt;
    use tracing_subscriber::EnvFilter;


    let filter = if verbose {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"))
    };

    // Build the base registry with filter
    let registry = tracing_subscriber::registry().with(filter);

    match (log_dir, telemetry_config.enabled) {
        // File logging + OTLP
        (Some(dir), true) => {
            fs::create_dir_all(dir).with_context(|| {
                format!("Failed to create log directory: {:?}", dir)
            })?;

            let file_appender = tracing_appender::rolling::daily(dir, "pie.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            std::mem::forget(_guard);

            if let Some(otel_layer) = telemetry::init_otel_layer(
                &telemetry_config.endpoint,
                &telemetry_config.service_name,
            ) {
                registry
                    .with(otel_layer)
                    .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
                    .init();
            } else {
                // OTLP creation failed, just use file logging
                registry
                    .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
                    .init();
            }
        }
        // File logging only
        (Some(dir), false) => {
            fs::create_dir_all(dir).with_context(|| {
                format!("Failed to create log directory: {:?}", dir)
            })?;

            let file_appender = tracing_appender::rolling::daily(dir, "pie.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            std::mem::forget(_guard);

            registry
                .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
                .init();
        }
        // Stdout + OTLP
        (None, true) => {
            if let Some(otel_layer) = telemetry::init_otel_layer(
                &telemetry_config.endpoint,
                &telemetry_config.service_name,
            ) {
                registry
                    .with(otel_layer)
                    .with(fmt::layer())
                    .init();
            } else {
                // OTLP creation failed, just use stdout
                registry.with(fmt::layer()).init();
            }
        }
        // Stdout only
        (None, false) => {
            registry.with(fmt::layer()).init();
        }
    }

    Ok(())
}
