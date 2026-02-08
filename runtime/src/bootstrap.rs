use anyhow::{bail, ensure, Context, Result};

use std::fs;
use std::path::PathBuf;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::adapter;
use crate::auth;
use crate::context;
use crate::device;
use crate::inference;
use crate::inference::kvcache::PageStore;
use crate::model;
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
    pub stop_tokens: Vec<u32>,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub devices: Vec<DeviceConfig>,
    pub scheduler: SchedulerConfig,
}

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub hostname: String,
    pub total_pages: usize,
    pub max_batch_tokens: usize,
    pub max_batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_in_flight_batches: usize,
    pub request_timeout_secs: u64,
    pub max_wait_ms: u64,
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

    verify_config(&config)?;

    init_tracing(&config.log_dir, config.verbose, &config.telemetry)?;
    let wasm_engine = init_wasmtime();

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

    for cfg in config.models.iter() {

        model::register(
            cfg.name.clone(),
            cfg.chat_template.clone(),
            cfg.stop_tokens.clone(),
            cfg.kv_page_size as u32,
            cfg.tokenizer_path.clone(),
        )?;

        let devices: Vec<usize> = cfg.devices.iter().map(|d| {
            device::spawn(&d.hostname, d.total_pages, d.max_batch_size, d.max_batch_tokens)
        }).collect();

        let page_store = PageStore::new(cfg.kv_page_size, &devices).await;

        context::spawn(page_store.clone());
        inference::spawn(
            page_store,
            cfg.scheduler.max_in_flight_batches,
            cfg.scheduler.request_timeout_secs,
            cfg.scheduler.max_wait_ms,
            cfg.scheduler.min_batch_for_optimization,
        ).await;
        adapter::spawn(&devices);
    }

    Ok(auth::get_internal_auth_token().await?)
}

fn verify_config(config: &Config) -> Result<()> {

    fs::create_dir_all(&config.cache_dir)
        .with_context(|| format!("Could not create cache dir: {:?}", config.cache_dir))?;

    if config.auth.enabled {
        fs::create_dir_all(&config.auth.authorized_users_dir)
            .with_context(|| format!("Could not create auth users dir: {:?}", config.auth.authorized_users_dir))?;
    }

    ensure!(!config.models.is_empty(), "No models configured");

    let mut seen_names = std::collections::HashSet::new();
    for model in &config.models {
        ensure!(
            seen_names.insert(&model.name),
            "Duplicate model name: {:?}", model.name
        );
        ensure!(!model.name.is_empty(), "Model name must not be empty");
        ensure!(!model.devices.is_empty(), "Model {:?} has no devices", model.name);
        ensure!(
            model.tokenizer_path.exists(),
            "Model {:?}: tokenizer not found at {:?}", model.name, model.tokenizer_path
        );

        for (i, dev) in model.devices.iter().enumerate() {
            ensure!(dev.total_pages > 0, "Model {:?} device {i}: total_pages must be > 0", model.name);
            ensure!(dev.max_batch_size > 0, "Model {:?} device {i}: max_batch_size must be > 0", model.name);
            ensure!(dev.max_batch_tokens > 0, "Model {:?} device {i}: max_batch_tokens must be > 0", model.name);
        }

        let sched = &model.scheduler;
        ensure!(sched.max_in_flight_batches > 0, "Model {:?}: max_in_flight_batches must be > 0", model.name);
        ensure!(sched.request_timeout_secs > 0, "Model {:?}: request_timeout_secs must be > 0", model.name);
    }

    Ok(())
}


fn init_wasmtime() -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();
    wasm_config.async_support(true);

    // TODO: Adjust settings later: https://docs.wasmtime.dev/api/wasmtime/struct.PoolingAllocationConfig.html
    // let mut pooling_config = PoolingAllocationConfig::default();
    // wasm_config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling_config));
    
    wasmtime::Engine::new(&wasm_config).unwrap()
}

/// Initialize the tracing subscriber with optional file logging and OTLP export.
fn init_tracing(
    log_dir: &Option<PathBuf>,
    verbose: bool,
    telemetry_config: &TelemetryConfig,
) -> Result<()> {
    use tracing_subscriber::fmt;
    use tracing_subscriber::EnvFilter;

    let default_level = if verbose { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    // Optional file writer layer
    let file_layer = if let Some(dir) = log_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create log directory: {dir:?}"))?;

        let file_appender = tracing_appender::rolling::daily(dir, "pie.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        std::mem::forget(guard);

        Some(fmt::layer().with_writer(non_blocking).with_ansi(false))
    } else {
        None
    };

    // Optional OTLP layer
    let otel_layer = if telemetry_config.enabled {
        telemetry::init_otel_layer(&telemetry_config.endpoint, &telemetry_config.service_name)
    } else {
        None
    };

    // Stdout layer (only when no file logging)
    let stdout_layer = if log_dir.is_none() {
        Some(fmt::layer())
    } else {
        None
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(otel_layer)
        .with(stdout_layer)
        .init();

    Ok(())
}
