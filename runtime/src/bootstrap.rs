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
use crate::linker;
use crate::messaging;
use crate::model;
use crate::process;
use crate::program;
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
    pub runtime: RuntimeConfig,
    pub models: Vec<ModelConfig>,
    /// Allow inferlets to access a sandboxed scratch filesystem.
    pub allow_filesystem: bool,
    /// Skip tracing initialization (for tests — can only init once per process).
    pub skip_tracing: bool,
    /// Hard cap on the number of concurrent processes.
    /// `None` means no limit; `Some(n)` caps admission to `n`.
    pub max_concurrent_processes: Option<usize>,
    /// Whether to apply host-side snapshot optimization to Python components.
    /// Disable via `python_snapshot = false` in the engine config or the
    /// `--no-snapshot` CLI flag.
    pub python_snapshot: bool,
}

/// Tokio runtime tuning. All fields are opt-in — leaving them at their
/// defaults yields the stock `tokio::runtime::Runtime::new()` behavior.
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    /// Number of tokio worker threads. `None` = let tokio default to
    /// `num_cpus`. On boxes with high logical-core counts and many
    /// in-flight tasks, lowering this (e.g. to 8) cuts migration /
    /// context-switch overhead and can give a substantial throughput
    /// win — see comments in `RuntimeConfig` on the Python side.
    pub worker_threads: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub arch_name: String,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub devices: Vec<DeviceConfig>,
    pub scheduler: SchedulerConfig,
    /// Default compute-wallet cap for processes that do not declare an
    /// explicit token budget at launch. `None` = unlimited (no cap); most
    /// deployments should leave this as `None` and let clients opt into
    /// caps per-launch. `Some(n)` = default hard cap at `n` tokens.
    pub default_token_budget: Option<usize>,
    /// Default market endowment (in KV pages) assigned to processes that
    /// do not declare an explicit token budget. One endowment unit = one
    /// page of long-run guaranteed GPU residency under contention.
    pub default_endowment_pages: usize,
    /// Admission oversubscription factor: maximum allowed ratio of
    /// `Σ endowment / total_gpu_pages`. Must be > 0. At 1.0 the provider
    /// guarantees every admitted process its full endowment at all times;
    /// at higher values the provider sells more entitlement than physical
    /// capacity, betting on non-peak duty cycles (like a typical airline).
    pub oversubscription_factor: f64,
}

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub hostname: String,
    pub total_pages: usize,
    pub cpu_pages: usize,
    pub max_batch_tokens: usize,
    pub max_batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub request_timeout_secs: u64,
    /// Optional batch-firing policy. `None` = use built-in default.
    /// Recognized values: `"adaptive"`, `"eager"`, `"greedy"`.
    pub policy: Option<String>,
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

    if !config.skip_tracing {
        init_tracing(&config.log_dir, config.verbose, &config.telemetry)?;
    }
    let wasm_engine = init_wasmtime();

    // Load the Python runtime shared modules (full + stripped variants) before
    // the linker and program services spawn, so both can read from the shared
    // runtime state rather than loading their own copies.
    crate::program::python::runtime::init(&wasm_engine, config.python_snapshot);

    auth::spawn(
        config.auth.enabled,
        &config.auth.authorized_users_dir,
    );

    program::spawn(
        &wasm_engine,
        config.registry_url.clone(),
        config.cache_dir.clone(),
    );

    linker::spawn(&wasm_engine, config.allow_filesystem);
    server::spawn(&config.host, config.port);
    messaging::spawn();
    process::init_admission(config.max_concurrent_processes);
    
    for cfg in config.models.iter() {

        model::register(
            cfg.name.clone(),
            &cfg.arch_name,
            cfg.kv_page_size as u32,
            cfg.tokenizer_path.clone(),
        )?;

        let devices: Vec<usize> = cfg.devices.iter().map(|d| {
            device::spawn(&d.hostname, d.total_pages, d.max_batch_size, d.max_batch_tokens)
        }).collect();

        let num_gpu_pages: Vec<usize> = cfg.devices.iter().map(|d| d.total_pages).collect();
        let num_cpu_pages: Vec<usize> = cfg.devices.iter().map(|d| d.cpu_pages).collect();

        context::spawn(
            cfg.kv_page_size,
            num_gpu_pages,
            num_cpu_pages,
            cfg.default_endowment_pages.max(1),
            cfg.default_token_budget,
            cfg.oversubscription_factor,
        );
        inference::spawn(
            &devices,
            cfg.kv_page_size as u32,
            cfg.scheduler.request_timeout_secs,
            cfg.scheduler.policy.clone(),
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
        if let Some(t) = model.default_token_budget {
            ensure!(
                t > 0,
                "Model {:?}: default_token_budget, when set, must be > 0 (got {})",
                model.name, t
            );
        }
        ensure!(
            model.default_endowment_pages > 0,
            "Model {:?}: default_endowment_pages must be > 0 (got {})",
            model.name, model.default_endowment_pages
        );
        ensure!(
            model.oversubscription_factor > 0.0 && model.oversubscription_factor.is_finite(),
            "Model {:?}: oversubscription_factor must be > 0 and finite (got {})",
            model.name, model.oversubscription_factor
        );

        for (i, dev) in model.devices.iter().enumerate() {
            ensure!(dev.total_pages > 0, "Model {:?} device {i}: total_pages must be > 0", model.name);
            ensure!(dev.max_batch_size > 0, "Model {:?} device {i}: max_batch_size must be > 0", model.name);
            ensure!(dev.max_batch_tokens > 0, "Model {:?} device {i}: max_batch_tokens must be > 0", model.name);
        }

        let sched = &model.scheduler;
        ensure!(sched.request_timeout_secs > 0, "Model {:?}: request_timeout_secs must be > 0", model.name);
    }

    Ok(())
}


fn init_wasmtime() -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();
    wasm_config.async_support(true);

    let pooling_config = wasmtime::PoolingAllocationConfig::default();
    wasm_config
        .allocation_strategy(wasmtime::InstanceAllocationStrategy::Pooling(pooling_config));

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
