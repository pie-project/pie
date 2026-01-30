use anyhow::{Context, Result};
// use ring::rand::{SecureRandom, SystemRandom};
use std::fs;
use std::path::PathBuf;
use tokio::sync::oneshot;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use wasmtime::{Config as WasmConfig, Engine as WasmEngine};

use crate::auth::AuthorizedUsers;
use crate::kvs;
use crate::messaging;
use crate::runtime;
use crate::server;
use crate::telemetry::{self, TelemetryConfig};

/// Configuration for the PIE engine.
#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log_dir: Option<PathBuf>,
    pub registry: String,
    pub telemetry: TelemetryConfig,
}

/// Runs the PIE server logic within an existing Tokio runtime.
///
/// This async function sets up all the engine's services and listens for a shutdown
/// signal to terminate gracefully.
pub async fn run_server(
    config: Config,
    authorized_users: AuthorizedUsers,
    ready_tx: oneshot::Sender<String>,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<()> {
    // Install panic hook to exit process on any Rust panic.
    // This ensures Python parent process is notified when Rust crashes.
    std::panic::set_hook(Box::new(|panic_info| {
        // Log the panic with location if available
        let location = panic_info.location().map(|loc| {
            format!("{}:{}:{}", loc.file(), loc.line(), loc.column())
        }).unwrap_or_else(|| "unknown location".to_string());
        
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        
        eprintln!("\n[FATAL] Rust runtime panic at {}: {}", location, message);
        eprintln!("[FATAL] Terminating process to signal Python shutdown.");
        
        // Exit with non-zero code to signal failure to Python
        //std::process::exit(1);
    }));

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


    let server_url = format!("{}:{}", config.host, config.port);

    // Generate a random 64-character string for internal client connection authentication.
    let internal_auth_token = crate::auth::generate_internal_auth_token()?;

    // Create the Wasmtime engine (shared between runtime and server)
    let mut wasm_config = WasmConfig::default();
    wasm_config.async_support(true);

    // TODO: Adjust settings later: https://docs.wasmtime.dev/api/wasmtime/struct.PoolingAllocationConfig.html
    // let mut pooling_config = PoolingAllocationConfig::default();
    // wasm_config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling_config));
    
    let wasm_engine = WasmEngine::new(&wasm_config).unwrap();

    runtime::start_service(wasm_engine.clone());
    server::start_service(
        &server_url,
        config.enable_auth,
        authorized_users,
        internal_auth_token.clone(),
        config.registry.clone(),
        config.cache_dir.clone(),
        wasm_engine,
    );
    kvs::start_service();
    messaging::start_service();

    ready_tx.send(internal_auth_token).unwrap();

    shutdown_rx.await?;

    Ok(())
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

            if let Some(otel_layer) = telemetry::init_otel_layer(telemetry_config) {
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
            if let Some(otel_layer) = telemetry::init_otel_layer(telemetry_config) {
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
