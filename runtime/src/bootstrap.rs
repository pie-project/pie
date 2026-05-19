use anyhow::{Context, Result, bail, ensure};

use std::fs;
use std::path::PathBuf;
use tokio::net::TcpListener;

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
use crate::http as http_mod;
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
    /// Skip tracing initialization (for tests — can only init once per process).
    pub skip_tracing: bool,
    /// Hard cap on the number of concurrent processes.
    /// `None` means no limit; `Some(n)` caps admission to `n`.
    pub max_concurrent_processes: Option<usize>,
    /// Whether to apply host-side snapshot optimization to Python components.
    /// Disable via `python_snapshot = false` in the engine config or the
    /// `--no-snapshot` CLI flag.
    pub python_snapshot: bool,
    /// `host:port` to bind the HTTP control plane (`/healthz`,
    /// `/v1/models`). `None` disables the listener. `host:0` asks the
    /// OS for an ephemeral port; the bound port is then surfaced on
    /// stdout (`HTTP_LISTEN=<host>:<port>`) and written to
    /// `$PIE_HOME/http.port`.
    pub http_listen: Option<String>,
}

/// Runtime tuning — tokio worker pool + wasmtime engine pool +
/// per-instance security policies (filesystem / network).
///
/// Every field is required: Python is the source of truth for defaults,
/// Rust just consumes whatever the caller sends. No fallback logic.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Number of tokio worker threads.
    pub worker_threads: usize,

    // ── wasmtime engine pool ────────────────────────────────────────
    //
    // The pooling allocator caps four resource classes (core_instances,
    // component_instances, memories, tables) — pie uses one of each per
    // inferlet, so we expose them as a single `wasm_max_instances` knob
    // and bump them in lockstep.
    /// Concurrent-inferlet cap (sets all four wasmtime `total_*` caps).
    pub wasm_max_instances: u32,
    /// Per-inferlet linear-memory cap, in MiB.
    pub wasm_max_memory_mb: usize,
    /// RAM kept warm per slot to skip remapping on respawn, in MiB.
    pub wasm_warm_memory_mb: usize,
    /// Prepared-but-idle inferlet slots kept ready for fast respawn.
    pub wasm_warm_slots: u32,

    // ── filesystem ───────────────────────────────────────────────────
    /// Mount per-process scratch dir at `/scratch` with full read+write.
    pub allow_fs: bool,
    /// Base dir under which per-process scratch dirs are created.
    /// Each instance gets `<base>/<process_id>`.
    pub fs_scratch_dir: PathBuf,

    // ── network ──────────────────────────────────────────────────────
    /// Expose the host network to inferlets (both `wasi:sockets` and
    /// `wasi:http`). When false, sockets are denied and the `wasi:http`
    /// linker binding is dropped entirely.
    pub allow_network: bool,
    /// Allowlist of `cidr[:port]` / `cidr:lo-hi`. `["*"]` = no
    /// restriction. NOTE: only filters `wasi:sockets`; `wasi:http`
    /// bypasses the per-socket hook. Set `allow_network = false` for
    /// tight outbound HTTP control.
    pub network_allowed_hosts: Vec<String>,

    // ── upload cap ───────────────────────────────────────────────────
    /// Per-upload cap on cumulative bytes (program installs +
    /// `session.send_file` blobs), in MiB.
    pub max_upload_mb: usize,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub arch_name: String,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub devices: Vec<DeviceConfig>,
    pub scheduler: SchedulerConfig,
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
    /// Batch-firing policy. Recognized: `"adaptive"`, `"eager"`, `"greedy"`.
    pub batch_policy: String,
    /// Wall-clock cap on a single forward-pass request, in seconds.
    pub request_timeout_secs: u64,
    /// Default compute-wallet cap for processes that do not declare an
    /// explicit token limit at launch. `None` = unlimited (no cap).
    /// `Some(n)` = default hard cap at `n` tokens.
    pub default_token_limit: Option<usize>,
    /// Default market endowment (in KV pages) for processes that do not
    /// declare an explicit token limit. One endowment unit = one page of
    /// long-run guaranteed GPU residency under contention.
    pub default_endowment_pages: usize,
    /// Admission oversubscription factor: maximum allowed ratio of
    /// `Σ endowment / total_gpu_pages`. Must be > 0. At 1.0 the provider
    /// guarantees every admitted process its full endowment at all times;
    /// at higher values the provider sells more entitlement than physical
    /// capacity, betting on non-peak duty cycles (like a typical airline).
    pub admission_oversubscription_factor: f64,
    /// Hard admission gate for the restore loop: pause restoring suspended
    /// contexts when any device's GPU page utilization exceeds this fraction.
    /// Prevents the evict→restore→re-evict thrash cascade. Range: (0.0, 1.0].
    pub restore_pause_at_utilization: f64,
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

#[derive(Debug, Clone)]
pub struct BootstrapHandle {
    pub token: String,
    pub port: u16,
    /// Bound HTTP control-plane address (`host:port`) if
    /// `Config::http_listen` was set, else `None`.
    pub http_listen: Option<String>,
}

pub async fn bootstrap(config: Config) -> Result<BootstrapHandle> {
    bootstrap_inner(config, None).await
}

pub async fn bootstrap_with_listener(
    config: Config,
    listener: TcpListener,
) -> Result<BootstrapHandle> {
    bootstrap_inner(config, Some(listener)).await
}

async fn bootstrap_inner(config: Config, listener: Option<TcpListener>) -> Result<BootstrapHandle> {
    verify_config(&config)?;

    if !config.skip_tracing {
        init_tracing(&config.log_dir, config.verbose, &config.telemetry)?;
    }
    let wasm_engine = init_wasmtime(&config.runtime);

    // Load the Python runtime shared modules (full + stripped variants) before
    // the linker and program services spawn, so both can read from the shared
    // runtime state rather than loading their own copies.
    crate::program::python::runtime::init(&wasm_engine, config.python_snapshot);

    auth::spawn(config.auth.enabled, &config.auth.authorized_users_dir);

    program::spawn(
        &wasm_engine,
        config.registry_url.clone(),
        config.cache_dir.clone(),
    );

    // Compile per-instance security policies once. Network policy
    // parsing fails fast on bad config (typo'd CIDRs, `"*"` mixed with
    // rules, etc.) — better here than on the first inferlet launch.
    let fs_policy = crate::policy::FsPolicy {
        allow: config.runtime.allow_fs,
        base_dir: config.runtime.fs_scratch_dir.clone(),
    };
    let network_policy = crate::policy::NetworkPolicy::parse(
        config.runtime.allow_network,
        &config.runtime.network_allowed_hosts,
    )?;

    linker::spawn(&wasm_engine, fs_policy, network_policy);
    let max_upload_bytes = config.runtime.max_upload_mb.saturating_mul(1024 * 1024);
    let bound_port = match listener {
        Some(listener) => server::spawn_listener(listener, max_upload_bytes)?,
        None => server::spawn(&config.host, config.port, max_upload_bytes).await?,
    };
    messaging::spawn();
    process::init_admission(config.max_concurrent_processes);

    for cfg in config.models.iter() {
        model::register(
            cfg.name.clone(),
            &cfg.arch_name,
            cfg.kv_page_size as u32,
            cfg.tokenizer_path.clone(),
        )?;

        let devices: Vec<usize> = cfg
            .devices
            .iter()
            .map(|d| {
                device::spawn(
                    &d.hostname,
                    d.total_pages,
                    d.max_batch_size,
                    d.max_batch_tokens,
                )
            })
            .collect();

        let num_gpu_pages: Vec<usize> = cfg.devices.iter().map(|d| d.total_pages).collect();
        let num_cpu_pages: Vec<usize> = cfg.devices.iter().map(|d| d.cpu_pages).collect();

        context::spawn(
            cfg.kv_page_size,
            num_gpu_pages,
            num_cpu_pages,
            cfg.scheduler.default_endowment_pages.max(1),
            cfg.scheduler.default_token_limit,
            cfg.scheduler.admission_oversubscription_factor,
            cfg.scheduler.restore_pause_at_utilization,
        );
        inference::spawn(
            &devices,
            cfg.kv_page_size as u32,
            cfg.scheduler.request_timeout_secs,
            cfg.scheduler.batch_policy.clone(),
        )
        .await;
        adapter::spawn(&devices);
    }

    let http_listen = match &config.http_listen {
        Some(spec) => Some(start_http_listener(spec, &config).await?),
        None => None,
    };

    Ok(BootstrapHandle {
        token: auth::get_internal_auth_token().await?,
        port: bound_port,
        http_listen,
    })
}

/// Parse `host:port`, bind the HTTP control plane, write
/// `$PIE_HOME/http.port`, and print the stdout handshake
/// (`HTTP_LISTEN=<host>:<port>`). Returns the bound `host:port`.
async fn start_http_listener(spec: &str, config: &Config) -> Result<String> {
    let (host, port) = parse_host_port(spec)
        .with_context(|| format!("parsing --http-listen value {spec:?}"))?;
    let model_name = config
        .models
        .first()
        .map(|m| m.name.clone())
        .unwrap_or_default();
    let state = http_mod::State_ {
        model_name,
        created_unix: http_mod::now_unix(),
        started: std::time::Instant::now(),
    };
    let handle = http_mod::spawn(&host, port, state)
        .await
        .context("spawning http control plane")?;
    let bound = format!("{}:{}", host, handle.addr.port());

    // Best-effort `$PIE_HOME/http.port` so external supervisors (the
    // macOS test harness's IsolatedTestCase.boundHTTPPort poll) can
    // discover the ephemeral port without parsing stdout.
    let port_file = crate::path::get_pie_home().join("http.port");
    if let Some(parent) = port_file.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            tracing::warn!("create {parent:?} for http.port: {e}");
        }
    }
    if let Err(e) = fs::write(&port_file, format!("{}\n", handle.addr.port())) {
        tracing::warn!("write {port_file:?}: {e}");
    }

    // Stdout handshake — parsed by the standalone CLI / launcher.
    // Keep on one line, prefixed `HTTP_LISTEN=`.
    println!("HTTP_LISTEN={bound}");

    Ok(bound)
}

/// Split a `host:port` listener spec. Tolerates bracketed IPv6
/// literals so `[::1]:8080` parses correctly.
fn parse_host_port(spec: &str) -> Result<(String, u16)> {
    let trimmed = spec.trim();
    ensure!(!trimmed.is_empty(), "empty listener spec");
    // IPv6 bracketed form: `[<addr>]:<port>`.
    if let Some(rest) = trimmed.strip_prefix('[') {
        let (host, tail) = rest
            .split_once("]:")
            .ok_or_else(|| anyhow::anyhow!("missing ']:port' suffix in {trimmed:?}"))?;
        let port: u16 = tail
            .parse()
            .with_context(|| format!("parsing port in {trimmed:?}"))?;
        return Ok((host.to_string(), port));
    }
    let (host, port_s) = trimmed
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("expected host:port, got {trimmed:?}"))?;
    let port: u16 = port_s
        .parse()
        .with_context(|| format!("parsing port in {trimmed:?}"))?;
    Ok((host.to_string(), port))
}

/// Boot-time checks for the values pie's Python layer cannot validate
/// itself: filesystem-side effects (cache/auth dirs) and worker-handshake
/// outputs (tokenizer file, device capability numbers). Field-level
/// validation of user-supplied scalars (`batch_policy`, timeouts, market
/// knobs, etc.) happens in `pie.config.*.__post_init__` — by the time
/// they reach Rust they're already known-good.
fn verify_config(config: &Config) -> Result<()> {
    fs::create_dir_all(&config.cache_dir)
        .with_context(|| format!("Could not create cache dir: {:?}", config.cache_dir))?;

    if config.auth.enabled {
        fs::create_dir_all(&config.auth.authorized_users_dir).with_context(|| {
            format!(
                "Could not create auth users dir: {:?}",
                config.auth.authorized_users_dir
            )
        })?;
    }

    for model in &config.models {
        ensure!(
            model.tokenizer_path.exists(),
            "Model {:?}: tokenizer not found at {:?}",
            model.name,
            model.tokenizer_path
        );
        for (i, dev) in model.devices.iter().enumerate() {
            ensure!(
                dev.total_pages > 0,
                "Model {:?} device {i}: total_pages must be > 0",
                model.name
            );
            ensure!(
                dev.max_batch_size > 0,
                "Model {:?} device {i}: max_batch_size must be > 0",
                model.name
            );
            ensure!(
                dev.max_batch_tokens > 0,
                "Model {:?} device {i}: max_batch_tokens must be > 0",
                model.name
            );
        }
    }

    Ok(())
}

fn init_wasmtime(runtime: &RuntimeConfig) -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();
    wasm_config.async_support(true);

    // Every wasmtime knob comes from the caller — Python is the source
    // of truth for defaults. The `wasm_max_instances` knob covers four
    // wasmtime resource classes (pie uses one of each per inferlet).
    let mut pooling_config = wasmtime::PoolingAllocationConfig::default();
    // Lockstep bump on the five "total_*" caps. wasmtime defaults all of them
    // to 1000; pie uses exactly one of each per inferlet (one core instance,
    // one component instance, one memory, one table, one async fiber stack).
    pooling_config.total_core_instances(runtime.wasm_max_instances);
    pooling_config.total_component_instances(runtime.wasm_max_instances);
    pooling_config.total_memories(runtime.wasm_max_instances);
    pooling_config.total_tables(runtime.wasm_max_instances);
    pooling_config.total_stacks(runtime.wasm_max_instances);
    pooling_config.max_memory_size(runtime.wasm_max_memory_mb.saturating_mul(1024 * 1024));
    pooling_config
        .linear_memory_keep_resident(runtime.wasm_warm_memory_mb.saturating_mul(1024 * 1024));
    pooling_config.max_unused_warm_slots(runtime.wasm_warm_slots);

    wasm_config.allocation_strategy(wasmtime::InstanceAllocationStrategy::Pooling(
        pooling_config,
    ));

    wasmtime::Engine::new(&wasm_config).unwrap()
}

/// Initialize the tracing subscriber with optional file logging and OTLP export.
fn init_tracing(
    log_dir: &Option<PathBuf>,
    verbose: bool,
    telemetry_config: &TelemetryConfig,
) -> Result<()> {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;

    let default_level = if verbose { "debug" } else { "info" };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

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

#[cfg(test)]
mod tests {
    use super::parse_host_port;

    #[test]
    fn parse_host_port_ipv4() {
        let (host, port) = parse_host_port("127.0.0.1:8080").unwrap();
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 8080);
    }

    #[test]
    fn parse_host_port_zero_port() {
        let (host, port) = parse_host_port("0.0.0.0:0").unwrap();
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 0);
    }

    #[test]
    fn parse_host_port_hostname() {
        let (host, port) = parse_host_port("localhost:9090").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 9090);
    }

    #[test]
    fn parse_host_port_ipv6_bracketed() {
        let (host, port) = parse_host_port("[::1]:8080").unwrap();
        assert_eq!(host, "::1");
        assert_eq!(port, 8080);
    }

    #[test]
    fn parse_host_port_rejects_empty() {
        assert!(parse_host_port("").is_err());
        assert!(parse_host_port("   ").is_err());
    }

    #[test]
    fn parse_host_port_rejects_missing_port() {
        assert!(parse_host_port("127.0.0.1").is_err());
    }

    #[test]
    fn parse_host_port_rejects_bad_port() {
        assert!(parse_host_port("127.0.0.1:abc").is_err());
        assert!(parse_host_port("127.0.0.1:99999").is_err());
    }
}
