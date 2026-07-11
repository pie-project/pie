use anyhow::{Context, Result, ensure};

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::driver;
use crate::inferlet::sandbox::{FsPolicy, NetworkPolicy};
use crate::inferlet::{linker, process, program, python};
use crate::server;
use crate::telemetry;
use pie_model as model;

static RUNTIME_ACTIVE: AtomicBool = AtomicBool::new(false);

struct ActiveRuntimeGuard {
    armed: bool,
}

impl ActiveRuntimeGuard {
    fn acquire() -> Result<Self> {
        ensure!(
            !RUNTIME_ACTIVE.swap(true, Ordering::AcqRel),
            "runtime bootstrap is single-use in this process; start a fresh process for another engine"
        );
        Ok(Self { armed: true })
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ActiveRuntimeGuard {
    fn drop(&mut self) {
        if self.armed {
            RUNTIME_ACTIVE.store(false, Ordering::Release);
        }
    }
}

struct RuntimeShutdown {
    scheduler: crate::scheduler::SchedulerShutdownHandle,
    driver_ids: Vec<usize>,
}

impl RuntimeShutdown {
    async fn shutdown(self) -> Result<()> {
        let scheduler_result = self.scheduler.shutdown().await;
        for driver_id in self.driver_ids {
            let _ = driver::registry::unregister_driver(driver_id);
        }
        scheduler_result
    }
}

pub struct Config {
    pub host: String,
    pub port: u16,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log_dir: Option<PathBuf>,
    pub registry_url: String,
    pub telemetry: TelemetryConfig,
    pub runtime: RuntimeConfig,
    pub model: ModelConfig,
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
    /// Concrete py-runtime root passed in by the embedding worker.
    pub py_runtime_dir: PathBuf,
}

pub struct ModelConfig {
    pub name: String,
    pub arch_name: String,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub drivers: Vec<DriverConfig>,
    pub scheduler: SchedulerConfig,
}

pub struct DriverConfig {
    pub total_pages: usize,
    pub cpu_pages: usize,
    pub rs_cache_required: bool,
    pub rs_cache_slots: usize,
    pub rs_cache_slot_bytes: u64,
    pub limits: crate::driver::SchedulerLimits,
    pub native_driver: crate::driver::NativeDriver,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Wall-clock cap on a single forward-pass request, in seconds.
    pub request_timeout_secs: u64,
    /// Hard admission gate for the restore loop: pause restoring suspended
    /// contexts when any driver's GPU page utilization exceeds this fraction.
    /// Prevents the evict→restore→re-evict thrash cascade. Range: (0.0, 1.0].
    pub restore_pause_at_utilization: f64,
}

#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub service_name: String,
}

pub struct BootstrapHandle {
    pub port: u16,
    shutdown: Option<RuntimeShutdown>,
}

impl BootstrapHandle {
    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            shutdown.shutdown().await
        } else {
            Ok(())
        }
    }
}

pub async fn bootstrap(config: Config) -> Result<BootstrapHandle> {
    bootstrap_inner(config).await
}

pub async fn bootstrap_with_listener(
    config: Config,
    _listener: tokio::net::TcpListener,
) -> Result<BootstrapHandle> {
    // WebSocket listeners are no longer used; keep this shim so older callers
    // compile while migrating to edge-rpc.
    bootstrap_inner(config).await
}

async fn bootstrap_inner(config: Config) -> Result<BootstrapHandle> {
    verify_config(&config)?;
    let mut active_guard = ActiveRuntimeGuard::acquire()?;

    if !config.skip_tracing {
        init_tracing(&config.log_dir, config.verbose, &config.telemetry)?;
    }
    let wasm_engine = init_wasmtime(&config.runtime);

    // Load the Python runtime shared modules (full + stripped variants) before
    // the linker and program services spawn, so both can read from the shared
    // runtime state rather than loading their own copies.
    // The Python runtime shared modules must load before the linker and
    // program services spawn, so both read from shared runtime state.
    python::runtime::init(
        &wasm_engine,
        &config.runtime.py_runtime_dir,
        config.python_snapshot,
    );

    program::spawn(
        &wasm_engine,
        config.registry_url.clone(),
        config.cache_dir.clone(),
    );

    // Compile per-instance security policies once. Network policy
    // parsing fails fast on bad config (typo'd CIDRs, `"*"` mixed with
    // rules, etc.) — better here than on the first inferlet launch.
    let fs_policy = FsPolicy {
        allow: config.runtime.allow_fs,
        base_dir: config.runtime.fs_scratch_dir.clone(),
    };
    let network_policy = NetworkPolicy::parse(
        config.runtime.allow_network,
        &config.runtime.network_allowed_hosts,
    )?;

    linker::spawn(&wasm_engine, fs_policy, network_policy);
    let max_upload_bytes = config.runtime.max_upload_mb.saturating_mul(1024 * 1024);
    server::init(max_upload_bytes);
    let bound_port = config.port;
    process::init_admission(config.max_concurrent_processes);

    let ModelConfig {
        name,
        arch_name,
        kv_page_size,
        tokenizer_path,
        drivers: driver_configs,
        scheduler,
    } = config.model;
    // RS working-set caps from the driver handshake (uniform across a model's
    // drivers → take [0]). bravo-authored bootstrap bundle.
    let rs_caps = {
        let d0 = driver_configs.first();
        let is_rs = d0.map(|d| d.rs_cache_slots > 0).unwrap_or(false);
        model::RsCaps {
            state_size: d0.map(|d| d.rs_cache_slot_bytes).unwrap_or(0),
            buffer_page_size: if is_rs { kv_page_size as u32 } else { 0 },
            fold_granularity: 1, // token-causal; 0-RS models never read it
        }
    };
    model::register(
        name.clone(),
        &arch_name,
        kv_page_size as u32,
        rs_caps,
        tokenizer_path.clone(),
    )?;

    let arena_kv_pages: Vec<usize> = driver_configs.iter().map(|d| d.total_pages).collect();
    let arena_cpu_pages: Vec<usize> = driver_configs.iter().map(|d| d.cpu_pages).collect();
    let arena_rs_slots: Vec<usize> = driver_configs.iter().map(|d| d.rs_cache_slots).collect();
    let driver_count = driver_configs.len();
    let drivers: Vec<usize> = driver_configs
        .into_iter()
        .map(|d| {
            driver::register_native_driver(
                driver::DriverSpec {
                    num_kv_pages: d.total_pages,
                    limits: d.limits,
                },
                d.native_driver,
            )
        })
        .collect();

    // Register this model's per-driver typed stores (KvStore/RsStore) in the
    // standalone registry. Capacities are read straight from `cfg.drivers[]`.
    // The registry is where the WIT working-set resources and the PTIR fire
    // path lock `store::registry::get(...)`.
    let _ = driver_count;
    let arena_model_idx = crate::store::registry::register_model(
        kv_page_size as u32,
        &arena_kv_pages,
        &arena_rs_slots,
    );
    let _ = arena_cpu_pages;

    // Task-B contention orchestrator (`PIE_KV_CONTENTION=preempt`): KV pool
    // exhaustion becomes FCFS preempt/restore instead of an inferlet error;
    // `max_concurrent_processes` stays a physical safety cap only. Off by
    // default — the legacy error path is byte-for-byte unchanged. The
    // existing `scheduler.restore_pause_at_utilization` config gates the
    // restore anti-thrash pause (its long-reserved consumer).
    if crate::store::reclaim::contention_mode() == crate::store::reclaim::ContentionMode::Preempt {
        // Backend tiers: passive v1 (waiters ride natural frees — proven
        // e2e, M-AB ②③) by default; `PIE_KV_PREEMPT_ACTIVE=1` selects the
        // v2 self-suspend backend (active FCFS victim preempt).
        let backend: Box<dyn crate::store::reclaim::ReclaimBackend> =
            if crate::store::reclaim::preempt_active() {
                Box::new(crate::store::reclaim::SelfSuspendBackend::new(
                    arena_model_idx,
                    0,
                ))
            } else {
                Box::new(crate::store::reclaim::KvPoolBackend::new(
                    arena_model_idx,
                    0,
                ))
            };
        crate::store::reclaim::init_contention(crate::store::reclaim::ContentionOrchestrator::new(
            backend,
            scheduler.restore_pause_at_utilization,
        ));
    }

    // (Context actor `context::spawn` removed — Phase 5. The unified arena
    // registry above is the per-model/driver physical home now.)
    let scheduler_shutdown = crate::scheduler::spawn(
        &drivers,
        kv_page_size as u32,
        scheduler.request_timeout_secs,
    )
    .await?;

    active_guard.disarm();
    Ok(BootstrapHandle {
        port: bound_port,
        shutdown: Some(RuntimeShutdown {
            scheduler: scheduler_shutdown,
            driver_ids: drivers,
        }),
    })
}

/// Boot-time checks for the values pie's Python layer cannot validate
/// itself: filesystem-side effects (cache dir) and worker-handshake
/// outputs (tokenizer file, driver capability numbers). Field-level
/// validation of user-supplied scalars (timeouts, etc.) happens in
/// `pie.config.*.__post_init__` — by the time they reach Rust they're
/// already known-good.
fn verify_config(config: &Config) -> Result<()> {
    fs::create_dir_all(&config.cache_dir)
        .with_context(|| format!("Could not create cache dir: {:?}", config.cache_dir))?;

    let model = &config.model;
    ensure!(
        model.tokenizer_path.exists(),
        "Model {:?}: tokenizer not found at {:?}",
        model.name,
        model.tokenizer_path
    );
    for (i, dev) in model.drivers.iter().enumerate() {
        ensure!(
            dev.total_pages > 0,
            "Model {:?} driver {i}: total_pages must be > 0",
            model.name
        );
        ensure!(
            dev.limits.max_forward_tokens > 0,
            "Model {:?} driver {i}: max_forward_tokens must be > 0",
            model.name
        );
        ensure!(
            dev.limits.max_forward_requests > 0,
            "Model {:?} driver {i}: max_forward_requests must be > 0",
            model.name
        );
        ensure!(
            dev.limits.max_page_refs > 0,
            "Model {:?} driver {i}: max_page_refs must be > 0",
            model.name
        );
    }

    // `restore_pause_at_utilization` is a GPU-utilization fraction in (0.0, 1.0]:
    // the restore loop pauses while any driver's page utilization exceeds it.
    // A value <= 0.0 makes the check `utilization > threshold` true even on an
    // empty GPU (utilization == 0.0), so restore is paused forever and any
    // suspended context's deferred page op deadlocks. Reject it fast here rather
    // than hang at runtime.
    ensure!(
        model.scheduler.restore_pause_at_utilization > 0.0
            && model.scheduler.restore_pause_at_utilization <= 1.0,
        "Model {:?}: scheduler.restore_pause_at_utilization must be in (0.0, 1.0], got {}",
        model.name,
        model.scheduler.restore_pause_at_utilization
    );

    Ok(())
}

fn init_wasmtime(runtime: &RuntimeConfig) -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();
    // wasmtime 46: `async_support` is a deprecated no-op (async is always
    // compiled in) and the Component Model Async feature is on by default, so
    // no explicit flags are needed to enable async host calls / fibers.

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
