#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use anyhow::{Result, ensure};

use std::borrow::Cow;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::layer::SubscriberExt;
#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::util::SubscriberInitExt;

use crate::engine;
use crate::inferlet::sandbox::{FsPolicy, NetworkPolicy};
use crate::inferlet::{linker, process, program};
use crate::model::{self, ModelMetadata};
use crate::server;
#[cfg(not(target_arch = "wasm32"))]
use crate::telemetry;

static RUNTIME_ACTIVE: AtomicBool = AtomicBool::new(false);

struct ActiveRuntimeGuard {
    armed: bool,
}

impl ActiveRuntimeGuard {
    fn acquire() -> Result<Self> {
        ensure!(
            !RUNTIME_ACTIVE.swap(true, Ordering::AcqRel),
            "runtime bootstrap is single-use in this process; start a fresh process for another runtime"
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
    engine_ids: Vec<usize>,
}

impl RuntimeShutdown {
    async fn shutdown(self) -> Result<()> {
        let scheduler_result = self.scheduler.shutdown().await;
        for engine_id in self.engine_ids {
            let _ = engine::backend::unregister_engine(engine_id);
        }
        scheduler_result
    }
}

pub struct Config {
    pub host: String,
    pub port: u16,
    pub cache_dir: PathBuf,
    /// The built-in inferlets: registered at boot beside whatever
    /// `cache_dir` holds, each shadowed by a copy of its own name and
    /// version installed there.
    pub builtin_programs: Vec<BuiltinProgram>,
    pub verbose: bool,
    pub log_dir: Option<PathBuf>,
    pub telemetry: TelemetryConfig,
    pub runtime: RuntimeConfig,
    pub model: ModelConfig,
    pub skip_tracing: bool,
    pub max_concurrent_processes: Option<usize>,
    pub policy: Option<(String, Cow<'static, [u8]>)>,
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinProgram {
    pub name: &'static str,
    pub version: &'static str,
    pub component: &'static [u8],
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub worker_threads: usize,

    pub wasm_max_instances: u32,
    pub wasm_max_memory_mb: usize,
    pub wasm_warm_memory_mb: usize,
    pub wasm_warm_slots: u32,

    pub allow_fs: bool,
    pub fs_scratch_dir: PathBuf,

    pub allow_network: bool,
    pub network_allowed_hosts: Vec<String>,

    pub max_upload_mb: usize,
    /// Where the `<language>.wasm` language components live.
    pub languages_dir: PathBuf,
    /// Where wasmtime caches compiled code across boots; `None` compiles
    /// every component on every boot.
    pub compile_cache_dir: Option<PathBuf>,
}

pub struct ModelConfig {
    pub name: String,
    pub model_id: String,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub metadata: ModelMetadata,
    pub engines: Vec<EngineConfig>,
    pub scheduler: SchedulerConfig,
}

pub struct EngineConfig {
    pub total_pages: usize,
    pub window_pages: u32,
    pub window_tokens: u32,
    pub cpu_pages: usize,
    pub cpu_window_pages: u32,
    pub disk_pages: u32,
    /// The disk tier is kept for the next run (`disk_kv_persistent`).
    pub disk_kept: Option<KeptDisk>,
    pub kv_copy: ::engine::caps::KvCopyDomains,
    pub backend_kind: String,
    pub rs_cache_required: bool,
    pub rs_cache_slots: usize,
    /// The engine asks for one frame a lane when its slots seat fewer lanes
    /// than it batches (`PoolFacts::one_frame_when_short`).
    pub one_frame_when_short: bool,
    /// The engine asks the planner to serve a decoding lane's pages before a
    /// prompt waiting to be seated (`PoolFacts::seated_first`).
    pub seated_first: bool,
    pub rs_cache_slot_bytes: u64,
    pub rs_host_slots: usize,
    pub rs_buffer_pages: u32,
    pub rs_host_buffer_pages: u32,
    pub has_mtp_logits: bool,
    pub mtp_depth: u32,
    pub draft_block: u32,
    pub draft_mask_token: u32,
    pub draft_proposals_from: u32,
    pub has_value_head: bool,
    pub has_kv_envelopes: bool,
    pub has_attn_score: bool,
    pub has_attn_page_mask: bool,
    pub has_lora: bool,
    /// The engine takes an attachment for an adapter alone
    /// ([`::engine::fire::Boundary::Adapter`]): an adapted pass can be cut.
    pub adapter_attachments: bool,
    /// The rows inside a lane whose recurrent state one fire lands.
    pub rs_snapshots: u32,
    pub device_geometry_port_mask: eta_ir::registry::PortMask,
    pub limits: crate::engine::SchedulerLimits,
    /// The facts the engine's trace branches on.
    pub facts: poem_ir::Facts,
    pub engine_backend: crate::engine::EngineBox,
}

/// A disk tier kept across runs: where its journal is, and whether the
/// engine found its slot files as the last run left them.
#[derive(Debug, Clone)]
pub struct KeptDisk {
    pub journal: PathBuf,
    pub kept: bool,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub submit_deadline_us: u64,
    pub silence_timeout_secs: u64,
    pub frame_size: u32,
    pub frame_dispatch_depth: u32,
}

#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub service_name: String,
}

pub struct BootstrapHandle {
    pub port: u16,
    pub model_idx: usize,
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

#[cfg(not(target_arch = "wasm32"))]
pub async fn bootstrap_with_listener(
    config: Config,
    _listener: tokio::net::TcpListener,
) -> Result<BootstrapHandle> {
    bootstrap_inner(config).await
}

async fn bootstrap_inner(config: Config) -> Result<BootstrapHandle> {
    verify_config(&config)?;
    let mut active_guard = ActiveRuntimeGuard::acquire()?;

    if !config.skip_tracing {
        init_tracing(config.log_dir.as_deref(), config.verbose, &config.telemetry)?;
    }
    let wasm_engine = init_wasmtime(&config.runtime);

    program::spawn(
        &wasm_engine,
        config.cache_dir.clone(),
        config.runtime.languages_dir.clone(),
        &config.builtin_programs,
    )?;

    let fs_policy = FsPolicy {
        allow: config.runtime.allow_fs,
        base_dir: config.runtime.fs_scratch_dir.clone(),
    };
    let network_policy = NetworkPolicy::parse(
        config.runtime.allow_network,
        &config.runtime.network_allowed_hosts,
    )?;

    linker::spawn(
        &wasm_engine,
        fs_policy,
        network_policy,
        config
            .runtime
            .wasm_max_memory_mb
            .saturating_mul(1024 * 1024),
    );
    let max_upload_bytes = config.runtime.max_upload_mb.saturating_mul(1024 * 1024);
    server::init(max_upload_bytes);
    let bound_port = config.port;

    let ModelConfig {
        name,
        model_id,
        kv_page_size,
        tokenizer_path,
        metadata,
        engines: engine_configs,
        scheduler,
    } = config.model;

    crate::scheduler::set_dispatch_depth(scheduler.frame_dispatch_depth as usize);
    // A running recurrent sequence holds one state slot, so the pool seats
    // as many as it has slots; what it buffers unfolded is in pages of
    // their own.
    let rs_seat_cap = engine_configs
        .iter()
        .map(|d| d.rs_cache_slots)
        .filter(|&slots| slots > 0)
        .min();
    // Each posted frame buffers what it leaves unfolded: where every engine asks
    // and the slots seat fewer lanes than they batch at the configured depth, post
    // one frame a lane.
    let lanes = engine_configs
        .iter()
        .map(|d| d.limits.max_forward_requests)
        .min()
        .unwrap_or(0);
    let depth = crate::scheduler::configured_dispatch_depth().max(1);
    if let Some(slots) = rs_seat_cap
        && depth > 1
        && slots / depth < lanes
        && engine_configs
            .iter()
            .all(|d| d.rs_cache_slots == 0 || d.one_frame_when_short)
    {
        tracing::info!(
            lanes,
            state_slots = slots,
            seated_at_depth = slots / depth,
            seated_at_one = slots.min(lanes),
            "admission: the state slots seat fewer lanes than the engine batches; posting one \
             frame a lane, as the engine's profile asks"
        );
        crate::scheduler::set_dispatch_depth(1);
    }
    let admission_cap = config
        .max_concurrent_processes
        .or_else(|| {
            engine_configs
                .iter()
                .map(|d| d.limits.max_forward_requests)
                .min()
                .filter(|&r| r > 0)
        })
        .map(|cap| match rs_seat_cap {
            Some(seats) if cap > seats => {
                tracing::warn!(
                    requested = cap,
                    seated = seats,
                    "admission: more lanes than the state pool seats; capping, because each \
                     running sequence holds one recurrent-state slot. Raise `[engine] \
                     max_state_slots` to seat more"
                );
                seats
            }
            _ => cap,
        });
    process::init_admission(admission_cap);

    let rs_caps = {
        let d0 = engine_configs.first();
        let mut caps = model::RsCaps {
            state_size: d0.map(|d| d.rs_cache_slot_bytes).unwrap_or(0),
            buffer_page_size: 0,
            window_tokens: d0.map_or(0, |d| d.window_tokens),
        };
        if caps.has_state() {
            caps.buffer_page_size = kv_page_size as u32;
        }
        caps
    };
    let all = |has: fn(&EngineConfig) -> bool| {
        !engine_configs.is_empty() && engine_configs.iter().all(has)
    };
    let eta_caps = model::EtaCaps {
        has_lora: all(|d| d.has_lora),
        has_mtp_logits: all(|d| d.has_mtp_logits),
        mtp_depth: match engine_configs.first().map(|d| d.mtp_depth) {
            Some(depth) if engine_configs.iter().all(|d| d.mtp_depth == depth) => depth,
            _ => 0,
        },
        draft_block: match engine_configs.first().map(|d| d.draft_block) {
            Some(rows) if engine_configs.iter().all(|d| d.draft_block == rows) => rows,
            _ => 0,
        },
        draft_mask_token: engine_configs.first().map_or(0, |d| d.draft_mask_token),
        draft_proposals_from: engine_configs.first().map_or(1, |d| d.draft_proposals_from),
        has_value_head: all(|d| d.has_value_head),
        has_kv_envelopes: all(|d| d.has_kv_envelopes),
        has_attn_score: all(|d| d.has_attn_score),
        has_attn_page_mask: all(|d| d.has_attn_page_mask),
        adapter_attachments: all(|d| d.adapter_attachments),
        rs_snapshots: engine_configs
            .iter()
            .map(|d| d.rs_snapshots)
            .min()
            .unwrap_or(0),
    };
    let facts = engine_configs
        .first()
        .map(|engine| engine.facts.clone())
        .unwrap_or_default();
    if let Some(other) = engine_configs.iter().find(|engine| engine.facts != facts) {
        anyhow::bail!(
            "the engines of one model classify requests by different facts ({:?} and {:?}); \
             they run different traces",
            facts.table,
            other.facts.table
        );
    }
    model::register(
        name.clone(),
        &model_id,
        facts,
        kv_page_size as u32,
        rs_caps,
        eta_caps,
        tokenizer_path.clone(),
        &metadata,
    )?;

    let arena_kv_pages: Vec<usize> = engine_configs.iter().map(|d| d.total_pages).collect();
    let arena_cpu_pages: Vec<usize> = engine_configs.iter().map(|d| d.cpu_pages).collect();
    let arena_disk_pages: Vec<u32> = engine_configs.iter().map(|d| d.disk_pages).collect();
    let arena_disk_kept: Vec<Option<KeptDisk>> =
        engine_configs.iter().map(|d| d.disk_kept.clone()).collect();
    let arena_state: Vec<crate::store::registry::StatePool> = engine_configs
        .iter()
        .map(|d| crate::store::registry::StatePool {
            slots: d.rs_cache_slots as u32,
            host_slots: d.rs_host_slots as u32,
            pages: d.rs_buffer_pages,
            host_pages: d.rs_host_buffer_pages,
            window_tokens: d.window_tokens,
            window_pages: d.window_pages,
            host_window_pages: d.cpu_window_pages,
        })
        .collect();
    let arena_max_context: Vec<usize> = engine_configs
        .iter()
        .map(|d| d.limits.max_context)
        .collect();
    let kv_swap_capable = engine_configs
        .first()
        .is_some_and(|d| d.cpu_pages > 0 || d.disk_pages > 0);
    let seated_first = !engine_configs.is_empty() && engine_configs.iter().all(|d| d.seated_first);
    if let Some(d) = engine_configs.first() {
        if kv_swap_capable {
            tracing::info!(
                device_pages = d.total_pages,
                host_pages = d.cpu_pages,
                disk_pages = d.disk_pages,
                "kv tiers: suspended kv goes to the host, then the disk"
            );
        } else {
            tracing::warn!(
                device_pages = d.total_pages,
                "no host or disk kv tier: pie cannot compute kv again, so kv \
                 pressure restarts processes"
            );
        }
    }
    let engine_count = engine_configs.len();
    let engines: Vec<usize> = engine_configs
        .into_iter()
        .map(|d| {
            engine::register_engine_backend(
                engine::EngineSpec {
                    device_domain: ::engine::MemoryDomain::HostPinned,
                    num_kv_pages: d.total_pages,
                    limits: d.limits,
                    device_geometry_port_mask: d.device_geometry_port_mask,
                },
                d.engine_backend,
            )
        })
        .collect();

    let _ = engine_count;
    let arena_model_idx = crate::store::registry::register_model_with_swap(
        kv_page_size as u32,
        &arena_kv_pages,
        &arena_cpu_pages,
        &arena_disk_pages,
        &arena_state,
        &arena_max_context,
    );
    for (d, kept) in arena_disk_kept.iter().enumerate() {
        let Some(kept) = kept else { continue };
        let stores = crate::store::registry::get(arena_model_idx, d);
        let taken = crate::store::registry::with_kv_lock(&stores.kv, "keep-disk", |kv| {
            kv.keep_disk(&kept.journal, kept.kept)
        });
        if let Err(error) = taken {
            tracing::warn!(%error, journal = %kept.journal.display(), "disk kv tier not kept");
        }
    }

    crate::planner::init_planner(
        arena_model_idx,
        0,
        crate::planner::ResidencyPlanner::new(std::sync::Arc::new(
            crate::planner::RegistryPool::new(arena_model_idx, 0, kv_swap_capable),
        ))
        .seated_first(seated_first),
    );
    // After the gates and the planner it orders exist.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some((name, component)) = &config.policy {
        crate::policy::start(name, component).with_context(|| format!("worker policy {name}"))?;
    }
    if let Some(period) = std::env::var("PIE_CONTENTION_TRACE_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        && let Some(planner) = crate::planner::planner_for(arena_model_idx, 0)
    {
        crate::rt::spawn(async move {
            let mut interval = crate::rt::time::interval(std::time::Duration::from_millis(period));
            loop {
                interval.tick().await;
                let d = planner.diagnostics();
                let (lk_n, lk_wait, lk_hold, lk_wmax, lk_hmax) = planner.lock_census();
                if d.queue.is_empty() && d.proc_states[1..].iter().all(|&n| n == 0) {
                    continue;
                }
                println!(
                    "[planner-trace] queue={} unmet={} head_pages={} head_kind={} bypass={}/{} accum={} \
                     free={}/{} host_free={}/{} head_rs={} rs_free={}/{} \
                     parks={} serves={} evictions={} deferrals={} \
                     evict_rollbacks={} restores={} restore_failures={} gate_parks={} \
                     hogs={} starved={} restarted={} salvaged={} swapfull={}/{} e6_relax={} rshort={} \
                     lock_n={} lock_wait_ms={} lock_hold_ms={} lock_wmax_us={} lock_hmax_us={} \
                     d2h_pages={} h2d_pages={} d2h_rs={} h2d_rs={} d2h_ms={} h2d_ms={} \
                     disk_free={}/{} disk_w_pages={} disk_r_pages={} disk_w_ms={} disk_r_ms={} \
                     resident={} evicting={} evicted={} restoring={} admitted={} \
                     runners=[{}]",
                    d.queue.len(),
                    d.unmet_queued,
                    d.unmet_head_pages,
                    d.unmet_head_kind,
                    d.bypassable_entries,
                    d.bypassable_pages,
                    d.accumulation,
                    d.device_pages_free,
                    d.device_pages_total,
                    d.host_slots_free,
                    d.host_slots_total,
                    d.queue.first().map_or(0, |w| w.rs.slots),
                    d.rs_free.slots,
                    d.rs_total.slots,
                    d.parks_total,
                    d.serves_total,
                    d.evictions_total,
                    d.eviction_deferrals_total,
                    d.eviction_rollbacks_total,
                    d.restores_total,
                    d.restore_failures_total,
                    d.gate_parks_total,
                    d.hog_failures_total,
                    d.starvations_total,
                    d.starvation_restarts_total,
                    d.salvages_total,
                    d.host_swap_exhaustions_total,
                    d.host_swap_unblocks_total,
                    d.e6_relaxations_total,
                    d.restore_absorb_short_total,
                    lk_n,
                    lk_wait / 1000,
                    lk_hold / 1000,
                    lk_wmax,
                    lk_hmax,
                    d.d2h_pages_total,
                    d.h2d_pages_total,
                    d.d2h_rs_slots_total,
                    d.h2d_rs_slots_total,
                    d.d2h_copy_us_total / 1000,
                    d.h2d_copy_us_total / 1000,
                    d.disk_slots_free,
                    d.disk_slots_total,
                    d.disk_write_pages_total,
                    d.disk_read_pages_total,
                    d.disk_write_us_total / 1000,
                    d.disk_read_us_total / 1000,
                    d.proc_states[0],
                    d.proc_states[1],
                    d.proc_states[2],
                    d.proc_states[3],
                    d.admitted_procs,
                    d.runners
                        .iter()
                        .map(|(seq, held, progressed)| format!("{seq}:h{held}:p{progressed}"))
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
        });
    }

    crate::scheduler::set_submit_deadline(std::time::Duration::from_micros(
        scheduler.submit_deadline_us,
    ));
    crate::scheduler::set_silence_timeout(std::time::Duration::from_secs(
        scheduler.silence_timeout_secs,
    ));
    crate::scheduler::set_frame_size(scheduler.frame_size as usize);
    crate::scheduler::set_dispatch_depth(scheduler.frame_dispatch_depth as usize);
    let scheduler_shutdown = crate::scheduler::spawn(&engines, kv_page_size as u32).await?;
    active_guard.disarm();
    Ok(BootstrapHandle {
        port: bound_port,
        model_idx: arena_model_idx,
        shutdown: Some(RuntimeShutdown {
            scheduler: scheduler_shutdown,
            engine_ids: engines,
        }),
    })
}

fn verify_config(config: &Config) -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    fs::create_dir_all(&config.cache_dir)
        .with_context(|| format!("Could not create cache dir: {:?}", config.cache_dir))?;

    let model = &config.model;
    ensure!(
        model.metadata.tokenizer.is_some() || model.tokenizer_path.exists(),
        "Model {:?}: tokenizer not found at {:?}",
        model.name,
        model.tokenizer_path
    );
    for (i, dev) in model.engines.iter().enumerate() {
        ensure!(
            dev.total_pages > 0,
            "Model {:?} engine {i}: total_pages must be > 0",
            model.name
        );
        ensure!(
            dev.limits.max_forward_tokens > 0,
            "Model {:?} engine {i}: max_forward_tokens must be > 0",
            model.name
        );
        ensure!(
            dev.limits.max_forward_requests > 0,
            "Model {:?} engine {i}: max_forward_requests must be > 0",
            model.name
        );
        ensure!(
            dev.limits.max_page_refs > 0,
            "Model {:?} engine {i}: max_page_refs must be > 0",
            model.name
        );
    }
    Ok(())
}

#[cfg(not(any(
    target_arch = "wasm32",
    all(target_vendor = "apple", not(target_os = "macos"))
)))]
const CORE_RESOURCES_PER_COMPONENT: u32 = 16;

#[cfg(target_arch = "wasm32")]
fn init_wasmtime(_runtime: &RuntimeConfig) -> wasmtime::Engine {
    browser_wasm_engine()
}

/// The engine inferlets run on in a browser tab. A component precompiled for
/// it (`program::precompile_component`) loads only into an engine built the
/// same way, so precompiling goes through here too.
#[cfg(target_arch = "wasm32")]
pub fn browser_wasm_engine() -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();
    wasmtime_web::configure(&mut wasm_config).expect("configure wasmtime for the browser");
    wasm_config.wasm_component_model_async(true);
    wasmtime::Engine::new(&wasm_config).unwrap()
}

#[cfg(not(target_arch = "wasm32"))]
fn init_wasmtime(runtime: &RuntimeConfig) -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();

    if let Some(dir) = &runtime.compile_cache_dir {
        let mut cache_config = wasmtime::CacheConfig::new();
        cache_config.with_directory(dir);
        match wasmtime::Cache::new(cache_config) {
            Ok(cache) => {
                wasm_config.cache(Some(cache));
            }
            Err(error) => tracing::warn!(
                "compile cache at {} unavailable, every component compiles on this boot: {error:#}",
                dir.display()
            ),
        }
    }

    configure_execution(&mut wasm_config, runtime);
    wasmtime::Engine::new(&wasm_config).unwrap()
}

/// Apple's platforms other than macOS (iOS, visionOS, tvOS, watchOS) give an
/// app no executable memory: inferlets run as Pulley bytecode, with memories
/// allocated on demand.
#[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
fn configure_execution(wasm_config: &mut wasmtime::Config, _runtime: &RuntimeConfig) {
    wasm_config
        .target("pulley64")
        .expect("pulley64 is compiled in where an app may not JIT");
    wasm_config.signals_based_traps(false);
    wasm_config.memory_reservation(0);
    wasm_config.memory_guard_size(0);
    wasm_config.memory_init_cow(false);
}

#[cfg(not(any(
    target_arch = "wasm32",
    all(target_vendor = "apple", not(target_os = "macos"))
)))]
fn configure_execution(wasm_config: &mut wasmtime::Config, runtime: &RuntimeConfig) {
    let mut pooling_config = wasmtime::PoolingAllocationConfig::default();
    pooling_config.total_component_instances(runtime.wasm_max_instances);
    pooling_config.total_memories(runtime.wasm_max_instances);
    pooling_config.total_stacks(runtime.wasm_max_instances);
    pooling_config.max_core_instances_per_component(CORE_RESOURCES_PER_COMPONENT);
    pooling_config.max_tables_per_component(CORE_RESOURCES_PER_COMPONENT);
    pooling_config.total_core_instances(
        runtime
            .wasm_max_instances
            .saturating_mul(CORE_RESOURCES_PER_COMPONENT),
    );
    pooling_config.total_tables(
        runtime
            .wasm_max_instances
            .saturating_mul(CORE_RESOURCES_PER_COMPONENT),
    );
    pooling_config.max_memory_size(runtime.wasm_max_memory_mb.saturating_mul(1024 * 1024));
    pooling_config
        .linear_memory_keep_resident(runtime.wasm_warm_memory_mb.saturating_mul(1024 * 1024));
    pooling_config.max_unused_warm_slots(runtime.wasm_warm_slots);

    wasm_config.allocation_strategy(wasmtime::InstanceAllocationStrategy::Pooling(
        pooling_config,
    ));
}

#[cfg(target_arch = "wasm32")]
fn init_tracing(
    _log_dir: Option<&Path>,
    _verbose: bool,
    _telemetry_config: &TelemetryConfig,
) -> Result<()> {
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn init_tracing(
    log_dir: Option<&Path>,
    verbose: bool,
    telemetry_config: &TelemetryConfig,
) -> Result<()> {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;

    let default_level = if verbose { "debug" } else { "info" };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

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
    let otel_layer = if telemetry_config.enabled {
        telemetry::init_otel_layer(&telemetry_config.endpoint, &telemetry_config.service_name)
    } else {
        None
    };

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
