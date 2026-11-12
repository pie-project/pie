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
    /// Where snapshots of handles are kept, and how many bytes of them.
    pub snapshots: Option<(PathBuf, u64)>,
    /// The built-in inferlets: the system's programs, each under whatever
    /// the operator installs of its name in `cache_dir`.
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
    /// Whether a client may install its own programs; without, the
    /// system's alone run.
    pub allow_upload: bool,
    /// Where the `<language>.wasm` language components live.
    pub languages_dir: PathBuf,
    /// Where wasmtime caches compiled code across boots; `None` compiles
    /// every component on every boot.
    pub compile_cache_dir: Option<PathBuf>,
}

pub struct ModelConfig {
    /// How much congestion the system may show before this gives memory back.
    pub memory_tolerance: crate::governor::Tolerance,
    /// Whether streamable weights may be evicted under pressure; resolved
    /// per engine against [`EngineConfig::weights_fit`], and again against
    /// the pool the weight shapes take: weights it cannot seat stream under
    /// `auto` and fail the load under `off`.
    pub weight_streaming: crate::store::weights::WeightStreaming,
    pub name: String,
    pub model_id: String,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub metadata: ModelMetadata,
    pub engines: Vec<EngineConfig>,
    pub scheduler: SchedulerConfig,
}

pub struct EngineConfig {
    /// The model's weights fit the device budget at load
    /// (`LoadFacts::weights_resident`): what `weight_streaming = auto`
    /// reads.
    pub weights_fit: bool,
    pub total_pages: usize,
    /// The pools as chunks: what the engine states
    /// ([`::engine::Capabilities::chunks`]).
    pub chunks: ::engine::chunk::ChunkFacts,
    pub window_tokens: u32,
    pub cpu_pages: usize,
    pub disk_pages: u32,
    /// The disk tier is kept for the next run (`disk_kv_persistent`).
    pub disk_kept: Option<KeptDisk>,
    pub rs_cache_slots: usize,
    /// The engine asks the planner to serve a decoding lane's pages before a
    /// prompt waiting to be seated (`PoolFacts::seated_first`).
    pub seated_first: bool,
    /// The engine asks the planner to start a process only while the pools
    /// hold the started ones to the end (`PoolFacts::seat_to_end`).
    pub seat_to_end: bool,
    /// The engine asks the planner to spread the starts the pools do not hold
    /// to the end at once over a life (`PoolFacts::seat_spread`).
    pub seat_spread: bool,
    pub rs_cache_slot_bytes: u64,
    pub draft_block: u32,
    pub draft_mask_token: u32,
    pub draft_proposals_from: u32,
    pub has_attn_page_mask: bool,
    pub has_lora: bool,
    /// The engine takes an attachment for an adapter alone
    /// ([`::engine::fire::Attachment::adapter_only`]): an adapted pass can be cut.
    pub adapter_attachments: bool,
    /// The rows inside a lane whose recurrent state one fire lands.
    pub rs_snapshots: u32,
    pub device_geometry_port_mask: eta_ir::registry::PortMask,
    pub limits: crate::engine::SchedulerLimits,
    /// The facts the engine's trace branches on.
    pub facts: poem_ir::Facts,
    pub inputs: Vec<poem_ir::InputDecl>,
    pub classes: poem_ir::LaneClasses,
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
    server::init(max_upload_bytes, config.runtime.allow_upload);
    let bound_port = config.port;

    let ModelConfig {
        memory_tolerance,
        weight_streaming,
        name,
        model_id,
        kv_page_size,
        tokenizer_path,
        metadata,
        engines: engine_configs,
        scheduler,
    } = config.model;

    crate::scheduler::set_dispatch_depth(scheduler.frame_dispatch_depth as usize);
    // A model with fixed recurrent state is admitted by its planner, which
    // charges each process the fixed set its working set holds against the
    // state pool, beside a copy-on-write set per lane a fire runs
    // (`ResidencyPlanner::admit`): no cap here.
    let has_state = engine_configs.iter().any(|d| d.rs_cache_slots > 0);
    let admission_cap = config
        .max_concurrent_processes
        .or_else(|| {
            engine_configs
                .iter()
                .map(|d| d.limits.max_forward_requests)
                .min()
                .filter(|&r| r > 0)
        })
        .map(|cap| {
            // A seated-first pool of kv pages alone seats no more than five
            // sequences a sequence at the declared context holds: past that,
            // admitted prompts seat lanes the pool then suspends for each
            // other's growth (Gemma-4-26B on an A100, c512: 2.9k tok/s
            // admitting all 512, 4.7k at 224; its 5791 pages seat 45 at 2048).
            // A pool that spreads its starts holds them to what it seats itself.
            let seats = engine_configs
                .iter()
                .filter(|d| d.seated_first && !d.seat_spread && d.limits.max_context > 0)
                .map(|d| 5 * d.total_pages * kv_page_size / d.limits.max_context)
                .min();
            match seats {
                Some(seats) if !has_state && cap > seats.max(1) => {
                    tracing::info!(
                        requested = cap,
                        seated = seats,
                        "admission: capped at five sequences a sequence at the declared context \
                         holds in the kv pool"
                    );
                    seats.max(1)
                }
                _ => cap,
            }
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
        draft_block: match engine_configs.first().map(|d| d.draft_block) {
            Some(rows) if engine_configs.iter().all(|d| d.draft_block == rows) => rows,
            _ => 0,
        },
        draft_mask_token: engine_configs.first().map_or(0, |d| d.draft_mask_token),
        draft_proposals_from: engine_configs.first().map_or(1, |d| d.draft_proposals_from),
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
    let inputs = engine_configs
        .first()
        .map(|engine| engine.inputs.clone())
        .unwrap_or_default();
    if engine_configs.iter().any(|engine| engine.inputs != inputs) {
        anyhow::bail!(
            "the engines of one model read different guest inputs; they run different traces"
        );
    }
    let classes = engine_configs
        .first()
        .map(|engine| engine.classes.clone())
        .unwrap_or_default();
    if engine_configs
        .iter()
        .any(|engine| engine.classes != classes)
    {
        anyhow::bail!("the engines of one model run different classes; they run different traces");
    }
    model::register(
        name.clone(),
        &model_id,
        facts,
        inputs,
        classes,
        kv_page_size as u32,
        rs_caps,
        eta_caps,
        tokenizer_path.clone(),
        &metadata,
    )?;

    let scales: Vec<crate::policy::pressure::Scale> = engine_configs
        .iter()
        .map(|d| {
            crate::store::shared::SharedPool::scale_of(&d.chunks).unwrap_or(
                crate::policy::pressure::Scale {
                    kv_page_bytes: 0,
                    state_slot_bytes: d.rs_cache_slot_bytes,
                },
            )
        })
        .collect();
    // Each engine's weight shapes and the chunks they may take.
    let chunk_facts: Vec<::engine::chunk::ChunkFacts> =
        engine_configs.iter().map(|d| d.chunks.clone()).collect();
    let arena_disk_kept: Vec<Option<KeptDisk>> =
        engine_configs.iter().map(|d| d.disk_kept.clone()).collect();
    // Each engine's pools as chunks: what it states, else today's pools as
    // the degenerate mode's pinned shapes.
    let arena_stores = engine_configs
        .iter()
        .map(|d| {
            let mut pools = crate::store::chunks::StorePools::of(&d.chunks)
                .map_err(|error| anyhow::anyhow!("engine pools: {error}"))?;
            // A window the engine holds no pages for is still a window.
            if pools.window_tokens == 0 {
                pools.window_tokens = d.window_tokens;
            }
            // An engine that states a shared pool: the stores draw from one
            // pool built of its facts, a copy-on-write record kept back for
            // each lane a fire runs.
            let shared = if pools.shared {
                anyhow::ensure!(
                    pools.page_size == 0 || pools.page_size as usize == kv_page_size,
                    "engine pools: the shared pool's slots hold {} tokens, the kv pages {kv_page_size}",
                    pools.page_size
                );
                let pool = crate::store::shared::SharedPool::new(
                    &d.chunks,
                    u32::try_from(d.limits.max_context).unwrap_or(u32::MAX),
                )
                .map_err(|error| anyhow::anyhow!("engine pools: {error}"))?;
                Some(std::sync::Arc::new(std::sync::Mutex::new(pool)))
            } else {
                None
            };
            // Below a shared pool: the host chunks and the kv list's disk
            // slots the engine states (CONTRACT-BC 22).
            let tiers = if pools.shared {
                crate::store::shared::Tiers::of(&d.chunks)
                    .map_err(|error| anyhow::anyhow!("engine tiers: {error}"))?
            } else {
                crate::store::shared::Tiers::default()
            };
            Ok(crate::store::registry::EngineStores {
                pools,
                max_context: d.limits.max_context,
                shared,
                tiers,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // A shared pool's tiers are its host chunks and disk slots; today's
    // pools state theirs in pages.
    let kv_swap_capable = engine_configs.first().is_some_and(|d| {
        if crate::store::shared::SharedPool::states_shared(&d.chunks) {
            d.chunks.host_chunks > 0 || d.chunks.disk_chunks > 0
        } else {
            d.cpu_pages > 0 || d.disk_pages > 0
        }
    });
    // The lanes one fire runs on each engine: the copy-on-write sets its
    // state pool keeps back from admission.
    let fire_lanes: Vec<u32> = engine_configs
        .iter()
        .map(|d| d.limits.max_forward_requests as u32)
        .collect();
    let seated_first = !engine_configs.is_empty() && engine_configs.iter().all(|d| d.seated_first);
    let seat_to_end = !engine_configs.is_empty() && engine_configs.iter().all(|d| d.seat_to_end);
    let seat_spread = seat_to_end && engine_configs.iter().all(|d| d.seat_spread);
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
            // Weights the engine states as shapes start off the device (the
            // engine reads them from host until the starting residency), so
            // its `weights_fit` says nothing of them: `auto` holds them
            // unless the pool cannot seat them all, which
            // `store::weights::boot_cache` decides.
            let streams = match weight_streaming {
                crate::store::weights::WeightStreaming::Auto if !d.chunks.weights.is_empty() => {
                    false
                }
                mode => mode.streams(d.weights_fit),
            };
            let id = engine::register_engine_backend(
                engine::EngineSpec {
                    device_domain: ::engine::MemoryDomain::HostPinned,
                    limits: d.limits,
                    device_geometry_port_mask: d.device_geometry_port_mask,
                },
                d.engine_backend,
            );
            crate::store::weights::set_streaming(id, streams);
            tracing::info!(
                engine = id,
                mode = ?weight_streaming,
                weights_fit = d.weights_fit,
                streams,
                "weight streaming"
            );
            id
        })
        .collect();

    let _ = engine_count;
    let arena_model_idx =
        crate::store::registry::register_model_with_chunks(kv_page_size as u32, &arena_stores);
    // The worker policy reads tier bytes off the stores' slot pools.
    crate::policy::pressure::install_meter(Some(std::sync::Arc::new(
        crate::store::chunks::SlotMeter,
    )));
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

    // A planner for each engine: each holds its own kv pages, and a process
    // runs on one of them.
    let mut shared_grows: Vec<crate::store::shared::SharedGrow> = Vec::new();
    let mut governed = Vec::new();
    for engine in 0..engines.len().max(1) {
        // What the worker policy reads tier bytes at; a page of unknown size
        // leaves them unread.
        crate::policy::pressure::scale(engine, scales.get(engine).copied().unwrap_or_default());
        let mut pool = crate::planner::RegistryPool::new(arena_model_idx, engine, kv_swap_capable);
        let shared = arena_stores
            .get(engine)
            .and_then(|stores| stores.shared.clone());
        let steered = match &shared {
            Some(shared) => {
                let grow =
                    crate::store::shared::SharedGrow::new(arena_model_idx, engine, shared.clone());
                shared_grows.push(grow.clone());
                pool = pool.with_shared_grow(grow.clone());
                shared_steered_pool(
                    arena_model_idx,
                    engine,
                    weight_streaming,
                    chunk_facts.get(engine),
                    grow,
                )?
            }
            None => steered_pool(
                arena_model_idx,
                engine,
                weight_streaming,
                chunk_facts.get(engine),
            )?,
        };
        crate::planner::init_planner(
            arena_model_idx,
            engine,
            crate::planner::ResidencyPlanner::new(std::sync::Arc::new(pool))
                .seated_first(seated_first)
                .cow_lanes(fire_lanes.get(engine).copied().unwrap_or(1))
                .seat_to_end(seat_to_end)
                .seat_spread(seat_spread)
                .weight_shed(steered.shed_hook()),
        );
        governed.push(steered);
    }
    // After the gates and the planner it orders exist.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some((name, component)) = &config.policy {
        crate::policy::start(name, component).with_context(|| format!("worker policy {name}"))?;
    } else {
        // No policy: the thread still keeps its record, and the default
        // responder answers tier pressure.
        crate::policy::start_default().context("the default worker policy thread")?;
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
                     disk_free={}/{} disk_w_pages={} host_disk_pages={} disk_r_pages={} disk_w_ms={} disk_r_ms={} \
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
                    d.host_disk_pages_total,
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
    // Each engine takes the weight residency it starts with before the load
    // returns, so before any fire is submitted: none reads from the host a
    // row the cache holds. An engine that streams and refuses it fails the
    // load: its fires would read every row from the host.
    for (engine, steered) in governed.iter().enumerate() {
        if let Err(error) = steered.boot().await {
            scheduler_shutdown.shutdown().await.ok();
            return Err(error.context(format!(
                "engine {engine} refused the weight residency it starts with"
            )));
        }
    }
    for grow in &shared_grows {
        grow.start();
    }
    for steered in governed {
        crate::governor::run::start(steered, memory_tolerance);
    }
    active_guard.disarm();
    Ok(BootstrapHandle {
        port: bound_port,
        shutdown: Some(RuntimeShutdown {
            scheduler: scheduler_shutdown,
            engine_ids: engines,
        }),
    })
}

/// Engine `engine`'s governor target where its pools are pinned (they never
/// change size), with its weight cache in front ([`boot_cache`]: weights
/// the pool cannot seat stream under `auto` and fail an `off` load).
///
/// [`boot_cache`]: crate::store::weights::boot_cache
fn steered_pool(
    model: usize,
    engine: usize,
    mode: crate::store::weights::WeightStreaming,
    facts: Option<&::engine::chunk::ChunkFacts>,
) -> Result<crate::governor::steered::Shared<crate::governor::steered::Steered>> {
    use crate::store::weights::{WeightFirst, boot_cache, port::EnginePort, slot_pool};
    let none = ::engine::chunk::ChunkFacts::default();
    let facts = facts.unwrap_or(&none);
    let mut slots = slot_pool(facts);
    let cache = boot_cache(engine, mode, &facts.weights, 0, &mut slots)?;
    let quiet = cache.shapes() == 0;
    let pool = crate::governor::steered::EnginePool::new(model, engine, slots);
    Ok(crate::governor::steered::Shared::new(WeightFirst::new(
        pool,
        cache,
        EnginePort::new(engine, quiet),
    )))
}

/// Engine `engine`'s governor target where its stores draw from a shared
/// pool: `grow` moves the pool's segments, and the weight shapes take their
/// slots from the same pool, after the state shapes.
fn shared_steered_pool(
    model: usize,
    engine: usize,
    mode: crate::store::weights::WeightStreaming,
    facts: Option<&::engine::chunk::ChunkFacts>,
    grow: crate::store::shared::SharedGrow,
) -> Result<crate::governor::steered::Shared<crate::governor::steered::Steered>> {
    use crate::store::weights::{WeightFirst, boot_cache, port::EnginePort};
    let none = ::engine::chunk::ChunkFacts::default();
    let facts = facts.unwrap_or(&none);
    let cache = {
        let mut shared = grow.shared().lock().unwrap();
        let first = shared.weights_first();
        boot_cache(engine, mode, &facts.weights, first, shared.pool_mut())?
    };
    grow.hold_mapped();
    let quiet = cache.shapes() == 0;
    let pool = crate::governor::steered::EnginePool::shared(model, engine, grow);
    Ok(crate::governor::steered::Shared::new(WeightFirst::new(
        pool,
        cache,
        EnginePort::new(engine, quiet),
    )))
}

fn verify_config(config: &Config) -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some((dir, budget)) = &config.snapshots {
        crate::handle::snapshot::configure(dir.clone(), *budget);
    }
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
