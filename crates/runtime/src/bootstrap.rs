use anyhow::{Context, Result, ensure};

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::engine;
use crate::inferlet::sandbox::{FsPolicy, NetworkPolicy};
use crate::inferlet::{linker, process, program, python};
use crate::model::{self, ModelMetadata};
use crate::server;
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
    /// Disable via `python_snapshot = false` in the runtime config or the
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
    /// Catalog id the engine loaded, as it reported it.
    ///
    /// The only model fact this bundle carries, and every other one is read
    /// off the row it names: the chat template, the two numbers, and the
    /// family label the vision and speech front-ends dispatch on. `arch_name`
    /// STOOD beside it, copied off the engine's capabilities, which left the
    /// sampler following the row while the processor followed the engine.
    /// Empty from an engine not yet on the catalog, and `register` says so
    /// rather than guessing a template.
    pub model_id: String,
    pub kv_page_size: usize,
    /// The tokenizer file, for a model served from a HuggingFace snapshot.
    ///
    /// Only consulted when `metadata.tokenizer` is `None`. A served `.zt`
    /// carries its tokenizer compiled, so there is no file to point at — this
    /// then holds the artifact's own path, which is what the diagnostics want
    /// anyway.
    pub tokenizer_path: PathBuf,
    /// The served model's compiled metadata, lifted once by the worker.
    ///
    /// Not optional: the descriptor inside it is where the runtime's model
    /// facts come from, for a `.zt` and for a snapshot alike. The runtime used
    /// to probe `config.json` itself when this was absent — two hand-written
    /// key walks that had to agree with the engine's parser by coincidence.
    pub metadata: ModelMetadata,
    pub engines: Vec<EngineConfig>,
    pub scheduler: SchedulerConfig,
}

pub struct EngineConfig {
    pub total_pages: usize,
    pub cpu_pages: usize,
    /// Which `copy_kv` directions this engine serves.
    ///
    /// Was `kv_copy_domain_mask: u32` over four `KV_COPY_*` bit constants;
    /// four named booleans is what four bits with four names are, and a
    /// caller reads the one it is about (`engine::caps`).
    pub kv_copy: ::engine::caps::KvCopyDomains,
    pub backend_kind: String,
    pub rs_cache_required: bool,
    pub rs_cache_slots: usize,
    pub rs_cache_slot_bytes: u64,
    // `elastic_page_bytes` and `elastic_budget_pages` stood here and were
    // WRITE-ONLY: `translate.rs` copied them off `PoolFacts` and nothing in
    // this crate ever read either one. Wave C gave the elastic supply its own
    // reader — `Supply::trim` asks the engine, and `PoolFacts` is where a gate
    // reads the numbers back — so a second copy in the scheduler's config was
    // a field that could only go stale. Deleted rather than wired: the facts
    // are one call away from every caller that wants them (alto wave P).
    pub has_mtp_logits: bool,
    pub has_mtp_drafts: bool,
    pub has_value_head: bool,
    pub has_kv_envelopes: bool,
    pub has_attn_score: bool,
    pub has_attn_page_mask: bool,
    pub has_lora: bool,
    /// Which descriptor ports it resolves on the device, in the port
    /// registry's own numbering (decision 19).
    pub device_geometry_port_mask: eta_ir::registry::PortMask,
    pub limits: crate::engine::SchedulerLimits,
    pub engine_backend: crate::engine::EngineBox,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Wall-clock cap on a single forward-pass request, in seconds.
    pub request_timeout_secs: u64,
    /// How long a lane holding the frame wait-set may go without submitting
    /// before the leash drops it from the wait-set. Not a verdict. See
    /// `crate::scheduler::configured_submit_deadline`.
    pub submit_deadline_us: u64,
    /// How long a lane may stay silent in total before its process is
    /// terminated. See `crate::scheduler::configured_silence_timeout`.
    pub silence_timeout_secs: u64,
    /// Waves per frame (k). See `crate::scheduler::configured_frame_size`.
    pub frame_size: u32,
    /// Frames the runtime keeps posted to the engine. See
    /// `crate::scheduler::frame::configured_dispatch_depth`.
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

    let ModelConfig {
        name,
        model_id,
        kv_page_size,
        tokenizer_path,
        metadata,
        engines: engine_configs,
        scheduler,
    } = config.model;

    // Admission defaults to the engine's `max_forward_requests` (R), not to
    // "unlimited". A forward can carry at most R rows and at most one fire
    // per process, so admitting more than R processes cannot widen a batch —
    // it only makes the set that happens to be READY at seal time a random
    // subset of a larger pool, and batches come out ragged. Measured on
    // Qwen3.6-35B-A3B (R=64, 128 requests): uncapped gave 775 batches with
    // rows spread across 16-64, capped at R gave 530 batches with 510 of
    // them full. Throughput +31%, and wall, mean and p99 latency ALL
    // improved -- oversubscribing R is strictly worse, not a trade.
    // An explicit operator setting always wins.
    //
    // Deliberately NOT also clamped to the engine's RS folded-slot count.
    // That looks right -- a recurrent process holds a folded slot for its
    // whole life, so the pool does bound residency -- but the pool serves
    // folded state AND buffer pages from the same slots, so admitting exactly
    // `rs_cache_slots` processes seats every one of them and leaves nothing to
    // buffer with. Measured on Qwen3.6-27B (24 slots): capping admission at 24
    // turned a loud failure into a 300 s hang, because every process is
    // legitimately running and the planner is right to keep waiting.
    //
    // The divisor that bound is missing is `frame_dispatch_depth`: a lane
    // keeps that many frames posted to the engine at once, and each posted
    // frame holds a slot, so one admitted seat costs D slots, not one. When
    // `lanes * D` exceeds the pool every lane ends up holding its first slot
    // and waiting for a second that no one can return -- a resource deadlock,
    // which the planner reports as `StarveCause::NoRsSlots` and fails rather
    // than waits out, because it is right that it cannot be waited out.
    // Measured on Qwen3.6-27B (24 slots, D=2, 32 requests of 64 tokens):
    // c=12 completed 32/32 at 28.8 tok/s, c=16 failed 18 of 32 and delivered
    // 0.10 tok/s, and c=16 with D=1 -- the same `lanes * D` -- completed
    // 32/32 again. So the product is the bound, and both factors are known
    // here.
    //
    // This clamp applies to an explicit operator setting too, unlike every
    // other default in this block. Admission is documented below as "a
    // physical safety cap only", and a seat count that cannot physically be
    // seated is not a preference to honour -- it is a request failure with
    // extra steps.
    crate::scheduler::set_dispatch_depth(scheduler.frame_dispatch_depth as usize);
    let seat_cost = crate::scheduler::configured_dispatch_depth().max(1);
    // The binding engine's seat pool, kept WITH its page pool: the warning
    // below is the only notice an operator gets that their concurrency was
    // overruled, and "seated=32" alone does not say by what. The seat count is
    // not a knob — `Budgets::slots` is the page pool divided by one
    // FULL-CONTEXT block per seat (`Paging::pages_per_slot`) — so the two
    // numbers that produced it are the two an operator can act on.
    let rs_pool = engine_configs
        .iter()
        .filter(|d| d.rs_cache_slots > 0)
        .min_by_key(|d| d.rs_cache_slots)
        .map(|d| (d.rs_cache_slots, d.total_pages));
    let rs_seat_cap = rs_pool.map(|(slots, _)| (slots / seat_cost).max(1));
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
                let (slots, pages) = rs_pool.unwrap_or((0, 0));
                tracing::warn!(
                    requested = cap,
                    seated = seats,
                    seat_cost,
                    pool_seats = slots,
                    pool_pages = pages,
                    pages_per_seat = pages.checked_div(slots).unwrap_or(0),
                    "admission: more lanes than the pools can seat; capping, because each \
                     lane holds one seat per posted frame. THE SEAT COUNT IS THE PAGE POOL \
                     DIVIDED BY ONE FULL-CONTEXT BLOCK PER SEAT, so it falls when the KV \
                     page pool is capped: to seat `requested * seat_cost` sequences, raise \
                     the pool (`[model.engine.options] max_total_pages`, or leave it unset \
                     to derive it from gpu_mem_utilization) or lower the context ceiling. \
                     Every batch this deployment fires is bounded by `seated`, not by the \
                     engine's max_lanes"
                );
                seats
            }
            _ => cap,
        });
    process::init_admission(admission_cap);

    // RS working-set caps from the engine handshake (uniform across a model's
    // engines → take [0]). bravo-authored bootstrap bundle.
    let rs_caps = {
        let d0 = engine_configs.first();
        let is_rs = d0.map(|d| d.rs_cache_slots > 0).unwrap_or(false);
        model::RsCaps {
            state_size: d0.map(|d| d.rs_cache_slot_bytes).unwrap_or(0),
            buffer_page_size: if is_rs { kv_page_size as u32 } else { 0 },
            fold_granularity: 1, // token-causal; 0-RS models never read it
        }
    };
    let eta_caps = model::EtaCaps {
        // tart: the span-grouped adapter capability, re-threaded onto
        // the 0.3 handshake (engine context.cpp reports it; the worker
        // translate carries it; every engine must honour the sink for
        // the runtime to advertise it).
        has_lora: !engine_configs.is_empty() && engine_configs.iter().all(|d| d.has_lora),
        has_mtp_logits: !engine_configs.is_empty()
            && engine_configs.iter().all(|d| d.has_mtp_logits),
        has_mtp_drafts: !engine_configs.is_empty()
            && engine_configs.iter().all(|d| d.has_mtp_drafts),
        has_value_head: !engine_configs.is_empty()
            && engine_configs.iter().all(|d| d.has_value_head),
        has_kv_envelopes: !engine_configs.is_empty()
            && engine_configs.iter().all(|d| d.has_kv_envelopes),
        has_attn_score: !engine_configs.is_empty()
            && engine_configs.iter().all(|d| d.has_attn_score),
        has_attn_page_mask: !engine_configs.is_empty()
            && engine_configs.iter().all(|d| d.has_attn_page_mask),
    };
    model::register(
        name.clone(),
        &model_id,
        kv_page_size as u32,
        rs_caps,
        eta_caps,
        tokenizer_path.clone(),
        &metadata,
    )?;

    let arena_kv_pages: Vec<usize> = engine_configs.iter().map(|d| d.total_pages).collect();
    let arena_cpu_pages: Vec<usize> = engine_configs.iter().map(|d| d.cpu_pages).collect();
    let arena_rs_slots: Vec<usize> = engine_configs.iter().map(|d| d.rs_cache_slots).collect();
    // Whether engine 0 (the contention-managed pool; working sets are
    // hardwired to (0, 0) until the per-engine core lands) can physically
    // move KV bytes to/from host swap — arms the suspend rung.
    let kv_swap_capable = engine_configs
        .first()
        .is_some_and(|d| d.kv_copy.device_to_host && d.kv_copy.host_to_device);
    let engine_count = engine_configs.len();
    let engines: Vec<usize> = engine_configs
        .into_iter()
        .map(|d| {
            engine::register_engine_backend(
                engine::EngineSpec {
                    // Overwritten by `register_engine_backend` from the
                    // backend itself; see `EngineSpec::device_domain`.
                    device_domain: ::engine::MemoryDomain::HostPinned,
                    num_kv_pages: d.total_pages,
                    limits: d.limits,
                    device_geometry_port_mask: d.device_geometry_port_mask,
                },
                d.engine_backend,
            )
        })
        .collect();

    // Register this model's per-engine typed stores (KvStore/RsStore) in the
    // standalone registry. Capacities are read straight from `cfg.engines[]`.
    // The registry is where the WIT working-set resources and the ETA fire
    // path lock `store::registry::get(...)`.
    let _ = engine_count;
    let arena_model_idx = crate::store::registry::register_model_with_swap(
        kv_page_size as u32,
        &arena_kv_pages,
        &arena_cpu_pages,
        &arena_rs_slots,
    );

    // Residency planner (Project Rainer) — ALWAYS installed: KV pool
    // exhaustion is FCFS eviction/restore, never an inferlet error; there
    // is no legacy mode. `max_concurrent_processes` stays a physical safety
    // cap only. Eviction arms by CAPABILITY, not policy: an engine that
    // advertises D2H+H2D KV copies gets planner-driven eviction; one that
    // cannot move KV bytes degrades to pool-only planning — parked asks
    // ride idle reclaim and natural frees. Uncontended fires never touch
    // the planner beyond two atomic loads.
    crate::planner::init_planner(
        arena_model_idx,
        0,
        crate::planner::ResidencyPlanner::new(std::sync::Arc::new(
            crate::planner::RegistryPool::new(arena_model_idx, 0, kv_swap_capable),
        )),
    );
    // Opt-in stall sampler: `PIE_CONTENTION_TRACE_MS=500` emits one line
    // per tick while anything is queued, so a stalling run reports whether
    // pages are MOVING (churn) or FROZEN (liveness). Off by default.
    if let Some(period) = std::env::var("PIE_CONTENTION_TRACE_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        && let Some(planner) = crate::planner::planner_for(arena_model_idx, 0)
    {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(period));
            loop {
                interval.tick().await;
                let d = planner.diagnostics();
                let (lk_n, lk_wait, lk_hold, lk_wmax, lk_hmax) = planner.lock_census();
                if d.queue.is_empty() && d.proc_states[1..].iter().all(|&n| n == 0) {
                    continue;
                }
                // `println!`, not `tracing`: the embedded (pyo3) server
                // boots with `skip_tracing` and installs no subscriber, so
                // a tracing event here would go nowhere.
                println!(
                    "[planner-trace] queue={} unmet={} head_pages={} head_kind={} bypass={}/{} accum={} \
                     free={}/{} host_free={}/{} head_rs={} rs_free={}/{} \
                     parks={} serves={} evictions={} deferrals={} \
                     evict_rollbacks={} restores={} restore_failures={} gate_parks={} \
                     hogs={} starved={} restarted={} salvaged={} swapfull={}/{} e6_relax={} rshort={} \
                     runway={}/{} \
                     lock_n={} lock_wait_ms={} lock_hold_ms={} lock_wmax_us={} lock_hmax_us={} \
                     d2h_pages={} h2d_pages={} d2h_ms={} h2d_ms={} \
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
                    d.queue.first().map_or(0, |w| w.rs_slots),
                    d.rs_slots_free,
                    d.rs_slots_total,
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
                    d.runway_rounds_total,
                    d.runway_pages_total,
                    lk_n,
                    lk_wait / 1000,
                    lk_hold / 1000,
                    lk_wmax,
                    lk_hmax,
                    d.d2h_pages_total,
                    d.h2d_pages_total,
                    d.d2h_copy_us_total / 1000,
                    d.h2d_copy_us_total / 1000,
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

    // (Context actor `context::spawn` removed — Phase 5. The unified arena
    // registry above is the per-model/engine physical home now.)
    crate::scheduler::set_submit_deadline(std::time::Duration::from_micros(
        scheduler.submit_deadline_us,
    ));
    crate::scheduler::set_silence_timeout(std::time::Duration::from_secs(
        scheduler.silence_timeout_secs,
    ));
    // Both are read once into a `OnceLock` and are guest-visible through
    // `model.frame-size()` / `model.channel-capacity()`, so they must be
    // installed before anything touches the scheduler.
    crate::scheduler::set_frame_size(scheduler.frame_size as usize);
    crate::scheduler::set_dispatch_depth(scheduler.frame_dispatch_depth as usize);
    let scheduler_shutdown = crate::scheduler::spawn(
        &engines,
        kv_page_size as u32,
        scheduler.request_timeout_secs,
    )
    .await?;
    // **THE 10-SECOND ELASTIC-TRIM POLL IS GONE** (alto design §8, wave C;
    // survey §2 debt 9). It woke every ten seconds, scanned this runtime's
    // own page free list for a high water, and sent the engine three
    // `resize_pool` messages — each of which cost a full device drain,
    // measured at 51-75 ms. Elasticity is a side effect of admission now: a
    // frame's union demand is committed atomically by the engine before any
    // of it runs, so the pools hold what has been asked for rather than what
    // somebody polled for. The high water it computed was a supply question
    // asked of a policy structure; the engine owns that number and reports it
    // (`LoadFacts::pool_high_water_bytes`).
    // (The old reclaim-ladder leave/kill/probe hook seams are gone: the
    // planner lives ABOVE both `store` and `scheduler` and calls
    // `scheduler::worker::notify_pipeline_leave_owned` and the residency
    // registry directly.)
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

/// Boot-time checks for the values pie's Python layer cannot validate
/// itself: filesystem-side effects (cache dir) and worker-handshake
/// outputs (tokenizer file, engine capability numbers). Field-level
/// validation of user-supplied scalars (timeouts, etc.) happens in
/// `pie.config.*.__post_init__` — by the time they reach Rust they're
/// already known-good.
fn verify_config(config: &Config) -> Result<()> {
    fs::create_dir_all(&config.cache_dir)
        .with_context(|| format!("Could not create cache dir: {:?}", config.cache_dir))?;

    let model = &config.model;
    // An artifact carries its tokenizer inside it, so there is no file to
    // check for — the artifact itself was already opened to lift the metadata
    // out. Only the tokenizer half is asked about: the descriptor is present
    // for either input form, and its absence is not a shape this type admits.
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

/// Per-component ceiling on the wasmtime resource classes a COMPONENT
/// multiplies. A component is not one core module: a Rust `wasm32-wasip2`
/// guest is linked from the guest module plus the preview1 adapter plus a
/// shim, and instantiating it takes one core-instance slot per module and one
/// table slot per module that defines a table. Measured over all 34 inferlets
/// pie ships, every one is exactly 3 core instances / 2 tables / 1 memory /
/// 1 fiber stack.
///
/// Declaring the ceiling matters as much as its value: without it, a guest
/// built from more modules than expected does not fail at instantiation, it
/// silently divides the runtime's effective inferlet capacity and then fails
/// under load at a concurrency that depends on the traffic. With it, wasmtime
/// rejects such a guest deterministically and names the limit.
///
/// The headroom over the measured 3/2 is cheap: these pools cost reserved
/// address space for instance metadata, not committed memory or KV.
const CORE_RESOURCES_PER_COMPONENT: u32 = 16;

fn init_wasmtime(runtime: &RuntimeConfig) -> wasmtime::Engine {
    let mut wasm_config = wasmtime::Config::default();
    // wasmtime 46: `async_support` is a deprecated no-op (async is always
    // compiled in) and the Component Model Async feature is on by default, so
    // no explicit flags are needed to enable async host calls / fibers.

    // Every wasmtime knob comes from the caller — Python is the source
    // of truth for defaults. `wasm_max_instances` is a cap on concurrent
    // INFERLETS, and each pool below is sized so that it can actually seat
    // that many.
    let mut pooling_config = wasmtime::PoolingAllocationConfig::default();
    // One per inferlet, exactly: pie runs one component instance in one
    // store with one linear memory and one async fiber stack. These are also
    // the expensive pools — a memory slot reserves a whole wasm32 range so
    // bounds checks can be elided — so they must not be inflated.
    pooling_config.total_component_instances(runtime.wasm_max_instances);
    pooling_config.total_memories(runtime.wasm_max_instances);
    pooling_config.total_stacks(runtime.wasm_max_instances);
    // Several per inferlet, however many core modules the component was
    // linked from. Sizing these at `wasm_max_instances` capped pie at
    // `wasm_max_instances / 3` concurrent inferlets: at 512-wide admission,
    // where prewarm + bind hold ~1536 live inferlets, a 4096 pool ran out of
    // core instances (1536 x 3 = 4608) and 55% of requests died with
    // "maximum concurrent limit of 4096 for core instances reached".
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
