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
        // Each planner's census once more at the end: a clean run shows
        // its passes, the map's, and the wakes the edge-triggering skipped.
        for planner in crate::planner::planners() {
            let pass = planner.pass_census();
            let map = planner.map_census();
            let edge = planner.edge_census();
            let stuck = planner.stuck_census();
            tracing::info!(
                passes = pass.0,
                pass_ms = pass.1 / 1000,
                pass_mean_us = pass.2,
                map_passes = map.0,
                map_pass_ms = map.1 / 1000,
                quiet_waits = map.2,
                skipped_wakes = edge.0,
                map_cache_hits = edge.1,
                stuck_settles = stuck.0,
                forced_passes = stuck.1,
                oldest_wait_ms = stuck.2,
                refusal_repeats = planner.refusal_repeats(),
                "planner census at shutdown"
            );
        }
        // Every process ends as the host ends it, and its teardown closes
        // its channels and instances on the schedulers, before they stop:
        // a step queued for a lane whose process went is dropped with the
        // process, never fired into a ring the gone process fills no more.
        let live = crate::inferlet::process::list();
        for &pid in &live {
            crate::inferlet::process::terminate_by(
                pid,
                Err("the server is shutting down".to_string()),
                crate::policy::By::Host,
            );
        }
        crate::inferlet::process::teardown::all_torn_down().await;
        tracing::info!(
            processes = live.len(),
            "every process ended and its instances closed before the schedulers stop"
        );
        let scheduler_result = self.scheduler.shutdown().await;
        for engine_id in self.engine_ids {
            let _ = engine::backend::unregister_engine(engine_id);
        }
        scheduler_result
    }
}

/// Clears what a boot set process-wide, and frees bootstrap for the next
/// boot in this process, which may serve another model. Call it once the
/// boot's [`BootstrapHandle::shutdown`] returned and the tokio runtime it
/// ran on is dropped, so no task of the boot is left to see the state go.
pub fn reset() {
    server::reset();
    process::reset();
    linker::stop();
    program::stop();
    crate::policy::reset();
    crate::pipeline::reset();
    crate::offload::reset();
    crate::handle::snapshot::reset();
    crate::scheduler::reset();
    crate::planner::reset();
    crate::store::reset();
    model::reset();
    engine::backend::reset();
    RUNTIME_ACTIVE.store(false, Ordering::Release);
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
    /// The worker policy (name, component). Every native worker runs one
    /// (`noop` writes no rows); the browser runs none.
    pub policy: Option<(String, Cow<'static, [u8]>)>,
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinProgram {
    pub name: &'static str,
    pub component: &'static [u8],
    /// What it was built from, by content (`builtins::SOURCE`): the same in
    /// any tree of the same source, where `component`'s bytes are not.
    pub source: &'static str,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub worker_threads: usize,

    /// Instances the wasmtime pool holds; `None` holds what this process's
    /// address space fits.
    pub wasm_max_instances: Option<u32>,
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
    pub allow_weight_reload: bool,
    /// Where the `<language>.wasm` language components live.
    pub languages_dir: PathBuf,
    /// Where wasmtime caches compiled code across boots; `None` compiles
    /// every component on every boot.
    pub compile_cache_dir: Option<PathBuf>,
}

pub struct ModelConfig {
    /// How much of the device budget this load gives back to the system's
    /// memory pressure (`[engine.memory] yield`); `None` where the
    /// deployment states none, resolved at load by the engines' memory
    /// ([`crate::governor::yield_for`]).
    pub memory_yield: Option<crate::governor::Yield>,
    pub name: String,
    pub model_id: String,
    pub kv_page_size: usize,
    pub tokenizer_path: PathBuf,
    pub metadata: ModelMetadata,
    pub engines: Vec<EngineConfig>,
    pub scheduler: SchedulerConfig,
    /// The CUDA device a media encode (NVENC) runs on: the engine's own.
    pub media_device: u32,
}

pub struct EngineConfig {
    /// The pools as chunks: what the engine states
    /// ([`::engine::Capabilities::chunks`]).
    pub chunks: ::engine::chunk::ChunkFacts,
    /// The engine serves windowed layers as ring + buffer
    /// (`Capabilities::windowed_state`, CONTRACT-BC 19, 21).
    pub windowed_state: bool,
    /// The disk tier is kept for the next run (`disk_kv_persistent`).
    pub disk_kept: Option<KeptDisk>,
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
    /// What a guest stage's cost is held to
    /// ([`::engine::Capabilities::guest_cost`]).
    pub guest_cost: ::engine::GuestCost,
    /// The frames the engine keeps in flight
    /// ([`::engine::Capabilities::frames_in_flight`]): the scheduler's
    /// dispatch depth and a beam group's pool's `F`.
    pub frames_in_flight: u32,
    pub device_geometry_port_mask: eta_ir::registry::PortMask,
    /// The host ops the engine serves ([`::engine::Capabilities::host_ops`]):
    /// a program needing another is refused at its bind.
    pub host_ops: ::engine::HostOps,
    pub limits: crate::engine::SchedulerLimits,
    /// The step compositions the engine computed before its arm walk and
    /// those it runs eager ([`::engine::Capabilities::steps`]): held to the
    /// set the runtime's rules form over the load's stated facts.
    pub steps: ::engine::StatedSteps,
    /// The mixed steps the engine lays out ([`::engine::Capabilities::mixed_lanes`]):
    /// the set the runtime forms holds no mixed step past them.
    pub mixed_lanes: ::engine::MixedLanes,
    /// The facts the engine's trace branches on.
    pub facts: poem_ir::Facts,
    pub inputs: Vec<poem_ir::InputDecl>,
    pub classes: poem_ir::LaneClasses,
    pub engine_backend: crate::engine::EngineBox,
}

/// The budget a guest program binds within on every engine of the model
/// ([`model::guest_budget`]): the least room of theirs. A model served by
/// no engine, or by one stating no state room, binds no guest program, and
/// is refused by name here rather than refuse each registration.
fn guest_budget(engines: &[EngineConfig]) -> Result<eta_ir::validate::Budget> {
    let budgets = engines
        .iter()
        .map(|engine| model::guest_budget(&engine.chunks, &engine.guest_cost))
        .collect::<std::result::Result<Vec<_>, String>>()
        .map_err(anyhow::Error::msg)?;
    // The least room, and of the bounds stated the tightest, at the
    // slowest launch and elementwise rate any engine measured: a stage
    // within it is within every engine's.
    let submit = budgets
        .iter()
        .filter_map(|budget| budget.submit)
        .reduce(|a, b| eta_ir::validate::SubmitBound {
            bound_ns: a.bound_ns.min(b.bound_ns),
            launch_ns: a.launch_ns.max(b.launch_ns),
            elementwise_bytes_per_ns: a.elementwise_bytes_per_ns.min(b.elementwise_bytes_per_ns),
        });
    let budget = budgets
        .iter()
        .map(|budget| eta_ir::validate::Budget {
            submit,
            ..*budget
        })
        .min_by_key(|budget| budget.room_bytes)
        .ok_or_else(|| anyhow::anyhow!("a model served by no engine has no guest budget"))?;
    if budget.room_bytes == 0 {
        anyhow::bail!(
            "an engine of this model states no state room past what it holds back \
             ({} state chunks of {} bytes, {} held back); no guest program could bind",
            engines.iter().map(|e| e.chunks.state_chunks()).min().unwrap_or(0),
            engines.first().map_or(0, |e| e.chunks.chunk_bytes),
            engines.first().map_or(0, |e| e.chunks.held_bytes),
        );
    }
    tracing::info!(
        room_bytes = budget.room_bytes,
        submit = ?budget.submit,
        "a guest program's stages bind within an instance's state room and, where \
         the device states one, its bound on one submission"
    );
    Ok(budget)
}

/// The frames the model's engines keep in flight
/// ([`::engine::Capabilities::frames_in_flight`]): one number, so engines
/// stating different ones are refused by name.
fn frames_in_flight(engines: &[EngineConfig]) -> Result<u32> {
    let mut stated = engines.iter().map(|e| e.frames_in_flight);
    let Some(first) = stated.next() else {
        anyhow::bail!("a model served by no engine states no frames in flight");
    };
    if first == 0 {
        anyhow::bail!("an engine states no frame in flight; a fire needs one");
    }
    if let Some(other) = stated.find(|&f| f != first) {
        anyhow::bail!(
            "the model's engines keep {first} and {other} frames in flight; the scheduler \
             dispatches by one number"
        );
    }
    Ok(first)
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

/// Records the address space and memory maps this process holds before
/// its engines load: what the sandbox pool leaves the rest of the process
/// to grow by. Call
/// it before the engines load; [`bootstrap`] comes after them.
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_vendor = "apple", not(target_os = "macos"))
)))]
pub use crate::sandbox_pool::note_maps_before_engines;
/// No sandbox pool sizes itself by the process's maps on this target.
#[cfg(any(
    target_arch = "wasm32",
    all(target_vendor = "apple", not(target_os = "macos"))
))]
pub fn note_maps_before_engines() {}

pub async fn bootstrap(config: Config) -> Result<BootstrapHandle> {
    bootstrap_inner(config).await
}

async fn bootstrap_inner(config: Config) -> Result<BootstrapHandle> {
    verify_config(&config)?;
    let mut active_guard = ActiveRuntimeGuard::acquire()?;

    if !config.skip_tracing {
        init_tracing(config.log_dir.as_deref(), config.verbose, &config.telemetry)?;
    }
    let wasm_engine = init_wasmtime(&config.runtime)?;

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
    // The built-ins compiled and linked beside the model's load: a request
    // that launches one compiles and links nothing.
    let builtins_ready = {
        let names: Vec<&'static str> = config.builtin_programs.iter().map(|b| b.name).collect();
        let (tx, rx) = tokio::sync::oneshot::channel();
        crate::rt::spawn(async move {
            let _ = tx.send(prepare_builtins(names).await);
        });
        rx
    };
    let max_upload_bytes = config.runtime.max_upload_mb.saturating_mul(1024 * 1024);
    server::init(
        max_upload_bytes,
        config.runtime.allow_upload,
        config.runtime.allow_weight_reload,
    );
    let bound_port = config.port;

    let ModelConfig {
        memory_yield,
        name,
        model_id,
        kv_page_size,
        tokenizer_path,
        metadata,
        engines: mut engine_configs,
        scheduler,
        media_device,
    } = config.model;
    crate::inferlet::host::frames::set_media_device(media_device);

    // The library forms programs are planned with: what every engine served
    // here states (`pipeline::program::serve_forms`), before any program
    // (the standard samplers first) is registered.
    for d in engine_configs.iter() {
        let facts = d.engine_backend.device_facts().ok_or_else(|| {
            anyhow::anyhow!(
                "an engine states no device facts, so no library forms its emitters serve; \
                 programs are planned with the forms every engine states"
            )
        })?;
        crate::pipeline::program::serve_forms(&facts.backend, facts.library_forms);
    }
    // An unset yield follows the engines' memory: unified memory (the pool
    // and the host share one RAM) yields fully, separate memory holds.
    let unified_memory = engine_configs.iter().any(|d| {
        d.engine_backend
            .device_facts()
            .is_some_and(|facts| facts.unified_memory)
    });
    // Where an engine reads the host's room itself, the yield reacts only to
    // the rest of the system (`governor::yield_for`).
    let room = crate::governor::Room::of(
        engine_configs
            .iter()
            .any(|d| d.engine_backend.host_room().is_some()),
    );
    let (memory_yield, stated) = crate::governor::yield_for(memory_yield, unified_memory, room);
    tracing::info!("{stated}");

    // Each engine's readouts, a fire bound (`Engine::readout_facts`): a
    // fixed store's rows bound the rows one fire reads out, which the
    // scheduler cuts its fires to; rows that land beside the activation
    // are counted at every rung of a shared pool, below.
    let readouts = engine_configs
        .iter_mut()
        .map(|d| {
            let facts = d
                .engine_backend
                .readout_facts()
                .map_err(|error| anyhow::anyhow!("engine readout facts: {error}"))?;
            d.limits = d.limits.read_out_by(&facts);
            // The step compositions the step builder forms (CONTRACT-BC 24):
            // the runtime's rules over the load's stated facts, the set the
            // engine walked before ready, or the load is refused by name.
            let rules = ::engine::StepRules::runtime();
            let step_facts = d
                .limits
                .step_facts(
                    &d.chunks,
                    ::engine::Attachments::served(
                        d.has_lora,
                        !d.inputs.is_empty(),
                        d.rs_snapshots > 0,
                    ),
                )
                .with_mixed(d.mixed_lanes.clone());
            d.steps.agrees(&rules, &step_facts).map_err(|why| {
                anyhow::anyhow!("engine {}: step shapes: {why}", d.engine_backend.kind())
            })?;
            tracing::info!(
                engine = d.engine_backend.kind(),
                eager = d.steps.eager.len(),
                "the step builder forms {}; each is armed before ready or stated eager",
                d.steps.shapes.census()
            );
            // One set a load, shared by every step it builds for its life.
            d.limits.shapes = Some(Box::leak(Box::new(d.steps.shapes.clone())));
            Ok(facts)
        })
        .collect::<Result<Vec<::engine::ReadoutFacts>>>()?;

    // The frames the engines keep in flight: the one number the scheduler
    // dispatches by and a beam group's pool is sized with.
    let engine_frames = frames_in_flight(&engine_configs)?;
    // the warm run that measures the host cover stays within every engine's
    // context (`measure_host_cover`)
    let warm_context = engine_configs
        .iter()
        .map(|d| d.limits.max_context)
        .filter(|&c| c > 0)
        .min()
        .unwrap_or(0);
    // The frame size, its ceiling and the depth, before anything reads a
    // channel's capacity (`scheduler::channel_capacity`): the standard
    // samplers' prewarm below traces its programs at the capacity a
    // request's programs take.
    crate::scheduler::set_frame_size(scheduler.frame_size as usize);
    crate::scheduler::set_frame_ceiling(scheduler.frame_size as usize);
    crate::scheduler::set_dispatch_depth(engine_frames as usize);
    // No count stands for memory: a newcomer is admitted once its first
    // ask, what it holds at its first step, passes the planner's capacity
    // gate at the order's last place (`ResidencyPlanner::admits`). A
    // count of sequences at the declared context seated too few where
    // prompts are short (gpt-oss-20b c64: 62 of 64) and stood for no
    // model's mix. What stays is a count stated as one: the operator's, or
    // the most requests an engine takes into one fire.
    let admission_cap = config.max_concurrent_processes.or_else(|| {
        engine_configs
            .iter()
            .map(|d| d.limits.max_forward_requests)
            .min()
            .filter(|&r| r > 0)
    });
    process::init_admission(admission_cap);

    // Windowed layers as ring + buffer (CONTRACT-BC 19, 21): stated only by
    // an engine that serves them so, and the same way by every engine of
    // the model; such a load is a recurrent-state load whose record holds
    // the rings.
    for (at, d) in engine_configs.iter().enumerate() {
        anyhow::ensure!(
            d.chunks.windowed.is_empty() || d.windowed_state,
            "engine {at} states windowed layers served as ring + buffer \
             (ChunkFacts::windowed) but not Capabilities::windowed_state"
        );
    }
    let window_state = engine_configs
        .first()
        .is_some_and(|d| d.chunks.window_buffer().is_some());
    if let Some(at) = engine_configs
        .iter()
        .position(|d| d.chunks.window_buffer().is_some() != window_state)
    {
        anyhow::bail!(
            "engine {at} serves the model's windowed layers otherwise than engine 0 (a window \
             buffer on one, none on the other); the engines of one model hold one state"
        );
    }
    let rs_caps = {
        let d0 = engine_configs.first();
        let mut caps = model::RsCaps {
            state_size: d0.map_or(0, |d| d.chunks.record_bytes),
            buffer_page_size: 0,
            window_state,
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
        budget: guest_budget(&engine_configs)?,
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
    // A class that reads no kv is a float lane's (an encoder's, an
    // embedding's): it reads out every row through the readout landing.
    let float_lanes = engine_configs
        .first()
        .is_some_and(|engine| engine.classes.classes.iter().any(|class| !class.reads_kv));
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

    // The standard samplers' guest code, built before ready: a request that
    // registers one of them builds nothing. One that cannot be built fails
    // the load, as a rung that cannot be armed does; a backend that builds
    // no guest code keeps the default, which builds nothing.
    // Registered as a request's are: a per-row channel declared at the rows
    // one fire binds, `model.max-pass-rows()` (the first engine's).
    let rows = engine_configs
        .first()
        .map_or(u32::MAX, |engine| {
            u32::try_from(engine.limits.max_forward_tokens).unwrap_or(u32::MAX)
        });
    for (at, engine) in engine_configs.iter_mut().enumerate() {
        let backend =
            crate::engine::verbs::codegen_backend(&engine.engine_backend).map(str::to_owned);
        let registrations = crate::pipeline::standard::registrations(backend.as_deref(), rows)?;
        let Some(programs) = engine.engine_backend.programs() else {
            continue;
        };
        programs
            .prewarm(&registrations)
            .map_err(|error| {
                anyhow::anyhow!("engine {at}: the standard samplers' guest code: {error}")
            })?;
    }

    let scales: Vec<crate::policy::pressure::Scale> = engine_configs
        .iter()
        .map(|d| {
            crate::store::shared::SharedPool::scale_of(&d.chunks).unwrap_or(
                crate::policy::pressure::Scale {
                    kv_page_bytes: 0,
                    state_slot_bytes: d.chunks.record_bytes,
                },
            )
        })
        .collect();
    // Each engine's weight shapes and the chunks they may take.
    let chunk_facts: Vec<::engine::chunk::ChunkFacts> =
        engine_configs.iter().map(|d| d.chunks.clone()).collect();
    let arena_disk_kept: Vec<Option<KeptDisk>> =
        engine_configs.iter().map(|d| d.disk_kept.clone()).collect();
    // Each engine's pools as chunks: the shared pool it states.
    let arena_stores = engine_configs
        .iter()
        .zip(&readouts)
        .map(|(d, readouts)| {
            let pools = crate::store::chunks::StorePools::of(&d.chunks)
                .map_err(|error| anyhow::anyhow!("engine pools: {error}"))?;
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
                let mut pool = pool;
                // What the engine holds back of the range it stated is no
                // sequence's from the first page on: a load fact, seeded
                // here. An engine that arms at the weight residency it
                // starts with says there what it still holds
                // (`Shared::<Steered>::boot`), before the load returns.
                pool.hold_back(d.chunks.held_bytes);
                // A rung of `r` tokens carries `min(r, max_lanes)` lanes, as
                // the engine counts them: what a program's fire term takes.
                pool.set_max_lanes(u32::try_from(d.limits.max_forward_requests).unwrap_or(u32::MAX));
                // The rows a fire reads out: counted beside the guest term
                // where they land beside the activation, a lane reading
                // more than a fixed store holds refused at its bind.
                pool.set_readouts(
                    readouts,
                    u32::try_from(d.limits.max_readouts).unwrap_or(u32::MAX),
                    float_lanes,
                );
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
    // Test builds: a slot the shared pool gives back is poisoned by its
    // engine before the engine's next verb (`Pool::poison_queue`), so a
    // fire that reads a cell its owner never wrote reads NaN.
    #[cfg(feature = "test-slot-fill")]
    for (engine, stores) in engine_configs.iter_mut().zip(&arena_stores) {
        if let Some(shared) = &stores.shared {
            let queue = ::engine::chunk::PoisonQueue::default();
            shared.lock().unwrap().pool_mut().poison_into(queue.clone());
            if let Some(pool) = engine.engine_backend.pool() {
                pool.poison_queue(queue);
            }
        }
    }
    // A shared pool's tiers are its host chunks and disk slots; today's
    // pools state theirs in pages.
    let kv_swap_capable = engine_configs.first().is_some_and(|d| {
        if crate::store::shared::SharedPool::states_shared(&d.chunks) {
            d.chunks.host_chunks > 0 || d.chunks.disk_chunks > 0
        } else {
            d.chunks.kv_host_pages() > 0 || d.chunks.kv_disk_pages() > 0
        }
    });
    if let Some(d) = engine_configs.first() {
        // What the planner admits against on a shared pool (its total,
        // `SharedPool::room`), beside the facts' count: the two count the
        // same pages, the span a lowered rung lends in neither; said once
        // here so they cannot part silently.
        let (planner_pages, lent_chunks) = arena_stores
            .first()
            .and_then(|stores| stores.shared.as_ref())
            .map_or((d.chunks.kv_device_pages(), 0), |shared| {
                let room = shared.lock().unwrap().room();
                (u64::from(room.list_slots), room.lent_chunks)
            });
        if planner_pages != d.chunks.kv_device_pages() {
            tracing::warn!(
                facts_pages = d.chunks.kv_device_pages(),
                planner_pages,
                "the planner's kv total is not the engine's: the pinned weights or the \
                 pool's ceiling take the difference"
            );
        }
        let unified = d
            .engine_backend
            .device_facts()
            .is_some_and(|facts| facts.unified_memory);
        if kv_swap_capable && unified && d.chunks.host_chunks == 0 {
            // No host tier: on a device whose memory is the host's a move to
            // the host frees nothing, so the engine states none.
            tracing::info!(
                device_pages = d.chunks.kv_device_pages(),
                planner_pages,
                lent_chunks,
                host = "none",
                disk_chunks = d.chunks.disk_chunks,
                chunk_bytes = d.chunks.chunk_bytes,
                "kv tiers: no host tier (the device's memory is the host's); suspended kv \
                 goes to the disk"
            );
        } else if kv_swap_capable && crate::store::shared::SharedPool::states_shared(&d.chunks) {
            // A shared pool states its tiers in chunks, not in the page
            // fields its engine leaves at 0.
            tracing::info!(
                device_pages = d.chunks.kv_device_pages(),
                planner_pages,
                lent_chunks,
                host_chunks = d.chunks.host_chunks,
                disk_chunks = d.chunks.disk_chunks,
                chunk_bytes = d.chunks.chunk_bytes,
                "kv tiers: suspended kv goes to {}",
                below_device(d.chunks.host_chunks > 0, d.chunks.disk_chunks > 0)
            );
        } else if kv_swap_capable {
            tracing::info!(
                device_pages = d.chunks.kv_device_pages(),
                planner_pages,
                host_pages = d.chunks.kv_host_pages(),
                disk_pages = d.chunks.kv_disk_pages(),
                "kv tiers: suspended kv goes to {}",
                below_device(d.chunks.kv_host_pages() > 0, d.chunks.kv_disk_pages() > 0)
            );
        } else {
            tracing::warn!(
                device_pages = d.chunks.kv_device_pages(),
                planner_pages,
                lent_chunks,
                "no host or disk kv tier: pie cannot compute kv again, so kv \
                 pressure restarts processes"
            );
        }
    }
    let engine_count = engine_configs.len();
    // Each engine's share of the host's room, where its device shares the
    // host's memory: its governor holds its pool within it. Every engine of
    // this process is on one host, so they share one ledger of its room (the
    // first reader stated reads it for all), and each pool's grant is
    // debited before the next reads: N pools are granted the room once.
    // Where the host cover is kept between boots, keyed by every engine's
    // fingerprint (`host_cover`): none where an engine states none.
    let cover_cache = engine_configs
        .iter()
        .map(|d| d.engine_backend.cover_fingerprint())
        .collect::<Option<Vec<String>>>()
        .filter(|prints| !prints.is_empty())
        .zip(
            // The home's cache: where wasmtime's compile cache lies in it.
            config
                .runtime
                .compile_cache_dir
                .as_ref()
                .and_then(|wasmtime| wasmtime.parent())
                .map(Path::to_path_buf),
        )
        .map(|(prints, dir)| CoverCache {
            dir,
            engines: prints.join(";"),
            deployment: model_id.clone(),
            sampler_builtin: config
                .builtin_programs
                .iter()
                .find(|b| b.name == "compat-openai")
                .map(|b| (program::ProgramHash::of(b.component), b.source)),
            committed: engine_configs
                .iter()
                .flat_map(|d| d.engine_backend.committed_covers().iter().copied())
                .collect(),
        });
    // Every engine skips measuring (`[engine] measure = false`): the host
    // cover no layer states is taken by its estimate, not warm runs.
    let cover_skipped = !engine_configs.is_empty()
        && engine_configs
            .iter()
            .all(|d| d.engine_backend.skips_measuring());
    let readers: Vec<Option<::engine::HostRoom>> = engine_configs
        .iter()
        .map(|d| d.engine_backend.host_room())
        .collect();
    let host = readers
        .iter()
        .flatten()
        .next()
        .cloned()
        .map(crate::governor::run::HostShare::new);
    let mut rooms: Vec<Option<crate::governor::run::PoolRoom>> = readers
        .iter()
        .map(|reader| reader.as_ref().and(host.as_ref()).map(|host| host.pool()))
        .collect();
    // Every engine's own measurements beside the cover's, said once the
    // load stands: "measured 0" where none was timed.
    let mut measured: Vec<String> = Vec::new();
    for d in &engine_configs {
        measured.extend(d.engine_backend.measured_at_load());
    }
    let engines: Vec<usize> = engine_configs
        .into_iter()
        .map(|d| {
            let engine = engine::register_engine_backend(
                engine::EngineSpec {
                    device_domain: ::engine::MemoryDomain::HostPinned,
                    limits: d.limits,
                    device_geometry_port_mask: d.device_geometry_port_mask,
                    host_ops: d.host_ops,
                    kind: "",
                    codegen_backend: None,
                },
                d.engine_backend,
                d.chunks.eager_rungs,
            );
            // What the driver holds outside the engine's span is a load
            // fact: the policy's ledger takes it once.
            crate::policy::pressure::set_outside(engine, d.chunks.outside_bytes);
            engine
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
                shared_steered_pool(arena_model_idx, engine, chunk_facts.get(engine), grow)?
            }
            None => steered_pool(arena_model_idx, engine, chunk_facts.get(engine))?,
        };
        crate::planner::init_planner(
            arena_model_idx,
            engine,
            crate::planner::ResidencyPlanner::new(std::sync::Arc::new(pool))
                .weight_shed(steered.shed_hook()),
        );
        governed.push(steered);
    }
    // After the gates and the planner it orders exist.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (name, component) = config
            .policy
            .as_ref()
            .context("a worker names its policy (`noop` writes no rows)")?;
        crate::policy::start(name, component).with_context(|| format!("worker policy {name}"))?;
    }

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
    // A yield that moves some pool needs the system's memory pressure: with
    // none to read, the load is refused by name rather than yielding
    // nothing. At yield 0, or where every pool is laid whole, nothing is
    // read.
    let moves = governed.iter().any(crate::governor::run::yield_applies);
    let source = if moves {
        crate::governor::source_for(memory_yield, || room.source())
    } else {
        Ok(None)
    };
    let source = match source {
        Ok(source) => source,
        Err(why) => {
            scheduler_shutdown.shutdown().await.ok();
            return Err(anyhow::anyhow!(why));
        }
    };
    for (engine, steered) in governed.into_iter().enumerate() {
        let room = rooms.get_mut(engine).and_then(Option::take);
        crate::governor::run::start(steered, memory_yield, room, source.clone());
    }
    let builtins = match builtins_ready.await {
        Ok(prepared) => prepared,
        Err(_) => Err(anyhow::anyhow!("preparing the built-in programs ended unanswered")),
    };
    if let Err(error) = builtins {
        scheduler_shutdown.shutdown().await.ok();
        return Err(error);
    }
    // The host cover, measured through the ordinary pipeline before ready,
    // or read where an earlier boot measured it under the same key.
    match host_cover(&name, warm_context, cover_cache.as_ref(), cover_skipped).await {
        Ok((cover, timed)) => {
            crate::scheduler::set_host_cover(cover);
            if timed {
                measured.push("host cover".to_string());
            }
            if measured.is_empty() {
                tracing::info!("load: measured 0");
            } else {
                tracing::info!("load: measured {}: {}", measured.len(), measured.join(", "));
            }
        }
        Err(error) => {
            tracing::error!(%error, "measuring the host cover at load");
            scheduler_shutdown.shutdown().await.ok();
            return Err(error.context("measuring the host cover at load"));
        }
    }
    active_guard.disarm();
    // Ready: what an engine left to measure behind serving runs from here,
    // where its lane idles.
    crate::scheduler::open_behind();
    Ok(BootstrapHandle {
        port: bound_port,
        shutdown: Some(RuntimeShutdown {
            scheduler: scheduler_shutdown,
            engine_ids: engines,
        }),
    })
}

/// The host cover (`scheduler::host_cover`), measured before ready on warm
/// runs of the standard sampler through the ordinary pipeline at one frame
/// in flight: the cover is a lane's guest turnaround (its frame landing to
/// its next frame's submit, the quorum probes' turnaround) over one frame's
/// device time (the fires' launch to retire, at depth 1 nothing queued),
/// rounded up. Rounds of twice the tokens each, from one frame's; each
/// round's counters' growth over it alone, and the cover from two rounds'
/// difference (a decode frame's marginal cost: the prompt fire and every
/// fixed cost cancel, as in the beam rule's T(256) - T(32)), until two
/// successive differences give the same cover (or a round would pass the
/// engines' context). The cover is the largest any difference gave: a
/// difference spanning few frames swings with the turnaround's noise, and a
/// cover too small leaves the device idle where one too large costs only
/// window.
/// The host cover: read from `cache` (`<cache>/cover/<key>.json`) where an
/// earlier boot measured it under the same key, else measured
/// ([`measure_host_cover`]) and written there. The key names what the cover
/// depends on: each engine's [`::engine::Engine::cover_fingerprint`] (its
/// device, driver, kernels and guest code), the deployment, the sampler
/// program's own hash, the frame size, the frames in flight and the warm
/// runs' context. Not the runtime's build: a build that changes none of
/// those reuses the cover. `cache` is `None` where an engine states no
/// fingerprint: the cover is measured.
/// Looked up the home's own cache first, then the covers committed for the
/// engines' devices (`::engine::Engine::committed_covers`, the same entries
/// under the same key), then measured; where every engine skips measuring
/// (`skipped`, `[engine] measure = false`), the estimate in place of the
/// warm runs ([`cover_estimate`]), said by name and kept nowhere.
async fn host_cover(
    model: &str,
    context: usize,
    cache: Option<&CoverCache>,
    skipped: bool,
) -> Result<(usize, bool)> {
    let Some(cache) = cache else {
        if skipped {
            return Ok((cover_estimate("no engine keys the cover"), false));
        }
        return measure_host_cover(model, context)
            .await
            .map(|(cover, ..)| (cover, true));
    };
    let reference = program::ProgramRef::parse("compat-openai")?;
    let built = program::resolve(None, &reference).await?;
    // The sampler by what it was built from where it is the built-in (its
    // bytes differ tree to tree, its source does not), else by its hash.
    let sampler = match &cache.sampler_builtin {
        Some((hash, source)) if *hash == built.hash => format!("source {source}"),
        _ => built.hash.to_string(),
    };
    let key = format!(
        "{}|{}|sampler {sampler}|frame {}|depth {}|context {context}",
        cache.engines,
        cache.deployment,
        crate::scheduler::configured_frame_size().max(1),
        crate::scheduler::configured_dispatch_depth(),
    );
    let name = key
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    let path = cache.dir.join("cover").join(format!("{name:016x}.json"));
    let read = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter(|held| held.get("key").and_then(serde_json::Value::as_str) == Some(key.as_str()));
    if let Some(held) = read {
        let field = |name: &str| held.get(name).and_then(serde_json::Value::as_f64);
        if let (Some(cover), Some(frame_us), Some(turnaround_us)) =
            (field("cover"), field("frame_us"), field("turnaround_us"))
        {
            let cover = cover as usize;
            tracing::info!(
                cover,
                frame_us,
                turnaround_us,
                path = %path.display(),
                "the host cover, read where an earlier boot measured it under this key"
            );
            return Ok((cover, false));
        }
    }
    if let Some(held) = cache.committed.iter().find_map(|text| committed_cover(text, &key)) {
        tracing::info!(
            cover = held.cover,
            frame_us = held.frame_us,
            turnaround_us = held.turnaround_us,
            "the host cover, committed for the engines' devices under this key; nothing measured"
        );
        return Ok((held.cover, false));
    }
    if skipped {
        return Ok((
            cover_estimate("neither the home's cache nor a committed cover holds this key"),
            false,
        ));
    }
    let (cover, frame_us, turnaround_us) = measure_host_cover(model, context).await?;
    let held = serde_json::json!({
        "key": key,
        "cover": cover,
        "frame_us": frame_us,
        "turnaround_us": turnaround_us,
    });
    if let Err(why) = std::fs::create_dir_all(path.parent().unwrap_or(&cache.dir))
        .and_then(|()| std::fs::write(&path, held.to_string()))
    {
        tracing::warn!(%why, path = %path.display(), "the host cover is not kept for the next boot");
    }
    Ok((cover, true))
}

/// Where the host cover is kept between boots, and what of the load its key
/// names ([`host_cover`]).
struct CoverCache {
    dir: PathBuf,
    /// Each engine's `cover_fingerprint`, in order.
    engines: String,
    /// The model the cover's warm runs serve.
    deployment: String,
    /// The covers committed for the engines' devices: each file's text.
    committed: Vec<&'static str>,
    /// The built-in sampler's hash and source fingerprint, where it is
    /// installed: the key names its source where the resolved sampler is it.
    sampler_builtin: Option<(program::ProgramHash, &'static str)>,
}

/// One kept host cover, as the cache and a committed file state it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeldCover {
    pub cover: usize,
    pub frame_us: f64,
    pub turnaround_us: f64,
}

/// The entry of `text` (a committed covers file: a JSON list of the cache's
/// entries) whose key is `key`, where it holds one.
#[must_use]
pub fn committed_cover(text: &str, key: &str) -> Option<HeldCover> {
    let all: Vec<serde_json::Value> = serde_json::from_str(text).ok()?;
    all.iter()
        .find(|entry| entry.get("key").and_then(serde_json::Value::as_str) == Some(key))
        .and_then(|entry| {
            let field = |name: &str| entry.get(name).and_then(serde_json::Value::as_f64);
            Some(HeldCover {
                cover: field("cover")? as usize,
                frame_us: field("frame_us")?,
                turnaround_us: field("turnaround_us")?,
            })
        })
}

/// The host cover where every engine skips measuring and no layer states
/// it: the frames the scheduler dispatches ahead at most
/// (`scheduler::configured_dispatch_depth`). A cover is a guest turnaround
/// over one frame's device time; one too small leaves the device idle
/// between frames, one too large costs only window, never a result, so the
/// most the dispatch depth allows is the estimate that never idles the
/// device. Said by name.
fn cover_estimate(why: &str) -> usize {
    let cover = crate::scheduler::configured_dispatch_depth().max(1);
    tracing::info!(
        cover,
        "the host cover: skipped (`[engine] measure = false`; {why}), served by the estimate: \
         the frames dispatched ahead at most, so no turnaround idles the device (a cover too \
         large costs window, never a result); no warm runs, kept nowhere"
    );
    cover
}

async fn measure_host_cover(model: &str, context: usize) -> Result<(usize, f64, f64)> {
    let depth = crate::scheduler::configured_dispatch_depth();
    crate::scheduler::set_dispatch_depth(1);
    let reference = program::ProgramRef::parse("compat-openai")?;
    let built = program::resolve(None, &reference).await?;
    program::load(&built).await?;
    let frame = crate::scheduler::configured_frame_size().max(1);
    let mut tokens = frame;
    let mut last: Option<usize> = None;
    let mut most = 0usize;
    // the previous round's growth: fires, device us, turnarounds, turnaround us
    let mut prev: Option<(u64, u64, u64, u64)> = None;
    let measured = loop {
        let before = crate::scheduler::get_stats().await;
        let input = serde_json::json!({
            "path": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": "Write a short story about a robot."}],
                "max_tokens": tokens,
                "temperature": 0.8,
                "top_p": 0.9,
            },
            "time": 0,
        })
        .to_string();
        let (tx, rx) = tokio::sync::oneshot::channel();
        process::spawn(
            "pie".to_string(),
            built.clone(),
            input,
            std::sync::Arc::new(Default::default()),
            None,
            false,
            Some(tx),
            None,
            crate::policy::Arrival::default(),
            Vec::new(),
        )?;
        match rx.await {
            Ok(Ok(_)) => {}
            Ok(Err(why)) => anyhow::bail!("the warm run failed: {why}"),
            Err(_) => anyhow::bail!("the warm run ended unanswered"),
        }
        let after = crate::scheduler::get_stats().await;
        let fires = after.total_batches.saturating_sub(before.total_batches);
        let device_us = after
            .cumulative_batch_latency_us
            .saturating_sub(before.cumulative_batch_latency_us);
        let turns = after
            .fire
            .quorum
            .turnaround_n
            .saturating_sub(before.fire.quorum.turnaround_n);
        let turn_us = after
            .fire
            .quorum
            .turnaround_sum_us
            .saturating_sub(before.fire.quorum.turnaround_sum_us);
        let grown = (fires, device_us, turns, turn_us);
        let next = tokens.saturating_mul(2);
        let past_context = context > 0 && next >= context;
        if let Some((f0, d0, n0, t0)) = prev {
            let (df, dd, dn, dt) = (
                fires.saturating_sub(f0),
                device_us.saturating_sub(d0),
                turns.saturating_sub(n0),
                turn_us.saturating_sub(t0),
            );
            // too short a difference to hold a frame and a turnaround: the
            // next round, twice as long, measures
            if df == 0 || dn == 0 || dd == 0 {
                if past_context {
                    anyhow::bail!(
                        "warm runs up to the engines' context of {context} timed no decode \
                         frame's turnaround ({df} fires, {dn} turnarounds more)"
                    );
                }
                prev = Some(grown);
                tokens = next;
                continue;
            }
            let frame_us = dd as f64 / df as f64;
            let turnaround_us = dt as f64 / dn as f64;
            let cover = (turnaround_us / frame_us).ceil() as usize;
            tracing::info!(
                tokens,
                frame_us,
                turnaround_us,
                cover,
                "a warm run of the standard sampler, measuring the host cover"
            );
            most = most.max(cover);
            if last == Some(cover) || past_context {
                break (most, frame_us, turnaround_us);
            }
            last = Some(cover);
        } else if past_context {
            anyhow::bail!("the engines' context of {context} holds no second warm run");
        }
        prev = Some(grown);
        tokens = next;
    };
    crate::scheduler::set_dispatch_depth(depth);
    let (cover, frame_us, turnaround_us) = measured;
    tracing::info!(
        cover,
        frame_us,
        turnaround_us,
        frames_in_flight = depth,
        "the host cover, measured at load: a guest turnaround over one frame's device time"
    );
    Ok((cover, frame_us, turnaround_us))
}

/// What each built-in's name runs for the system (the operator's install of
/// that name, else the built-in), compiled and linked.
async fn prepare_builtins(names: Vec<&'static str>) -> Result<()> {
    for name in names {
        let reference = program::ProgramRef::parse(name)?;
        let built = program::resolve(None, &reference).await?;
        program::load(&built)
            .await
            .with_context(|| format!("the built-in program {name}"))?;
    }
    Ok(())
}

/// Engine `engine`'s governor target where its pools are pinned (they never
/// change size), with its weight cache in front ([`boot_cache`]: held where
/// the pool seats every weight, else streaming).
///
/// [`boot_cache`]: crate::store::weights::boot_cache
fn steered_pool(
    model: usize,
    engine: usize,
    facts: Option<&::engine::chunk::ChunkFacts>,
) -> Result<crate::governor::steered::Shared<crate::governor::steered::Steered>> {
    use crate::store::weights::{WeightFirst, boot_cache, port::EnginePort, slot_pool};
    let none = ::engine::chunk::ChunkFacts::default();
    let facts = facts.unwrap_or(&none);
    let mut slots = slot_pool(facts);
    let cache = boot_cache(engine, &facts.weights, 0, &mut slots)?;
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
    facts: Option<&::engine::chunk::ChunkFacts>,
    grow: crate::store::shared::SharedGrow,
) -> Result<crate::governor::steered::Shared<crate::governor::steered::Steered>> {
    use crate::store::weights::{WeightFirst, boot_cache, port::EnginePort};
    let none = ::engine::chunk::ChunkFacts::default();
    let facts = facts.unwrap_or(&none);
    let cache = {
        let mut shared = grow.shared().lock().unwrap();
        let first = shared.weights_first();
        boot_cache(engine, &facts.weights, first, shared.pool_mut())?
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
fn init_wasmtime(_runtime: &RuntimeConfig) -> Result<wasmtime::Engine> {
    Ok(browser_wasm_engine())
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
fn init_wasmtime(runtime: &RuntimeConfig) -> Result<wasmtime::Engine> {
    let mut wasm_config = wasmtime::Config::default();
    configure_execution(&mut wasm_config, runtime)?;

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

    wasmtime::Engine::new(&wasm_config)
        .map_err(|error| anyhow::anyhow!("the inferlet sandbox's wasmtime engine: {error:#}"))
}

/// Apple's platforms other than macOS (iOS, visionOS, tvOS, watchOS) give an
/// app no executable memory: inferlets run as Pulley bytecode, with memories
/// allocated on demand.
#[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
fn configure_execution(wasm_config: &mut wasmtime::Config, _runtime: &RuntimeConfig) -> Result<()> {
    wasm_config.target("pulley64").map_err(|error| {
        anyhow::anyhow!("pulley64 is compiled in where an app may not JIT: {error:#}")
    })?;
    wasm_config.signals_based_traps(false);
    wasm_config.memory_reservation(0);
    wasm_config.memory_guard_size(0);
    wasm_config.memory_init_cow(false);
    Ok(())
}

/// The pooling allocator, its instance count held to what this process's
/// address space fits (`sandbox_pool`).
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_vendor = "apple", not(target_os = "macos"))
)))]
fn configure_execution(wasm_config: &mut wasmtime::Config, runtime: &RuntimeConfig) -> Result<()> {
    use crate::sandbox_pool;

    let mut pooling_config = wasmtime::PoolingAllocationConfig::default();
    pooling_config.max_core_instances_per_component(CORE_RESOURCES_PER_COMPONENT);
    pooling_config.max_tables_per_component(CORE_RESOURCES_PER_COMPONENT);
    pooling_config.max_memory_size(runtime.wasm_max_memory_mb.saturating_mul(1024 * 1024));
    pooling_config
        .linear_memory_keep_resident(runtime.wasm_warm_memory_mb.saturating_mul(1024 * 1024));
    pooling_config.max_unused_warm_slots(runtime.wasm_warm_slots);

    // What an instance reserves, read off an engine of this configuration
    // that allocates on demand (it reserves nothing): its memory
    // reservation, guard and stack, as wasmtime resolves them.
    let probe = wasmtime::Engine::new(wasm_config)
        .map_err(|error| anyhow::anyhow!("the inferlet sandbox's wasmtime engine: {error:#}"))?;
    let layout = sandbox_pool::PoolLayout {
        page_bytes: sandbox_pool::page_bytes(),
        memory_reservation: probe.get_memory_reservation(),
        max_memory: pooling_config.get_max_memory_size() as u64,
        memory_guard: probe.get_memory_guard_size(),
        guard_before: probe.get_guard_before_linear_memory(),
        stack_bytes: probe.get_async_stack_size() as u64,
        tables_per_instance: u64::from(CORE_RESOURCES_PER_COMPONENT),
        table_elements: pooling_config.get_table_elements() as u64,
        table_element_bytes: std::mem::size_of::<usize>() as u64,
        // wasmtime lays its table pool accessible whole, which Windows
        // commits (`MEM_COMMIT`); elsewhere it is only mapped.
        tables_committed: cfg!(windows),
    };
    drop(probe);
    let cost = layout.cost();
    let space = sandbox_pool::read_address_space();
    let has = space.as_ref().ok().map_or_else(
        || "an address space not read".to_string(),
        |space| {
            format!(
                "{}, {} and {}",
                space.bytes_shown(),
                space.maps_shown(),
                space.commit_shown()
            )
        },
    );
    let instances = sandbox_pool::size_pool(runtime.wasm_max_instances, space, &cost)?;
    tracing::info!(
        "sandbox pool: {instances} instances{}, {} of address space ({} and {} memory maps each); \
         this process has {has}",
        if runtime.wasm_max_instances.is_some() {
            " as configured"
        } else {
            " (what this process's address space fits)"
        },
        sandbox_pool::show(cost.bytes(u64::from(instances))),
        sandbox_pool::show(cost.instance_bytes),
        cost.instance_maps,
    );

    pooling_config.total_component_instances(instances);
    pooling_config.total_memories(instances);
    pooling_config.total_stacks(instances);
    pooling_config.total_core_instances(instances.saturating_mul(CORE_RESOURCES_PER_COMPONENT));
    pooling_config.total_tables(instances.saturating_mul(CORE_RESOURCES_PER_COMPONENT));

    wasm_config.allocation_strategy(wasmtime::InstanceAllocationStrategy::Pooling(
        pooling_config,
    ));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn init_tracing(
    _log_dir: Option<&Path>,
    _verbose: bool,
    _telemetry_config: &TelemetryConfig,
) -> Result<()> {
    Ok(())
}

/// Crates whose own debug output buries the runtime's, held down under a
/// level that would let it through: naga's WGSL frontend writes ~880k lines
/// at `debug` while a load builds its pipelines, and tarpc writes seven per
/// gateway heartbeat. A filter that names either target keeps its own say,
/// and one whose base level is already as quiet gets nothing added: a quiet
/// target is never made louder than the operator's level.
#[cfg(not(target_arch = "wasm32"))]
const QUIET: [(&str, tracing_subscriber::filter::LevelFilter); 2] = [
    ("naga", tracing_subscriber::filter::LevelFilter::WARN),
    ("tarpc", tracing_subscriber::filter::LevelFilter::INFO),
];

/// The tracing filter for `level`: `RUST_LOG` when it is set, as it says;
/// otherwise `level` with [`QUIET`]'s targets held down.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn log_filter(level: &str) -> tracing_subscriber::EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(quieted(level)))
}

/// `filter` with a directive for each [`QUIET`] target it names nowhere and
/// whose quiet level is below the filter's base level. The base level is
/// the last directive with no target (`off` where there is none, as an
/// `EnvFilter` reads it); a directive's target is what precedes its `[`
/// span or its `=`.
#[cfg(not(target_arch = "wasm32"))]
fn quieted(filter: &str) -> String {
    use tracing_subscriber::filter::LevelFilter;

    let directives: Vec<&str> = filter
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .collect();
    let target_of = |directive: &str| -> Option<String> {
        let head = directive.split(['[', '=']).next().unwrap_or("");
        let names_a_target = directive.contains('=') || directive.contains('[');
        (names_a_target || head.parse::<LevelFilter>().is_err()).then(|| head.trim().to_owned())
    };
    let base = directives
        .iter()
        .rev()
        .filter(|d| target_of(d).is_none())
        .find_map(|d| d.parse::<LevelFilter>().ok())
        .unwrap_or(LevelFilter::OFF);
    let named = |target: &str| {
        directives.iter().filter_map(|d| target_of(d)).any(|t| {
            t == target
                || t.starts_with(&format!("{target}::"))
                || target.starts_with(&format!("{t}::"))
        })
    };
    QUIET
        .iter()
        .filter(|(target, at)| *at < base && !named(target))
        .fold(filter.to_owned(), |filter, (target, at)| {
            format!("{filter},{target}={}", at.to_string().to_lowercase())
        })
}

#[cfg(not(target_arch = "wasm32"))]
fn init_tracing(
    log_dir: Option<&Path>,
    verbose: bool,
    telemetry_config: &TelemetryConfig,
) -> Result<()> {
    use tracing_subscriber::fmt;

    let filter = log_filter(if verbose { "debug" } else { "info" });

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

#[cfg(test)]
mod tests {
    use super::quieted;
    use crate::model::RsCaps;

    #[test]
    fn a_loud_base_quiets_naga_and_tarpc_and_never_makes_them_louder() {
        assert_eq!(quieted("debug"), "debug,naga=warn,tarpc=info");
        assert_eq!(quieted("trace"), "trace,naga=warn,tarpc=info");
        // At info, naga is held to warn; tarpc already is as quiet.
        assert_eq!(quieted("info"), "info,naga=warn");
        // A base at or under each quiet level gets nothing added.
        assert_eq!(quieted("warn"), "warn");
        assert_eq!(quieted("error"), "error");
        assert_eq!(quieted("off"), "off");
        // A filter that names a target keeps its say, whether the target
        // itself, one of its modules, or a span directive on it.
        assert_eq!(quieted("debug,naga=trace"), "debug,naga=trace,tarpc=info");
        assert_eq!(
            quieted("debug,naga::front=trace"),
            "debug,naga::front=trace,tarpc=info"
        );
        assert_eq!(
            quieted("debug,tarpc[RPC]=debug"),
            "debug,tarpc[RPC]=debug,naga=warn"
        );
        // A target that merely contains a quiet one's name is not it.
        assert_eq!(
            quieted("debug,naga_extra=info"),
            "debug,naga_extra=info,naga=warn,tarpc=info"
        );
        // Only targets and no base: nothing passes but them, so nothing to quiet.
        assert_eq!(quieted("runtime=debug"), "runtime=debug");
        // A bare target name with no level is a target, not the base.
        assert_eq!(quieted("debug,naga"), "debug,naga,tarpc=info");
        for filter in [
            "debug",
            "info",
            "warn",
            "debug,naga::front=trace",
            "debug,tarpc[RPC]=debug",
        ] {
            assert!(tracing_subscriber::EnvFilter::try_new(quieted(filter)).is_ok());
        }
    }

    /// What bootstrap reads of a load's chunk facts for the model's
    /// recurrent state: whether it holds any, and its RsCaps.
    fn read(facts: &::engine::chunk::ChunkFacts) -> (bool, RsCaps) {
        (
            facts.record_bytes > 0,
            RsCaps {
                state_size: facts.record_bytes,
                buffer_page_size: 0,
                window_state: false,
            },
        )
    }

    /// A load whose records take no bytes says the model holds no state:
    /// its RsCaps state nothing (no rs working set, no rs count a fire is
    /// held to). A hybrid's record bytes are its RsCaps' state size.
    #[test]
    fn the_model_s_record_bytes_say_whether_it_holds_state() {
        let mut facts = ::engine::chunk::ChunkFacts::default();
        let (dense, caps) = read(&facts);
        assert!(!dense && !caps.has_state(), "dense: no rs work");

        facts.record_bytes = 3000;
        let (hybrid, caps) = read(&facts);
        assert!(hybrid && caps.has_state());
        assert_eq!(caps.state_size, 3000, "a record's bytes, not the slot's");
    }
}

/// The tiers below the device suspended kv goes to, as a load stated them.
fn below_device(host: bool, disk: bool) -> &'static str {
    match (host, disk) {
        (true, true) => "the host, then the disk",
        (true, false) => "the host (no disk tier)",
        (false, true) => "the disk (no host tier)",
        (false, false) => "nowhere: no tier below the device",
    }
}
