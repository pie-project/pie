//! Create, destroy, and everything that happens once per model.
//!
//! Standing the shell up, reading a checkpoint onto the device, wiring trace
//! names onto checkpoint names, answering what the deployment can do, and
//! adopting a program — all before any fire, none of it again.

use super::state::{LoadedModel, Shell, retire};
use crate::fire::launch::sg_trace;
use crate::fire::scratch::Scratch;
use driver_api::CompletionBroker;
use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED, PIE_STATUS_INVALID_ARGUMENT,
};

/// Stand the shell up.
pub(crate) fn create_impl(config_bytes: &[u8], broker: CompletionBroker) -> Result<Shell, i32> {
    // The boot TOML rides in the bytes.
    let boot = std::str::from_utf8(config_bytes)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .unwrap_or_default();
    let boot_config = boot
        .get("model")
        .and_then(|m| m.get("config")?.as_str())
        .map(std::path::PathBuf::from);
    // One string: a model id, not a document. Optional — absent, the checkpoint
    // is identified from its tensors.
    let boot_model_id = boot
        .get("model")
        .and_then(|m| m.get("id")?.as_str())
        .map(str::to_owned);
    // Per-driver, not per-process, so one caller can opt into async completions
    // without deciding for every driver in the process. One parse; see `crate::boot`.
    let cfg = crate::boot::Boot::from_boot(Some(&boot));
    let runahead = cfg.runahead;
    // An unrecognised spelling is refused, not defaulted: quietly giving bf16 to
    // a caller who asked fp8 reads as a slightly worse model.
    let kv_format = match crate::layout::KvCacheFormat::from_name(
        boot.get("driver")
            .and_then(|d| d.get("kv_cache_dtype")?.as_str())
            .unwrap_or("auto"),
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[driver-cuda] create: {e:?}");
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
    };
    // The fire path's attention is FlashInfer's bf16 entry point, which ignores
    // the scheme, so a non-native format would be attended as bf16 — refused.
    if !kv_format.is_native_bf16() {
        eprintln!(
            "[driver-cuda] create: kv_cache_dtype '{}' can be written but not \
             read back — the fire path's attention is FlashInfer's bf16 entry \
             point, which ignores the scheme. Refusing rather than returning \
             garbage logits.",
            kv_format.name()
        );
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    let driver_u32 = |key: &str, default: u32| {
        boot.get("driver")
            .and_then(|d| d.get(key)?.as_integer())
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(default)
    };
    let calibrating = cfg.calibrating;
    let device_ordinal = boot
        .get("driver")
        .and_then(|d| d.get("device")?.as_integer())
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(0);
    let tp_size = driver_u32("tp_size", 1).max(1);
    let tp_rank = driver_u32("tp_rank", 0).min(tp_size - 1);
    // Bind the device on the thread that will fire: `cudaSetDevice` is
    // per-thread, so binding only in `load_model` would strand later calls on 0.
    if let Err(e) = crate::device::Device::bind(device_ordinal) {
        eprintln!("[driver-cuda] create: cannot bind CUDA device {device_ordinal}: {e}");
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }

    // The key a tensor-parallel group finds itself by; see `Shell::tp_group_id`.
    let tp_group_id = boot
        .get("driver")
        .and_then(|d| d.get("tp_group_id")?.as_str())
        .unwrap_or_default()
        .to_owned();

    // A group of more than one is still refused, naming what it waits on. See
    // `tp_serving_refusal`.
    if let Err(why) = tp_serving_refusal(tp_size, &tp_group_id) {
        eprintln!("[driver-cuda] create: {why}");
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    // Stated, not parsed from a `CAPS_JSON`: see `state::device_facts`.
    Ok(Shell {
        facts: super::state::device_facts(),
        boot_config,
        boot_model_id,
        runahead,
        boot: cfg.clone(),
        kv_format,
        cublas: None,
        lowerings: std::collections::BTreeMap::new(),
        calibrating,
        device_ordinal,
        preds: None,
        peel_win: None,
        logits_staging: None,
        retired_staging: Vec::new(),
        tp_rank,
        tp_size,
        model: None,
        load_generation: 0,
        programs: std::collections::BTreeMap::new(),
        instances: std::collections::BTreeMap::new(),
        next_id: 1,
        broker,
        fire_arrays: Scratch::default(),
        supergraph: crate::fire::recordings::Recordings::new(),
        all_reduce: None,
        tp_group_id,
        kv: None,
        gdn: None,
        channels: std::collections::BTreeMap::new(),
        swap: None,
        lora_arena: crate::fire::lora::LoraStageArena::default(),
        scratch: None,
        fire_stream: None,
        in_flight: std::collections::VecDeque::new(),
        fire_alloc: None,
        ptir: crate::program::Runtime::default(),
        ptir_control: None,
        ptir_rings: None,
        ptir_channel_slots: std::collections::BTreeMap::new(),
        ptir_sessions: std::collections::BTreeMap::new(),
        ptir_programs: crate::program::Programs::new(),
        ptir_plans: std::collections::BTreeMap::new(),
    })
}

/// World sizes the P2P plane's constructor (`CustomAllReduce::initialise`)
/// admits: `2 <= w <= 8`, `w % 2 == 0`.
const CONSTRUCTIBLE_WORLD_SIZES: &[u32] = &[2, 4, 6, 8];

/// The fused landing's world size — a single value: the fusion workspace is
/// built only for two (`Plane::can_fuse_residual_rmsnorm` refuses anything else).
const FUSED_WORLD_SIZE: u32 = 2;

/// Why this build refuses to serve a tensor-parallel group, or `Ok(())`.
///
/// The blocking conditions are read from [`kernels_cuda::comm::CAN_LAUNCH`],
/// not restated here. The ceilings: launchable device text, a
/// [`CONSTRUCTIBLE_WORLD_SIZES`] world size, and a `[driver] tp_group_id` —
/// without one the ranks never meet in `layout::rendezvous` and each builds a
/// plane pointing at its own memory, every reduction a silent no-op. The fused
/// landing is [`FUSED_WORLD_SIZE`] only, a warning: the plain reduction still
/// serves at 4, 6 and 8.
///
/// Not checked: that a reduction sums correctly. The per-message ceiling (8
/// MiB, the 16-byte multiple, the NCCL crossover) is `CustomAllReduce::can_handle`'s,
/// answered per fire, with `bind/arms/comm.rs` falling back rather than failing.
///
/// # Errors
///
/// The sentence to print, naming the condition and its value.
pub(crate) fn tp_serving_refusal(tp_size: u32, tp_group_id: &str) -> Result<(), String> {
    if tp_size <= 1 {
        return Ok(());
    }
    if !kernels_cuda::comm::CAN_LAUNCH {
        return Err(format!(
            "[driver] tp_size = {tp_size} is refused: `kernels_cuda::comm::CAN_LAUNCH` is false, \
             which is the launch half saying both `comm::all_reduce_bf16` and \
             `comm::all_reduce_residual_rmsnorm_bf16` decline every call. `cudarc`'s `nccl` \
             feature is off, so the `dist::` fallback has no bindings either. A rank that served \
             anyway would return its own shard as if it were the whole answer, with no error \
             anywhere -- which is the worst failure a driver has. What that constant means is \
             `kernels_cuda::comm`'s to say; this refusal does not restate it, because the \
             version that did went stale twice."
        ));
    }
    if !CONSTRUCTIBLE_WORLD_SIZES.contains(&tp_size) {
        return Err(format!(
            "[driver] tp_size = {tp_size} is refused: the vllm P2P plane is built only for world \
             sizes {CONSTRUCTIBLE_WORLD_SIZES:?} (`CustomAllReduce::initialise`), and vllm's \
             plain reduction is instantiated at exactly those \
             (`kernels_cuda::comm::PLAIN_NRANKS`). flashinfer's fused landing is instantiated at \
             {{2, 4, 8, 16}} and served at 2, which ceiling 3 below covers"
        ));
    }
    if tp_group_id.is_empty() {
        return Err(format!(
            "[driver] tp_size = {tp_size} is refused: [driver] tp_group_id is unset, and the \
             ranks of a group rendezvous on it. Without a key each rank would build a plane \
             pointing at its own memory and every reduction would silently be a no-op"
        ));
    }
    if tp_size != FUSED_WORLD_SIZE {
        eprintln!(
            "[driver-cuda] create: [driver] tp_size = {tp_size}: the FUSED landing \
             (`comm::all_reduce_residual_rmsnorm_bf16`) is world size {FUSED_WORLD_SIZE} only and \
             will decline at its first fire. The plain reduction serves at this width."
        );
    }
    Ok(())
}

/// Build this rank's P2P all-reduce plane, and publish it for the bind arms.
///
/// `group_devices` is gathered (`tp_host_allgather`, an in-process barrier —
/// TP ranks are threads), not configured, and must be real device ordinals,
/// never rank indices, or the second group on a multi-GPU box corrupts. The
/// fusion extents use the model's full hidden width, not this rank's shard:
/// each rank holds a partial sum of the whole vector, so the collective is a
/// sum, not a concatenation.
///
/// # Errors
///
/// The sentence to print; each failure names itself through `crate::error::Error`.
///
/// # Unexercised
///
/// Never run: the box has one GPU, so nothing here has met a second rank.
pub(crate) fn build_tp_plane(
    tp_size: u32,
    tp_rank: u32,
    tp_group_id: &str,
    device_ordinal: i32,
    fusion_max_tokens: i32,
    fusion_hidden: i32,
) -> Result<crate::fire::all_reduce::ResidentPlane, String> {
    use crate::fire::all_reduce::{Config, CustomAllReduce, HostAllgather, ResidentPlane};

    let world = i32::try_from(tp_size).map_err(|_| format!("tp_size {tp_size} is not an i32"))?;
    let rank = i32::try_from(tp_rank).map_err(|_| format!("tp_rank {tp_rank} is not an i32"))?;
    let gather = crate::layout::rendezvous::tp_host_allgather(world, tp_group_id, rank)
        .ok_or_else(|| {
            format!(
                "no rendezvous for rank {rank} of {world} on `{tp_group_id}` -- a group of one, \
                 an unset [driver] tp_group_id, or a rank outside its own group"
            )
        })?;

    // Round one: every rank's device ordinal, so `group_devices` is what the
    // ranks actually bound rather than what a config file claims.
    let mut ordinals = vec![0u8; 4 * tp_size as usize];
    gather(&device_ordinal.to_ne_bytes(), &mut ordinals);
    let group_devices: Vec<i32> = ordinals
        .chunks_exact(4)
        .map(|w| i32::from_ne_bytes([w[0], w[1], w[2], w[3]]))
        .collect();

    let ag = HostAllgather { rank, world_size: world, gather };
    let cfg = Config {
        same_process: true,
        group_devices,
        fusion_max_tokens,
        fusion_hidden,
        ..Config::default()
    };
    let car = CustomAllReduce::new(ag, &cfg).map_err(|e| format!("{e:?}"))?;
    Ok(ResidentPlane::publish(car))
}

/// Teardown, as a destructor.
impl Drop for Shell {
    fn drop(&mut self) {
        let shell = self;
        // Every queued fire first: a fire still on the stream runs its callback
        // against a `ChannelState` the frees below would reclaim underneath it.
        for fire in std::mem::take(&mut shell.in_flight) {
            let _ = fire.done.synchronize();
            retire(fire);
        }
        for ch in shell.channels.values() {
            ch.free();
        }
        if let Some(swap) = &shell.swap {
            swap.free();
        }
        // The driver owns the handle; `CublasHandle` asserts it was released,
        // not dropped.
        if let Some(mut h) = shell.cublas.take() {
            h.release(&mut crate::device::cublas::LiveCublas);
        }
        if let Some(mut scratch) = shell.scratch.take() {
            let mut sops = crate::fire::attention_workspace::LiveStagingOps;
            scratch.ws.release(&mut sops);
            // `AttentionWorkspace` has no working `Drop` (CUDA calls need
            // `&mut O`), so an unreleased workspace is a pinned-host leak.
            scratch.prefill_ws.release(&mut sops);
            // The peel tail's own workspace, released for the same reason.
            scratch.tail_ws.release(&mut sops);
            drop(scratch.decode_plan);
            drop(scratch.decode_plan_full);
            drop(scratch.prefill_plan);
            drop(scratch.tail_plan);
        }
    }
}

/// The load itself; `i32` errors are the ABI's status codes.
pub(crate) fn load_impl(state: &mut Shell, snapshot: &std::path::Path) -> Result<(), i32> {
    use model_loader::checkpoint::read::{parse_checkpoint_metadata, read_meta};

    let meta = parse_checkpoint_metadata(snapshot)
        .map_err(|e| crate::Error::invalid("load_model: checkpoint parse", format!("{e:?}")))?;

    // The checkpoint's own `config.json`: embedded, else the boot TOML's path.
    // Only the declared quantization is read out — a catalog row can't state it.
    let config_json = match read_meta(&meta, model::encoding::CONFIG_OBJECT) {
        Ok(Some(bytes)) => {
            String::from_utf8(bytes).map_err(|e| i32::from(crate::Error::from(e)))?
        }
        Ok(None) => {
            let Some(path) = &state.boot_config else {
                return Err(crate::Error::unsupported(
                    "load_model",
                    "no embedded model/config and no [model] config in \
                     the boot TOML",
                )
                .into());
            };
            std::fs::read_to_string(path).map_err(|e| i32::from(crate::Error::from(e)))?
        }
        Err(e) => {
            return Err(crate::Error::invalid("load_model: read_meta", format!("{e:?}")).into());
        }
    };

    // Which model this is, asked of the tensors, not the config: identification
    // and validation are one operation — a checkpoint is a known model or not.
    let chosen = state
        .boot_model_id
        .as_ref()
        .map_or(model::catalog::Override::None, |id| {
            model::catalog::Override::Id(id.clone())
        });
    let row = model::catalog::identify(&meta, &chosen)
        .map_err(|e| crate::Error::unsupported("load_model: identify", e.to_string()))?;

    // How the numbers are stored — not part of what model this is.
    let encoding = model::encoding::Encoding::from_config_json(&config_json)
        .map_err(|e| crate::Error::invalid("load_model: config", e.to_string()))?;

    // The load is `model-loader`'s plan, executed onto the device: it decides
    // which encodings are loadable (a transform outside `CUDA_TILE_MAP_MASK` is
    // refused at compile, not mis-bound at launch) and which projections fuse.
    let target = crate::weights::plan::cuda_storage_target(state.tp_rank, state.tp_size);
    let (plan, _moe) =
        crate::weights::plan::compile_load_plan_for(snapshot, &meta, &target, row, &encoding)
            .map_err(|e| crate::Error::unsupported("load_model: load plan", e))?;
    let alloc = crate::device::Allocator::new();
    // `Error::from` spelled out because `?` won't chain two of them: the orphan
    // rule forbids `From<LoaderError> for i32` on a primitive this crate lacks.
    let staged = crate::weights::stage::stage_plan_weights(&plan, snapshot, &alloc)
        .map_err(crate::Error::from)?;

    let mut model = LoadedModel {
        id: row.id(),
        // Filled below once the checkpoint view exists; it reads the weight map.
        deployment: model::deployment::Deployment::empty(),
        load_caps: Vec::new(),
        weights: staged.spans,
        owned: staged.owned,
        aliases: std::collections::BTreeMap::new(),
        layer_scalars: Vec::new(),
        tp_size: state.tp_size,
    };
    wire_trace_names(&mut model);

    // Once, at load; see `LoadedModel::deployment`. From the matched row.
    model.deployment = row
        .deployment(model::catalog::Deployed {
            // The row serves both backends; the caller states which asks.
            backend: model::catalog::Backend::Cuda,
            tp_size: state.tp_size,
            layer_scalars: &model.layer_scalars,
        })
        .map_err(|e| i32::from(crate::Error::from(e)))?;

    // A KV shape this shell has no pool for is refused at load, not at first
    // fire: the `match` on `KvStyle` cannot forget a variant. MLA and
    // compressed planes are ported but have no forward path yet.
    if let Some(what) = match &model.deployment.kv {
        model::deployment::KvStyle::Paged => None,
        model::deployment::KvStyle::Mla { .. } => Some(
            "this checkpoint attends through a latent ckv/kpe pair, which this \
             driver does not build — `pools::mla_cache` is ported and has \
             no forward path to serve",
        ),
        model::deployment::KvStyle::CompressedPlane { .. } => Some(
            "this checkpoint attends through a compressed KV plane, which this \
             driver does not build — `pools::compressed_plane_cache` is ported \
             and has no forward path to serve",
        ),
    } {
        return Err(crate::error::Error::Unsupported {
            what: what.to_string(),
        }
        .into());
    }

    // The GQA ratio, refused at load: FlashInfer's decode instantiates a fixed
    // set of group sizes and reports anything else by throwing, and a throw
    // across the C ABI is undefined behaviour. `DECODE_GQA_GROUPS` is this build's.
    model
        .deployment
        .servable_by(super::DECODE_GQA_GROUPS)
        .map_err(|why| -> i32 {
            // `servable_by` distinguishes a fractional ratio from an
            // uninstantiated one, so a malformed shape isn't a missing kernel.
            crate::error::Error::Unsupported {
                what: format!(
                    "{why}: {} q heads over {} kv heads; this build \
                     instantiates {:?}",
                    model.deployment.shape.q_heads,
                    model.deployment.shape.kv_heads,
                    super::DECODE_GQA_GROUPS,
                ),
            }
            .into()
        })?;

    // A new residency drops every cache keyed on the old one. Replacing
    // `state.model` frees the previous weight arena a captured graph baked
    // addresses into, so a same-depth reload would replay into freed memory.
    state.load_generation += 1;
    state.lowerings.clear();
    state.supergraph = crate::fire::recordings::Recordings::new();
    // And the P2P plane: the fusion workspace's `hidden` is compared for
    // equality, so a plane built for the previous checkpoint declines (or
    // mis-reduces) the next one's fused fires. Safe to drop — ranks rebuild.
    state.all_reduce = None;
    state.model = Some(model);
    // After the model is stored: a calibration boot fires the ordinary path.
    let caps = capabilities_json(state, snapshot)?;
    state.model.as_mut().expect("just stored").load_caps = caps;

    // The P2P plane, built once here — not in `create`: its stride is baked at
    // construction, so it needs the checkpoint's `hidden` and the advertised
    // token ceiling (hence after the caps). These addresses are what every
    // peer's mapping points at, so a rebuild mid-serve is unagreed. Needs two GPUs.
    if state.tp_size > 1 && state.all_reduce.is_none() {
        let hidden = state
            .model
            .as_ref()
            .and_then(|m| i32::try_from(m.deployment.shape.hidden).ok())
            .unwrap_or(0);
        let max_tokens = state
            .model
            .as_ref()
            .and_then(|m| {
                serde_json::from_slice::<driver_api::DriverCapabilities>(&m.load_caps).ok()
            })
            .and_then(|c| i32::try_from(c.max_forward_tokens).ok())
            .unwrap_or(0);
        match build_tp_plane(
            state.tp_size,
            state.tp_rank,
            &state.tp_group_id,
            state.device_ordinal,
            max_tokens,
            hidden,
        ) {
            Ok(plane) => state.all_reduce = Some(plane),
            Err(why) => {
                return Err(i32::from(crate::Error::invalid(
                    "load_model: tensor-parallel plane",
                    why,
                )));
            }
        }
    }

    if state.calibrating {
        calibrate_planner(state);
    }
    Ok(())
}

/// The calibration sweep: time the reachable fire shapes and cache the fastest
/// for the next boot. `[batching] calibrate_planner` turns it on. Runs at load,
/// after weights and caps, because a probe fires the ordinary path. Failures
/// are swallowed: a boot that cannot measure still serves analytically.
fn calibrate_planner(state: &mut Shell) {
    use crate::layout::calibrate::{Ceiling, Point, StepTimer, sweep};
    use crate::serve::state::{InstanceEntry, ProgramEntry};

    let Some(model) = state.model.as_ref() else {
        return;
    };
    // The ceiling is what the driver just advertised: sweeping above it would
    // measure shapes no scheduler will send.
    let Ok(caps) = serde_json::from_slice::<driver_api::DriverCapabilities>(&model.load_caps)
    else {
        eprintln!("[driver-cuda] calibrate: the caps did not parse; nothing to sweep around");
        return;
    };
    let ceiling = Ceiling {
        max_forward_tokens: i32::try_from(caps.max_forward_tokens).unwrap_or(0),
        max_forward_requests: i32::try_from(caps.max_forward_requests).unwrap_or(0),
    };
    if ceiling.max_forward_tokens < 1 || ceiling.max_forward_requests < 1 {
        eprintln!("[driver-cuda] calibrate: the advertised rectangle is empty");
        return;
    }
    let page_size = i32::try_from(caps.kv_page_size).unwrap_or(16).max(1);
    let total_pages = caps.total_pages;
    let template = crate::layout::profile_key::ProfileShape {
        policy_profile: "generic".to_owned(),
        kv_page_size: page_size,
        max_forward_tokens: ceiling.max_forward_tokens,
        max_forward_requests: ceiling.max_forward_requests,
        budget_bytes: 0,
    };

    /// Times one point by firing a synthetic batch of that shape. `None` for a
    /// shape it cannot fire — the ladder starts at the ceiling, likeliest to decline.
    struct FireTimer<'a> {
        state: &'a mut Shell,
        instances: Vec<u64>,
        page_size: i32,
        total_pages: u32,
    }
    impl StepTimer for FireTimer<'_> {
        fn step_ms(&mut self, point: Point) -> Option<f64> {
            let t = std::time::Instant::now();
            match synthetic_fire(
                self.state,
                point,
                &self.instances,
                self.page_size,
                self.total_pages,
            ) {
                Ok(()) => Some(t.elapsed().as_secs_f64() * 1e3),
                Err(status) => {
                    eprintln!(
                        "[driver-cuda] calibrate: N={} R={} declined ({status})",
                        point.max_forward_tokens, point.max_forward_requests
                    );
                    None
                }
            }
        }
    }

    let key = crate::layout::profile_key::ProfileKey {
        gpu_name: String::new(),
        compute_major: 0,
        compute_minor: 0,
        sm_count: 0,
        kv_cache_dtype: state.kv_format.name().to_owned(),
        tp_size: i32::try_from(state.tp_size).unwrap_or(1),
        // The row, not its family: keyed on the id keys the model; the four
        // extents below guard against a row that moved under a stale file.
        model_type: model.id.to_owned(),
        hidden_size: i32::try_from(model.deployment.shape.hidden).unwrap_or(0),
        num_hidden_layers: i32::try_from(model.deployment.layers).unwrap_or(0),
        num_attention_heads: i32::try_from(model.deployment.shape.q_heads).unwrap_or(0),
        num_key_value_heads: i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
        head_dim: i32::try_from(model.deployment.shape.head_dim_alloc()).unwrap_or(0),
    };

    // ONE probe program and as many instances as the widest point needs.
    // Registration is a map insert, so the full set up front costs nothing.
    let probe_program = state.next_id;
    state.next_id += 1;
    state.programs.insert(
        probe_program,
        ProgramEntry {
            program_hash: 0,
            emitter_version: 0,
        },
    );
    let mut instances = Vec::with_capacity(ceiling.max_forward_requests as usize);
    for _ in 0..ceiling.max_forward_requests {
        let id = state.next_id;
        state.next_id += 1;
        state.instances.insert(
            id,
            InstanceEntry {
                program_id: probe_program,
                geometry_class: driver_api::local::PIE_GEOMETRY_CLASS_HOST,
                channel_ids: Vec::new(),
                seeds: Vec::new(),
            },
        );
        instances.push(id);
    }

    let mut timer = FireTimer {
        state,
        instances,
        page_size,
        total_pages,
    };
    let outcome = sweep(ceiling, &template, &mut timer);

    // The probe leaves nothing behind: its instances hold KV pages and its
    // program is not one the engine registered.
    for id in &timer.instances {
        state.instances.remove(id);
    }
    state.programs.remove(&probe_program);

    let Some(cal) = outcome else {
        eprintln!(
            "[driver-cuda] calibrate: no point on the ladder could be fired, so \
             there is nothing to record — the analytic pick stands"
        );
        return;
    };
    for s in &cal.samples {
        eprintln!(
            "[driver-cuda] calibrate: N={} R={} -> {:.2} ms (+/- {:.2}), {:.0} tok/s",
            s.max_forward_tokens,
            s.max_forward_requests,
            s.step_ms,
            s.step_ms_stddev,
            s.tokens_per_s
        );
    }
    eprintln!(
        "[driver-cuda] calibrate: winner N={} R={}",
        cal.shape.max_forward_tokens, cal.shape.max_forward_requests
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0));
    match crate::layout::profile_cache::ProfileCache::discover("") {
        Ok(c) => match c.store(&key, &cal.shape, &cal.samples, now) {
            Ok(()) => eprintln!("[driver-cuda] calibrate: stored at {}", c.path().display()),
            Err(e) => eprintln!("[driver-cuda] calibrate: the winner could not be stored: {e:?}"),
        },
        Err(e) => eprintln!("[driver-cuda] calibrate: no cache directory: {e:?}"),
    }
}

/// Fire one synthetic batch shaped like `point` and wait for it to retire.
///
/// `R` requests of `tokens_per_request` tokens, each a fresh prefill over its
/// own pages — both axes at once, the worst case. Synchronous by design
/// (`owes: None`): a fire that returned early would time the enqueue, not the step.
fn synthetic_fire(
    state: &mut Shell,
    point: crate::layout::calibrate::Point,
    instances: &[u64],
    page_size: i32,
    total_pages: u32,
) -> Result<(), i32> {
    let reqs = usize::try_from(point.max_forward_requests)
        .unwrap_or(0)
        .min(instances.len());
    let per = usize::try_from(point.tokens_per_request())
        .unwrap_or(1)
        .max(1);
    if reqs == 0 {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    let page = usize::try_from(page_size).unwrap_or(16).max(1);
    // Pages this batch needs if every request runs alone. A footprint past the
    // pool is not a candidate — cheaper to refuse here than in the allocator.
    let pages_each = per.div_ceil(page);
    let pages_total = reqs * pages_each;
    if pages_total > usize::try_from(total_pages).unwrap_or(0) {
        return Err(PIE_STATUS_EXHAUSTED);
    }

    let mut roster_rows = Vec::with_capacity(reqs);
    let mut sub_batch_indptr = Vec::with_capacity(reqs + 1);
    let mut sub_batch_class = Vec::with_capacity(reqs);
    let mut token_ids = Vec::with_capacity(reqs * per);
    let mut position_ids = Vec::with_capacity(reqs * per);
    let mut kv_page_indices = Vec::with_capacity(pages_total);
    let mut kv_page_indptr = Vec::with_capacity(reqs + 1);
    let mut kv_last_page_lens = Vec::with_capacity(reqs);
    let mut qo_indptr = Vec::with_capacity(reqs + 1);
    let mut cells = Vec::with_capacity(reqs);

    sub_batch_indptr.push(0u32);
    kv_page_indptr.push(0u32);
    qo_indptr.push(0u32);
    for r in 0..reqs {
        roster_rows.push(u32::try_from(r).unwrap_or(0));
        sub_batch_indptr.push(u32::try_from(r + 1).unwrap_or(0));
        sub_batch_class.push(driver_api::local::PIE_GEOMETRY_CLASS_HOST);
        for t in 0..per {
            // Token zero for every row: the sweep times a shape and every
            // kernel is dense, so the token in a slot changes nothing.
            token_ids.push(0);
            position_ids.push(u32::try_from(t).unwrap_or(0));
        }
        for p in 0..pages_each {
            kv_page_indices.push(u32::try_from(r * pages_each + p).unwrap_or(0));
        }
        kv_page_indptr.push(u32::try_from((r + 1) * pages_each).unwrap_or(0));
        let tail = per - (pages_each - 1) * page;
        kv_last_page_lens.push(u32::try_from(tail).unwrap_or(1));
        qo_indptr.push(u32::try_from((r + 1) * per).unwrap_or(0));
        cells.push(driver_api::local::TerminalCell::pending());
    }
    let cell_ptrs: Vec<*mut driver_api::local::TerminalCell> =
        cells.iter_mut().map(|c| c as *mut _).collect();

    let step = driver_api::StepSubmission {
        plan: driver_api::LaunchPlan {
            token_ids,
            position_ids,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            qo_indptr,
            ..Default::default()
        },
        roster_rows,
        sub_batch_indptr,
        sub_batch_class,
        terminal_cells: cell_ptrs,
        ..Default::default()
    };
    let frame = driver_api::FrameSubmission {
        instance_ids: instances[..reqs].to_vec(),
        required_kv_pages: u32::try_from(pages_total).unwrap_or(0),
        steps: vec![step],
        ..Default::default()
    };
    let step = &frame.steps[0];
    crate::fire::launch::step_impl(state, &frame, step, None)
}

/// Answer the trace names a launch will ask for, from `model`'s tables. Most
/// naming is family knowledge (`model::shared::weight_names`); two need the driver.
///
/// Whether a join is a rename or nothing: a checkpoint shipping pre-joined
/// projections (Phi-3) has its contract split them, so the halves are adjacent
/// in the arena (the plan wrote them in file order). The driver checks
/// contiguity rather than assumes: a GEMM handed a discontiguous operand reads
/// what lies between.
///
/// And reading a load-time scalar to the host: gemma-4's `layer_scalar` is one
/// bf16 on device; `model` says which and in what order.
fn wire_trace_names(model: &mut LoadedModel) {
    let Some(row) = model::catalog::find(model.id) else {
        return; // no row, no names; the load already refused
    };
    let published: Vec<String> = model.weights.keys().cloned().collect();
    let set: std::collections::BTreeSet<&str> = published.iter().map(String::as_str).collect();
    let has = |n: &str| set.contains(n);
    let wiring = model::shared::weight_names::wire(row.load_shape(), &has);

    for (trace, name) in wiring.aliases {
        model.aliases.insert(trace, name);
    }
    for (trace, parts) in wiring.joins {
        let mut spans = Vec::with_capacity(parts.len());
        for p in &parts {
            let Some(span) = model.weights.get(p).copied() else {
                spans.clear();
                break;
            };
            spans.push(span);
        }
        if spans.len() != parts.len() {
            continue;
        }
        let abut = spans.windows(2).all(|p| {
            std::ptr::eq(
                p[0].ptr.wrapping_byte_add(p[0].bytes).cast_const(),
                p[1].ptr.cast_const(),
            )
        });
        if abut {
            model.weights.insert(
                trace,
                crate::weights::stage::WeightSpan {
                    ptr: spans[0].ptr,
                    bytes: spans.iter().map(|s| s.bytes).sum(),
                },
            );
        }
    }
    model.layer_scalars = wiring
        .scalars
        .iter()
        .map(|n| {
            model
                .weights
                .get(n)
                .map_or(1.0f32, |b| match crate::weights::stage::read_span(*b) {
                    Ok(back) if back.len() == 2 => {
                        f32::from_bits(u32::from(u16::from_le_bytes([back[0], back[1]])) << 16)
                    }
                    _ => 1.0,
                })
        })
        .collect();
}

/// What `load_model` answers: a `driver_api::DriverCapabilities` document.
///
/// The KV pool is sized here. `total_pages` is what a scheduler admits against.
/// The budget (`memory_planner::budget_for`) is measured after weights are
/// resident, so `cudaMemGetInfo`'s free figure already subtracts them. It
/// covers the KV pool and the fire's activations; the activation share is a
/// fifth (the C++'s rule of thumb), not a computed arena, because no fire has
/// been lowered yet.
fn capabilities_json(state: &mut Shell, snapshot: &std::path::Path) -> Result<Vec<u8>, i32> {
    use crate::layout::memory_planner::{
        DeviceMemory, DeviceProps, ModelCosts, ModelShape, NoProfiles, PlannerConfig,
        ProfileSource, ShapeKnees, plan,
    };
    use crate::layout::model_costs::{CheckpointCosts, DiskProfiles};

    let model = state.model.as_ref().expect("the model is stored");
    // No `HfConfig` read: a program and this driver fire off the same row, so
    // they can't be two models. Cloned because the planner wants `state` back.
    let deployment = model.deployment.clone();
    let model_tp = model.tp_size;
    // Copied out (`&'static str`) to end the borrow of `state` before the planner.
    let model_id = model.id;
    let device = crate::device::Device::bind(state.device_ordinal)?;
    let (free, total) = device.memory_info()?;
    let (major, minor) = device.compute_capability().unwrap_or((0, 0));
    let cfg = PlannerConfig {
        gpu_mem_utilization: 0.90,
        memory_profile: "auto".to_owned(),
        max_forward_tokens: 0,
        max_forward_requests: 0,
        // Pinned, a coupling not a preference: the fire path builds 16-token
        // pages, so sweeping page sizes would answer a geometry it never builds.
        kv_page_size: 16,
        // The driver's own format, so the planner sizes the pages it allocates:
        // a bf16 planner under-counts a quantized cache up to 4x.
        kv_cache_dtype: state.kv_format.name().to_owned(),
        tp_size: i32::try_from(model_tp).unwrap_or(1),
        mtp_num_drafts: 0,
        // False even when calibrating: the attention workspace is a fixed 32 MB
        // and a fire wider than it supports fails inside CUDA rather than
        // returning a status, so the sweep stays at the reachable shapes.
        calibrating: false,
        rs_slot_mult: 1,
        // The key both rendezvous read: `String::new()` reads as "no group",
        // leaving cross-rank plan agreement inert — a deadlock risk.
        nccl_unique_id_hex: state.tp_group_id.clone(),
    };
    // The row's own numbers, so the pool the planner sizes and the one the fire
    // builds share one shape — a second parse could leave the pool pages short.
    let costs = CheckpointCosts::new(&deployment, model_tp);
    let shape = ModelShape {
        hidden_size: i32::try_from(deployment.shape.hidden).unwrap_or(0),
        num_hidden_layers: i32::try_from(deployment.layers).unwrap_or(0),
        num_attention_heads: i32::try_from(deployment.shape.q_heads).unwrap_or(0),
        num_key_value_heads: i32::try_from(deployment.shape.kv_heads).unwrap_or(0),
        head_dim_kernel: i32::try_from(deployment.shape.head_dim_alloc()).unwrap_or(0),
        model_id: model_id.to_owned(),
    };
    let props = DeviceProps {
        name: String::new(),
        major,
        minor,
        sm_count: device.sm_count().unwrap_or(0),
    };
    let mem = DeviceMemory {
        free_bytes: free as u64,
        total_bytes: total as u64,
    };
    // A measurement beats the score: `DiskProfiles` reads the calibration
    // cache; no file is a miss, and the planner falls back to the analytic pick.
    let disk = DiskProfiles::discover("").ok();
    let profiles: &dyn ProfileSource = disk.as_ref().map_or(&NoProfiles, |d| d);
    let planned = plan(
        &cfg,
        &shape,
        &props,
        mem,
        ShapeKnees::default(),
        &costs,
        profiles,
    )
    .map_err(|e| {
        eprintln!("[driver-cuda] load_model: memory planner: {e:?}");
        PIE_STATUS_EXHAUSTED
    })?;
    for note in &planned.notes {
        eprintln!("[driver-cuda] {note}");
    }

    // Pages against what the arena leaves, not the full budget: `Scratch` keeps
    // the arena, seam buffers and descriptor arrays for the driver's life — a
    // captured graph bakes their addresses and a freed arena can't be replayed
    // into. Both resident allocations share one budget; charging one over-counts.
    let per_page = planned.plan.kv_page_bytes.max(1);
    let resident_arena = costs
        .arena_bytes(
            planned.plan.capacity.max_forward_tokens,
            planned.plan.capacity.max_logit_rows,
            0,
        )
        .saturating_add(planned.plan.attn_float_workspace_bytes)
        .saturating_add(planned.plan.persistent_input_bytes)
        .saturating_add(planned.plan.runtime_quant_scratch_bytes);
    let total_pages = planned.budget.saturating_sub(resident_arena) / per_page;

    let caps = driver_api::DriverCapabilities {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        total_pages: u32::try_from(total_pages).unwrap_or(u32::MAX),
        kv_page_size: u32::try_from(planned.plan.kv_page_size).unwrap_or(16),
        // The lattice's answer, not a stated ceiling: the arena is sized for
        // exactly this rectangle — a wider fire has no workspace.
        max_forward_tokens: u32::try_from(planned.plan.capacity.max_forward_tokens).unwrap_or(0),
        max_forward_requests: u32::try_from(planned.plan.capacity.max_forward_requests)
            .unwrap_or(0),
        max_page_refs: u32::try_from(planned.plan.capacity.max_page_refs).unwrap_or(0),
        arch_name: deployment.advertised.arch.to_owned(),
        // The catalog id, which `arch_name` above is not: `qwen3` names
        // twelve checkpoints of six shapes, `qwen3-0.6b` names one.
        model_id: model_id.to_owned(),
        vocab_size: deployment.shape.vocab,
        max_model_len: deployment.advertised.max_model_len,
        activation_dtype: "bf16".to_owned(),
        hidden_size: deployment.shape.hidden,
        rs_cache_required: costs.has_linear_state(),
        snapshot_dir: snapshot.display().to_string(),
        // No swap pool, elastic accounting, MTP, value head, or sink yet. Each
        // is a claim a program binds against, so a false one is a silent no-op.
        swap_pool_size: 0,
        kv_copy_domain_mask: 0,
        rs_cache_slots: 0,
        rs_cache_slot_bytes: costs.state_slot_bytes(),
        elastic_page_bytes: 0,
        elastic_budget_pages: 0,
        has_mtp_logits: false,
        has_mtp_drafts: false,
        has_value_head: false,
        has_kv_envelopes: false,
        has_attn_score: false,
        has_attn_page_mask: false,
        // True, made safe by the region table: a lane that does not resolve
        // degrades to the arm's no-op, not a wrong answer.
        has_lora: true,
        model_site_summary: driver_api::ModelSiteSummary::default(),
        // Exactly the three ports `fire::envelope::compose` reads —
        // `EmbedTokens | Positions | KvLen`; the rest a decode derives from the
        // positions. `DEVICE_GEOMETRY_PORTS` is deliberately absent: it wins the
        // pool-owned class this driver does not build. At 0 the decode is `Host`.
        device_geometry_port_mask: driver_api::PIE_DECODE_ENVELOPE_PORTS,
        // False, a different question from the mask: it only matters for the
        // pool-owned `devgeo` class this driver does not claim, so a decode
        // envelope is unaffected. Saying true would advertise a refused class.
        resolves_geometry_per_step: false,
        // True when this checkpoint has a tower `pie_cuda_encode` serves. A
        // false negative makes the worker build no encode executor, so gemma-4's
        // towers go unreachable. Qwen3-VL is deliberately absent — its tower
        // writes the fire's hidden rows in-fire, not through encode.
        supports_media_encode: deployment.advertised.media_encode,
        kv_handle: None,
        // This driver compiles its own PTIR through NVRTC; nothing upstream
        // needs to generate a kernel for it.
        codegen_backend: String::new(),
    };
    serde_json::to_vec(&caps).map_err(|_| PIE_STATUS_DRIVER_ERROR)
}

/// Adopt one non-empty launch package and compile what it generates.
pub(crate) fn adopt_and_compile(
    state: &mut Shell,
    id: u64,
    desc: &driver_api::ProgramRegistration,
    package: driver::driver_api::plan::LaunchPackage,
    kernels: &[driver_api::EmittedKernel],
) -> Result<(), i32> {
    let plan = driver::adopt_launch_package(package)
        .map_err(|error| crate::Error::unsupported("register_program", error))?;

    // The compile, when there is a device to compile for (`load_model` binds
    // it). A registration arriving first is fine — guessing arch mis-targets.
    if plan.executable && state.model.is_some() {
        let target = ptir_target(state.device_ordinal)?;
        let versions = driver::Versions::from_compiler(desc.emitter_version);
        match state
            .ptir
            .compile(desc.program_hash, &plan, kernels, versions, target)
        {
            Ok(compiled) => {
                sg_trace(|| {
                    format!(
                        "  ptir program {:#018x}: {} stage(s) compiled",
                        desc.program_hash,
                        plan.stages.len()
                    )
                });
                state.ptir_programs.insert(id, compiled);
            }
            Err(failure) => {
                return Err(crate::Error::unsupported(
                    "register_program",
                    format_args!(
                        "cannot compile program {:#018x}: {}",
                        desc.program_hash,
                        failure.reason()
                    ),
                )
                .into());
            }
        }
    } else if !plan.executable {
        // Recorded rather than refused: a launch needing this unexecutable plan
        // must be able to report why, so the reason is kept.
        eprintln!(
            "[driver-cuda] register_program: program {:#018x} adopted but is \
             not executable by this driver: {}",
            desc.program_hash,
            plan.reject_reason.as_deref().unwrap_or("no reason given")
        );
    }

    state.ptir_plans.insert(id, plan);
    Ok(())
}

/// What the compile cache needs to know about the GPU it is compiling for.
///
/// Read per registration, not cached: the cheap numbers stay fresh, and the
/// NVRTC version would go stale against a runtime swap.
pub(crate) fn ptir_target(ordinal: i32) -> Result<crate::program::Target, i32> {
    // The CUDA error already is one of ours — `Device::bind` and
    // `compute_capability` return `crate::Error::Driver`.
    let device = crate::device::Device::bind(ordinal)?;
    let (major, minor) = device.compute_capability()?;
    let nvrtc = crate::program::compile::version().map_err(|error| {
        eprintln!("[driver-cuda] register_program: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    Ok(crate::program::Target {
        major,
        minor,
        // The ordinal, widened: a stable per-GPU id, and with one device bound
        // per process the ordinal is that id.
        device: u64::try_from(device.ordinal()).unwrap_or(0),
        nvrtc,
    })
}

#[cfg(test)]
mod tests {
    use super::{CONSTRUCTIBLE_WORLD_SIZES, tp_serving_refusal};

    #[test]
    fn a_single_rank_needs_no_collective_and_is_never_refused() {
        assert!(tp_serving_refusal(1, "").is_ok(), "one rank reduces nothing");
        assert!(tp_serving_refusal(0, "").is_ok(), "and neither does a mis-stated zero");
    }

    /// Asserts on the message content, not just `is_err`: the first ceiling that
    /// bites must name the value it read.
    #[test]
    fn a_group_is_refused_by_the_condition_that_actually_blocks_it() {
        // A two-rank group's missing piece is the key, and the refusal says so.
        let why = tp_serving_refusal(2, "").expect_err("a group with no key cannot rendezvous");
        assert!(why.contains("tp_group_id"), "names the value it read: {why}");
        assert!(
            !why.contains("CAN_LAUNCH"),
            "there IS device text, so that is not what blocks it: {why}"
        );
        assert!(
            !why.contains("no `CustomAllReduce` handle to pass"),
            "the claim that had already gone stale: {why}"
        );
        // And a fully configured two-rank group is admitted.
        assert!(
            tp_serving_refusal(2, "group-a").is_ok(),
            "two ranks with a key is what `CAN_LAUNCH` being true admits"
        );
    }

    /// A world size nothing can build a plane for is refused, and the message
    /// names the set rather than the value.
    #[test]
    fn a_world_size_no_plane_can_be_built_for_is_refused() {
        let odd = tp_serving_refusal(3, "group-a").expect_err("three is refused");
        assert!(
            odd.contains(&format!("{CONSTRUCTIBLE_WORLD_SIZES:?}")),
            "the ceiling is named rather than the number that missed it: {odd}"
        );
        assert!(tp_serving_refusal(9, "group-a").is_err(), "and so is nine");
    }

    /// The ceilings are still stated, and the two sets they are drawn from
    /// are asked of the crate that owns them rather than copied.
    #[test]
    fn the_ceilings_are_written_down() {
        assert_eq!(
            CONSTRUCTIBLE_WORLD_SIZES,
            &[2, 4, 6, 8],
            "`CustomAllReduce::initialise` takes 2..=8 even, and nothing else"
        );
        // The plain reduction is instantiated at exactly the constructible
        // set, which is what lets one constant answer both.
        for &size in CONSTRUCTIBLE_WORLD_SIZES {
            assert!(
                kernels_cuda::comm::PLAIN_NRANKS.contains(&i32::try_from(size).unwrap()),
                "the plane can be built at {size} and vllm has no kernel for it"
            );
        }
        // The fused landing's set is a different one; six shows it: the plane
        // builds and the plain kernel exists, but flashinfer instantiated no
        // fused launcher for it.
        assert!(!kernels_cuda::comm::NRANKS.contains(&6));
        assert!(kernels_cuda::comm::NRANKS.contains(&16));
        assert!(!CONSTRUCTIBLE_WORLD_SIZES.contains(&16));
    }
}
