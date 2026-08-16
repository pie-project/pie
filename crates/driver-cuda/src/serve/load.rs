//! Create, destroy, and everything that happens once per model.
//!
//! The verbs behind four of the thirteen exports: standing the shell up,
//! reading a checkpoint onto the device, wiring trace names onto checkpoint
//! names, answering what this deployment can do, and adopting a program.
//! All of it happens before any fire; none of it happens again.

use super::state::{LoadedModel, Shell, retire};
use crate::fire::launch::sg_trace;
use crate::fire::scratch::Scratch;
use driver_api::CompletionBroker;
use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED, PIE_STATUS_INVALID_ARGUMENT,
};

/// Stand the shell up.
///
/// It took a `*const PieDriverCreateDesc` and answered a `*mut PieDriver`
/// plus a JSON blob through a second out-parameter, and every refusal below
/// returned a bare null — a caller learned WHICH of nine causes only from
/// stderr. It takes the bytes and the broker now, and answers `Result`.
pub(crate) fn create_impl(config_bytes: &[u8], broker: CompletionBroker) -> Result<Shell, i32> {
    // The boot TOML rides in the bytes. Three keys are read today:
    // `[model] id`, `[model] config` and `[driver] runahead`.
    let boot = std::str::from_utf8(config_bytes)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .unwrap_or_default();
    let boot_config = boot
        .get("model")
        .and_then(|m| m.get("config")?.as_str())
        .map(std::path::PathBuf::from);
    // WHAT THE PROCESS BOUNDARY CARRIES.
    //
    // One string. It used to carry a `pie.model/1` JSON document — the
    // worker wrote it beside the boot TOML and named the PATH here, and
    // this driver parsed it back with its own reader while
    // `driver-metal` parsed the same document with a DIFFERENT reader
    // into a different struct, under a different failure policy (facts
    // swallowed a missing field with a default; the descriptor refused).
    // One document, two readers, two answers.
    //
    // An id cannot do that. Both drivers link the same `const` table, so
    // the worst a bad id can do is fail to resolve — loudly, at the
    // door, with the nearest ids named. And it is OPTIONAL rather than
    // required: absent, the driver identifies the checkpoint from its
    // TENSORS, which is the answer that does not depend on anyone having
    // written anything down.
    let boot_model_id = boot
        .get("model")
        .and_then(|m| m.get("id")?.as_str())
        .map(str::to_owned);
    // PER-DRIVER, not per-process, so a caller that wants asynchronous
    // completions can ask for them without deciding for every other
    // driver alive in the same process. The env var is the default the
    // boot key overrides.
    // ONE PARSE, and every knob in it. See `crate::boot`.
    let cfg = crate::boot::Boot::from_boot(Some(&boot));
    let runahead = cfg.runahead;
    // An unrecognised spelling is refused rather than defaulted: silently
    // giving bf16 to a caller who asked for fp8 is the kind of wrong
    // answer that reads as a slightly worse model.
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
    // The pages can be written in every catalogued format — `kv_paged.cu`
    // switches on the scheme — but only `attn::attention_naive_paged`
    // READS one back. The fire path's prefill and decode are FlashInfer's
    // `_bf16` entry points, which take the view and ignore its scheme, so
    // a non-native format today would be appended correctly and attended
    // to as though the bytes were bf16.
    //
    // That is a wrong answer rather than a crash, so it is refused here.
    // Lifting it is a kernel change (a scheme-aware attention fast path),
    // not a plumbing one, and the plumbing should not wait for it.
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
    // `[batching] calibrate_planner`, which is where it is DOCUMENTED
    // and where the C++ reads it. This site read `[driver]`, so the key
    // every doc in the tree names had no effect.
    let calibrating = cfg.calibrating;
    let device_ordinal = boot
        .get("driver")
        .and_then(|d| d.get("device")?.as_integer())
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(0);
    let tp_size = driver_u32("tp_size", 1).max(1);
    let tp_rank = driver_u32("tp_rank", 0).min(tp_size - 1);
    // BIND THE DEVICE HERE, on the thread that will fire.
    //
    // `cudaSetDevice` is per-THREAD, so binding it only inside `load_model`
    // would leave every later call on whatever device the thread last had --
    // which is device 0, which is why the hardwiring was invisible. Doing it
    // at create is what makes `[driver] device` mean anything.
    if let Err(e) = crate::device::Device::bind(device_ordinal) {
        eprintln!("[driver-cuda] create: cannot bind CUDA device {device_ordinal}: {e}");
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }

    // The key a tensor-parallel group finds itself by. See
    // `Shell::tp_group_id`: `layout::rendezvous` needs one and this driver
    // supplied none, so the plan reduction it already calls was inert.
    let tp_group_id = boot
        .get("driver")
        .and_then(|d| d.get("tp_group_id")?.as_str())
        .unwrap_or_default()
        .to_owned();

    // A GROUP OF MORE THAN ONE IS STILL REFUSED, and the refusal now NAMES
    // WHAT IT IS WAITING ON rather than restating a claim about the tree.
    //
    // The LAYOUT half of tensor parallelism works, and so does everything
    // around the collective now: `bind/arms/comm.rs` binds both `comm::`
    // symbols and all three `dist::` ones, `build_tp_plane` below constructs
    // a `CustomAllReduce` out of `layout::rendezvous::tp_host_allgather`, and
    // `Shell::all_reduce` holds it for the fires. What is missing is one
    // thing and it is the last one: DEVICE TEXT. See `tp_serving_refusal`.
    if let Err(why) = tp_serving_refusal(tp_size, &tp_group_id) {
        eprintln!("[driver-cuda] create: {why}");
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    // The facts were PARSED here, out of a `CAPS_JSON` that was not a
    // `DeviceFacts` — see `state::device_facts` for what that cost. Stating
    // them removes the only failure this function had that no input could
    // avoid.
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

/// World sizes the P2P plane's CONSTRUCTOR takes —
/// `fire::all_reduce::CustomAllReduce::initialise`, which refuses anything
/// outside `2 <= w <= 8` with `w % 2 == 0`.
const CONSTRUCTIBLE_WORLD_SIZES: &[u32] = &[2, 4, 6, 8];

/// The fused landing's world size, and it is a single value —
/// `custom_all_reduce.cu:308` builds a fusion workspace only for two, and
/// `Plane::can_fuse_residual_rmsnorm` refuses anything else.
const FUSED_WORLD_SIZE: u32 = 2;

/// Why this build refuses to SERVE a tensor-parallel group, or `Ok(())`.
///
/// # Why this is a function and why it reads a constant from `kernels-cuda`
///
/// The refusal it replaced was a paragraph, and the paragraph had already
/// gone stale in three places: it said the collective rows *"were declared in
/// the ARCHIVE crate's `src/gemm.rs`"* (deleted), that they are *"the live
/// kernels crate's `src/not_yet_crossed.rs`'s now"* (also deleted — they are
/// `comm::ROUTINES` and `dist::ROUTINES`), and that there is *"no
/// `CustomAllReduce` handle to pass"* (there is one; [`build_tp_plane`] makes
/// it). Every one of those was a hand-written second account of a fact owned
/// somewhere else, and each drifted when its owner moved.
///
/// So the blocking condition is read from the thing that knows it.
/// [`kernels_cuda::comm::CAN_LAUNCH`] is the launch half's statement about
/// itself, pinned to its own bodies by `the_bodies_agree_with_can_launch`.
/// Flipping it here is not possible; flipping it there requires the bodies to
/// agree.
///
/// # The ceilings, in the order a caller can act on them
///
/// 1. **No device text** — `CAN_LAUNCH == false`. **This no longer fires**:
///    `csrc/src/attn/flashinfer/comm/` holds both headers, `comm::ROOT`
///    compiles them, and the constant is `true`. The branch stays because the
///    whole point of reading it is that this file does not decide it — the
///    launch half does, and it can say `false` again for a reason this file
///    should not be enumerating.
/// 2. **The world size the plane can be built for** —
///    [`CONSTRUCTIBLE_WORLD_SIZES`], which is what
///    `CustomAllReduce::initialise` admits AND what vllm's plain kernels are
///    instantiated at (`comm::PLAIN_NRANKS`). The two sets are the same
///    `{2, 4, 6, 8}`, which is why one constant answers both.
/// 3. **The fused landing is world size two only** —
///    [`FUSED_WORLD_SIZE`]. A model text that states
///    `comm::all_reduce_residual_rmsnorm_bf16` at any other width gets
///    `Decline::FusionWorldSize` at its first fire. This is a warning rather
///    than a refusal because the plain reduction still serves at 4, 6 and 8,
///    and which of the two a checkpoint lowers to is the model's call.
/// 4. **A group key** — `[driver] tp_group_id`. Without one the ranks cannot
///    find each other in `layout::rendezvous`, and a group whose members
///    never meet would bootstrap N planes each pointing at their own memory:
///    every reduction a no-op, silently.
///
/// # WHAT THIS FUNCTION DOES NOT CHECK, AND CANNOT
///
/// That a reduction produces the right sum. `CAN_LAUNCH` means there is
/// device text and a template-id for it, and `kernels_cuda::comm`'s own
/// header says at length that no launch in it has ever run on more than one
/// device — the box it was written on has one GPU. So a `tp_size = 2`
/// deployment that gets past this function is the FIRST thing in this
/// repository to fire a collective, and that is what an operator is agreeing
/// to. It is not stated in the returned message because there is no message
/// on the `Ok` path; it is stated here, where somebody widening this gate
/// will read it.
///
/// What is NOT here is the per-message ceiling — 8 MiB, the 16-byte multiple,
/// the NCCL crossover. Those are `CustomAllReduce::can_handle`'s, they are
/// answered per fire, and `bind/arms/comm.rs` falls back rather than failing
/// on them. A load-time gate on a per-message limit would refuse deployments
/// that never send a message that large.
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
/// # Where the pieces come from
///
/// * **The all-gather** is `layout::rendezvous::tp_host_allgather`, keyed on
///   `[driver] tp_group_id`, which is the same key `tp_min_plan` reduces the
///   memory plan on. The ranks of a TP group are threads of one process, so
///   an in-process barrier IS the bootstrap collective — that is the shape
///   `HostAllgather` was designed for and the reason its constructor takes a
///   callback rather than an `NcclComm&`.
/// * **`group_devices`** is gathered rather than configured. `[driver]
///   device` is one rank's ordinal and `CustomAllReduce` needs every rank's,
///   indexed by rank; the first all-gather round supplies exactly that, so
///   there is no second config key to keep consistent with the first.
///   `enable_peer_access`'s doc warns these must be real ORDINALS and never
///   rank indices — a mistake that works on every single-group box and
///   corrupts the second group on a four-GPU one — and gathering the ordinal
///   each rank actually bound is what makes that unmistakable.
/// * **`same_process: true`**, because they are. Raw pointers cross instead
///   of `cudaIpcMemHandle_t`s, which is what `custom_all_reduce.cu:264-272`
///   is for.
/// * **The fusion extents** are the fire ceiling this driver advertises and
///   the model's FULL hidden width. Full, not this rank's shard: every rank
///   holds a partial sum of the whole vector, which is what makes the
///   collective a sum rather than a concatenation.
///
/// # Errors
///
/// The sentence to print. Every failure is a device or configuration one —
/// unavailable peer access, a failing IPC exchange, an exhausted `RankData`
/// slab — and each already names itself through `crate::error::Error`.
///
/// # Unexercised
///
/// **This has never run**, and the reason narrowed. It used to be that
/// [`tp_serving_refusal`] refused `tp_size > 1` before any caller reached it,
/// because `kernels_cuda::comm::CAN_LAUNCH` was `false`; both all-reduce
/// headers are internalised now and that constant is `true`, so a group with
/// a key at a world size in [`CONSTRUCTIBLE_WORLD_SIZES`] does reach here.
/// What has not changed is the box: it has one GPU, so nothing here has met a
/// second rank.
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

    // ROUND ONE: every rank's device ordinal, so `group_devices` is what the
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
///
/// It was `destroy_impl(driver: *mut PieDriver)`, called by a
/// `pie_cuda_destroy` export and doing `Box::from_raw` to take the shell
/// back. Nothing leaks the shell any more, so the compiler runs this at the
/// end of the owner's scope and the export has nothing left to do.
impl Drop for Shell {
    fn drop(&mut self) {
        let shell = self;
        // EVERY QUEUED FIRE FIRST, because they may still be writing.
        //
        // A fire that is on the stream when the driver is dropped will run
        // its stream-ordered callback against a `ChannelState` copy, and the
        // frees below would take that memory back underneath it. Waiting is
        // the only correct answer here: unlike the reclaim in `step_impl`
        // there is no later call to defer to.
        //
        // It also frees the channels those fires were holding for a
        // `close_channel` that arrived while they were queued.
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
        // The handle is the DRIVER's, so its destructor is the driver's
        // too — `CublasHandle` asserts it was released rather than dropped.
        if let Some(mut h) = shell.cublas.take() {
            h.release(&mut crate::device::cublas::LiveCublas);
        }
        if let Some(mut scratch) = shell.scratch.take() {
            let mut sops = crate::fire::attention_workspace::LiveStagingOps;
            scratch.ws.release(&mut sops);
            // The prefill plan's own workspace, released beside the
            // decode plans'. `AttentionWorkspace` has no working `Drop`
            // -- every CUDA call goes through `StagingOps` and `Drop` has
            // no `&mut O` -- so a workspace nobody releases is a pinned
            // host leak and a debug assert.
            scratch.prefill_ws.release(&mut sops);
            drop(scratch.decode_plan);
            drop(scratch.decode_plan_full);
            drop(scratch.prefill_plan);
        }
    }
}

/// The load itself; `i32` errors are the ABI's status codes.
pub(crate) fn load_impl(state: &mut Shell, snapshot: &std::path::Path) -> Result<(), i32> {
    use model_loader::checkpoint::read::{parse_checkpoint_metadata, read_meta};

    let meta = parse_checkpoint_metadata(snapshot)
        .map_err(|e| crate::Error::invalid("load_model: checkpoint parse", format!("{e:?}")))?;

    // The checkpoint's own `config.json`: embedded in an artifact, else
    // the boot TOML's path.
    //
    // ONE FIELD IS READ OUT OF IT. This used to be a `pie.model/1`
    // descriptor — ~40 numbers, normalized by 845 lines from a
    // 136-field schema — parsed back here into a resident `HfConfig`
    // that the load path, the launch path, the cost model and the
    // capability report all read from. Every one of those numbers is a
    // catalog row's now.
    //
    // What is left is the declared quantization, and it stays because
    // it is the one thing a row genuinely cannot state: Qwen3-8B is one
    // model and four downloads, and a group size is not an extent of
    // any tensor.
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

    // WHICH MODEL THIS IS, asked of the TENSORS.
    //
    // The config above no longer DECIDES anything: the row that authors
    // the contract, projects the deployment and speaks the chat template
    // is matched here, once, against the checkpoint's own tensor names
    // and extents.
    //
    // Identification and validation are the same operation, which is the
    // point. A config that lies about its geometry used to be believed
    // by the derivation and contradicted by an assertion several frames
    // later, if at all. A checkpoint is now a known model or it is not.
    let chosen = state
        .boot_model_id
        .as_ref()
        .map_or(model::catalog::Override::None, |id| {
            model::catalog::Override::Id(id.clone())
        });
    let row = model::catalog::identify(&meta, &chosen)
        .map_err(|e| crate::Error::unsupported("load_model: identify", e.to_string()))?;

    // What the FILES say about how the numbers are stored, which is not
    // part of what model this is: Qwen3-8B is one row and four
    // downloads.
    let encoding = model::encoding::Encoding::from_config_json(&config_json)
        .map_err(|e| crate::Error::invalid("load_model: config", e.to_string()))?;

    // THE LOAD IS `model-loader`'s PLAN, EXECUTED ONTO THE DEVICE.
    //
    // What this replaced: a loop that read each checkpoint tensor into a
    // host `Vec`, uploaded it, and then read three of them BACK off the
    // device to concatenate a `qkv` and uploaded that — three round trips
    // for bytes a plan lays out once, and a thousand `cudaMalloc`s where
    // a resident plan wants one arena.
    //
    // The plan also decides what the driver used to decide by hand: which
    // encodings are loadable (a transform outside `CUDA_TILE_MAP_MASK` is
    // refused when the plan COMPILES, with the tensor named, rather than
    // mis-bound at launch), and which projections are fused.
    let target = crate::weights::plan::cuda_storage_target(state.tp_rank, state.tp_size);
    let (plan, _moe) =
        crate::weights::plan::compile_load_plan_for(snapshot, &meta, &target, row, &encoding)
            .map_err(|e| crate::Error::unsupported("load_model: load plan", e))?;
    let alloc = crate::device::Allocator::new();
    // ALREADY an `Error`, and the site was throwing it away: the old
    // line matched on nothing and returned `PIE_STATUS_EXHAUSTED` for
    // every staging failure, so a missing tensor and a full arena
    // reported the same thing.
    // `Error::from` spelled out because `?` does not chain two of them,
    // and the orphan rule forbids the shortcut: `From<LoaderError> for
    // i32` would be an impl on a primitive this crate does not own.
    let staged = crate::weights::stage::stage_plan_weights(&plan, snapshot, &alloc)
        .map_err(crate::Error::from)?;

    let mut model = LoadedModel {
        id: row.id(),
        // Filled below, once the checkpoint view can be built — the
        // derivation reads the weight map, which needs the model.
        deployment: model::deployment::Deployment::empty(),
        load_caps: Vec::new(),
        weights: staged.spans,
        owned: staged.owned,
        aliases: std::collections::BTreeMap::new(),
        layer_scalars: Vec::new(),
        tp_size: state.tp_size,
    };
    wire_trace_names(&mut model);

    // ONCE, at load, and never again. See `LoadedModel::deployment`.
    //
    // A PROJECTION of the matched row rather than a derivation from a
    // parsed config. The eleven `*_facts_from_hf` functions this
    // replaces read the same numbers out of the same checkpoint, one
    // family at a time, keyed on a `model_type` string that a second
    // table keyed differently.
    model.deployment = row
        .deployment(model::catalog::Deployed {
            // This driver, named. See the note at the trace call in
            // `fire/launch.rs`: the row serves both backends, so which
            // one is asking is the caller's to state.
            backend: model::catalog::Backend::Cuda,
            tp_size: state.tp_size,
            layer_scalars: &model.layer_scalars,
        })
        .map_err(|e| i32::from(crate::Error::from(e)))?;

    // A KV SHAPE THIS SHELL HAS NO POOL FOR IS REFUSED AT LOAD, not at
    // its first fire — and it is a MATCH now, which is §8 row 7.
    //
    // This used to ask `unbuilt_kv_store()`, a vtable method returning
    // `Option<&'static str>`: a family that forgot to implement it
    // answered `None` and loaded. A `match` on `KvStyle` cannot forget,
    // because a new variant fails to compile here until someone decides
    // what this shell does with it.
    //
    // The MLA and DSv4 caches are ported and waiting for a forward path.
    // Until there is one, saying so at the door is the honest answer:
    // otherwise the checkpoint loads, reports itself healthy through
    // `capabilities_json`, and dies inside a walk on a
    // `DispatchRefusal` — late, quiet, and the exact shape gpt-oss
    // failed in before its wiring landed.
    // THE REFUSAL CARRIES WHAT HAPPENED. This said it twice, on two
    // channels — the reason to stderr and `-1` to the caller — which is
    // exactly the defect §3.4 names: "an engine cannot learn which
    // layer, which kernel, or which fire refused, only that something
    // did." `Error::Unsupported` carries the sentence, and
    // `serve::status_of` is the one place that logs it.
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

    // THE GQA RATIO, refused at LOAD rather than discovered at launch.
    //
    // The same argument as the `KvStyle` match above, for the same
    // reason, at the same door: FlashInfer's decode instantiates a fixed
    // set of group sizes and reports anything else by THROWING, and a
    // throw crossing the C ABI is undefined behaviour. The shim prints
    // and dies, because a launcher signature has nowhere to put a
    // failure. This return does.
    //
    // It is asked HERE rather than in `model` because it is not a fact
    // about the checkpoint — it is a fact about what this build
    // instantiated, and `super::DECODE_GQA_GROUPS` is where this crate
    // states it. `model` states the shape; the driver states the set.
    model
        .deployment
        .servable_by(super::DECODE_GQA_GROUPS)
        .map_err(|why| -> i32 {
            // The refusal's OWN sentence, plus the numbers. `servable_by`
            // distinguishes a fractional ratio from an uninstantiated
            // one, and collapsing them here would report a malformed
            // shape as a missing kernel.
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

    // A NEW RESIDENCY, so every cache keyed on the old one is dropped.
    //
    // Replacing `state.model` frees the previous `LoadedModel`'s weight
    // arena, and a captured graph bakes addresses into it. Both caches
    // were keyed on the LAYER COUNT, so a second 32-layer checkpoint hit
    // the first one's entries — a replay into freed memory, with every
    // pointer looking exactly as valid as it did.
    //
    // Bumped AND cleared, which is belt to that braces: the counter makes
    // the key honest for anything that outlives this line, and the clear
    // returns the memory rather than leaving entries no key can reach.
    state.load_generation += 1;
    state.lowerings.clear();
    state.supergraph = crate::fire::recordings::Recordings::new();
    // AND THE P2P PLANE, for a reason the paragraph above already covers
    // once: the fusion workspace's `hidden` is compared for EQUALITY, so a
    // plane built for the previous checkpoint declines every fused fire of
    // the next one — and if the two happen to share a hidden size, it does
    // something worse and reduces through buffers sized for a different
    // token ceiling. Dropped here rather than reused, which is safe because
    // every rank reloads together and rebuilds together, below.
    state.all_reduce = None;
    state.model = Some(model);
    // AFTER the model is stored, because a calibration boot fires through the
    // ordinary path and that path reads `state.model`.
    let caps = capabilities_json(state, snapshot)?;
    state.model.as_mut().expect("just stored").load_caps = caps;

    // THE P2P PLANE, BUILT ONCE, HERE.
    //
    // Not in `create`, because the fusion workspace's stride is BAKED at
    // construction — `Plane::can_fuse_residual_rmsnorm` compares `hidden` for
    // EQUALITY, not against a bound — so it cannot be sized before the
    // checkpoint says what `hidden` is. And after the caps, because the other
    // extent it needs is the token ceiling this driver ADVERTISES: sizing the
    // workspace for anything smaller would decline fires a caller is entitled
    // to send, and for anything larger would reserve memory no fire can use.
    // `calibrate_planner` reads the published caps for exactly this reason,
    // one line down.
    //
    // Once, not per fire: the addresses this allocates are what every peer's
    // mapping points at, so a plane rebuilt mid-serve is a set of addresses
    // no rank could agree on twice.
    //
    // REACHABLE, AND NEVER TAKEN. `tp_serving_refusal` used to turn
    // `tp_size > 1` away at `create` on `comm::CAN_LAUNCH`, so `state.tp_size`
    // was 1 here in every configuration the build accepted. That gate is gone
    // -- the headers are vendored -- and this branch is now what a `tp_size =
    // 2` deployment with a group key runs. It has still never been taken,
    // because taking it needs two GPUs.
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

/// The calibration sweep: time the reachable fire shapes and write the
/// fastest to the profile cache the next boot reads.
///
/// `[batching] calibrate_planner` turns it on. It runs at LOAD, after the
/// weights are resident and the caps are published, because a probe fires
/// through the ordinary path and that path reads `state.model`.
///
/// WHY THIS COULD NOT BE WRITTEN BEFORE, and it was not about calibration:
/// a shape the driver cannot bind is the probe's ANSWER, and until today
/// the fence seam asserted on a CUDA error and took the process with it.
/// `StagingOps`'s pair reports now, so `None` from a timer is a point the
/// sweep skips rather than a crash.
///
/// Failures here are NOTED AND SWALLOWED. A calibration boot that cannot
/// measure still has a model loaded and an analytic plan to serve with;
/// refusing the load would turn a missing optimisation into an outage.
fn calibrate_planner(state: &mut Shell) {
    use crate::layout::calibrate::{Ceiling, Point, StepTimer, sweep};
    use crate::serve::state::{InstanceEntry, ProgramEntry};

    let Some(model) = state.model.as_ref() else {
        return;
    };
    // THE CEILING IS WHAT THE DRIVER JUST ADVERTISED, not what the planner
    // computed. Those differ: `capabilities_json` publishes the lattice's
    // rectangle, and a caller may not exceed it. Sweeping above the
    // advertisement would measure shapes no scheduler will ever send.
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

    /// Times one point by firing a synthetic batch of that shape.
    ///
    /// It reports `None` for a shape it cannot fire, which is what makes
    /// this a probe: the ladder walks DOWN from the ceiling and the first
    /// points are the ones most likely to be declined.
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
        // THE ROW, not its family. A cache keyed on `"qwen3"` plus a
        // hidden size was keyed on a shape summary; keyed on the id it
        // is keyed on the model, and the four extents below stay as a
        // guard against a build whose row moved under a stale file.
        model_type: model.id.to_owned(),
        hidden_size: i32::try_from(model.deployment.shape.hidden).unwrap_or(0),
        num_hidden_layers: i32::try_from(model.deployment.layers).unwrap_or(0),
        num_attention_heads: i32::try_from(model.deployment.shape.q_heads).unwrap_or(0),
        num_key_value_heads: i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
        head_dim: i32::try_from(model.deployment.shape.head_dim_alloc()).unwrap_or(0),
    };

    // ONE probe program and as many instances as the widest point needs.
    // Registration is a map insert, so it costs nothing to make the full
    // set up front and hand each point a prefix of it.
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

    // THE PROBE LEAVES NOTHING BEHIND. Its instances hold KV pages and its
    // program is not one the engine registered; a serving boot that
    // inherited them would be serving a rectangle nobody asked for.
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
/// The batch is `R` requests of `tokens_per_request` tokens each, every
/// request a fresh prefill from position zero over pages of its own. That is
/// the shape the planner's rectangle DESCRIBES — `max_forward_tokens` is a
/// prefill width and `max_forward_requests` a decode one — so a batch that
/// fills both axes at once is the worst case each point must survive.
///
/// SYNCHRONOUS, which is the whole point: the sweep is timing wall clock,
/// and a fire that returns before the device has finished has been timed at
/// the speed of the enqueue rather than of the step. `owes: None` makes
/// `step_impl` synchronize before it returns.
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
    // Pages this batch needs if every request gets its own run. A point
    // whose footprint exceeds the pool is not a candidate — refusing here
    // is cheaper than firing into an allocator that will refuse anyway.
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
            // TOKEN ZERO for every row. The sweep is timing a SHAPE, and
            // which token sits in a slot changes nothing about the work —
            // every kernel this fires is dense. Zero is in every vocabulary.
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

/// Answer the trace names a launch will ask for, from `model`'s tables.
///
/// The driver's whole part in naming, and it is deliberately small: which
/// trace name means which published tensor is FAMILY knowledge and lives in
/// `model::shared::weight_names`, beside the DSL that invents the trace names and the
/// contract author that invents the published ones. What is left here is the
/// two things only a driver can answer.
///
/// **Whether a join is a rename or nothing.** A checkpoint that ships its
/// projections pre-joined (Phi-3) has its contract SPLIT them, so
/// `Projections::Fused` has nothing to fuse and the halves are merely
/// adjacent. They are adjacent IN THE ARENA — the plan wrote them once, in
/// file order — so the fused operand exists and only wants a name. Only the
/// driver holds the addresses that decide it, and it checks rather than
/// assumes: a GEMM handed a discontiguous operand reads what lies between.
///
/// **Reading a load-time scalar to the host.** gemma-4's per-layer
/// `layer_scalar` is one bf16 on the device; `model` says which tensors they
/// are and in what order, and the copy is CUDA's.
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
/// **This used to be a five-field summary of the checkpoint** —
/// `{"model_type":…,"hidden":…,"layers":…,"vocab":…,"weights":…}` — which is
/// not a capability payload and which `DriverCapabilities` rejects outright,
/// field by field, at `unknown field \`model_type\``. So no engine could load
/// a model through this driver; the ABI tests call `pie_cuda_load_model`
/// directly and pass `null` for the caps, so nothing here noticed.
///
/// # The KV pool is SIZED HERE, and that is the substance
///
/// `total_pages` is what a scheduler admits against, so answering it is
/// answering how much context this device holds. The budget comes from
/// `layout::memory_planner::budget_for` — the ported planner's own reserve
/// arithmetic, which until now had no live caller — measured AFTER the
/// weights are resident, so `cudaMemGetInfo`'s free figure already has them
/// subtracted.
///
/// What the budget then has to cover is the KV pool and the fire's
/// activations. The activation share is a fraction rather than a computed
/// arena, because the arena is a property of the LOWERING and no fire has
/// been lowered yet; a fifth is the C++'s own rule of thumb and it is stated
/// here rather than hidden in a constant.
fn capabilities_json(state: &mut Shell, snapshot: &std::path::Path) -> Result<Vec<u8>, i32> {
    use crate::layout::memory_planner::{
        DeviceMemory, DeviceProps, ModelCosts, ModelShape, NoProfiles, PlannerConfig,
        ProfileSource, ShapeKnees, plan,
    };
    use crate::layout::model_costs::{CheckpointCosts, DiskProfiles};

    let model = state.model.as_ref().expect("the model is stored");
    // NO `HfConfig` READ ON THIS PATH.
    //
    // Three reads used to survive here — `model_type`,
    // `max_position_embeddings`, and whether the checkpoint carries
    // `gemma_vision`/`gemma_audio` — and they were the reason the driver
    // kept a whole parsed `config.json` resident for the life of a load.
    // They are `Deployment::advertised` now, off the same row that
    // authored the contract and traced the forward, so the model a
    // program is told about and the model this driver fires cannot be
    // two different models.
    //
    // CLONED for the same reason `model_tp` and `model_id` are copied:
    // the planner below wants `state` back.
    let deployment = model.deployment.clone();
    let model_tp = model.tp_size;
    // Copied out beside `model_tp` for the same reason: `&'static str`,
    // so this ends the borrow of `state` rather than extending it across
    // the planner below.
    let model_id = model.id;
    let device = crate::device::Device::bind(state.device_ordinal)?;
    let (free, total) = device.memory_info()?;
    let (major, minor) = device.compute_capability().unwrap_or((0, 0));
    let cfg = PlannerConfig {
        gpu_mem_utilization: 0.90,
        memory_profile: "auto".to_owned(),
        max_forward_tokens: 0,
        max_forward_requests: 0,
        // PINNED, and this is a coupling rather than a preference: the fire
        // path builds 16-token pages by construction (`page_size: usize = 16`
        // in `step_impl` and in `resize_pool`). Letting the lattice sweep page
        // sizes would have it answer a geometry the driver does not build.
        kv_page_size: 16,
        // The driver's OWN format, so the planner sizes the pages the
        // driver will actually allocate. It was hardwired while the shell
        // could only build bf16; now that the format is a boot key, a
        // hardwired planner would under-count a quantized cache by up to
        // 4x and hand back a page budget the device cannot hold.
        kv_cache_dtype: state.kv_format.name().to_owned(),
        tp_size: i32::try_from(model_tp).unwrap_or(1),
        mtp_num_drafts: 0,
        // FALSE EVEN WHEN CALIBRATING, and the divergence is deliberate.
        //
        // `calibrating` makes the planner build the CEILING of the feasible
        // region — the largest rectangle whose `arena + persistent` fits the
        // budget — on the reasoning that a bigger arena can run a smaller
        // shape. That holds when the arena is the only limit. It is not here:
        // the attention workspace is a FIXED 32 MB the shell allocates, and a
        // fire wider than it supports fails inside CUDA rather than returning
        // a status. On an L40S the ceiling is N=65536, whose logits buffer
        // alone is twenty gigabytes and whose fire aborts.
        //
        // So the sweep explores at or below the shape the driver is built to
        // fire, which is the scored pick. It measures which of the REACHABLE
        // shapes is fastest — a smaller claim than the C++'s and a true one.
        calibrating: false,
        rs_slot_mult: 1,
        // THE KEY THE RANKS ACTUALLY RENDEZVOUS ON. This was `String::new()`,
        // which is precisely the value `rendezvous::tp_min_plan` reads as
        // "no group to reconcile with" — so the cross-rank plan agreement
        // this driver calls was wired up and inert, and two ranks could leave
        // the planner with different `max_requests` and deadlock at the first
        // collective rather than failing here. One key, both rendezvous.
        nccl_unique_id_hex: state.tp_group_id.clone(),
    };
    // THE ROW'S OWN NUMBERS, so the pool the planner sizes and the
    // pool the fire builds come from one statement of the shape. Both
    // of these read a resident `HfConfig` — a second parse of the
    // checkpoint, whose disagreements with the first never surfaced as
    // an error, only as a KV pool a few thousand pages short.
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
    // MEASURED AFTER THE WEIGHTS ARE RESIDENT, so `cudaMemGetInfo`'s free
    // figure already has them subtracted and the budget is what is left.
    // A MEASUREMENT BEATS THE SCORE, when there is one. `DiskProfiles` reads
    // the cache a calibration boot writes; a machine that has never
    // calibrated has no file, which is a miss and not an error, and the
    // planner falls back to the analytic pick. `NoProfiles` is the honest
    // stand-in when no cache directory can even be derived.
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

    // PAGES AGAINST WHAT THE ARENA LEAVES, and this is a deliberate
    // divergence from the planner's own rule.
    //
    // The planner sizes pages against the FULL budget, and says why: "the
    // arena is a transient graph workspace that is freed between fires, while
    // the KV pool is the resident one". That was true of the C++. It is not
    // true here — `Scratch` keeps the arena, the named seam buffers and
    // the descriptor arrays for the life of the driver, because a captured
    // graph bakes the addresses it recorded and a freed arena can never be
    // replayed into. Persistence is the precondition for the supergraph and
    // for run-ahead, and it is not negotiable.
    //
    // So the two resident allocations share one budget, and charging only one
    // of them would advertise a page count whose pool cannot be built: on
    // qwen3-0.6B the full-budget figure is 22,016 pages against a budget the
    // arena also has to come out of.
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
        // THE LATTICE'S ANSWER, not a stated ceiling. These are what a
        // scheduler batches under, and the arena the planner chose is sized
        // for exactly this rectangle — a fire wider than it has no workspace.
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
        // No swap pool, no elastic accounting, no MTP or value head, and no
        // sink this shell honours yet. Every one of these is a claim a
        // program BINDS against, so a false advertisement is a program that
        // runs as a silent no-op rather than one that is refused.
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
        // TRUE, and it is the region table that makes it safe to say:
        // a fire marks its adapter rows, the `HasLora` guard states the
        // correction, and a lane that does not resolve degrades to the
        // arm's no-op rather than to a wrong answer.
        has_lora: true,
        model_site_summary: driver_api::ModelSiteSummary::default(),
        // EXACTLY THE THREE PORTS `fire::envelope::compose` READS, and not
        // one bit more.
        //
        // `PIE_DECODE_ENVELOPE_PORTS` is `EmbedTokens | Positions | KvLen`.
        // Those three come off the instance's channel rings before the
        // forward; everything else a decode needs — its pages, its page CSR,
        // its write descriptor — the class DERIVES from the positions, which
        // is the class's own contract (`LaunchPlan::geometry_class`).
        //
        // `PIE_DEVICE_GEOMETRY_PORTS` is deliberately NOT here. That claim
        // wins the pool-owned device-geometry class, whose members state their
        // own pages, page CSR, write descriptor and dense mask cell in-graph;
        // `compose` reads none of those and would hand the fire the pages the
        // program did not choose. `driver-vulkan`'s own note on widening this
        // mask is the rule: "widening the claim alone would have been the
        // silent version of the bug".
        //
        // At `0` the engine classified this driver's decode loops as `Host`
        // and they died at `EmbedTokens is not host-derivable: channel 0 has
        // no host-known value` — the whole of what this claim buys.
        device_geometry_port_mask: driver_api::PIE_DECODE_ENVELOPE_PORTS,
        // FALSE, and it is a different question from the mask above.
        //
        // `pipeline::fire` reads it to decide whether a frame slot may consume
        // a channel an earlier slot of the same frame publishes — and only for
        // a pass whose `devgeo` is set, which is the pool-owned class this
        // driver does not claim. A decode ENVELOPE is not covered by that rule
        // at all, so the bench's chained frame is admitted with this at
        // `false` and flipping it would buy nothing this driver can use.
        //
        // What it would CLAIM is that `launch` converts one step, fires it,
        // and only then converts the next. That is in fact what `launch_impl`
        // now does — `compose_step` runs inside `step_impl`, per step, after
        // the previous step has synchronized — so the day this driver serves
        // the pool-owned class, this is the line to revisit. Until it does,
        // saying `true` would advertise a class it refuses by name.
        resolves_geometry_per_step: false,
        // TRUE WHEN THIS CHECKPOINT HAS A TOWER `pie_cuda_encode` SERVES,
        // and it was hardwired false while four GPU tests fired the entry
        // point and passed.
        //
        // That is the failure a false negative makes: the worker refuses
        // to build an encode executor at all when this is clear
        // (`worker/src/executor/mod.rs:1341`), so gemma-4's vision and
        // audio towers — ported, bound, and matching HF's embeddings to
        // cosine — were unreachable through the engine. The tests never
        // saw it because they call the entry directly, which is exactly
        // the seam a capability is supposed to cover.
        //
        // Asked of the CHECKPOINT rather than stated of the driver: the
        // encode arms refuse a deployment with no `gemma_vision` /
        // `gemma_audio`, so answering true without one would advertise a
        // call that is guaranteed to fail. Qwen3-VL is deliberately NOT
        // here — its tower (`tower::qwen3_vl::scatter`) writes into the
        // fire's hidden rows rather than handing host rows back, so it is
        // an in-fire path and not an encode one.
        supports_media_encode: deployment.advertised.media_encode,
        kv_handle: None,
        // This driver compiles its own PTIR through NVRTC; nothing upstream
        // needs to generate a kernel for it.
        codegen_backend: String::new(),
    };
    serde_json::to_vec(&caps).map_err(|_| PIE_STATUS_DRIVER_ERROR)
}

/// Adopt one non-empty launch package and compile what it generates.
///
/// Split out so the id lifecycle above reads as the lifecycle: the empty
/// case, the dedup case and the id assignment are all one paragraph, and
/// the thing that can fail is one call.
pub(crate) fn adopt_and_compile(
    state: &mut Shell,
    id: u64,
    desc: &driver_api::ProgramRegistration,
    package: driver::driver_api::plan::LaunchPackage,
    kernels: &[driver_api::EmittedKernel],
) -> Result<(), i32> {
    let plan = driver::adopt_launch_package(package)
        .map_err(|error| crate::Error::unsupported("register_program", error))?;

    // The compile, when there is a device to compile FOR. `load_model`
    // binds it; a registration that arrives first is not an error, and
    // guessing an architecture would produce a cubin for the wrong GPU
    // rather than a diagnostic.
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
        // Recorded rather than refused: an unexecutable plan is a fact
        // about the program that the launch needing it must be able to
        // report, and losing the reason here would leave that launch with
        // nothing to say.
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
/// Read per registration rather than cached on the shell because the two
/// numbers that matter are cheap and the one that is not — the NVRTC
/// version — is a `dlopen`'d call the loader has already resolved by the
/// second registration. Caching it would trade nothing for a field that
/// can go stale against a runtime swap.
pub(crate) fn ptir_target(ordinal: i32) -> Result<crate::program::Target, i32> {
    // THE CUDA ERROR ALREADY IS one of ours — `Device::bind` and
    // `compute_capability` return `crate::Error::Driver`, and the old
    // lines logged it and then replaced it with a status that says
    // nothing about which call failed.
    let device = crate::device::Device::bind(ordinal)?;
    let (major, minor) = device.compute_capability()?;
    let nvrtc = crate::program::compile::version().map_err(|error| {
        eprintln!("[driver-cuda] register_program: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    Ok(crate::program::Target {
        major,
        minor,
        // The ordinal, widened. A stable per-GPU id is what the identity
        // wants and what stops one machine's cache answering for another
        // family; with one device bound per process the ordinal IS that
        // id, and it is the number the C++ used.
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

    /// The refusal has to NAME what it is waiting on, because the sentence it
    /// replaced named three things that had all moved. This asserts on the
    /// content rather than merely on `is_err`, which is the only way a
    /// message can be kept true by a test.
    ///
    /// **The subject changed when `CAN_LAUNCH` flipped.** It used to assert
    /// that a two-rank group IS refused and that the message names the
    /// missing device text; both headers are vendored now, so a two-rank
    /// group with a key is admitted and the sentence that named their absence
    /// is gone. What is still checkable is the shape: the first ceiling that
    /// bites names the value it read.
    #[test]
    fn a_group_is_refused_by_the_condition_that_actually_blocks_it() {
        // A key is what a two-rank group is missing now, and the refusal says
        // so rather than reaching for something further down.
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
        // And a fully configured two-rank group is admitted, which is the
        // whole of what vendoring the headers bought at this layer.
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
        // The fused landing's set is a DIFFERENT one, and six is the value
        // that shows it: the plane builds, the plain kernel exists, and
        // flashinfer never instantiated a fused launcher for it.
        assert!(!kernels_cuda::comm::NRANKS.contains(&6));
        assert!(kernels_cuda::comm::NRANKS.contains(&16));
        assert!(!CONSTRUCTIBLE_WORLD_SIZES.contains(&16));
    }
}
