//! Create, destroy, and everything that happens once per model.
//!
//! The verbs behind four of the thirteen exports: standing the shell up,
//! reading a checkpoint onto the device, wiring trace names onto checkpoint
//! names, answering what this deployment can do, and adopting a program.
//! All of it happens before any fire; none of it happens again.

use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR,
    PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_UNSUPPORTED,
    PieDriver,
    PieDriverCaps,
    PieDriverCreateDesc,
    PieProgramDesc,
    validate_create_out_params,
};
use super::launch::{runahead_env, sg_trace};
use super::checked;
use super::state::{CAPS_JSON, FireArrays, LoadedModel, Shell, retire, shell};

pub(crate) fn create_impl(
    desc: *const PieDriverCreateDesc,
    caps: *mut PieDriverCaps,
) -> *mut PieDriver {
    // `create` RETURNS A POINTER, not a status, so it cannot pass the
    // validator's message back — it can only refuse. The message still
    // reaches the log, which is the whole reason this call is here and not
    // an `abi_version` test: a caller that got a null handle learns WHY
    // from the line `checked` prints rather than from three candidate
    // causes.
    let Ok(desc) = checked(desc, driver_api::local::validate_driver_create_desc, "create") else {
        return std::ptr::null_mut();
    };
    if validate_create_out_params(caps).is_err() {
        eprintln!("[driver-cuda] create: the caps out-parameter must be non-null");
        return std::ptr::null_mut();
    }
    // The boot TOML rides in `config_bytes`. Two keys are read today:
    // `[model] descriptor` and `[driver] runahead`.
    let boot = (!desc.config_bytes.ptr.is_null())
        .then(|| unsafe {
            std::slice::from_raw_parts(desc.config_bytes.ptr, desc.config_bytes.len)
        })
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| text.parse::<toml::Table>().ok())
        .unwrap_or_default();
    let boot_descriptor = boot
        .get("model")
        .and_then(|m| m.get("descriptor")?.as_str())
        .map(std::path::PathBuf::from);
    // PER-DRIVER, not per-process, so a caller that wants asynchronous
    // completions can ask for them without deciding for every other
    // driver alive in the same process. The env var is the default the
    // boot key overrides.
    let runahead = boot
        .get("driver")
        .and_then(|d| d.get("runahead")?.as_bool())
        .unwrap_or_else(runahead_env);
    // An unrecognised spelling is refused rather than defaulted: silently
    // giving bf16 to a caller who asked for fp8 is the kind of wrong
    // answer that reads as a slightly worse model.
    let kv_format = match crate::store::KvCacheFormat::from_name(
        boot.get("driver").and_then(|d| d.get("kv_cache_dtype")?.as_str()).unwrap_or("auto"),
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[driver-cuda] create: {e:?}");
            return std::ptr::null_mut();
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
        return std::ptr::null_mut();
    }
    let driver_u32 = |key: &str, default: u32| {
        boot.get("driver")
            .and_then(|d| d.get(key)?.as_integer())
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(default)
    };
    let calibrating = boot
        .get("driver")
        .and_then(|d| d.get("calibrate_planner")?.as_bool())
        .unwrap_or(false);
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
    if let Err(e) = crate::cuda::Device::bind(device_ordinal) {
        eprintln!(
            "[driver-cuda] create: cannot bind CUDA device {device_ordinal}: {e}"
        );
        return std::ptr::null_mut();
    }

    // A GROUP OF MORE THAN ONE IS REFUSED, and refusing is the whole point.
    //
    // The LAYOUT half of tensor parallelism works: `tp_rank`/`tp_size` reach
    // `cuda_storage_target`, so a rank compiles a plan that reads only its own
    // bands, and `store::kv_geometry` divides the cache the same way. A rank
    // therefore holds a SHARD of every projection.
    //
    // The COLLECTIVE that puts the shards back together does not exist. The
    // three `comm::`/`dist::` all-reduce rows are declared in
    // `kernels-cuda/src/gemm.rs` and no launch reaches them; there is no NCCL
    // in this tree and no `CustomAllReduce` handle to pass. So a rank that
    // accepted `tp_size > 1` would run its own shard, skip the reduction, and
    // return an answer that is a fraction of the real one -- with no error
    // anywhere, which is the worst failure a driver has.
    //
    // Silence is the bug. Until a collective lands, this is a refusal.
    if tp_size > 1 {
        eprintln!(
            "[driver-cuda] create: [driver] tp_size = {tp_size} is refused. \
             This driver shards a rank's WEIGHTS and its KV cache correctly, and \
             has no all-reduce to combine the shards -- so serving would return \
             one rank's partial answer as if it were the whole one. See \
             .wiki/new-driver/next.md, Priority 3."
        );
        return std::ptr::null_mut();
    }
    let boxed = Box::new(Shell {
        caps: CAPS_JSON.as_bytes().to_vec(),
        boot_descriptor,
        runahead,
        kv_format,
        cublas: None,
        lowerings: std::collections::BTreeMap::new(),
        calibrating,
        device_ordinal,
        preds: None,
        peel_win: None,
        logits_staging: None,
        tp_rank,
        tp_size,
        model: None,
        programs: std::collections::BTreeMap::new(),
        instances: std::collections::BTreeMap::new(),
        next_id: 1,
        notify: desc.runtime.notify,
        notify_ctx: desc.runtime.ctx,
        fire_arrays: FireArrays::default(),
        supergraph: crate::model::supergraph::SupergraphCache::new(),
        kv: None,
        gdn: None,
        channels: std::collections::BTreeMap::new(),
        swap: None,
        scratch: None,
        fire_stream: None,
        in_flight: std::collections::VecDeque::new(),
        fire_alloc: None,
        ptir: crate::ptir::Runtime::default(),
        ptir_control: None,
        ptir_sessions: std::collections::BTreeMap::new(),
        ptir_programs: crate::ptir::Programs::new(),
        ptir_plans: std::collections::BTreeMap::new(),
    });
    let raw = Box::into_raw(boxed);
    if let Some(out) = unsafe { caps.as_mut() } {
        out.json_bytes = unsafe { (*raw).caps.as_ptr() };
        out.json_len = unsafe { (*raw).caps.len() };
    }
    raw.cast()
}

pub(crate) fn destroy_impl(driver: *mut PieDriver) {
    if !driver.is_null() {
        let mut shell = unsafe { Box::from_raw(driver.cast::<Shell>()) };
        // EVERY QUEUED FIRE FIRST, because they may still be writing.
        //
        // A fire that is on the stream when the driver is destroyed will run
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
        // The handle is the DRIVER's now, so its destructor is the driver's
        // too — `CublasHandle` asserts it was released rather than dropped.
        if let Some(mut h) = shell.cublas.take() {
            h.release(&mut crate::cuda::cublas::LiveCublas);
        }
        if let Some(mut scratch) = shell.scratch.take() {
            let mut sops = crate::model::attention_workspace::LiveStagingOps;
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
        drop(shell);
    }
}

/// The load itself; `i32` errors are the ABI's status codes.
pub(crate) fn load_impl(state: &mut Shell, snapshot: &std::path::Path) -> Result<(), i32> {
    use model_loader::checkpoint::read::{parse_checkpoint_metadata, read_meta};

    let meta = parse_checkpoint_metadata(snapshot).map_err(|e| {
        eprintln!("[driver-cuda] load_model: checkpoint parse: {e:?}");
        PIE_STATUS_INVALID_ARGUMENT
    })?;

    // The descriptor: embedded in an artifact, else the boot TOML's path.
    let descriptor_json = match read_meta(&meta, "model/descriptor") {
        Ok(Some(bytes)) => String::from_utf8(bytes).map_err(|_| PIE_STATUS_DRIVER_ERROR)?,
        Ok(None) => {
            let Some(path) = &state.boot_descriptor else {
                eprintln!(
                    "[driver-cuda] load_model: no embedded model/descriptor \
                     and no [model] descriptor in the boot config"
                );
                return Err(PIE_STATUS_UNSUPPORTED);
            };
            std::fs::read_to_string(path).map_err(|_| PIE_STATUS_INVALID_ARGUMENT)?
        }
        Err(e) => {
            eprintln!("[driver-cuda] load_model: read_meta: {e:?}");
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
    };
    let hf = crate::model::descriptor::parse_pie_model_descriptor(&descriptor_json)
        .map_err(|e| {
            eprintln!("[driver-cuda] load_model: descriptor: {e}");
            PIE_STATUS_INVALID_ARGUMENT
        })?;

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
    let target = crate::loader::plan::cuda_storage_target(state.tp_rank, state.tp_size);
    let (plan, _moe) =
        crate::loader::plan::compile_load_plan(snapshot, &meta, &target, &descriptor_json)
            .map_err(|e| {
                eprintln!("[driver-cuda] load_model: {e}");
                PIE_STATUS_UNSUPPORTED
            })?;
    let alloc = crate::cuda::Allocator::new();
    let staged = crate::loader::stage::stage_plan_weights(&plan, snapshot, &alloc).map_err(
        |e| {
            eprintln!("[driver-cuda] load_model: staging: {e:?}");
            PIE_STATUS_EXHAUSTED
        },
    )?;

    let mut model = LoadedModel {
        hf,
        load_caps: Vec::new(),
        weights: staged.spans,
        owned: staged.owned,
        aliases: std::collections::BTreeMap::new(),
        gemma_layer_scalars: Vec::new(),
        tp_size: state.tp_size,
    };
    wire_trace_names(&mut model);

    // A FAMILY THIS SHELL CANNOT FIRE IS REFUSED AT LOAD, not at its
    // first fire.
    //
    // The MLA lineage has facts rows, so `facts_from_hf` succeeds — and
    // its attention reads latent ckv/kpe planes rather than the paged k/v
    // pair `kv_pools_for` builds, with no executor arm naming an MLA
    // dispatch. Without this the checkpoint loads, reports itself healthy
    // through `capabilities_json`, and dies inside a walk on a
    // `DispatchRefusal` — which is late, quiet, and the exact shape
    // gpt-oss failed in before its wiring landed.
    //
    // `store::mla_cache` and `store::dsv4_compress_cache` are ported and
    // waiting for a forward path. Until there is one, saying so at the
    // door is the honest answer.
    if let Ok(family) = crate::facts::facts_from_hf(&model.checkpoint())
        && let Some(store) = family.unbuilt_kv_store()
    {
        eprintln!(
            "[driver-cuda] load_model: `{}` attends through {store}, which this \
             driver does not build — its cache is ported and has no forward \
             path to serve. Refusing here rather than at the first fire.",
            model.hf.model_type
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }

    state.model = Some(model);
    // AFTER the model is stored, because a calibration boot fires through the
    // ordinary path and that path reads `state.model`.
    let caps = capabilities_json(state, snapshot)?;
    state.model.as_mut().expect("just stored").load_caps = caps;
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
    use crate::store::calibrate::{Ceiling, Point, StepTimer, sweep};
    use crate::serve::state::{InstanceEntry, ProgramEntry};

    let Some(model) = state.model.as_ref() else { return };
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
    let template = crate::store::profile_key::ProfileShape {
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
            match synthetic_fire(self.state, point, &self.instances, self.page_size, self.total_pages)
            {
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

    let key = crate::store::profile_key::ProfileKey {
        gpu_name: String::new(),
        compute_major: 0,
        compute_minor: 0,
        sm_count: 0,
        kv_cache_dtype: state.kv_format.name().to_owned(),
        tp_size: i32::try_from(state.tp_size).unwrap_or(1),
        model_type: model.hf.model_type.clone(),
        hidden_size: model.hf.hidden_size,
        num_hidden_layers: model.hf.num_hidden_layers,
        num_attention_heads: model.hf.num_attention_heads,
        num_key_value_heads: model.hf.num_key_value_heads,
        head_dim: model.hf.head_dim_kernel.max(model.hf.head_dim),
    };

    // ONE probe program and as many instances as the widest point needs.
    // Registration is a map insert, so it costs nothing to make the full
    // set up front and hand each point a prefix of it.
    let probe_program = state.next_id;
    state.next_id += 1;
    state
        .programs
        .insert(probe_program, ProgramEntry { program_hash: 0, emitter_version: 0 });
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
            },
        );
        instances.push(id);
    }

    let mut timer = FireTimer { state, instances, page_size, total_pages };
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
            s.max_forward_tokens, s.max_forward_requests, s.step_ms, s.step_ms_stddev,
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
    match crate::store::profile_cache::ProfileCache::discover("") {
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
    point: crate::store::calibrate::Point,
    instances: &[u64],
    page_size: i32,
    total_pages: u32,
) -> Result<(), i32> {
    use driver_api::local::{PieFrameDesc, PieStepDesc, PieU32Slice, PieU64Slice};

    let reqs = usize::try_from(point.max_forward_requests).unwrap_or(0).min(instances.len());
    let per = usize::try_from(point.tokens_per_request()).unwrap_or(1).max(1);
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
        cells.push(driver_api::local::PieTerminalCell {
            outcome: driver_api::local::PIE_TERMINAL_OUTCOME_PENDING,
            reserved0: 0,
        });
    }
    let cell_ptrs: Vec<*mut driver_api::local::PieTerminalCell> =
        cells.iter_mut().map(|c| c as *mut _).collect();

    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let step = PieStepDesc {
        roster_rows: u32s(&roster_rows),
        sub_batch_indptr: u32s(&sub_batch_indptr),
        sub_batch_class: u32s(&sub_batch_class),
        terminal_cells: driver_api::local::PieTerminalCellPtrSlice {
            ptr: cell_ptrs.as_ptr(),
            len: cell_ptrs.len(),
        },
        token_ids: u32s(&token_ids),
        position_ids: u32s(&position_ids),
        kv_page_indices: u32s(&kv_page_indices),
        kv_page_indptr: u32s(&kv_page_indptr),
        kv_last_page_lens: u32s(&kv_last_page_lens),
        qo_indptr: u32s(&qo_indptr),
        ..Default::default()
    };
    let frame = PieFrameDesc {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        instance_ids: PieU64Slice { ptr: instances.as_ptr(), len: reqs },
        required_kv_pages: u32::try_from(pages_total).unwrap_or(0),
        steps: driver_api::local::PieStepDescSlice { ptr: &step, len: 1 },
        ..Default::default()
    };
    crate::serve::launch::step_impl(state, &frame, &step, None)
}

/// Answer the trace names a launch will ask for, from `model`'s tables.
///
/// The driver's whole part in naming, and it is deliberately small: which
/// trace name means which published tensor is FAMILY knowledge and lives in
/// `model::weight_names`, beside the DSL that invents the trace names and the
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
    let published: Vec<String> = model.weights.keys().cloned().collect();
    let set: std::collections::BTreeSet<&str> =
        published.iter().map(String::as_str).collect();
    let has = |n: &str| set.contains(n);
    let wiring = model::weight_names::wire(&model.hf, &has);

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
            model.weights.insert(trace, crate::loader::stage::WeightSpan {
                ptr: spans[0].ptr,
                bytes: spans.iter().map(|s| s.bytes).sum(),
            });
        }
    }
    model.gemma_layer_scalars = wiring
        .scalars
        .iter()
        .map(|n| {
            model.weights.get(n).map_or(1.0f32, |b| {
                match crate::loader::stage::read_span(*b) {
                    Ok(back) if back.len() == 2 => {
                        f32::from_bits(u32::from(u16::from_le_bytes([back[0], back[1]])) << 16)
                    }
                    _ => 1.0,
                }
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
/// `store::memory_planner::budget_for` — the ported planner's own reserve
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
    use crate::store::memory_planner::{
        DeviceMemory, DeviceProps, Family, ModelCosts, ModelShape, NoProfiles, PlannerConfig,
        ProfileSource, plan,
    };
    use crate::store::model_costs::{CheckpointCosts, DiskProfiles};

    let model = state.model.as_ref().expect("the model is stored");
    let hf = model.hf.clone();
    let hf = &hf;
    let model_tp = model.tp_size;
    let device =
        crate::cuda::Device::bind(state.device_ordinal).map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
    let (free, total) = device.memory_info().map_err(|_| PIE_STATUS_DRIVER_ERROR)?;
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
        nccl_unique_id_hex: String::new(),
    };
    let costs = CheckpointCosts::new(hf, model_tp);
    let shape = ModelShape {
        hidden_size: hf.hidden_size,
        num_hidden_layers: hf.num_hidden_layers,
        num_attention_heads: hf.num_attention_heads,
        num_key_value_heads: hf.num_key_value_heads,
        head_dim_kernel: hf.head_dim_kernel.max(hf.head_dim),
        model_type: hf.model_type.clone(),
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
    let planned = plan(&cfg, &shape, &props, mem, Family::Generic, &costs, profiles)
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
    // true here — `FireArrays` keeps the arena, the named seam buffers and
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
        arch_name: hf.model_type.clone(),
        vocab_size: u32::try_from(hf.vocab_size).unwrap_or(0),
        max_model_len: u32::try_from(hf.max_position_embeddings).unwrap_or(0),
        activation_dtype: "bf16".to_owned(),
        hidden_size: u32::try_from(hf.hidden_size).unwrap_or(0),
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
        has_lora: false,
        model_site_summary: driver_api::ModelSiteSummary::default(),
        device_geometry_port_mask: 0,
        supports_media_encode: false,
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
    desc: &PieProgramDesc,
    package: driver::driver_api::plan::LaunchPackage,
    kernels: &[driver_api::EmittedKernel],
) -> Result<(), i32> {
    let plan = match driver::adopt_launch_package(package) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("[driver-cuda] register_program: {error}");
            return Err(PIE_STATUS_UNSUPPORTED);
        }
    };

    // The compile, when there is a device to compile FOR. `load_model`
    // binds it; a registration that arrives first is not an error, and
    // guessing an architecture would produce a cubin for the wrong GPU
    // rather than a diagnostic.
    if plan.executable && state.model.is_some() {
        let target = ptir_target(state.device_ordinal)?;
        let versions = driver::Versions::mirrored(desc.emitter_version);
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
                eprintln!(
                    "[driver-cuda] register_program: cannot compile program \
                     {:#018x}: {}",
                    desc.program_hash,
                    failure.reason()
                );
                return Err(PIE_STATUS_UNSUPPORTED);
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
pub(crate) fn ptir_target(ordinal: i32) -> Result<crate::ptir::Target, i32> {
    let device = crate::cuda::Device::bind(ordinal).map_err(|error| {
        eprintln!("[driver-cuda] register_program: no device to compile for: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    let (major, minor) = device.compute_capability().map_err(|error| {
        eprintln!("[driver-cuda] register_program: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    let nvrtc = crate::ptir::nvrtc::version().map_err(|error| {
        eprintln!("[driver-cuda] register_program: {error}");
        PIE_STATUS_DRIVER_ERROR
    })?;
    Ok(crate::ptir::Target {
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


