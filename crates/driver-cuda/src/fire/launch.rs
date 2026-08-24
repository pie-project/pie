//! The forward path: one step, from a frame descriptor to logits.
//!
//! `step_impl` is the whole of a fire, and the phases around it are its
//! parts: admit, the lane, the KV pools and descriptor arrays, the fa2
//! schedules, the resident planes, the GDN context, the view arena, the
//! walk, sampling, delivery, retirement.
//!
//! # What this file was, until R2
//!
//! Twelve hundred more lines, and three of the phases above were different
//! ones: `build_lowering` (trace the legacy catalog's text for this fire's
//! shape, lower it under one of two guard modes, join the ops back onto the
//! launches), `capture_or_replay` (warm, capture, instantiate, replay a
//! unionized supergraph keyed on the fire's bucket), and the peel (split a
//! fire whose rows disagreed about an axis, give the tail its own CSRs and
//! its own schedule). Every one of them existed because the executor was
//! handed a FLAT LIST OF LAUNCHES with the guards still in it.
//!
//! It is handed a `model_compiler::program::Program` now — one per lane,
//! built at load, chosen here by the fact word — and a Program's statements
//! carry their own operands. There is nothing to lower, nothing to join,
//! no guard to resolve on a device and no axis to split on. The deletions
//! are named where each stood.
//!
//! THE PERF DEBT IS REAL: the walk is EAGER, one launch per step, every
//! fire. `.wiki/driver/graph.md` describes the machinery that is gone.

use crate::fire::scratch::slot;
use crate::serve::load::ptir_target;
use crate::serve::state::{
    ChannelState, FireDebt, FireScratch, GdnState, InFlight, InstanceEntry, KvState, LoadedModel,
    RUNAHEAD_DEPTH, Shell, instance_ring_shapes, retire, retire_fire,
};
use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED, PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_UNSUPPORTED,
};
use driver_api::submission::FrameSubmission;

/// `PIE_CUDA_TRACE_SUPERGRAPH=1`: say what each fire did with the graph.
///
/// Lazy: an unset variable costs a `getenv` and no formatting.
pub(crate) fn sg_trace(what: impl FnOnce() -> String) {
    if std::env::var_os("PIE_CUDA_TRACE_SUPERGRAPH").is_some() {
        eprintln!("[sg] {}", what());
    }
}

/// The fire's class, read off its shape: one row per request is a decode,
/// anything else is prefill-shaped.
pub(crate) fn fire_class_of(
    _step: &driver_api::StepSubmission,
    rows: usize,
    requests: usize,
) -> Result<model_ir::trace::FireClass, i32> {
    use model_ir::trace::FireClass;
    Ok(if rows == requests {
        FireClass::Decode
    } else {
        FireClass::Prefill
    })
}

// `run_resolved` AND `capture_or_replay` STOOD HERE — 250 lines, the whole
// of the legacy walk's graph story: warm once per variant, open a capture on
// the fire allocator, record the union lowering with its guards left to the
// device, instantiate, and replay on every later fire of that bucket, with an
// eager re-lowering under `GuardMode::Resolve` as the fallback for every path
// that gave up.
//
// THE PERF DEBT, MEASURED AND NOT HIDDEN: on a Qwen3-0.6B decode the replay
// cost 12 ms against 3.0–4.8 s to capture 535 launches — and, more to the
// point, against the cost of ISSUING those 535 launches by hand every fire.
// The baker walk is eager: `baker::fire::Fire::step` issues one launch per
// `program.steps` entry, every fire, with no exec to replay.
//
// It is not a knob that was switched off. A baker capture needs the walk to
// be replayable, which needs its arena addresses and its `views` fields to be
// stable across fires (they are — both are pooled) and needs the lane's guard
// story, which `program::bound` answers by BUILDING ONE PROGRAM PER LANE
// rather than by leaving a predicate for the device. So the design is
// "capture a lane", which is simpler than what died here and is not written
// yet.

/// The fire itself, run against the shell's own state.
#[allow(clippy::too_many_lines)]
pub(crate) fn launch_impl(
    state: &mut Shell,
    frame: &FrameSubmission,
    completion: driver_api::completion::CompletionTarget,
) -> Result<(), i32> {
    let steps = frame.steps.as_slice();
    if steps.is_empty() {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    // Steps run SEQUENTIALLY, each a fire of its own — the frame's
    // producer→consumer ordering. One shared KV, per-step everything else.
    for step in &steps[..steps.len() - 1] {
        step_impl(state, frame, step, None)?;
    }
    // The last step carries the frame's debt: every step's terminal cells and
    // the completion the runtime waits on, enqueuing one async retire because
    // a frame completes once. Every step's cells, not just the last's — a cell
    // is per member, and an unpublished one leaves that member Pending forever.
    let step = steps.last().expect("nonempty");
    let cells: Vec<*mut driver_api::local::TerminalCell> = steps
        .iter()
        .flat_map(|s| s.terminal_cells.as_slice().iter().copied())
        .collect();
    step_impl(state, frame, step, Some((completion, cells)))
}

// `build_lowering` STOOD HERE. It called `row.trace(class, deployed)` on the
// legacy catalog's `Variant`, ran `model_compiler::lower::lower_with` over the
// fire's rows in one of two guard modes, joined the ops back onto the launches
// through `bind::DispatchPlan`, and refused the fire when any symbol was
// unfireable. Every one of those four is deleted: the trace is the new
// catalog's (`baker::load`), the lowering is `model_compiler::program`'s (one
// `Program` per lane, at LOAD), the join is the claim table's, and the refusal
// moved to `serve::load` where it belongs — a checkpoint whose program will
// not resolve is not loaded at all.
//
// `Shell::lowerings`, `LoweringKey` and `digest_rows` died with it: the cache
// existed because the trace-lower-join chain cost ~3.3 ms per fire shape, and
// a `Program` is built once per load.

/// The admit phase's result.
///
/// What a step must satisfy before anything is allocated, and the facts that
/// survive the asking. Returns owned values, not slices borrowed from
/// `state`, which is `&mut` for the rest of the fire.
struct Admitted {
    /// The service class the row/request ratio implies — the fact word that
    /// picks the lane.
    class: model_ir::trace::FireClass,
    /// Token rows in this step.
    rows: usize,
    /// Requests the step's CSR partitions those rows into.
    requests: usize,
    /// Which rows carry a read-out, in ascending order. Read off the step's
    /// OWN readout table (see [`sampled_rows_of`]), not off a lowering.
    sampled_rows: Vec<u32>,
}

/// Which rows of the fire the step reads out, in ascending order.
///
/// # Where this came from
///
/// `model_compiler::lower::Readouts::samples` decided this, because the
/// lowering needed a `Row::samples` bit to resolve a gather guard against.
/// The rule is the WIRE's and never was the lowering's: a request that names
/// no readout row reads its last (`qo_indptr[r + 1] - 1`), and one that names
/// rows names them in its OWN numbering, so row `k` of request `r` is fire
/// row `qo_indptr[r] + k`.
///
/// Both refusals are kept, because both are still reachable and neither is a
/// guess:
///
/// * a readout CSR that does not have one segment per request — the table
///   does not say which request named which row, and picking any reading
///   would deliver one request's logits to another;
/// * a named row past the end of the request that named it — likewise.
///
/// # What is NOT read any more, stated
///
/// The step's `region_sig` / `region_k` / `region_row_indptr` triple is the
/// SERIATION's output: per-region axis bits (multi-token, custom mask, hook,
/// LoRA, truncated depth) that `lower_with` resolved guards against. With no
/// guards there is nothing to resolve — a `Program` is picked whole, by the
/// fact word — so the driver no longer reads those three fields at all.
/// `FrameSubmission::validate` still checks their structure; this file no
/// longer gives them meaning, and saying so is cheaper than a translation
/// nothing consumes.
fn sampled_rows_of(rows: usize, step: &driver_api::StepSubmission) -> Result<Vec<u32>, i32> {
    let indices = step.plan.sampling_indices.as_slice();
    let indptr = step.plan.sampling_indptr.as_slice();
    let qo_indptr = step.plan.qo_indptr.as_slice();
    let requests = qo_indptr.len().saturating_sub(1);
    if !indices.is_empty() && indptr.len() != requests + 1 {
        eprintln!(
            "[driver-cuda] launch: the step reads out {} row(s) under a CSR of \
             {} segment(s) for {requests} request(s); nothing says which \
             request named which row",
            indices.len(),
            indptr.len().saturating_sub(1),
        );
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    let mut samples = vec![false; rows];
    for r in 0..requests {
        let (lo, hi) = (qo_indptr[r], qo_indptr[r + 1]);
        let span = hi.saturating_sub(lo);
        if span == 0 {
            continue;
        }
        let named = match (indptr.get(r), indptr.get(r + 1)) {
            (Some(&x), Some(&y)) if y >= x => indices.get(x as usize..y as usize).unwrap_or(&[]),
            _ => &[],
        };
        if named.is_empty() {
            // The default read-out: the request's last row.
            if let Some(slot) = samples.get_mut((hi - 1) as usize) {
                *slot = true;
            }
            continue;
        }
        for &row in named {
            if row >= span {
                eprintln!(
                    "[driver-cuda] launch: request {r} reads out its row {row} \
                     and holds {span} row(s)",
                );
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            }
            if let Some(slot) = samples.get_mut((lo + row) as usize) {
                *slot = true;
            }
        }
    }
    Ok(samples
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| s.then_some(u32::try_from(i).unwrap_or(0)))
        .collect())
}

/// See [`Admitted`].
#[cfg(feature = "abi")]
fn admit(state: &Shell, step: &driver_api::StepSubmission) -> Result<Admitted, i32> {
    use model_ir::trace::FireClass;

    let sub_batches = step.sub_batch_indptr.as_slice();
    if sub_batches.len() > 2 {
        eprintln!("[driver-cuda] launch: one sub-batch per step today");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let Some(model) = state.model.as_ref() else {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    };
    // The value derived at load (`LoadedModel::deployment`), not re-derived.
    let dep = &model.deployment;

    let token_ids = step.plan.token_ids.as_slice();
    let position_ids = step.plan.position_ids.as_slice();
    let kv_indptr = step.plan.kv_page_indptr.as_slice();
    let kv_lens = step.plan.kv_last_page_lens.as_slice();
    let qo_indptr = step.plan.qo_indptr.as_slice();
    if token_ids.is_empty()
        || token_ids.len() != position_ids.len()
        || kv_indptr.len() < 2
        || kv_indptr.len() != kv_lens.len() + 1
        || qo_indptr.len() != kv_indptr.len()
    {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    let rows = token_ids.len();
    let requests = kv_lens.len();
    let class = fire_class_of(step, rows, requests)?;
    let sampled_rows = sampled_rows_of(rows, step)?;

    // A family that does not declare a service class must be refused, not
    // fired: the fact word that picks the lane is computed from the class,
    // and a class no lane's word answers has no program behind it.
    if !matches!(class, FireClass::Decode | FireClass::Prefill) && dep.recurrent.is_none() {
        eprintln!(
            "[driver-cuda] launch: {class:?} is an MTP service pass and \
             this family declares no trace for it"
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    Ok(Admitted {
        class,
        rows,
        requests,
        sampled_rows,
    })
}

/// Run the instance's registered program over the fire's logits — the
/// sampling phase (top-p, top-k, temperature and argmax are PTIR ops a
/// caller's program states, not driver flags).
///
/// Returns `Ok(false)` when there is nothing to run: no program, a program
/// that compiled to nothing, or channels this shell does not hold. A program
/// that declines or is not ready is not a failed step — the cursors are left
/// where they were, so the next fire sees the same decision. Only a device
/// error propagates.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn run_program(
    // The disjoint `Shell` fields this phase touches, not `&mut Shell`: the
    // caller has borrowed `model`, `named_bufs`, `stream` and `alloc` out of
    // the shell, so a whole-shell borrow here would conflict.
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    programs: &crate::program::Programs,
    control: &mut Option<crate::program::Control>,
    sessions: &mut std::collections::BTreeMap<u64, crate::program::session::Session>,
    rings: &mut crate::program::channel::Rings,
    disk: &crate::program::Disk,
    device_ordinal: i32,
    instance_id: u64,
    logits: (u64, u32, u32),
    rows: usize,
    // The row of `logits` this instance's program reads — the last row of
    // its token span, not its index.
    row: usize,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
) -> Result<bool, i32> {
    use crate::program::session::Fired;

    // A DOOR THE BRING-UP NEEDS. A sampling program runs on device and
    // publishes through a ring, so when its answer is wrong there is nothing
    // to read: the same fire either samples on device or hands the host raw
    // logits, and only the second can be checked against the buffer this
    // driver just wrote. `PIE_NO_PTIR_SAMPLER=1` takes the second.
    if std::env::var_os("PIE_NO_PTIR_SAMPLER").is_some() {
        return Ok(false);
    }

    // THE BYTES THE PROGRAM IS ABOUT TO SAMPLE, at the address and row it was
    // pointed at. Everything else about the binding is checkable from the
    // host; the CONTENT at the moment of the fire is not, and a sampler that
    // answers zero looks the same whether it read the wrong buffer or the
    // right one wrongly.
    if std::env::var_os("PIE_TRACE_VALUES").is_some() && logits.0 != 0 {
        let (base, vocab, stride) = logits;
        let at = base + (row as u64) * u64::from(stride) * 2;
        let mut host = [0u16; 8];
        // SAFETY: the readout buffer's own claim -- `rows * stride` live bf16
        // at `base` -- and this row is inside it.
        let rc = unsafe {
            cudarc::runtime::sys::cudaMemcpy(
                host.as_mut_ptr().cast(),
                at as *const std::ffi::c_void,
                core::mem::size_of_val(&host),
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        let seen: Vec<f32> = host
            .iter()
            .map(|b| f32::from_bits(u32::from(*b) << 16))
            .collect();
        eprintln!(
            "[sample] base={base:#x} row={row} vocab={vocab} stride={stride} rc={rc:?} {seen:.4?}"
        );
    }
    let Some(instance) = instances.get(&instance_id) else {
        return Ok(false);
    };
    let Some(compiled) = programs.get(instance.program_id) else {
        return Ok(false);
    };
    // A PROGRAM has no single plan to pick. It used to: the fire prepared one
    // stage, so this had to choose which, and the choice had to be by KIND
    // rather than by position -- `plans.first()` is the epilogue only by
    // accident, and a package carrying an adapter puts its sink in a prologue,
    // so first would have fired the adapter and never sampled. `Session::fire`
    // now prepares every launching stage from its own plan, which is both the
    // reason that choice is gone and the reason an adapter program runs at all.
    //
    // The refusal that STOOD HERE -- a program with no plan at index `stage` --
    // is `stages_and_plans_agree`'s now, and states which index and why.
    if compiled.plans.is_empty() {
        return Ok(false);
    }
    let Some(shapes) = instance_ring_shapes(instance, channels) else {
        eprintln!(
            "[driver-cuda] launch: instance {instance_id} names a channel this \
             driver does not hold; its program cannot be given rings"
        );
        return Ok(false);
    };
    let channel_ids = instance.channel_ids.clone();
    let compiled = compiled.clone();

    // The control kernels, once. Same disk as the program runtime: they share
    // a key scheme, so a second cache directory would recompile both every
    // boot and never hit.
    if control.is_none() {
        let target = ptir_target(device_ordinal)?;
        let architecture = crate::program::compile::arch_flag(target.major, target.minor);
        match crate::program::Control::compile(disk, &architecture, "pie-cuda") {
            Ok(built) => *control = Some(built),
            Err(failure) => {
                eprintln!(
                    "[driver-cuda] launch: the PTIR control kernels will not \
                     compile ({}); this fire delivers raw logits",
                    failure.reason()
                );
                return Ok(false);
            }
        }
    }

    // `ensure_sessions` rings every instance the frame names before the
    // forward, so a missing session here is an instance whose channels this
    // shell does not hold — ringing one here would mean registering channels,
    // which is `ensure_sessions`'s job.
    if !sessions.contains_key(&instance_id) {
        eprintln!(
            "[driver-cuda] launch: instance {instance_id} was not ringed before the \
             forward; its program cannot be fired"
        );
        return Ok(false);
    }
    let _ = shapes;

    // The host planes, in the instance's channel order — the order a program
    // indexes them by, not the map's.
    let mut host: Vec<crate::program::channel::HostChannel> = Vec::with_capacity(channel_ids.len());
    for id in &channel_ids {
        let Some(channel) = channels.get(id) else {
            return Ok(false);
        };
        host.push(channel.host_plane());
    }

    let control = control.as_ref().expect("just ensured");
    let session = sessions.get_mut(&instance_id).expect("just ensured");
    let extents = driver::Extents {
        row_count: u32::try_from(rows).unwrap_or(1),
        token_count: u32::try_from(rows).unwrap_or(1),
        sampled_rows: 1,
        ..driver::Extents::default()
    };
    match session.fire(
        rings,
        &compiled,
        control,
        &mut host,
        logits,
        // One lane per fire, one row per lane: the fire is no longer
        // single-lane, but nothing groups yet, so this slice holds one and
        // the closure ignores the lane.
        |_lane| u32::try_from(row).unwrap_or(0),
        std::slice::from_ref(&extents),
        alloc,
        stream.as_ref(),
    ) {
        Ok(Fired::Committed { published }) => Ok(published > 0),
        Ok(Fired::Declined) => {
            sg_trace(|| format!("  ptir instance {instance_id} declined the fire"));
            Ok(false)
        }
        Ok(Fired::NotReady) => {
            sg_trace(|| format!("  ptir instance {instance_id} was not ready"));
            Ok(false)
        }
        Err(error) => {
            eprintln!("[driver-cuda] launch: the program refused: {error}");
            Err(PIE_STATUS_DRIVER_ERROR)
        }
    }
}

/// Publish the fire's readout — the last row's logits — through the
/// instance's reader channel. Convention until the channel table is parsed:
/// the roster's first instance, its first `READER` channel whose cell is
/// `[vocab]` f32. Device bf16 widens to the f32 wire on the host.
///
/// Takes `debt` by `&mut Option` because the paths differ in who waits: a step
/// owing a completion queues the D2H and hands its destination to the debt;
/// one owing nothing has synchronized and widens here.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
/// Where request `r`'s logits sit in the fire's logits buffer.
///
/// Its row is `qo_indptr[r + 1] - 1` (the last row of its token span); its
/// offset is that row's ordinal among the rows the fire read out, because the
/// gather packs the buffer as `[sampled, vocab]` in gather order. The two
/// coincide only when every row samples (the decode case).
fn logits_row_of(span_end: usize, rows: usize, sampled_rows: &[u32]) -> usize {
    let row = span_end.saturating_sub(1).min(rows.saturating_sub(1));
    if sampled_rows.len() == rows {
        return row;
    }
    sampled_rows
        .iter()
        .position(|&s| s as usize == row)
        .unwrap_or(row)
}

/// Grow the walk's activation arena for a fire of `rows` rows.
///
/// `rows * row_pitch` and NOTHING AFTER IT. The block used to carry a tail
/// of hand-carved columns for the three f32 planes `ssm.gdn_prep`'s routine
/// wrote and its statement did not state; both recurrence points stage those
/// out of `Ctx::scratch` now, beside every other plane a claim body needs,
/// so the arena holds exactly the values the walk assigned.
fn arena_for(
    program: &model_compiler::program::Program,
    fire_arrays: &mut crate::fire::scratch::Scratch,
    alloc: &crate::device::Allocator,
    rows: usize,
) -> Result<*mut std::ffi::c_void, i32> {
    let bytes = program.row_pitch as usize * rows;
    fire_arrays.baker_arena(alloc, bytes.max(64)).map_err(|e| {
        eprintln!("[driver-cuda] fire: the arena did not fit: {e:?}");
        PIE_STATUS_EXHAUSTED
    })
}

/// Fire this step's lane.
///
/// # What it borrows, and what it owns
///
/// Everything it needs is already built by the time it is called, and it is
/// all borrowed: `views` (this fire's KV pages, recurrent slabs, runtime
/// planes and raised schedules), the stream and cuBLAS handle, and `logits`
/// for the landing the delivery reads. It owns exactly one thing: the arena,
/// which is pooled.
///
/// The fa2 decode schedule rides `views.streams.decode_plan_cache` — what
/// `raise_attn_plans` raised for THIS fire, with stamped workspaces, the
/// variant as the lane stated it and `window_left = -1`. Reused rather than
/// replanned: `baker-smoke` carried its own `DecodePlanCache` because it had
/// no driver to ask, and planning a second one here would be a second 48 MB
/// workspace and a second answer to a question the fire already answered.
///
/// # The `Ok(usize)` is the step count
#[allow(clippy::too_many_arguments)]
fn baker_fire(
    // The disjoint `Shell` fields this phase touches, not `&mut Shell`:
    // `model` is borrowed out of the shell by the caller and lives until the
    // debt is built, so a whole-shell borrow here would conflict. The same
    // split `deliver_logits` takes, for the same reason.
    baked: &crate::baker::Baked,
    program: &model_compiler::program::Program,
    arena: *mut std::ffi::c_void,
    logits: Option<&crate::device::DeviceBuffer>,
    views: &crate::bind::views::FireViews,
    rows: usize,
    requests: usize,
    sampled_rows: &[u32],
    cublas: *mut std::ffi::c_void,
    raw_stream: *mut std::ffi::c_void,
) -> Result<usize, crate::bind::RunRefusal> {
    // A refusal that names the walk rather than a kernel.
    let refuse = |what: &str| {
        Err(crate::bind::RunRefusal {
            step: 0,
            kernel: "fire".to_string(),
            why: what.to_string(),
        })
    };

    let rows_i = match i32::try_from(rows) {
        Ok(r) if r > 0 => r,
        _ => return refuse("a fire with no rows"),
    };
    let requests_i = match i32::try_from(requests) {
        Ok(r) if r > 0 => r,
        _ => return refuse("a fire with no requests"),
    };
    // ROWS > 1 FIRES (W10). The refusal that stood here was real: the
    // executor cut packed rows with pointer arithmetic, which reports the
    // CUT's width as the row stride when the bytes stride by the PACKED
    // width — right at one row, silently wrong at two. It was fixed where
    // it belonged rather than papered over here: `ssm.gdn_prep` and
    // `ssm.gated_delta` are claim bodies now and every packed→compact cut
    // happens in a kernel that is told the packing, so the four
    // `Rect::column` calls (and `Rect::column` itself) are gone. The
    // remaining rule is one line and the whole design: an executor hands a
    // kernel DENSE rectangles only. `baker_serve`'s
    // `two_batched_requests_match_the_banked_rows` is the running gate.

    // SAFETY: `raw_stream` is this fire's stream and `cublas` is the handle
    // bound to it at create — which is exactly what `Ctx::with_cublas` asks
    // of its caller.
    //
    // `with_raised(views)` IS THE STAGING DOOR. A claim body pulls what the
    // statement does not carry off its own `Ctx` — the fa2 schedules, the
    // host CSR mirrors, the mask, the fire's streams — and `FireViews`
    // answers each by the key its `Raise` declares
    // (`bind::views::FireViews::raised`). This is the caller R2 left that
    // function waiting for.
    let cx = unsafe {
        kernels_cuda::jit::Ctx::on(raw_stream)
            .with_cublas(cublas)
            .with_raised(views)
    };
    let fire = crate::baker::fire::Fire {
        plan: &baked.plan,
        program,
        ctx: &cx,
        stream: raw_stream,
        arena,
        rows: rows_i,
        requests: requests_i,
        banks: &baked.banks,
        views,
    };

    for (i, step) in program.steps.iter().enumerate() {
        if let Err(refusal) = fire.step(step.op, &step.call) {
            let op = &baked.plan.ops[step.op as usize];
            eprintln!(
                "[driver-cuda] fire: step {i} (op {}, `{}`, layer {:?}) -> {refusal:?}",
                step.op, op.kernel, op.layer,
            );
            return Err(crate::bind::RunRefusal {
                step: i,
                kernel: op.kernel.clone(),
                why: format!("{refusal:?}"),
            });
        }
    }

    // The answer, into the buffer the delivery already reads.
    //
    // ONE STRIDED D2D, AND NOTHING DOWNSTREAM CHANGES. The walk's logits are
    // a rectangle of its own arena; the pinned staging `retire_fire` widens
    // reads them tightly packed at `(row * vocab + t) * 2`
    // (`serve/state.rs`). `cudaMemcpy2DAsync` is exactly that repitch, and
    // paying it buys every piece of machinery below unchanged — the D2H, the
    // widening, the ring publish, the terminal cells, the sampling programs.
    if let Some(buf) = logits {
        // The gather this copy does NOT model, named. `logits_row_of` maps
        // a request's last token row to its ordinal among SAMPLED rows,
        // because a compacting epilogue packs `[sampled, vocab]`. The lane
        // states no epilogue, so its `out` seam holds every row — which
        // agrees with the delivery exactly when every row is sampled. A
        // fire that compacts refuses rather than delivering row `k` as row
        // `j`.
        let identity = sampled_rows.is_empty()
            || (sampled_rows.len() == rows
                && sampled_rows
                    .iter()
                    .enumerate()
                    .all(|(i, &r)| r as usize == i));
        if !identity {
            eprintln!(
                "[driver-cuda] fire: this fire samples {} of {rows} rows; the \
                 lane states no gather epilogue",
                sampled_rows.len(),
            );
            return refuse("a gather epilogue");
        }
        let out = match fire.rect(baked.out) {
            Ok(r) => r,
            Err(_) => return refuse("the lane's `out` seam"),
        };
        let width = out.width as usize * out.dt.size() as usize;
        if buf.len() < rows * width {
            return refuse("a logits buffer this fire's rows do not fit");
        }
        // The source pitch is `width`, not `row_pitch`: the arena is read
        // VALUE-MAJOR (`baker::fire::Fire::rect`), so this rectangle's rows
        // are already contiguous. The 2D copy stays because the DESTINATION
        // is a different buffer and a plain 1D copy would hide which of the
        // two pitches is which.
        // SAFETY: `out` is a live rectangle of the arena and `buf` is at
        // least `rows * width` bytes, checked above.
        let code = unsafe {
            cudarc::runtime::sys::cudaMemcpy2DAsync(
                buf.as_ptr(),
                width,
                out.ptr.cast_const(),
                width,
                width,
                rows,
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                raw_stream.cast(),
            )
        };
        if code != cudarc::runtime::sys::cudaError::cudaSuccess {
            eprintln!("[driver-cuda] fire: the logits repitch failed: {code:?}");
            return refuse("the lane's logits");
        }
    }
    Ok(program.steps.len())
}

// `logits_value_of` STOOD HERE — the last launch whose first output named an
// SSA value, which is where the legacy epilogue left the logits. Both the
// delivery and the baker arm asked it, so that they could not drift about
// which buffer to read.
//
// They cannot drift now for a better reason: there is one buffer and the
// driver owns it (`Scratch::logits`). The baker walk repitches its `out` seam
// into it, and the delivery reads it. No lowering is consulted.

fn deliver_logits(
    // The disjoint `Shell` fields this phase touches, not `&mut Shell`:
    // `model` and `named_bufs` are borrowed out of the shell by the caller,
    // so a whole-shell borrow here would conflict.
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    logits_staging: &mut Option<crate::device::PinnedBuf>,
    retired_staging: &mut Vec<crate::device::PinnedBuf>,
    request_instances: &[u64],
    model: &LoadedModel,
    // Where the fire left its logits (`Scratch::logits`), repitched into by
    // the walk. `None` for a fire that produced none.
    logits: Option<&crate::device::DeviceBuffer>,
    stream: crate::device::StreamRef<'_>,
    rows: usize,
    // Where each request's token span ends, so its answer row can be
    // found. `qo_indptr[r + 1] - 1` is request `r`'s last row.
    qo_indptr: &[u32],
    // The rows the fire read out, in gather order: a request's logits live at
    // its ordinal here, not its row, once a gather packs `[sampled, vocab]`.
    sampled_rows: &[u32],
    // The requests this fallback is for — those whose PTIR program did not
    // publish; a request with a sampled answer must not also get a vocabulary.
    serve: &[usize],
    debt: &mut Option<FireDebt>,
) -> Result<(), i32> {
    // The step's instances, one per wire request (`request_instances`): the
    // frame's roster is not the step's.
    let instance_ids = request_instances;
    let vocab = model.deployment.shape.vocab as usize;

    // Every request, each its own reader channel and its own row: request `r`
    // owns `qo_indptr[r]..qo_indptr[r + 1]`, so its answer is at
    // `qo_indptr[r + 1] - 1` (equal to `r` only on a decode), and once the
    // epilogue compacts, at its ordinal among the sampled rows.
    let readouts: Vec<(ChannelState, usize)> = serve
        .iter()
        .filter_map(|&r| {
            let iid = instance_ids.get(r)?;
            let inst = instances.get(iid)?;
            let end = qo_indptr.get(r + 1).copied()? as usize;
            let row = logits_row_of(end, rows, sampled_rows);
            let ch = inst.channel_ids.iter().find_map(|cid| {
                channels.get(cid).filter(|ch| {
                    ch.host_role == driver_api::local::PIE_CHANNEL_HOST_ROLE_READER
                        && ch.cell_bytes == vocab * 4
                })
            })?;
            Some((*ch, row))
        })
        .collect();

    // The D2H is enqueued, not awaited: its destination belongs to the debt,
    // not this stack frame — a `Vec` here would be freed before an async copy
    // lands. One copy carries every row: N requests cost one D2H, N widenings.
    if let (Some(buf), false) = (logits, readouts.is_empty()) {
        match debt.as_mut() {
            Some(d) => {
                // The shell's buffer, grown to fit and reused (not the debt's).
                if logits_staging.as_ref().is_none_or(|p| p.len() < buf.len()) {
                    // Parked, not dropped: `PinnedBuf::drop` is a
                    // `cudaFreeHost` that is not stream-ordered, so freeing
                    // here would pull memory out from under an earlier fire's
                    // queued D2H.
                    retired_staging
                        .extend(logits_staging.replace(crate::device::PinnedBuf::new(buf.len())?));
                }
                let pin = logits_staging.as_mut().expect("just sized");
                let view = (pin.as_slice().as_ptr(), buf.len());
                buf.copy_to_host(&mut pin.as_mut_slice()[..buf.len()], stream)?;
                d.staging = Some(view);
                d.readouts = readouts;
            }
            None => {
                // A step that owes nothing has already synchronized.
                let mut bf16 = vec![0u8; buf.len()];
                buf.copy_to_host(&mut bf16, stream)?;
                stream.synchronize()?;
                for (ch, row) in &readouts {
                    let mut cell = vec![0u8; vocab * 4];
                    for t in 0..vocab {
                        let off = (row * vocab + t) * 2;
                        if off + 1 < bf16.len() {
                            let bits = u16::from_le_bytes([bf16[off], bf16[off + 1]]);
                            cell[t * 4..t * 4 + 4]
                                .copy_from_slice(&(u32::from(bits) << 16).to_le_bytes());
                        }
                    }
                    if !ch.publish(&cell) {
                        eprintln!(
                            "[driver-cuda] launch: logits ring full; a request dropped \
                         its output"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// Make the shell's persistent device state ready, and reclaim what the
/// last fires finished with.
///
/// The stream and allocator are the shell's, not per-fire: a pooling allocator
/// rebuilt every fire has no pool, and run-ahead needs a stream that outlives
/// the call so a second fire can queue behind the first.
///
/// Reclaim: a fire's scratch cannot be freed while it runs or from the
/// callback (CUDA forbids calling the runtime from a host function), so it is
/// freed here. Drop everything already retired without waiting, and wait only
/// when the queue is at `RUNAHEAD_DEPTH` — the backpressure that bounds how
/// much scratch the driver holds.
#[cfg(feature = "abi")]
fn ready_device_state(state: &mut Shell) -> Result<(), i32> {
    if state.fire_stream.is_none() {
        state.fire_stream = Some(crate::device::OwnedStream::new(0)?);
    }
    if state.fire_alloc.is_none() {
        state.fire_alloc = Some(crate::device::Allocator::new());
    }
    while state
        .in_flight
        .front()
        .is_some_and(|f| f.done.is_complete().unwrap_or(true))
    {
        let done = state.in_flight.pop_front().expect("just checked");
        retire(done);
    }
    // An empty `in_flight` is the only moment a replaced staging buffer is
    // certainly unreferenced: `retire_fire`'s host_fn is enqueued before the
    // `InFlight::done` event is recorded, both stream-ordered, so an entry
    // leaving `in_flight` proves its debt is already paid. The non-runahead
    // paths pay synchronously and never push an entry.
    if state.in_flight.is_empty() {
        state.retired_staging.clear();
    }
    while state.in_flight.len() >= RUNAHEAD_DEPTH {
        let oldest = state.in_flight.pop_front().expect("nonempty");
        oldest.done.synchronize()?;
        retire(oldest);
    }
    Ok(())
}

/// The hybrids' recurrent context: driver-owned slabs, instance slots.
///
/// Takes the disjoint `Shell` fields it touches (`gdn` mutable, `alloc`/
/// `stream` shared) rather than `&mut Shell`, which would conflict with the
/// borrows the caller holds.
///
/// Returns the context and the slot-id buffer it points into: the buffer is
/// returned, not dropped, because the context holds a raw pointer into it — a
/// fire that let it go would bind a freed address.
#[cfg(feature = "abi")]
fn gdn_context(
    gdn: &mut Option<GdnState>,
    // Bumped when the state pool grows: the bases move and a capture
    // baked them.
    epoch: &mut crate::fire::scratch::PlanEpoch,
    dep: &model::deployment::Deployment,
    step: &driver_api::StepSubmission,
    requests: usize,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
) -> Result<
    (
        Option<crate::bind::GdnCtx>,
        Option<crate::device::DeviceBuffer>,
    ),
    i32,
> {
    use crate::bind::GdnCtx;

    let mut gdn_ctx: Option<GdnCtx> = None;
    let mut _slot_ids_buf: Option<crate::device::DeviceBuffer> = None;
    if let Some(shape) = dep.recurrent.as_ref() {
        let (conv_stride, state_stride) = (shape.conv_stride_elems, shape.state_stride_elems);
        const GDN_SLOTS: u32 = 8;
        if (*gdn).is_none() {
            // The ported cache owns the layout: it pools the `(conv,
            // recurrent)` pairs and answers both strides.
            let is_linear: Vec<bool> = (0..dep.layers)
                .map(|l| shape.linear_layers.contains(&l))
                .collect();
            let cache =
                crate::pools::recurrent_state_cache::RecurrentStateCache::allocate_bf16_recurrent(
                    &is_linear,
                    u32::try_from(shape.conv_dim).unwrap_or(0),
                    u32::try_from(shape.conv_k).unwrap_or(0),
                    u32::try_from(shape.v_h).unwrap_or(0),
                    u32::try_from(shape.k_d).unwrap_or(0),
                    u32::try_from(shape.v_d).unwrap_or(0),
                    i32::try_from(GDN_SLOTS).unwrap_or(0),
                );
            let mut conv = alloc.alloc(
                usize::try_from(cache.layout().conv_total_bytes())
                    .unwrap_or(0)
                    .max(1),
            )?;
            let mut recurrent = alloc.alloc(
                usize::try_from(cache.layout().recurrent_total_bytes())
                    .unwrap_or(0)
                    .max(1),
            )?;
            conv.memset(0, stream.as_ref())?;
            recurrent.memset(0, stream.as_ref())?;
            (*gdn) = Some(GdnState {
                cache,
                conv,
                recurrent,
                is_linear,
                num_slots: GDN_SLOTS,
                conv_stride_elems: i64::try_from(conv_stride).unwrap_or(0),
                state_stride_elems: i64::try_from(state_stride).unwrap_or(0),
            });
        }
        let gdn_state = (*gdn).as_mut().expect("just ensured");
        // The engine assigns slots (`rs_slot_ids`, one per request): RESET
        // zeroes a slot before the fire; BUFFER_WRITE routes the pass's state
        // into a buffer slot instead of the live one; FOLD copies the accepted
        // prefix back afterwards.
        let rs_slot_ids = step.plan.rs_slot_ids.as_slice();
        let rs_flags = step.plan.rs_slot_flags.as_slice();
        if rs_slot_ids.len() != requests {
            eprintln!("[driver-cuda] launch: hybrid fire without rs_slot_ids");
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        // Buffer/fold flags: a speculative decode writes its tokens into a
        // buffer slot and folds only the accepted prefix into the live slot,
        // so a rejected token is never folded and there is nothing to repair.
        let rs_fold_lens = step.plan.rs_fold_lens.as_slice();
        let rs_buffer_slot_ids = step.plan.rs_buffer_slot_ids.as_slice();
        let rs_buffer_indptr = step.plan.rs_buffer_slot_indptr.as_slice();
        let need_slots = rs_slot_ids.iter().copied().max().map_or(1, |m| m + 1);
        gdn_state.ensure_slots(need_slots, epoch, &alloc, &stream)?;
        // RESET, asked of the cache: `reset_slot` emits one strided fill per
        // buffer (a `Memset2D` whose rows are the linear layers).
        for (r, &slot) in rs_slot_ids.iter().enumerate() {
            if rs_flags.get(r).copied().unwrap_or(0) & driver_api::local::PIE_RS_FLAG_RESET == 0 {
                continue;
            }
            let Ok(ops) = gdn_state
                .cache
                .reset_slot(i32::try_from(slot).unwrap_or(-1))
            else {
                eprintln!("[driver-cuda] launch: rs_slot_ids names a slot the cache lacks");
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            gdn_state.apply(&ops, stream.as_ref())?;
        }
        let slot_ids_h: Vec<i32> = rs_slot_ids
            .iter()
            .enumerate()
            .map(|(r, &live)| {
                // A BUFFER_WRITE row's pass writes the buffer slot, not the
                // live one: the buffer CSR names one slot per buffered token
                // and the pass writes the row's head (its first entry).
                let f = rs_flags.get(r).copied().unwrap_or(0);
                let slot = if f & driver_api::local::PIE_RS_FLAG_BUFFER_WRITE != 0 {
                    rs_buffer_indptr
                        .get(r)
                        .and_then(|&lo| rs_buffer_slot_ids.get(lo as usize))
                        .copied()
                        .unwrap_or(live)
                } else {
                    live
                };
                i32::try_from(slot).unwrap_or(0)
            })
            .collect();
        // Every slot the fire names has to exist, buffer slots included.
        let need_buffer = rs_buffer_slot_ids
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m + 1);
        gdn_state.ensure_slots(need_buffer.max(need_slots), epoch, &alloc, &stream)?;
        // The fold, recorded on the fire's stream so it lands after the pass
        // that filled the buffer: copy the accepted prefix's last state into
        // the live slot. A linear state is a running summary, so the state
        // after the accepted token is where the next fire continues; the
        // rejected tokens past it are never folded.
        for (r, &live) in rs_slot_ids.iter().enumerate() {
            let f = rs_flags.get(r).copied().unwrap_or(0);
            if f & driver_api::local::PIE_RS_FLAG_FOLD == 0 {
                continue;
            }
            let (lo, hi) = match (rs_buffer_indptr.get(r), rs_buffer_indptr.get(r + 1)) {
                (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
                _ => continue,
            };
            // A device-resolved length is clamped to the row's replay length
            // (the ABI bound). The port is not read yet, so a device row folds
            // its whole replay — the conservative answer.
            let span = hi.saturating_sub(lo);
            let want = if f & driver_api::local::PIE_RS_FLAG_FOLD_LEN_DEVICE != 0 {
                span
            } else {
                rs_fold_lens.get(r).copied().unwrap_or(0) as usize
            };
            let take = want.min(span);
            if take == 0 {
                continue;
            }
            let Some(&src_slot) = rs_buffer_slot_ids.get(lo + take - 1) else {
                continue;
            };
            // Linear halves only: a fold restores recurrent state to the
            // accepted prefix, but the MTP pending-hidden row was already
            // rebuilt from those accepted tokens, so copying it would
            // overwrite the newer value with an older one.
            let Ok(ops) = gdn_state.cache.copy_linear_state_slot_d2d(
                i32::try_from(src_slot).unwrap_or(-1),
                i32::try_from(live).unwrap_or(-1),
            ) else {
                eprintln!("[driver-cuda] launch: a fold names a slot the cache lacks");
                return Err(PIE_STATUS_INVALID_ARGUMENT);
            };
            gdn_state.apply(&ops, stream.as_ref())?;
        }
        let bytes: Vec<u8> = slot_ids_h.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut sbuf = alloc.alloc(bytes.len().max(4))?;
        sbuf.copy_from_host(&bytes, stream.as_ref())?;
        gdn_ctx = Some(GdnCtx {
            k_h: shape.k_h,
            v_h: shape.v_h,
            k_d: shape.k_d,
            v_d: shape.v_d,
            conv_dim: shape.conv_dim,
            conv_k: shape.conv_k,
            // mamba's B/C group count. ZERO, and stated here rather than
            // carried: no `#[points]` declaration in the catalog states a
            // group count (the mamba row that would is not in this tree),
            // and the legacy catalog's `RecurrentShape` set it to 0 for
            // every family it shipped. A number nothing states is a zero
            // written where a reader can see it, not a field to thread.
            n_groups: 0,
            // Still one base per model layer: pooling changed where a base
            // comes from, not what a launch is handed.
            conv_state: (0..gdn_state.is_linear.len())
                .map(|l| gdn_state.conv_base(l))
                .collect(),
            conv_stride_elems: gdn_state.conv_stride_elems,
            recurrent_state: (0..gdn_state.is_linear.len())
                .map(|l| gdn_state.recurrent_base(l))
                .collect(),
            state_stride_elems: gdn_state.state_stride_elems,
            slot_ids_d: sbuf.as_ptr().cast(),
            write_state: true,
        });
        _slot_ids_buf = Some(sbuf);
    }
    Ok((gdn_ctx, _slot_ids_buf))
}

/// Size the KV pools for this fire and describe them per layer.
///
/// Bumps the array epoch when it grows: growth moves base addresses, so every
/// capture that recorded one is stale.
///
/// A family may share one layer's cache with another (gemma-4's trailing
/// layers project no KV), so `kv_source(l)` says whose pool a layer reads and
/// only the sources get an allocation; the returned vector is as long as the
/// layer count, the pool vector is not.
///
/// Growth replaces the pools without migrating pages, so decode continuity
/// holds only while page demand is stable; migration rides with `resize_pool`.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn kv_pools_for(
    kv: &mut Option<KvState>,
    epoch: &mut crate::fire::scratch::PlanEpoch,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    need_pages: u32,
    page_size: i32,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
    format: crate::layout::KvCacheFormat,
) -> Result<Vec<crate::bind::abi::KvCacheLayerView>, i32> {
    // The provisioner asks which store, so a new `KvStyle` is a compile error
    // here rather than a family quietly served the wrong store. `load_model`
    // refuses the unbuilt styles at the door; this match is the other half.
    match dep.kv {
        model::deployment::KvStyle::Paged => {}
        model::deployment::KvStyle::Latent { .. } => {
            return Err(PIE_STATUS_UNSUPPORTED);
        }
    }
    let kv_heads_i = i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0);
    let n = dep.layers;
    // Per-layer geometry, family-decided: gemma-4's two layer kinds disagree
    // on head dim AND on head count (e4b keeps 2 across both, 31b reads 16 at
    // 256 and 4 at 512), and its trailing layers own no pages (they attend
    // through their source's). All three columns come off `Deployment` now;
    // the head count was `vec![one; n]` until `Deployment::of` read the rows.
    let per_layer = crate::pools::kv_cache::PerLayer {
        head_dim: dep.attention.iter().map(|a| a.head_dim as i32).collect(),
        kv_source_layer: dep.attention.iter().map(|a| a.kv_source as i32).collect(),
        num_kv_heads: dep.attention.iter().map(|a| a.kv_heads as i32).collect(),
    };
    // One set of pages has one shape, so a layer that reads through another's
    // must share its dims. A violation would not crash — each shared layer
    // would read its source's pages at its own stride and emit plausible
    // tokens — which is why it is checked here, where `layer_view` reports an
    // aliased layer's dims as its source's.
    per_layer.check_sharing()?;

    let grow = !matches!(&(*kv), Some(kv) if kv.num_pages >= need_pages);
    if grow {
        let layout = crate::pools::kv_cache::KvCacheLayout::plan_per_layer(
            n as i32,
            need_pages as i32,
            page_size,
            kv_heads_i,
            per_layer,
            format,
            false,
        )?;

        let mut ops = crate::pools::kv_cache_live::LiveKvCacheOps::new(stream.as_ref(), alloc);
        let cache = crate::pools::kv_cache_live::KvCache::materialize(layout, &mut ops)?;
        let mut held = ops.into_held();
        // `materialize` does not zero, and a page read before its first write
        // is otherwise whatever the allocator last had.
        for b in &mut held {
            b.memset(0, stream.as_ref())?;
        }

        // Growth replaces the pages without migrating them (decode continuity
        // holds while page demand is stable) and moves base addresses, so
        // every capture that recorded one is stale. `install_kv` owns the
        // epoch bump so no moved pool installs without one.
        crate::serve::state::install_kv(
            kv,
            epoch,
            KvState {
                cache,
                _held: held,
                num_pages: need_pages,
            },
        );
    }
    Ok((*kv).as_ref().expect("just ensured").views())
}

#[cfg(feature = "abi")]
#[allow(clippy::too_many_lines)]
/// The step descriptor's arrays, already borrowed out of the FFI slices.
///
/// Grouped because they travel together and because a phase taking seven
/// bare `&[u32]` parameters can transpose two of them silently.
struct StepArrays<'a> {
    token_ids: &'a [u32],
    position_ids: &'a [u32],
    kv_indices: &'a [u32],
    kv_indptr: &'a [u32],
    kv_lens: &'a [u32],
    qo_indptr: &'a [u32],
    required_kv_pages: u32,
}

/// What the fire's first phase leaves on the device.
struct FireInputs {
    /// KV page size, in tokens.
    page_size: i32,
    /// KV head count, unsharded.
    kv_heads_i: i32,
    /// The kernel-facing head dim.
    head_dim_i: i32,
    /// Per-layer views of the KV pool, in layer order.
    layers: Vec<crate::bind::abi::KvCacheLayerView>,
    d_ids: *const u32,
    d_pos: *const u32,
    d_kv_indices: *const u32,
    d_kv_indptr: *const u32,
    d_kv_lens: *const u32,
    d_qo: *const u32,
    d_w_page: *const u32,
    d_w_off: *const u32,
    /// One byte per row, all ones — the mask the sampler ANDs against.
    d_valid: *mut core::ffi::c_void,
}

/// Grow the KV pool to fit the step and upload every descriptor array it needs
/// — the fire's first phase.
///
/// Takes the disjoint `Shell` fields it writes (`kv`, `fire_arrays`) and reads
/// (`fire_alloc`, `fire_stream`) rather than `&mut Shell`, which would not
/// compile.
#[allow(clippy::too_many_arguments)]
fn kv_and_arrays(
    kv: &mut Option<KvState>,
    fire_arrays: &mut crate::fire::scratch::Scratch,
    format: crate::layout::KvCacheFormat,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
    step: StepArrays<'_>,
    rows: usize,
    requests: usize,
) -> Result<FireInputs, i32> {
    let StepArrays {
        token_ids,
        position_ids,
        kv_indices,
        kv_indptr,
        kv_lens,
        qo_indptr,
        required_kv_pages,
    } = step;
    let need_pages = required_kv_pages.max(kv_indices.iter().copied().max().map_or(1, |m| m + 1));
    let page_size: i32 = crate::boot::KV_PAGE_SIZE;
    // Re-derived here as well as in `kv_pools_for`: the attention plans below
    // want the same two numbers.
    let (kv_heads_i, head_dim_i) = (
        i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
        i32::try_from(model.deployment.shape.head_dim_kernel).unwrap_or(0),
    );
    let layers = kv_pools_for(
        kv,
        &mut fire_arrays.epoch,
        dep,
        model,
        need_pages,
        page_size,
        alloc,
        stream,
        format,
    )?;

    // The fire's descriptor arrays, pooled like the arena: a capture bakes an
    // address, so the buffer must be the same one next fire with only its
    // contents refreshed. Slots are positional.
    let d_ids = fire_arrays.upload_u32(alloc, slot::IDS, token_ids, stream.as_ref())?;
    let d_pos = fire_arrays.upload_u32(alloc, slot::POS, position_ids, stream.as_ref())?;
    let d_kv_indices =
        fire_arrays.upload_u32(alloc, slot::KV_INDICES, kv_indices, stream.as_ref())?;
    let d_kv_indptr = fire_arrays.upload_u32(alloc, slot::KV_INDPTR, kv_indptr, stream.as_ref())?;
    let d_kv_lens = fire_arrays.upload_u32(alloc, slot::KV_LENS, kv_lens, stream.as_ref())?;
    let d_qo = fire_arrays.upload_u32(alloc, slot::QO, qo_indptr, stream.as_ref())?;
    // A `d_sampled` UPLOAD STOOD HERE, filling `slot::SAMPLED` whenever the
    // fire did not sample every row, and reaching exactly one destination:
    // `FireStreams::sampling_indices`, which no door answered for. A fire
    // that states a gather is refused by `baker_fire` before the pointer
    // could be read ("this fire samples N of M rows; the lane states no
    // gather epilogue"), so this was work done for a case that cannot reach
    // it. `sampled_rows` itself stays: it is what that refusal compares.

    // Write targets: each request appends its new tokens at the CSR tail —
    // decode one token at `len - 1`, prefill its whole window ending there.
    let mut w_page = Vec::with_capacity(rows);
    let mut w_off = Vec::with_capacity(rows);
    for r in 0..requests {
        let pages = &kv_indices[kv_indptr[r] as usize..kv_indptr[r + 1] as usize];
        let total = (pages.len() as u32 - 1) * page_size as u32 + kv_lens[r];
        let toks = (qo_indptr[r + 1] - qo_indptr[r]) as usize;
        for t in 0..toks {
            let pos = total - toks as u32 + t as u32;
            w_page.push(pages[(pos / page_size as u32) as usize]);
            w_off.push(pos % page_size as u32);
        }
    }
    let d_w_page = fire_arrays.upload_u32(alloc, slot::W_PAGE, &w_page, stream.as_ref())?;
    let d_w_off = fire_arrays.upload_u32(alloc, slot::W_OFF, &w_off, stream.as_ref())?;
    // Pooled, because a capture bakes `row_valid_d`.
    let d_valid = fire_arrays.row_valid(alloc, rows, stream.as_ref())?;

    Ok(FireInputs {
        page_size,
        kv_heads_i,
        head_dim_i,
        layers,
        d_ids,
        d_pos,
        d_kv_indices,
        d_kv_indptr,
        d_kv_lens,
        d_qo,
        d_w_page,
        d_w_off,
        d_valid,
    })
}

/// One attention workspace, at the sizes the planner budgets for.
///
/// THE SIZES ARE THE BUDGET'S. `raise_attn_plans` allocates one of these per
/// attention class the lanes state and `Scratch` holds every one of them for
/// the driver's life, so `CheckpointCosts::attn_float_workspace_bytes`
/// multiplies its per-workspace figure by the class count
/// (`baker::Baked::attn_workspaces`). Both call sites wrote the numbers out;
/// there is one spelling now, and it is the one the budget reads.
///
/// # Errors
///
/// The device or its pinned staging could not be allocated.
fn attn_workspace<O: crate::fire::attention_workspace::StagingOps<Event = E>, E>(
    ops: &mut O,
) -> Result<crate::fire::attention_workspace::AttentionWorkspace<E>, i32> {
    use crate::layout::model_costs::{
        ATTN_FLOAT_WORKSPACE_BYTES, ATTN_INT_WORKSPACE_BYTES, ATTN_PLAN_STAGING_SLOTS,
    };
    crate::fire::attention_workspace::AttentionWorkspace::allocate(
        ops,
        usize::try_from(ATTN_FLOAT_WORKSPACE_BYTES).unwrap_or(usize::MAX),
        usize::try_from(ATTN_INT_WORKSPACE_BYTES).unwrap_or(usize::MAX),
        ATTN_PLAN_STAGING_SLOTS,
    )
    .map_err(i32::from)
}

/// The shapes an attention plan is raised against.
///
/// BUILT ONCE PER FIRE AND PASSED TWICE. It was built twice, from the same
/// six locals, seven lines apart — `raise_attn_plans` and
/// `publish_seam_pins` each got their own literal. `Copy` is what makes one
/// enough: every field is a shared slice or an `i32`, so the second pass is
/// a move of six words and neither callee can change what the other sees.
#[derive(Clone, Copy)]
struct PlanGeometry<'a> {
    kv_indptr: &'a [u32],
    kv_lens: &'a [u32],
    qo_indptr: &'a [u32],
    kv_heads: i32,
    head_dim: i32,
    page_size: i32,
}

/// The attention plans and workspaces a fire binds against.
///
/// Raw pointers and copies, deliberately: the plans live in `FireScratch` for
/// the driver's lifetime, and returning borrows would keep `state.scratch`
/// mutably borrowed across the rest of the fire.
struct AttnPlans {
    /// One raised decode schedule per CLASS the lane states, as the fire's
    /// view table takes them. A body asks by class
    /// (`kernels::raises::Class`), so this is a table and not a pointer.
    decode_plans: Vec<(kernels::raises::Class, *mut std::ffi::c_void)>,
    /// The same table for the PREFILL cache: one PRE-planned schedule per
    /// masked class the lane states, or — for a lane that states none — the
    /// single classless cache the planless leg carves into.
    prefill_plans: Vec<(kernels::raises::Class, *mut std::ffi::c_void)>,
    /// Does the lane state the flashinfer DECODE dispatch? Read by the score
    /// sink, which sizes a one-wide window for it.
    states_decode_dispatch: bool,
    /// Does it state `attention.prefill`, which plans INSIDE the fire out of
    /// the host CSR mirrors? Those mirrors are published for it.
    states_own_prefill: bool,
}

/// Allocate the workspaces on first fire, then raise the schedules this
/// lane's statements ask for.
///
/// # What changed when the legacy walk died
///
/// This used to read `lowered.preps` and raise EVERY schedule the geometry
/// permitted, "not just the one this fire's text states", because a union
/// capture recorded both arms of an attention guard and an arm whose plan was
/// never raised abandoned the capture. There is no union and no capture: a
/// `Program` is one lane, its statements are all issued, and
/// [`Baked::attn_ask`] reads exactly what they ask for.
///
/// # ONE SCHEDULE PER CLASS, and the workspace is why there is a Vec
///
/// The pair of decode schedules the legacy walk kept went with it, and came
/// back — as a set the lane's own statements measure rather than a pair the
/// lowering remembered. A FlashInfer plan writes its work list into the
/// workspace it was raised against, so each class owns one; they are
/// allocated on the fire that first states the class and kept for the
/// driver's life, which is what the single pair always did.
///
/// A lane's classes are the attention GEOMETRIES its text states — one for
/// every shipping SKU but gemma-4's two.
///
/// # AND THE MASKED ARM READS THE SAME TABLE
///
/// The paragraph above was written for decode while the prefill cache below
/// was still a fixed pair, and gemma's MASKED lane is the same tower saying
/// the same thing through the other arm: 35 statements at `(256, 512)` beside
/// 7 at `(512, 0)`, one lane, and a schedule planned at one geometry. So the
/// two loops are one loop said twice, and what separates them is only which
/// planner carves the work list.
///
/// The classless ANSWER is the seam between them. A decode lane with no decode
/// statement gets a schedule answered under `Class::ANY` and PLANNED (a null
/// one would turn a later absence into a fault); a lane with no masked
/// statement gets a cache answered under `Class::ANY` and only STAMPED,
/// because its reader — the planless prefill leg — carves it per statement out
/// of the host CSR mirrors. A lane that states masked therefore publishes no
/// classless prefill cache at all, and a lane stating both arms would be
/// refused at the bind by the `"fa2.prefill"` key rather than handed a cache
/// the other arm replans mid-fire. No text states both: gemma's three-way
/// `split` is a predicate over the fact word, so the masked lane states
/// `attention.masked` and neither sibling.
///
/// `Class::ANY` IS THE ANSWER'S KEY AND NOT THE ENTRY'S. `scratch` is keyed by
/// GEOMETRY — a workspace per `(head_dim, window, kv_heads)`, grown across
/// every lane of the model — so the fallback's entry is shared with a stated
/// class that happens to be the same geometry. gemma is exactly that case on
/// the prefill side: its fallback is `(512, 0, kv 2)`, which is its second
/// masked class, so a masked fire PLANS the entry an unmasked fire STAMPS.
/// Sound because a `Program` is one lane and a lane states one arm — each fire
/// (re)carves what it is about to read, which is what both loops do
/// unconditionally.
fn raise_attn_plans(
    scratch_slot: &mut Option<FireScratch>,
    model: &LoadedModel,
    ask: &crate::baker::AttnAsk,
    geom: PlanGeometry<'_>,
    raw_stream: *mut std::ffi::c_void,
) -> Result<AttnPlans, i32> {
    use crate::baker::DecodeClass;
    use crate::bind::{DecodePlan, PrefillPlan};
    use crate::fire::attention_workspace::LiveStagingOps;
    use crate::serve::state::{DecodeSchedule, PrefillSchedule};

    let PlanGeometry {
        kv_indptr,
        kv_lens,
        qo_indptr,
        kv_heads,
        head_dim,
        page_size,
    } = geom;
    let mut sops = LiveStagingOps;
    if scratch_slot.is_none() {
        *scratch_slot = Some(FireScratch {
            decode: Vec::new(),
            prefill: Vec::new(),
        });
    }
    let scratch = scratch_slot.as_mut().expect("just ensured");
    let states_decode_dispatch = !ask.decode.is_empty();
    let q_heads_i = i32::try_from(model.deployment.shape.q_heads).unwrap_or(0);

    // A LANE THAT STATES NO DECODE ATTENTION STILL GETS A SCHEDULE, at the
    // deployment's own widest head: the plan is cheap, and a null one would
    // turn a later statement's absence into a fault rather than a refusal.
    // It is raised under `Class::ANY`, which is the ask nothing makes — a
    // decode body always names its statement's class — so it is answered only
    // by a key lookup that finds nothing else.
    let fallback = DecodeClass {
        head_dim: head_dim.max(0).unsigned_abs(),
        window: 0,
        kv_heads: kv_heads.max(0).unsigned_abs(),
    };
    let wanted: Vec<DecodeClass> = if ask.decode.is_empty() {
        vec![fallback]
    } else {
        ask.decode.clone()
    };

    let mut decode_plans = Vec::with_capacity(wanted.len());
    for want in wanted {
        if !scratch.decode.iter().any(|s| s.class == want) {
            // ONE PER CLASS, WHICH IS WHAT THE PLANNER NOW CHARGES FOR. The
            // three sizes are `layout::model_costs`' constants rather than
            // literals: the budget term and the allocation were two spellings
            // of `32 << 20` that had to agree, with nothing making them.
            let ws = attn_workspace(&mut sops)?;
            scratch.decode.push(DecodeSchedule {
                class: want,
                ws,
                plan: DecodePlan::new(),
            });
        }
        let held = scratch
            .decode
            .iter_mut()
            .find(|s| s.class == want)
            .expect("just ensured");
        held.ws.begin_plan_update(&mut sops)?;
        // `enable_cuda_graph = true` on the raise: the padded batch size stays
        // constant between fires, which is what the flag buys and what a
        // future capture of this walk will need. It costs nothing to an eager
        // fire.
        //
        // THE STATED VARIANT RIDES ALONG, and dropping it was once a silent
        // numerics bug: `plan_decode` hardcodes `full_attention_variant =
        // false`, so a stack with NO sliding window — every llama, qwen3 and
        // mistral — planned the windowed schedule and every decode ran the
        // wrong kernel. The class says which variant it wants and this passes
        // it; `attn::variant_agrees` holds the answer to it at the body.
        //
        // `window_left = -1` UNCHANGED, and it is not the class's window: the
        // decode planner reads that argument in exactly one place, an
        // env-gated split-kv choice, and the WINDOW the launch attends over is
        // the statement's own `Const<i32>` (`attn::window_left`). What the
        // schedule carries about the window is the variant.
        held.plan.plan_decode_variant(
            kv_indptr,
            q_heads_i,
            i32::try_from(want.kv_heads).unwrap_or(kv_heads),
            i32::try_from(want.head_dim).unwrap_or(head_dim),
            page_size,
            held.ws.view(),
            raw_stream,
            true,
            want.full(),
            -1,
        );
        held.ws.end_plan_update(&mut sops, raw_stream)?;
        decode_plans.push((
            if ask.decode.is_empty() {
                kernels::raises::Class::ANY
            } else {
                want.class()
            },
            held.plan.as_ptr(),
        ));
    }

    // THE PREFILL CACHE IS RAISED EITHER WAY, and only PLANNED for the masked
    // arm. `attention.prefill`'s body plans its own schedule into this cache
    // per statement (`fa2::plan_own_prefill`), which is why the workspace is
    // stamped on the leg that does not plan here: a cache with a null
    // workspace would have the planless leg carving into nothing.
    //
    // SO THE LOOP ABOVE, WITH THAT ONE BRANCH IN IT. A lane that states the
    // masked arm gets one PLANNED schedule per class it states; a lane that
    // states none gets the one classless STAMPED cache it has always had, at
    // the same allocation and in the same place.
    let masked = !ask.masked.is_empty();
    let wanted: Vec<DecodeClass> = if masked {
        ask.masked.clone()
    } else {
        vec![fallback]
    };

    let mut prefill_plans = Vec::with_capacity(wanted.len());
    for want in wanted {
        if !scratch.prefill.iter().any(|s| s.class == want) {
            let ws = attn_workspace(&mut sops)?;
            scratch.prefill.push(PrefillSchedule {
                class: want,
                ws,
                plan: PrefillPlan::new(),
            });
        }
        let held = scratch
            .prefill
            .iter_mut()
            .find(|s| s.class == want)
            .expect("just ensured");
        if masked {
            held.ws.begin_plan_update(&mut sops)?;
            // `window_left = -1` FOR THE DECODE LOOP'S REASON, and the masked
            // body holds it to that by name. The prefill planner reads the
            // argument in exactly one place — the split-kv chunk-size search
            // — where a shorter effective kv length is the OPTIMISATION and
            // the whole prefix is the conservative answer; the window the
            // launch attends over is the statement's own `Const<i32>`, ANDed
            // with the caller's mask bit in one `LogitsMask`. A schedule
            // carved for a narrower reading than the launch attends over
            // would leave the chunks outside it unscheduled, which is silent.
            //
            // WHICH IS WHY THE CLASS CARRIES A WINDOW THE PLAN DOES NOT. The
            // class is the statement's own `(head_dim, window)` — it says
            // WHICH schedule a body means, and gemma's two masked geometries
            // are two entries because their statements are two classes.
            held.plan.plan_prefill(
                qo_indptr,
                kv_indptr,
                kv_lens,
                q_heads_i,
                i32::try_from(want.kv_heads).unwrap_or(kv_heads),
                i32::try_from(want.head_dim).unwrap_or(head_dim),
                page_size,
                held.ws.view(),
                raw_stream,
                true,
                -1,
            );
            // The fence is the point: `end_plan_update` records the event
            // that says the schedule upload landed, so a launch cannot read a
            // schedule that is not there yet.
            held.ws.end_plan_update(&mut sops, raw_stream)?;
        } else {
            held.plan.stamp_workspace(held.ws.view());
        }
        prefill_plans.push((
            if masked {
                want.class()
            } else {
                kernels::raises::Class::ANY
            },
            held.plan.as_ptr(),
        ));
    }

    Ok(AttnPlans {
        decode_plans,
        prefill_plans,
        states_decode_dispatch,
        states_own_prefill: ask.states_own_prefill,
    })
}

/// Every resident plane the fire publishes for its views.
struct SeamPins {
    /// The observed-rows CSR the `"attn.score"` view carries.
    d_score_indptr: *const i32,
    d_mask: *const u8,
    d_mask_indptr: *const i32,
}

/// Stage the resident planes this fire's views answer with: the attention
/// mask and the attention-score observation.
///
/// # What left, and what stayed
///
/// The seam-pin walk STOOD HERE — one pooled device buffer per `Arg::Named`
/// the legacy lowering placed, sized `rows * width * 4`, zeroed every fire.
/// A `Program`'s values live in its own arena at offsets the walk assigned
/// (`baker::fire::Fire::rect`), so there is nothing to publish per value.
///
/// The two that stayed are the ones a RUNTIME OBJECT carries, and both are
/// the driver's own policy rather than any statement's:
///
/// * `"attention_mask"` — published on every fire: the CALLER'S mask when
///   the frame carries one, and the plan's own causal mask when it does
///   not. Which planner runs is the only difference; the plane, the CSR and
///   the view are the same either way, and `MaskView::enabled` stays the
///   POINTER'S PRESENCE, so a claim body reading the view through
///   `FireViews::raised` gets this fire's own mask rather than a null.
/// * `"attn.score"` — the per-request CSR of observed rows plus the
///   boot-configured window (`ScoreView`). The sink BLOCK is still sized and
///   allocated around that CSR, and its score/folded halves have no writer
///   any more: the legacy attention arms took them as loose operands, and no
///   point declares an observation output yet. That is the honest state of
///   the hook — the channel is answered, the buffer is waiting for a
///   declaration — and it is a few KB a fire.
#[allow(clippy::too_many_arguments)]
fn publish_seam_pins(
    fire_arrays: &mut crate::fire::scratch::Scratch,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
    model: &LoadedModel,
    step: &driver_api::StepSubmission,
    geom: PlanGeometry<'_>,
    states_decode_dispatch: bool,
    // How many score rows the sink keeps — `crate::boot`'s, so the one parse of
    // the knob reaches here.
    attn_score_window: u32,
) -> Result<SeamPins, i32> {
    let PlanGeometry {
        kv_indptr,
        kv_lens,
        qo_indptr,
        page_size,
        ..
    } = geom;
    let score_window = if states_decode_dispatch {
        1
    } else {
        attn_score_window
    };
    let sink = crate::fire::scratch::plan_score_sink(
        kv_indptr,
        kv_lens,
        page_size,
        model.deployment.shape.q_heads,
        score_window,
    );
    let d_score_indptr = match sink {
        // A sink too large to publish (the prefill window grows with context)
        // keeps the old answer: null, and the view says it observes nothing.
        None => core::ptr::null(),
        Some(p) => {
            let base = fire_arrays.score(alloc, &p, stream.as_ref())?;
            unsafe { base.cast::<u8>().add(p.indptr_offset) }
                .cast::<i32>()
                .cast_const()
        }
    };

    // A CALLER'S MASK IS SERVED, AND THE REFUSAL MOVED TO THE LANE. What
    // stood here refused the FLAG — because `attention.masked` had no arm to
    // read a staged mask with, so staging one meant staging it and attending
    // causally anyway. It has one now (a claim body reading the raise door;
    // the mask is not an operand), so the question is no longer whether this
    // driver can serve a mask but whether THIS TEXT states an arm that does:
    // `baker::word_of` asks that when it picks the lane, and refuses there,
    // naming the text. By this line the lane already reads a mask.
    //
    // The two planners' `None` mean opposite things, which is why this is a
    // branch and not a fallback. `plan_causal` returning `None` is an empty
    // fire with nothing to publish — a null `MaskView`, which every fa2 arm
    // reads as "no mask". `from_words` returning `None` is the fire's rows
    // and the caller's table DISAGREEING, and falling back to causal there is
    // the one answer that would look right, so it is a refusal.
    let element_mask = if step.plan.has_user_mask {
        let words = step.plan.bitmask_words();
        let Some(planned) = crate::fire::page_mask::element_mask::from_words(
            qo_indptr,
            kv_indptr,
            kv_lens,
            page_size,
            &words.request_indptr,
            &words.word_indptr,
            &words.words,
        ) else {
            eprintln!(
                "[driver-cuda] launch: this frame's mask table does not describe \
                 its rows — a row count, a bitset shorter than its own KV extent, \
                 or more mask bytes than a fire may hold. Refusing rather than \
                 attending causally, which is what a fallback would look like."
            );
            return Err(PIE_STATUS_UNSUPPORTED);
        };
        Some(planned)
    } else {
        crate::fire::page_mask::element_mask::plan_causal(qo_indptr, kv_indptr, kv_lens, page_size)
    };
    let (d_mask, d_mask_indptr) = match element_mask {
        None => (core::ptr::null(), core::ptr::null()),
        Some(p) => {
            let base = fire_arrays.mask(alloc, &p, stream.as_ref())?;
            (
                base.cast::<u8>().cast_const(),
                unsafe { base.cast::<u8>().add(p.indptr_offset) }
                    .cast::<i32>()
                    .cast_const(),
            )
        }
    };

    Ok(SeamPins {
        d_score_indptr,
        d_mask,
        d_mask_indptr,
    })
}

/// Where a sampling program reaches into the shell.
///
/// A struct of disjoint `Shell` fields, not `&mut Shell`: the phase writes
/// `ptir_control` and `ptir_sessions` while reading `model` and the lowering.
struct SamplingSites<'a> {
    instances: &'a std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &'a std::collections::BTreeMap<u64, ChannelState>,
    programs: &'a crate::program::Programs,
    control: &'a mut Option<crate::program::Control>,
    sessions: &'a mut std::collections::BTreeMap<u64, crate::program::session::Session>,
    rings: &'a mut crate::program::channel::Rings,
    disk: &'a crate::program::Disk,
    device_ordinal: i32,
    /// Where the fire left its logits; a sampler reads its own row out of it.
    logits: Option<&'a crate::device::DeviceBuffer>,
}

/// Run each request's sampling program over its own row, and report which
/// requests still need raw logits.
///
/// A frame can be mixed — one request bound to a program, another not — so the
/// result is a set, not a flag. No program, a decline, inputs not ready, or
/// channels this shell does not hold all fall through to raw logits.
#[allow(clippy::too_many_arguments)]
fn run_sampling_programs(
    sites: SamplingSites<'_>,
    model: &LoadedModel,
    request_instances: &[u64],
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
    qo_indptr: &[u32],
    sampled_rows: &[u32],
    rows: usize,
) -> Result<Vec<usize>, i32> {
    let SamplingSites {
        instances,
        channels,
        programs,
        control,
        sessions,
        rings,
        disk,
        device_ordinal,
        logits,
    } = sites;
    let vocab = model.deployment.shape.vocab;
    let logits_base = logits.map_or(0, |b| b.as_ptr() as u64);
    // A ZERO BASE IS A SAMPLER READING ADDRESS ZERO, and it fails silently:
    // every request draws from whatever is there, which is why a forward pass
    // whose logits are provably right can still emit token 0 forever.
    if std::env::var_os("PIE_TRACE_VALUES").is_some() {
        eprintln!("[readout] logits_base={logits_base:#x} vocab={vocab}");
    }
    // Every request over its own row: request `r`'s logits row is the last of
    // its token span, `qo_indptr[r + 1] - 1` (`r` only on a decode). Still one
    // lane per fire, so this is N single-lane fires, not one N-lane fire.
    // The step's instances, one per wire request (`request_instances`), not
    // the frame's roster.
    let instance_ids = request_instances;
    // Which requests still need raw logits: those whose program did not
    // publish. A mixed frame serves each half, so this is a set.
    let mut unsampled: Vec<usize> = Vec::new();
    for (r, &iid) in instance_ids.iter().enumerate() {
        let Some(&end) = qo_indptr.get(r + 1) else {
            break;
        };
        // The ordinal, not the row (see `logits_row_of`): a sampling program
        // reads the same compacted buffer the raw readback does.
        let row = logits_row_of(end as usize, rows, &sampled_rows);
        if run_program(
            instances,
            channels,
            programs,
            control,
            sessions,
            rings,
            disk,
            device_ordinal,
            iid,
            (logits_base, vocab, vocab),
            rows,
            row,
            alloc,
            stream,
        )? {
            continue;
        }
        unsampled.push(r);
    }
    Ok(unsampled)
}

/// Which instance owns each of the step's wire requests.
///
/// `frame.instance_ids` is the frame's roster and a step's `roster_rows` are
/// indices into it, so request `r`'s instance is not `instance_ids[r]` unless
/// the step uses the whole roster in order (a single-slot frame does).
/// `program_row_indptr` is the attribution the wire states; absent, a roster
/// the length of the request list is one request per member in order, and a
/// differently-sized one falls back to the frame's roster.
#[cfg(feature = "abi")]
fn request_instances(
    frame: &FrameSubmission,
    step: &driver_api::StepSubmission,
    requests: usize,
) -> Vec<u64> {
    let roster = frame.instance_ids.as_slice();
    let rows = step.roster_rows.as_slice();
    let mut out: Vec<u64> = (0..requests)
        .map(|r| roster.get(r).copied().unwrap_or(0))
        .collect();
    if step.program_row_indptr.len() >= 2 {
        for (member, &row) in rows.iter().enumerate() {
            let Some(&id) = roster.get(row as usize) else {
                continue;
            };
            let Some((first, last)) =
                crate::fire::envelope::member_requests(&step.program_row_indptr, member, requests)
            else {
                continue;
            };
            for slot in out.iter_mut().take(last.min(requests)).skip(first) {
                *slot = id;
            }
        }
    } else if rows.len() == requests {
        for (r, &row) in rows.iter().enumerate() {
            if let Some(&id) = roster.get(row as usize) {
                out[r] = id;
            }
        }
    }
    out
}

/// Read the step's device-resolved descriptors and translate its pages.
///
/// `None` when the step's own tables are already the fire — every member is
/// host class and the frame states no page translation.
///
/// An empty descriptor channel is a refusal here, not a retry: `launch_impl`
/// fires the frame's steps in order and synchronizes each one that does not
/// carry the completion, and `Session::fire` synchronizes after its regions, so
/// by the time a slot is composed every earlier slot's epilogue has committed.
/// An empty cell is a chain the guest built that the driver did not walk.
#[cfg(feature = "abi")]
fn compose_step(
    state: &mut Shell,
    frame: &FrameSubmission,
    step: &driver_api::StepSubmission,
) -> Result<Option<Box<driver_api::StepSubmission>>, i32> {
    use crate::fire::envelope::{Composed, Sites, compose};

    let Some(stream) = state.fire_stream.as_ref() else {
        return Ok(None);
    };
    let page = state.facts.page_size;
    // A shell with no registry has ringed no instance, so it has no
    // device-class member to resolve. `compose` applies the translation half
    // but needs a registry to borrow, and building one here would allocate for
    // a frame that names no program.
    let Some(rings) = state.ptir_rings.as_mut() else {
        return Ok(None);
    };
    let composed = compose(
        Sites {
            instances: &state.instances,
            channels: &state.channels,
            plans: &state.ptir_plans,
            sessions: &mut state.ptir_sessions,
            rings,
        },
        frame,
        step,
        page,
        stream.as_ref(),
    );
    match composed {
        Ok(Composed::Wire) => Ok(None),
        Ok(Composed::Ready(step)) => Ok(Some(step)),
        Ok(Composed::Early {
            instance,
            channel,
            port,
        }) => {
            eprintln!(
                "[driver-cuda] launch: instance {instance}'s {port:?} port names channel \
                 {channel}, whose ring holds no value — and every earlier slot of this \
                 frame has already run and synchronized, so nothing later will fill it"
            );
            Err(PIE_STATUS_INVALID_ARGUMENT)
        }
        Err(why) => {
            eprintln!(
                "[driver-cuda] launch: this step's geometry cannot be composed: {}",
                why.0
            );
            Err(PIE_STATUS_INVALID_ARGUMENT)
        }
    }
}

/// Ring every instance the frame names, before the forward runs.
///
/// Not lazy: a channel cell's address comes from a session's `Rings`, and
/// `fwd.adapter` puts its `lora` sink in the program's prologue, which runs
/// before the forward and so needs the address up front.
///
/// Failures are noted and swallowed: a frame whose instance cannot be ringed
/// still has a forward to run and raw logits to deliver, and `run_program`
/// declines when it finds no session.
fn ensure_sessions(state: &mut Shell, frame: &FrameSubmission) {
    // The stream and allocator are separate fields on purpose: grouping them
    // collapses a disjoint borrow the fire path depends on.
    let (Some(alloc), Some(stream)) = (state.fire_alloc.as_ref(), state.fire_stream.as_ref())
    else {
        return;
    };
    let ids: Vec<u64> = frame.instance_ids.as_slice().to_vec();
    for id in ids {
        if state.ptir_sessions.contains_key(&id) {
            continue;
        }
        let Some(instance) = state.instances.get(&id) else {
            continue;
        };
        let Some(shapes) = instance_ring_shapes(instance, &state.channels) else {
            continue;
        };
        let channel_ids = instance.channel_ids.clone();
        // The registry first: a channel two instances name is registered once,
        // so the prefill's `tok_in` and the decode's `EmbedTokens` share one
        // ring and the decode reads what the prefill published.
        let rings = state.ptir_rings.get_or_insert_with(|| {
            crate::program::channel::Rings::new(alloc, &[], stream.as_ref())
                .expect("an empty registry allocates nothing that can fail")
        });
        let mut slots = Vec::with_capacity(channel_ids.len());
        let mut refused = None;
        for (dense, channel) in channel_ids.iter().enumerate() {
            if let Some(&slot) = state.ptir_channel_slots.get(channel) {
                slots.push(slot);
                continue;
            }
            match rings.register(alloc, shapes[dense], stream.as_ref()) {
                Ok(slot) => {
                    // The seed, at the one moment the ring is known empty: a
                    // channel a previous instance registered keeps its cells
                    // (the `continue` above), so re-seeding would overwrite
                    // what that instance published.
                    if let Some((_, wire)) = instance.seeds.iter().find(|(id, _)| id == channel) {
                        let shape = shapes[dense];
                        match crate::program::channel::wire_to_native(
                            shape.dtype,
                            shape.numel,
                            wire,
                        ) {
                            Ok(native) => {
                                if let Err(error) =
                                    rings.seed(slot as usize, 0, &native, stream.as_ref())
                                {
                                    refused = Some(error);
                                    break;
                                }
                                // And into the MIRROR, which the seed
                                // otherwise skips entirely.
                                //
                                // A seed rides the bind as a `ChannelValue`,
                                // out of band, so the cell above never travels
                                // through the engine-shared plane and neither
                                // cursor word moves. The engine counts it all
                                // the same -- `pipeline::channel::bind` sets
                                // `writer_tail = max(1)` for a seeded writer,
                                // because from the guest's side one cell HAS
                                // been put -- and then two things it owns are
                                // wrong at once, with no fault from either
                                // side:
                                //
                                // * The guest's next `put` asks
                                //   `writer_tail - head >= capacity`, reads the
                                //   head word at zero, and gets `1 - 0 >= 1` at
                                //   the default capacity of one. `Full`, and
                                //   forever: the only thing that moves `head`
                                //   is a `pull_channels` take out of a mirror
                                //   that is empty. The guest waits on a cell it
                                //   cannot stage, the driver waits on a fire
                                //   that is never submitted, and nothing logs.
                                // * `Channel::set` -- which is how a
                                //   latest-value port like `KvLen` is advanced,
                                //   since nothing consumes it -- refuses with
                                //   `Empty` unless the cell it replaces is
                                //   still committed (`committed_tail <= head`).
                                //   With an empty mirror there is no front to
                                //   replace and every `set` after the seed is
                                //   an error.
                                //
                                // Publishing it here fixes both, because it is
                                // the truth: the cell was produced by the guest
                                // and has not been consumed by anything. Tail
                                // moves, head does not. `tart-masked` is the
                                // gate -- it seeds `token_in` AND `set`s
                                // `klen`, so it needed both halves.
                                if let Some(entry) = state.channels.get(channel) {
                                    let mut plane = entry.host_plane();
                                    if plane.engine_writes() && !plane.publish(wire) {
                                        eprintln!(
                                            "[driver-cuda] launch: instance {id}'s seed for \
                                             channel {channel} did not fit the host mirror"
                                        );
                                    }
                                }
                            }
                            Err(why) => {
                                eprintln!(
                                    "[driver-cuda] launch: instance {id}'s seed for channel \
                                     {channel} does not decode: {why}"
                                );
                            }
                        }
                    }
                    state.ptir_channel_slots.insert(*channel, slot);
                    slots.push(slot);
                }
                Err(error) => {
                    refused = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = refused {
            eprintln!("[driver-cuda] launch: cannot ring instance {id}: {error}");
            continue;
        }
        match crate::program::session::Session::new(slots, shapes) {
            Ok(session) => {
                state.ptir_sessions.insert(id, session);
            }
            Err(error) => {
                eprintln!("[driver-cuda] launch: cannot ring instance {id}: {error}");
            }
        }
    }
}

/// The adapter phase: one lane per instance, and the state it stages.
///
/// Takes its fields, not the shell: `named_bufs` is a shared borrow of
/// `state.fire_arrays` and growing the pool is a unique one, so they cannot be
/// live together (the scratch is resolved before the closure for that reason).
#[allow(clippy::too_many_arguments)]
// `lora_phase` STOOD HERE — 130 lines that walked the frame's instances for
// adapter lanes, sized the xAᵀ scratch, resolved the correction's three
// operands off the lowering and staged the adapters into a device arena. It
// is deleted with the arm that fired what it staged; see the note above
// `SamplingSites` for the hook's full verdict.

// `TailCsrs`, `peel_word`, `tail_csrs`, `peel_tail_ctx` and
// `fire_experts_per_token` STOOD HERE.
//
// THE PEEL was the legacy lowering's answer to a fire whose rows disagreed
// about an axis — hooked rows, masked rows — under `GuardMode::Union`: lower
// both regions, split the fire at the first marked row, give the tail its own
// rebased CSRs and its own fa2 schedule, and let a device word decide which
// rows each form ran. `lowered.launches[..].peel` is where the split came
// from and there are no launches. A `Program` has one shape per lane and the
// lane is chosen by the fact word, so a fire is one region by construction.
//
// `fire_experts_per_token` counted a routed launch's fan-out off the lowering
// when the deployment did not state one; `program::sweep` reads `k` off the
// routes' own width (the MoE rows algebra), so the count is in the Program.

pub(crate) fn step_impl(
    state: &mut Shell,
    frame: &FrameSubmission,
    step: &driver_api::StepSubmission,
    // `owes` is the debt the frame's last step carries; `None` for earlier
    // steps, which owe nothing because a frame completes once. A step handed
    // one enqueues an async completion and does not synchronize; a step handed
    // `None` synchronizes, because the next step depends on it.
    owes: Option<(
        driver_api::completion::CompletionTarget,
        Vec<*mut driver_api::local::TerminalCell>,
    )>,
) -> Result<(), i32> {
    use crate::bind::AttnCtx;

    let t_head = std::time::Instant::now();
    // The mutation first, then the borrows: `ready_device_state` takes
    // `&mut Shell`, so every shared borrow below (`model`, the lane, the
    // stream, the allocator) must be taken after it. This is only the refusal —
    // a fire with no model loaded stops here, carrying no value across.
    if state.model.is_none() {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    ready_device_state(state)?;
    // Before the forward, so a prologue's channel cells have addresses.
    ensure_sessions(state, frame);
    // And before `admit`, the ordering this step exists for: a `DecodeEnvelope`
    // member's wire plan carries a zero token, a zero position and no KV tables
    // — `admit` refuses that (`kv_indptr.len() < 2`) because it is not a fire
    // until its descriptors are read off the channel rings. `compose` reads
    // them and translates each member's working-set pages to physical ones.
    let composed = compose_step(state, frame, step)?;
    let step = composed.as_deref().unwrap_or(step);
    let Admitted {
        class,
        rows,
        requests,
        sampled_rows,
    } = admit(state, step)?;

    let model = state.model.as_ref().ok_or(PIE_STATUS_INVALID_ARGUMENT)?;
    // Derived at load, read here. See `LoadedModel::deployment`.
    let dep = &model.deployment;
    let token_ids = step.plan.token_ids.as_slice();
    let position_ids = step.plan.position_ids.as_slice();
    let kv_indices = step.plan.kv_page_indices.as_slice();
    let kv_indptr = step.plan.kv_page_indptr.as_slice();
    let kv_lens = step.plan.kv_last_page_lens.as_slice();
    let qo_indptr = step.plan.qo_indptr.as_slice();

    // ── THE LANE, FIRST. ────────────────────────────────────────────────
    //
    // Before a byte is allocated, because everything below is sized from it:
    // the arena is `rows * program.row_pitch`, and the fa2 schedules are the
    // ones this lane's own statements ask for. A fire with no lane owes no
    // allocation and no upload.
    //
    // A load that could not build a lane REFUSED (`serve::load`), so `None`
    // here means a fire arrived before a load — an engine bug, named rather
    // than unwrapped. A lane the fact word does not reach is the other case
    // and is a real refusal: `qwen35-d0.8b`'s PREFILL lane states
    // `ssm.gated_delta_chunked`, which nothing claims yet, so a prefill fire
    // on that SKU says so and does not run something else.
    let baked = state.baker.as_ref().ok_or_else(|| {
        eprintln!("[driver-cuda] launch: no lane is built; nothing has been loaded");
        PIE_STATUS_INVALID_ARGUMENT
    })?;
    let program = match baked.lane(class, step.plan.has_user_mask) {
        Ok((_, p)) => p,
        Err(why) => {
            eprintln!("[driver-cuda] launch: {why}");
            return Err(PIE_STATUS_UNSUPPORTED);
        }
    };
    let ask = baked.attn_ask(program).map_err(|why| {
        eprintln!("[driver-cuda] launch: {why}");
        PIE_STATUS_UNSUPPORTED
    })?;
    let steps_in_lane = program.steps.len();
    sg_trace(|| format!("  head {:?} ({steps_in_lane} steps)", t_head.elapsed()));
    let mut phase = std::time::Instant::now();
    let mut lap = |what: &str| {
        sg_trace(|| format!("  {what} {:?}", phase.elapsed()));
        phase = std::time::Instant::now();
    };

    let stream = state.fire_stream.as_ref().expect("just ensured");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = state.fire_alloc.as_ref().expect("just ensured");

    let FireInputs {
        page_size,
        kv_heads_i,
        head_dim_i,
        layers,
        d_ids,
        d_pos,
        d_kv_indices,
        d_kv_indptr,
        d_kv_lens,
        d_qo,
        d_w_page,
        d_w_off,
        d_valid,
    } = kv_and_arrays(
        &mut state.kv,
        &mut state.fire_arrays,
        state.kv_format,
        dep,
        model,
        alloc,
        stream,
        StepArrays {
            token_ids,
            position_ids,
            kv_indices,
            kv_indptr,
            kv_lens,
            qo_indptr,
            required_kv_pages: frame.required_kv_pages,
        },
        rows,
        requests,
    )?;

    lap("kv+arrays");

    // The six numbers both raises are cut from, read once. See
    // [`PlanGeometry`] for why one value serves two callees.
    let geom = PlanGeometry {
        kv_indptr,
        kv_lens,
        qo_indptr,
        kv_heads: kv_heads_i,
        head_dim: head_dim_i,
        page_size,
    };

    // Workspace + plan caches: driver-lifetime, first-launch built.
    let AttnPlans {
        decode_plans,
        prefill_plans,
        states_decode_dispatch,
        states_own_prefill,
    } = raise_attn_plans(&mut state.scratch, model, ask, geom, raw_stream)?;

    // The walk's arena and the logits landing, both carved HERE rather than
    // at the walk, and the reason is a borrow: the delivery below takes a
    // shared borrow of `state.fire_arrays` and holds it, so growing a pooled
    // buffer after that point is a mutable borrow of a field already lent
    // out. Both sizes need only the lane and the row count, and `admit` has
    // settled the row count, so early is also correct.
    let arena_ptr = arena_for(program, &mut state.fire_arrays, alloc, rows)?;
    let logits_bytes = rows * dep.shape.vocab as usize * 2;
    state
        .fire_arrays
        .logits(alloc, logits_bytes.max(64))
        .map_err(i32::from)?;

    // A `"request_of_token"` DERIVE STOOD HERE, behind
    // `program.slots.iter().any(|s| matches!(s, Slot::Runtime(n) if n ==
    // "request_of_token"))` — a guard that cannot be true. `Slot::Runtime`
    // comes from `ValueDef::Runtime`, which `model-dsl`'s `Recorder::runtime`
    // is the only producer of, and its seven call sites in the whole tree
    // spell three names: `token_ids`, `positions`, `qo_indptr`.
    // `model-compiler` says the same in its own doc. So the block derived
    // nothing, uploaded nothing, and published a null — every fire, on every
    // catalog row.
    //
    // The name is still ASKED for, by key, from `kernels-cuda`'s
    // `pool.attention_lse`, and that ask has always met the null and refused.
    // It still refuses. What is gone is the appearance that some plan could
    // turn it on.
    let SeamPins {
        d_score_indptr,
        d_mask,
        d_mask_indptr,
    } = publish_seam_pins(
        &mut state.fire_arrays,
        alloc,
        stream,
        model,
        step,
        geom,
        states_decode_dispatch,
        state.boot.attn_score_window,
    )?;

    lap("attn-plan");
    // The hybrid's GDN context: driver-owned slabs, instance slots.
    let (gdn_ctx, _slot_ids_buf) = gdn_context(
        &mut state.gdn,
        &mut state.fire_arrays.epoch,
        dep,
        step,
        requests,
        alloc,
        stream,
    )?;

    // The KV/mask/score descriptors every view is cut from — a struct literal
    // because every field is already computed above.
    let attn = AttnCtx {
        layers,
        mask_d: d_mask,
        mask_indptr_d: d_mask_indptr,
        score_indptr_d: d_score_indptr,
        kv_page_indices_d: d_kv_indices.cast(),
        kv_page_indptr_d: d_kv_indptr.cast(),
        kv_last_page_lens_d: d_kv_lens.cast(),
        qo_indptr_d: d_qo.cast(),
        num_requests: requests as i32,
        num_pages_in_batch: kv_indices.len() as i32,
        max_pages_per_request: i32::try_from(
            kv_indptr
                .windows(2)
                .map(|w| w[1].saturating_sub(w[0]))
                .max()
                .unwrap_or(0),
        )
        .unwrap_or(0),
        w_page_d: d_w_page.cast(),
        w_off_d: d_w_off.cast(),
        row_valid_d: d_valid.cast(),
        score_window: state.boot.attn_score_window,
    };

    // One handle for the driver, its stream rebound per fire: creating and
    // destroying one per fire cost 3.2 ms.
    let mut cublas_ops = crate::device::cublas::LiveCublas;
    if state.cublas.is_none() {
        state.cublas = Some(crate::device::cublas::CublasHandle::create(
            &mut cublas_ops,
            raw_stream,
        )?);
    }
    let cublas = state.cublas.as_mut().expect("just ensured");
    cublas.set_stream(&mut cublas_ops, raw_stream)?;
    let cublas_handle = cublas.handle().expect("created").cast();

    // ── The per-fire VIEW ARENA (`bind::views`): every runtime object and
    // stream this driver answers, built once from the descriptors above.
    // Copies, not borrows -- the arena holds no reference into `state`.
    let streams = crate::bind::views::FireStreams {
        positions: d_pos.cast_mut().cast(),
        token_ids: d_ids.cast_mut().cast(),
        qo_indptr: d_qo.cast_mut().cast(),
        row_valid: d_valid,
        // The planless prefill's two host mirrors, published for the lane
        // that walks them — which is the lane that STATES `attention.prefill`,
        // not the lane that does not. That polarity was inverted here until
        // the prefill point's body was read for what it does: it always
        // carves its own schedule, and a null mirror is the refusal
        // `plan_own_prefill` opens with. No lane reached it (`baker_serve`
        // ingests one token at a time and the decode lane states neither),
        // which is why nothing caught it.
        qo_indptr_host: if states_own_prefill {
            qo_indptr.as_ptr()
        } else {
            core::ptr::null()
        },
        kv_page_indptr_host: if states_own_prefill {
            kv_indptr.as_ptr()
        } else {
            core::ptr::null()
        },
        prefill_plan_caches: prefill_plans,
        decode_plan_caches: decode_plans,
    };
    let views = crate::bind::views::FireViews::build(Some(&attn), gdn_ctx.as_ref(), streams);

    lap("views");
    // ── THE WALK. ───────────────────────────────────────────────────────
    //
    // Everything real is already built: `views` holds this fire's KV pages,
    // recurrent slabs and runtime planes; `attn` holds the descriptors they
    // were cut from; the fa2 schedule is raised; the logits buffer the
    // delivery reads is grown. The walk borrows all of it and owns one
    // thing, its arena.
    //
    // EAGER, AND THE PERF DEBT IS NAMED: one launch per `program.steps`
    // entry, every fire, with no captured exec to replay. See the note where
    // `capture_or_replay` stood.
    let result = baker_fire(
        baked,
        program,
        arena_ptr,
        state.fire_arrays.logits_buf(),
        &views,
        rows,
        requests,
        &sampled_rows,
        cublas_handle,
        raw_stream,
    );
    lap("run");
    // A step that owes nothing synchronizes, because the next step reads what
    // this one wrote. A step that owes the completion does not: its debt rides
    // a stream-ordered callback and this call returns with the work queued.
    let sync = if owes.is_some() && state.runahead {
        Ok(())
    } else {
        stream.as_ref().synchronize()
    };
    lap("sync");
    match (result, sync) {
        (Ok(_), Ok(())) => {}
        (Err(e), _) => {
            eprintln!("[driver-cuda] launch: refused: {e}");
            return Err(PIE_STATUS_UNSUPPORTED);
        }
        (_, Err(e)) => {
            eprintln!("[driver-cuda] launch: stream: {e:?}");
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
    }

    lap("post-match");
    // The frame's debt, built before the delivery below because it is owed
    // whether or not this fire has logits: paying it inside delivery would
    // leave a fire with no readout channel never publishing its terminal cells.
    let mut debt = owes.map(|(completion, cells)| FireDebt {
        staging: None,
        readouts: Vec::new(),
        vocab: model.deployment.shape.vocab as usize,
        cells,
        completion,
        broker: state.broker.clone(),
    });

    // Sampling: the instance's program, if it has one — before the delivery
    // below, because a program that published has already sent its answer and
    // raw logits beside it would deliver twice. No program, a decline, inputs
    // not ready, or missing channels all fall through to raw logits.
    let alloc = state
        .fire_alloc
        .as_ref()
        .expect("the fire allocator exists");
    // One instance per wire request, off the step's own roster.
    let request_instances = request_instances(frame, step, requests);
    // No registry means no instance this shell ringed, so every request falls
    // through to raw logits — the empty-roster case.
    let unsampled = match state.ptir_rings.as_mut() {
        None => (0..qo_indptr.len().saturating_sub(1)).collect(),
        Some(rings) => run_sampling_programs(
            SamplingSites {
                instances: &state.instances,
                channels: &state.channels,
                programs: &state.ptir_programs,
                control: &mut state.ptir_control,
                sessions: &mut state.ptir_sessions,
                rings,
                disk: state.ptir.disk(),
                device_ordinal: state.device_ordinal,
                logits: state.fire_arrays.logits_buf(),
            },
            model,
            &request_instances,
            alloc,
            stream,
            qo_indptr,
            &sampled_rows,
            rows,
        )?,
    };
    lap("sample");

    if !unsampled.is_empty() {
        deliver_logits(
            &state.instances,
            &state.channels,
            &mut state.logits_staging,
            &mut state.retired_staging,
            &request_instances,
            model,
            state.fire_arrays.logits_buf(),
            stream.as_ref(),
            rows,
            qo_indptr,
            &sampled_rows,
            &unsampled,
            &mut debt,
        )?;
    }

    // The debt goes last in stream order, so it runs after every launch and
    // after the D2H above.
    if let Some(d) = debt {
        let raw = Box::into_raw(Box::new(d)).cast::<std::ffi::c_void>();
        // One set of debts, two ways to pay them: with runahead off, this
        // thread pays after the synchronize above; with it on, a stream-ordered
        // callback pays and this call returns with the work queued.
        if !state.runahead {
            // The D2H above was enqueued, so paying here means waiting here —
            // without it the staging is read before the copy into it lands.
            stream.as_ref().synchronize()?;
            unsafe { retire_fire(raw) };
            return Ok(());
        }
        if let Err(e) = unsafe { stream.as_ref().host_fn(retire_fire, raw) } {
            // The callback never ran, so nothing will reclaim the box
            // or pay the debt — do both here rather than leak a frame
            // the runtime is waiting on.
            eprintln!("[driver-cuda] launch: cannot enqueue completion: {e:?}");
            let _ = stream.as_ref().synchronize();
            unsafe { retire_fire(raw) };
            return Err(PIE_STATUS_DRIVER_ERROR);
        }
        lap("debt");
        // A live debt with no entry is the one hole in "nothing in flight
        // proves every debt is paid": `host_fn` has succeeded, so the debt is
        // on the stream, but if `Event::new`/`record` fails no `InFlight` is
        // pushed and the next `ready_device_state` would clear `retired_staging`
        // under a callback that has not run. Pay inline instead.
        let done = match crate::device::Event::new()
            .and_then(|e| stream.as_ref().record(&e).map(|()| e))
        {
            Ok(e) => e,
            Err(_) => {
                let _ = stream.as_ref().synchronize();
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
        };
        state.in_flight.push_back(InFlight {
            done,
            // `d_valid` is pooled, so it is not here: handing a pooled buffer
            // to `InFlight` would free the pool.
            scratch: [_slot_ids_buf].into_iter().flatten().collect(),
            closed_channels: Vec::new(),
        });
    }
    lap("tail");
    Ok(())
}

// `mod peel_tests` and `mod lora_pin_tests` STOOD HERE — the unit gates on
// `peel_word`, `tail_csrs` and `lora_pin`, deleted with the three functions
// they held.
