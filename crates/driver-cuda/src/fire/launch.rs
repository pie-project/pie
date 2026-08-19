//! The forward path: one step, from a frame descriptor to logits.
//!
//! The largest of these modules and the one that earns it — `step_impl` is
//! the whole of a decode step, and the phases around it (admit, lower,
//! capture-or-replay, the GDN context, the KV pools, delivery) are its
//! parts. `.wiki/driver/graph.md` is about this file.

use crate::fire::scratch::slot;
use crate::serve::load::ptir_target;
use crate::serve::state::{
    ChannelState, FireDebt, FireScratch, GdnState, InFlight, InstanceEntry, KvState, LoadedModel,
    LoweredFire, LoweringKey, RUNAHEAD_DEPTH, Shell, digest_rows, instance_ring_shapes, retire,
    retire_fire,
};
use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR, PIE_STATUS_EXHAUSTED, PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_UNSUPPORTED,
};
use driver_api::submission::FrameSubmission;

/// The arena offset the attention dispatch at `fi` writes.
///
/// Two readings; only the op join (reading 1) is right under a union
/// lowering. The next launch's first operand is the o_proj only under
/// `Resolve`, where the guard has deleted every arm the fire did not take;
/// under `Union` every arm is present and the neighbour is some other body.
fn attention_landing(
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
    fi: usize,
) -> Option<usize> {
    use model_compiler::lower::Arg;
    match dplan.spec(fi).outs.first() {
        Some(Arg::Arena { at, .. }) => Some(*at),
        _ => match lowered.launches.get(fi + 1).map(|n| &lowered.args[n.args.start as usize]) {
            Some(Arg::Arena { at, .. }) => Some(*at),
            _ => None,
        },
    }
}

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
pub fn fire_class_of(
    _step: &driver_api::StepSubmission,
    rows: usize,
    requests: usize,
) -> Result<model_ir::trace::FireClass, i32> {
    use model_ir::trace::FireClass;
    Ok(if rows == requests { FireClass::Decode } else { FireClass::Prefill })
}

/// Replay this fire's bucket if it is captured, and capture it if not.
///
/// A capture must be taken warm: a launcher that allocates on first use
/// cannot allocate inside a capture, and the warm-up must walk a valid
/// program, so each variant warms once with its own resolved lowering. The
/// epoch is bumped whenever a pool grew — growth moves a base address out
/// from under a recorded launch, so a stale exec is recaptured, not replayed.
#[allow(clippy::too_many_arguments)]
fn capture_or_replay<R: crate::bind::Resolver>(
    cache: &mut crate::fire::recordings::Recordings,
    epoch: crate::fire::recordings::PlanEpoch,
    model_id: u64,
    plan: &model_ir::trace::ForwardPlan,
    rows_desc: &[model_compiler::lower::Row],
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
    frame: crate::bind::Frame,
    resolver: &mut R,
    ctx: &crate::bind::DispatchCtx,
    regions: crate::bind::AttnRegions<'_>,
    gdn: Option<&crate::bind::GdnCtx>,
    alloc: &mut crate::device::Allocator,
    preds: &mut crate::device::PredicateWord,
    stream: crate::device::StreamRef<'_>,
    requests: usize,
    rows: usize,
    class: model_ir::trace::FireClass,
) -> Result<usize, crate::bind::RunRefusal> {
    use crate::bind::{DispatchPlan, run};
    use crate::fire::recordings::{BucketKey, fire_predicates, union_eligibility};

    // SAFETY: the pointer is the `LoraFireState` `lora_phase` staged for this
    // fire, alive for the whole call — `step_impl` holds it in `lora_state`
    // past `capture_or_replay`.
    let lora = ctx.lora.map(|(s, _)| unsafe { &*s });
    let eligibility = union_eligibility(lora);
    let key = BucketKey::new(
        u32::try_from(requests).unwrap_or(0),
        u32::try_from(rows).unwrap_or(0),
        class,
        model_id,
    )
    // The bucket must carry the adapter shape: a capture bakes the adapter
    // device pointers, lane count and ranks, so two fires with the same
    // requests and tokens but different adapters must not share a bucket.
    .with_lora(lora.map_or(0, |l| l.capture_fingerprint));

    // The fire's own bits, the only thing that differs between two replays of
    // one exec. Not synchronized after: upload and replay are ordered on the
    // same stream.
    if fire_predicates(rows_desc, &lowered.conds, preds).is_err() || preds.upload(stream).is_err() {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }

    // What this fire would hand the graph, against what the graph recorded.
    let digest = crate::fire::recordings::capture_digest(ctx, regions, gdn);
    if cache.replay(key, epoch, digest, stream).unwrap_or(false) {
        sg_trace(|| format!("replay {key:?}"));
        return Ok(lowered.launches.len());
    }
    sg_trace(|| format!("miss {key:?} launches={}", lowered.launches.len()));

    // One warm fire per variant, each a resolved program — only variants this
    // fire can prepare: a `wants_scores` warm-up would lower the score-
    // capturing dispatch, which refuses without a score sink.
    for marks in [
        model_compiler::lower::Row { samples: true, ..Default::default() },
        model_compiler::lower::Row { samples: true, write_desc: true, ..Default::default() },
    ] {
        let warm_rows = vec![marks; rows];
        let Ok(warm) = model_compiler::lower::lower_with(
            plan,
            &warm_rows,
            model_compiler::lower::Fire { captures_across_splits: false },
            model_compiler::lower::GuardMode::Resolve,
        ) else {
            return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
        };
        // Resolve against the fire's own boot, not a default: a
        // `Boot::default()` states no KV dtype, so the warm-up would exercise
        // a different program from the one captured.
        let warm_dplan = DispatchPlan::with_boot(plan, &warm, dplan.boot());
        run(&warm, &warm_dplan, frame, resolver, ctx, regions, gdn)?;
        let _ = stream.synchronize();
    }

    let captured = {
        // Open on the fire's own allocator: a `cudaFree` defers only on the
        // allocator that owns the buffer, so a throwaway allocator would free
        // immediately inside the open capture and the graph would fault on
        // destroy.
        let Ok(scope) = alloc.begin_capture(stream) else {
            return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
        };
        let mut b = crate::device::SupergraphBuilder::new(scope.stream(), preds);
        let ran =
            crate::bind::run_captured(lowered, dplan, frame, resolver, ctx, regions, gdn, &mut b);
        // The retained nodes, taken before the builder is dropped: one per
        // launch, letting a later fire of a different row count retune this
        // exec's rectangles instead of recapturing.
        let nodes = b.nodes().to_vec();
        drop(b);
        // A refused capture is not a refused fire: some arms cannot be
        // recorded because their prepared state is something the fire declined
        // to build, so the capture is abandoned and the fire runs eager.
        let ended = scope.end();
        sg_trace(|| format!("capture ran={ran:?} ended_ok={}", ended.is_ok()));
        match (ran, ended) {
            (Ok(n), Ok(g)) => Some((n, g, nodes)),
            (Err(_), Ok(g)) => {
                // An abandoned capture must not be destroyed: its nodes are
                // already dropped and `cudaGraphDestroy` would fault inside
                // the driver (host segfault), so the template is leaked.
                // `ManuallyDrop`, not `mem::forget`, so the leak reads as one.
                let _leaked = std::mem::ManuallyDrop::new(g);
                None
            }
            (_, Err(_)) => None,
        }
    };
    let Some((ran, graph, nodes)) = captured else {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    };
    let Ok(exec) = graph.instantiate() else {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    };
    if exec.launch(stream).is_err() {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }
    let _ = cache.insert_with_nodes(key, exec, epoch, nodes, digest, eligibility);
    Ok(ran)
}

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

/// Trace a family's forward for one fire shape, lower it, and join the ops
/// back onto the launches.
///
/// Split out of `step_impl` so its result can be cached — nothing here reads
/// the fire's data, only its shape, which is what makes the answer reusable.
fn build_lowering(
    row: &'static dyn model::catalog::Variant,
    deployed: model::catalog::Deployed<'_>,
    class: model_ir::trace::FireClass,
    fire_rows: &[model_compiler::lower::Row],
    union_asked: bool,
    boot: crate::bind::Boot,
) -> Result<LoweredFire, i32> {
    use crate::bind::DispatchPlan;
    use model_compiler::lower::{Fire, GuardMode, lower_with};

    let plan = row.trace(class, deployed).map_err(|e| i32::from(crate::Error::from(e)))?;
    // `captures_across_splits` must follow the guard mode: with it clear the
    // host's counts are the truth, so a capture would bake this fire's split
    // into the graph — wrong only on replay. Under `Union` both regions lower
    // with `rows_device` launches and early-out on the `PeelWindowWord`, so
    // one exec serves every split and the split need not be a `BucketKey` axis.
    let lower_as = |g: GuardMode| {
        let captures_across_splits = g == GuardMode::Union;
        lower_with(&plan, fire_rows, Fire { captures_across_splits }, g).map_err(|e| {
            eprintln!("[driver-cuda] launch: uncovered: {e:?}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let union = union_asked;
    if !union {
        sg_trace(|| "union off at the gate".into());
    }
    let lowered = lower_as(if union { GuardMode::Union } else { GuardMode::Resolve })?;

    let dplan = DispatchPlan::with_boot(&plan, &lowered, boot);
    // The load-time refusal: unfireable symbols refuse here, before any
    // operand is bound, so a fire stays a straight line. Two classes reach
    // `unfireable` — `Undeclared` (no contract or row declares the symbol)
    // and `Unstated` (fn-world declares it but no bind can ever fire it). It
    // refuses in both guard modes: under `Union` every arm is issued during
    // capture, so an unfireable symbol fails there exactly as under `Resolve`.
    let unfireable = dplan.unfireable();
    if !unfireable.is_empty() {
        for u in unfireable {
            eprintln!("[driver-cuda] {}: cannot fire {u}", plan.family);
        }
        sg_trace(|| {
            format!(
                "refused at load: {} unfireable symbol(s): {}",
                unfireable.len(),
                unfireable.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")
            )
        });
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    // How much of this model still fires through the row world.
    let (rows_left, total) = dplan.sweep_progress();
    sg_trace(|| format!("routes: {rows_left}/{total} still row-world"));
    sg_trace(|| format!("built: launches={} union={union}", lowered.launches.len()));
    Ok(LoweredFire { plan, lowered, dplan, union })
}

/// The admit phase's result.
#[allow(clippy::too_many_lines)]
/// What a step must satisfy before anything is traced, lowered or allocated,
/// and the facts that survive the asking. Returns owned values, not slices
/// borrowed from `state`, which is `&mut` for the rest of the fire.
struct Admitted {
    /// The service class the row/request ratio implies.
    pub(crate) class: model_ir::trace::FireClass,
    /// Token rows in this step.
    pub(crate) rows: usize,
    /// Requests the step's CSR partitions those rows into.
    pub(crate) requests: usize,
    /// The rows the lowering resolves its guards against, read from the step's
    /// region table.
    pub(crate) fire_rows: Vec<model_compiler::lower::Row>,
}

/// See [`Admitted`].
#[cfg(feature = "abi")]
fn admit(
    state: &Shell,
    step: &driver_api::StepSubmission,
) -> Result<(Admitted, &'static dyn model::catalog::Variant), i32> {
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
    // `trace()` is the one question a `Deployment` does not answer, so the row
    // comes along — a `&'static` borrow of a const table.
    let Some(row) = model::catalog::find(model.id) else {
        eprintln!(
            "[driver-cuda] launch: this build has no catalog row named \
             {:?}, which cannot happen after a load that matched one",
            model.id
        );
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    };

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
    // The region table (the seriation's output); an empty one is the legacy
    // discipline, not a refusal. It arrives in wire-row space but is read in
    // token-row space: the engine counts one wire row per request, the
    // lowering one row per token. `qo_indptr` maps between them — wire row `i`
    // owns token rows `qo_indptr[i]..qo_indptr[i + 1]` — so the table is
    // translated here, not reinterpreted.
    let region_row_indptr: Vec<u32> = step
        .region_row_indptr
        .as_slice()
        .iter()
        .map(|&wire_row| {
            // Out of range must not be clamped — a clamp would describe the
            // wrong rows; pass it through so `rows_from_regions` refuses it.
            qo_indptr.get(wire_row as usize).copied().unwrap_or(u32::MAX)
        })
        .collect();
    let mut fire_rows = model_compiler::lower::rows_from_regions(
        rows,
        model_compiler::lower::Readouts {
            indices: step.plan.sampling_indices.as_slice(),
            indptr: step.plan.sampling_indptr.as_slice(),
            qo_indptr: qo_indptr.as_ref(),
        },
        &region_row_indptr,
        step.region_sig.as_slice(),
        step.region_k.as_slice(),
    )
    .map_err(|drift| {
        // Report the shapes: the variant alone does not locate the fault, and
        // a refusal that names neither structure sends its reader to the wrong
        // crate.
        eprintln!(
            "[driver-cuda] launch: the step's region table does not describe \
             its rows: {drift:?}; rows={rows} requests={requests} \
             region_row_indptr(wire)={:?} region_row_indptr(token)={:?} \
             region_sig.len()={} region_k.len()={} \
             sampling_indptr={:?} sampling_indices.len()={}",
            step.region_row_indptr.as_slice(),
            region_row_indptr.as_slice(),
            step.region_sig.as_slice().len(),
            step.region_k.as_slice().len(),
            step.plan.sampling_indptr.as_slice(),
            step.plan.sampling_indices.as_slice().len(),
        );
        PIE_STATUS_INVALID_ARGUMENT
    })?;
    // `multi_token` is derived from the CSR, not taken on trust:
    // `GuardPred::WindowOne` reads it, so a row that under-claims it puts a
    // ragged fire on the decode arm — the wrong kernel and wrong logits, not a
    // refusal. The region bit states the same fact, but an empty table is
    // legal (default point is `multi_token: false`), and `qo_indptr` cannot be
    // silent: a request with more than one token row is multi-token, so the
    // two are ORed rather than one replacing the other.
    for r in 0..requests {
        let (lo, hi) = (qo_indptr[r] as usize, qo_indptr[r + 1] as usize);
        if hi.saturating_sub(lo) > 1 {
            for row in fire_rows.get_mut(lo..hi.min(rows)).unwrap_or_default() {
                row.multi_token = true;
            }
        }
    }
    // No LoRA refusal here: the adapter is applied now, and a fire whose lanes
    // do not resolve gets the same correct no-op an adapter-free fire gets.

    // A family that does not declare a service class must be refused, not
    // traced: its text answers with `unreachable!`, and a panic across the
    // entry point costs the whole request. Only MTP composes these passes.
    if !matches!(class, FireClass::Decode | FireClass::Prefill) && dep.recurrent.is_none() {
        eprintln!(
            "[driver-cuda] launch: {class:?} is an MTP service pass and \
             this family declares no trace for it"
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    Ok((Admitted { class, rows, requests, fire_rows }, row))
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

    let Some(instance) = instances.get(&instance_id) else {
        return Ok(false);
    };
    let Some(compiled) = programs.get(instance.program_id) else {
        return Ok(false);
    };
    // The epilogue's plan, not the first: `plans.first()` is the epilogue only
    // by accident, and a package carrying an adapter puts its sink in a
    // prologue, so first would fire the adapter and never sample. Fall back to
    // the first stage for a package that states no kinds.
    let stage = compiled.stage_of_kind(crate::program::runtime::stage_kind::EPILOGUE).unwrap_or(0);
    let Some(plan) = compiled.plans.get(stage).cloned() else {
        return Ok(false);
    };
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
        &plan,
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
    sampled_rows.iter().position(|&s| s as usize == row).unwrap_or(row)
}

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
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
    named_bufs: &std::collections::BTreeMap<
        model_ir::trace::ValueId,
        crate::device::DeviceBuffer,
    >,
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
    use model_compiler::lower::Arg;
    // The last launch output that names a value — the logits buffer.
    let logits_value = (0..lowered.launches.len()).rev().find_map(|i| {
        dplan.spec(i).outs.first().and_then(|a| match a {
            Arg::Named { value, .. } => Some(*value),
            Arg::Arena { .. } | Arg::Weight(_) => None,
        })
    });
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
    if let (Some(lv), false) = (logits_value, readouts.is_empty())
        && let Some(buf) = named_bufs.get(&lv)
    {
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
    while state.in_flight.front().is_some_and(|f| f.done.is_complete().unwrap_or(true)) {
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
    epoch: &mut crate::fire::recordings::PlanEpoch,
    dep: &model::deployment::Deployment,
    step: &driver_api::StepSubmission,
    requests: usize,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
) -> Result<(Option<crate::bind::GdnCtx>, Option<crate::device::DeviceBuffer>), i32> {
    use crate::bind::GdnCtx;

    let mut gdn_ctx: Option<GdnCtx> = None;
    let mut _slot_ids_buf: Option<crate::device::DeviceBuffer> = None;
    if let Some(shape) = dep.recurrent.as_ref() {
        let (conv_stride, state_stride) = (shape.conv_stride, shape.state_stride);
        const GDN_SLOTS: u32 = 8;
        if (*gdn).is_none() {
            // The ported cache owns the layout: it pools the `(conv,
            // recurrent)` pairs and answers both strides.
            let is_linear: Vec<bool> =
                (0..dep.layers).map(|l| shape.linear_layers.contains(&l)).collect();
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
            let mut conv = alloc
                .alloc(usize::try_from(cache.layout().conv_total_bytes()).unwrap_or(0).max(1))?;
            let mut recurrent = alloc.alloc(
                usize::try_from(cache.layout().recurrent_total_bytes()).unwrap_or(0).max(1),
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
            let Ok(ops) = gdn_state.cache.reset_slot(i32::try_from(slot).unwrap_or(-1)) else {
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
        let need_buffer = rs_buffer_slot_ids.iter().copied().max().map_or(0, |m| m + 1);
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
            // mamba's B/C group count, off the statement.
            n_groups: shape.n_groups,
            // Still one base per model layer: pooling changed where a base
            // comes from, not what a launch is handed.
            conv_state: (0..gdn_state.is_linear.len()).map(|l| gdn_state.conv_base(l)).collect(),
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
    epoch: &mut crate::fire::recordings::PlanEpoch,
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
        model::deployment::KvStyle::Mla { .. }
        | model::deployment::KvStyle::CompressedPlane { .. } => {
            return Err(PIE_STATUS_UNSUPPORTED);
        }
    }
    let kv_heads_i = i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0);
    let n = dep.layers;
    // Per-layer geometry, family-decided: gemma-4's two layer kinds disagree
    // on head dim, and its trailing layers own no pages (they attend through
    // their source's).
    let per_layer = crate::pools::kv_cache::PerLayer {
        head_dim: dep.attention.iter().map(|a| a.head_dim as i32).collect(),
        kv_source_layer: dep.attention.iter().map(|a| a.kv_source as i32).collect(),
        num_kv_heads: vec![kv_heads_i; n as usize],
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
            KvState { cache, _held: held, num_pages: need_pages },
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
    /// Which rows carry a sampled logit, by fire-row index. Returned as well
    /// as uploaded because delivery indexes by sampled ordinal —
    /// `logits_row_of` needs the host copy after the fire.
    sampled_rows: Vec<u32>,
    d_ids: *const u32,
    d_pos: *const u32,
    d_kv_indices: *const u32,
    d_kv_indptr: *const u32,
    d_kv_lens: *const u32,
    d_qo: *const u32,
    /// The gather list, or null when every row samples and none is stated.
    d_sampled: *const u32,
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
    fire_rows: &[model_compiler::lower::Row],
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
    let page_size: i32 = 16;
    // Re-derived here as well as in `kv_pools_for`: the attention plans below
    // want the same two numbers.
    let (kv_heads_i, head_dim_i) = (
        i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
        i32::try_from(model.deployment.shape.head_dim_alloc()).unwrap_or(0),
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
    // Which rows the epilogue gathers, derived here rather than from
    // `sampling_indices` so the pointer and the guard that produced it cannot
    // disagree: both count the same `Row::samples`.
    let sampled_rows: Vec<u32> = fire_rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.samples.then_some(u32::try_from(i).unwrap_or(0)))
        .collect();
    let d_sampled = if sampled_rows.len() == rows {
        // Every row sampled means no gather is stated.
        core::ptr::null()
    } else {
        fire_arrays.upload_u32(alloc, slot::SAMPLED, &sampled_rows, stream.as_ref())?
    };

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
        sampled_rows,
        d_ids,
        d_pos,
        d_kv_indices,
        d_kv_indptr,
        d_kv_lens,
        d_qo,
        d_sampled,
        d_w_page,
        d_w_off,
        d_valid,
    })
}

/// The shapes an attention plan is raised against.
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
    decode_plan: *mut std::ffi::c_void,
    decode_plan_full: *mut std::ffi::c_void,
    prefill_plan: *mut std::ffi::c_void,
    workspace: crate::bind::abi::AttentionWorkspaceView,
    /// The workspace the prefill arm binds — the decode one for the planless
    /// family, because it never raised a prefill plan and a view of an
    /// unplanned workspace is not one a kernel may read.
    prefill_workspace: crate::bind::abi::AttentionWorkspaceView,
    /// Does the lowered text state the flashinfer DECODE dispatch? Read by
    /// the score sink, which sizes a one-wide window for it.
    states_decode_dispatch: bool,
    /// Does it plan its prefill INSIDE the fire? True when no arm states a
    /// prefill schedule, which is the claim `PrefillStyle::Planless` used to
    /// make one layer up.
    planless_prefill: bool,
    /// Does the stack keep a second decode schedule, one per layer kind?
    two_decode_kinds: bool,
}

/// Allocate the workspaces on first fire, then raise every plan the geometry
/// permits — not just the one this fire's text states: under `GuardMode::Union`
/// both arms of an attention guard are recorded, and a capture that walks an
/// arm whose plan was never raised is abandoned.
fn raise_attn_plans(
    scratch_slot: &mut Option<FireScratch>,
    model: &LoadedModel,
    lowered: &model_compiler::lower::Lowered,
    geom: PlanGeometry<'_>,
    raw_stream: *mut std::ffi::c_void,
) -> Result<AttnPlans, i32> {
    use crate::bind::{DecodePlan, PrefillPlan};
    use crate::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};

    let PlanGeometry { kv_indptr, kv_lens, qo_indptr, kv_heads, head_dim, page_size } = geom;
    let mut sops = LiveStagingOps;
    if scratch_slot.is_none() {
        let ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2)?;
        let prefill_ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2)?;
        // A third workspace for the peel tail: a plan writes into the
        // workspace it was raised against, and a peel launches both regions,
        // so the tail cannot plan into the prefix's.
        let tail_ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2)?;
        *scratch_slot = Some(FireScratch {
            ws,
            prefill_ws,
            tail_ws,
            decode_plan: DecodePlan::new(),
            decode_plan_full: DecodePlan::new(),
            prefill_plan: PrefillPlan::new(),
            tail_plan: DecodePlan::new(),
        });
    }
    let scratch = scratch_slot.as_mut().expect("just ensured");
    let (ws, prefill_ws, decode_plan, decode_plan_full, prefill_plan) = (
        &mut scratch.ws,
        &mut scratch.prefill_ws,
        &mut scratch.decode_plan,
        &mut scratch.decode_plan_full,
        &mut scratch.prefill_plan,
    );
    // What the text stated, not what its symbols imply and not the fire class:
    // a union trace carries both classes as guard arms, so a capture stands on
    // the schedules its arms state.
    use model_ir::trace::PrepKind;
    let mut decode_wanted: Vec<(u32, bool)> = Vec::new();
    let mut prefill_head_dim: Option<u32> = None;
    for p in &lowered.preps {
        match p.kind {
            PrepKind::DecodeAttention { head_dim, full_attention } => {
                if !decode_wanted.contains(&(head_dim, full_attention)) {
                    decode_wanted.push((head_dim, full_attention));
                }
            }
            // The widest stated, so a stack whose layers disagree plans the
            // schedule that addresses the most: one prefill plan is raised.
            PrepKind::PrefillAttention { head_dim } => {
                prefill_head_dim =
                    Some(prefill_head_dim.map_or(head_dim, |d: u32| d.max(head_dim)));
            }
        }
    }
    let states_decode_dispatch = !decode_wanted.is_empty();
    ws.begin_plan_update(&mut sops)?;
    let q_heads_i = i32::try_from(model.deployment.shape.q_heads).unwrap_or(0);
    // `enable_cuda_graph = true` on every raise: the deployment always
    // captures, so the padded batch size stays constant between fires.
    //
    // The PAIR is what a stack with two head dims states: the windowed arm and
    // the full-attention arm. `attn_plan` picks the full one on an unbounded
    // window, so the full-attention entry is the one that fills `_full`.
    let full = decode_wanted.iter().find(|(_, f)| *f).copied();
    let windowed = decode_wanted.iter().find(|(_, f)| !*f).copied();
    let decode_plan_full_ptr = match (windowed, full) {
        // Two schedules that differ in WIDTH -- the planner bakes the head dim,
        // so a stack running two needs two. Two that differ only in window are
        // one schedule: the width is what the plan is.
        (Some((d_win, _)), Some((d_full, _))) if d_win != d_full => {
            decode_plan.plan_decode_variant(
                kv_indptr, q_heads_i, kv_heads, d_win as i32, page_size,
                ws.view(), raw_stream, true, false, -1,
            );
            decode_plan_full.plan_decode_variant(
                kv_indptr, q_heads_i, kv_heads, d_full as i32, page_size,
                ws.view(), raw_stream, true, true, -1,
            );
            decode_plan_full.as_ptr()
        }
        // One schedule, whichever arm stated it. A text that states none is a
        // pure-prefill trace: `head_dim` keeps the geometry's answer so the
        // plan still stands for a capture whose other arm is a decode.
        //
        // THE STATED VARIANT RIDES ALONG, and dropping it was a silent
        // numerics bug: this arm called `plan_decode`, which hardcodes
        // `full_attention_variant = false`, so a stack with NO sliding window
        // -- every llama, qwen3 and mistral -- planned the windowed schedule,
        // answered `keys::Fa2DecodeFullAttention` with `false`, and
        // `decode_arm` fell through to `DecodeArm::Window`. The text says
        // which variant it wants (`PrepKind::DecodeAttention::full_attention`)
        // and the two-schedule arm above already honours it; this one threw
        // the answer away and every decode ran the wrong kernel.
        //
        // `windowed.or(full)`: where a text states BOTH at one width, one
        // schedule serves both and the windowed variant is the general case.
        _ => {
            let stated = windowed.or(full);
            let d = stated.map_or(head_dim, |(d, _)| d as i32);
            decode_plan.plan_decode_variant(
                kv_indptr, q_heads_i, kv_heads, d, page_size,
                ws.view(), raw_stream, true, stated.is_some_and(|(_, f)| f), -1,
            );
            core::ptr::null_mut()
        }
    };
    // A text whose prefill plans inside the fire states no prep.
    let planless_prefill = prefill_head_dim.is_none();
    if let Some(d) = prefill_head_dim {
        prefill_ws.begin_plan_update(&mut sops)?;
        prefill_plan.plan_prefill(
            qo_indptr, kv_indptr, kv_lens, q_heads_i, kv_heads, d as i32, page_size,
            prefill_ws.view(), raw_stream, true, -1,
        );
        // The fence is the point: `end_plan_update` records the event that
        // says the schedule upload landed, so a launch cannot read a schedule
        // that is not there yet.
        prefill_ws.end_plan_update(&mut sops, raw_stream)?;
    }
    ws.end_plan_update(&mut sops, raw_stream)?;

    Ok(AttnPlans {
        decode_plan: decode_plan.as_ptr(),
        decode_plan_full: decode_plan_full_ptr,
        prefill_plan: prefill_plan.as_ptr(),
        workspace: ws.view(),
        prefill_workspace: if planless_prefill { ws.view() } else { prefill_ws.view() },
        states_decode_dispatch,
        planless_prefill,
        two_decode_kinds: !decode_plan_full_ptr.is_null(),
    })
}

/// Every pin the seam publishes for one fire, resolved before the named
/// map is borrowed.
struct SeamPins {
    d_scores: *mut std::ffi::c_void,
    d_folded: *mut std::ffi::c_void,
    d_score_indptr: *const i32,
    d_mask: *const u8,
    d_mask_indptr: *const i32,
    /// The driver's own attention landing, or null when the family states
    /// its attention output as an SSA arg (gemma-4 does).
    d_attn_out: *mut std::ffi::c_void,
}

/// Size and publish the resident seam buffers this fire's arms may read,
/// unconditionally: `WantsAttnScore` and `HasCustomMask` are folded predicates,
/// so one exec serves the fire that wants a buffer and the fire that does not,
/// and a capture that walks an arm whose pin was never published is abandoned.
/// The cost is resident memory, not runtime — the untaken arm is skipped.
#[allow(clippy::too_many_arguments)]
fn publish_seam_pins(
    fire_arrays: &mut crate::fire::scratch::Scratch,
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    step: &driver_api::StepSubmission,
    named_widths: &std::collections::BTreeMap<model_ir::trace::ValueId, u32>,
    geom: PlanGeometry<'_>,
    rows: usize,
    states_decode_dispatch: bool,
    // How many score rows the sink keeps — `crate::boot`'s, so the one parse of
    // the knob reaches here.
    attn_score_window: u32,
) -> Result<SeamPins, i32> {
    let PlanGeometry { kv_indptr, kv_lens, qo_indptr, page_size, .. } = geom;
    for (&v, &w) in named_widths {
        // fp32-wide: the GDN seam pins are f32; llama-like's are bf16 and
        // simply leave half the pin unread.
        fire_arrays.named(alloc, v, rows * w as usize * 4, stream.as_ref())?;
    }
    // The score sink is published unconditionally: `WantsAttnScore` is a folded
    // predicate, so the capturing arm must be recordable whether this fire
    // wants scores or not.
    let score_window = if states_decode_dispatch { 1 } else { attn_score_window };
    let sink = crate::fire::attn_score::plan_score_sink(
        kv_indptr,
        kv_lens,
        page_size,
        model.deployment.shape.q_heads,
        score_window,
    );
    let (d_scores, d_folded, d_score_indptr) = match sink {
        // A sink too large to publish (the prefill window grows with context)
        // keeps the old answer: null, and the capturing arm declines.
        None => (core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null()),
        Some(p) => {
            let base = fire_arrays.score(alloc, &p, stream.as_ref())?;
            (
                base,
                unsafe { base.cast::<u8>().add(p.folded_offset) }.cast::<std::ffi::c_void>(),
                unsafe { base.cast::<u8>().add(p.indptr_offset) }.cast::<i32>().cast_const(),
            )
        }
    };

    // The custom mask, likewise unconditional (`HasCustomMask` is folded): with
    // nothing staged the resident form is plain causal, so taking the `_custom`
    // arm is correct, not merely safe. The caller's mask is used when there is
    // one, but refused — not replaced — when it does not describe this fire,
    // because attending causally over a supplied mask looks like a right answer.
    let staged = step.plan.has_user_mask.then(|| {
        let masks = step.plan.bitmask_words();
        crate::fire::page_mask::element_mask::from_words(
            qo_indptr,
            kv_indptr,
            kv_lens,
            page_size,
            &masks.request_indptr,
            &masks.word_indptr,
            &masks.words,
        )
        .ok_or_else(|| {
            eprintln!(
                "[driver-cuda] launch: this frame sets `has_user_mask` and its \
                 mask table does not describe its rows -- one mask per query \
                 row, each at least the request's KV extent. Refusing rather \
                 than attending causally, which would look like an answer."
            );
            PIE_STATUS_INVALID_ARGUMENT
        })
    });
    let element_mask = match staged {
        Some(r) => Some(r?),
        None => crate::fire::page_mask::element_mask::plan_causal(
            qo_indptr, kv_indptr, kv_lens, page_size,
        ),
    };
    let (d_mask, d_mask_indptr) = match element_mask {
        None => (core::ptr::null(), core::ptr::null()),
        Some(p) => {
            let base = fire_arrays.mask(alloc, &p, stream.as_ref())?;
            (
                base.cast::<u8>().cast_const(),
                unsafe { base.cast::<u8>().add(p.indptr_offset) }.cast::<i32>().cast_const(),
            )
        }
    };

    // The driver-owned attention landing, resolved before `named_bufs` borrows
    // the map: a fire whose op join names no output slot lands here instead of
    // losing its graph. Null when the family states its attention output as an
    // SSA arg (gemma-4 does).
    let d_attn_out = if dep.attn_output == model::deployment::AttnOutput::DriverPinned {
        fire_arrays.attn_out(
            alloc,
            rows * model.deployment.shape.q_heads as usize
                * model.deployment.shape.head_dim as usize
                * 2,
        )?
    } else {
        core::ptr::null_mut()
    };

    Ok(SeamPins { d_scores, d_folded, d_score_indptr, d_mask, d_mask_indptr, d_attn_out })
}

/// Which SSA value holds the attention query, and where its output lands.
/// Both are read off the lowering's join, not counted off launch positions,
/// which is false under `Union`. `(None, None)` for a family that states
/// [q, o] as SSA args (gemma-4).
///
/// Which SSA values the adapter correction reads and writes. `(q, v, x)`: the
/// two projection outputs the correction adds into (bound as `args[0]`/`[1]`)
/// and the projection input it reads — a foreign aux operand (`aux[0]`, bound
/// as `aux_slot(0)`), which is what makes it resolvable. `None` when the
/// lowering states no correction.
struct LoraPins {
    /// The q-site output rows.
    q: model_ir::trace::ValueId,
    /// The v-site output rows.
    v: model_ir::trace::ValueId,
    /// The projection input — normed value under `Pre`, residual stream under
    /// `Post`; the lowering knows which.
    x: model_ir::trace::ValueId,
}

fn lora_pins(
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
) -> Option<LoraPins> {
    use model_compiler::lower::Arg;
    let at = lowered
        .launches
        .iter()
        .position(|x| lowered.kernels[x.kernel as usize] == "gemm::lora_qkv_correction")?;
    let named = |a: &Arg| match a {
        Arg::Named { value, .. } => Some(*value),
        Arg::Arena { .. } | Arg::Weight(_) => None,
    };
    let mut args =
        lowered.launches[at].args.clone().filter_map(|ai| named(&lowered.args[ai as usize]));
    let q = args.next()?;
    let v = args.next()?;
    let x = dplan.spec(at).aux.first().and_then(named)?;
    Some(LoraPins { q, v, x })
}

fn attention_pins(
    dep: &model::deployment::Deployment,
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
    states_decode_dispatch: bool,
) -> Result<(Option<model_ir::trace::ValueId>, Option<usize>), i32> {
    use model_compiler::lower::Arg;
    // The guard-owned attention values, discovered from the lowering. gemma-4
    // has none: both its attention forms state [q, o] as SSA args, so the pins
    // stay null.
    if dep.attn_output != model::deployment::AttnOutput::DriverPinned {
        return Ok((None, None));
    }
    // Both decode spellings, as `states_decode_dispatch` above: a single-name
    // lookup would refuse a `_lse`-stating family outright, since the `else`
    // arm below is `PIE_STATUS_UNSUPPORTED`.
    let dispatch_names: &[&str] = if states_decode_dispatch {
        &[
            "attn::dispatch_attention_flashinfer_decode",
            "attn::dispatch_attention_flashinfer_decode_lse",
        ]
    } else {
        &["attn::dispatch_attention_flashinfer_prefill_bf16"]
    };
    let Some(fi) = lowered
        .launches
        .iter()
        .position(|x| dispatch_names.contains(&lowered.kernels[x.kernel as usize].as_str()))
    else {
        eprintln!("[driver-cuda] launch: the lowering states no {}", dispatch_names[0]);
        return Err(PIE_STATUS_UNSUPPORTED);
    };
    let q_pin = lowered.launches[fi].args.clone().find_map(|ai| match &lowered.args[ai as usize] {
        Arg::Named { value, .. } => Some(*value),
        _ => None,
    });
    // The dispatch's output, read off its op join — the stated read: the
    // attention statement carries its output placement, the slot the o_proj
    // reads. Preferred over the neighbour launch (the old positional read)
    // because positional breaks under `Union`, where every arm is present and
    // the next launch belongs to some other body. A join that names no slot is
    // not a refusal: `AttnCtx::o_out` is driver-owned, so the fire gets a
    // driver-owned landing and keeps its graph.
    Ok((q_pin, attention_landing(lowered, dplan, fi)))
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
    named_bufs:
        &'a std::collections::BTreeMap<model_ir::trace::ValueId, crate::device::DeviceBuffer>,
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
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
    request_instances: &[u64],
    alloc: &crate::device::Allocator,
    stream: &crate::device::OwnedStream,
    qo_indptr: &[u32],
    sampled_rows: &[u32],
    rows: usize,
) -> Result<Vec<usize>, i32> {
    use model_compiler::lower::Arg;
    let SamplingSites {
        instances,
        channels,
        programs,
        control,
        sessions,
        rings,
        disk,
        device_ordinal,
        named_bufs,
    } = sites;
    let vocab = model.deployment.shape.vocab;
    let readout = (0..lowered.launches.len()).rev().find_map(|i| {
        dplan.spec(i).outs.first().and_then(|a| match a {
            Arg::Named { value, .. } => Some(*value),
            Arg::Arena { .. } | Arg::Weight(_) => None,
        })
    });
    let logits_base = readout.and_then(|v| named_bufs.get(&v)).map_or(0, |b| b.as_ptr() as u64);
    // A ZERO BASE IS A SAMPLER READING ADDRESS ZERO, and it fails silently:
    // every request draws from whatever is there, which is why a forward pass
    // whose logits are provably right can still emit token 0 forever.
    if std::env::var_os("PIE_TRACE_VALUES").is_some() {
        eprintln!("[readout] value={readout:?} logits_base={logits_base:#x} vocab={vocab}");
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
        Ok(Composed::Early { instance, channel, port }) => {
            eprintln!(
                "[driver-cuda] launch: instance {instance}'s {port:?} port names channel \
                 {channel}, whose ring holds no value — and every earlier slot of this \
                 frame has already run and synchronized, so nothing later will fill it"
            );
            Err(PIE_STATUS_INVALID_ARGUMENT)
        }
        Err(why) => {
            eprintln!("[driver-cuda] launch: this step's geometry cannot be composed: {}", why.0);
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
                    if let Some((_, wire)) =
                        instance.seeds.iter().find(|(id, _)| id == channel)
                    {
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
fn lora_phase(
    programs: &crate::program::Programs,
    sessions: &std::collections::BTreeMap<u64, crate::program::session::Session>,
    rings: Option<&crate::program::channel::Rings>,
    instances: &std::collections::BTreeMap<u64, crate::serve::state::InstanceEntry>,
    scratch: &mut crate::fire::scratch::Scratch,
    lora_arena: &mut crate::fire::lora::LoraStageArena,
    tp_size: u32,
    frame: &FrameSubmission,
    qo_indptr: &[u32],
    stream: crate::device::StreamRef<'_>,
    raw_stream: *mut core::ffi::c_void,
    alloc: &crate::device::Allocator,
    model: &LoadedModel,
    dep: &model::deployment::Deployment,
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::bind::DispatchPlan,
    rows: usize,
) -> Option<(crate::fire::lora::LoraFireState, *mut core::ffi::c_void)> {
    // One lane per instance, and the token span is the request's own —
    // `qo_indptr[r]..qo_indptr[r+1]`, which is what makes an adapter apply to
    // the rows that asked for it and no others.
    let lora_lanes: Vec<crate::fire::lora::LoraLaneView> = frame
        .instance_ids
        .as_slice()
        .iter()
        .enumerate()
        .filter_map(|(r, &iid)| {
            let start = *qo_indptr.get(r)?;
            let end = *qo_indptr.get(r + 1)?;
            crate::fire::lora::lane_for_instance(
                programs,
                sessions,
                rings?,
                instances,
                iid,
                start,
                end.saturating_sub(start),
                stream,
            )
        })
        .collect();
    // The scratch is resolved first, outside the closure: `named_bufs` is a
    // shared borrow of `state.fire_arrays` and growing the pool is a unique
    // one, so the two cannot be live together.
    let lora_gate = if lora_lanes.is_empty() {
        core::ptr::null_mut()
    } else {
        scratch
            .attn_out(alloc, rows * model.deployment.shape.intermediate.max(1) as usize * 2)
            .unwrap_or(core::ptr::null_mut())
    };
    let named_bufs = &scratch.named;
    let lora_state =
        (!lora_lanes.is_empty()).then(|| lora_pins(lowered, dplan)).flatten().and_then(|pins| {
            let ptr = |v: model_ir::trace::ValueId| {
                named_bufs.get(&v).map(crate::device::DeviceBuffer::as_ptr)
            };
            let (q, v, x) = (ptr(pins.q)?, ptr(pins.v)?, ptr(pins.x)?);
            // The xAᵀ scratch, from the driver's own pool: it is not a value
            // any text states, and it is sized by the widest adapter in the
            // batch rather than by the fire.
            let gate = lora_gate;
            if gate.is_null() {
                return None;
            }
            let table = crate::fire::lora::LoraTable { lanes: &lora_lanes };
            let mut ops = crate::fire::lora::LiveLoraOps::new(raw_stream);
            let post = dep.norm == model::deployment::NormPlacement::Post;
            let stage_rows = crate::fire::lora::LoraStageRows {
                // Under post-norm the projection input is the residual stream,
                // under pre it is the normed value; both slots name the same
                // buffer here because the lowering resolved whichever one this
                // text states.
                y: x.cast_const(),
                norm_x: x.cast_const(),
                q,
                v,
                gate,
            };
            let (fingerprint, staged) = crate::fire::lora::stage_qkv_adapters(
                &mut ops,
                lora_arena,
                Some(&table),
                i32::try_from(model.deployment.layers).unwrap_or(0),
                i32::try_from(rows).unwrap_or(0),
                i32::try_from(model.deployment.shape.hidden).unwrap_or(0),
                i32::try_from(model.deployment.shape.q_heads).unwrap_or(0),
                i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
                i32::try_from(model.deployment.shape.intermediate).unwrap_or(0),
                i32::try_from(tp_size).unwrap_or(1),
                post,
                &stage_rows,
                false,
            )
            .ok()?;
            let _ = fingerprint;
            staged.map(|s| (s, gate))
        });
    lora_state
}

/// A peel tail's attention state: its own plan, its own rebased CSRs.
///
/// A peel's tail serves rows `[split, N)` — a different request count, so
/// FlashInfer's planner produces a different schedule and the fire's plan would
/// not describe the launch. The CSRs are rebased, not sliced: FlashInfer reads
/// a prefix sum starting at zero, so the tail's `kv_page_indptr` is the fire's
/// suffix minus its own first entry. Everything else the tail inherits.
#[allow(clippy::too_many_arguments)]
/// A peel tail's rebased CSRs — the part that needs no device.
#[derive(Debug, PartialEq, Eq)]
struct TailCsrs {
    /// Where the tail's pages begin inside the fire's index array.
    base: usize,
    /// The tail's page indptr, starting at zero.
    indptr: Vec<u32>,
    /// The tail's token indptr, starting at zero.
    qo: Vec<u32>,
}

/// Rebased, not sliced: `indptr[i]` counts the pages before request `split + i`
/// within the tail (the fire's suffix entry minus its entry at `split`).
/// Extracted from `peel_tail_ctx` because it is the half that can be silently
/// wrong — an off-by-one plans for the wrong requests rather than faulting.
///
/// The `[start, count]` a `_devwin` launch reads — the only statement of where
/// a peel splits. Two kernel forms read it differently: the prefix form runs
/// rows `[0, start)`, the tail form runs `[start, start + count)`. So an
/// unpeeled fire wants `(rows, 0)` — the prefix runs everything, the tail
/// nothing — which matters under `Union`, where both regions lower whether
/// this fire marks a row or not.
fn peel_word(
    fire_rows: &[model_compiler::lower::Row],
    axis: Option<model_ir::trace::PeelWindow>,
    rows: usize,
) -> (u32, u32) {
    let Some(axis) = axis else {
        // No peel in the lowering at all: one region, every row.
        return (u32::try_from(rows).unwrap_or(0), 0);
    };
    // The same predicate `lower::split_at` uses — two derivations of one split
    // is how they drift, and under `Union` both regions carry the whole window.
    let marked: fn(&model_compiler::lower::Row) -> bool = match axis {
        model_ir::trace::PeelWindow::HookFreePrefix => |r| r.hooked,
        model_ir::trace::PeelWindow::UnmaskedPrefix => |r| r.custom_mask,
    };
    let start = fire_rows.iter().position(marked).unwrap_or(fire_rows.len());
    (
        u32::try_from(start).unwrap_or(0),
        u32::try_from(fire_rows.len().saturating_sub(start)).unwrap_or(0),
    )
}

/// The fire's routed-expert fanout, read out of the lowered plan.
///
/// The fanout sizes a grid before any operand is read, so it cannot arrive as
/// an operand. It is read from the mixture statements' wire params, keyed on
/// the symbol — never by blind `params[k]`, because index `k` is `window_left`
/// on an attention dispatch, `w.width` on an unrouted `qmv`, a row pitch on a
/// strided copy. A statement whose layout is not in `ROUTED_FANOUT_AT` is
/// invisible here, which reads as a refusal downstream.
///
/// It must agree: a fire's mixture layers all route to the same fanout, so
/// disagreement is a misread plan, and the answer to it is `0` — absence,
/// which every reading rule refuses. A fire with no routed statement gets `0`
/// too, true for a dense model.
///
/// The seven symbols are metal's; no `kernels-cuda` fire matches one, so this
/// returns `0` for every CUDA fire. The construction site therefore reads
/// `model.deployment.shape.experts_per_token` first and calls this only as a
/// fallback when the deployment states nothing.
fn fire_experts_per_token(lowered: &model_compiler::lower::Lowered) -> i32 {
    /// Where a routed statement states its fanout, by symbol. Each entry is
    /// `(symbol, index into that launch's params)`, transcribed from the `dsl`
    /// constructor that emits it. `route_sort` and `route_gather` share one
    /// layout so the sort's padding and the gather's bounds cannot disagree;
    /// the routed GEMV is handled below because `dsl` builds its symbol by
    /// `format!` from the weight repr.
    const ROUTED_FANOUT_AT: &[(&str, usize)] = &[
        ("router_topk_bfloat16", 1),
        ("router_topk_scaled_bfloat16", 1),
        ("route_sort", 2),
        ("route_gather", 2),
        ("combine_sorted", 1),
    ];
    /// The routed GEMV's family: all three variants take the same params, so
    /// the index is one and the match is on the stem. A prefix, not a
    /// `contains`: `affine_qmv_routed` starts every routed GEMV symbol and
    /// nothing else, where a substring test could match a differently-laid-out
    /// symbol.
    const ROUTED_QMV_STEMS: &[&str] = &["mxfp4_qmv_routed", "affine_qmv_routed"];
    const ROUTED_QMV_FANOUT_AT: usize = 4;

    let mut seen: Option<u32> = None;
    for l in &lowered.launches {
        let Some(sym) = lowered.kernels.get(l.kernel as usize) else {
            continue;
        };
        let at = ROUTED_FANOUT_AT
            .iter()
            .find_map(|&(s, i)| (s == sym.as_str()).then_some(i))
            .or_else(|| {
                ROUTED_QMV_STEMS.iter().any(|s| sym.starts_with(s)).then_some(ROUTED_QMV_FANOUT_AT)
            });
        let Some(at) = at else { continue };
        // The run this launch's `params` names, then the slot inside it. A
        // launch whose run is shorter than the layout says is a launch this
        // table has misidentified, so it is skipped rather than read at a
        // clamped index -- reading `params[len-1]` because `params[4]` is out
        // of range is exactly the invented number this whole function avoids.
        let run = l.params.start as usize..l.params.end as usize;
        let Some(v) = lowered.params.get(run).and_then(|p| p.get(at).copied()) else {
            continue;
        };
        // A stated ZERO is not a fanout either, and it is what a text that
        // did not fill the slot leaves. Absence, uniformly.
        if v == 0 {
            continue;
        }
        match seen {
            None => seen = Some(v),
            // Disagreement is absence: the answer to a plan this driver has
            // misread is a refusal, not the first reading.
            Some(prev) if prev != v => return 0,
            Some(_) => {}
        }
    }
    seen.and_then(|v| i32::try_from(v).ok()).unwrap_or(0)
}

fn tail_csrs(kv_indptr: &[u32], qo_indptr: &[u32], split: usize) -> TailCsrs {
    let page0 = kv_indptr.get(split).copied().unwrap_or(0);
    let tok0 = qo_indptr.get(split).copied().unwrap_or(0);
    TailCsrs {
        base: page0 as usize,
        indptr: kv_indptr
            .get(split..)
            .unwrap_or(&[])
            .iter()
            .map(|p| p.saturating_sub(page0))
            .collect(),
        qo: qo_indptr.get(split..).unwrap_or(&[]).iter().map(|p| p.saturating_sub(tok0)).collect(),
    }
}

fn peel_tail_ctx(
    fire: &crate::bind::AttnCtx,
    scratch: &mut crate::fire::scratch::Scratch,
    plan: &mut crate::bind::DecodePlan,
    ws: &mut crate::fire::attention_workspace::AttentionWorkspace<
        cudarc::runtime::sys::cudaEvent_t,
    >,
    alloc: &crate::device::Allocator,
    stream: crate::device::StreamRef<'_>,
    raw_stream: *mut core::ffi::c_void,
    kv_indptr: &[u32],
    kv_lens: &[u32],
    kv_indices: &[u32],
    qo_indptr: &[u32],
    // The tail's first ROW, which is what the lowering states.
    split: usize,
    // The fire's row count, to tell rows from requests.
    rows: usize,
    // Whether the family keeps a second decode plan per layer kind.
    two_kind: bool,
    // Bytes per row of the guard-owned attention landing, and of the log-sum-
    // exp beside it. Both are pinned by the driver, so both are addressed from
    // a base the tail must advance past the prefix.
    o_row_bytes: usize,
    lse_row_bytes: usize,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    page_size: i32,
) -> Result<Option<crate::bind::AttnCtx>, i32> {
    // A split at zero is no split.
    if split == 0 {
        return Ok(None);
    }
    // One row per request, or this function cannot do its arithmetic: `split`
    // is a row index but `kv_indptr`/`kv_lens`/`qo_indptr` are per request. In
    // a decode they coincide; a prefill would mis-index silently, so it is a
    // refusal, not an assumption (a peeled prefill is not lowered today).
    if rows != kv_lens.len() || split >= kv_lens.len() {
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    // A two-kind family needs two tail plans and this builds one: gemma-4's
    // layer kinds disagree on head dim, so a tail with only the sliding plan
    // would serve its full layers from the wrong plan. Refuse rather than
    // half-serve.
    if two_kind {
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let TailCsrs { base, indptr: tail_indptr, qo: tail_qo } =
        tail_csrs(kv_indptr, qo_indptr, split);

    let d_indptr = scratch.upload_u32(alloc, slot::TAIL_INDPTR, &tail_indptr, stream)?;
    let d_lens = scratch.upload_u32(alloc, slot::TAIL_LENS, &kv_lens[split..], stream)?;
    let d_indices = scratch.upload_u32(
        alloc,
        slot::TAIL_INDICES,
        &kv_indices[base.min(kv_indices.len())..],
        stream,
    )?;
    let d_qo = scratch.upload_u32(alloc, slot::TAIL_QO, &tail_qo, stream)?;

    let mut sops = crate::fire::attention_workspace::LiveStagingOps;
    ws.begin_plan_update(&mut sops)?;
    plan.plan_decode(
        &tail_indptr,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        ws.view(),
        raw_stream,
        // `true`, with `raise_attn_plans`' four: a tail is planned against the
        // same workspace the prefix's capture replays over, so a tail raised as
        // if there were no graph would disagree with the capture digest.
        true,
        // `-1`, matching the fire's own `plan_decode`, not `fire.window_left`:
        // the window is per layer and the plan is raised unbounded, so a tail
        // planned against a different bound than the prefix is two regions of
        // one fire disagreeing about the cache.
        -1,
    );
    ws.end_plan_update(&mut sops, raw_stream)?;

    Ok(Some(crate::bind::AttnCtx {
        decode_plan: plan.as_ptr(),
        // The workspace its own plan was raised in — the reason `tail_ws`
        // exists: a FlashInfer launcher reads the schedule out of the
        // workspace it is handed, so a tail beside the prefix's workspace would
        // run the prefix's schedule while its own sits unread.
        workspace: ws.view(),
        // Null, so a tail that reaches for either declines: `decode_plan_full`
        // and `prefill_plan` were not replanned for this sub-batch, and the
        // `!attn_plan(..).is_null()` guard turns each into a refusal.
        decode_plan_full: core::ptr::null_mut(),
        prefill_plan: core::ptr::null_mut(),
        kv_page_indptr_d: d_indptr,
        kv_last_page_lens_d: d_lens,
        kv_page_indices_d: d_indices,
        qo_indptr_d: d_qo,
        num_requests: i32::try_from(kv_lens.len() - split).unwrap_or(0),
        num_pages_in_batch: i32::try_from(kv_indices.len().saturating_sub(base)).unwrap_or(0),
        max_pages_per_request: i32::try_from(
            kv_indptr
                .windows(2)
                .map(|w| w[1].saturating_sub(w[0]))
                .max()
                .unwrap_or(0),
        )
        .unwrap_or(0),
        first_token: i32::try_from(split).unwrap_or(0),
        // The driver-pinned outputs advance past the prefix: these guard-owned
        // landings are not windowed (a stated output is, by
        // `resolve_arg_windowed`), so the tail would otherwise write its rows
        // over the prefix's.
        o_out: fire.o_out.wrapping_byte_add(split * o_row_bytes),
        lse_out_d: fire.lse_out_d.wrapping_byte_add(split * lse_row_bytes),
        // Null, so a tail that wants them declines rather than writing through
        // the fire's: the score CSR and mask indptr are indexed by the fire's
        // rows, so a tail addressing them tail-relative reads someone else's
        // rows. `Facts`' accessors return `Option` and test for null.
        score_out: core::ptr::null_mut(),
        score_indptr_d: core::ptr::null(),
        folded_out: core::ptr::null_mut(),
        mask_d: core::ptr::null(),
        mask_indptr_d: core::ptr::null(),
        ..fire.clone()
    }))
}

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
    use crate::bind::{AttnCtx, AttnRegions, DispatchCtx, Frame, Resolver, run};
    use model_compiler::lower::Arg;
    use model_ir::trace::ValueId;

    let t_head = std::time::Instant::now();
    // The mutation first, then the borrows: `ready_device_state` takes
    // `&mut Shell`, so every shared borrow below (`model`, the lowering, the
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
    let (Admitted { class, rows, requests, fire_rows }, row) = admit(state, step)?;
    let model = state.model.as_ref().ok_or(PIE_STATUS_INVALID_ARGUMENT)?;
    // Derived at load, read here. See `LoadedModel::deployment`.
    let dep = &model.deployment;
    let token_ids = step.plan.token_ids.as_slice();
    let position_ids = step.plan.position_ids.as_slice();
    let kv_indices = step.plan.kv_page_indices.as_slice();
    let kv_indptr = step.plan.kv_page_indptr.as_slice();
    let kv_lens = step.plan.kv_last_page_lens.as_slice();
    let qo_indptr = step.plan.qo_indptr.as_slice();

    sg_trace(|| format!("  head {:?}", t_head.elapsed()));
    let t_low = std::time::Instant::now();
    // The lowering, or the one this shape already has: everything to
    // `DispatchPlan` is a pure function of the key, cached because it costs
    // ~3.3 ms on a 0.6B decode.
    let key = LoweringKey {
        model_id: state.load_generation,
        class,
        rows: u32::try_from(rows).unwrap_or(0),
        rows_digest: digest_rows(&fire_rows),
        union_asked: state.boot.supergraph && dep.recurrent.is_none(),
    };
    if !state.lowerings.contains_key(&key) {
        let built = build_lowering(
            row,
            model::catalog::Deployed {
                // CUDA, stated rather than defaulted: the row answers for
                // either backend, so a default would be a silent assumption.
                backend: model::catalog::Backend::Cuda,
                tp_size: model.tp_size,
                layer_scalars: &model.layer_scalars,
            },
            class,
            &fire_rows,
            key.union_asked,
            // The boot's KV scheme, read once here rather than branched on at
            // every append — the same value the layer view is built from.
            crate::bind::Boot { kv_native_bf16: Some(state.kv_format.is_native_bf16()) },
        )?;
        state.lowerings.insert(key, built);
    }
    let LoweredFire { plan, lowered, dplan, union } =
        state.lowerings.get(&key).expect("just built");
    let union = *union;
    sg_trace(|| format!("  lowering {:?}", t_low.elapsed()));
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
        sampled_rows,
        d_ids,
        d_pos,
        d_kv_indices,
        d_kv_indptr,
        d_kv_lens,
        d_qo,
        d_sampled,
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
        &fire_rows,
        rows,
        requests,
    )?;

    lap("kv+arrays");
    // Workspace + plan caches: driver-lifetime, first-launch built.
    let AttnPlans {
        decode_plan,
        decode_plan_full,
        prefill_plan,
        workspace,
        prefill_workspace,
        states_decode_dispatch,
        planless_prefill,
        two_decode_kinds,
    } = raise_attn_plans(
        &mut state.scratch,
        model,
        lowered,
        PlanGeometry {
            kv_indptr,
            kv_lens,
            qo_indptr,
            kv_heads: kv_heads_i,
            head_dim: head_dim_i,
            page_size,
        },
        raw_stream,
    )?;

    let arena_bytes = lowered.arena_bytes.max(64);
    let arena_ptr = state.fire_arrays.arena(&alloc, arena_bytes)?;
    let exec_frame = Frame { arena: arena_ptr, arena_bytes };

    let mut named_widths: std::collections::BTreeMap<ValueId, u32> =
        std::collections::BTreeMap::new();
    for a in &lowered.args {
        if let Arg::Named { value, width, .. } = a {
            // Max, not last: several values can share one id (they alias in
            // place), so the buffer must fit the widest read.
            let slot = named_widths.entry(*value).or_insert(*width);
            *slot = (*slot).max(*width);
        }
    }
    for i in 0..lowered.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width, .. } = a {
                let slot = named_widths.entry(*value).or_insert(*width);
                *slot = (*slot).max(*width);
            }
        }
    }
    let SeamPins { d_scores, d_folded, d_score_indptr, d_mask, d_mask_indptr, d_attn_out } =
        publish_seam_pins(
            &mut state.fire_arrays,
            alloc,
            stream,
            dep,
            model,
            step,
            &named_widths,
            PlanGeometry {
                kv_indptr,
                kv_lens,
                qo_indptr,
                kv_heads: kv_heads_i,
                head_dim: head_dim_i,
                page_size,
            },
            rows,
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

    // Pooled, because a capture bakes `lse_out_d`.
    let lse = state.fire_arrays.lse(alloc, rows * model.deployment.shape.q_heads as usize * 4)?;

    // The guard-owned attention values, discovered from the lowering.
    let (q_pin, o_off) = attention_pins(dep, lowered, dplan, states_decode_dispatch)?;

    struct LiveResolver<'a> {
        model: &'a LoadedModel,
        named: &'a std::collections::BTreeMap<ValueId, crate::device::DeviceBuffer>,
    }
    impl Resolver for LiveResolver<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            self.model.weight(name)
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // The family's attention scalars. `sm_scale` varies by layer kind, not
    // layer, so the first is the stack's (gemma-4 runs 1.0 — its q/k norms
    // carry the scaling — with per-layer windows and host CSR mirrors for its
    // planless prefill).
    let sm_scale = dep.attention.first().map_or(1.0, |a| a.sm_scale);
    let window_by_layer = dep.windows();
    // The attention state a rectangle executes against — a struct literal because every
    // field is already computed above.
    let attn = AttnCtx {
        decode_plan,
        decode_plan_full,
        prefill_plan,
        workspace,
        prefill_workspace,
        layers,
        // A temporary borrow, not the long-lived `named_bufs`: the adapter
        // phase below grows the scratch, which a borrow held across forbids.
        q_out: q_pin
            .and_then(|v| state.fire_arrays.named.get(&v).map(|b| b.as_ptr()))
            .unwrap_or(core::ptr::null_mut()),
        score_out: d_scores.cast(),
        folded_out: d_folded.cast(),
        score_indptr_d: d_score_indptr.cast(),
        mask_d: d_mask,
        mask_indptr_d: d_mask_indptr,
        o_out: match o_off {
            Some(off) => unsafe { arena_ptr.cast::<u8>().add(off) }.cast(),
            // No stated slot: the driver's own landing buffer, pooled so a
            // capture that baked its address keeps addressing something.
            None => d_attn_out,
        },
        kv_page_indices_d: d_kv_indices.cast(),
        kv_page_indptr_d: d_kv_indptr.cast(),
        kv_last_page_lens_d: d_kv_lens.cast(),
        qo_indptr_d: d_qo.cast(),
        qo_indptr_h: if planless_prefill { qo_indptr.as_ptr() } else { core::ptr::null() },
        kv_page_indptr_h: if planless_prefill { kv_indptr.as_ptr() } else { core::ptr::null() },
        num_requests: requests as i32,
        num_pages_in_batch: kv_indices.len() as i32,
        max_pages_per_request: i32::try_from(
            kv_indptr.windows(2).map(|w| w[1].saturating_sub(w[0])).max().unwrap_or(0),
        )
        .unwrap_or(0),
        first_token: 0,
        w_page_d: d_w_page.cast(),
        w_off_d: d_w_off.cast(),
        row_valid_d: d_valid.cast(),
        lse_out_d: lse.cast(),
        window_left: -1,
        window_left_by_layer: window_by_layer,
        // The attention cap, off the statement: gemma-2 states one
        // (`attn_logit_softcapping: 50.0`), and a literal `0.0` here would
        // attend uncapped while its facts claim capped.
        logits_soft_cap: model.deployment.attn_logit_softcap,
        sm_scale,
        score_window: state.boot.attn_score_window,
    };

    // The phase takes its fields, not the shell: `model` is a shared borrow of
    // `state` that outlives this call, so a `&mut Shell` phase would not compile.
    let lora_state = lora_phase(
        &state.ptir_programs,
        &state.ptir_sessions,
        state.ptir_rings.as_ref(),
        &state.instances,
        &mut state.fire_arrays,
        &mut state.lora_arena,
        state.tp_size,
        frame,
        &qo_indptr,
        stream.as_ref(),
        raw_stream,
        alloc,
        model,
        dep,
        lowered,
        dplan,
        rows,
    );

    // One handle for the driver, its stream rebound per fire: creating and
    // destroying one per fire cost 3.2 ms.
    let mut cublas_ops = crate::device::cublas::LiveCublas;
    if state.cublas.is_none() {
        state.cublas =
            Some(crate::device::cublas::CublasHandle::create(&mut cublas_ops, raw_stream)?);
    }
    let cublas = state.cublas.as_mut().expect("just ensured");
    cublas.set_stream(&mut cublas_ops, raw_stream)?;
    // The family's per-layer tables, off the value: a stack whose rope is one
    // theta answers with an empty table, because the binder checks emptiness
    // and would walk a table of identical values for nothing.
    let theta_by_layer = dep.theta_by_layer();
    let rotary_by_layer = dep.rotary_by_layer();
    let softcap = dep.logit_softcap;
    // `u32` on the deployment, `i32` on the ctx, narrowed here. Saturating, not
    // wrapping: a PLE width past `i32::MAX` is a corrupt config, and a grid
    // divided by a negative extent launches nothing rather than refusing.
    let ple_dim = i32::try_from(dep.ple_dim).unwrap_or(i32::MAX);
    let scales = dep.scales.clone();
    // The peel window word, re-derived from the fire's rows rather than read
    // off the tail rectangle: under `Union` both regions get the whole window
    // as their rectangle (the launches are full-window grids), so the rectangle
    // no longer carries the split and this word is the only thing that does.
    // Derived from the same place `lower::split_at` uses — the rows and the
    // axis's predicate — so the two cannot drift.
    if state.peel_win.is_none() {
        state.peel_win = Some(crate::device::PeelWindowWord::new(alloc)?);
    }
    let peel_win = state.peel_win.as_mut().expect("just ensured");
    let peel_axis = lowered.launches.iter().find_map(|l| l.peel.map(|p| p.axis));
    let (peel_start, peel_count) = peel_word(&fire_rows, peel_axis, rows);
    peel_win.set(peel_start, peel_count);
    peel_win.upload(stream.as_ref())?;
    let peel_window_ptr = peel_win.device_ptr();

    // The dispatch context: what any rectangle reads (the stream, the cublas
    // handle, the per-layer tables, the fire's own words) — the other half of
    // the pair with `AttnCtx`.
    let ctx = DispatchCtx {
        sampling_indices: d_sampled.cast::<i32>(),
        sampled_rows: i32::try_from(sampled_rows.len()).unwrap_or(0),
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: model.deployment.norm_eps,
        // The first layer's base, which is what a single `rope_theta` meant;
        // the per-layer values are read from `rope_theta_by_layer`. Zero for a
        // deployment with no attention layers.
        rope_theta: model.deployment.attention.first().map_or(0.0, |a| a.rope_theta),
        rope_theta_by_layer: theta_by_layer,
        rotary_by_layer,
        head_dim: i32::try_from(model.deployment.shape.head_dim).unwrap_or(0),
        num_q_heads: i32::try_from(model.deployment.shape.q_heads).unwrap_or(0),
        num_kv_heads: i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
        vocab: i32::try_from(model.deployment.shape.vocab).unwrap_or(0),
        gate_second: false,
        rope_interleaved: false,
        token_ids: d_ids.cast_mut().cast(),
        positions: d_pos.cast_mut().cast(),
        final_logit_softcap: softcap,
        ple_dim,
        scales,
        // The row's convention, not a constant: `moe::topk_sigmoid_bias` takes
        // this as `Source::Ctx`, and a `false` here would route every mixture
        // on weights summing to less than one — wrong for GLM-4.5, which
        // publishes `norm_topk_prob: true`.
        moe_norm_topk: model.deployment.norm_topk_prob,
        // As above: the row's, not a constant. DeepSeek-V3 and GLM-4.5 publish
        // 2.5 and Kimi-K2 publishes 2.0, so a `1.0` here would deliver a
        // fraction of a routed token's trained contribution.
        moe_routed_scaling: model.deployment.routed_scaling,
        // YaRN's four, off the statement: `rope_yarn_original` takes all of
        // them, so a zeroed array is not a neutral default for the rows that
        // reach it — a factor of zero is a degenerate ramp, not an absent one.
        yarn: match model.deployment.rope_scaling {
            Some(model::deployment::RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                attention_factor,
                ..
            }) => [factor, beta_fast, beta_slow, attention_factor],
            _ => [0.0; 4],
        },
        yarn_original_max: match model.deployment.rope_scaling {
            Some(model::deployment::RopeScaling::Yarn { original_max_position, .. }) => {
                i32::try_from(original_max_position).unwrap_or(0)
            }
            _ => 0,
        },
        // gpt-oss's clamped GLU, off the statement not left at zero: `alpha`
        // scales the gate inside the sigmoid, so `0.0` collapses `silu(a*x)` to
        // `x/2`, and `limit` clamps both halves to nothing at `0.0`.
        glu_limit: match model.deployment.mlp_gate {
            model::deployment::MlpGate::SiluClamped { limit, .. } => limit,
            _ => 0.0,
        },
        glu_alpha: match model.deployment.mlp_gate {
            model::deployment::MlpGate::SiluClamped { alpha, .. } => alpha,
            _ => 0.0,
        },
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        // The fire's own fanout, read off the deployment and derived only if it
        // is silent: `experts_per_token` is a stated field, and
        // `fire_experts_per_token` stays below it as the fallback.
        experts_per_token: {
            let stated = i32::try_from(model.deployment.shape.experts_per_token).unwrap_or(0);
            if stated > 0 { stated } else { fire_experts_per_token(lowered) }
        },
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        // The staged adapter, or `None` for a fire that carries none. `None` is
        // not a refusal: the executor's arm returns `Ok(())`, load-bearing for
        // union captures — every arm lowers and the predicate decides at
        // replay, so the arm must be issuable with nothing to correct.
        lora: lora_state.as_ref().map(|(s, scratch)| (std::ptr::from_ref(s), *scratch)),
        // The fire's peel window, published so a `_devwin` tail statement can
        // early-out per lane: the prefix is the rows that do not carry the
        // axis's mark, so with no marked rows the word says the whole fire.
        peel_window: peel_window_ptr,
        rows_total: i32::try_from(rows).unwrap_or(0),
        moe_ptrs: std::cell::Cell::new(None),
    };

    lap("bind");
    // The tail's own state, when the lowering peeled: a tail serves `[split, N)`,
    // a different request count and so a different FlashInfer schedule. Built
    // after the adapter phase because it takes the scratch mutably. An empty
    // tail is not a tail: under `Union` `peel_axis` is `Some` even on an
    // unpeeled fire, with the split at `rows`, so the guard below requires rows
    // in the tail.
    let tail_ctx = if peel_start > 0 && peel_count > 0 && (peel_start as usize) < rows {
        let fs = state.scratch.as_mut().ok_or(PIE_STATUS_DRIVER_ERROR)?;
        let (tail_plan, tail_ws) = (&mut fs.tail_plan, &mut fs.tail_ws);
        peel_tail_ctx(
            &attn,
            &mut state.fire_arrays,
            tail_plan,
            tail_ws,
            alloc,
            stream.as_ref(),
            raw_stream,
            &kv_indptr,
            &kv_lens,
            &kv_indices,
            &qo_indptr,
            peel_start as usize,
            rows,
            two_decode_kinds,
            // The same two extents `publish_seam_pins` and the LSE allocation
            // are sized from, so a stride cannot drift from its buffer.
            model.deployment.shape.q_heads as usize * model.deployment.shape.head_dim as usize * 2,
            model.deployment.shape.q_heads as usize * 4,
            i32::try_from(model.deployment.shape.q_heads).unwrap_or(0),
            i32::try_from(model.deployment.shape.kv_heads).unwrap_or(0),
            i32::try_from(model.deployment.shape.head_dim_alloc()).unwrap_or(0),
            page_size,
        )?
    } else {
        None
    };
    let named_bufs = &state.fire_arrays.named;
    let mut resolver = LiveResolver { model, named: named_bufs };
    // Which prepared state each rectangle gets.
    let regions = match tail_ctx.as_ref() {
        Some(tail) => AttnRegions::split(&attn, tail),
        None => AttnRegions::whole(Some(&attn)),
    };
    // The last use of `alloc` is above, so the shared borrow is dead and the
    // capture can take the same allocator mutably — a capture must be opened on
    // the allocator that owns what the fire frees, or the frees are not deferred.
    if state.preds.is_none() {
        state.preds = crate::device::PredicateWord::new(
            state.fire_alloc.as_ref().expect("the fire allocator exists"),
        )
        .ok();
    }
    let (capture_alloc, capture_preds) = match (&mut state.fire_alloc, &mut state.preds) {
        (Some(a), Some(p)) => (a, p),
        _ => return Err(PIE_STATUS_EXHAUSTED),
    };
    lap("ctx");
    // The walk: capture, replay, or straight onto the stream.
    let result = if union {
        capture_or_replay(
            &mut state.supergraph,
            state.fire_arrays.epoch,
            state.load_generation,
            &plan,
            &fire_rows,
            &lowered,
            &dplan,
            exec_frame,
            &mut resolver,
            &ctx,
            regions,
            gdn_ctx.as_ref(),
            capture_alloc,
            capture_preds,
            stream.as_ref(),
            requests,
            rows,
            class,
        )
    } else {
        run(&lowered, &dplan, exec_frame, &mut resolver, &ctx, regions, gdn_ctx.as_ref())
    };
    lap("run");
    // A step that owes nothing synchronizes, because the next step reads what
    // this one wrote. A step that owes the completion does not: its debt rides
    // a stream-ordered callback and this call returns with the work queued.
    let sync =
        if owes.is_some() && state.runahead { Ok(()) } else { stream.as_ref().synchronize() };
    lap("sync");
    match (result, sync) {
        (Ok(_), Ok(())) => {}
        (Err(e), _) => {
            eprintln!("[driver-cuda] launch: refused: {e:?}");
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
    // A fresh borrow of the allocator: the capture above takes
    // `&mut state.fire_alloc`, so reusing the earlier binding here would extend
    // a shared borrow across it.
    let alloc = state.fire_alloc.as_ref().expect("the fire allocator exists");
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
            named_bufs: &state.fire_arrays.named,
        },
        model,
        lowered,
        dplan,
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
            lowered,
            dplan,
            named_bufs,
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
        // The scratch survives the call: dropping it here would `cudaFree`
        // while the fire runs, synchronizing the device. The next launch
        // reclaims it.
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
            // `lse` and `d_valid` are pooled, so they are not here: handing a
            // pooled buffer to `InFlight` would free the pool.
            scratch: [_slot_ids_buf].into_iter().flatten().collect(),
            closed_channels: Vec::new(),
        });
    }
    lap("tail");
    Ok(())
}

#[cfg(test)]
mod peel_tests {
    use super::{TailCsrs, tail_csrs};

    use super::peel_word;
    use model_compiler::lower::Row;
    use model_ir::trace::PeelWindow;

    fn rows(hooked: &[bool]) -> Vec<Row> {
        hooked.iter().map(|&h| Row { hooked: h, ..Row::default() }).collect()
    }

    /// The three cases the two kernel forms must agree on: the prefix form runs
    /// `[0, start)`, the tail form `[start, start + count)`.
    #[test]
    fn the_peel_word_says_which_rows_each_form_runs() {
        // No peel: prefix runs all four, tail none.
        assert_eq!(peel_word(&rows(&[false; 4]), None, 4), (4, 0));

        // A peel lowered, no row marked — what `Union` does to an unpeeled
        // fire, since both regions lower unconditionally. The whole fire is the
        // prefix.
        assert_eq!(
            peel_word(&rows(&[false; 4]), Some(PeelWindow::HookFreePrefix), 4),
            (4, 0),
            "an unpeeled fire under Union must still run its prefix over \
             every row — `(0, 4)` would make the prefix form compute \
             NOTHING and the tail form run over all four"
        );

        // A contiguous marked suffix: prefix [0,2), tail [2,4).
        assert_eq!(
            peel_word(&rows(&[false, false, true, true]), Some(PeelWindow::HookFreePrefix), 4),
            (2, 2)
        );

        // Every row marked: no prefix, the tail is the fire.
        assert_eq!(peel_word(&rows(&[true; 4]), Some(PeelWindow::HookFreePrefix), 4), (0, 4));
    }

    /// The axis picks the predicate, and the two axes are different marks.
    #[test]
    fn the_axis_decides_which_mark_splits() {
        let r = rows(&[false, false, true, true]);
        assert_eq!(peel_word(&r, Some(PeelWindow::HookFreePrefix), 4), (2, 2));
        // The same rows carry no `custom_mask`, so the mask axis sees no split:
        // a fire hooked but unmasked is one region on the mask axis and two on
        // the hook axis.
        assert_eq!(peel_word(&r, Some(PeelWindow::UnmaskedPrefix), 4), (4, 0));
    }

    /// The tail's prefix sums start at zero, and its pages start where
    /// the split's entry points.
    #[test]
    fn the_tail_csrs_are_rebased_not_sliced() {
        // Four requests holding 2, 1, 3, 1 pages; tokens 1 each (decode).
        let kv_indptr = [0u32, 2, 3, 6, 7];
        let qo_indptr = [0u32, 1, 2, 3, 4];

        let got = tail_csrs(&kv_indptr, &qo_indptr, 2);
        assert_eq!(
            got,
            TailCsrs { base: 3, indptr: vec![0, 3, 4], qo: vec![0, 1, 2] },
            "requests 2..4 hold 3 then 1 pages, and their pages begin at \
             index 3 — the value `kv_indptr[2]` holds"
        );

        // A slice would have been `[3, 6, 7]` and `[2, 3, 4]`, which FlashInfer
        // reads as a batch whose first request already has three pages behind
        // it — a plan for requests that are not these.
        assert_ne!(got.indptr, kv_indptr[2..].to_vec());
    }

    /// A split at the last request leaves one, not zero.
    #[test]
    fn the_last_request_is_a_tail_of_one() {
        let kv_indptr = [0u32, 2, 3];
        let qo_indptr = [0u32, 1, 2];
        let got = tail_csrs(&kv_indptr, &qo_indptr, 1);
        assert_eq!(got.indptr, vec![0, 1], "one request holding one page");
        assert_eq!(got.base, 2);
    }

    /// A split past the array answers empty rather than panicking —
    /// `peel_tail_ctx` refuses that case first, and this pins that the
    /// arithmetic does not depend on the refusal.
    #[test]
    fn a_split_past_the_end_is_empty_not_a_panic() {
        let got = tail_csrs(&[0, 2], &[0, 1], 9);
        assert_eq!(got, TailCsrs { base: 0, indptr: Vec::new(), qo: Vec::new() });
    }
}
