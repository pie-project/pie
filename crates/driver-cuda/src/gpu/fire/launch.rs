//! The forward path: one step, from a frame descriptor to logits.
//!
//! The largest of these modules and the one that earns it — `step_impl` is
//! the whole of a decode step, and the phases around it (admit, lower,
//! capture-or-replay, the GDN context, the KV pools, delivery) are its
//! parts. `.wiki/driver/graph.md` is about this file.

use driver_api::local::{
    PIE_STATUS_DRIVER_ERROR,
    PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_UNSUPPORTED,
    PieCompletion,
    PieFrameDesc,
};
use crate::gpu::serve::load::ptir_target;
use crate::gpu::serve::state::{
    retire_fire,
    ChannelState,
    FireDebt,
    FireScratch,
    GdnState,
    InFlight,
    InstanceEntry,
    KvState,
    LoadedModel,
    LoweredFire,
    LoweringKey,
    RUNAHEAD_DEPTH,
    Shell,
    digest_rows,
    instance_ring_shapes,
    retire,
    slice_of,
};

/// The loaded model's facts, family-dispatched: the qwen3_5 hybrid by
/// its `linear_*` geometry + layer schedule, else the llama-like
/// mapping. Only the qwen3-family pre-norm shape is claimed on the
/// llama-like side; anything else refuses rather than mis-executes.
/// The `Scratch::named` key the SCORE pin is pooled under.
///
/// A reserved id rather than a traced one: no statement names this value,
/// the driver publishes it, and the pool is keyed by `ValueId` because
/// every other thing in it is a traced seam. `u32::MAX` cannot collide
/// with a trace value — a plan with four billion values would have failed
/// long before.
const SCORE_PIN: model_compiler::trace::ValueId = model_compiler::trace::ValueId::MAX;

/// Does a fire's completion ride a stream callback? **YES unless told not to**
/// (`PIE_CUDA_RUNAHEAD=0`, or `[driver] runahead = false` per driver).
///
/// It was off, and the reason was honest: `pie_cuda_launch` used to finish the
/// fire before it returned, and a caller reading the ring on the next line was
/// asserting that. The gate protected a CONTRACT CHANGE.
///
/// It is on because the contract is now the ABI's: the notify says the fire
/// retired, the engine waits for it, and this tree's tests do too. And because
/// there is finally something to gain — a warm decode issues in 0.68 ms and
/// retires 3.75 ms later, so the call returns with 3 ms of GPU work queued
/// behind it. When the gate was written those two numbers were the same, and
/// run-ahead bought nothing.
pub(crate) fn runahead_env() -> bool {
    !std::env::var_os("PIE_CUDA_RUNAHEAD").is_some_and(|v| v == "0" || v == "false" || v == "off")
}

/// The arena offset the attention dispatch at `fi` WRITES.
///
/// Two readings, and only the first is right under a union lowering.
///
/// 1. **The dispatch's own op join.** The attention statement carries its
///    output placement, which is exactly the slot the o_proj goes on to read.
/// 2. **The next launch's first operand** — "the launch after the dispatch is
///    the o_proj". True under `Resolve`, where the guard has already deleted
///    every arm the fire did not take. False under `Union`, where every arm is
///    present and the next launch belongs to some other body, which is why a
///    union that has to fall back here declines instead.
///
/// # And why every DECODE declines
///
/// `dsl::seam::attn_at` records an output only when the statement is NOT
/// inside a value-producing region:
///
/// ```ignore
/// let out = q.t.inner.borrow().inside_value_region();
/// let shape = (!out).then(|| …);   // None inside a region
/// ```
///
/// The decode arm states its attention inside one and the prefill arm does
/// not. So reading 1 is empty for every decode, the union is declined, and a
/// one-token fire walks all 396 launches — ~9 ms on a 0.6B model, which is
/// most of what `.wiki/new-driver/next.md`'s table measures.
///
/// **Recovering it from the enclosing region here does not work, and the
/// failure is quiet.** Two attempts, both green on 20 of 21 ABI tests and
/// both wrong on `multi_step_resize_and_copy_preserve_the_kv`: the nearest
/// preceding region is as often a closed guard from an earlier layer as the
/// enclosing one; requiring coverage and matching the output's shape to q's
/// still picks the wrong construct for some fires, because several covering
/// regions can carry a q-shaped value. A wrong offset here is not a refusal —
/// it binds the attention plan over another activation.
///
/// The fix belongs at the STATEMENT, not here: `attn_at` should record its
/// landing in the join even inside a region, so the value has one stated
/// producer instead of being reverse-engineered from op adjacency.
fn attention_landing(
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::gpu::bind::DispatchPlan,
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
/// The one instrument that answers the question the measurements raise —
/// whether a fire REPLAYED or walked its launches, and if it walked, which
/// clause of the servability test turned it away. Lazy, so an unset variable
/// costs a `getenv` and no formatting.
pub(crate) fn sg_trace(what: impl FnOnce() -> String) {
    if std::env::var_os("PIE_CUDA_TRACE_SUPERGRAPH").is_some() {
        eprintln!("[sg] {}", what());
    }
}

/// Is the unionized supergraph armed for this process?
///
/// **ON by default now**, and `PIE_CUDA_SUPERGRAPH=0` turns it off.
///
/// It was off, deliberately, with this reason: every A/B in the tree pins
/// the EAGER leg, and a capture is an optimisation that has to prove
/// itself against that rather than replace it silently. It has now proved
/// it, on the three claims that were the actual doubt:
///
/// - the whole ABI suite records and replays (19/19 with the gate on),
///   which is every family this shell opens and every fire shape it
///   serves;
/// - one exec runs two structurally distinct KV-write programs and
///   returns byte-identical logits, selected by a byte of device memory
///   (`bridge_smoke::the_union_captures_and_replays_the_same_decode`);
/// - and one exec serves a SECOND fire's tokens
///   (`a_cached_exec_serves_the_next_fire`), which is the property that
///   makes a cached exec worth caching and the only one that can tell a
///   baked address from baked contents.
///
/// What cannot be replayed still refuses rather than being captured
/// wrong: recurrent-state families stay eager at the LOWERING decision,
/// and an arm whose prepared state the fire declines to build is refused.
/// So default-on changes which leg runs, not which answers are possible.
///
/// The env var inverts rather than disappears, because a default is a
/// judgement and a judgement should stay reversible without a rebuild.
fn supergraph_enabled() -> bool {
    !std::env::var_os("PIE_CUDA_SUPERGRAPH")
        .is_some_and(|v| v == "0" || v == "false" || v == "off")
}

/// The fire's CLASS, read off its SHAPE — one row per request is a
/// decode, anything else is prefill-shaped.
///
/// It used to read the recurrent-state flags too, and derive three MTP
/// service classes from them (`CommitAdvance` where every row replayed
/// buffered tokens, `FrozenVerify` where any row wrote buffered slabs,
/// `StateOnly` where a recurrent fire had no readout rows). Those classes
/// are gone — `.wiki/driver/graph.md` §4.2: a speculative decode buffers
/// its tokens and folds only the accepted prefix, so a rejected token is
/// never folded and there is nothing to repair. The driver executes the
/// flags now (see the fold in `gdn_context`) instead of classifying on
/// them.
///
/// What remains is a shape question, and it is on its way out too: the
/// window class is `GuardPred::WindowOne` now, so the two surviving
/// values pick nothing the trace does not already guard. See §4.1.
pub fn fire_class_of(
    _step: &driver_api::local::PieStepDesc,
    rows: usize,
    requests: usize,
) -> Result<model_compiler::trace::FireClass, i32> {
    use model_compiler::trace::FireClass;
    Ok(if rows == requests { FireClass::Decode } else { FireClass::Prefill })
}

/// Replay this fire's bucket if it is captured, and capture it if not.
///
/// The whole supergraph arc, at its one live call site. What it does, in
/// the order the pieces were built:
///
/// 1. **Eligibility.** A fire whose staged LoRA did not group cannot be
///    recorded at all — `apply`'s solo path is a host loop whose launch
///    count follows the adapter set. Ineligible means eager, which is the
///    C++ arc's own device for what cannot be replayed.
/// 2. **The bucket.** `(R, N, fire class, model)` plus the lora group
///    shape. Every `GuardPred` axis is deliberately absent: those are what
///    the conditionals fold.
/// 3. **The epoch.** `Scratch` bumps it whenever a pool grew, because
///    growth moves a base address out from under a recorded launch. A
///    stale exec is dropped and recaptured rather than replayed.
/// 4. **Dual-prepare.** A capture must be taken warm — a launcher that
///    allocates on first use cannot do so inside a capture — and a warm-up
///    must walk a VALID program, so warm once per variant with its own
///    resolved lowering. A union records arms no single valid program
///    takes, which is why one warm fire is not enough.
/// 5. **The predicates**, uploaded before every launch: this is the fire's
///    own shape, and the only thing that differs between two replays of
///    one exec.
#[allow(clippy::too_many_arguments)]
fn capture_or_replay<R: crate::gpu::bind::Resolver>(
    cache: &mut crate::gpu::fire::recordings::Recordings,
    epoch: u64,
    model_id: u64,
    plan: &model_compiler::trace::ForwardPlan,
    rows_desc: &[model_compiler::lower::Row],
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::gpu::bind::DispatchPlan,
    frame: crate::gpu::bind::Frame,
    resolver: &mut R,
    ctx: &crate::gpu::bind::DispatchCtx,
    regions: crate::gpu::bind::AttnRegions<'_>,
    gdn: Option<&crate::gpu::bind::GdnCtx>,
    alloc: &mut crate::gpu::device::Allocator,
    preds: &mut crate::gpu::device::PredicateWord,
    stream: crate::gpu::device::StreamRef<'_>,
    requests: usize,
    rows: usize,
    class: model_compiler::trace::FireClass,
) -> Result<usize, crate::gpu::bind::RunRefusal> {
    use crate::gpu::bind::{DispatchPlan, run};
    use crate::gpu::fire::recordings::{BucketKey, fire_predicates, union_eligibility};

    let eligibility = union_eligibility(None);
    let key = BucketKey::new(
        u32::try_from(requests).unwrap_or(0),
        u32::try_from(rows).unwrap_or(0),
        class,
        model_id,
    );

    // The fire's own bits, and the only thing that differs between two
    // replays of one exec. NOT synchronized after: the upload and the replay
    // are ordered on the same stream, so waiting here only made the call
    // block on work it had just enqueued.
    if fire_predicates(rows_desc, &lowered.conds, preds).is_err()
        || preds.upload(stream).is_err()
    {
        return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
    }

    if cache.replay(key, epoch, stream).unwrap_or(false) {
        sg_trace(|| format!("replay {key:?}"));
        return Ok(lowered.launches.len());
    }
    sg_trace(|| format!("miss {key:?} launches={}", lowered.launches.len()));

    // DUAL-PREPARE: one warm fire per variant, each a resolved program.
    // Only variants this fire can PREPARE. A `wants_scores` warm-up would
    // lower the score-capturing dispatch, which refuses without a score
    // sink — and warming is not the place to discover that. It is also
    // why scores are not a union axis: the north star's list is "hook
    // attachment, mask kind, correction arm, depth, LoRA rank", and every
    // one of those is a branch rather than a different prepared state.
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
        let warm_dplan = DispatchPlan::new(plan, &warm);
        run(&warm, &warm_dplan, frame, resolver, ctx, regions, gdn)?;
        let _ = stream.synchronize();
    }

    let captured = {
        // THE CAPTURE OPENS ON THE FIRE'S OWN ALLOCATOR, and it used to open
        // on a throwaway one made right here. That looked harmless — nothing
        // allocates during a capture — but the flag it raises is what makes a
        // `cudaFree` DEFER, and the deferral has to happen on the allocator
        // that OWNS the buffer. A temporary dropped inside `run_captured`
        // therefore freed immediately, in the middle of an open stream
        // capture, and the graph that came out faulted when it was destroyed.
        //
        // Phi-3's decode found it, because no decode had ever been captured.
        let Ok(scope) = alloc.begin_capture(stream) else {
            return run(lowered, dplan, frame, resolver, ctx, regions, gdn);
        };
        let mut b = crate::gpu::device::SupergraphBuilder::new(scope.stream(), preds);
        let ran = crate::gpu::bind::run_captured(
            lowered, dplan, frame, resolver, ctx, regions, gdn, &mut b,
        );
        // The nodes the capture retained, taken BEFORE the builder is
        // dropped: one per launch, and what lets a later fire of a
        // different row count retune this exec's rectangles instead of
        // recapturing (`.wiki/driver/graph.md` §6.2).
        let nodes = b.nodes().to_vec();
        drop(b);
        // A REFUSED CAPTURE IS NOT A REFUSED FIRE.
        //
        // Some arms cannot be recorded at all, and the reason is always
        // the same shape: their prepared state is something the fire
        // declined to build. The score-capturing prefill dispatch wants a
        // plan raised for the full-attention variant, buffers laid out for
        // an observation window, and a positive window — none of which a
        // fire that wants no scores has any reason to prepare.
        //
        // So the capture is abandoned and the fire runs eagerly. That is
        // the same answer ungrouped LoRA gets from `union_eligibility`,
        // and the same one the C++ arc gives mixed peels: what cannot be
        // replayed stays eager. The alternative — failing the fire — would
        // make an optimisation into a correctness requirement.
        let ended = scope.end();
        sg_trace(|| format!("capture ran={ran:?} ended_ok={}", ended.is_ok()));
        match (ran, ended) {
            (Ok(n), Ok(g)) => Some((n, g, nodes)),
            (Err(_), Ok(g)) => {
                // AN ABANDONED CAPTURE IS NOT DESTROYED, it is forgotten.
                //
                // A run that refused part-way leaves a recording whose nodes
                // the builder has already dropped, and `cudaGraphDestroy` on
                // that faults inside the CUDA driver — no device error, a
                // host segfault. Found through phi-3, where an unservable
                // arm made every decode capture abandon.
                //
                // Leaking one graph template per abandoned capture is the
                // cheaper wrong answer: captures are per (bucket, epoch) and
                // a refusal is meant to be rare. The refusals that are NOT
                // rare belong in the servability test above, which is where
                // phi-3's went. `ManuallyDrop` rather than `mem::forget`
                // because the leak is the POINT and should read as one.
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
    let _ = cache.insert_with_nodes(key, exec, epoch, nodes, eligibility);
    Ok(ran)
}

/// The fire itself. Everything here is the proven smoke assembly, run
/// against the shell's own state.
#[allow(clippy::too_many_lines)]
pub(crate) fn launch_impl(
    state: &mut Shell,
    frame: &PieFrameDesc,
    completion: PieCompletion,
) -> Result<(), i32> {
    let steps = slice_of(frame.steps.ptr, frame.steps.len);
    if steps.is_empty() {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    }
    // Steps run SEQUENTIALLY, each a fire of its own — the frame's
    // producer→consumer ordering. One shared KV, per-step everything else.
    for step in &steps[..steps.len() - 1] {
        step_impl(state, frame, step, None)?;
    }
    // The LAST step carries the frame's debt: its terminal cells and the
    // completion the runtime waits on. Only it enqueues an asynchronous
    // retire, because a frame completes once.
    let step = steps.last().expect("nonempty");
    let cells = slice_of(step.terminal_cells.ptr, step.terminal_cells.len).to_vec();
    step_impl(state, frame, step, Some((completion, cells)))
}

/// Trace a family's forward for one fire shape, lower it, and join the ops
/// back onto the launches.
///
/// Split out of `step_impl` so its result can be CACHED — see
/// [`Shell::lowerings`]. Nothing here reads the fire's data; it reads the
/// shape, which is what makes the answer reusable.
fn build_lowering(
    family: &dyn model::deployment_cuda::PlannedFamily,
    class: model_compiler::trace::FireClass,
    fire_rows: &[model_compiler::lower::Row],
    union_asked: bool,
) -> Result<LoweredFire, i32> {
    use crate::gpu::bind::DispatchPlan;
    use model_compiler::lower::{Fire, GuardMode, lower_with};

    let plan = family.trace(class);
    let lower_as = |g: GuardMode| {
        lower_with(&plan, fire_rows, Fire { captures_across_splits: false }, g).map_err(|e| {
            eprintln!("[driver-cuda] launch: uncovered: {e:?}");
            PIE_STATUS_UNSUPPORTED
        })
    };
    let mut union = union_asked;
    if !union {
        sg_trace(|| "union off at the gate".into());
    }
    let mut lowered = lower_as(if union { GuardMode::Union } else { GuardMode::Resolve })?;

    // NOTHING DECLINES THE UNION ANY MORE, and the two clauses that used
    // to are worth naming because their removal is the point of §5 (1)
    // and (2) in `.wiki/driver/graph.md`.
    //
    // The first refused any lowering mentioning `_capture` or `_custom` --
    // the exact case a folded `WantsAttnScore` / `HasCustomMask` predicate
    // exists for. Its cause was PER-ARM PREPARED STATE: under `Union`
    // every arm is recorded whether this fire takes it or not, so the
    // capture walked the arm whose state the fire declined to build and
    // abandoned the whole recording. Prepared state belongs to the BUCKET
    // now -- the score sink is published every fire, every plan the
    // geometry permits is raised, a causal element mask stays resident.
    //
    // The second refused a fire whose attention output slot the op JOIN
    // could not name. The old fallback -- "the launch after the dispatch
    // is the o_proj, so its input is the slot" -- is a fact read off where
    // a statement SITS, true under `Resolve` (the guard has deleted every
    // arm the fire did not take) and false under `Union` (every arm is
    // present and the neighbour belongs to some other body). But declining
    // was never the only answer: `AttnCtx::o_out` is a DRIVER-owned
    // pointer, so a fire whose join names no slot gets a driver-owned
    // buffer instead of losing its graph. See `o_off` at the fire site.

    let dplan = DispatchPlan::new(&plan, &lowered);
    sg_trace(|| format!("built: launches={} union={union}", lowered.launches.len()));
    Ok(LoweredFire { plan, lowered, dplan, union })
}

/// One step's fire — the former single-step body.
#[allow(clippy::too_many_lines)]
/// What a step must satisfy before anything is traced, lowered or
/// allocated for it — and the handful of facts that survive the asking.
///
/// The FIRST of `step_impl`'s phases, promoted out of it. Its boundary is
/// the one place in the fire path where "this driver cannot serve this"
/// is still a cheap answer: nothing has been built, nothing has been
/// bound, and no device memory has moved. Every refusal below is
/// therefore an early return and not a rollback, which is the north
/// star's third rule read forwards — decide, then move.
///
/// The borrow shape is why this returns indices into `step` rather than
/// the slices themselves: `state` is `&mut` for the rest of the fire, so
/// a phase that handed back `&[u32]` borrowed from `state.model` would
/// pin it. The caller re-slices from `step`, which it owns.
struct Admitted {
    /// The service class the row/request ratio implies.
    pub(crate) class: model_compiler::trace::FireClass,
    /// Token rows in this step.
    pub(crate) rows: usize,
    /// Requests the step's CSR partitions those rows into.
    pub(crate) requests: usize,
    /// The rows the lowering will resolve its guards against, read from
    /// the step's REGION TABLE.
    ///
    /// This shell used to build `vec![Row { samples: true, ..default() };
    /// rows]` and never look at `region_sig` — zero reads in the whole
    /// file. So `HasLora`, `HasCustomMask`, `HasStageHooks` and the depth
    /// truncation could not hold no matter what the engine sent, and
    /// every fire claimed to sample every row. LoRA looked like a missing
    /// feature; the wire had been carrying it the whole time.
    pub(crate) fire_rows: Vec<model_compiler::lower::Row>,
}

/// See [`Admitted`].
#[cfg(feature = "abi")]
fn admit(
    state: &Shell,
    step: &driver_api::local::PieStepDesc,
) -> Result<(Admitted, Box<dyn model::deployment_cuda::PlannedFamily>), i32> {
    use model_compiler::trace::FireClass;

    // A USER MASK IS SERVED NOW, and it used to be refused.
    //
    // The refusal was right for its time and said so: this shell read
    // neither `step.masks` nor the region bit, launched no custom-mask
    // kernel, and would have attended CAUSALLY over a mask the caller
    // supplied -- a wrong answer that looks like a right one.
    //
    // Three things landed since. `.wiki/driver/graph.md` §5 (1) made the
    // mask BUCKET state, so a resident element mask is published on every
    // fire and the `_custom` dispatch has an arm; the region table's
    // `PIE_REGION_SIG_MASK` bit reaches `Row::custom_mask`, so the
    // `HasCustomMask` guard can hold; and `brle` turns out never to have
    // needed porting -- the engine decodes its own runs host-side and
    // ships a packed bitset (`MaskWordsStorage::from_plan`).
    //
    // So what is left is a widen and a relayout, which is
    // `element_mask::from_words`. It REFUSES a table that does not
    // describe this fire rather than falling back, for the same reason
    // the whole entry used to refuse.
    let sub_batches = slice_of(step.sub_batch_indptr.ptr, step.sub_batch_indptr.len);
    if sub_batches.len() > 2 {
        eprintln!("[driver-cuda] launch: one sub-batch per step today");
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    let Some(model) = state.model.as_ref() else {
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    };
    // THE VALUE, not a fresh `Box<dyn PlannedFamily>`. Derived at load
    // (`LoadedModel::deployment`); this is the read.
    let dep = &model.deployment;
    // `trace()` is the one question a `Deployment` does not answer,
    // because a `ForwardPlan` is `model-compiler`'s and the family text
    // is what produces it. Everything else below is the value.
    let family = model::deployment_cuda::facts_from_hf(&model.checkpoint())
            .map_err(|e| i32::from(crate::Error::from(e)))?;

    let token_ids = slice_of(step.token_ids.ptr, step.token_ids.len);
    let position_ids = slice_of(step.position_ids.ptr, step.position_ids.len);
    let kv_indptr = slice_of(step.kv_page_indptr.ptr, step.kv_page_indptr.len);
    let kv_lens = slice_of(step.kv_last_page_lens.ptr, step.kv_last_page_lens.len);
    let qo_indptr = slice_of(step.qo_indptr.ptr, step.qo_indptr.len);
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
    // THE REGION TABLE, which is the seriation's output stated once. An
    // empty one is the legacy discipline — no seriation ran, so the fire
    // is one region of the default point — and not a refusal.
    let mut fire_rows = model_compiler::lower::rows_from_regions(
        rows,
        slice_of(step.sampling_indices.ptr, step.sampling_indices.len),
        slice_of(step.region_row_indptr.ptr, step.region_row_indptr.len),
        slice_of(step.region_sig.ptr, step.region_sig.len),
        slice_of(step.region_k.ptr, step.region_k.len),
    )
    .map_err(|drift| {
        eprintln!(
            "[driver-cuda] launch: the step's region table does not describe \
             its rows: {drift:?}"
        );
        PIE_STATUS_INVALID_ARGUMENT
    })?;
    // THE READOUT ROWS ARE THE WIRE'S NOW, and this used to be an
    // override that forced `samples: true` on every row.
    //
    // The reason it did is gone. `lower::epilogue` states a gather when
    // the fire samples fewer rows than it computes -- a prefill reads one
    // distribution per request out of a stream of one row per token --
    // but it used to emit the gather over the `LmHead` op's own operands,
    // whose output IS the logits buffer. The gather therefore wrote
    // `[sampled, hidden]` into a `[sampled, vocab]` allocation and the
    // head read what it had overwritten: all-zero logits on gemma-4 and
    // the hybrid. Claiming every row was the way around it, at the cost
    // of a prefill running the head over every token.
    //
    // The epilogue names its temp now (`Buffers::epilogue_gather`, sized
    // from the statement and carried on `Lowered` all along), so the
    // compaction is real and the wire's answer stands.
    //
    // A fire that names NO readout rows is the legacy discipline rather
    // than a service pass, and the answer is one row per REQUEST: the
    // last row a request contributes, `qo_indptr[r + 1] - 1`. Not the
    // fire's last row, which is what a shape that knows no request
    // boundaries can say and is only right at one request.
    if step.sampling_indices.len == 0 {
        for r in &mut fire_rows {
            r.samples = false;
        }
        for r in 0..requests {
            let last = (qo_indptr[r + 1] as usize).saturating_sub(1);
            if let Some(row) = fire_rows.get_mut(last) {
                row.samples = true;
            }
        }
    }

    // AND `multi_token` IS DERIVED FROM THE CSR, not taken on trust.
    //
    // `GuardPred::WindowOne` reads it (`.wiki/driver/graph.md` §4.1: the
    // window class is a row property now, not a class), so a row that
    // under-claims it puts a RAGGED fire on the decode arm — the wrong
    // attention kernel, and wrong logits rather than a refusal.
    //
    // `PIE_REGION_SIG_MULTI_TOKEN` is the engine's statement of the same
    // fact, but an EMPTY region table is legal — it is the legacy
    // discipline, "one region of the default point" — and the default
    // point is `multi_token: false`. So every prefill fired by a caller
    // that sends no table would answer WindowOne, which is exactly the
    // disagreement `fire_class_of` would not have had.
    //
    // `qo_indptr` cannot be silent: it is how the fire says which rows
    // belong to which request. A request contributing more than one token
    // row IS multi-token, whatever else was said, so the two are ORed
    // rather than one replacing the other.
    for r in 0..requests {
        let (lo, hi) = (qo_indptr[r] as usize, qo_indptr[r + 1] as usize);
        if hi.saturating_sub(lo) > 1 {
            for row in fire_rows.get_mut(lo..hi.min(rows)).unwrap_or_default() {
                row.multi_token = true;
            }
        }
    }
    // AN ADAPTER THIS DRIVER CANNOT APPLY IS REFUSED.
    //
    // Now that the region bit is read, a marked row reaches the
    // `HasLora` guard and the trace states `pie_lora_qkv_correction`.
    // The executor's arm for it returns `Ok(())` when `ctx.lora` is
    // `None` — which it always is, because nothing stages the table
    // yet — and that no-op is LOAD-BEARING for union captures: under
    // `GuardMode::Union` every arm lowers and the predicate is decided
    // at replay, so the arm has to be issuable with nothing to correct.
    //
    // So the refusal cannot live in the arm. It lives here, where the
    // question is whether this FIRE asked for something the driver
    // cannot do. Running it would apply no correction and return tokens
    // that look like a slightly worse model, which is the one failure
    // mode worth refusing over.
    //
    // THE ADAPTER IS APPLIED NOW, so there is nothing to refuse here.
    //
    // This used to turn away any fire whose region table carried
    // `PIE_REGION_SIG_LORA`, because the correction's executor arm
    // returns `Ok(())` when `ctx.lora` is `None` — and that no-op is
    // load-bearing for union captures, so the refusal could not live in
    // the arm. It lived here instead, where the question is whether
    // this FIRE asked for something the driver cannot do.
    //
    // It can now. `lane_for_instance` resolves each request's adapter
    // to an address, `lora_pins` names the q, v and x the correction
    // binds, and `llama_like_lora_stage` builds the state `ctx.lora`
    // carries. A fire whose lanes do not resolve still gets `None`,
    // which is the same correct no-op an adapter-free fire gets — the
    // difference being that it is now a fallback rather than the only
    // outcome.

    // A family that does not DECLARE a service class must be turned away
    // rather than traced: its text answers the three with `unreachable!`,
    // and a panic crossing the entry point is caught but costs the whole
    // request. Only the MTP family composes those passes.
    if !matches!(class, FireClass::Decode | FireClass::Prefill) && dep.recurrent.is_none() {
        eprintln!(
            "[driver-cuda] launch: {class:?} is an MTP service pass and \
             this family declares no trace for it"
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    }
    Ok((Admitted { class, rows, requests, fire_rows }, family))
}

/// Run the instance's registered program over the fire's logits.
///
/// `step_impl`'s SAMPLING phase, and the reason `ptir_programs` had a
/// writer and no reader until now. Sampling is a PTIR stage and not a
/// driver flag — top-p, top-k, temperature and argmax are ops a caller's
/// program states — so a fire that skipped this returned raw logits and
/// ignored every sampling parameter the request carried.
///
/// Returns `Ok(false)` when there is nothing to run, which is the common
/// case and not a failure: an instance with no program, a program that
/// compiled to nothing, or an instance whose channels this shell does not
/// hold. The caller then delivers logits the old way.
///
/// # Why a refusal here is not a failed request
///
/// A program that DECLINES a fire has said so deliberately, and one whose
/// inputs are not ready is waiting on the engine rather than broken.
/// Neither is a reason to fail the step: the cursors are left where they
/// were, so the next fire sees the same inputs and the same decision.
/// Only a device error propagates.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn run_program(
    // THE FIVE FIELDS THIS PHASE TOUCHES, not `&mut Shell` — the caller
    // has already borrowed `model`, `named_bufs`, `stream` and `alloc` out
    // of the shell, so a whole-shell borrow here is a conflict. Same wall
    // `deliver_logits` and `gdn_context` hit, same answer.
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    programs: &crate::gpu::program::Programs,
    control: &mut Option<crate::gpu::program::Control>,
    sessions: &mut std::collections::BTreeMap<u64, crate::gpu::program::session::Session>,
    disk: &crate::gpu::program::Disk,
    device_ordinal: i32,
    instance_id: u64,
    logits: (u64, u32, u32),
    rows: usize,
    // The row of `logits` this instance's program reads — the last row of
    // its token span, not its index.
    row: usize,
    alloc: &crate::gpu::device::Allocator,
    stream: &crate::gpu::device::OwnedStream,
) -> Result<bool, i32> {
    use crate::gpu::program::session::{Fired, Session};

    let Some(instance) = instances.get(&instance_id) else {
        return Ok(false);
    };
    let Some(compiled) = programs.get(instance.program_id) else {
        return Ok(false);
    };
    // THE EPILOGUE'S plan, not the first one.
    //
    // Sampling is an epilogue stage. `plans.first()` was the epilogue
    // only by the accident that no program in the tree has a prologue —
    // and `fwd.adapter` puts its `lora` sink in one, so the first program
    // that carries an adapter would have had its ADAPTER fired here and
    // its sampler never run, with the fire reporting a successful publish
    // either way.
    //
    // Falling back to the first stage keeps a package that states no
    // kinds working, which is what every fixture in the tree is.
    let stage = compiled
        .stage_of_kind(crate::gpu::program::runtime::stage_kind::EPILOGUE)
        .unwrap_or(0);
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

    // THE CONTROL KERNELS, ONCE. Same disk as the program runtime's on
    // purpose: the two share a key scheme, so a second cache directory
    // would recompile both every boot and neither would ever hit.
    if control.is_none() {
        let target = ptir_target(device_ordinal)?;
        let architecture = crate::gpu::program::compile::arch_flag(target.major, target.minor);
        match crate::gpu::program::Control::compile(disk, &architecture, "pie-cuda") {
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

    if !sessions.contains_key(&instance_id) {
        let session = Session::new(alloc, &shapes, stream.as_ref()).map_err(|error| {
            eprintln!("[driver-cuda] launch: cannot ring instance {instance_id}: {error}");
            PIE_STATUS_EXHAUSTED
        })?;
        sessions.insert(instance_id, session);
    }

    // The host planes, in the instance's channel ORDER — which is the
    // order a program indexes them by, and not the map's.
    let mut host: Vec<crate::gpu::program::channel::HostChannel> = Vec::with_capacity(channel_ids.len());
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
        &compiled,
        &plan,
        control,
        &mut host,
        logits,
        // ONE LANE per fire, and one ROW per lane. The fire itself is no
        // longer single-lane — `Prepared::build` takes one `Extents` per
        // lane and writes a record, a descriptor row and a channel-slot
        // row for each — so what is left is a caller that groups. When
        // one exists, this slice grows and the closure indexes it.
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

/// Publish the fire's readout: the LAST row's logits, out through the
/// instance's reader channel.
///
/// `step_impl`'s DELIVERY phase, promoted out of it. The convention until
/// the launch package's channel table is parsed: the roster's first
/// instance, its first registered channel with `host_role == READER`
/// whose cell is `[vocab]` f32. Device bf16 widens to the f32 wire on the
/// host.
///
/// Takes `debt` by `&mut Option` rather than returning a copy because the
/// two paths through here differ in WHO waits, not in what is produced: a
/// step that owes a completion hands the D2H's destination to the debt
/// and returns with the copy still queued, and a step that owes nothing
/// has already synchronized and can widen on this stack. Splitting those
/// into two functions would duplicate the channel search, which is the
/// part that is actually shared.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
/// Where request `r`'s logits sit in the fire's logits buffer.
///
/// Its ROW is `qo_indptr[r + 1] - 1` — the last row of its token span.
/// Its OFFSET is that row's ORDINAL among the rows the fire read out,
/// because `lower::epilogue` states a gather whenever the fire samples
/// fewer rows than it computes and the buffer is then `[sampled, vocab]`
/// in gather order.
///
/// The two coincide when every row samples, which is the decode case and
/// was the only case while the shell forced `samples: true`. A fire whose
/// readout rows are the wire's makes them differ on every prefill.
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

fn deliver_logits(
    // THE THREE FIELDS OF `Shell` THIS PHASE TOUCHES, and not `&mut
    // Shell`. Not a style choice: `model` and `named_bufs` below are
    // borrowed OUT of the shell by the caller, so a `&mut Shell` here is
    // a borrow conflict, and the fix that keeps working is to name the
    // disjoint fields rather than to widen the borrow. It also documents
    // the phase — delivery reads the roster and the channel table and
    // writes exactly one buffer.
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    logits_staging: &mut Option<crate::gpu::device::PinnedBuf>,
    frame: &PieFrameDesc,
    model: &LoadedModel,
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::gpu::bind::DispatchPlan,
    named_bufs: &std::collections::BTreeMap<
        model_compiler::trace::ValueId,
        crate::gpu::device::DeviceBuffer,
    >,
    stream: crate::gpu::device::StreamRef<'_>,
    rows: usize,
    // Where each request's token span ends, so its answer row can be
    // found. `qo_indptr[r + 1] - 1` is request `r`'s last row.
    qo_indptr: &[u32],
    // The rows the fire read out, in the order the epilogue's gather
    // compacted them. A request's logits live at its ORDINAL here, not at
    // its row: the buffer holds `[sampled, vocab]` once a gather runs.
    sampled_rows: &[u32],
    // The requests this fallback is for — the ones whose PTIR program did
    // not publish. A frame can be mixed, and a request that already has a
    // sampled answer must not also get a vocabulary.
    serve: &[usize],
    debt: &mut Option<FireDebt>,
) -> Result<(), i32> {
    use model_compiler::lower::Arg;
// ── Delivery: the LAST row's logits, out through the instance's
// reader channel. The convention until the launch package's channel
// table is parsed: the roster's first instance, its first registered
// channel with `host_role == READER` whose cell is `[vocab]` f32.
// Device bf16 widens to the f32 wire on the host.
let logits_value = (0..lowered.launches.len())
    .rev()
    .find_map(|i| {
        dplan.spec(i).outs.first().and_then(|a| match a {
            Arg::Named { value, .. } => Some(*value),
            Arg::Arena { .. } | Arg::Weight(_) => None,
        })
    });
let instance_ids = slice_of(frame.instance_ids.ptr, frame.instance_ids.len);
let vocab = usize::try_from(model.hf.vocab_size).unwrap_or(0);

// EVERY REQUEST, each its OWN reader channel and its OWN row.
//
// This used to take `instance_ids.first()` and `rows - 1`, so a frame
// with a roster of two published request 0's vocabulary and returned
// request 1 nothing at all. The row is not the index: request `r` owns
// `qo_indptr[r]..qo_indptr[r + 1]`, so its answer is at
// `qo_indptr[r + 1] - 1` — equal to `r` on a decode, and not on a
// prefill.
//
// And the row is not the OFFSET either, once the epilogue compacts. A
// fire that reads fewer rows than it computes states a gather, so the
// logits buffer holds `[sampled, vocab]` in gather order and a
// request's answer sits at its ORDINAL among the sampled rows. Those
// coincide exactly when every row samples — which is what the shell
// used to force, and why the distinction could be ignored.
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

// THE D2H IS ENQUEUED, NOT AWAITED. Its destination belongs to
// the debt rather than to this stack frame — a `Vec` here would
// be freed the moment this call returns, which with an
// asynchronous completion is before the copy lands.
//
// ONE copy carries every row, so N requests cost one D2H and N
// widenings rather than N copies.
if let (Some(lv), false) = (logits_value, readouts.is_empty())
    && let Some(buf) = named_bufs.get(&lv)
{
    match debt.as_mut() {
        Some(d) => {
            // The shell's buffer, grown to fit and reused. Not the
            // debt's: see `FireDebt::staging`.
            if logits_staging.as_ref().is_none_or(|p| p.len() < buf.len()) {
                *logits_staging = Some(
                    crate::gpu::device::PinnedBuf::new(buf.len())?,
                );
            }
            let pin = logits_staging.as_mut().expect("just sized");
            let view = (pin.as_slice().as_ptr(), buf.len());
            buf.copy_to_host(&mut pin.as_mut_slice()[..buf.len()], stream)?;
            d.staging = Some(view);
            d.readouts = readouts;
        }
        None => {
            // A step that owes nothing has already synchronized,
            // so the old shape is still correct for it.
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
/// `step_impl`'s DEVICE STATE phase, promoted out of it — and the reason
/// it could be is that it is the only phase before the launch that takes
/// `&mut Shell` and borrows nothing else. Extracting it is what lets
/// every shared borrow below it (`model`, the lowering, the stream, the
/// allocator) be taken AFTER the mutation rather than across it, which is
/// the borrow conflict that stopped `deliver_logits` from taking a
/// `&mut Shell` and would have stopped each of these too.
///
/// The stream and the allocator used to be built per fire. The stream
/// because nothing needed it to outlive the call, and the allocator
/// because it was convenient — but an allocator that POOLS and is rebuilt
/// every fire has no pool, so every buffer a fire wanted was a fresh
/// `cudaMalloc`. Both are the shell's now, which is a saving on its own
/// and is the precondition for run-ahead: a second fire cannot queue
/// behind the first onto a stream that dies with the first call.
///
/// # Reclaim, and when it waits
///
/// A fire's scratch cannot be freed while it runs and cannot be freed
/// from the callback — CUDA forbids calling the runtime from a host
/// function, and `cudaFree` is the runtime — so it is freed here. What
/// matters is WHEN.
///
/// This used to hold exactly one holder and `synchronize()` on it, which
/// is a run-ahead that never runs ahead: the call that would queue fire
/// n+1 blocked until fire n had finished. It cost nothing to notice while
/// issuing a fire took longer than running one, and it is the whole game
/// now that issue is 0.81 ms against 2.9 ms of work.
///
/// So: drop everything already retired without asking the driver to wait,
/// and wait only when the queue is at depth. `RUNAHEAD_DEPTH` is the
/// backpressure — the driver runs at most that many fires ahead of the
/// GPU, which bounds the SCRATCH it is holding rather than the time.
#[cfg(feature = "abi")]
fn ready_device_state(state: &mut Shell) -> Result<(), i32> {
    if state.fire_stream.is_none() {
        state.fire_stream =
            Some(crate::gpu::device::OwnedStream::new(0)?);
    }
    if state.fire_alloc.is_none() {
        state.fire_alloc = Some(crate::gpu::device::Allocator::new());
    }
    while state
        .in_flight
        .front()
        .is_some_and(|f| f.done.is_complete().unwrap_or(true))
    {
        let done = state.in_flight.pop_front().expect("just checked");
        retire(done);
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
/// `step_impl`'s GDN phase, promoted out of it — and the LARGEST of its
/// phases at a hundred lines, which is why it is worth the six
/// parameters. It also settles the design question the cut left open.
///
/// # Named fields, not a carrier
///
/// The middle phases each read the shell one way and write it another,
/// so `&mut Shell` is a borrow conflict against the `model`, `stream`
/// and `alloc` the caller already holds — the same wall `deliver_logits`
/// hit. Two answers were on the table: a `Fire<'a>` struct holding the
/// borrowed halves, or naming the fields each phase touches.
///
/// Naming them wins, and doing it three times is what says so. This
/// phase touches exactly ONE field of the shell (`gdn`) and reads four
/// things the caller has already resolved; a carrier would have handed
/// it the whole fire so it could reach one `Option`. The parameter list
/// IS the phase's contract, and a reader who wants to know whether the
/// recurrent slabs can affect the attention plan can answer it from the
/// signature.
///
/// The `alloc`/`stream` pair is shared and `gdn` is mutable, which the
/// compiler accepts at the call site because they are disjoint fields —
/// and that is the property a carrier would have hidden rather than
/// removed.
///
/// Returns the context the dispatch reads, and the slot-id buffer it
/// points into: the buffer is returned rather than dropped because the
/// context holds a raw pointer to it, and a fire that let it go would
/// bind a freed address.
#[cfg(feature = "abi")]
fn gdn_context(
    gdn: &mut Option<GdnState>,
    // Bumped when the state pool grows: the bases move and a capture
    // baked them.
    epoch: &mut u64,
    dep: &model::deployment::Deployment,
    step: &driver_api::local::PieStepDesc,
    requests: usize,
    alloc: &crate::gpu::device::Allocator,
    stream: &crate::gpu::device::OwnedStream,
) -> Result<(Option<crate::gpu::bind::GdnCtx>, Option<crate::gpu::device::DeviceBuffer>), i32> {
        use crate::gpu::bind::GdnCtx;

    let mut gdn_ctx: Option<GdnCtx> = None;
    let mut _slot_ids_buf: Option<crate::gpu::device::DeviceBuffer> = None;
    if let Some(shape) = dep.recurrent.as_ref() {
        let (conv_stride, state_stride, state_elem) =
            (shape.conv_stride, shape.state_stride, shape.state_elem);
        const GDN_SLOTS: u32 = 8;
        if (*gdn).is_none() {
            // THE PORTED CACHE OWNS THE LAYOUT. This used to allocate a
            // `(conv, recurrent)` pair per linear layer and derive every
            // stride here; `RecurrentStateCache` pools them and answers
            // both, which is what its 1,467 tested lines were ported for.
            let is_linear: Vec<bool> =
                (0..dep.layers).map(|l| shape.linear_layers.contains(&l)).collect();
            let cache = crate::gpu::pools::recurrent_state_cache::RecurrentStateCache::allocate_bf16_recurrent(
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
            let mut recurrent = alloc
                .alloc(
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
                state_elem_bytes: state_elem,
            });
        }
        let gdn_state = (*gdn).as_mut().expect("just ensured");
        // The ENGINE assigns slots: `rs_slot_ids`, one per request. RESET
        // zeroes a slot before the fire; BUFFER_WRITE routes the pass's
        // state into a buffer slot instead of the live one; FOLD copies
        // the accepted prefix back afterwards. See the fold below.
        let rs_slot_ids = slice_of(step.rs_slot_ids.ptr, step.rs_slot_ids.len);
        let rs_flags = slice_of(step.rs_slot_flags.ptr, step.rs_slot_flags.len);
        if rs_slot_ids.len() != requests {
            eprintln!("[driver-cuda] launch: hybrid fire without rs_slot_ids");
            return Err(PIE_STATUS_INVALID_ARGUMENT);
        }
        // THE BUFFER/FOLD FLAGS ARE ACCEPTED NOW, and that is directive
        // 4.2 of `.wiki/driver/graph.md`.
        //
        // They used to be refused wholesale — "rs fold/buffer flags await
        // spec-decode" — which left the tree carrying TWO mechanisms for
        // one job: this one, complete on the ABI side since v23/v24, and
        // the repair fire classes (`CommitAdvance`, `StateOnly`,
        // `FrozenVerify`) that existed only because this one was off.
        //
        // The mechanism is the whole argument for deleting them. A
        // speculative decode writes its tokens into a BUFFER slot and
        // folds only the accepted prefix into the live slot; a rejected
        // token is simply never folded, so there is nothing to repair.
        // `FrozenVerify` is "prefill plus a verify-stash store" — the
        // buffer IS the stash. `CommitAdvance` is "replay the confirmed
        // prefix" — the fold length IS that prefix.
        let rs_fold_lens = slice_of(step.rs_fold_lens.ptr, step.rs_fold_lens.len);
        let rs_buffer_slot_ids =
            slice_of(step.rs_buffer_slot_ids.ptr, step.rs_buffer_slot_ids.len);
        let rs_buffer_indptr =
            slice_of(step.rs_buffer_slot_indptr.ptr, step.rs_buffer_slot_indptr.len);
        let need_slots = rs_slot_ids.iter().copied().max().map_or(1, |m| m + 1);
        gdn_state.ensure_slots(need_slots, epoch, &alloc, &stream)?;
        // RESET, asked of the cache rather than written out. `reset_slot`
        // emits one strided fill per buffer -- a `Memset2D` whose rows are
        // the linear layers -- where this walked layers and issued a
        // contiguous fill each. Same bytes, one call, and the semantics
        // now live where they are tested.
        for (r, &slot) in rs_slot_ids.iter().enumerate() {
            if rs_flags.get(r).copied().unwrap_or(0) & driver_api::local::PIE_RS_FLAG_RESET
                == 0
            {
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
                // A BUFFER_WRITE row's pass writes the BUFFER slot, not
                // the live one — that is the whole of "the pass scatters
                // its own tokens". The buffer CSR names one slot per
                // buffered token; the pass writes the row's head, which
                // is its first entry.
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
        // THE FOLD, recorded on the fire's stream so it lands after the
        // pass that filled the buffer. Copying the accepted prefix's LAST
        // state into the live slot is the whole operation: a linear
        // state is a running summary, so the state after token `k` is the
        // state the next fire continues from, and the tokens past `k`
        // were rejected and are simply never folded.
        for (r, &live) in rs_slot_ids.iter().enumerate() {
            let f = rs_flags.get(r).copied().unwrap_or(0);
            if f & driver_api::local::PIE_RS_FLAG_FOLD == 0 {
                continue;
            }
            let (lo, hi) = match (rs_buffer_indptr.get(r), rs_buffer_indptr.get(r + 1)) {
                (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
                _ => continue,
            };
            // A device-resolved length is CLAMPED to the row's replay
            // length, which the ABI names as the bound. The port itself
            // is not read yet, so a device row folds its whole replay —
            // the conservative answer, and the one a non-speculative
            // fire produces anyway.
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
            let Some(&src_slot) = rs_buffer_slot_ids.get(lo + take - 1) else { continue };
            // The LINEAR halves only, which is the cache's own asymmetry
            // and worth taking from it rather than reinventing: a fold
            // restores recurrent state to the accepted prefix, but the MTP
            // pending-hidden row was already rebuilt from exactly those
            // accepted tokens, so copying it would overwrite the newer
            // value with an older one.
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
        let to_i32 = |v: u32| i32::try_from(v).unwrap_or(0);
        let _ = to_i32;
        gdn_ctx = Some(GdnCtx {
            k_h: shape.k_h,
            v_h: shape.v_h,
            k_d: shape.k_d,
            v_d: shape.v_d,
            conv_dim: shape.conv_dim,
            conv_k: shape.conv_k,
            n_groups: 0,
            // Still one base per MODEL layer, so nothing downstream moved:
            // the pooling changed where a base comes FROM, not what a
            // launch is handed.
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
/// `step_impl`'s KV phase. It touches ONE field of the shell — `kv` — and
/// bumps the array epoch when it grows, which is the second reason it is
/// worth being a function: growth MOVES base addresses, so every capture
/// that recorded one is stale, and the bump is what says so. Keeping that
/// pair in one place is what keeps a reader from having to notice it.
///
/// # Per-layer geometry, and the layers that own no pool
///
/// A family may share one layer's cache with another — gemma-4's trailing
/// layers project no KV of their own — so `kv_source(l)` says whose pool a
/// layer reads and only the SOURCES get an allocation. The views then
/// point every layer at its source's pages, which is why the returned
/// vector is as long as the layer count and the pool vector is not.
///
/// # What growth does not do
///
/// It REPLACES the pools without migrating pages. Decode continuity holds
/// while page demand is stable, which is the single-frame world; page
/// migration rides with `resize_pool`.
#[cfg(feature = "abi")]
#[allow(clippy::too_many_arguments)]
fn kv_pools_for(
    kv: &mut Option<KvState>,
    epoch: &mut u64,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    need_pages: u32,
    page_size: i32,
    alloc: &crate::gpu::device::Allocator,
    stream: &crate::gpu::device::OwnedStream,
    format: crate::layout::KvCacheFormat,
) -> Result<Vec<crate::gpu::bind::abi::KvCacheLayerView>, i32> {
    let (kv_heads_i, head_dim_i) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
    let head_dim_u = u32::try_from(head_dim_i).unwrap_or(0);
    let n = dep.layers;
    // Per-layer geometry, family-decided: gemma-4's two layer kinds
    // disagree on head dim, and its trailing layers own NO pages (they
    // attend through their source's — the load-time decision).
    let per_layer = crate::gpu::pools::kv_cache::PerLayer {
        head_dim: dep.attention.iter().map(|a| a.head_dim as i32).collect(),
        kv_source_layer: dep.attention.iter().map(|a| a.kv_source as i32).collect(),
        num_kv_heads: vec![kv_heads_i; n as usize],
    };
    // One set of pages has ONE shape, so a layer that reads through
    // another's must have that layer's dims. gemma-4 holds this without
    // trying — `kv_source` searches by the same predicate `head_dim_of`
    // keys on — but that is an invariant spread across two functions in
    // another crate. A violation would not crash: every shared layer
    // would read its source's pages at its own stride and emit plausible
    // tokens. It matters HERE because `layer_view` reports an aliased
    // layer's dims as its SOURCE's.
    per_layer.check_sharing()?;

    let grow = !matches!(&(*kv), Some(kv) if kv.num_pages >= need_pages);
    if grow {
        let layout = crate::gpu::pools::kv_cache::KvCacheLayout::plan_per_layer(
            n as i32,
            need_pages as i32,
            page_size,
            kv_heads_i,
            per_layer,
            format,
            false,
        )?;

        let mut ops = crate::gpu::pools::kv_cache_live::LiveKvCacheOps::new(
            stream.as_ref().as_raw().cast(),
            alloc,
        );
        let cache = crate::gpu::pools::kv_cache_live::KvCache::materialize(layout, &mut ops)?;
        let mut held = ops.into_held();
        // `materialize` does not zero, and the C++ did not either — but
        // the shell's hand-built pools did, and a page read before its
        // first write is otherwise whatever the allocator last had.
        for b in &mut held {
            b.memset(0, stream.as_ref())?;
        }

        // NOTE: growth REPLACES the pages without migrating them — decode
        // continuity holds while the page demand is stable, which is the
        // single-frame smoke's world. Page migration rides with resize_pool.
        //
        // AND IT MOVES BASE ADDRESSES, so every capture that recorded one is
        // stale. Same rule as `Scratch`'s own growth, and the same
        // relocation: `install_kv` owns the bump so neither this path nor
        // `resize_pool` can install a moved pool without one.
        crate::gpu::serve::state::install_kv(
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
    layers: Vec<crate::gpu::bind::abi::KvCacheLayerView>,
    /// Which rows carry a sampled logit, by fire-row index.
    ///
    /// Returned as well as uploaded because delivery indexes by SAMPLED
    /// ORDINAL — `logits_row_of` needs the host copy after the fire.
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
    d_valid: crate::gpu::device::DeviceBuffer,
}

/// Grow the KV pool to fit the step and upload every descriptor array it
/// needs. The fire's first phase, `lap("kv+arrays")`.
///
/// TAKES ITS FIELDS, NOT THE SHELL, and that is not a style choice. The
/// phase writes `kv` and `fire_arrays` while reading `fire_alloc` and
/// `fire_stream`, which the borrow checker permits only because they are
/// distinct fields of one struct. A `&mut Shell` parameter collapses that
/// distinction and the body stops compiling. The parameter list is
/// therefore an accurate statement of what the phase touches — which is
/// the property that makes the cut worth making.
#[allow(clippy::too_many_arguments)]
fn kv_and_arrays(
    kv: &mut Option<KvState>,
    fire_arrays: &mut crate::gpu::fire::scratch::Scratch,
    format: crate::layout::KvCacheFormat,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    alloc: &crate::gpu::device::Allocator,
    stream: &crate::gpu::device::OwnedStream,
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
    let need_pages = required_kv_pages.max(
        kv_indices.iter().copied().max().map_or(1, |m| m + 1),
    );
    let page_size: i32 = 16;
    // Re-derived here as well as inside `kv_pools_for`, because the
    // attention plans below want the same two numbers and returning them
    // would make the phase's result a tuple whose second half is a
    // restatement of its arguments.
    let (kv_heads_i, head_dim_i) =
        (model.hf.num_key_value_heads, model.hf.head_dim_kernel.max(model.hf.head_dim));
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

    // The fire's descriptor arrays, POOLED like the arena and for the same
    // reason: a capture bakes an address, so the buffer has to be the same
    // one next fire with only its contents refreshed. Slots are positional
    // and this is the whole list of them.
    const S_IDS: usize = 0;
    const S_POS: usize = 1;
    const S_KV_INDICES: usize = 2;
    const S_KV_INDPTR: usize = 3;
    const S_KV_LENS: usize = 4;
    const S_QO: usize = 5;
    const S_W_PAGE: usize = 6;
    const S_W_OFF: usize = 7;
    const S_SAMPLED: usize = 8;
    let d_ids = fire_arrays.upload_u32(alloc, S_IDS, token_ids, stream.as_ref())?;
    let d_pos = fire_arrays.upload_u32(alloc, S_POS, position_ids, stream.as_ref())?;
    let d_kv_indices =
        fire_arrays.upload_u32(alloc, S_KV_INDICES, kv_indices, stream.as_ref())?;
    let d_kv_indptr =
        fire_arrays.upload_u32(alloc, S_KV_INDPTR, kv_indptr, stream.as_ref())?;
    let d_kv_lens =
        fire_arrays.upload_u32(alloc, S_KV_LENS, kv_lens, stream.as_ref())?;
    let d_qo = fire_arrays.upload_u32(alloc, S_QO, qo_indptr, stream.as_ref())?;
    // WHICH ROWS the epilogue gathers, from the rows the step described.
    // Derived here rather than taken from `sampling_indices` directly, so
    // the pointer and the guard that produced it cannot disagree: the
    // lowering states a gather exactly when `sampled < window.len()`, and
    // it counts the same `Row::samples` this reads.
    let sampled_rows: Vec<u32> = fire_rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.samples.then_some(u32::try_from(i).unwrap_or(0)))
        .collect();
    let d_sampled = if sampled_rows.len() == rows {
        // Every row sampled means no gather is stated, and a pointer for
        // a launch nobody makes is one more thing to keep in step.
        core::ptr::null()
    } else {
        fire_arrays.upload_u32(alloc, S_SAMPLED, &sampled_rows, stream.as_ref())?
    };

    // Write targets: each request appends its NEW tokens at the CSR tail.
    // Decode appends one token at `len - 1`; prefill appends its whole
    // window ending there.
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
    let d_w_page = fire_arrays.upload_u32(alloc, S_W_PAGE, &w_page, stream.as_ref())?;
    let d_w_off = fire_arrays.upload_u32(alloc, S_W_OFF, &w_off, stream.as_ref())?;
    let mut d_valid = alloc.alloc(rows)?;
    d_valid
        .copy_from_host(&vec![1u8; rows], stream.as_ref())?;

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
/// RAW POINTERS AND COPIES, deliberately: the plans live in `FireScratch`
/// for the driver's lifetime, and returning borrows of them would keep
/// `state.scratch` mutably borrowed across the whole rest of the fire.
struct AttnPlans {
    decode_plan: *mut std::ffi::c_void,
    decode_plan_full: *mut std::ffi::c_void,
    prefill_plan: *mut std::ffi::c_void,
    workspace: crate::gpu::bind::abi::AttentionWorkspaceView,
    /// The workspace the prefill arm binds — which is the DECODE one for
    /// the planless family, because it never raised a prefill plan and a
    /// view of an unplanned workspace is not one a kernel may read.
    prefill_workspace: crate::gpu::bind::abi::AttentionWorkspaceView,
    /// Does the lowered text state the flashinfer DECODE dispatch? Read by
    /// the score sink, which sizes a one-wide window for it.
    states_decode_dispatch: bool,
}

/// Allocate the workspaces on first fire, then raise EVERY plan the
/// geometry permits. The fire's `lap("attn-plan")` phase.
///
/// Every plan, not just the one this fire's text states: under
/// `GuardMode::Union` both arms of an attention guard are recorded, and a
/// capture that walks an arm whose plan was never raised is abandoned.
/// Prepared state belongs to the BUCKET rather than to the arm
/// (`.wiki/driver/graph.md` §5 ①).
fn raise_attn_plans(
    scratch_slot: &mut Option<FireScratch>,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    lowered: &model_compiler::lower::Lowered,
    geom: PlanGeometry<'_>,
    raw_stream: *mut std::ffi::c_void,
) -> Result<AttnPlans, i32> {
    use crate::gpu::fire::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use crate::gpu::bind::{DecodePlan, PrefillPlan};

    let PlanGeometry { kv_indptr, kv_lens, qo_indptr, kv_heads, head_dim, page_size } = geom;
    let mut sops = LiveStagingOps;
    if scratch_slot.is_none() {
        let ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2)?;
        let prefill_ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2)?;
        *scratch_slot = Some(FireScratch {
            ws,
            prefill_ws,
            decode_plan: DecodePlan::new(),
            decode_plan_full: DecodePlan::new(),
            prefill_plan: PrefillPlan::new(),
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
    // Plan for the dispatch the LOWERED text actually states — not the
    // fire class: the hybrid's `prefill_decode` fact routes a
    // single-request decode through the PREFILL flashinfer path
    // (`TokensLE(1)` resolves at lower time).
    let states_decode_dispatch = lowered
        .kernels
        .iter()
        .any(|k| k == "attn::dispatch_attention_flashinfer_decode");
    ws.begin_plan_update(&mut sops)?;
    // EVERY PLAN THE GEOMETRY PERMITS, not just the one this fire's text
    // states. Under `GuardMode::Union` both arms of an attention guard are
    // recorded, and a capture that walks an arm whose plan was never
    // raised is abandoned — so prepared state belongs to the bucket rather
    // than to the arm (`.wiki/driver/graph.md` §5 ①). It is also the
    // precondition for §4.1: a merged Decode/Prefill graph needs BOTH
    // classes' plans standing whenever either is captured.
    //
    // The cost is plan-raise time. It is not runtime waste — the arm a
    // fire does not take is skipped by the conditional, not executed.
    let planless_prefill = dep.prefill == model::deployment::PrefillStyle::Planless;
    let decode_plan_full_ptr = if let Some((d_sliding, d_full)) = dep.decode_head_dims() {
        // TWO decode plans, one per layer kind — the C++'s
        // `decode_plan_sliding` / `decode_plan_full` pair, because
        // the kinds disagree on head dim and the planner bakes it in.
        decode_plan.plan_decode_variant(
            kv_indptr,
            model.hf.num_attention_heads,
            kv_heads,
            d_sliding as i32,
            page_size,
            ws.view(),
            raw_stream,
            false,
            false,
            -1,
        );
        decode_plan_full.plan_decode_variant(
            kv_indptr,
            model.hf.num_attention_heads,
            kv_heads,
            d_full as i32,
            page_size,
            ws.view(),
            raw_stream,
            false,
            true,
            -1,
        );
        decode_plan_full.as_ptr()
    } else {
        decode_plan.plan_decode(
            kv_indptr,
            model.hf.num_attention_heads,
            kv_heads,
            head_dim,
            page_size,
            ws.view(),
            raw_stream,
            false,
            -1,
        );
        core::ptr::null_mut()
    };
    // gemma-4's prefill is PLANLESS (it plans internally per fire, off the
    // host CSR mirrors) and its 512-wide layers take the naive kernel —
    // there is nothing to pre-plan, so it is the one plan that stays
    // unraised.
    if !planless_prefill {
        prefill_ws.begin_plan_update(&mut sops)?;
        prefill_plan.plan_prefill(
            qo_indptr,
            kv_indptr,
            kv_lens,
            model.hf.num_attention_heads,
            kv_heads,
            head_dim,
            page_size,
            prefill_ws.view(),
            raw_stream,
            false,
            -1,
        );
        prefill_ws.end_plan_update(&mut sops, raw_stream);
    }
    ws.end_plan_update(&mut sops, raw_stream);

    Ok(AttnPlans {
        decode_plan: decode_plan.as_ptr(),
        decode_plan_full: decode_plan_full_ptr,
        prefill_plan: prefill_plan.as_ptr(),
        workspace: ws.view(),
        prefill_workspace: if planless_prefill { ws.view() } else { prefill_ws.view() },
        states_decode_dispatch,
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

/// Size and publish the resident seam buffers this fire's arms may read.
///
/// UNCONDITIONALLY, and that is the point. `WantsAttnScore` and
/// `HasCustomMask` are FOLDED predicates: one exec serves the fire that
/// wants scores and the fire that does not, and under `GuardMode::Union`
/// both arms are recorded whether or not this fire takes either. So the
/// question is not "what does this fire need" but "what could any arm
/// need". A capture that walks an arm whose pin was never published is
/// abandoned. The cost is resident memory, not runtime: the arm a fire
/// does not take is skipped by the conditional rather than executed.
#[allow(clippy::too_many_arguments)]
fn publish_seam_pins(
    fire_arrays: &mut crate::gpu::fire::scratch::Scratch,
    alloc: &crate::gpu::device::Allocator,
    stream: &crate::gpu::device::OwnedStream,
    dep: &model::deployment::Deployment,
    model: &LoadedModel,
    step: &driver_api::local::PieStepDesc,
    named_widths: &std::collections::BTreeMap<model_compiler::trace::ValueId, u32>,
    geom: PlanGeometry<'_>,
    rows: usize,
    states_decode_dispatch: bool,
    // How many score rows the sink keeps — `crate::boot`'s, so the one
    // parse of the knob reaches here rather than a second read of it.
    attn_score_window: u32,
) -> Result<SeamPins, i32> {
    let PlanGeometry { kv_indptr, kv_lens, qo_indptr, page_size, .. } = geom;
    for (&v, &w) in named_widths {
        // fp32-wide: the GDN seam pins are f32; llama-like's are bf16 and
        // simply leave half the pin unread.
        fire_arrays.named(alloc, v, rows * w as usize * 4, stream.as_ref())?;
    }
    // THE SCORE SINK IS UNCONDITIONAL, and that is a reversal.
    //
    // It used to be null on purpose: a fire that wants no scores prepares
    // no score path, the capturing dispatch refuses without one, and
    // refusing before the launcher is reached beats an exception crossing
    // the C ABI with nowhere to go. Sound — but it makes the union decline
    // every lowering that so much as MENTIONS `_capture`, which is the one
    // case the union was built for. `WantsAttnScore` is a FOLDED
    // predicate: one exec is supposed to serve the fire that wants scores
    // and the fire that does not, and under `GuardMode::Union` both arms
    // are recorded whether or not this fire takes either.
    //
    // So the question is not "what does this fire need" but "what could
    // any arm need", and the answer is published every time. The cost is
    // resident memory and plan-raise time; it is NOT runtime waste,
    // because the arm a fire does not take is skipped by the conditional
    // rather than executed. `.wiki/driver/graph.md` §5 ①.
    let score_window = if states_decode_dispatch {
        1
    } else {
        attn_score_window
    };
    let sink = crate::gpu::fire::attn_score::plan_score_sink(
        kv_indptr,
        kv_lens,
        page_size,
        u32::try_from(model.hf.num_attention_heads).unwrap_or(0),
        score_window,
    );
    let (d_scores, d_folded, d_score_indptr) = match sink {
        // A sink too large to publish (the prefill window grows with the
        // context) keeps the old answer: null, and the capturing arm
        // declines exactly as it did.
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

    // THE CUSTOM MASK, likewise unconditional. `HasCustomMask` is folded
    // too, so the `_custom` arm has to be recordable whether or not this
    // fire stages a mask. With nothing staged the resident form is plain
    // causal — the same answer the unmasked arm computes — so taking the
    // arm then is correct rather than merely safe. The addresses do not
    // move either way.
    // THE CALLER'S MASK WHEN THERE IS ONE, causal otherwise -- and the
    // caller's is REFUSED rather than replaced when it does not describe
    // this fire, because a fire asked to attend over a supplied mask and
    // served causally returns an answer that looks exactly like a correct
    // one. That was the whole reason `admit` used to turn the frame away.
    let staged = (step.has_user_mask != 0).then(|| {
        crate::gpu::fire::page_mask::element_mask::from_words(
            qo_indptr,
            kv_indptr,
            kv_lens,
            page_size,
            slice_of(step.masks.request_indptr.ptr, step.masks.request_indptr.len),
            slice_of(step.masks.word_indptr.ptr, step.masks.word_indptr.len),
            slice_of(step.masks.words.ptr, step.masks.words.len),
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
        None => crate::gpu::fire::page_mask::element_mask::plan_causal(
            qo_indptr,
            kv_indptr,
            kv_lens,
            page_size,
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

    // The driver-owned attention landing, resolved BEFORE `named_bufs`
    // borrows the map: a fire whose op join names no output slot lands
    // here instead of losing its graph. Null when the family states its
    // attention output as an SSA arg (gemma-4 does).
    let d_attn_out = if dep.attn_output == model::deployment::AttnOutput::DriverPinned {
        fire_arrays.attn_out(
            alloc,
            rows * model.hf.num_attention_heads as usize
                * usize::try_from(model.hf.head_dim).unwrap_or(0)
                * 2,
        )?
    } else {
        core::ptr::null_mut()
    };


    Ok(SeamPins { d_scores, d_folded, d_score_indptr, d_mask, d_mask_indptr, d_attn_out })
}

/// Which SSA value holds the attention QUERY, and where its output lands.
///
/// Both are read off the lowering's own join rather than counted off
/// launch positions. A value found by counting is a fact derived from
/// where a statement SITS, and that is false under `GuardMode::Union`,
/// where every guard arm is present and the launch after the dispatch
/// belongs to some other body.
///
/// `(None, None)` for a family that states [q, o] as SSA args — gemma-4
/// does — because then there is no pin to find.
/// Which SSA values the adapter correction reads and writes.
///
/// The same read [`attention_pins`] makes, for the same reason: a value
/// found by COUNTING launches is a fact derived from where a statement
/// sits, and that is false under `GuardMode::Union`, where every guard
/// arm is present and the neighbour belongs to some other body. So this
/// finds the launch by NAME and reads its own operands.
///
/// `(q, v, x)` — the two projection outputs the correction adds into,
/// and the projection INPUT it reads. The first two are the launch's
/// own args, which is what the executor's arm binds as `bound.args[0]`
/// and `[1]`; the third is its FOREIGN operand (`LaunchSpec::aux[0]`),
/// which is what the arm binds as `aux_slot(0)`.
///
/// The input being an aux is what makes this resolvable at all. It is
/// not one of the correction's args — the statement does not carry it —
/// and finding it any other way would mean knowing which named value
/// the family's norm placement produces. The lowering already wrote it
/// down.
///
/// `None` when the lowering states no correction, which is every
/// adapter-free fire.
struct LoraPins {
    /// The q-site output rows.
    q: model_compiler::trace::ValueId,
    /// The v-site output rows.
    v: model_compiler::trace::ValueId,
    /// The projection input — normed value under `Pre`, residual stream
    /// under `Post`, and the lowering knows which because the family
    /// text stated it.
    x: model_compiler::trace::ValueId,
}

fn lora_pins(
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::gpu::bind::DispatchPlan,
) -> Option<LoraPins> {
    use model_compiler::lower::Arg;
    let at = lowered
        .launches
        .iter()
        .position(|x| lowered.kernels[x.kernel as usize] == "pie_lora_qkv_correction")?;
    let named = |a: &Arg| match a {
        Arg::Named { value, .. } => Some(*value),
        Arg::Arena { .. } | Arg::Weight(_) => None,
    };
    let mut args = lowered.launches[at]
        .args
        .clone()
        .filter_map(|ai| named(&lowered.args[ai as usize]));
    let q = args.next()?;
    let v = args.next()?;
    let x = dplan.spec(at).aux.first().and_then(named)?;
    Some(LoraPins { q, v, x })
}

fn attention_pins(
    dep: &model::deployment::Deployment,
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::gpu::bind::DispatchPlan,
    states_decode_dispatch: bool,
) -> Result<(Option<model_compiler::trace::ValueId>, Option<usize>), i32> {
    use model_compiler::lower::Arg;
    // The guard-owned attention values, discovered from the lowering as
    // the smokes discovered them. gemma-4 has NONE: both its attention
    // forms state [q, o] as SSA args, so the pins stay null.
    if dep.attn_output != model::deployment::AttnOutput::DriverPinned {
        return Ok((None, None));
    }
    let dispatch_name = if states_decode_dispatch {
        "attn::dispatch_attention_flashinfer_decode"
    } else {
        "attn::dispatch_attention_flashinfer_prefill_bf16"
    };
    let Some(fi) = lowered
        .launches
        .iter()
        .position(|x| lowered.kernels[x.kernel as usize] == dispatch_name)
    else {
        eprintln!(
            "[driver-cuda] launch: the lowering states no {dispatch_name}"
        );
        return Err(PIE_STATUS_UNSUPPORTED);
    };
    let q_pin = lowered.launches[fi]
        .args
        .clone()
        .find_map(|ai| match &lowered.args[ai as usize] {
            Arg::Named { value, .. } => Some(*value),
            _ => None,
        });
    // The dispatch's OUTPUT, read off its own op join.
    //
    // This used to be `launches[fi + 1]`'s first operand — "the launch
    // after the dispatch is the o_proj, and its input is the slot the
    // dispatch wrote". True under `Resolve`, where the guard has
    // already deleted every arm the fire did not take, and false under
    // `Union`, where every arm is present and the next launch belongs
    // to some other guard's body.
    //
    // A value found by counting launches is a fact derived from where
    // a statement SITS. The join says it: the attention statement
    // carries one arg (q) and its output placement, which is exactly
    // the slot wanted. Same read the executor's arms make.
    // Prefer the join, fall back to the neighbour.
    //
    // The join is the STATED read: the attention statement carries its
    // output placement, which is the slot the o_proj goes on to read.
    // Where a deployment spells the attention with [q, o] as SSA args
    // the join records no output of its own, and there the old
    // positional read is still the only answer available.
    //
    // Positional is what breaks under `Union` — every guard arm is
    // present, so the launch after the dispatch belongs to some other
    // body — which is why the join is tried first rather than second.
    // A JOIN THAT NAMES NO SLOT IS NOT A REFUSAL any more.
    //
    // This used to return `PIE_STATUS_UNSUPPORTED`, and before the
    // union it also fell back to the neighbour read -- "the launch
    // after the dispatch is the o_proj, so its input is the slot the
    // dispatch wrote". That is a fact read off where a statement SITS,
    // and it is false under `GuardMode::Union`, where every arm is
    // present and the neighbour belongs to some other body.
    //
    // But `AttnCtx::o_out` is a DRIVER-owned pointer -- the whole
    // reason it exists is that the region's launches record no SSA
    // output of their own. So a fire whose join names no slot gets a
    // driver-owned landing buffer, and keeps its graph
    // (`.wiki/driver/graph.md` §5 (2)).
    Ok((q_pin, attention_landing(lowered, dplan, fi)))
}

/// Where a sampling program reaches into the shell.
///
/// A STRUCT OF FIELDS, not the shell: the phase writes `ptir_control` and
/// `ptir_sessions` while reading `model` and the lowering, both of which
/// are themselves borrows of the shell. Naming the fields is what keeps
/// those disjoint.
struct SamplingSites<'a> {
    instances: &'a std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &'a std::collections::BTreeMap<u64, ChannelState>,
    programs: &'a crate::gpu::program::Programs,
    control: &'a mut Option<crate::gpu::program::Control>,
    sessions: &'a mut std::collections::BTreeMap<u64, crate::gpu::program::session::Session>,
    disk: &'a crate::gpu::program::Disk,
    device_ordinal: i32,
    named_bufs: &'a std::collections::BTreeMap<model_compiler::trace::ValueId, crate::gpu::device::DeviceBuffer>,
}

/// Run each request's sampling PROGRAM, and report which requests still
/// need raw logits.
///
/// EVERY REQUEST, each over its OWN row. This used to fire only
/// `instance_ids.first()` and then let a successful publish suppress
/// `deliver_logits` for the whole frame — so a two-request batch sampled
/// request 0 and returned request 1 NOTHING AT ALL: no sample, because
/// its program never ran, and no logits, because request 0's had.
///
/// A frame can be MIXED — one request bound to a sampling program and
/// another not — which is why the result is a set rather than a flag.
///
/// Everything about this degrades to the old behaviour: no program, a
/// program that declines, inputs not ready, or channels this shell does
/// not hold all fall through to the raw logits.
#[allow(clippy::too_many_arguments)]
fn run_sampling_programs(
    sites: SamplingSites<'_>,
    model: &LoadedModel,
    lowered: &model_compiler::lower::Lowered,
    dplan: &crate::gpu::bind::DispatchPlan,
    frame: &PieFrameDesc,
    alloc: &crate::gpu::device::Allocator,
    stream: &crate::gpu::device::OwnedStream,
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
        disk,
        device_ordinal,
        named_bufs,
    } = sites;
    let vocab = u32::try_from(model.hf.vocab_size).unwrap_or(0);
    let readout = (0..lowered.launches.len()).rev().find_map(|i| {
        dplan.spec(i).outs.first().and_then(|a| match a {
            Arg::Named { value, .. } => Some(*value),
            Arg::Arena { .. } | Arg::Weight(_) => None,
        })
    });
    let logits_base = readout
        .and_then(|v| named_bufs.get(&v))
        .map_or(0, |b| b.as_ptr() as u64);
    // EVERY REQUEST, each over its OWN row.
    //
    // This used to fire only `instance_ids.first()`, and then let a
    // successful publish suppress `deliver_logits` for the whole frame —
    // so a two-request batch sampled request 0 and returned request 1
    // NOTHING AT ALL: no sample, because its program never ran, and no
    // logits, because request 0's had.
    //
    // A request's logits row is the last row of its token span, which is
    // what `qo_indptr` states: request `r` owns `qo_indptr[r]
    // ..qo_indptr[r + 1]`, so its row is `qo_indptr[r + 1] - 1`. On a
    // decode that is `r`; on a prefill it is not, and the difference is
    // the whole reason to read the indptr rather than count.
    //
    // Still one lane per fire — `program::run`'s grouping is unbuilt — so
    // this is N single-lane fires rather than one N-lane fire. Slower and
    // correct, which is the right order.
    let instance_ids = slice_of(frame.instance_ids.ptr, frame.instance_ids.len);
    // Which requests still need raw logits: the ones whose program did
    // not publish. A frame can be MIXED — one request bound to a sampling
    // program and another not — and each half has to be served, which is
    // why this is a set rather than a flag.
    let mut unsampled: Vec<usize> = Vec::new();
    for (r, &iid) in instance_ids.iter().enumerate() {
        let Some(&end) = qo_indptr.get(r + 1) else { break };
        // The ORDINAL, not the row — see `logits_row_of`. A sampling
        // program reads the same compacted buffer the raw readback does.
        let row = logits_row_of(end as usize, rows, &sampled_rows);
        if run_program(
            instances,
            channels,
            programs,
            control,
            sessions,
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

/// Ring every instance the frame names, BEFORE the forward runs.
///
/// # Why this is not lazy any more
///
/// `run_program` used to create a session on first use, and `run_program`
/// runs AFTER the forward — it is the sampler. That is fine for a
/// sampler and wrong for anything else, because a channel cell's ADDRESS
/// comes from a session's `Rings`, and the one thing that needs an
/// address before the forward is the thing the forward is supposed to
/// apply: `fwd.adapter` puts its `lora` sink in the program's PROLOGUE,
/// and a prologue is by definition before.
///
/// So the ordering was the whole of LoRA's remaining blocker — not a
/// missing function. `model::lora::read_lora_sink` already resolves
/// which channels an adapter arrives on; what it could not do was turn
/// a channel index into a pointer, because the ring did not exist yet.
///
/// FAILURES ARE NOTED AND SWALLOWED, deliberately. A frame whose
/// instance cannot be ringed still has a forward to run and raw logits
/// to deliver; refusing here would turn a missing sampler into a dead
/// request. `run_program` finds no session and declines, which is the
/// path it already had.
fn ensure_sessions(state: &mut Shell, frame: &PieFrameDesc) {
    // The stream and the allocator are separate FIELDS on purpose — see
    // the north star §7: grouping them into one struct collapses a
    // disjoint borrow the fire path depends on.
    let (Some(alloc), Some(stream)) = (state.fire_alloc.as_ref(), state.fire_stream.as_ref())
    else {
        return;
    };
    let ids: Vec<u64> = slice_of(frame.instance_ids.ptr, frame.instance_ids.len).to_vec();
    for id in ids {
        if state.ptir_sessions.contains_key(&id) {
            continue;
        }
        let Some(instance) = state.instances.get(&id) else { continue };
        let Some(shapes) = instance_ring_shapes(instance, &state.channels) else {
            continue;
        };
        match crate::gpu::program::session::Session::new(alloc, &shapes, stream.as_ref()) {
            Ok(session) => {
                state.ptir_sessions.insert(id, session);
            }
            Err(error) => {
                eprintln!("[driver-cuda] launch: cannot ring instance {id}: {error}");
            }
        }
    }
}

pub(crate) fn step_impl(
    state: &mut Shell,
    frame: &PieFrameDesc,
    step: &driver_api::local::PieStepDesc,
    // `owes` is the debt this step carries when it is the frame's LAST:
    // `None` for the earlier steps, which owe nothing because a frame
    // completes once. A step handed one enqueues an asynchronous
    // completion and does NOT synchronize; a step handed `None`
    // synchronizes, because the next step's work depends on it and the
    // producer→consumer ordering inside a frame is what makes steps
    // sequential in the first place.
    owes: Option<(PieCompletion, Vec<*mut driver_api::local::PieTerminalCell>)>,
) -> Result<(), i32> {
    use crate::gpu::bind::{AttnCtx, AttnRegions, DispatchCtx, Frame, Resolver, run};
    use model_compiler::lower::Arg;
    use model_compiler::trace::ValueId;

    let t_head = std::time::Instant::now();
    let (Admitted { class, rows, requests, fire_rows }, family) = admit(state, step)?;
    // THE MUTATION FIRST, AND THEN THE BORROWS. `ready_device_state` takes
    // `&mut Shell`, so every shared borrow this function goes on to hold —
    // `model`, the lowering, the stream, the allocator — has to be taken
    // after it. Reading the layer count out as a NUMBER rather than
    // keeping `model` alive across the call is the whole of what that
    // costs, and it is what lets the phase be a function at all.
    let layers = state
        .model
        .as_ref()
        .ok_or(PIE_STATUS_INVALID_ARGUMENT)?
        .hf
        .num_hidden_layers;
    ready_device_state(state)?;
    // BEFORE the forward, so a prologue's channel cells have addresses.
    // See `ensure_sessions`.
    ensure_sessions(state, frame);
    let model = state.model.as_ref().ok_or(PIE_STATUS_INVALID_ARGUMENT)?;
    // Derived at load, read here. See `LoadedModel::deployment`.
    let dep = &model.deployment;
    let token_ids = slice_of(step.token_ids.ptr, step.token_ids.len);
    let position_ids = slice_of(step.position_ids.ptr, step.position_ids.len);
    let kv_indices = slice_of(step.kv_page_indices.ptr, step.kv_page_indices.len);
    let kv_indptr = slice_of(step.kv_page_indptr.ptr, step.kv_page_indptr.len);
    let kv_lens = slice_of(step.kv_last_page_lens.ptr, step.kv_last_page_lens.len);
    let qo_indptr = slice_of(step.qo_indptr.ptr, step.qo_indptr.len);

    sg_trace(|| format!("  head {:?}", t_head.elapsed()));
    let t_low = std::time::Instant::now();
    // ── The lowering, or the one this shape already has. ──
    //
    // Everything between here and `DispatchPlan` is a pure function of the
    // key, and it costs ~3.3 ms on a 0.6B decode. See `Shell::lowerings`.
    let key = LoweringKey {
        model_id: u64::from(layers.unsigned_abs()),
        class,
        rows: u32::try_from(rows).unwrap_or(0),
        rows_digest: digest_rows(&fire_rows),
        union_asked: state.boot.supergraph && dep.recurrent.is_none(),
    };
    if !state.lowerings.contains_key(&key) {
        let built = build_lowering(family.as_ref(), class, &fire_rows, key.union_asked)?;
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
    // ── Workspace + plan caches: DRIVER-lifetime, first-launch built. ──
    let AttnPlans {
        decode_plan,
        decode_plan_full,
        prefill_plan,
        workspace,
        prefill_workspace,
        states_decode_dispatch,
    } = raise_attn_plans(
        &mut state.scratch,
        dep,
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
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..lowered.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
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

    let named_bufs = &state.fire_arrays.named;

    lap("attn-plan");
    // ── The hybrid's GDN context: driver-owned slabs, instance slots. ──
    let (gdn_ctx, _slot_ids_buf) =
        gdn_context(
            &mut state.gdn,
            &mut state.fire_arrays.epoch,
            dep,
            step,
            requests,
            alloc,
            stream,
        )?;

    let lse = alloc
        .alloc(rows * model.hf.num_attention_heads as usize * 4)?;


    // The guard-owned attention values, discovered from the lowering as
    // the smokes discovered them.
    let (q_pin, o_off) =
        attention_pins(dep, lowered, dplan, states_decode_dispatch)?;

    struct LiveResolver<'a> {
        model: &'a LoadedModel,
        named: &'a std::collections::BTreeMap<ValueId, crate::gpu::device::DeviceBuffer>,
    }
    impl Resolver for LiveResolver<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            self.model.weight(name)
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // The family's attention scalars: gemma-4 runs sm_scale 1.0 (the
    // q/k norms carry the scaling), per-layer windows (sliding at
    // `sliding_window`, full unbounded), and needs the HOST CSR mirrors
    // for its planless prefill.
    // PER LAYER in the value; the binder wants one scalar, and every
    // family that varies it varies it by layer KIND rather than by
    // layer, so the first is the stack's.
    let sm_scale = dep.attention.first().map_or(1.0, |a| a.sm_scale);
    let window_by_layer = dep.windows();
    // THE LINE THE NORTH STAR QUOTES, twice over.
    //
    // It was `let is_gemma4 = family.planless_prefill();` — wrapping a
    // family name in a virtual predicate and then recovering the name
    // at the call site, which means the axis was the family all along.
    //
    // The value fixed the read and the NAME still said gemma, which
    // `tests/no_family_names.rs` caught. That is the guard doing its
    // job on something a compiler cannot see: the code was correct and
    // the word was wrong, and a wrong word is how the next reader
    // learns to branch on a family again.
    let planless = dep.prefill == model::deployment::PrefillStyle::Planless;
    let attn = AttnCtx {
        decode_plan,
        decode_plan_full,
        prefill_plan,
        workspace,
        prefill_workspace,
        layers,
        q_out: q_pin
            .and_then(|v| named_bufs.get(&v).map(|b| b.as_ptr()))
            .unwrap_or(core::ptr::null_mut()),
        score_out: d_scores.cast(),
        folded_out: d_folded.cast(),
        score_indptr_d: d_score_indptr.cast(),
        mask_d: d_mask,
        mask_indptr_d: d_mask_indptr,
        o_out: match o_off {
            Some(off) => unsafe { arena_ptr.cast::<u8>().add(off) }.cast(),
            // No stated slot: the driver's own landing buffer, sized to
            // the fire's attention output and pooled like the rest so a
            // capture that baked its address keeps addressing something.
            None => d_attn_out,
        },
        kv_page_indices_d: d_kv_indices.cast(),
        kv_page_indptr_d: d_kv_indptr.cast(),
        kv_last_page_lens_d: d_kv_lens.cast(),
        qo_indptr_d: d_qo.cast(),
        qo_indptr_h: if planless { qo_indptr.as_ptr() } else { core::ptr::null() },
        kv_page_indptr_h: if planless { kv_indptr.as_ptr() } else { core::ptr::null() },
        num_requests: requests as i32,
        num_pages_in_batch: kv_indices.len() as i32,
        first_token: 0,
        w_page_d: d_w_page.cast(),
        w_off_d: d_w_off.cast(),
        row_valid_d: d_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: window_by_layer,
        logits_soft_cap: 0.0,
        sm_scale,
        score_window: state.boot.attn_score_window,
    };

    // ── The adapter, if any request carries one ──
    //
    // Every piece is in place now: `read_lora_sink` resolves the plan,
    // `lane_for_instance` resolves the addresses (which needed
    // `ensure_sessions` to have run), `lora_pins` names the q and v the
    // correction writes, and `llama_like_lora_stage` builds the state.
    //
    // ONE LANE PER INSTANCE, and the token span is the request's own —
    // `qo_indptr[r]..qo_indptr[r+1]`, which is what makes an adapter
    // apply to the rows that asked for it and no others.
    let lora_lanes: Vec<crate::gpu::fire::lora::LoraLaneView> =
        slice_of(frame.instance_ids.ptr, frame.instance_ids.len)
            .iter()
            .enumerate()
            .filter_map(|(r, &iid)| {
                let start = *qo_indptr.get(r)?;
                let end = *qo_indptr.get(r + 1)?;
                crate::gpu::fire::lora::lane_for_instance(
                    &state.ptir_programs,
                    &state.ptir_sessions,
                    &state.instances,
                    iid,
                    start,
                    end.saturating_sub(start),
                    stream.as_ref(),
                )
            })
            .collect();
    // THE ROWS THE STAGING READS, from the correction's own operand
    // join — the same read `attention_pins` makes, and false under
    // `Union` if it were positional.
    //
    // `q` and `v` are the launch's own args; `x` is its FOREIGN operand
    // (`LaunchSpec::aux[0]`), which is what makes the projection input
    // resolvable at all. The statement does not carry it, so finding it
    // any other way would have meant knowing which named value the
    // family's norm placement produces — and the lowering already wrote
    // it down.
    // THE SCRATCH IS RESOLVED FIRST, outside the closure. Same wall the
    // phase extraction hit: `named_bufs` is a shared borrow of
    // `state.fire_arrays` and growing the pool is a unique one, so the
    // two cannot be live together — and a closure capturing `state`
    // makes them so.
    let lora_gate = if lora_lanes.is_empty() {
        core::ptr::null_mut()
    } else {
        state
            .fire_arrays
            .attn_out(alloc, rows * model.hf.intermediate_size.max(1) as usize * 2)
            .unwrap_or(core::ptr::null_mut())
    };
    let named_bufs = &state.fire_arrays.named;
    let lora_arena = &mut state.lora_arena;
    let lora_state = (!lora_lanes.is_empty())
        .then(|| lora_pins(lowered, dplan))
        .flatten()
        .and_then(|pins| {
            let ptr = |v: model_compiler::trace::ValueId| {
                named_bufs.get(&v).map(crate::gpu::device::DeviceBuffer::as_ptr)
            };
            let (q, v, x) = (ptr(pins.q)?, ptr(pins.v)?, ptr(pins.x)?);
            // The xAᵀ scratch, from the driver's own pool — it is not a
            // value any text states, and it is sized by the widest
            // adapter in the batch rather than by the fire.
            let gate = lora_gate;
            if gate.is_null() {
                return None;
            }
            let table = crate::gpu::fire::lora::LoraTable { lanes: &lora_lanes };
            let mut ops = crate::gpu::fire::lora::LiveLoraOps::new(raw_stream);
            let post = dep.norm == model::deployment::NormPlacement::Post;
            let stage_rows = crate::gpu::fire::lora::LoraStageRows {
                // UNDER POST-NORM the projection input is the residual
                // stream and under PRE it is the normed value; the
                // staging picks with `post_norm`, and both slots name
                // the same buffer here because the lowering resolved
                // whichever one this text states.
                y: x.cast_const(),
                norm_x: x.cast_const(),
                q,
                v,
                gate,
            };
            let (fingerprint, staged) = crate::gpu::fire::lora::stage_qkv_adapters(
                &mut ops,
                lora_arena,
                Some(&table),
                model.hf.num_hidden_layers,
                i32::try_from(rows).unwrap_or(0),
                model.hf.hidden_size,
                model.hf.num_attention_heads,
                model.hf.num_key_value_heads,
                model.hf.intermediate_size,
                i32::try_from(state.tp_size).unwrap_or(1),
                post,
                &stage_rows,
                false,
            )
            .ok()?;
            let _ = fingerprint;
            staged.map(|s| (s, gate))
        });

    // ONE HANDLE FOR THE DRIVER, its stream rebound per fire. See
    // `Shell::cublas`: creating and destroying one per fire cost 3.2 ms.
    let mut cublas_ops = crate::gpu::device::cublas::LiveCublas;
    if state.cublas.is_none() {
        state.cublas = Some(
            crate::gpu::device::cublas::CublasHandle::create(&mut cublas_ops, raw_stream)?,
        );
    }
    let cublas = state.cublas.as_mut().expect("just ensured");
    cublas
        .set_stream(&mut cublas_ops, raw_stream)?;
    // The family's per-layer tables and named constants — the C++
    // parse-time vectors (`per_layer_rope_theta`, `rotary_of`) and the
    // prologue's `scale.*` values plus the load-read layer scalars. A
    // family whose rope is one theta and whose epilogue caps nothing
    // answers with empties.
    // OFF THE VALUE, and the empties are the value's too: a stack whose
    // rope is one theta answers with an empty table, because the binder
    // checks emptiness and a table of identical values is one it would
    // walk for nothing. `Deployment` stores per layer and folds here.
    let theta_by_layer = dep.theta_by_layer();
    let rotary_by_layer = dep.rotary_by_layer();
    let softcap = dep.logit_softcap;
    let ple_dim = dep.ple_dim;
    let scales = dep.scales.clone();
    // THE PEEL WINDOW, and this is where layer 3 stops being vocabulary.
    //
    // It used to be `set(0, rows)` on every fire — "start 0, count ALL",
    // which is the word for NO SPLIT. Nothing ever computed a boundary, so
    // the per-row polymorphism the `_devwin` kernels exist for had never
    // once executed (`.wiki/driver/graph.md` §2, §5 ③).
    //
    // Worse than unused: WRONG whenever a peel did lower. `lower::split_at`
    // computes the real boundary from the fire's rows and gives the tail
    // region the rectangle `[split, N)` — but a `_devwin` launch ignores
    // `bound.rows.start` by contract and reads this word instead. Saying
    // "all rows" therefore ran the tail's program over the PREFIX rows
    // too, silently, on every peeled fire.
    //
    // The window is read off the lowering rather than re-derived, because
    // the lowering already answered: the tail rectangle IS the window its
    // launches want, and two derivations of one split is how they drift.
    if state.peel_win.is_none() {
        state.peel_win = Some(
            crate::gpu::device::PeelWindowWord::new(alloc)?,
        );
    }
    let peel_win = state.peel_win.as_mut().expect("just ensured");
    let (peel_start, peel_count) = lowered
        .launches
        .iter()
        .find(|l| l.peel.is_some_and(|p| p.tail))
        .map_or_else(
            // No peel lowered: the whole fire is one region, which is what
            // an unpeeled fire means.
            || (0, u32::try_from(rows).unwrap_or(0)),
            |l| (l.rows.start, l.rows.end.saturating_sub(l.rows.start)),
        );
    peel_win.set(peel_start, peel_count);
    peel_win.upload(stream.as_ref())?;
    let peel_window_ptr = peel_win.device_ptr();

    let ctx = DispatchCtx {
        sampling_indices: d_sampled.cast::<i32>(),
        sampled_rows: i32::try_from(sampled_rows.len()).unwrap_or(0),
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        eps: model.hf.rms_norm_eps,
        rope_theta: model.hf.rope_theta,
        rope_theta_by_layer: theta_by_layer,
        rotary_by_layer,
        head_dim: model.hf.head_dim,
        num_q_heads: model.hf.num_attention_heads,
        num_kv_heads: model.hf.num_key_value_heads,
        vocab: model.hf.vocab_size,
        gate_second: false,
        rope_interleaved: false,
        token_ids: d_ids.cast_mut().cast(),
        positions: d_pos.cast_mut().cast(),
        final_logit_softcap: softcap,
        ple_dim,
        scales,
        moe_norm_topk: false,
        moe_routed_scaling: 1.0,
        yarn: [0.0; 4],
        yarn_original_max: 0,
        glu_limit: 0.0,
        glu_alpha: 0.0,
        situ_beta: 0.0,
        situ_linear_beta: 0.0,
        wna16_group_size: 0,
        altup_streams: 0,
        altup_active: 0,
        altup_std_mult_by_layer: Vec::new(),
        // THE STAGED ADAPTER, or `None` for the fires that carry none —
        // which is every fire until a program states a `lora` sink in
        // its prologue.
        //
        // `None` is not a refusal here: the executor's arm returns
        // `Ok(())` for it, and that no-op is load-bearing for union
        // captures, because under `GuardMode::Union` every arm lowers
        // and the predicate decides at replay. The arm has to be
        // issuable with nothing to correct.
        lora: lora_state.as_ref().map(|(s, scratch)| (std::ptr::from_ref(s), *scratch)),
        // The fire's peel window, published so a `_devwin` statement in a
        // tail region can early-out per lane. The prefix is the rows that
        // do NOT carry the axis's mark, so the tail begins where the
        // marked suffix does; with no marked rows there is no split and
        // the word says the whole fire.
        peel_window: peel_window_ptr,
        rows_total: i32::try_from(rows).unwrap_or(0),
    };

    lap("bind");
    let mut resolver = LiveResolver { model, named: &named_bufs };
    let regions = AttnRegions::whole(Some(&attn));
    // The last use of `alloc` is above, so the shared borrow is dead and the
    // capture can take the same allocator mutably — which is the point: a
    // capture has to be opened on the allocator that owns what the fire
    // frees, or the frees are not deferred.
    if state.preds.is_none() {
        state.preds = crate::gpu::device::PredicateWord::new(
            state.fire_alloc.as_ref().expect("the fire allocator exists"),
        )
        .ok();
    }
    let (capture_alloc, capture_preds) = match (&mut state.fire_alloc, &mut state.preds) {
        (Some(a), Some(p)) => (a, p),
        _ => return Err(PIE_STATUS_EXHAUSTED),
    };
    lap("ctx");
    let result = if union {
        capture_or_replay(
            &mut state.supergraph,
            state.fire_arrays.epoch,
            u64::from(model.hf.num_hidden_layers.unsigned_abs()),
            &plan, &fire_rows, &lowered, &dplan, exec_frame, &mut resolver, &ctx,
            regions, gdn_ctx.as_ref(), capture_alloc, capture_preds, stream.as_ref(),
            requests, rows, class,
        )
    } else {
        run(&lowered, &dplan, exec_frame, &mut resolver, &ctx, regions, gdn_ctx.as_ref())
    };
    lap("run");
    // A step that owes nothing SYNCHRONIZES, because the next step in the
    // frame reads what this one wrote. A step that owes the frame's
    // completion does not: its debt rides a stream-ordered callback and
    // this call returns with the work still queued, which is the whole
    // point.
    let sync = if owes.is_some() && state.runahead {
        Ok(())
    } else {
        stream.as_ref().synchronize()
    };
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
    // THE FRAME'S DEBT, built before the delivery below because it is
    // owed whether or not this fire has logits to deliver. Paying it
    // inside the delivery block meant a fire with no readout channel
    // never published its terminal cells and never notified — the
    // runtime waited forever on a frame the driver had finished.
    let mut debt = owes.map(|(completion, cells)| FireDebt {
        staging: None,
        readouts: Vec::new(),
        vocab: usize::try_from(model.hf.vocab_size).unwrap_or(0),
        cells,
        completion,
        notify: state.notify,
        notify_ctx: state.notify_ctx,
    });

    // ── Sampling: the instance's PROGRAM, if it has one. ──
    //
    // Before the delivery below and not after, because a program that
    // published is a fire whose answer has already gone out — top-p,
    // top-k, temperature and argmax are its stages, and handing the
    // caller raw logits beside them would deliver twice and disagree
    // with itself.
    //
    // Everything about this degrades to the old behaviour: no program, a
    // program that declines, inputs not ready, or channels this shell
    // does not hold all return `false` and fall through to the raw
    // logits. That is what the driver did before this existed, so the
    // worst case is the status quo rather than a broken fire.
    // A FRESH borrow of the allocator, not the one from the top of the
    // fire. The capture above takes `&mut state.fire_alloc` — deliberately,
    // since a capture must be opened on the allocator that owns what the
    // fire frees — and reusing the earlier binding here would extend a
    // shared borrow across it. Re-deriving is one line and says where the
    // mutable window ended.
    let alloc = state.fire_alloc.as_ref().expect("the fire allocator exists");
    let unsampled = run_sampling_programs(
        SamplingSites {
            instances: &state.instances,
            channels: &state.channels,
            programs: &state.ptir_programs,
            control: &mut state.ptir_control,
            sessions: &mut state.ptir_sessions,
            disk: state.ptir.disk(),
            device_ordinal: state.device_ordinal,
            named_bufs: &state.fire_arrays.named,
        },
        model,
        lowered,
        dplan,
        frame,
        alloc,
        stream,
        qo_indptr,
        &sampled_rows,
        rows,
    )?;
    lap("sample");

    if !unsampled.is_empty() {
        deliver_logits(
            &state.instances,
            &state.channels,
            &mut state.logits_staging,
            frame,
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

    // The debt goes last in stream order, so it runs after every
    // launch and after the D2H above.
    if let Some(d) = debt {
        let raw = Box::into_raw(Box::new(d)).cast::<std::ffi::c_void>();
        // ONE SET OF DEBTS, TWO WAYS TO PAY THEM. Gated off, this
        // thread pays after the synchronize above — which is what the
        // driver always did, and what every caller reading a result
        // on the next line still expects. Gated on, a stream-ordered
        // callback pays them and this call returns with the work
        // still queued.
        if !state.runahead {
            // The D2H above was ENQUEUED, so paying here means waiting
            // here. Without this the callback's staging is read before
            // the copy into it has landed.
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
        // AND THE SCRATCH SURVIVES THE CALL. Dropping it here would
        // `cudaFree` while the fire runs, which synchronizes the
        // device and undoes everything above. The next launch
        // reclaims it — see `InFlight`.
        lap("debt");
        let done = crate::gpu::device::Event::new()?;
        stream.as_ref().record(&done)?;
        state.in_flight.push_back(InFlight {
            done,
            scratch: [Some(lse), Some(d_valid), _slot_ids_buf]
                .into_iter()
                .flatten()
                .collect(),
            closed_channels: Vec::new(),
        });
    }
    lap("tail");
    Ok(())
}

