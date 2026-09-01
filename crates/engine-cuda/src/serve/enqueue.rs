//! **THE STREAM HALF OF ONE STEP** — the prologue, the fresh-slot memsets,
//! the staging write, the arena and pool tables, the schedules, the walk, and
//! the epilogue that no longer waits for itself.
//!
//! A child module of [`serve`](super), and `prepare`'s twin: that phase
//! touches no stream and this one is nothing but stream. `serve.rs` keeps the
//! wrapper — [`FrameShell::enqueue`](engine::frame::Shell::enqueue) is next
//! door in [`prepare`](super::prepare), where the three verbs of the seam sit
//! together — and the twelve hundred lines it delegates to are here.
//!
//! What travelled with `Shell::enqueue_on` is what only it reads:
//!
//! ```text
//! GuestBatch        one boundary's guest fires, left on the stream
//! AirborneFires     the fires of one boundary, enqueued and unsettled
//! reap_guest_fires  and read back at the next frame, usually for free
//! committed_or      the sentence for a pass that did not commit
//! ```
//!
//! `Shell::reap_guests` — the door a caller outside the fire path collects a
//! deferred boundary through — stays in `serve.rs` with the other doors, and
//! calls [`reap_guest_fires`] like every other caller does.

use std::cell::Cell;

use kernels_cuda::attn::plan::Shape;
use model_exec::fire::walk;
use model_ir::Dtype;

use engine::fire::Boundary;

use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::{MTP_SEAM, SCORES_SEAM};
use crate::program::launch::INTRINSIC_STORAGE_RAW_BF16;
use crate::program::{Fired, Plane as ProgramPlane};
use crate::record;
use crate::run::{
    CacheGeometry, CachePlanning, Ceilings, FireBindings, FireTables, RsMove, RsSeat, Run,
    ScheduleSeat,
};
use crate::window::{At, Cursor, Lanes};

use super::{Prepared, Readback, Shell};

/// **ONE BOUNDARY'S GUEST FIRES, LEFT ON THE STREAM.**
///
/// What a deferred settlement has to carry is small, and deliberately so:
/// which instances owe a verdict, and two ways of asking whether the verdict
/// is readable yet.
#[derive(Debug)]
pub(super) struct GuestBatch {
    /// `(lane, instance)` for every launch owing a settlement, in launch
    /// order. The lane is carried only to name the fault; by the time this is
    /// read the `Prepared` that described the frame is gone, which is why the
    /// instance id travels rather than a borrow of its `Attached`.
    launched: Vec<(usize, u64)>,
    /// **The step whose settlement callback proves this batch landed.** Read
    /// FIRST, because it costs nothing: `Airborne` is two atomics on the host
    /// and a `true` here means the reap takes no CUDA call at all. The event
    /// is the fallback for the frame that has not called back yet.
    seq: u64,
}

impl Shell {
    /// `enqueue`'s body — see the wrapper for what it does not do.
    ///
    /// # Errors
    ///
    /// The shell's fault, for a launch the backend refused at enqueue time.
    pub(super) fn enqueue_on(
        &mut self,
        p: &mut Prepared<'_>,
        slot: &crate::inputs::SlotGuard,
    ) -> Result<(u32, Option<Readback>)> {
        // **THE STEP THIS FIRE WILL SETTLE AT**, stamped onto the graph cache
        // before anything launches. Every exec launched below carries it, and
        // it is what eviction compares against the settled count — the
        // arithmetic that replaced "every fire ends synchronized". Read rather
        // than consumed: `settle` is what takes the number, one host statement
        // later with nothing in between.
        let seq = self.airborne.next_seq();
        self.cache.at_step(seq);
        let Shell {
            device,
            trace,
            compiled,
            weights,
            arena,
            pools,
            inputs,
            facts,
            graphs,
            pad,
            arming,
            cache,
            // NAMED, NOT ABSORBED BY THE `..`: the guest-program plane is
            // touched at the fire's BOUNDARIES and nowhere between them, and
            // spelling the field out is what makes that a statement rather
            // than an omission.
            programs,
            exports,
            held,
            // NAMED FOR THE SAME REASON: the recurrent plane is touched at
            // exactly two instants — the predicate, before the walk, and the
            // scatter/gather, inside it — and spelling the two fields out is
            // what makes that a statement rather than an omission.
            buffers,
            predicate,
            // NAMED FOR THE THIRD TIME AND FOR THE SAME REASON: the readout's
            // row-pointer tables are staged at ONE instant — the epilogue
            // binding, below — and by one writer.
            readout_rows,
            budget,
            // NAMED FOR THE FOURTH TIME AND FOR THE SAME REASON: the deferred
            // epilogue batch is parked at ONE instant and collected at two,
            // all three of them in this function.
            owed,
            guest_landed,
            airborne,
            // NAMED FOR THE FIFTH TIME AND FOR THE SAME REASON: the score
            // slab is touched at exactly two instants — the seat handed to
            // the walk, and the epilogue binding that points a guest at it —
            // and both of them are in this function.
            scores,
            // NAMED FOR THE SIXTH TIME AND FOR THE SAME REASON: the per-region
            // "this region moves its own base" slice is read at exactly two
            // instants — the bodies gate in `prepare`, and the `Run` built
            // below — and this is the second of them (the bodies design's
            // chunk 2b-ii).
            shifted,
            // NAMED FOR THE SEVENTH TIME AND FOR THE SAME REASON: which
            // classes a decode lane lands in is read at exactly two instants
            // — the ladder `prepare` builds, and the `record::Fire` below,
            // which hands it on so `fire_body` re-keys with the arguments
            // `prepare` keyed with (`record::Ladder::rung`).
            decoding,
            ..
        } = self;
        let graphs = *graphs;
        let pad = *pad;
        let arming = *arming;

        // ── The prologue. Channel reads, state, token prep — never the
        //    readout, which does not exist yet.
        //
        //    **THE VERDICTS ARE COLLECTED AND THE PREDICATE IS WRITTEN
        //    BEFORE ANY OF THEM IS JUDGED** (alto design §6's change (a)).
        //    `channel::pull_validate` — inside each pass — is what seeds the
        //    commit word, and the recurrent fold has to be predicated on that
        //    same word, so the mask kernel stands between the pull and the
        //    forward and not on the far side of a refusal. This shell's own
        //    policy also ABORTS a fire whose prologue did not commit
        //    (`committed_or`), so today the two agree twice over; the order
        //    below is what keeps the fold's predicate true on its own terms
        //    the day the policy softens, which is what article 3 asks of it.
        //    **AND THE BOUNDARY TAKES ONE WAIT, NOT ONE PER ATTACHMENT**
        //    (alto §14 exception #1, closed). Every prologue is ENQUEUED
        //    here; the verdicts are read below, after one synchronize for the
        //    whole boundary. See [`Boundary`]'s own note and
        //    [`Session::launch`](crate::program::Session::launch).
        let mut verdicts: Vec<(usize, Fired)> = Vec::new();
        let mut prologues = AirborneFires::default();
        // The same obligation the epilogue loop below has, and paid only when
        // there is a prologue to stage: a decode guest attaches its sampler at
        // the epilogue and nothing here, so the common frame does not reach
        // this at all.
        if p.attachments.iter().any(|a| a.at == Boundary::Prologue) {
            reap_guest_fires(programs, owed, airborne, guest_landed)?;
        }
        for (at, attached) in p.attachments.iter().enumerate() {
            if attached.at != Boundary::Prologue {
                continue;
            }
            if let Some(fired) = prologues.stage(device, programs, at, attached.instance)? {
                verdicts.push((at, fired));
            }
        }
        // **THE PROLOGUE'S FIRES LEAVE THE GROUND HERE**, before the fold
        // predicate is written, because the predicate is each lane's own
        // commit word and `channel::pull_validate` is what seeds it. Staging
        // decided nothing; this is the launch.
        prologues.fly(device, programs)?;

        // ── The fold predicate, as device data (design §6, §12 finding 4).
        //
        //    One byte per lane: the lane's own pass commit word where it has
        //    a prologue, the standing ONE where it has none — an unattached
        //    lane folds, which is what keeps the plain path plain — and the
        //    standing ZERO where the lane's verb is a buffered scatter, which
        //    is the verb's own predicate riding the same kernel.
        let lane_count = p.composition.lane_count();
        if p.rs.predicated || p.rs.truncates {
            let mut commits: Vec<u64> = vec![predicate.always(); lane_count as usize];
            for (at, verb) in p.rs.moves.iter().enumerate() {
                if matches!(verb, RsMove::Scatter { fold: 0, .. }) {
                    commits[at] = predicate.never();
                }
            }
            for attached in p.attachments.iter().filter(|a| a.at == Boundary::Prologue) {
                let Some(&lane) = p.rs.order.get(attached.lane as usize) else {
                    continue;
                };
                let Some(session) = programs.instance(attached.instance) else {
                    continue;
                };
                if let Some(slot) = commits.get_mut(lane as usize) {
                    *slot = session.commit_word();
                }
            }
            predicate.write(device.stream(), &commits, &p.rs.lens)?;
            if p.rs.predicated {
                kernels_cuda::channel::mask_from_commit(
                    device.ctx(),
                    predicate.commits(),
                    predicate.indptr(),
                    predicate.mask(lane_count).ptr,
                    lane_count,
                )
                .map_err(Fault::from)?;
            }
        }

        // ── THE PROLOGUE BOUNDARY'S ONE WAIT, and then every verdict.
        //
        //    It stands HERE, before the forward, because that is what it
        //    always meant: a prologue that did not commit is a fire nobody can
        //    replay, and `committed_or` refuses to build a forward on top of
        //    one. What changed is the count — one synchronize for the whole
        //    boundary rather than one per attachment — and a boundary with
        //    nothing enqueued takes none at all, which is the common shape:
        //    a decode guest attaches a sampler at the epilogue and nothing
        //    here.
        prologues.settle_into(device, programs, &mut verdicts)?;
        for (at, fired) in verdicts {
            committed_or(fired, p.attachments[at].instance, "prologue")?;
        }

        // ── The fresh slots' recurrent banks, zeroed. `prepare` decided
        //    which; this is the memset, and it stands where the lane loop
        //    used to do it — after the prologue, in front of the staging.
        //
        //    **ON THE STREAM** (alto F2b). It was `cudaMemset`, which is
        //    synchronous — so the first fire of every sequence drained
        //    everything airborne, a host wait between two waves that article 2
        //    forbids and that F1's own end-of-fire sync hid. Ordered on the
        //    fire's stream it means what it always meant: zeroed before the
        //    launches that read the bank, and free.
        for slot in &p.fresh {
            pools.clear_on(device.stream(), *slot)?;
        }

        let rows = p.composition.rows();

        // 5. Commit the slot `prepare` wrote onto the fire's stream, in front
        //    of the launches that read it.
        //
        //    **THIS IS THE FIRST STREAM TOUCH, AND THEREFORE THE PHASE
        //    BOUNDARY**, and F2b is where the two halves finally are two.
        //    Design §4 splits it — `staging.write(slot, ..)` on the host in
        //    `prepare`, `staging.commit(s, desc)` on the stream here — because
        //    a ring of staging slots is what lets W+1's descriptor be WRITTEN
        //    while W's is still being READ. The device destination stays one
        //    pointer-stable region (article 7: a captured graph reads baked
        //    addresses), and what keeps two in-flight frames from colliding on
        //    it is not a second buffer but stream order: W+1's copies are
        //    enqueued behind W's kernels on the one compute stream.
        let handles = inputs.commit(device.stream(), slot, &p.lengths)?;
        p.windows.bind(handles.windows);
        // And the live-rows seat beside them — `None` for a fire that staged
        // no words, which is every fire today, and then `Windows::live_at`
        // answers the disarmed `0` the whole plane is built to be identical
        // under.
        p.windows.bind_live(handles.live_rows);
        // And the absolute reading of the qo boundaries beside it — `None` for
        // a fire that staged none, and then `Windows::qo_absolute` answers
        // `None` and every ragged view takes the rebased vector it always
        // took.
        p.windows.bind_qo_absolute(handles.qo_absolute);

        // ── **THE PATCH BYTES, WRITTEN INSIDE THE ENQUEUE** (multimodal
        //    §5.4). Three copies onto the same compute stream, in front of
        //    the launches that read them, from pageable `Vec`s the prepare
        //    pass made — which is what lets them ride no staging ring and
        //    cost a text-only load nothing. `None` for a fire no lane
        //    submitted an image into, and then not one of the three copies
        //    happens.
        let patches = if p.patch_payload.is_empty() {
            None
        } else {
            Some(inputs.stage_patches(
                device.stream(),
                &p.patch_payload,
                &p.patch_segments,
                &p.patch_routes,
                &p.patch_positions,
                &p.patch_embed_rows,
                &p.patch_embed_weights,
            )?)
        };

        // ── **AND THE TRUNK'S ROTATION STREAM, ONE COPY BESIDE THEM.** Same
        //    stream, same instant, same argument for riding no ring; empty
        //    for a plan that does not declare it, and then no copy at all.
        let mrope = if p.mrope_positions.is_empty() {
            None
        } else {
            Some(inputs.stage_mrope_positions(device.stream(), &p.mrope_positions)?)
        };

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        // **BOTH AXES' COUNTS** (multimodal §5.1): a tower rectangle is
        // `Dim::Patches`-rowed and resolves through this same table, so a call
        // that stated only the token pair would size every one of them at zero
        // — which does not fault, it computes, and the failure arrives inside
        // a GEMM whose destination has no rows. The composition holds both
        // pairs because it seriated both axes.
        // **AND THE TOKEN COLUMN IS CARVED AT THE BUCKET FOR A BODIED FIRE**
        // (the grid-at-ceiling wave). `Run::cut` hands a launch in a region
        // that owns a retirement the KEY's row ceiling rather than this
        // window's live span, and its last line clamps that extent to the
        // rectangle the value RESOLVES to — so a column cut at the live rows
        // would clamp the ceiling straight back down to them and the grids
        // would follow the fire after all.
        //
        // **A COLUMN HEIGHT AND NOT A ROW COUNT, WHICH IS WHY THIS IS A
        // SECOND NUMBER RATHER THAN A WIDER `rows`.** The arena's offsets are
        // static — `model_exec::store::arena::rect` moves only the rectangle's
        // HEIGHT with this argument, and the allocation behind it is
        // `max_tokens` tall on every load (P0 promises every bucket sits under
        // that) — so raising it names bytes the carve already holds and no
        // value's neighbour moves. `rows` beside it stays the fire's own and
        // goes on being the fire's own everywhere it is read: the pool seats
        // below take it, and what they mean by it is how many rows the page
        // geometry describes, which padding does not change.
        //
        // **AND THE PATCH COLUMN IS CARVED AT THE PATCH BUCKET, ON EXACTLY
        // THE SAME ARGUMENT** (the multi-unit bodies wave). This used to read
        // "the PATCH axis takes no such ceiling: a body carries one bucket and
        // a multi-unit artifact is refused from the path outright" — and that
        // was true of a key with one lattice point in it. A `record::BodyKey`
        // now carries a `record::AxisKey` for the tower unit, so a tower
        // region of a bodied fire is gridded at the PATCH lattice point
        // (`Run::carve_rows` off `run::Ceilings::pad_on`) exactly as a trunk region is
        // gridded at the token one — and a column cut at this fire's live
        // patch rows would clamp that grid straight back down to them, which
        // is the same failure the token half describes and the reason this
        // second number exists at all.
        //
        // **AND THE HOLE IT CLOSES WAS A REPLAY-OVERRUN AND NOT A CLAMP.**
        // A body captured for a key whose patch bucket is 128 is replayed by
        // every fire of that key, including one bringing 64 patch rows; the
        // capture's tower launches are gridded at 128 and the seat retires the
        // rest. With the column cut at the CAPTURING fire's patch rows, the
        // rectangle the replay's baked pointers address would be shorter than
        // the grid that addresses it whenever a later fire of the key carved
        // smaller — which does not fault, it computes, off the next value's
        // bytes. Carved at the bucket, every fire of the key resolves the same
        // column height, which is what "a ceiling is a function of the key"
        // has to mean on this axis too.
        //
        // A HEIGHT AND NOT A COUNT, for the token clause's reason: the arena's
        // offsets are static and the allocation behind them is
        // `PatchLadder::max_patches` tall, which P0 promises sits above every
        // patch rung. `images` beside it stays the fire's own — it is how many
        // images the geometry vectors describe, and padding does not change
        // that.
        //
        // `p.bodied` alone, because it implies the pad: `prepare`'s gate takes
        // that clause once, where it is a sentence about the deployment.
        let carve_rows = if p.bodied {
            u64::from(p.composition.bucket()).max(u64::from(rows))
        } else {
            u64::from(rows)
        };
        let carve_patches = if p.bodied {
            u64::from(p.composition.patch_bucket()).max(u64::from(p.composition.patch_rows()))
        } else {
            u64::from(p.composition.patch_rows())
        };
        let slots = arena.slots(
            &compiled.arena,
            model_compiler::FireRows {
                tokens: carve_rows,
                lanes: u64::from(lane_count),
                patches: carve_patches,
                images: u64::from(p.composition.images()),
            },
        );
        let caches = pools.table(
            &inputs
                .seats(&handles, p.pages, rows, lane_count)
                // **THE THREE RS SEATS, AND THE PLAIN FIRE BINDS NONE OF
                // THEM.** `Tensor::ABSENT` is the null pointer every optional
                // seat in `attn/ssm.cuh` already tests for, so a fire that
                // predicates nothing and truncates nothing hands the launches
                // exactly the arguments they took before F3.
                .rs(
                    p.rs.write_state,
                    if p.rs.predicated {
                        predicate.mask(lane_count)
                    } else {
                        kernels_cuda::Tensor::ABSENT
                    },
                    if p.rs.truncates {
                        predicate.commit_len(lane_count)
                    } else {
                        kernels_cuda::Tensor::ABSENT
                    },
                )
                // **THE SAME VECTOR, READ FROM THE OTHER END** (wave F3b's
                // 2R split): a row's fold boundary is one number, the head
                // launch stops at it and the tail launch starts at it. A
                // fire no row splits binds nothing and makes one launch.
                .splitting(if p.rs.splits {
                    predicate.commit_len(lane_count)
                } else {
                    kernels_cuda::Tensor::ABSENT
                }),
        )?;
        let paging = pools.paging();

        // 7. The geometry seats, and their host twins. THE DUALITY: the IR
        //    names `kv_indptr` as a device input and the plan builders are
        //    host functions that walk its CONTENTS, so the same vector is
        //    bound twice — once as a handle for the launches, once as a
        //    `Vec<i32>` for `plan_decode`/`plan_prefill`.
        let mut geometry = Vec::with_capacity(p.geometries.len());
        for (space, host) in p.geometries.iter().enumerate() {
            let seat = handles.spaces[space];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(handles.row_valid),
                request_of_token: None,
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
                // The custom-mask slab, bound whole: its entries are bits and
                // `Run::cut` excludes it for the same reason it excludes the
                // page-id list. Every space gets the same handle, because
                // every space of a v1 plan is paged over the same lanes with
                // the same extents — the day two spaces hold different
                // readable extents, this reads `staged` per space.
                mask: handles.mask,
                planning: Some(CachePlanning {
                    kv_indptr: host.indptr.clone(),
                    kv_len: host.kv_len.clone(),
                }),
            });
        }

        // 7b. The schedule seats. One per PLAN VALUE, because a schedule is
        //    carved for ONE reading — head width, query heads, window — and a
        //    family may carve two out of one page-id space (gemma's sliding
        //    beside its global). The FIRE's lanes go in; `Run::planning`
        //    narrows `num_requests` to the asking node's window, which is the
        //    count a schedule is actually built at.
        //
        //    ONE SEAT PER (RUN, PLAN VALUE), because a region P4 could not
        //    seat builds one schedule per interval of its window and all of
        //    them are alive between the prepare pass and the capture pass.
        //    `windows.max_runs()` is 1 for every artifact P4 seated whole, and
        //    this is then the flat table it always was.
        let runs = p.windows.max_runs();
        let inputs = &*inputs;
        let schedules: Vec<Option<ScheduleSeat>> = (0..runs)
            .flat_map(|run| {
                facts.plans.iter().enumerate().map(move |(at, seat)| {
                    let seat = (*seat)?;
                    Some(ScheduleSeat {
                        shape: Shape {
                            num_requests: lane_count,
                            // The FIRE's lanes go in and `Run::planning`
                            // narrows both: the request count to the asking
                            // node's window, and this origin to the window's
                            // own `lane_offset` — but only where the pointers
                            // beside it are the plane's, which is where a
                            // launch is handed lane tables it did not have
                            // sliced for it.
                            lane_offset: 0,
                                                        num_q_heads: seat.reading.q_heads,
                            num_kv_heads: seat.reading.kv_heads,
                            head_dim: seat.reading.head_dim,
                            page_size: paging.page_size,
                            hnd_layout: false,
                        },
                        window: seat.reading.window,
                        workspace: inputs.grant(at as u32, run).unwrap_or_else(|| {
                            panic!(
                                "plan value {at} carries a reading but no grant for \
                                 run {run}; `Inputs::reserve` carves one per probed \
                                 plan per run the artifact can split into"
                            )
                        }),
                    })
                })
            })
            .collect();

        // 8. The walk. The prepare regions build and stage the attention
        //    schedules — one per window, so a mixed fire builds both — and
        //    the capture regions enqueue. The sink records nothing, as
        //    `EagerSink` would: in an eager fire the walk's own control flow
        //    IS the structure. What it does carry is the region number, which
        //    is how a `Run` knows whose window it is resolving in.
        let bindings = FireBindings {
            tokens: handles.tokens,
            positions: handles.positions,
            adapter_routes: handles.adapter_routes,
            patches: patches.as_ref().map(|seats| seats.patches),
            patch_segments: patches.as_ref().map(|seats| seats.segments),
            patch_routes: patches.as_ref().map(|seats| seats.routes),
            patch_positions: patches.as_ref().map(|seats| seats.positions),
            patch_embed_rows: patches.as_ref().and_then(|seats| seats.embed_rows),
            patch_embed_weights: patches.as_ref().and_then(|seats| seats.embed_weights),
            mrope_positions: mrope,
            geometry,
            schedules,
            plan_values: facts.plans.len(),
            tables: FireTables {
                // Fire-wide going in and window-sliced coming out
                // (`Run::mask_indptr`): the plan-building arm takes its own
                // window's lanes, and the byte offsets inside stay absolute
                // because the slab they point into is not sliced.
                mask_indptr: handles.mask_indptr,
                pool_state: None,
            },
            // **A SEAT ONLY WHEN SOMEBODY ASKED** (attn-score §4's
            // zero-cost-when-off, gate S-3). `None` is what makes the capture
            // arm's observation cost a non-capturing fire nothing at all —
            // not a disabled node, not an empty launch, not a predicated
            // store: `Run::capture_scores` returns before it reaches a
            // stream, so the fire this shell fires is the fire it always
            // fired, launch for launch.
            scores: scores
                .as_ref()
                .filter(|_| p.lanes.iter().any(|seated| seated.captures_scores))
                .map(crate::scores::Scores::seat),
            device: device.device(),
            toggles: device.toggles(),
            // The shell's policy word going in: under a mode that records,
            // the builders carve graph-shaped, padded schedules, so that the
            // numbers a capture bakes into its launches are a function of the
            // fire's SHAPE and not of its contents.
            capture: graphs.shaped(),
        };
        // The one piece of state between the two halves of the walk: the sink
        // writes which region is running and which run of its window, the
        // `Run` reads both to know which window to resolve in. They cannot be
        // one object — `walk` takes two `&mut` — and this is the smallest
        // thing that stands between them.
        let place = At::new();
        // P6's twin of it: which STREAM the walk is on. Written by the same
        // cursor at the same instant, read by the same `Run` — one more `u32`
        // between the sink and the dispatch, and nothing else changes about
        // either.
        let stream = Cell::new(0u32);
        let side_ctx = device.side_ctx();
        let side_streams = device.side_streams();
        let forked = (!side_ctx.is_empty()).then(|| Lanes {
            side: &side_streams,
            main: device.stream(),
            events: device.events(),
            at: &stream,
        });
        // **THE CONDITIONAL BUNDLE, AND IT IS `Some` ONLY FOR AN ARTIFACT
        // THAT ASKED** (palo design §4). `Context::open_conditional` is called
        // at load and only when P3 stamped a `Lowering` on some region, so
        // this reads `None` for every SKU in the catalog but the drafting
        // ones — and a walk that never meets a conditional never looks at it.
        //
        // It carries the SAME stream cell `forked` does, which is what lets a
        // load with a conditional and no side streams write a stream number
        // the `Run` reads: there is one cell per fire, not one per bundle.
        let conditionals = device
            .conditional_ctx()
            .map(|_| crate::window::Conditionals {
                main: device.stream(),
                body: device.conditional_stream(),
                setter: device.ctx(),
                windows: &p.windows,
                at: &stream,
            });
        // **D4: THE BUCKET REACHES THE ENTRIES, AND NOTHING ELSE MOVES**
        // (`.wiki/palo/cuda-abi.md` §3, refined form). `Composition::bucket`
        // has been computed on every fire since compose was written and read
        // by nobody but the fallback menu's position lookup above. Here it
        // stops being decorative: the pair (this fire's rows, the lattice
        // point above them) rides into the walk, and the entries that hand a
        // shape to cuBLASLt — and only those, and only in a region whose
        // window is the whole fire — round their `M` up to it, so the
        // library's unpublished shape→kernel table stops being a function of
        // the batch the runtime happened to assemble.
        //
        // **HANDED TO THE `Run` AND NOT TO THE CONTEXTS**, which is what makes
        // the windowed boundary structural rather than conventional: the pad
        // is gated per REGION, by `Run::ctx`, against the window the shell
        // built from that region's mask. A pad written onto a context here
        // would still be armed when the walk stepped into a windowed region,
        // and the only thing an entry could then check is one extent against
        // another — a test a window whose rows happen to equal the fire's
        // passes. It also reaches every side stream for free, because
        // `Run::ctx` is what picks the side stream too.
        //
        // The composition is the ONE source: rows and bucket come off the
        // same `Composition` that carved the windows this walk resolves in, so
        // there is no second reading to fall out of step with. The off arm
        // hands `bucket == rows`, which is the same nothing a deployment with
        // no lattice hands.
        let armed = kernels_cuda::Pad {
            rows: p.composition.rows(),
            bucket: if pad {
                p.composition.bucket()
            } else {
                p.composition.rows()
            },
        };
        // **AND THE SECOND ROW AXIS'S PAIR** (the multi-unit bodies wave).
        // Same instant, same composition, same off-arm: a shell serving the
        // pad off hands `bucket == rows` here for the reason it hands it
        // above, and an artifact with no patch axis hands `0` and `0`, which
        // is the default pair no extent equals. Two readers — `run::Ceilings::pad_on`,
        // which is the axis a launch's `M` is rounded on and the total its
        // window's span is judged against, and `record::Fire::carve_patch_bucket`,
        // which is the ledger's twin of the same number.
        let armed_patches = kernels_cuda::Pad {
            rows: p.composition.patch_rows(),
            bucket: if pad {
                p.composition.patch_bucket()
            } else {
                p.composition.patch_rows()
            },
        };
        let mut run = Run::new(
            device.ctx(),
            &trace.values,
            &trace.nodes,
            weights.table(),
            &slots,
            &caches,
            bindings,
            &p.windows,
            &place,
        )
        .across(&side_ctx, &stream)
        // **THE SIX FACTS THE WALK CANNOT RECOMPUTE, ARMED IN ONE PIECE**
        // (`run::Ceilings`) — and one piece is the point: every ceiling on the
        // launch side is a function of the pad pair AND the admission AND the
        // ladder together (`run::Standing`), so a `Run` holding two of the three
        // would be a `Run` whose answers are about no fire. They used to be
        // three builders, and the split said the three could be armed apart,
        // which this call site has never done.
        .ceilings(Ceilings {
            // **ONE PAD PAIR PER ROW AXIS** (the multi-unit bodies wave),
            // filled in `RowAxis::ALL`'s order. Same instant, same
            // composition, same off-arm on both: a shell serving the pad off
            // hands `bucket == rows` on each, and an artifact with no patch
            // axis hands the default pair that no extent equals. What the
            // second entry arms is `run::Ceilings::pad_on` for a TOWER region
            // — the axis a launch's `M` is rounded on and the total its
            // window's span is judged against — and a region on the token
            // axis never reads it.
            pads: model_ir::PerAxis::new([armed, armed_patches]),
            // **THE LAUNCH PLANE'S HALF OF THE BODIES GATE** (chunk 2b-ii).
            // The two words the walk needs to hand a shifting region its
            // plane's base instead of its window's slice, and to arm the seat
            // that then tells it where its rows are. `p.bodied` is `prepare`'s
            // own answer — the same one that put the live-rows words into the
            // slot — so a fire that staged no seat resolves exactly the
            // pointers it always did, and the eager path is byte for byte what
            // it was.
            bodied: p.bodied,
            shifted: shifted.as_slice(),
            admits: p.admits.as_ref(),
            // **AND THE KEY'S CEILINGS** (the ceiling design's Option B): the
            // ladder `prepare` built beside the key, over the class table it
            // was built from, which is what `Run::planning` turns a window's
            // span into a carve with. `None` off the bodies path, where there
            // is no key and therefore no ceiling to take.
            // **AND THE SECOND UNIT'S PAIR OF THE SAME THREE OBJECTS** (the
            // multi-unit bodies wave): this fire's PATCH class table and the
            // key's patch ladder, with no lane ceiling — an image count is
            // carved at nothing today (`record::AxisCarve::lane_ceiling`).
            // `Run::planning` and `Run::carve_rows` index by the region's own
            // axis, which is the only thing that can pick: a span and the
            // table it is classified against have to come out of one
            // seriation.
            carve: p.bodied.then(|| record::Carve {
                per_axis: model_ir::PerAxis::new([
                    Some(record::AxisCarve {
                        classes: p.composition.table(model_ir::RowAxis::Tokens),
                        ladder: &p.ladder,
                        lane_ceiling: Some(p.lane_ceiling),
                    }),
                    p.patch_ladder.as_ref().map(|ladder| record::AxisCarve {
                        classes: p.composition.table(model_ir::RowAxis::Patches),
                        ladder,
                        lane_ceiling: None,
                    }),
                ]),
            }),
        });
        // The other half of the bundle above: where a conditional body's
        // launches land. The cursor writes `window::BODY` into the cell for
        // exactly the span between a `cond_begin` and its `cond_end`.
        if let Some(body) = device.conditional_ctx() {
            run = run.conditional(body, &stream);
        }
        // **THE BUFFERED PLANE, SEATED ONLY WHEN A LANE MOVES BYTES.** A fire
        // whose every lane folds hands the walk nothing, so the two dispatch
        // arms that could scatter or gather test one `Option` and return.
        if p.rs.buffered
            && let Some(pool) = buffers.as_ref()
        {
            run = run.buffered(RsSeat {
                buffers: pool,
                lanes: &p.rs.moves,
            });
        }
        // TWO MODES, ONE WALK (design §6, decision #11). Off and Shaped run
        // it whole; On splits it at the phase boundary — prepare on the open
        // stream, then the capture regions either replayed from this
        // composition's body or run and recorded into one. Which is why
        // `record::Graphs::fire_body` takes the same arguments `walk` does and
        // answers the same errors: it is not another path, it is the same one
        // at two instants.
        // **A BUFFERED FIRE IS NOT GRAPH-REPLAYABLE, AND THAT IS DESIGN §6'S
        // OWN SENTENCE** ("the default is the only RS shape that
        // graph-replays"). The scatter and the gather are copies whose page
        // slots, in-page offsets and lengths are THIS fire's — not this
        // shape's — so baking them into a captured graph would replay one
        // window's addressing over another window's tokens. So a fire that
        // moves buffered bytes takes the eager walk, whatever mode the shell
        // is in: the same walk, the same launches, nothing recorded.
        // **AND A ROTATING LOAD IS NOT GRAPH-REPLAYABLE EITHER**, for a
        // reason with the same shape (alto streaming §3 item 4, D2b). The
        // dense pump rotates a slot's contents at each region boundary, and
        // its backpressure is a HOST cursor the walk advances; a replayed
        // graph has no walk, so a captured rotation would bake one fire's ring
        // state into an exec that outlives it. So a load whose weights rotate
        // takes the eager walk, whatever mode the shell is in: the same walk,
        // the same launches, nothing recorded. `crate::rotate`'s header
        // carries the whole argument.
        let records = graphs.records() && !p.rs.buffered && !weights.rotating();
        let walked = if records {
            // **THE ROUTER, AND IT IS TWO ARMS AND A HOLE** (the tier-2
            // campaign). A fire that reaches here either has a body or is the
            // load's own synthetic that could not have one; everything else
            // walks. There used to be two more arms — the keyed cache and the
            // fold's template — and collapsing them is what makes `p.bodied`
            // the whole question at this line.
            if arming && !p.bodied {
                // **THE ARMING PASS'S SYNTHETIC, WHOSE COMPOSITION THE GATE
                // REFUSED** (`Shell::arm_bodies`). There is nothing to record
                // — `prepare` already named the refusal into `bodies_refused`
                // — and there is nothing worth running either, because this
                // fire is nobody's: its numbers are read by no caller and its
                // launches would warm a composition the map will never hold.
                Ok(())
            } else if p.bodied {
                // **THE BODY ARM, AND SINCE THE TIER-2 CAMPAIGN THE ONLY
                // RECORDED ONE** (the bodies design's chunk B).
                //
                // Every clause that put this fire here was decided in
                // `prepare` and is in `Prepared::bodied`: the outer three this
                // `records` already carries, the armed pad, the unit-count
                // clause (`Shell::keyable_units`, which the multi-unit bodies
                // wave narrowed from `CompiledModel::fold_refused` — the
                // compiler's fact survives and says what it always said about
                // the FOLD; what died is the bodies path inheriting it), and
                // the admissibility rule `record::BodyKey` argues. Nothing is
                // re-asked, because the seat's words are already in the slot
                // and a second opinion here could only disagree with them.
                //
                // **AND THE ARMING INSTANT IS EITHER A REAL FIRE'S OR THE
                // LOAD'S, AND BOTH ARRIVE HERE.** A body's capture rides its
                // key's `record::WARM_FIRES`-th miss, whose own eager pass is
                // what warms the JIT, grows the scratch slabs and gets the
                // dense tuner's tuned ladder into the graph (`record`'s
                // header argues all three). `Shell::arm_bodies` climbs that
                // exact ladder at load with synthetic compositions — same
                // call, same counters, same number of eager walks — so a
                // load-armed body and a traffic-armed one are the same body,
                // and this line is the only place either is made.
                let fire = record::Fire {
                    trace,
                    compiled,
                    descriptor: &p.descriptor,
                    // The same table the `Run` above resolves in, handed to the
                    // record mode for the one thing the descriptor cannot say:
                    // how many rows each LAUNCH runs over, which is what a
                    // resident body's grids are compared against now that a
                    // windowed region can be one of them (chunk 2b-ii).
                    windows: &p.windows,
                    stream: device.stream(),
                    lanes: forked,
                    conditionals,
                    bucket: p.composition.bucket(),
                    // **AND THE TWO CONSTANTS THE KEY'S SECOND HALF IS CARVED
                    // FROM.** `prepare` built a ladder already
                    // (`Prepared::ladder`); this hands the INPUTS rather than that
                    // ladder so that `fire_body` builds its key exactly as the
                    // gate did — one function, `BodyKey::of`, off one composition,
                    // one phase apart. It used to hand the lattice, because a rung
                    // was `rung_of` over the class's rows; a rung is a ceiling
                    // now, so what has to travel is which classes are decode
                    // classes and how many lanes the load can seat.
                    decoding,
                    lane_ceiling: p.lane_ceiling,
                    // **AND WHETHER THIS ARTIFACT HAS A SECOND UNIT AT ALL**
                    // (the multi-unit bodies wave) — a LOAD constant, read off
                    // the bake at `Shell::load` and carried through `prepare`,
                    // so that `fire_body` builds the same key `prepare` built
                    // for the same reason every other input here travels: two
                    // readings of one fact are two keys waiting to disagree.
                    towered: p.towered,
                    // **AND THE TWO THE LEDGER NEEDS** (the grid-at-ceiling
                    // wave). `record::launch_grid` restates `Run::carve_rows` and
                    // `Run::carve_lanes` from outside the walk, so it needs the
                    // same two facts the `Run` above was handed: which regions
                    // move their own plane, and the bucket the pad was ARMED at
                    // — `armed.bucket`, not `Composition::bucket`, because a
                    // shell with the pad off carved nothing and its grids are
                    // live spans that must go on being able to grow.
                    shifted: shifted.as_slice(),
                    // **AND THE TABLE THAT SAYS WHICH REGIONS ANY OF THAT
                    // APPLIES TO** (the tier-2 campaign). The same slice the
                    // `Run` above was handed: `record::cuts` turns it into the
                    // capture script, and `record::launch_grids` and
                    // `record::grew_past` keep the ledger to the CAPTURED
                    // regions on the write and on the read alike. Handed
                    // rather than recomputed for `shifted`'s reason — the
                    // host's answer and the walk's have to be one answer.
                    admits: p.admits.as_ref(),
                    // **AND IT IS THE ARMED BUCKET WHOLE, WITH NO SLACK TEST IN
                    // FRONT OF IT** (the tier-1 key-collapse wave). It used to be
                    // zeroed where `bucket == rows`, to keep the ledger quiet on
                    // the `[engine] pad = off` arm; the bodies route now REQUIRES
                    // the pad (`prepare`'s gate), so the only fires that read this
                    // are padded ones and the only thing the old test could still
                    // reach was the padded fire that lands exactly on its lattice
                    // point — where zeroing it made the ledger describe live spans
                    // while the launches were issued at ceilings.
                    carve_bucket: armed.bucket,
                    // **AND THE SECOND AXIS'S, OFF THE PAIR THE SAME LINE
                    // ARMED** (the multi-unit bodies wave) — `armed_patches`
                    // rather than `Composition::patch_bucket`, for the reason
                    // the line above takes `armed.bucket`: the ledger has to
                    // describe the ceiling the WALK issued at, and a shell
                    // with the pad off carved nothing on either axis.
                    carve_patch_bucket: armed_patches.bucket,
                };

                cache.fire_body(&fire, &mut run, &place)
            } else {
                // **AND A RECORDING MODE WITH NO BODY FOR THIS FIRE WALKS**,
                // which is TIER 3 and is an answer rather than a fallback.
                // Since the tier-2 campaign what puts a fire here is never a
                // window's shape — a gathered or grouped region is an ISLAND
                // inside a body that serves the rest of the composition — but
                // one of the two things a cut cannot rescue: an artifact
                // with more capture units than the key names — which is none
                // of them today, since a `record::BodyKey` carries a pair per
                // row axis and `model_ir::RowAxis` has two variants — or a
                // composition the widening left no captured stretch in
                // (`record::widen`, `record::Uncut::Eager`).
                // Either way the refusal was already named, once per
                // composition, into `record::BodyTally::refusals`. Counted per
                // COMPOSITION and not per fire, deliberately: what an operator
                // needs is how many of its SHAPES this tier cannot serve.
                //
                // No pump is threaded onto this cursor and none can be: the
                // `records` line above already excluded every rotating load,
                // so `weights.rotor()` is `None` on every fire that reaches
                // this branch. The pumped cursor lives in the eager `else`
                // below, which is the one that serves them.
                let mut cursor = Cursor::new(&place);
                walk(trace, compiled, &p.descriptor, &mut run, &mut cursor)
                    .map_err(Fault::from)
            }
        } else {
            // **AND AN EAGER WALK UNDER A RECORDING MODE IS A WARNING, SO IT
            // IS COUNTED HERE.** Every other way a fire can miss its graph is
            // already a number the cache keeps — warming, declined, refused,
            // evicted — and the two clauses on the `records` line above were
            // the only ones that took a fire out of every graph without
            // leaving a trace anywhere. An operator who states `[engine]
            // graphs on` and reads a steady hit count has bought what it
            // thought it bought; one who reads these two moving instead now
            // knows WHICH sentence above spent its replays.
            //
            // **ONLY WHILE THE MODE RECORDS**, which is the whole of the
            // gate: `Graphs::Off` and `Graphs::Shaped` walk eagerly BY
            // CHOICE, and a counter that moved under them would be measuring
            // the knob. Both clauses are handed over rather than one, because
            // a fire can be disqualified twice and the second reason does not
            // stop mattering — `record::BodyTally::eager_buffered` states
            // that rule and what it costs (their sum is not a fire count).
            //
            // **AND THE LOAD'S OWN SYNTHETIC FIRES WOULD BE COUNTED LIKE ANY
            // OTHER**, because they are ordinary fires: `arm_bodies` climbs
            // the warm ladder through this same call, and nothing here knows
            // or cares that nobody is waiting on the answer.
            //
            // **WHICH IS EXACTLY WHY THAT LOOP NO LONGER RUNS ON A ROTATING
            // LOAD.** Its rungs used to land in this branch — real executed
            // walks at boot, `eager_rotating` moving before a caller had
            // connected, and not one exec captured at the end of it — so the
            // pass is now refused at its own gate for this counter's reason
            // (`Shell::arm_bodies`). What that buys the reading is that the
            // first nonzero `eager_rotating` on any load is a CALLER's fire:
            // the load no longer spends walks it knew in advance would be
            // spent for nothing, and the boot line is where the rotor is
            // announced instead.
            if graphs.records() {
                cache.eager_walk(weights.rotating(), p.rs.buffered);
            }
            // **THE ROTATION RIDES THE EAGER CURSOR** (alto streaming §3 item
            // 4). `Cursor::pumping` is the region seam: release, issue,
            // acquire, once per `region_begin`, on the fire's own compute
            // stream. `None` for every load that armed no pump, and then this
            // is the line it always was.
            let mut cursor = Cursor::new(&place);
            if let Some(rotor) = weights.rotor() {
                cursor = cursor.pumping(crate::window::Pump {
                    rotor,
                    compute: device.stream(),
                });
            }
            walk(trace, compiled, &p.descriptor, &mut run, &mut cursor).map_err(Fault::from)
        };
        drop(run);
        // **THE PAD IS THE FIRE'S, SO IT ENDS WITH THE FIRE** — including the
        // fire that ended in a refusal, which is why the walk's answer is held
        // rather than `?`-ed above. A context outlives every fire on it and a
        // pad left armed would still name the last fire's row count; the next
        // thing to fire on this stream is a guest program's epilogue, a
        // registration's copy or the next fire's warm pass, and none of them
        // is the fire that number was true of.
        //
        // **AND THE SEAT ENDS WITH IT, FOR THE SAME SENTENCE** (bodies
        // design): the address `Run::ctx` stamped points into the staging slot
        // this fire is about to release, so a stage left armed would be an
        // entry reading the next fire's words — or a freed carve's — through
        // an argument nobody re-checked. Every context the walk could have
        // armed is put back, including the conditional body's, which
        // `Run::ctx` reaches through `window::BODY` rather than through the
        // side list.
        device.ctx().disarm();
        device.ctx().disarm_stage();
        for ctx in &side_ctx {
            ctx.disarm();
            ctx.disarm_stage();
        }
        if let Some(body) = device.conditional_ctx() {
            body.disarm_stage();
        }
        walked?;

        // ── THE EPILOGUE AND THE BOOKKEEPING, BOTH MOVED UP OUT OF `settle`
        //    (F2b, and it is two of the sync's own five obligations).
        //
        //    They stood below the synchronize because everything did; neither
        //    ever needed it. An epilogue binds `IntrinsicId::Logits` to a
        //    rectangle the ARENA CARVE placed and a row the COMPOSITION
        //    numbered — both host arithmetic, both known here — and then
        //    launches, which is stream work and therefore this phase's.
        //    `held` is the count the NEXT step's `prepare` reads, and with
        //    settlement asynchronous "the next prepare" happens long before
        //    the callback: leaving it below would have step k+1 composing
        //    against step k's stale extent. Article 4 is what makes advancing
        //    it here honest — past admission the stream work is success-only,
        //    so a step that reached this line is a step whose KV WILL be
        //    written.
        let readback = if arming {
            // The synthetic arming pass computed nothing (capture does not
            // execute), so there is no readout to plan, no epilogue to run
            // and — load-bearing — no `held` to advance: its lanes borrowed
            // real slots for their page arithmetic and stated `held`
            // explicitly so nothing of the shell's counting moves.
            None
        } else {
            let out = exports.out;
            let logits = slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, the out seam, which the carve gave no rectangle",
                    out.0
                ),
            })?;
            if logits.dtype != Dtype::Bf16 {
                return Err(Fault::Unbound {
                    what: format!(
                        "an out seam landed as {:?}, which this shell cannot read back",
                        logits.dtype
                    ),
                });
            }
            // Which ROW of the arena's logits rectangle each SUBMITTED lane
            // reads — the fire order is the seriated one, so a lane's row is a
            // fact the composition holds and nothing else does. It is what the
            // readback indexes and what an epilogue's `logits` intrinsic is
            // offset by, and computing it twice is how the two would come to
            // disagree.
            let lane_count = p.lanes.len();
            let mut last_row = vec![0u32; lane_count];
            // And which rows it OWNS, first and count — the draft readout is
            // indexed by the first (`eta_exec`'s `mtp_draft_row`) and
            // the capture readout copies the whole run, so both come off the
            // same reading of the same composition.
            let mut first_row = vec![0u32; lane_count];
            let mut lane_rows = vec![0u32; lane_count];
            for row in p.composition.lanes() {
                let at = row.source as usize;
                last_row[at] = row.row_offset + row.rows - 1;
                first_row[at] = row.row_offset;
                lane_rows[at] = row.rows;
            }

            // ── THE CAPTURE COLUMNS' RECTANGLES (design §9, palo C4b). One
            //    per exported attention layer, each `[fire rows, heads]` F32.
            //    Resolved here, where the logits rectangle is resolved and for
            //    the same reason: the carve holds an export open past the last
            //    node, and this is the reader that knows where it is.
            let mut columns = Vec::with_capacity(exports.scores.len());
            if p.lanes.iter().any(|seated| seated.captures_scores) {
                for export in &exports.scores {
                    let column =
                        slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "value {}, an `{SCORES_SEAM}` export, which the carve gave no \
                                 rectangle",
                                export.value.0
                            ),
                        })?;
                    if column.dtype != Dtype::F32 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an `{SCORES_SEAM}` export landed as {:?}; the kernel's \
                                 log-sum-exp is F32 and this shell reads back no other",
                                column.dtype
                            ),
                        });
                    }
                    columns.push((export.layer, column));
                }
            }

            // ── The epilogue. The readout does not exist yet and does not
            //    need to: the intrinsic points at this lane's ROW of the arena
            //    rectangle, read where it lies rather than copied anywhere, and
            //    the launches behind it are ordered after the forward by the
            //    stream.
            //
            //    `INTRINSIC_STORAGE_RAW_BF16` and not a widened f32 buffer: the
            //    emitted kernel widens a bf16 column with `bits << 16`, which is
            //    the same arithmetic `bf16()` below does, so the guest reads
            //    exactly the f32 the caller is handed — bit for bit, which is
            //    what makes a parity diff against the host interpreter mean
            //    anything.
            //    **AND THE DRAFT COLUMN IS BOUND BESIDE IT** (palo C3b). The MTP
            //    export is a rectangle of its own — `mtp` and `out` are two
            //    values and the carve is what keeps them two — so
            //    `IntrinsicId::MtpLogits` takes that rectangle's base rather
            //    than an offset into the trunk's, and `mtp_draft_row` is the
            //    first row of this lane's draft window off the composition's own
            //    lane table. Bound only when the plan declares the export, which
            //    is exactly when `ModelProfile::has_mtp_logits` let the program
            //    declare the intrinsic in the first place; a shell that bound it
            //    otherwise would hand the guest the trunk's logits under the
            //    draft's name.
            let vocab = u32::try_from(logits.width as usize).unwrap_or(u32::MAX);
            let draft = match &exports.mtp {
                Some(export) => {
                    let column =
                        slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "value {}, the `{MTP_SEAM}` export, which the carve gave no rectangle",
                                export.value.0
                            ),
                        })?;
                    if column.dtype != Dtype::Bf16 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an `{MTP_SEAM}` export landed as {:?}, which this shell cannot \
                                 point an intrinsic at",
                                column.dtype
                            ),
                        });
                    }
                    Some(column)
                }
                None => None,
            };
            // **THE PREVIOUS FRAME'S EPILOGUES, COLLECTED HERE AND NOWHERE
            //    EARLIER.** A session may hold one airborne fire, so its lane
            //    has to be free before this loop stages the next — and this is
            //    the LATEST point that is true, which is the whole of why the
            //    wait is free: the forward of THIS frame is already on the
            //    stream above, so the device runs on across whatever the host
            //    blocks for.
            reap_guest_fires(programs, owed, airborne, guest_landed)?;
            let mut epilogues = AirborneFires::default();
            for attached in p
                .attachments
                .iter()
                .filter(|a| a.at == Boundary::Epilogue)
            {
                // ── **THE GUEST'S OWN ROWS, AND NOT THE LAST ONE THREE
                //    TIMES** (`palo B-readout`, the device half).
                //
                //    A lane's readout has two readers and this is the one the
                //    host never sees: an epilogue that reads
                //    `IntrinsicId::Logits` and argmaxes on the device, which
                //    is how every speculative verifier in the corpus gets its
                //    tokens. It reads `k` rows from wherever this call points
                //    it, `k` being the extent the GUEST declared — so a shell
                //    that pointed it at `last_row` handed a `k`-row verifier
                //    its own last row followed by `k - 1` rows past the end of
                //    the fire's rectangle. Zeros, and an argmax over zeros is
                //    token 0: the verifier then rejected every draft it made
                //    and speculation ran strictly more forward passes than no
                //    speculation at all.
                //
                //    `Seated::readout` is the lane's own list, by index within
                //    the lane, and `first_row` is where the lane's run starts.
                let lane = attached.lane as usize;
                let owned = lane_rows.get(lane).copied().unwrap_or(0);
                let stated = p.lanes.get(lane).and_then(|seated| seated.readout);
                let wanted: Vec<u32> = match stated {
                    // `Readout::Last` and `Readout::None` both arrive as
                    // `None`, and both mean the row every epilogue has been
                    // given since there were epilogues.
                    None => vec![last_row[lane]],
                    Some(rows) => {
                        let mut arena_rows = Vec::with_capacity(rows.len());
                        for &row in rows {
                            if row >= owned {
                                return Err(Fault::Ceiling {
                                    what: "rows in the lane a readout names",
                                    need: u64::from(row) + 1,
                                    have: u64::from(owned),
                                });
                            }
                            arena_rows.push(first_row[lane] + row);
                        }
                        // A stated-but-empty list is `Readout::None` reaching
                        // here as `Some(&[])`; the epilogue still runs and
                        // still reads a row, so it gets the one it always had.
                        if arena_rows.is_empty() {
                            arena_rows.push(last_row[lane]);
                        }
                        arena_rows
                    }
                };
                // **A CONSECUTIVE RUN IS STILL A BASE AND AN OFFSET**, which
                // is every `Readout::Last` and every verifier in the corpus
                // (`start .. start + k`). Only the shape a stride cannot spell
                // — a list that skips or descends — pays for a pointer table,
                // and `readout_rows` stays cold on every other fire.
                let consecutive = wanted
                    .windows(2)
                    .all(|pair| pair[1] == pair[0].wrapping_add(1));
                if consecutive {
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::Logits,
                        logits.ptr,
                        INTRINSIC_STORAGE_RAW_BF16,
                        vocab,
                        vocab,
                        wanted[0],
                    )?;
                } else {
                    // One `u64` per requested row, in REQUEST order — the
                    // kernel's `mode == 2` arm indexes this table and reads
                    // the row it finds, so the order the caller wrote is the
                    // order the guest sees.
                    let row_bytes = u64::from(vocab) * 2;
                    let table: Vec<u8> = wanted
                        .iter()
                        .flat_map(|row| {
                            (logits.ptr + u64::from(*row) * row_bytes).to_le_bytes()
                        })
                        .collect();
                    let at = u64::from(budget.max_tokens)
                        .saturating_mul(8)
                        .saturating_mul(lane as u64);
                    readout_rows.stage(device.stream(), at, &table)?;
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::Logits,
                        readout_rows.ptr() + at,
                        crate::program::launch::INTRINSIC_STORAGE_ROW_POINTERS,
                        vocab,
                        vocab,
                        0,
                    )?;
                }
                if let Some(column) = draft {
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::MtpLogits,
                        column.ptr,
                        INTRINSIC_STORAGE_RAW_BF16,
                        column.width,
                        column.width,
                        first_row[attached.lane as usize],
                    )?;
                }
                // ── **THE OBSERVABILITY DOOR** (`.wiki/alto/attn-score.md`
                //    §4). The capture arm wrote this lane's block of planes
                //    as the graph ran; this points the epilogue at it and
                //    nothing is copied anywhere. Bound at F32 and not at
                //    `INTRINSIC_STORAGE_RAW_BF16`, because a probability that
                //    a policy divides by is not a bf16 quantity — the slab is
                //    the one place in this shell where the four bytes are
                //    what they say.
                //
                //    **THE STRIDE IS THE SLAB'S AND THE ROWS ARE THE
                //    PROGRAM'S**, which is the whole contract
                //    (`eta_ir::registry::ATTN_SCORE_KV_MAX`): a guest states
                //    how many planes it means to read and reads a prefix of
                //    the layers, while the pitch between them is a number it
                //    could not have been told and must not guess.
                if let Some(slab) = scores.as_ref().filter(|_| {
                    p.lanes
                        .get(attached.lane as usize)
                        .is_some_and(|seated| seated.captures_scores)
                }) {
                    if attached.lane >= slab.lanes() {
                        return Err(Fault::Ceiling {
                            what: "fire lanes the score slab seats",
                            need: u64::from(attached.lane) + 1,
                            have: u64::from(slab.lanes()),
                        });
                    }
                    // **AND THE DECLARED CEILING IS REFUSED, NOT TRUNCATED.**
                    // The rows are the program's own claim and the pitch is
                    // the slab's, so a program claiming more planes than this
                    // load exports would read straight on into the NEXT
                    // lane's mass — silently, deterministically, and wrong.
                    // The type rule in `eta_ir::validate` can only check the
                    // width (the plane count is not in the profile), so this
                    // is where the other half of that contract is kept.
                    let declared = programs.declared_score_planes(attached.instance);
                    if let Some(declared) = declared
                        && declared > slab.planes()
                    {
                        return Err(Fault::Ceiling {
                            what: "attention-score planes this load exports",
                            need: u64::from(declared),
                            have: u64::from(slab.planes()),
                        });
                    }
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::AttnScore,
                        slab.lane_base(attached.lane),
                        crate::program::launch::INTRINSIC_STORAGE_F32,
                        crate::scores::KV_MAX,
                        crate::scores::KV_MAX,
                        0,
                    )?;
                }
                if let Some(fired) =
                    epilogues.stage(device, programs, attached.lane as usize, attached.instance)?
                {
                    committed_or(fired, attached.instance, "epilogue")?;
                }
            }

            // ── **THE EPILOGUE BOUNDARY'S WAIT, GONE** — the line this wave
            //    is about, and the last one in the fire path.
            //
            //    Sixty-four samplers are enqueued back to back above. What
            //    stood here read their verdicts, which meant a
            //    `cudaStreamSynchronize` for the whole frame: the device had
            //    nothing left when it returned and stayed idle for as long as
            //    the host took to build the next one. ~826 of them a c64 run,
            //    26% of the GPU's own span.
            //
            //    Three things made it removable and none of them is here.
            //    `channel::settle` advances the endpoint counters the next
            //    mint predicts off, on the device, in stream order.
            //    `Endpoint::predicted` answers where a shared ring stands
            //    without consulting a word at all. And a verdict is only ever
            //    an error path — nothing downstream reads one. So the fires
            //    are parked and `reap_guest_fires` collects them at the next
            //    frame, in front of the stage that needs the lane free, by
            //    which time the device has passed them and the reap costs two
            //    atomic loads.
            //
            //    A mid-batch flush's verdicts are the exception and are read
            //    here: a shared ring already forced that wait, so they are
            //    final now and naming them late would be worse.
            let mut settled: Vec<(usize, Fired)> = Vec::new();
            *owed = epilogues.defer(device, programs, guest_landed, seq, &mut settled)?;
            for (lane, fired) in settled {
                let attached = p
                    .attachments
                    .iter()
                    .find(|a| a.at == Boundary::Epilogue && a.lane as usize == lane)
                    .ok_or_else(|| {
                        Fault::program(
                            "serve::enqueue",
                            format!("lane {lane} settled an epilogue nothing attached"),
                        )
                    })?;
                committed_or(fired, attached.instance, "epilogue")?;
            }

            // The fire is enqueued, so the sequences are longer. Only the
            // slots this shell counts for — a caller that owns the page table
            // owns the count too, and writing into `held` under its slot
            // numbering would be writing into somebody else's table.
            for (seat, table) in p.seats.iter().zip(&p.tables) {
                if table.is_empty()
                    && let Some(slot) = held.get_mut(seat.slot as usize)
                {
                    *slot = seat.have + seat.rows;
                }
            }

            Some(Readback {
                logits,
                columns,
                last_row,
                first_row,
                lane_rows,
                captures: p.lanes.iter().map(|s| s.captures_scores).collect(),
            })
        };

        Ok((p.windows.launches(), readback))
    }
}

/// **THE FIRES OF ONE BOUNDARY, ENQUEUED AND UNSETTLED** (alto §14 exception
/// #1, closed).
///
/// A boundary is a run of independent guest passes — sixty-four samplers at
/// c=64, one per lane — and until this wave the shell fired them one at a
/// time, each ending in `Session::fire`'s own `cudaStreamSynchronize`. A
/// profile put the bill at 16,898 synchronize calls for 869 ms, 44% of all
/// CUDA API time, with the GPU idle 45% of its kernel span in ~56 µs bubbles
/// that matched the fires one for one: the host was waiting ~72 µs for a
/// 51 µs epilogue before it would mint the next lane's.
///
/// So the boundary enqueues everything and waits once. This holds what is
/// airborne between the two.
///
/// # And then the epilogue stopped waiting at all
///
/// One wait a boundary is still one wait a frame, and it drained the stream:
/// the device had nothing left when it returned and stayed idle for as long as
/// the host took to build the next frame. [`AirborneFires::defer`] is what
/// replaced it — the fires are parked as a [`GuestBatch`] and
/// [`reap_guest_fires`] collects them at the next frame — and it became
/// possible when `channel::settle` moved the endpoint counters onto the device
/// and `Endpoint::predicted` moved the shared rings' host answer off the
/// words. The PROLOGUE still waits, because its verdicts gate the forward
/// launched a few lines after them.
///
/// # The one ordering the batch may not flatten
///
/// A DEVICE-ONLY RING SHARED BY TWO ATTACHMENTS (design §5's draft→verify
/// chaining) is a putting pass and a taking pass, and the taker's admission
/// depends on the putter's settlement having happened. **That is a launch
/// order, not a host visibility problem, and it survived the move of the
/// prediction onto `Endpoint`**: `channel::pull_validate` runs ONCE at the
/// front of a wave, for every lane, before any lane's regions — so a taker
/// batched with its putter is validated against words the putter's
/// `channel::settle` has not reached yet, `REQUIRE_INPUT`'s `tail > head` is
/// false, and the fire is refused. Whatever the host believes, and however
/// the host came to believe it.
///
/// So two attachments of one ring must be two waves, and this reinstates that:
/// an attachment whose shared rings collide with one already airborne FLUSHES
/// the batch first — one synchronize, every verdict, a clean slate — and only
/// then launches. Nothing is lost but the batching, and only for the passes
/// that genuinely chain.
#[derive(Default)]
struct AirborneFires {
    /// `(tag, instance)` for every launch owing a settlement, in launch order.
    /// `tag` is whatever the caller wants back beside the verdict — an
    /// attachment index at the prologue, a lane at the epilogue.
    launched: Vec<(usize, u64)>,
    /// The identities of the shared rings the airborne fires hold, as
    /// `Session::shared_rings` answers them.
    rings: Vec<usize>,
    /// Settled verdicts a flush produced, kept until `settle_into` hands the
    /// whole boundary's back in one list.
    settled: Vec<(usize, Fired)>,
    /// **HAS THIS BATCH LEFT THE GROUND?** `stage` only mints; `fly` is what
    /// puts the pull, the regions and the tail on the stream, and it is
    /// idempotent because two callers reach for it — the prologue, which
    /// needs the fires enqueued before it writes the fold predicate, and the
    /// flush, which needs them enqueued before it waits.
    flown: bool,
}

impl AirborneFires {
    /// Stage instance `instance` into the plane's wave, flushing first if it
    /// chains onto a shared ring already airborne.
    ///
    /// Answers `Some(fired)` for a fire that never launched — a blocked
    /// channel or a poisoned instance, whose verdict is final without a wait
    /// — and `None` for one now holding a lane of the wave.
    ///
    /// **NOTHING IS ON THE STREAM WHEN THIS RETURNS.** The whole point of the
    /// wave is that a boundary's lanes are staged before any of them flies,
    /// so the three control kernels can launch once with a block per lane
    /// rather than once per attachment with one block. A caller that binds
    /// intrinsics or writes side tables between two `stage` calls is still
    /// ordered correctly: every one of those copies is enqueued before `fly`
    /// puts the first region on the stream.
    ///
    /// # Errors
    ///
    /// Whatever the mint, the flush's synchronize or a settlement said.
    fn stage(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        tag: usize,
        instance: u64,
    ) -> Result<Option<Fired>> {
        // **DEBRIS FROM A FAULTED BOUNDARY IS NOT THIS BATCH'S TO FLY.** The
        // wave is the plane's and lives across boundaries; a fault raised
        // between some earlier boundary's first stage and its landing unwinds
        // past the landing that would have cleared it. This batch's first
        // lane is the one moment nothing of ours is in there, so anything
        // that is belongs to fires nobody will settle.
        if self.launched.is_empty() && !self.flown && programs.staged() != 0 {
            programs.abandon_wave();
        }
        let rings = programs.shared_rings(instance);
        if rings.iter().any(|ring| self.rings.contains(ring)) {
            self.flush(device, programs)?;
        }
        match programs.stage(instance)? {
            crate::program::Launched::Airborne => {
                self.rings.extend(rings);
                self.launched.push((tag, instance));
                Ok(None)
            }
            crate::program::Launched::Refused(fired) => Ok(Some(fired)),
        }
    }

    /// **THE BATCH, ON THE STREAM**: one `pull_validate` over every staged
    /// lane, then each fire's regions in staging order, then one
    /// `commit_bump` and one `scatter_publish` over the same lanes.
    ///
    /// The order within a fire is what it always was — pull, regions, bump,
    /// publish — and the order BETWEEN fires is nothing, which is what makes
    /// the interleave sound: two lanes of one wave share no ring (a shared
    /// ring flushes at `stage`) and the stream orders each lane's own three
    /// phases around its own regions.
    ///
    /// Idempotent: a batch already flown is left alone.
    ///
    /// # Errors
    ///
    /// Whatever the copy and the launches said.
    fn fly(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.flown || self.launched.is_empty() {
            return Ok(());
        }
        programs.fly(device)?;
        programs.land(device)?;
        self.flown = true;
        Ok(())
    }

    /// Everything enqueued, one wait, then every airborne fire's verdict.
    fn flush(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(());
        }
        self.fly(device, programs)?;
        device.synchronize()?;
        for (tag, instance) in self.launched.drain(..) {
            let fired = programs.settle_launched(instance)?;
            self.settled.push((tag, fired));
        }
        self.rings.clear();
        self.flown = false;
        Ok(())
    }

    /// [`AirborneFires::flush`], appending every verdict this batch produced
    /// — including any a mid-batch flush already read — onto `into`.
    ///
    /// # Errors
    ///
    /// As [`AirborneFires::flush`].
    fn settle_into(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        into: &mut Vec<(usize, Fired)>,
    ) -> Result<()> {
        self.flush(device, programs)?;
        into.append(&mut self.settled);
        Ok(())
    }

    /// **EVERYTHING ENQUEUED AND NOTHING WAITED FOR** — the line this wave is
    /// about, and [`AirborneFires::settle_into`]'s replacement wherever a
    /// verdict can be read one frame late.
    ///
    /// Puts the batch on the stream, records `landed` behind it, and hands
    /// the airborne fires back as a [`GuestBatch`] for the caller to park.
    /// Any verdict a MID-BATCH flush already read is appended to `into` —
    /// those cost their wait when a shared ring forced one and are final now.
    ///
    /// `seq` is the step whose settlement callback will prove this batch
    /// landed; the reap reads it before it touches the event.
    ///
    /// # Errors
    ///
    /// Whatever the launches and the event record said.
    fn defer(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        landed: &crate::device::graph::Event,
        seq: u64,
        into: &mut Vec<(usize, Fired)>,
    ) -> Result<Option<GuestBatch>> {
        self.fly(device, programs)?;
        into.append(&mut self.settled);
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(None);
        }
        // **RECORDED ON THE COMPUTE STREAM, BEHIND THIS BATCH AND NOTHING
        //    MORE.** A stream synchronize would drain every launch enqueued
        //    after it too, which at the epilogue is the whole of the next
        //    frame; waiting on a point instead lets the device run past it
        //    while the host is still behind.
        landed.record(device.stream())?;
        let batch = GuestBatch {
            launched: core::mem::take(&mut self.launched),
            seq,
        };
        self.rings.clear();
        self.flown = false;
        Ok(Some(batch))
    }
}

/// **READ A DEFERRED BOUNDARY'S VERDICTS, WAITING ONLY IF THE DEVICE HAS NOT
/// PASSED THEM.**
///
/// The far half of [`AirborneFires::defer`], and the reason the boundary's
/// `cudaStreamSynchronize` could go at all. Three things had to become true
/// first, and each is somewhere else:
///
/// ```text
/// the endpoint counters the next mint predicts off  channel::settle, on the
///                                                   device, in stream order
/// where a SHARED ring stands, for either attachment Endpoint::predicted
/// the verdict itself                                only ever an error path
/// ```
///
/// So this is what is left of the wait: a check of two host atomics, and —
/// only when the frame that carried the batch has not called back yet — a
/// `cudaEventSynchronize` on the point the batch landed at. **The device is
/// not idle across it.** By the time anything reaps, the next frame's forward
/// is already enqueued behind the batch, so the host blocks and the GPU runs
/// on; that is the whole difference from the drain this replaced, where the
/// stream was empty on the far side of the wait and stayed empty for as long
/// as the host took to build the next frame.
///
/// **WHERE IT MUST BE CALLED, AND WHY EACH ONE.** In front of every path that
/// reads a guest ring on the host or stages a second fire into a session that
/// already has one:
///
/// ```text
/// serve::enqueue, before either boundary's stage loop   a session may hold
///                                                       ONE airborne fire
/// serve::prepare, before the descriptor-port read       the port is a cell
///                                                       `scatter_publish`
///                                                       writes
/// api's publish/take channel doors                      the same cells, from
///                                                       the runtime's side
/// close_instance                                        a session whose
///                                                       kernels are running
///                                                       may not be dropped
/// ```
///
/// # Errors
///
/// Whatever the wait said, and the first non-committing verdict — deferred by
/// one frame from where it used to be raised, which is the one semantic this
/// wave changes and is stated at [`committed_or`].
pub(super) fn reap_guest_fires(
    programs: &mut ProgramPlane,
    owed: &mut Option<GuestBatch>,
    airborne: &crate::settle::Airborne,
    landed: &crate::device::graph::Event,
) -> Result<()> {
    let Some(batch) = owed.take() else {
        return Ok(());
    };
    // The free question first. A batch whose frame has already settled is
    // reaped with no CUDA call at all, which is the steady state whenever the
    // host is not running ahead of the device.
    if !airborne.settled_past(batch.seq) {
        landed.settle()?;
    }
    let mut first: Option<crate::error::Fault> = None;
    for (lane, instance) in batch.launched {
        // **EVERY LANE IS SETTLED, EVEN AFTER ONE HAS FAULTED.** A session
        // that keeps its `pending` mint can never fire again, so an early
        // return here would turn one bad epilogue into a permanently stuck
        // instance for every lane behind it in the batch.
        let outcome = programs
            .settle_launched(instance)
            .and_then(|fired| committed_or(fired, instance, "epilogue"));
        if let Err(fault) = outcome {
            let _ = lane;
            first.get_or_insert(fault);
        }
    }
    match first {
        Some(fault) => Err(fault),
        None => Ok(()),
    }
}

/// A guest pass that ran, or the sentence for the one that did not.
///
/// **THREE VERDICTS ARE FAILURES HERE AND ONE IS NOT ELSEWHERE.** Fired on
/// its own, a [`Fired::Blocked`] program is a normal answer a caller retries
/// on. Attached to a model fire it is not: the gate already asked, before
/// anything launched, so a block at this point means the pass's own cursors
/// moved under it — which one attachment per instance is exactly the rule
/// that forbids. [`Fired::Declined`] is a stage clearing its commit slot and
/// [`Fired::Faulted`] is an instance that is unusable from now on; both leave
/// the guest's channels where they were, and both are the caller's to poison.
///
/// **AND AN EPILOGUE'S VERDICT NOW ARRIVES ONE FRAME LATE.** The epilogue
/// boundary is enqueue-only ([`AirborneFires::defer`]), so its fires are
/// settled by [`reap_guest_fires`] at the next frame and a fault raised here
/// fails THAT frame rather than the one that produced it. Nothing downstream
/// reads a verdict for anything but this: a guest's cells reach it through
/// device-written pinned words, and the fold predicate is the commit word
/// itself, on the device. The prologue boundary is unchanged and still waits,
/// because its verdicts gate the forward that follows them in the same call.
fn committed_or(fired: Fired, instance: u64, at: &str) -> Result<()> {
    match fired {
        Fired::Committed => Ok(()),
        Fired::Blocked(channel) => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {instance}'s {at} blocked on channel {channel} AFTER the gate \
                 admitted it, so something advanced its cursors between the two"
            ),
        )),
        Fired::Declined => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {instance}'s {at} declined: a stage cleared its commit slot, so \
                 nothing the guest computed this fire is visible"
            ),
        )),
        Fired::Faulted(why) => Err(Fault::program(
            "serve::fire",
            format!("instance {instance}'s {at} faulted and stays faulted: {why}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{Boundary, Fired, committed_or};
    use crate::serve::Attached;

    /// **A PASS THAT DID NOT COMMIT IS AN ERROR BY NAME, NEVER A REPLAY**
    /// (alto E; design §1 article 4, and the retry-fails-loudly gate).
    ///
    /// The readiness gate that used to stand in `prepare` answered
    /// `Fault::Blocked`, which `api::fault()` crossed as `Error::Exhausted`
    /// and the runtime's lane slept on and re-offered. Both are gone: static
    /// admission (`runtime::pipeline::fire::validate_frame`) proves ring
    /// occupancy, host-writer staging and reader pressure over the whole
    /// frame before it is admitted, so a pass that reaches its boundary and
    /// cannot commit means something moved cursors the admission had already
    /// proved — and an epilogue fires AFTER the forward wrote the lane's KV,
    /// so there is nothing to replay anyway.
    ///
    /// All three non-commit verdicts must therefore name the instance and say
    /// which one happened.
    #[test]
    fn a_pass_that_does_not_commit_on_an_admitted_fire_errors_by_name() {
        let attached = Attached {
            lane: 0,
            instance: 77,
            at: Boundary::Epilogue,
        };
        committed_or(Fired::Committed, attached.instance, "epilogue")
            .expect("a committed pass is the ordinary answer");

        for (fired, expected) in [
            (Fired::Blocked(3), "blocked on channel 3"),
            (Fired::Declined, "declined"),
            (Fired::Faulted("bad table".into()), "faulted"),
        ] {
            let fault = committed_or(fired, attached.instance, "epilogue")
                .expect_err("a pass that did not commit is not an outcome to retry");
            let said = fault.to_string();
            assert!(said.contains("77"), "the instance must be named: {said}");
            assert!(said.contains(expected), "{said}");
        }
    }
}
