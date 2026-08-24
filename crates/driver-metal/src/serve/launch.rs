//! The forward path: admit a sealed frame, then run its steps in order.
//!
//! The whole body is the four calls the executor is made of, with admission
//! in front. Nothing here decides what runs — the text states it, `lower`
//! flattens it, and `run` walks the result.

use crate::error::{Error, Result};
use crate::layout::region::Region;
use crate::serve::state::Shell;

/// What a frame did, when it did not fail.
///
/// Admission is NOT an error: a frame that does not fit is one the caller
/// re-posts, and one that could never fit is one it drops. Folding those into
/// `Err` would make the caller match on prose to tell a full pool from a
/// broken device.
#[derive(Debug)]
pub enum Launched {
    /// Every step was submitted, waited for, and its read-out handed to the
    /// programs bound to it.
    ///
    /// A fault poisons the one instance and does not fail the frame — the
    /// other requests batched with it did nothing wrong — so `faults` are
    /// returned for the caller to report rather than raised or logged.
    Ran {
        /// `(instance id, why)` for each program that faulted this frame.
        faults: Vec<(u64, String)>,
        /// How many of the frame's steps FIRED.
        ///
        /// `frame.steps.len()` in every ordinary case, and fewer when a step
        /// came back `Filled::Early` -- a device-resolved step whose
        /// descriptor channel was empty at fire time. The seam needs the
        /// number because it publishes one terminal outcome per member per
        /// step, and a member of a step that never ran must be told FAILED:
        /// an untouched cell holds `Pending`, which is not an outcome and
        /// which the scheduler reports as `work item completion terminal
        /// outcome is still Pending`.
        ran_steps: usize,
    },
    /// The pool cannot serve this frame now, but eviction could make room.
    Exhausted,
    /// The demand exceeds the physical pool, so waiting is waiting for
    /// something that cannot happen.
    Impossible,
}

impl Shell {
    /// Post one sealed frame: admit it, then run its steps in order.
    ///
    /// # Errors
    ///
    /// A frame whose step tables do not describe its rows, an architecture no
    /// text serves, or a device failure. Admission is not among them — see
    /// [`Launched`].
    pub fn launch(&mut self, frame: &driver_api::FrameSubmission) -> Result<Launched> {
        // ── Admission, against the frame-union demand. ──
        //
        // Before anything is encoded and without side effects, so the caller
        // can re-post. Against what the pool RESERVED, not what it holds: the
        // trim task gives pages back to a high-water mark, and a frame past
        // that mark is one the pool can serve after growing.
        if !self.need_pool("launch")?.admits(frame.required_kv_pages) {
            return Ok(Launched::Impossible);
        }

        // ── Grow to cover the pages this frame names. ──
        //
        // The highest page NAMED, not the count required: a frame's pages are
        // physical indices the scheduler chose anywhere in the pool, so a
        // frame needing two pages can name page 900. The trim task only
        // unmaps, so a pool left at the trimmed size would refuse this frame
        // at `translate` for a page the scheduler was right to hand out.
        let need = frame
            .kv_translation
            .iter()
            .copied()
            .filter(|&p| p != u32::MAX)
            .max()
            .map_or(0, |p| p.saturating_add(1))
            .max(frame.required_kv_pages);
        if self.pool.as_ref().is_some_and(|p| need > p.pages()) {
            let pool = self.pool.as_mut().expect("just inspected");
            pool.resize(&mut self.stepper, need)?;
        }

        let (Some(baked), Some(weights), Some(pool)) = (
            self.baked.as_ref(),
            self.weights.as_ref(),
            self.pool.as_ref(),
        ) else {
            return Err(Error::Unserved {
                what: "launch",
                message: "no checkpoint is loaded. `load_model` traces the lane and \
                          stages the tensors every fire's operands address."
                    .to_string(),
            });
        };

        // Which stack layers are linear-attention ones, in stack order.
        // Copied out of the deployment because the pool view built per fire
        // cannot hold a borrow of it, and it is a list of thirty integers.
        let rs_layers: Vec<u32> = baked
            .deployment
            .recurrent
            .as_ref()
            .map(|r| r.linear_layers.clone())
            .unwrap_or_default();

        // ── The page translation, checked per lane. ──
        //
        // A page past the pool addresses another layer's memory and attention
        // would read it without complaint, so this refuses rather than clamps.
        for lane in 0..frame.instance_ids.len() {
            crate::pools::kv::translate(
                pool,
                &frame.kv_translation,
                &frame.kv_translation_indptr,
                lane,
            )
            .map_err(|why| Error::Unserved {
                what: "launch",
                message: format!("this frame's kv translation: {why:?}"),
            })?;
        }

        // `dispatch::Geometry` STOOD HERE and was fourteen numbers a fire
        // could not read off anything else: the head counts, the affine point
        // the bytes arrived in, the router's fan, gemma-4's second attention
        // shape. Every one of them was an input to a GRID PLANNER, which read
        // a `kernel!` row's launch rule and decided how many threads a launch
        // took — and there is no such planner. A claim body computes its own
        // grid from the operands it was handed, and the numbers it cannot
        // derive are the ones its POINT DECLARES, arriving on the statement's
        // own params run.
        //
        // What is left of the geometry is the POOL's, and it goes on
        // `baker::KvGeometry`: three numbers the allocator settled, read by
        // the claim bodies through the view a `Cache` mark carries.
        //
        // `text_row` went with it. A plane is NAMED at load
        // (`Backend::Metal`), so there is no per-fire door onto a different
        // text and nothing for a driver's own measurements to select with.

        // A `named: HashMap<&str, Slice>` STOOD HERE and was the legacy
        // walk's weight table, joined by CHECKPOINT name. A fire binds
        // `weights.banks` — keyed by the name the PLAN names — straight into
        // `baker::walk::Fire::over`, so there is no second table and no join.

        // ONE timeline for the whole frame, so a step is QUEUED while the
        // previous one runs. `Stepper` waits for the step two back, because
        // there are two command allocators.
        //
        // Command buffers committed to one queue execute in submission order,
        // which is what makes this SAFE for steps that depend on each other:
        // step n+1 reads the KV step n appended.
        //
        // Per-FRAME rather than per-driver because `Stepper<'ctx>` borrows the
        // `Context` this struct owns, so holding one across `launch` calls is
        // a self-reference.
        let mut in_flight: Vec<(&driver_api::StepSubmission, InFlight)> = Vec::new();
        let mut faults: Vec<(u64, String)> = Vec::new();
        // How many steps of this frame actually fired. Every one of them, in
        // the ordinary case; fewer when a step came back `Early`, which the
        // seam turns into a FAILED terminal outcome for the members that
        // never ran. The scheduler resolves a member's work item by reading
        // that cell, and `Pending` -- what an untouched cell holds -- is not
        // an outcome, so a frame that stops halfway has to say where.
        let mut ran_steps = 0usize;
        // WHETHER THE FRAME MAY BE PIPELINED, which is one question with two
        // answers and they are not interchangeable.
        //
        // A frame of ordinary host-wire steps states every step's geometry up
        // front, so all of them can be encoded and committed before any is
        // waited for -- which is what this loop did unconditionally, and is
        // where this backend's run-ahead comes from.
        //
        // A frame with a DEVICE-RESOLVED member cannot be driven that way. A
        // decode envelope's tokens are the cells the PREVIOUS step's program
        // put on its channels, and that program runs in `run_programs`, after
        // its fire has retired. Encoding step n before step n-1 has been read
        // out means resolving step n's geometry from a ring whose front is
        // the fire-before-last's token: not a refusal, an answer, and the
        // wrong one -- the same token twice, or a position a page behind.
        //
        // So the frame is driven a step at a time exactly when it has to be,
        // and `serve::load` states the same fact to the engine as
        // `resolves_geometry_per_step`.
        let per_step = frame.steps.iter().any(crate::envelope::resolves_on_device);
        let page = pool.shape().page_size;

        for step in &frame.steps {
            // THE GEOMETRY THIS STEP FIRES OVER, which is not always the one
            // on the wire. See `crate::envelope`: a decode envelope arrives
            // with a placeholder token and empty page tables and has its real
            // ones read off the instance's channels, and EVERY class has its
            // working-set pages translated into the physical ones the frame
            // placed.
            let plan = match crate::envelope::fill(&self.registry, frame, step, page)? {
                crate::envelope::Filled::Ready { plan } => plan,
                // Nothing to fire and nothing wrong with the step itself: the
                // program that fills the channel has not run. v14 admission
                // is supposed to make that unreachable here -- the scheduler
                // does not seal a step whose producer is still owed a fire --
                // so this is reported rather than retried, and the members
                // that did not run are FAILED rather than left to publish a
                // silent SUCCESS. RETRY is not an outcome a terminal cell has.
                // `driver-vulkan`'s seam handles it the same way at the same
                // point.
                crate::envelope::Filled::Early { channel } => {
                    for &row in &step.roster_rows {
                        if let Some(&id) = frame.instance_ids.get(row as usize) {
                            faults.push((
                                id,
                                format!(
                                    "this step's geometry channel {channel} is unfilled at \
                                     fire time, so the step did not run"
                                ),
                            ));
                        }
                    }
                    break;
                }
            };
            let s = crate::baker::frame::Step {
                token_ids: &plan.token_ids,
                qo_indptr: &plan.qo_indptr,
                sampling_indices: &plan.sampling_indices,
                sampling_indptr: &plan.sampling_indptr,
            };
            let class = crate::baker::frame::fire_class(&s);
            // THE LANE THIS FIRE RUNS, picked by fact word off the programs
            // bound at LOAD.
            //
            // No lowering, no cache, no closure. The legacy path called
            // `binding::text(row, class, &binding)` here — a fresh
            // `ForwardPlan` per fire shape, flattened by
            // `model_compiler::lower` and memoised because deriving a
            // decode's graph cost 0.81 ms of a 4.9 ms step. `program::bound`
            // ran once at load and answered every lane the text states, so
            // what is left is a lookup.
            //
            // MASKED IS FALSE and it is a refusal rather than an omission:
            // this driver stages no custom mask (`Frame` carries none and
            // `bind::tables` writes the enable plane as zeros), so a text
            // that branches on `masked` would be handed its unmasked arm,
            // which is the right lane for the frame it is actually given.
            let (_word, program) = baked
                .lane(class, false)
                .map_err(|message| Error::Unserved {
                    what: "launch",
                    message,
                })?;
            // The gather's index list, in the fire's own numbering: the
            // wire's numbers are request-local and the gather's must not be.
            let sampled = crate::baker::frame::sampled_rows(&s).map_err(|why| Error::Program {
                message: format!("this step's read-out: {why}"),
            })?;

            // Every CSR invariant, checked BEFORE the pool is touched.
            //
            // A short or mis-sized CSR would resolve a token's physical KV
            // page to one belonging to another request, and the fire would
            // write this request's keys over that request's cache without
            // faulting. There is no safe fallback page, so the frame is
            // refused here, before anything is staged.
            plan.validate_geometry().map_err(|e| Error::Unserved {
                what: "launch",
                message: format!("this frame's geometry: {e}"),
            })?;
            plan.validate_kv_writes(pool.shape().page_size)
                .map_err(|e| Error::Unserved {
                    what: "launch",
                    message: format!("this frame's KV writes: {e}"),
                })?;
            // Where the paged append writes each token: its physical page and
            // the row inside it. Every lookup is infallible because
            // `validate_kv_writes` has already proved each token's virtual
            // page sits inside its own request's span.
            let (w_page, w_off) = {
                let page = pool.shape().page_size.max(1);
                let req = plan.req_of_token();
                let (mut pages, mut offs) = (Vec::new(), Vec::new());
                for (t, &pos) in plan.position_ids.iter().enumerate() {
                    let r = req[t] as usize;
                    let base = plan.kv_page_indptr[r] as usize;
                    let virt = base + (pos / page) as usize;
                    pages.push(plan.kv_page_indices[virt]);
                    offs.push(pos % page);
                }
                (pages, offs)
            };
            let req = plan.req_of_token();
            // The fire's own tables, staged into one device region: the
            // driver never reads what a table MEANS, only where the frame put
            // it.
            //
            // `i32` throughout: the shader reads some as `uint` and some as
            // `uchar`, and a `u32` written little-endian is the same first
            // byte. The narrowing is the kernel's and the width is the
            // frame's, which is the direction that is safe.
            // A fire's recurrent seats, straight off the wire.
            //
            // `RS_FLAG_RESET` is a seat starting a new request: its conv
            // window and its DeltaNet memory are both defined to be zero, and
            // a seat handed on from a finished request still holds that one's.
            // Cleared here rather than by a control op because the plan is
            // where the scheduler says it, and a reset at any other moment
            // would either wipe a seat mid-fire or leave the previous
            // request's memory in it for one step.
            if let Some(rs) = self.recurrent.as_ref() {
                for (i, &slot) in plan.rs_slot_ids.iter().enumerate() {
                    if step
                        .plan
                        .rs_slot_flags
                        .get(i)
                        .is_some_and(|f| f & driver_api::PIE_RS_FLAG_RESET != 0)
                    {
                        rs.clear_slot(slot)
                            .map_err(|e| crate::error::Error::Unserved {
                                what: "launch",
                                message: format!(
                                    "row {i} resets recurrent slot {slot}, which this pool does \
                                 not have: {e}. It holds {} seats -- `PIE_METAL_RS_SLOTS` \
                                 sizes it, and a scheduler admitting past what \
                                 `rs_cache_slots` advertised is the other way this happens.",
                                    rs.shape().slots
                                ),
                            })?;
                    }
                }
            }
            // One seat per fire ROW, which is not what the wire carries.
            //
            // `rs_slot_ids` is validated against `qo_indptr`, so it holds one
            // entry per REQUEST. Both slotted GDN kernels read it per TOKEN --
            // `gdn_prep_slotted` and `gdn_core_recurrent_slotted` index
            // `slot_ids[b_idx]` where `b_idx = tpig.z / Hv` runs over the
            // fire's rows. Passing the wire's vector straight through made
            // every token past the first request read PAST THE END of the
            // table region, and the garbage it read went on to index
            // `rstate` and `new_conv_state`: an unbounded device write from
            // an unbounded device read. What that cost is written up in
            // `model::qwen_3_5::forward::metal` -- a 128-token prefill wrote
            // NaN over its own fire tables, the positions among them, and the
            // next attention read a position of 0x7fc00000 and looped over
            // two billion key tiles. No fault, no wrong answer, just a GPU
            // that never came back.
            let rs_slots: Vec<u32> = if plan.rs_slot_ids.is_empty() {
                Vec::new()
            } else {
                req.iter()
                    .map(|&r| {
                        plan.rs_slot_ids
                            .get(r as usize)
                            .copied()
                            .ok_or_else(|| Error::Unserved {
                                what: "launch",
                                message: format!(
                                    "a token names request {r}, which has no recurrent seat: \
                                     the frame states {} of them",
                                    plan.rs_slot_ids.len()
                                ),
                            })
                    })
                    .collect::<Result<_>>()?
            };
            let staged = crate::bind::tables::stage(
                &self.context,
                &self.scratch,
                crate::bind::tables::Frame {
                    token_ids: &plan.token_ids,
                    position_ids: &plan.position_ids,
                    req_of_token: &req,
                    kv_page_indices: &plan.kv_page_indices,
                    kv_page_indptr: &plan.kv_page_indptr,
                    // The pool's, so `stage` can check the run reaches the
                    // last position. `Plan::validate` has already refused a
                    // short run on this path -- this states the divisor a
                    // second time so the CHECK is the same one on both paths
                    // and not a courtesy the serving path happens to get.
                    page_size: pool.shape().page_size,
                    kv_write_page: &w_page,
                    kv_write_offset: &w_off,
                    // The FIRE's rows, translated from the request-local
                    // numbering the wire uses. See `sampled_rows`.
                    sampling_indices: &sampled,
                    // Which seat each ROW's linear-attention state lives in,
                    // one per token -- see `rs_slots` above for why this is
                    // not `plan.rs_slot_ids`.
                    //
                    // Empty for a checkpoint with no recurrent stack, which is
                    // most of them, and `Staged::at` answers `None` for an
                    // empty table -- so a GDN symbol that reached such a fire
                    // refuses for want of the table rather than indexing slot
                    // zero.
                    recurrent_slots: &rs_slots,
                },
            )?;
            // The stand-in for an operand that addresses NOTHING:
            // `dispatch::bind` answers an unfilled slot with address zero,
            // which `encode` binds happily and a recorded command cannot. The
            // tables region is real, resident, and never written through.
            self.regions.add(staged.region());
            self.regions.set_null(staged.region());

            // ── WHAT THIS FIRE ADDRESSES BESIDE ITS ARENA. ──
            //
            // Three kinds of byte a `Program`'s slots cannot name: the
            // runtime planes the scheduler decided, the page translation it
            // chose, and the pools that outlive the fire. `baker::Pools` is
            // the one door onto all three, and this is the driver's answer to
            // it — plain regions, so that the walk itself never meets a Metal
            // type.
            //
            // `Names::mlx()` STOOD HERE, with a `Store` built over it: a
            // TRACE-NAME to CHECKPOINT-NAME map, because the legacy load
            // staged tensors under the file's own names and the text asked
            // for its own. `model::produce` answers with the names the plan's
            // params carry, so there is one name space and nothing to map.
            let staging = FireStaging {
                pool,
                recurrent: self.recurrent.as_ref(),
                rs_layers: &rs_layers,
                staged: &staged,
            };

            // THE FIRE'S OWN COUNTS, which the plan deliberately does not
            // hold: how many rows a fire carries is decided when the
            // scheduler batches, and `qo_indptr`'s segment count IS the
            // request count (the appender reads it that way).
            let rows = u32::try_from(plan.token_ids.len()).unwrap_or(u32::MAX);
            let requests = u32::try_from(plan.qo_indptr.len().saturating_sub(1)).unwrap_or(0);

            // ── THE ARENA, LEASED BEFORE THE WALK. ──
            //
            // The walk resolves every rectangle to an ADDRESS inside it, so
            // the lease has to exist first; `fire::run::submit` takes the
            // same one rather than allocating, which is what makes it
            // impossible to plan against one arena and bind another.
            let arena = self.scratch.take(
                &self.context,
                (program.row_pitch * u64::from(rows)).max(1),
                "activation arena",
            )?;
            // SAFETY: freshly leased; nothing is encoded against it yet.
            // Zeroed because a slot no kernel writes otherwise holds whatever
            // the ring handed over.
            unsafe { arena.zero(0, arena.len())? };

            let fire = crate::baker::walk::Fire::over(
                &baked.plan,
                program,
                crate::baker::walk::Extent {
                    arena: crate::baker::Slice {
                        address: arena.gpu_address(),
                        bytes: arena.len(),
                    },
                    rows: i32::try_from(rows).unwrap_or(i32::MAX),
                    requests: i32::try_from(requests).unwrap_or(i32::MAX),
                    layers: baked.deployment.layers as usize,
                },
                &weights.banks,
                &staging,
            );
            let encoder = crate::baker::encode::Encoder::over(&fire.bindings, &fire.cursor);
            fire.walk(&encoder).map_err(Error::from)?;
            let dispatches = encoder.finish();
            let blits = fire.blits.borrow().clone();
            // The read-out is the `out` seam's own rectangle, which the walk
            // sized like every other value. It travels with the fire because
            // the `Fire` is dropped at the end of this iteration.
            let readout = fire.rect(baked.out).ok().map(|r| Readout {
                offset: r.slice.address.saturating_sub(arena.gpu_address()),
                rows: r.rows,
                width: r.width,
                bytes: r.dt.size(),
            });
            drop(fire);

            let mut machine = crate::fire::run::Machine {
                context: &self.context,
                compiler: &self.compiler,
                pipelines: &mut self.pipelines,
                stepper: &mut self.stepper,
                scratch: &self.scratch,
                regions: &mut self.regions,
                recordings: Some(&mut self.recordings),
            };
            let fire = crate::fire::run::submit(&mut machine, arena, &dispatches, &blits)?;
            // Committed, not waited for. The `Fire` is dropped at the end
            // of this iteration, so the read-out's shape travels with it.
            //
            // `staged` travels with it, and that is not tidiness: a fire's
            // tables are a LEASE from `self.scratch`, returned to the pool
            // when the value drops, and a returned region is one the next fire
            // may be handed. Dropped here, the next step would stage its token
            // ids over the ones a running fire is still reading.
            in_flight.push((
                step,
                InFlight {
                    fire,
                    readout,
                    _tables: staged,
                },
            ));
            ran_steps += 1;
            // ── A step at a time, when a step at a time is what it takes. ──
            //
            // Same body as the drain below, one entry deep: the fire is
            // waited for and its read-out handed to the programs before the
            // NEXT step is encoded, so the next step's `envelope::fill` reads
            // channels this step's program has already written. That
            // ordering is the whole of what makes a decode envelope work, and
            // it costs the frame's pipelining -- which is why it is asked for
            // rather than always done.
            if per_step {
                retire(
                    &mut self.stepper,
                    self.recurrent.as_ref(),
                    &mut self.registry,
                    &frame.instance_ids,
                    &mut in_flight,
                    &mut faults,
                )?;
            }
        }

        // ── The read-outs, and the channel plane over them. ──
        //
        // After the whole frame is committed, in submission order: reading an
        // arena before its fire retires is a plausible tensor and the wrong
        // one. Empty already when the frame was driven a step at a time,
        // which is what lets the two disciplines share one body instead of
        // two copies that drift.
        retire(
            &mut self.stepper,
            self.recurrent.as_ref(),
            &mut self.registry,
            &frame.instance_ids,
            &mut in_flight,
            &mut faults,
        )?;
        Ok(Launched::Ran { faults, ran_steps })
    }
}

/// One committed step, and everything that has to outlive its fire.
///
/// A named struct rather than the tuple this was, because [`retire`] takes it
/// as a parameter now: the two disciplines -- a frame committed whole and a
/// frame driven a step at a time -- share one body, and a body that names its
/// argument type is one the compiler checks rather than one three call sites
/// agree about by position.
struct InFlight {
    /// The timeline value to wait on, and the arena to read out.
    fire: crate::fire::run::InFlight,
    /// The read-out's shape, which travels with the fire because the `Fire`
    /// that computed it is dropped at the end of the iteration that encoded
    /// it.
    readout: Option<Readout>,
    /// The fire's tables, HELD and never read again.
    ///
    /// Not tidiness: a fire's tables are a LEASE from `Shell::scratch`,
    /// returned to the pool when the value drops, and a returned region is
    /// one the next fire may be handed. Dropped at the end of the encode, the
    /// next step would stage its token ids over the ones a running fire is
    /// still reading.
    _tables: crate::bind::tables::Staged,
}

/// Wait for every committed step, then run the programs bound to it.
///
/// Drains `in_flight` in submission order, which is the order the fires
/// retire in: one queue, and command buffers on one queue execute in the
/// order they were committed.
///
/// The fields are taken one by one rather than as `&mut Shell`, and that is
/// not a style choice: the caller is holding borrows of `Shell::model`,
/// `Shell::pool` and `Shell::scratch` across the whole launch, so a `&mut
/// self` here would conflict with all three. Disjoint fields do not.
///
/// # Errors
///
/// A roster row that names no bound instance, or a device failure while
/// waiting.
fn retire(
    stepper: &mut crate::device::Stepper<'static>,
    recurrent: Option<&crate::pools::recurrent::Pool>,
    registry: &mut crate::channel::Registry,
    instance_ids: &[u64],
    in_flight: &mut Vec<(&driver_api::StepSubmission, InFlight)>,
    faults: &mut Vec<(u64, String)>,
) -> Result<()> {
    for (step, committed) in in_flight.drain(..) {
        stepper.wait_for(committed.fire.value)?;
        // What this fire's gated-DeltaNet layers wrote becomes what the
        // next one reads. After the wait, because it is a host `memmove`
        // over the same planes the fire was writing.
        if let Some(rs) = recurrent {
            // SAFETY: this fire has retired and the next has not been
            // encoded -- both statements are this loop's own structure.
            unsafe { rs.carry_forward()? };
        }
        // What the fire COMPUTED, handed to the programs bound to this
        // frame. Until this landed the seam ran every launch and dropped
        // the arena, so a green frame and a frame that computed the wrong
        // thing were the same observation — `pipeline::step` had no
        // production caller at all, and the interpreter was exercised
        // only by tests that built their own inputs.
        let logits = read_logits(&committed.fire.arena, committed.readout);
        run_programs(registry, instance_ids, step, logits.as_ref(), faults)?;
    }
    Ok(())
}

/// Run the channel-plane pass for every program batched into one step.
///
/// One instance per roster row, in sub-batch order, each over ITS OWN rows of
/// the read-out: member `p` reads `program_row_indptr[p]..[p+1]` and nothing
/// else.
///
/// Neither a blocked pass nor a fault is an error. Readiness is the program's
/// own gate; a fault poisons the one instance, and failing the whole frame
/// would take down every other request batched with it. Faults are appended to
/// `faults` for the caller to report.
///
/// # Errors
///
/// A roster row that names no bound instance — which is a frame the scheduler
/// built against a registry it did not have.
fn run_programs(
    registry: &mut crate::channel::Registry,
    instance_ids: &[u64],
    step: &driver_api::StepSubmission,
    logits: Option<&(Vec<f32>, u32, u32)>,
    faults: &mut Vec<(u64, String)>,
) -> Result<()> {
    for (member, &row) in step.roster_rows.iter().enumerate() {
        let id = *instance_ids
            .get(row as usize)
            .ok_or_else(|| Error::Unserved {
                what: "launch",
                message: format!("roster row {row} is outside the frame's instances"),
            })?;
        // THIS member's rows of the read-out, and nothing else. Slicing rather
        // than passing an offset keeps `bind_intrinsic`'s `base_row = 0` TRUE
        // for each member instead of making it a parameter every caller could
        // forget: the interpreter's view is its own rows, so there is no row
        // it could reach that is not its.
        let inputs = match logits {
            None => crate::channel::PassInputs::none(),
            Some((values, rows, vocab)) => {
                let sampling = Sampling {
                    qo_indptr: &step.plan.qo_indptr,
                    indices: &step.plan.sampling_indices,
                    indptr: &step.plan.sampling_indptr,
                };
                let (start, end) = member_rows(&step.program_row_indptr, &sampling, member, *rows)
                    .ok_or_else(|| Error::Unserved {
                        what: "launch",
                        message: format!(
                            "member {member} is not described by the {}-entry attribution CSR \
                             over the {} wire request(s) whose {rows} read-out row(s) this fire \
                             produced",
                            step.program_row_indptr.len(),
                            step.plan.qo_indptr.len().saturating_sub(1)
                        ),
                    })?;
                let span = (end - start) as usize * *vocab as usize;
                let from = start as usize * *vocab as usize;
                if from + span > values.len() {
                    return Err(Error::Unserved {
                        what: "launch",
                        message: format!(
                            "member {member} claims read-out rows {start}..{end} of \
                             {rows}, which is past the {} values this fire produced",
                            values.len()
                        ),
                    });
                }
                crate::channel::PassInputs {
                    logits: Some(&values[from..from + span]),
                    rows: end - start,
                    vocab: *vocab,
                    mtp_draft_row: None,
                }
            }
        };
        match registry.fire(id, &inputs) {
            Ok(crate::channel::StepOutcome::Committed)
            | Ok(crate::channel::StepOutcome::Blocked(_)) => {}
            Ok(crate::channel::StepOutcome::Faulted(why)) => faults.push((id, why.to_string())),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// This fire's logits, widened to `f32`, with the two extents beside them.
///
/// `None` when the text states no exit seam — a fire that computes something
/// other than a distribution, which is not an error.
///
/// The interpreter's `PassInputs` wants `&[f32]` and the metal read-out is
/// **bf16** (`affine_qmv_fast` writes bf16 whatever the text declares), so the
/// bytes are reinterpreted anyway and a widening reinterpretation is a copy.
/// bf16 → f32 is exact: the low sixteen bits are zero.
fn read_logits(
    arena: &crate::fire::scratch::Lease,
    readout: Option<Readout>,
) -> Option<(Vec<f32>, u32, u32)> {
    let r = readout?;
    let bytes = usize::try_from(r.bytes).ok()?;
    let at = usize::try_from(r.offset).ok()?;
    let span = r.rows as usize * r.width as usize * bytes;
    if at + span > arena.len() as usize {
        return None;
    }
    // SAFETY: the arena is `StorageModeShared`, so its contents are host
    // addressable, and the caller waits on the fire (`stepper.wait_for`)
    // before reading. The wait is what makes the read valid, so a caller that
    // stops waiting has to stop calling this.
    let raw =
        unsafe { std::slice::from_raw_parts(arena.contents().cast::<u8>().as_ptr().add(at), span) };
    let values = match bytes {
        2 => raw
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        4 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => return None,
    };
    Some((values, r.rows.unsigned_abs(), r.width.unsigned_abs()))
}

/// Where a fire's logits are, as an offset into its own arena.
///
/// `model_compiler::lower::Readout` STOOD HERE and was the lowering's answer
/// to the same question: an offset, a row count, a vocab width and an element
/// size, computed while flattening the text. The walk answers it with the
/// `out` seam's own slot — the same rectangle every other value gets, sized
/// by the same rule — so this is a `Rect` with its address expressed relative
/// to the arena it is a span of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Readout {
    /// Bytes into the fire's arena.
    offset: u64,
    /// Rows the fire read out.
    rows: i32,
    /// The logits width — the model's vocabulary, as the plan sized it.
    width: i32,
    /// Bytes per element, from the slot's own dtype.
    bytes: u64,
}

/// What this fire addresses beside its arena: the pools, the page
/// translation, and the planes the scheduler staged.
///
/// THE DRIVER'S ANSWER TO `baker::Pools`, and the reason the walk itself
/// never meets a Metal type. Every method here turns a device object into a
/// plain `(address, bytes)`; the executor reads regions and mints handles,
/// and what is behind them is this struct's business alone.
struct FireStaging<'a> {
    pool: &'a crate::pools::kv::Pool,
    recurrent: Option<&'a crate::pools::recurrent::Pool>,
    /// Which STACK layers are linear-attention ones, in stack order.
    ///
    /// The pool allocated one plane per LINEAR layer with nothing between
    /// them, and a text names the layer it IS — qwen3.6's gated-DeltaNet
    /// layers are 0, 1, 2, 4, 5, 6, 8, … — so layer 4 is plane 3. Allocating
    /// a plane per stack layer would have made the two indices agree and
    /// thrown away a quarter of the pool on full-attention layers that have
    /// no state at all.
    rs_layers: &'a [u32],
    staged: &'a crate::bind::tables::Staged,
}

impl crate::baker::Pools for FireStaging<'_> {
    fn kv(&self, layer: u32, values: bool) -> Option<crate::baker::Slice> {
        self.pool.layer(layer).map(|l| {
            let h = if values { &l.v } else { &l.k };
            crate::baker::Slice {
                address: h.gpu_address(),
                // THIS layer's, not the pool's: full-attention layers hold a
                // different page size from sliding ones, and an over-stated
                // length is one attention reads past.
                bytes: self.pool.shape().layer_bytes_at(layer),
            }
        })
    }

    fn slab(&self, layer: u32, which: crate::baker::Slab) -> Option<crate::baker::Slice> {
        let pool = self.recurrent?;
        let ord = self.rs_layers.iter().position(|&x| x == layer)?;
        let l = pool.layer(u32::try_from(ord).ok()?)?;
        // `conv` and `new_conv` are two DISTINCT planes: the kernel reads the
        // window while shifting it, from different threadgroups, so the
        // shifted one cannot land where the taps are being read from.
        // `Pool::carry_forward` makes the second the first again once the
        // fire retires.
        let (region, bytes) = match which {
            crate::baker::Slab::Conv => (&l.conv, pool.shape().conv_bytes_per_layer()),
            crate::baker::Slab::NewConv => (&l.new_conv, pool.shape().conv_bytes_per_layer()),
            crate::baker::Slab::State => (&l.state, pool.shape().state_bytes_per_layer()),
        };
        Some(crate::baker::Slice {
            address: region.gpu_address(),
            bytes,
        })
    }

    fn kv_geometry(&self) -> crate::baker::KvGeometry {
        let shape = self.pool.shape();
        crate::baker::KvGeometry {
            page_size: i32::try_from(shape.page_size).unwrap_or(0),
            seq_stride: u64::from(shape.kv_heads) * u64::from(shape.head_dim),
            head_stride: u64::from(shape.page_size) * u64::from(shape.head_dim),
        }
    }

    fn table(&self, which: crate::baker::FireTable) -> Option<crate::baker::Slice> {
        self.staged.at(which)
    }
}

/// Which rows of a fire's read-out belong to batch member `member`.
///
/// `program_row_indptr` is the frame's own attribution CSR — member `p` owns
/// WIRE REQUEST rows `[indptr[p], indptr[p+1])` — and an ABSENT one is the
/// single-member case, where the whole read-out is that member's.
///
/// # The CSR counts requests and the read-out counts sampled rows
///
/// Those are two numberings, and this function used to spend the first as if
/// it were the second: `[indptr[p], indptr[p+1])` was sliced straight out of
/// the logits. Today they agree, because every request a fire serves reads
/// exactly one row — its last — and one request is therefore one read-out
/// row. Nothing in the frame *says* that. A speculative verifier names one
/// read-out row per drafted token (`sampling_indices` is exactly that list),
/// and the moment one appears the two numberings come apart at the FIRST
/// member that follows it: request 2 is read-out row 4, and the untranslated
/// read hands the member rows 2..3, which are another conversation's drafts.
/// Real distributions of the right width, no fault, no stall — the same shape
/// of silence the `(0, rows)` fallback used to have, one batch member over.
///
/// So the request span is translated through [`Sampling`], which counts what
/// each request actually contributes to the read-out under the same rule
/// `model_compiler::lower::Readouts::samples` used to build it. `driver-wgpu`
/// resolves the identical question through `Step::readouts_of` in
/// `frames.rs`, and Metal had no equivalent at all.
///
/// # Why a CSR that does not describe this member is `None`
///
/// It used to be the same fallback: any unusable CSR answered `(0, rows)`.
/// That is the right answer for a frame that states no attribution and the
/// wrong one for a frame whose table disagrees with its own roster, because
/// the interpreter's `base_row` is 0 — the member reads from read-out row 0,
/// which in a batched frame is ANOTHER CONVERSATION'S distribution. On
/// `[0, 1, 2, 9]` over three read-out rows, members 0 and 1 answer `(0, 1)`
/// and `(1, 2)` — correct — and member 2 answered `(0, 3)`, sampling request
/// 0's tokens and returning them as its own. One member out of three, in a
/// frame whose other members are fine, with nothing faulted. A CSR shorter
/// than the roster does the same.
///
/// "Absent" and "present but not describing this member" are different
/// claims, and `driver-wgpu::frames::member_requests`,
/// `driver-vulkan::frames::member_requests` and this crate's own
/// `envelope::member_requests` all keep them apart and refuse the second.
/// This copy did not, and it is the one that reads a DISTRIBUTION.
///
/// The bound is the read-out's own row count: a span past it is a frame whose
/// tables disagree with the fire they were built for. [`run_programs`]
/// already refused that a line later, by measuring the slice against the
/// values the fire produced — the check is here as well so that both ways of
/// being undescribed are one answer rather than two.
fn member_rows(
    program_row_indptr: &[u32],
    sampling: &Sampling<'_>,
    member: usize,
    rows: u32,
) -> Option<(u32, u32)> {
    if program_row_indptr.len() < 2 {
        return Some((0, rows));
    }
    let (&from, &to) = (
        program_row_indptr.get(member)?,
        program_row_indptr.get(member + 1)?,
    );
    if to < from {
        return None;
    }
    let (start, end) = sampling.read_out(from, to)?;
    (end <= rows).then_some((start, end))
}

/// The read-out table a fire was lowered with, read back the same way.
///
/// # Why this recounts rather than trusting a length
///
/// The count a member needs is "how many read-out rows did requests
/// `0..r` contribute", and there is no array carrying it: `sampling_indptr`
/// segments `sampling_indices`, which is not the same list. A request that
/// names NO row still reads one — its own last, which is what a decode means
/// — and a request with no tokens at all reads none. Naming the same row
/// twice is one row, because the read-out is built from a per-row `samples`
/// bitmap and a bit set twice is set once.
///
/// All three rules live in `Readouts::samples`, and they are mirrored here
/// rather than approximated, because the number this produces indexes into
/// the values that function's answer laid out. Approximating it is not a
/// smaller answer, it is a different member's rows.
struct Sampling<'a> {
    /// Request → token row CSR for the fire.
    qo_indptr: &'a [u32],
    /// Rows read, each numbered inside its own request.
    indices: &'a [u32],
    /// Request → readout CSR over [`Self::indices`], empty when none are
    /// named.
    indptr: &'a [u32],
}

impl Sampling<'_> {
    /// The read-out rows contributed by wire requests `[from, to)`.
    ///
    /// `None` when a request in that span is not described by `qo_indptr` at
    /// all, which is a frame whose tables were built for a different fire.
    fn read_out(&self, from: u32, to: u32) -> Option<(u32, u32)> {
        let mut start = 0u32;
        let mut end = 0u32;
        for r in 0..to as usize {
            let n = self.rows_of(r)?;
            if r < from as usize {
                start = start.checked_add(n)?;
            }
            end = end.checked_add(n)?;
        }
        Some((start, end))
    }

    /// How many read-out rows request `r` contributes.
    fn rows_of(&self, r: usize) -> Option<u32> {
        let (&lo, &hi) = (self.qo_indptr.get(r)?, self.qo_indptr.get(r + 1)?);
        if hi <= lo {
            // A request with no token rows has no last row to read.
            return Some(0);
        }
        let named = self.named(r);
        if named.is_empty() {
            return Some(1);
        }
        let mut rows: Vec<u32> = named.to_vec();
        rows.sort_unstable();
        rows.dedup();
        u32::try_from(rows.len()).ok()
    }

    /// The rows request `r` names, in its own numbering.
    ///
    /// Lenient exactly where `Readouts::of` is: this reports what the fire
    /// DID, and the fire already ran.
    fn named(&self, r: usize) -> &[u32] {
        if self.indices.is_empty() {
            return &[];
        }
        let (Some(&lo), Some(&hi)) = (self.indptr.get(r), self.indptr.get(r + 1)) else {
            return &[];
        };
        if hi < lo {
            return &[];
        }
        self.indices.get(lo as usize..hi as usize).unwrap_or(&[])
    }
}

#[cfg(test)]
mod readout_rows {
    use super::{Sampling, member_rows};

    /// A decode: every request one token row, every one reading it.
    fn decodes(requests: u32) -> Vec<u32> {
        (0..=requests).collect()
    }

    /// The read-out table of a fire whose requests name no row of their own,
    /// which is what a decode and an ordinary prefill both submit.
    fn plain<'a>(qo_indptr: &'a [u32]) -> Sampling<'a> {
        Sampling {
            qo_indptr,
            indices: &[],
            indptr: &[],
        }
    }

    /// Three requests batched into one fire, one read-out row each.
    #[test]
    fn each_member_of_a_batched_frame_reads_its_own_row() {
        let qo = decodes(3);
        let s = plain(&qo);
        let indptr = [0, 1, 2, 3];
        assert_eq!(member_rows(&indptr, &s, 0, 3), Some((0, 1)));
        assert_eq!(member_rows(&indptr, &s, 1, 3), Some((1, 2)));
        assert_eq!(member_rows(&indptr, &s, 2, 3), Some((2, 3)));
    }

    /// A member may own several REQUESTS, and the span is all of their rows.
    #[test]
    fn a_member_that_owns_several_requests_gets_all_of_their_rows() {
        let qo = decodes(5);
        let s = plain(&qo);
        let indptr = [0, 4, 5];
        assert_eq!(member_rows(&indptr, &s, 0, 5), Some((0, 4)));
        assert_eq!(member_rows(&indptr, &s, 1, 5), Some((4, 5)));
    }

    /// A PREFILL'S REQUEST IS MANY TOKEN ROWS AND ONE READ-OUT ROW.
    ///
    /// The attribution CSR counts requests, so nothing here moves when the
    /// requests get wider: two prefills of four and three tokens are still
    /// read-out rows 0 and 1. This is the case that made the untranslated
    /// read look correct for as long as it did.
    #[test]
    fn a_prefills_width_does_not_reach_the_readout() {
        let qo = [0, 4, 7];
        let s = plain(&qo);
        let indptr = [0, 1, 2];
        assert_eq!(member_rows(&indptr, &s, 0, 2), Some((0, 1)));
        assert_eq!(member_rows(&indptr, &s, 1, 2), Some((1, 2)));
    }

    /// A SPECULATIVE MEMBER MOVES EVERY MEMBER AFTER IT, AND THE UNTRANSLATED
    /// READ HANDED THEM SOMEBODY ELSE'S DRAFTS.
    ///
    /// Request 0 verifies three drafted tokens and names three rows of its
    /// own span; requests 1 and 2 are ordinary decodes. The read-out is five
    /// rows — `0..3` are request 0's, row 3 is request 1's, row 4 is request
    /// 2's — while the attribution CSR still says `[0, 1, 2, 3]`, because it
    /// counts requests.
    ///
    /// Spending those entries as read-out rows gives member 1 row 1 and
    /// member 2 row 2, which are request 0's SECOND and THIRD drafts. Both
    /// are real distributions of the right width over the right vocabulary,
    /// so the interpreter samples them, commits them, and two conversations
    /// continue with a third one's tokens. Nothing faults; the frame is
    /// green. Only the transcripts show it, which is the same failure mode
    /// as the `(0, rows)` fallback and one batch member further along.
    #[test]
    fn a_speculative_member_does_not_shift_the_members_after_it() {
        let qo = [0, 3, 4, 5];
        let s = Sampling {
            qo_indptr: &qo,
            indices: &[0, 1, 2],
            indptr: &[0, 3, 3, 3],
        };
        let indptr = [0, 1, 2, 3];
        assert_eq!(member_rows(&indptr, &s, 0, 5), Some((0, 3)));
        assert_eq!(member_rows(&indptr, &s, 1, 5), Some((3, 4)));
        assert_eq!(member_rows(&indptr, &s, 2, 5), Some((4, 5)));
        // What the untranslated read answered, stated so a regression is
        // named rather than merely unequal.
        assert_ne!(member_rows(&indptr, &s, 1, 5), Some((1, 2)));
        assert_ne!(member_rows(&indptr, &s, 2, 5), Some((2, 3)));
    }

    /// A row named twice is one read-out row, because the read-out is built
    /// from a per-row bitmap and a bit set twice is set once.
    #[test]
    fn a_row_named_twice_is_counted_once() {
        let qo = [0, 3, 4];
        let s = Sampling {
            qo_indptr: &qo,
            indices: &[2, 2, 0],
            indptr: &[0, 3, 3],
        };
        let indptr = [0, 1, 2];
        assert_eq!(member_rows(&indptr, &s, 0, 3), Some((0, 2)));
        assert_eq!(member_rows(&indptr, &s, 1, 3), Some((2, 3)));
    }

    /// A request with no token rows contributes no read-out row: it has no
    /// last row to read, and `Readouts::samples` skips it for that reason.
    #[test]
    fn a_request_with_no_rows_contributes_none() {
        let qo = [0, 1, 1, 2];
        let s = plain(&qo);
        let indptr = [0, 1, 2, 3];
        assert_eq!(member_rows(&indptr, &s, 0, 2), Some((0, 1)));
        assert_eq!(member_rows(&indptr, &s, 1, 2), Some((1, 1)));
        assert_eq!(member_rows(&indptr, &s, 2, 2), Some((1, 2)));
    }

    /// No attribution CSR is the single-member case, and the whole read-out
    /// is that member's — the behaviour every frame used to get.
    #[test]
    fn an_absent_csr_gives_the_whole_readout_to_the_one_member() {
        let qo = decodes(7);
        let s = plain(&qo);
        assert_eq!(member_rows(&[], &s, 0, 7), Some((0, 7)));
        assert_eq!(member_rows(&[0], &s, 0, 7), Some((0, 7)));
    }

    /// A CSR THAT DOES NOT DESCRIBE THIS MEMBER IS REFUSED, NOT ANSWERED
    /// WITH ROW ZERO.
    ///
    /// This function used to fall back to `(0, rows)` for EVERY unusable
    /// CSR — absent, too short, or inverted alike — and `run_programs` fed
    /// that span to the channel interpreter, whose `base_row` is 0. So a
    /// member the frame's own table failed to place did not fault and did not
    /// stall: it sampled read-out row 0, which in a batched frame is the
    /// FIRST member's distribution, and returned another conversation's token
    /// as its own. Nothing observes that from inside the member — the logits
    /// are real, the sampler is fine, and only the two conversations put side
    /// by side show it.
    ///
    /// The relation, rather than a literal: whatever a described member
    /// answers, an undescribed one answers NOTHING, and in particular it does
    /// not answer the span the frame gave to member 0. `driver-wgpu` and
    /// `driver-vulkan` both refuse here, and so does this crate's own
    /// `envelope::member_requests` over the same table; this copy is the one
    /// that reads a distribution rather than a geometry, so it was the one
    /// where the lenient answer cost tokens.
    #[test]
    fn a_member_the_csr_does_not_place_is_refused_rather_than_given_row_zero() {
        let qo = decodes(3);
        let s = plain(&qo);
        // Three members' worth of roster, a table that places two of them.
        let short = [0, 1, 2];
        let first = member_rows(&short, &s, 0, 3).expect("member 0 is placed");
        assert!(member_rows(&short, &s, 2, 3).is_none());
        assert_ne!(member_rows(&short, &s, 2, 3), Some(first));

        // Present, long enough, and inverted at this member.
        assert!(member_rows(&[0, 2, 1], &s, 1, 3).is_none());

        // Present and reaching past the read-out the fire produced: the span
        // would run off the end of the values, which is the same claim the
        // slice check in `run_programs` makes a line later.
        assert!(member_rows(&[0, 1, 9], &s, 1, 3).is_none());
    }

    /// A CSR naming a request the fire does not have is refused, and it is
    /// the `qo_indptr` that says so — the read-out row count alone cannot,
    /// because a request past the end contributes nothing to it and the span
    /// would come back empty and in bounds.
    #[test]
    fn a_request_the_fire_does_not_have_is_refused() {
        let qo = decodes(2);
        let s = plain(&qo);
        assert!(member_rows(&[0, 1, 5], &s, 1, 2).is_none());
    }
}
