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
/// re-posts, and a frame that could never fit is one it drops. Both are
/// answers about the pool rather than failures of the driver, and a `Result`
/// that folded them into `Err` would make the caller match on prose to tell a
/// full pool from a broken device.
#[derive(Debug)]
pub enum Launched {
    /// Every step was submitted, waited for, and its read-out handed to the
    /// programs bound to it.
    ///
    /// `faults` are the instances whose channel-plane pass faulted. A fault
    /// poisons the one instance and does not fail the frame — the other
    /// requests batched with it did nothing wrong — so they are REPORTED
    /// rather than raised. They are returned rather than logged because
    /// choosing a logging backend is the caller's business, not a driver's.
    Ran {
        /// `(instance id, why)` for each program that faulted this frame.
        faults: Vec<(u64, String)>,
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
        // Before anything is encoded, and without side effects, which is what
        // lets the caller re-post: a frame that took an arena and then failed
        // to admit would have to be undone.
        //
        // Against what the pool RESERVED, not what it currently holds. The
        // trim task gives pages back down to a high-water mark, and a frame
        // past that mark is one the pool can serve after growing -- calling it
        // impossible would have the scheduler drop work it had correctly
        // admitted, permanently, because the pool had been idle.
        if !self.need_pool("launch")?.admits(frame.required_kv_pages) {
            // Impossible rather than Exhausted when no eviction could
            // make room — the demand exceeds the physical pool, so
            // waiting is waiting for something that cannot happen.
            return Ok(Launched::Impossible);
        }

        // ── Grow to cover the pages this frame names. ──
        //
        // The trim task only ever UNMAPS: it passes an empty `map_ranges` and
        // a target, and takes memory back one tick after the high-water mark
        // that justified it rose. That tick is after the frame that raised
        // it, so a pool left at the trimmed size would refuse this frame at
        // `translate` for a page the scheduler was right to hand out.
        //
        // The highest page NAMED, not the count required: a frame's pages are
        // physical indices the scheduler chose anywhere in the pool, so a
        // frame needing two pages can name page 900. Idempotent when the pool
        // is already big enough, which is every frame in the steady state.
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

        let (Some(model), Some(pool)) = (self.model.as_ref(), self.pool.as_ref()) else {
            return Err(Error::Unserved {
                what: "launch",
                message: "no checkpoint is loaded. `load_model` stages the tensors every \
                          fire's operands address."
                    .to_string(),
            });
        };

        // ── The page translation, checked per lane. ──
        //
        // A page past the pool addresses another layer's memory and attention
        // would read it without complaint, so this is a refusal and not a
        // clamp.
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

        // THE ROW that was identified at load, and what that load OBSERVED.
        //
        // Both `Copy`, where this used to `clone()` two fact structs per
        // frame — and the clone was the smaller cost. What it cloned was
        // twenty-nine model facts this driver had rebuilt for itself from
        // nine tensor probes, so every fire ran a text selected from the
        // driver's reading of a checkpoint rather than from the row that
        // checkpoint had already been matched to.
        let (row, binding) = self.text_row.ok_or_else(|| Error::Unserved {
            what: "launch",
            message: "no row. `load_model` identifies one from the tensors and \
                      records what its weights arrived as; a fire before a load \
                      has nothing to trace."
                .to_string(),
        })?;
        // The fire's geometry, off the SAME projection the pool was sized
        // from — read once here rather than per step, because a deployment
        // does not change between the steps of a frame.
        //
        // `head_dim_alloc()` and not `head_dim`: phi-3's heads are 96 wide
        // and run on the 128-wide kernel, so a dispatch that states the
        // checkpoint's width addresses two thirds of the buffer the pool
        // allocated. This is the value `DecodeGeometry::head_dim` carried and
        // the value the deleted facts passed on, taken from the row's own
        // rounding rule rather than from a re-derivation of it.
        //
        // The two mixture counts are zero because they can only be zero:
        // `geometry_from_deployment` REFUSES a routed mixture outright — a
        // `Deployment` states no top-k, and a mixture fired at the wrong one
        // routes each token to almost the right experts and returns fluent
        // nonsense — so a load that reached this line is a dense stack. Zeros
        // are what `is_moe` reads as "not that", stated rather than inherited.
        let geometry = {
            let d = self.deployment.as_ref().ok_or_else(|| Error::Unserved {
                what: "launch",
                message: "no deployment. The row projects one at load and every \
                          consumer reads that projection."
                    .to_string(),
            })?;
            crate::lowering::dispatch::Geometry {
                q_heads: d.shape.q_heads,
                kv_heads: d.shape.kv_heads,
                head_dim: d.shape.head_dim_alloc(),
                rotary_dims: d.shape.head_dim_alloc(),
                n_experts: 0,
                experts_per_token: 0,
                // What the BYTES arrived in, which is the one pair of facts a
                // catalog row cannot state: `mlx-community` publishes one
                // model at g64/b4 and at g128/b8 and the two pack to
                // identical extents, so `Loaded::affine_point` asks the
                // tensors and refuses a checkpoint that answers twice.
                group: binding.quant_group,
                bits: binding.quant_bits,
            }
        };
        let named = std::collections::HashMap::new();

        // ONE timeline for the whole frame, so a step is QUEUED while the
        // previous one runs rather than after it finishes.
        //
        // Every step used to build its own `Stepper` and end in a wait, which
        // made the frame N submissions and N full GPU stalls. `Stepper` is
        // bounded internally -- it waits for the step two back, because there
        // are two command allocators -- so this is one fire in flight while
        // one runs, which is the shape run-ahead wants
        // (`.wiki/new-driver/next.md`, priority 1).
        //
        // Command buffers committed to one queue execute in submission order,
        // which is what makes this SAFE for steps that depend on each other:
        // step n+1 reads the KV step n appended.
        //
        // Still per-FRAME rather than per-driver, and the reason is a
        // lifetime: `Stepper<'ctx>` borrows the `Context` this struct owns, so
        // holding one across `launch` calls is a self-reference. Making it own
        // an `Arc<Context>` is what across-frame run-ahead needs next.
        let mut in_flight: Vec<(&driver_api::StepSubmission, _)> = Vec::new();

        for step in &frame.steps {
            let s = crate::lowering::frame::Step {
                token_ids: &step.plan.token_ids,
                qo_indptr: &step.plan.qo_indptr,
                region_row_indptr: &step.region_row_indptr,
                region_sig: &step.region_sig,
                region_k: &step.region_k,
                sampling_indices: &step.plan.sampling_indices,
                sampling_indptr: &step.plan.sampling_indptr,
            };
            let class = crate::lowering::frame::fire_class(&s);
            // THE ROW'S OWN TEXT for this fire class, and the driver's only
            // door to one.
            //
            // What this replaced took an architecture STRING and two fact
            // structs — `plan_for(arch, class, &facts, &metal)` — and matched
            // the string against an eleven-entry table to pick a text. That
            // table was a third dispatch key for an identity `catalog::identify`
            // had already settled from the tensors, and it disagreed with the
            // load path about gemma-4 in the two directions a second list
            // always eventually disagrees.
            //
            // The refusal is carried rather than summarized, the way
            // `driver-cuda/src/fire/launch.rs` carries it: `Error: From<Refusal>`
            // maps `Unsupported` and `Malformed` onto this driver's own
            // `Unserved`, so what reaches the operator is the row's sentence.
            // In practice a refusal here is unreachable — `load_model` asked
            // the same row the same question before it staged a byte — and it
            // is propagated anyway, because "unreachable" is a claim about
            // today's `serves` and this is the place that would find out it
            // had stopped being true.
            //
            // CACHED, by the fire shape. Both halves of this -- the plan and
            // the lowering -- are pure functions of things that do not change
            // between the steps of a generation, and deriving them per token
            // cost 0.21 ms and 0.60 ms of a 4.9 ms decode. The closure is why
            // the refusal above is still propagated on a miss and not paid
            // for on a hit.
            let lowered = self
                .lowerings
                .for_step(class, &s, || {
                    crate::model::binding::text(row, class, &binding)
                })
                .map_err(|why| match why {
                    crate::lowering::cached::Miss::Plan(refusal) => Error::from(refusal),
                    crate::lowering::cached::Miss::Lower(why) => Error::Program {
                        message: format!("step did not lower: {why:?}"),
                    },
                })?;
            // The gather's index list, in the fire's own numbering. Taken
            // beside the lowering because it comes from the same read of the
            // same table -- the wire's numbers are request-local and the
            // gather's must not be.
            let sampled =
                crate::lowering::frame::sampled_rows(&s).map_err(|why| Error::Program {
                    message: format!("step did not lower: {why:?}"),
                })?;

            // Every CSR invariant, checked BEFORE the pool is touched.
            //
            // The derivation below used `unwrap_or(0)` three times, and the
            // third one was the defect: a short or mis-sized CSR resolved a
            // token's physical KV page to **0**, which belongs to some other
            // request, and the fire wrote this request's keys over that
            // request's cache. Nothing faults and the damage lands on a
            // request that did nothing wrong.
            //
            // There is no safe fallback page, so the only correct answer is to
            // refuse the frame — and refusing has to happen here, before
            // anything is staged, which is the `decide, then move` rule
            // `serve/transfer.rs` records the cost of breaking.
            step.plan.validate_geometry().map_err(|e| Error::Unserved {
                what: "launch",
                message: format!("this frame's geometry: {e}"),
            })?;
            step.plan
                .validate_kv_writes(pool.shape().page_size)
                .map_err(|e| Error::Unserved {
                    what: "launch",
                    message: format!("this frame's KV writes: {e}"),
                })?;
            // Where the paged append writes each token: its physical page and
            // the row inside it. Driver arithmetic over a driver allocation --
            // the frame states a POSITION in a sequence and a page table, and
            // this normalizes the pair.
            //
            // Every lookup is infallible now: `validate_kv_writes` has already
            // proved each token's virtual page sits inside its own request's
            // span, so an `expect` here states a checked fact rather than
            // papering over an unchecked one.
            let (w_page, w_off) = {
                let page = pool.shape().page_size.max(1);
                let req = step.plan.req_of_token();
                let (mut pages, mut offs) = (Vec::new(), Vec::new());
                for (t, &pos) in step.plan.position_ids.iter().enumerate() {
                    let r = req[t] as usize;
                    let base = step.plan.kv_page_indptr[r] as usize;
                    let virt = base + (pos / page) as usize;
                    pages.push(step.plan.kv_page_indices[virt]);
                    offs.push(pos % page);
                }
                (pages, offs)
            };
            let req = step.plan.req_of_token();
            // The fire's own tables, staged into one device region. The row
            // names which a slot wants and this answers — the driver never
            // reads what a table MEANS, only where the frame put it.
            //
            // `i32` throughout: the shader reads some as `uint` and some as
            // `uchar`, and a `u32` written little-endian is the same first
            // byte. The narrowing is the kernel's and the width is the
            // frame's, which is the direction that is safe.
            let staged = crate::bind::tables::stage(
                &self.context,
                &self.scratch,
                crate::bind::tables::Frame {
                    token_ids: &step.plan.token_ids,
                    position_ids: &step.plan.position_ids,
                    req_of_token: &req,
                    kv_page_indices: &step.plan.kv_page_indices,
                    kv_page_indptr: &step.plan.kv_page_indptr,
                    kv_write_page: &w_page,
                    kv_write_offset: &w_off,
                    rope_frequencies: &self.inv_freq,
                    // The FIRE's rows, translated from the request-local
                    // numbering the wire uses. See `sampled_rows`.
                    sampling_indices: &sampled,
                },
            )?;
            // The fire's tables, and the stand-in for an operand that
            // addresses NOTHING -- `dispatch::bind` answers an unfilled slot
            // with address zero, which `encode` binds happily and a recorded
            // command cannot. The tables region serves as that stand-in: it
            // is real, resident, and no statement writes through a slot it
            // did not fill.
            self.regions.add(staged.region());
            self.regions.set_null(staged.region());
            let tables = |which| staged.at(which);

            let names = crate::lowering::resolve::Names::mlx();
            // The KV pages a statement's state reference resolves through. A
            // closure, because the map is portable and the pool is not.
            let pages = |layer: u16, values: bool| {
                pool.layer(u32::from(layer)).map(|l| {
                    let h = if values { &l.v } else { &l.k };
                    crate::lowering::executor::Slice {
                        address: h.gpu_address(),
                        // THIS layer's, not the pool's: gemma-4's
                        // full-attention layers hold a different page size
                        // from its sliding ones, and a slice length that
                        // over-states the region is one an attention reads
                        // past the end of.
                        bytes: pool.shape().layer_bytes_at(u32::from(layer)),
                    }
                })
            };
            let mut store = crate::lowering::resolve::Store::new(names, &model.tensors, &named)
                .with_kv(&pages)
                .with_fire(&tables)
                // The shape the pool was allocated at, which is where the
                // attention kernels' strides come from. A store without it
                // answers zero, and a zero seq stride is every step of the
                // scan reading the same token.
                .with_pool(pool.shape());
            let mut machine = crate::fire::run::Machine {
                context: &self.context,
                compiler: &self.compiler,
                pipelines: &mut self.pipelines,
                stepper: &mut self.stepper,
                scratch: &self.scratch,
                regions: &mut self.regions,
                recordings: Some(&mut self.recordings),
            };
            let fire = crate::fire::run::submit(&mut machine, &lowered, geometry, &mut store)
                .map_err(|e| {
                    // A fire that could not bind names them all, because a
                    // checkpoint missing one tensor is usually missing a
                    // family of them and stopping at the first costs a round
                    // trip each.
                    let missed = store.missed();
                    if missed.is_empty() {
                        e
                    } else {
                        Error::Unserved {
                            what: "launch",
                            message: format!("{e}; unresolved names: {missed:?}"),
                        }
                    }
                })?;
            // Committed, not waited for. `lowered` is dropped at the end of
            // this iteration, so the read-out's shape is carried forward with
            // the fire rather than looked up again.
            //
            // `staged` travels with it, and that is not tidiness. A fire's
            // tables are a LEASE from `self.scratch`, returned to the pool
            // when the value drops -- and a returned region is one the next
            // fire may be handed. Dropped here, the next step would stage its
            // token ids over the ones a running fire is still reading. Under
            // the old bare `Allocation::new` the failure was the mirror of
            // that: the region was never returned at ALL, so this loop leaked
            // one tables region per step forever, and the leak was the only
            // thing making the lifetime look right.
            // `fire::run::InFlight` already carries the argument table and
            // the scalars for exactly this reason ("held for the GPU, not for
            // the caller"); the tables are staged out here, so this is where
            // they have to be held.
            in_flight.push((step, (fire, lowered.readout, staged)));
        }

        // ── The read-outs, and the channel plane over them. ──
        //
        // After the whole frame is committed, in submission order. Reading an
        // arena before its fire retires is reading whatever the last fire left
        // there, which is a plausible tensor and the wrong one.
        let mut faults = Vec::new();
        for (step, (fire, readout, _tables)) in &in_flight {
            self.stepper.wait_for(fire.value)?;
            // What the fire COMPUTED, handed to the programs bound to this
            // frame. Until this landed the seam ran every launch and dropped
            // the arena, so a green frame and a frame that computed the wrong
            // thing were the same observation — `pipeline::step` had no
            // production caller at all, and the interpreter was exercised
            // only by tests that built their own inputs.
            let logits = read_logits(&fire.arena, *readout);
            run_programs(
                &mut self.registry,
                &frame.instance_ids,
                step,
                logits.as_ref(),
                &mut faults,
            )?;
        }
        Ok(Launched::Ran { faults })
    }
}

/// Run the channel-plane pass for every program batched into one step.
///
/// One instance per roster row, in sub-batch order, each over ITS OWN rows of
/// the read-out: the fire produced one distribution per request and the
/// members of a batch are those requests, so member `p` reads
/// `program_row_indptr[p]..[p+1]` and nothing else.
///
/// A blocked pass is not an error. Readiness is the program's own gate and
/// missing it means the fire did not happen for that member — the interpreter
/// changed nothing, and the caller re-posts. A FAULT is also not an error
/// here, for a different reason: it poisons the one instance that faulted, and
/// failing the whole frame would take down every other request batched with it
/// for a fault that is one program's. Faults are appended to `faults` for the
/// caller to report.
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
        // THIS member's rows of the read-out, and nothing else.
        //
        // Every instance in the frame used to be handed the whole logits
        // buffer, and `bind_intrinsic` reads it from `base_row = 0` — so
        // in an M>1 frame every request sampled the FIRST request's
        // distribution and returned its token. One fire, N requests, one
        // answer repeated. Nothing faults, and a single-request frame
        // (which is what most tests build) cannot tell the difference.
        //
        // `program_row_indptr` is the mapping and it was already here:
        // member `p` owns wire request rows `[indptr[p], indptr[p+1])`.
        // Slicing rather than passing an offset keeps `base_row = 0`
        // TRUE for each member instead of making it a parameter every
        // caller could forget — the interpreter's view is its own rows,
        // so there is no row it could reach that is not its.
        let inputs = match logits {
            None => crate::channel::PassInputs::none(),
            Some((values, rows, vocab)) => {
                let (start, end) = member_rows(&step.program_row_indptr, member, *rows);
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
/// # Why a copy, and why widening
///
/// The interpreter's `PassInputs` wants `&[f32]`, and the metal read-out is
/// **bf16**: `affine_qmv_fast` writes bf16 whatever the text declares, which
/// is a defect the reference gate found by reading a vocabulary that was
/// exactly half zeros. So the bytes have to be reinterpreted anyway, and a
/// widening reinterpretation is a copy. The alternative — teaching the
/// interpreter a dtype — buys nothing while there is one read-out format.
///
/// bf16 → f32 is exact: the low sixteen bits are zero and every bf16 is an
/// f32. Nothing is lost here, and nothing is gained either — the precision
/// was lost in the kernel.
fn read_logits(
    arena: &crate::device::Handle,
    readout: Option<model_compiler::lower::Readout>,
) -> Option<(Vec<f32>, u32, u32)> {
    let r = readout?;
    let span = r.rows as usize * r.vocab as usize * r.bytes as usize;
    if r.at + span > arena.len() as usize {
        return None;
    }
    // SAFETY: the arena is `StorageModeShared`, so its contents are host
    // addressable, and every launch encoded against it has completed — the
    // caller waits on the fire (`stepper.wait_for`) before reading, which is
    // the loop just above this. The wait is what makes the read valid, so a
    // caller that stops waiting has to stop calling this.
    let raw = unsafe {
        std::slice::from_raw_parts(arena.contents().cast::<u8>().as_ptr().add(r.at), span)
    };
    let values = match r.bytes {
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
    Some((values, r.rows, r.vocab))
}

/// Which rows of a fire's read-out belong to batch member `member`.
///
/// `program_row_indptr` is the frame's own attribution CSR — member `p` owns
/// wire request rows `[indptr[p], indptr[p+1])` — and an empty one is the
/// single-member case, where the whole read-out is that member's.
///
/// Split out from `run_programs` so the M>1 case can be held to a number. It
/// was wrong in a way no single-instance test could see: every member was
/// handed the WHOLE buffer and `bind_intrinsic` reads from `base_row = 0`, so
/// each request in a batched frame sampled the first request's distribution.
fn member_rows(program_row_indptr: &[u32], member: usize, rows: u32) -> (u32, u32) {
    match (
        program_row_indptr.get(member),
        program_row_indptr.get(member + 1),
    ) {
        (Some(&s), Some(&e)) if e >= s => (s, e),
        _ => (0, rows),
    }
}

#[cfg(test)]
mod readout_rows {
    use super::member_rows;

    /// Three requests batched into one fire, one read-out row each.
    ///
    /// The defect this pins: every member used to get `(0, 3)`, so all three
    /// sampled row 0 and returned the same token. One fire, three requests,
    /// one answer repeated — and nothing faults.
    #[test]
    fn each_member_of_a_batched_frame_reads_its_own_row() {
        let indptr = [0, 1, 2, 3];
        assert_eq!(member_rows(&indptr, 0, 3), (0, 1));
        assert_eq!(member_rows(&indptr, 1, 3), (1, 2));
        assert_eq!(member_rows(&indptr, 2, 3), (2, 3));
    }

    /// A member may own several rows — a speculative fire reads out more than
    /// one row per request — and the span is the CSR's, not one row.
    #[test]
    fn a_member_that_owns_several_rows_gets_all_of_them() {
        let indptr = [0, 4, 5];
        assert_eq!(member_rows(&indptr, 0, 5), (0, 4));
        assert_eq!(member_rows(&indptr, 1, 5), (4, 5));
    }

    /// No attribution CSR is the single-member case, and the whole read-out
    /// is that member's — the behaviour every frame used to get.
    #[test]
    fn an_absent_csr_gives_the_whole_readout_to_the_one_member() {
        assert_eq!(member_rows(&[], 0, 7), (0, 7));
        // A CSR too short for this member is the same answer rather than a
        // panic: it is a frame the scheduler built inconsistently, and the
        // row-range check in `run_programs` is what refuses it.
        assert_eq!(member_rows(&[0, 1], 5, 7), (0, 7));
    }
}
