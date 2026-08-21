//! One instance's fire: the rings, the control kernels, and the sequence.
//!
//! A program names a channel by index into its instance's `channel_ids`, so a
//! session is that map from dense channel index to the driver-wide registry
//! slot whose ring holds the cells. The map is per instance; the
//! [`Rings`](super::channel::Rings) are one registry for the whole driver, so
//! two instances naming one channel share its ring — a chained decode frame
//! reads what the prefill published. The map cannot be rebuilt per fire: a slot
//! is the only thing tying a channel to the cursors that record what a previous
//! fire published. Control modules depend only on the architecture, so they are
//! compiled once and borrowed. [`Session::fire`] runs pull, readiness, bind,
//! launch, commit, push in that order — readiness is not a formality; a fire
//! that skips it reads a zeroed cell and publishes an answer computed from nothing.

use super::run::Lane;
use driver::Extents;
use driver::tensor_ir::op::IntrinsicId;

use super::channel::{ChannelShape, Rings};
use super::channel::{HostChannel, StageChannels};
use super::launch_control;
use super::run::Control;
use super::run::Prepared;
use super::runtime::Compiled;
use crate::device::{Allocator, StreamRef};
use crate::error::{Error, Result};

/// Where one bound instance's channels live in the driver's ring registry.
#[derive(Debug)]
pub struct Session {
    /// Dense channel index → registry slot, in `channel_ids` order.
    slots: Vec<u32>,
    shapes: Vec<ChannelShape>,
}

/// What one fire produced, so a caller can tell the three outcomes apart.
#[derive(Debug, PartialEq, Eq)]
pub enum Fired {
    /// The stage ran and committed.
    Committed {
        /// Cells pushed back into the host mirror. Can be fewer than produced:
        /// a host ring the engine has not drained refuses the publish, and the
        /// unconsumed device cell is retried next fire (back-pressure, lossless).
        published: usize,
    },
    /// The stage ran but its kernel declined to commit — not an error; the
    /// cursors are left so the next fire sees the same inputs.
    Declined,
    /// The inputs were not ready, so nothing launched. Opposite of `Declined`:
    /// the driver lacked an input, versus the program refusing what it got.
    NotReady,
}

impl Session {
    /// Name an instance's channels: `slots[dense]` is the registry slot for
    /// channel `dense`, `shapes[dense]` its geometry.
    ///
    /// # Errors
    ///
    /// If there are no channels, or the two lists disagree in length (which
    /// would give a channel another's cell width).
    pub fn new(slots: Vec<u32>, shapes: Vec<ChannelShape>) -> Result<Self> {
        if slots.is_empty() {
            return Err(Error::invalid(
                "ptir::session",
                "an instance with no channels",
            ));
        }
        if slots.len() != shapes.len() {
            return Err(Error::invalid(
                "ptir::session",
                format!(
                    "{} channel slot(s) and {} shape(s)",
                    slots.len(),
                    shapes.len()
                ),
            ));
        }
        Ok(Self { slots, shapes })
    }

    /// The registry slot channel `dense` lives at.
    #[must_use]
    pub fn slot(&self, dense: usize) -> Option<u32> {
        self.slots.get(dense).copied()
    }

    /// Every channel's registry slot, in the instance's own order.
    #[must_use]
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    /// This instance's dense channel indices as registry slots.
    fn global(&self, dense: &[u32]) -> Result<Vec<u32>> {
        dense
            .iter()
            .map(|&c| {
                self.slots.get(c as usize).copied().ok_or_else(|| {
                    Error::invalid("ptir::session", format!("no channel {c} in this instance"))
                })
            })
            .collect()
    }

    /// Copy every cell the host published, for the channels `sets` says the
    /// stage reads, into the device rings. Returns how many cells moved.
    ///
    /// # Errors
    ///
    /// If a channel index is out of range, a cell is the wrong width, or a copy fails.
    pub fn pull(
        &mut self,
        rings: &mut Rings,
        host: &mut [HostChannel],
        sets: &StageChannels,
        stream: StreamRef<'_>,
    ) -> Result<usize> {
        self.pull_channels(rings, host, &sets.need_full, stream)
    }

    /// [`Session::pull`], for a stated list of dense channels — the descriptor
    /// ports a host writer feeds. A seed reaches its ring at registration
    /// (`launch::ensure_sessions`), not here.
    ///
    /// # Errors
    ///
    /// As [`Session::pull`].
    pub fn pull_channels(
        &mut self,
        rings: &mut Rings,
        host: &mut [HostChannel],
        dense: &[u32],
        stream: StreamRef<'_>,
    ) -> Result<usize> {
        let mut moved = 0;
        for &c in dense {
            let index = c as usize;
            let shape = *self.shapes.get(index).ok_or_else(|| {
                Error::invalid("ptir::session", format!("no channel {c} in this instance"))
            })?;
            let global = self.slots[index] as usize;
            let Some(plane) = host.get_mut(index) else {
                return Err(Error::invalid(
                    "ptir::session",
                    format!("channel {c} has no host plane"),
                ));
            };
            // Only a plane the engine writes: the mirror shares one head/tail
            // for both directions, so taking from a plane this driver publishes
            // into re-injects the program's own output as an input.
            if !plane.engine_writes() {
                continue;
            }
            // Everything the host has, not one cell: pulling one would leave a
            // backlog that never drains.
            while let Some(wire) = plane.take() {
                let native = super::channel::wire_to_native(shape.dtype, shape.numel, &wire)
                    .map_err(|why| Error::invalid("ptir::session", why))?;
                let slot = rings.cursors(stream)?[global].tail;
                rings.seed(global, slot, &native, stream)?;
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// Copy everything the stage published back into the host mirror, consuming
    /// the device cells that landed there. Returns how many cells moved.
    ///
    /// A `chan_put` advances the ring's tail and nothing advances its head (no
    /// stage takes an output), so a reader channel's ring would fill after
    /// `capacity` fires and stall the loop unless the mirror consumes here.
    /// A refused publish is back-pressure, not a drop: the device cell stays
    /// unconsumed and the next fire's readiness declines until the engine
    /// drains, so the cell is consumed only when the publish took it.
    ///
    /// # Errors
    ///
    /// If a channel index is past the instance's, or a copy fails.
    pub fn push(
        &mut self,
        rings: &mut Rings,
        host: &mut [HostChannel],
        sets: &StageChannels,
        before: &[u32],
        stream: StreamRef<'_>,
    ) -> Result<usize> {
        let after = rings.cursors(stream)?;
        let mut moved = 0;
        for &c in &sets.put {
            let index = c as usize;
            let shape = *self.shapes.get(index).ok_or_else(|| {
                Error::invalid("ptir::session", format!("no channel {c} in this instance"))
            })?;
            let global = self.slots[index] as usize;
            let ring = shape.ring()?;
            let (was, now) = (
                *before.get(index).unwrap_or(&0),
                after.get(global).map_or(0, |c| c.tail),
            );
            // Tails are ring-modular, so count the forward distance, not a
            // subtraction — a wrap would publish a huge `u32` otherwise.
            let produced = (now + ring - was % ring) % ring;
            for step in 0..produced {
                let slot = (was + step) % ring;
                let native = rings.read_cell(global, slot, stream)?;
                let wire = super::channel::native_to_wire(shape.dtype, shape.numel, &native)
                    .map_err(|why| Error::invalid("ptir::session", why))?;
                let Some(plane) = host.get_mut(index) else {
                    return Err(Error::invalid(
                        "ptir::session",
                        format!("channel {c} has no host plane"),
                    ));
                };
                // Only a plane the engine reads — the mirror of `pull`'s rule;
                // else the driver fills a ring nobody drains and reads it back.
                if !plane.engine_reads() {
                    continue;
                }
                // WHAT THE PROGRAM ACTUALLY PUBLISHED. A sampling program that
                // commits and publishes the wrong cell is invisible from both
                // ends: the driver saw a successful fire and the engine saw a
                // token. `PIE_TRACE_VALUES=1` prints the wire bytes.
                if std::env::var_os("PIE_TRACE_VALUES").is_some() {
                    eprintln!(
                        "[put] channel={c} dtype={:?} numel={} wire={:?}",
                        shape.dtype,
                        shape.numel,
                        &wire[..wire.len().min(32)]
                    );
                }
                if !plane.publish(&wire) {
                    // The engine is behind; leave the cell so the next fire's
                    // readiness declines and retries — lossless.
                    break;
                }
                moved += 1;
                // Not for a channel the stage also takes: the commit kernel
                // advanced that head already; a second advance skips a cell.
                if !sets.taken.contains(&c) {
                    rings.consume_front(global, stream)?;
                }
            }
        }
        Ok(moved)
    }

    /// Run every stage of a compiled program, end to end, as ONE fire.
    ///
    /// `logits` is the readout buffer and `vocab` its row width; `row_of` says
    /// which row a lane samples. One [`Extents`] per lane: a grouped fire's
    /// members share a plan and cursors, differing only in how much each
    /// submitted. The buffer is the model fire's and is handed in, so a program
    /// that reads no intrinsic may pass a null base.
    ///
    /// # One `Prepared` per stage, one commit for the program
    ///
    /// `Prepared` is built from ONE plan — its value types size the scratch,
    /// its bindings index the lane table, its op count strides the params — so
    /// each stage gets its own. Nothing flows between them in scratch: stages
    /// are separate programs joined only by channels, and by the sinks a
    /// prologue configures the forward with.
    ///
    /// The COMMIT is the program's, not a stage's, because the channel cursors
    /// advance per fire. Three things follow, and all three are why this cannot
    /// be a loop over `fire`:
    ///
    /// * The readiness gate and the commit read the PROGRAM's channel sets
    ///   (`program_channels`), so a channel one stage reads and another puts is
    ///   gated once on its first touch anywhere.
    /// * Every stage's `Prepared` resolves its cell addresses from the same
    ///   cursors, which is correct precisely because nothing advances them
    ///   until the commit at the end.
    /// * A stage that refuses (clears its commit slot) refuses the whole fire.
    ///   Every stage still launches — the dummy run, so a blocked fire costs
    ///   what a running one does — and then nothing moves.
    ///
    /// What this REPLACES is a guard that refused any program with more than
    /// one stage. It also took the caller's chosen plan as an argument, which
    /// no longer means anything: a program's stages carry their own plans and
    /// the loop pairs them by index. See `stages_and_plans_agree`.
    ///
    /// # Errors
    ///
    /// If a launching stage has no plan, if the channel sets cannot be derived,
    /// if a copy fails, or if a region refuses to launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fire(
        &mut self,
        rings: &mut Rings,
        compiled: &Compiled,
        control: &Control,
        host: &mut [HostChannel],
        logits: (u64, u32, u32),
        row_of: impl Fn(u32) -> u32,
        lane_extents: &[Extents],
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<Fired> {
        stages_and_plans_agree(compiled)?;
        let sets = super::channel::program_channels(&compiled.plans)
            .map_err(|why| Error::invalid("ptir::session", why))?;

        self.pull(rings, host, &sets, stream)?;
        // The control kernels index the registry's flat arrays, so map the
        // dense channel numbers to slots before gating.
        let (need_full, need_empty) = (
            self.global(&sets.need_full)?,
            self.global(&sets.need_empty)?,
        );
        if !launch_control::readiness(control, rings, &need_full, &need_empty, alloc, stream)? {
            // Name which channel and direction, not a bare `NotReady`. The env
            // check is inline rather than `fire::launch::sg_trace` because that
            // module is `feature = "abi"` and this one is not.
            if std::env::var_os("PIE_CUDA_TRACE_SUPERGRAPH").is_some() {
                let cursors = rings.cursors(stream).unwrap_or_default();
                let say = |what: &str, dense: &[u32]| -> String {
                    dense
                        .iter()
                        .map(|&c| {
                            let g = self.slots.get(c as usize).copied().unwrap_or(u32::MAX);
                            let ring = self.shapes.get(c as usize).and_then(|s| s.ring().ok());
                            match cursors.get(g as usize) {
                                Some(k) => format!(
                                    " {what} chan {c} (slot {g}) head={} tail={} ring={:?} \
                                     full[head]={}",
                                    k.head,
                                    k.tail,
                                    ring,
                                    u8::from(k.is_readable())
                                ),
                                None => format!(" {what} chan {c} (slot {g}) unringed"),
                            }
                        })
                        .collect()
                };
                eprintln!(
                    "[sg]   ptir readiness refused:{}{}",
                    say("needs-full", &sets.need_full),
                    say("needs-room", &sets.need_empty)
                );
            }
            return Ok(Fired::NotReady);
        }

        // The tails before the fire: `push` counts what this fire added.
        // Indexed densely, which is how `push` reads it back.
        let cursors = rings.cursors(stream)?;
        let before: Vec<u32> = self
            .slots
            .iter()
            .map(|&g| cursors.get(g as usize).map_or(0, |c| c.tail))
            .collect();

        // One lane per member: a grouped fire's lanes are different instances,
        // each with its own dense→slot map over the one shared registry.
        let members: Vec<Lane<'_>> = lane_extents
            .iter()
            .map(|&extents| Lane {
                rings,
                slots: &self.slots,
                extents,
            })
            .collect();
        // WHAT THE PROGRAM IS -- printed per stage inside the loop below, since
        // "the plan" is now one of several and a sampler that publishes a
        // constant is indistinguishable from one that never read the logits.
        // WHICH REGIONS ACTUALLY LAUNCH. A stage's region list is what the
        // compile plane kept, not what the plan declared, so a region dropped
        // there is invisible from every other vantage point: the values it
        // would have written read back as the zeros `Prepared::build` wrote,
        // and nothing faults. `.wiki/migration.md` §11.21.
        let trace = std::env::var_os("PIE_TRACE_VALUES").is_some();
        let (base, vocab, row_stride) = logits;
        // Every stage's verdict, ANDed: one refusal refuses the fire. Starts
        // true so a program whose stages all launch nothing still commits --
        // the forward ran, the sinks configured it, and the cursors should
        // advance for the channels its ops named.
        let mut committed = true;
        for (index, stage) in compiled.stages.iter().enumerate() {
            // Its OWN plan. Reaching for the caller's would size stage 1's
            // scratch from stage 0's value types and launch into it without
            // faulting, which is the thing `one_prepared_only` used to prevent
            // by refusing the shape outright.
            let stage_plan = compiled.plans.get(index).ok_or_else(|| {
                Error::invalid(
                    "ptir::session",
                    format!("stage {index} has regions to launch and no plan to prepare from"),
                )
            })?;
            // A stage with nothing to launch needs no buffers. It is not an
            // empty case either: the adapter prologue is exactly this, its one
            // region being a `lora` sink the fire reads out of the plan.
            if stage.regions.is_empty() {
                continue;
            }
            if trace {
                eprintln!(
                    "[plan] stage={index} flags={:#x} ops={} values={} error={:?}",
                    stage_plan.flags,
                    stage_plan.ops.len(),
                    stage_plan.value_types.len(),
                    stage_plan.error
                );
                for (i, op) in stage_plan.ops.iter().enumerate() {
                    eprintln!(
                        "[op] {i} code={:#x} intrinsic={} results={}",
                        op.code, op.intrinsic, op.result_count
                    );
                }
            }
            let mut prepared = Prepared::build(alloc, stage_plan, &members, stream)?;
            if base != 0 {
                prepared.bind_intrinsic(
                    IntrinsicId::Logits,
                    base,
                    // Raw bf16, which is what the fire writes into the pin:
                    // `deliver_logits` reads the buffer as bf16 and widens
                    // `(bits as u32) << 16`, matching `m1_intrinsic_row_base`
                    // for mode 1. F32 mode (wire byte 0) would stride four
                    // bytes per logit through a two-byte buffer and select
                    // different tokens.
                    crate::program::params::INTRINSIC_STORAGE_RAW_BF16,
                    vocab,
                    row_stride,
                    &row_of,
                    stream,
                )?;
            }
            for region in stage.regions.iter() {
                if trace {
                    eprintln!(
                        "[region] stage={:#x} region={} entry={}",
                        stage.signature_hash,
                        region.region_index,
                        region.module.entry_name()
                    );
                }
                prepared.launch_region(region, stream)?;
            }
            // Per stage, because the next stage's regions may read a channel
            // cell this one wrote through the shared rings.
            stream.synchronize()?;
            if trace {
                prepared.trace_scratch(0, stream)?;
            }
            committed &= prepared.committed(stream)?;
        }

        let (taken, put) = (self.global(&sets.taken)?, self.global(&sets.put)?);
        launch_control::commit(control, rings, &taken, &put, committed, alloc, stream)?;
        stream.synchronize()?;
        if !committed {
            return Ok(Fired::Declined);
        }
        let published = self.push(rings, host, &sets, &before, stream)?;
        Ok(Fired::Committed { published })
    }
}

/// The pairing every stage's `Prepared` depends on, named so a test can reach
/// it without a GPU.
///
/// `Session::fire` builds one `Prepared` per launching stage from
/// `compiled.plans[index]`, so the two arrays have to be parallel and each
/// launching stage has to be the one its plan describes. A plan that is off by
/// one sizes a stage's scratch from someone else's value types, indexes the
/// lane table with someone else's bindings, and strides the params by someone
/// else's op count — none of which faults on the device.
///
/// # What this replaces
///
/// A guard that refused any program with more than one COMPILED STAGE, on the
/// grounds that the fire prepared exactly one and every region indexed it. That
/// was true when it was written and it is what refused every adapter program:
/// a `lora` prologue plus a sampling epilogue is two stages, and `lora-probe`
/// did not merely fail on it — the refusal reached the guest as a fire that
/// never completed, so it hung at one token with the forward pass running to
/// completion underneath.
fn stages_and_plans_agree(compiled: &Compiled) -> Result<()> {
    stage_plans_are_parallel(
        &compiled
            .stages
            .iter()
            .map(|stage| (stage.signature_hash, !stage.regions.is_empty()))
            .collect::<Vec<_>>(),
        &compiled
            .plans
            .iter()
            .map(|plan| plan.signature_hash)
            .collect::<Vec<_>>(),
    )
}

/// The decision [`stages_and_plans_agree`] makes, over the facts it reads.
///
/// Projected to signatures and "launches?" first, because a `Region` owns a
/// loaded cubin and a precondition nobody can test on the host is a
/// precondition nobody tests.
fn stage_plans_are_parallel(stages: &[(u64, bool)], plans: &[u64]) -> Result<()> {
    if stages.len() != plans.len() {
        return Err(Error::invalid(
            "ptir::session",
            format!(
                "this program has {} compiled stages and {} plans: the fire pairs \
                 them by index to prepare each stage's own scratch, so it cannot \
                 tell which plan describes which stage",
                stages.len(),
                plans.len()
            ),
        ));
    }
    for (index, (&(signature, launches), &plan)) in stages.iter().zip(plans).enumerate() {
        // Only a LAUNCHING stage is prepared, so only a launching stage's
        // signature has to match. A stage with no regions -- the adapter
        // prologue, whose one region is a sink read out of the plan rather than
        // compiled -- is skipped by the fire and carries nothing to compare.
        if launches && signature != plan {
            return Err(Error::invalid(
                "ptir::session",
                format!(
                    "stage {index} has regions to launch and the plan at that index \
                     describes a different stage: its regions would index scratch, \
                     descriptors and a channel table sized for someone else"
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::runtime::{Compiled, Stage};

    /// The served shape, and the one that used to be refused outright: an
    /// adapter prologue that launches nothing plus a sampling epilogue that
    /// does. Two stages, two plans, one fire.
    #[test]
    fn an_adapter_prologue_and_a_sampling_epilogue_are_one_fire() {
        super::stage_plans_are_parallel(&[(0xa11, false), (0xb22, true)], &[0xa11, 0xb22])
            .expect("the plans are parallel and the launching stage is its own");
    }

    /// Two stages that BOTH launch is now served too — each gets its own
    /// `Prepared` and they share the commit.
    #[test]
    fn two_launching_stages_each_get_their_own_plan() {
        super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xb22])
            .expect("both launch, both are prepared from their own plan");
    }

    /// A plan array that is not parallel is refused before anything is
    /// prepared: the fire pairs by index and has nothing else to pair on.
    #[test]
    fn stages_without_a_plan_apiece_are_refused() {
        let refusal = super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11])
            .expect_err("two stages, one plan");
        let text = format!("{refusal:?}");
        assert!(text.contains("2 compiled stages"), "names how many: {text}");
        assert!(
            text.contains("1 plans"),
            "and how many of the other: {text}"
        );
    }

    /// And a launching stage paired with somebody else's plan is refused,
    /// which is the failure nothing on the device would fault on.
    #[test]
    fn a_launching_stage_paired_with_the_wrong_plan_is_refused() {
        let refusal =
            super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xdead])
                .expect_err("stage 1 is not what plan 1 describes");
        let text = format!("{refusal:?}");
        assert!(text.contains("stage 1"), "names which: {text}");
        assert!(
            text.contains("scratch") || text.contains("descriptors"),
            "and what would go wrong: {text}"
        );
    }

    /// A stage that launches nothing is never prepared, so its signature is
    /// never compared -- and demanding a match would refuse the adapter
    /// prologue for a plan the fire does not read.
    #[test]
    fn a_stage_that_launches_nothing_is_not_compared() {
        super::stage_plans_are_parallel(&[(0xa11, false)], &[0xdead])
            .expect("nothing is prepared from it");
    }

    /// The projection the guard actually reads, wired to the real types once,
    /// so the two halves cannot drift: an empty region list is "launches
    /// nothing" and a plan's signature is what it describes.
    #[test]
    fn the_projection_reads_emptiness_and_the_plans_signature() {
        let compiled = Compiled {
            stages: Arc::new(vec![
                Stage {
                    signature_hash: 0xa11,
                    regions: Arc::new(Vec::new()),
                },
                Stage {
                    signature_hash: 0xb22,
                    regions: Arc::new(Vec::new()),
                },
            ]),
            plans: Arc::new(vec![
                driver::driver_api::plan::LaunchStagePlan {
                    signature_hash: 0xa11,
                    ..Default::default()
                },
                driver::driver_api::plan::LaunchStagePlan {
                    signature_hash: 0xb22,
                    ..Default::default()
                },
            ]),
            kinds: Arc::new(vec![
                super::super::runtime::stage_kind::PROLOGUE,
                super::super::runtime::stage_kind::EPILOGUE,
            ]),
        };
        super::stages_and_plans_agree(&compiled).expect("parallel and paired");

        let crossed = Compiled {
            plans: Arc::new(vec![driver::driver_api::plan::LaunchStagePlan::default()]),
            ..compiled
        };
        super::stages_and_plans_agree(&crossed).expect_err("two stages, one plan");
    }
}
