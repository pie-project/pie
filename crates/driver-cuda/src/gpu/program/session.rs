//! One instance's fire: the rings, the control kernels, and the sequence.
//!
//! Every piece of firing a PTIR program existed before this module and
//! nothing put them in order. `Runtime::compile` turns a plan into cubins,
//! `Prepared::build` lays out a stage's device state, `launch_control`
//! gates on the cursors, and [`super::bridge`] moves values between the
//! host mirror the engine polls and the device rings the kernels read.
//! What was missing is the thing that holds them for the length of an
//! instance and runs them in the one order that works.
//!
//! # What a session owns, and why it is per instance
//!
//! The rings are sized from the instance's OWN channels, in the order its
//! `channel_ids` lists them, because a program names a channel by index
//! into that list. So they cannot be shared and cannot be rebuilt per
//! fire — rebuilding would zero the cursors, and a cursor is the only
//! record of what a previous fire published.
//!
//! The control modules are the opposite: they depend on the architecture
//! and nothing else, so they are compiled once and shared. They are
//! borrowed rather than owned here for exactly that reason.
//!
//! # The order, and what each step is for
//!
//! ```text
//!   pull      host mirror -> device rings, for everything the stage reads
//!   readiness the inputs hold values and the outputs have room
//!   bind      the logits buffer, per lane, into the intrinsic tables
//!   launch    one region at a time, in the order the stage states them
//!   commit    the verdict the kernel wrote, then the cursors
//!   push      device rings -> host mirror, for everything it published
//! ```
//!
//! Readiness is not a formality. A fire launched without asking reads the
//! ZEROED cell and publishes a confident answer computed from nothing,
//! which is the failure this whole driver keeps choosing to refuse
//! instead.

use super::run::Lane;
use driver::Extents;
use driver::driver_api::plan::LaunchStagePlan;
use driver::tensor_ir::DType;
use driver::tensor_ir::op::IntrinsicId;

use super::channel::{HostChannel, StageChannels};
use super::run::Control;
use super::run::Prepared;
use super::launch_control;
use super::channel::{ChannelShape, Rings};
use super::runtime::Compiled;
use crate::gpu::device::{Allocator, StreamRef};
use crate::error::{Error, Result};

/// The device rings of one bound instance, and what a fire needs to know
/// about them.
#[derive(Debug)]
pub struct Session {
    rings: Rings,
    shapes: Vec<ChannelShape>,
}

/// What one fire produced, so a caller can tell the three outcomes apart.
#[derive(Debug, PartialEq, Eq)]
pub enum Fired {
    /// The stage ran and committed.
    Committed {
        /// Cells pushed back into the host mirror.
        ///
        /// Can be fewer than the fire produced: a host ring the engine has
        /// not drained refuses the publish, which is a dropped output
        /// rather than an error, because the alternative is a fire that
        /// blocks on a reader.
        published: usize,
    },
    /// The stage ran and its kernel declined to commit. Not an error —
    /// a program may refuse a fire, and the cursors are left where they
    /// were so the next one sees the same inputs.
    Declined,
    /// The inputs were not ready, so nothing launched.
    ///
    /// Distinct from `Declined` because the causes are opposite: this is
    /// the driver not having something the program needs, and that is the
    /// program not wanting what it was given.
    NotReady,
}

impl Session {
    /// Build the rings for an instance's channels.
    ///
    /// # Errors
    ///
    /// If the shapes are empty or the device refuses the allocation.
    pub fn new(
        alloc: &Allocator,
        shapes: &[ChannelShape],
        stream: StreamRef<'_>,
    ) -> Result<Self> {
        Ok(Self {
            rings: Rings::new(alloc, shapes, stream)?,
            shapes: shapes.to_vec(),
        })
    }

    /// The rings, for the control kernels and for tests.
    #[must_use]
    pub const fn rings(&self) -> &Rings {
        &self.rings
    }

    /// Copy every cell the host has published, for the channels `sets`
    /// says the stage reads, into the device rings.
    ///
    /// Returns how many cells moved. A channel the host has nothing in is
    /// not an error here — [`Session::fire`] asks readiness about that,
    /// which is where the answer belongs.
    ///
    /// # Errors
    ///
    /// If a channel index is past the instance's, if a cell is the wrong
    /// width, or if a copy fails.
    pub fn pull(
        &mut self,
        host: &mut [HostChannel],
        sets: &StageChannels,
        stream: StreamRef<'_>,
    ) -> Result<usize> {
        let mut moved = 0;
        for &c in &sets.need_full {
            let index = c as usize;
            let shape = *self.shapes.get(index).ok_or_else(|| {
                Error::invalid("ptir::session", format!("no channel {c} in this instance"))
            })?;
            let Some(plane) = host.get_mut(index) else {
                return Err(Error::invalid(
                    "ptir::session",
                    format!("channel {c} has no host plane"),
                ));
            };
            // EVERYTHING the host has, not one cell: the device ring is as
            // deep as the host's, and a fire that pulled one would leave a
            // backlog that never drains.
            while let Some(wire) = plane.take() {
                let native = super::channel::wire_to_native(shape.dtype, shape.numel, &wire)
                    .map_err(|why| Error::invalid("ptir::session", why))?;
                let slot = self.rings.cursors(stream)?[index].tail;
                self.rings.seed(index, slot, &native, stream)?;
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// Copy everything the stage published back into the host mirror.
    ///
    /// Returns how many cells moved. A host ring the engine has not
    /// drained refuses the publish, which is a DROPPED output rather than
    /// an error — the alternative is a fire that blocks on a reader.
    ///
    /// # Errors
    ///
    /// If a channel index is past the instance's, or a copy fails.
    pub fn push(
        &mut self,
        host: &mut [HostChannel],
        sets: &StageChannels,
        before: &[u32],
        stream: StreamRef<'_>,
    ) -> Result<usize> {
        let after = self.rings.cursors(stream)?;
        let mut moved = 0;
        for &c in &sets.put {
            let index = c as usize;
            let shape = *self.shapes.get(index).ok_or_else(|| {
                Error::invalid("ptir::session", format!("no channel {c} in this instance"))
            })?;
            let ring = shape.ring()?;
            let (was, now) = (
                *before.get(index).unwrap_or(&0),
                after.get(index).map_or(0, |c| c.tail),
            );
            // The tails are ring-modular, so the count is the forward
            // distance rather than a subtraction — a fire that wrapped
            // would otherwise publish a negative number of cells, which
            // in `u32` is most of them.
            let produced = (now + ring - was % ring) % ring;
            for step in 0..produced {
                let slot = (was + step) % ring;
                let native = self.rings.read_cell(index, slot, stream)?;
                let wire = super::channel::native_to_wire(shape.dtype, shape.numel, &native)
                    .map_err(|why| Error::invalid("ptir::session", why))?;
                let Some(plane) = host.get_mut(index) else {
                    return Err(Error::invalid(
                        "ptir::session",
                        format!("channel {c} has no host plane"),
                    ));
                };
                if plane.publish(&wire) {
                    moved += 1;
                }
            }
        }
        Ok(moved)
    }

    /// Run one stage of a compiled program, end to end.
    ///
    /// `logits` is the fire's readout buffer and `vocab` its row width;
    /// `row_of` says which row a lane samples. One [`Extents`] per LANE:
    /// the members of a grouped fire share a plan and a set of channel
    /// cursors, and differ only in how much each of them submitted. They are handed in rather
    /// than discovered because the buffer is the model fire's and this
    /// module has no way to ask for it — which is also why a program that
    /// reads no intrinsic may pass a null base and lose nothing.
    ///
    /// # One stage, and the refusal that says so
    ///
    /// `Prepared` is built from ONE plan: its value types size the
    /// scratch, its channel bindings index the lane table, its op count
    /// strides the params. Every region launched against it reads those
    /// strides. So a program with two stages — an adapter prologue and a
    /// sampling epilogue is exactly that shape — cannot have its second
    /// stage's regions launched here: they would index the FIRST stage's
    /// descriptors, scratch and channel table, and nothing on the device
    /// carries a length to fault on.
    ///
    /// This used to launch every stage's regions in one `Prepared`. It
    /// was invisible because no program in the tree has two stages yet.
    ///
    /// Making it work is not a bigger loop: each stage needs its own
    /// `Prepared` and they must share ONE commit, because the channel
    /// cursors advance per FIRE and not per stage. That is the shape of
    /// the fix, and until it exists a multi-stage program is refused
    /// rather than run against the wrong memory.
    ///
    /// # Errors
    ///
    /// If the program has more than one stage, if the plan's channel sets
    /// cannot be derived, if a copy fails, or if a region refuses to
    /// launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fire(
        &mut self,
        compiled: &Compiled,
        plan: &LaunchStagePlan,
        control: &Control,
        host: &mut [HostChannel],
        logits: (u64, u32, u32),
        row_of: impl Fn(u32) -> u32,
        lane_extents: &[Extents],
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<Fired> {
        one_stage_only(compiled)?;
        let sets = super::channel::stage_channels(plan)
            .map_err(|why| Error::invalid("ptir::session", why))?;

        self.pull(host, &sets, stream)?;
        if !launch_control::readiness(
            control,
            &self.rings,
            &sets.need_full,
            &sets.need_empty,
            alloc,
            stream,
        )? {
            return Ok(Fired::NotReady);
        }

        // The tails BEFORE, because the push below counts what this fire
        // added and the control kernels are what move them.
        let before: Vec<u32> = self.rings.cursors(stream)?.iter().map(|c| c.tail).collect();

        // ONE LANE PER MEMBER, each with its OWN rings — which for a
        // single-instance fire is this session's, and for a grouped one
        // would be each member's. The fire takes the pairing rather
        // than one ring set, so the caller cannot group instances and
        // silently have them share channels.
        let members: Vec<Lane<'_>> =
            lane_extents.iter().map(|&extents| Lane { rings: &self.rings, extents }).collect();
        let mut prepared = Prepared::build(alloc, plan, &members, stream)?;
        let (base, vocab, row_stride) = logits;
        if base != 0 {
            prepared.bind_intrinsic(
                IntrinsicId::Logits,
                base,
                DType::F32,
                vocab,
                row_stride,
                row_of,
                stream,
            )?;
        }
        for stage in compiled.stages.iter() {
            for region in stage.regions.iter() {
                prepared.launch_region(region, stream)?;
            }
        }
        stream.synchronize()?;

        let committed = prepared.committed(stream)?;
        launch_control::commit(
            control,
            &self.rings,
            &sets.taken,
            &sets.put,
            committed,
            alloc,
            stream,
        )?;
        stream.synchronize()?;
        if !committed {
            return Ok(Fired::Declined);
        }
        let published = self.push(host, &sets, &before, stream)?;
        Ok(Fired::Committed { published })
    }
}

/// The one-stage precondition, named so a test can reach it.
///
/// A function rather than an inline `if`, because the alternative is a
/// test that asserts on a string it built itself — which passes whatever
/// `fire` does.
fn one_stage_only(compiled: &Compiled) -> Result<()> {
    if compiled.stages.len() > 1 {
        return Err(Error::invalid(
            "ptir::session",
            format!(
                "this program has {} stages and the fire prepares one: every \
                 region would index the first stage's descriptors, scratch and \
                 channel table. Each stage needs its own `Prepared`, sharing one \
                 commit because the channel cursors advance per fire",
                compiled.stages.len()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::runtime::{Compiled, Stage};

    /// A two-stage program is refused before anything is prepared.
    ///
    /// No program in the tree has two stages yet, which is exactly why
    /// `fire` launched every stage's regions against ONE `Prepared` for
    /// as long as it did. An adapter prologue plus a sampling epilogue is
    /// that shape, and it is the next program anyone writes.
    ///
    /// The check is on the COMPILED stage count rather than on anything
    /// the caller passes, because the caller passes one plan and cannot
    /// see the mismatch: `run_program` takes `plans.first()` and has no
    /// way to know the compiled form has more.
    ///
    /// A GPU is not needed and none is touched: the refusal precedes
    /// every device call in `fire`, which is the property being asserted.
    #[test]
    fn a_two_stage_program_is_refused_rather_than_run_against_one_prepared() {
        let stage = || Stage { signature_hash: 0, regions: Arc::new(Vec::new()) };
        let two = Compiled {
            stages: Arc::new(vec![stage(), stage()]),
            plans: Arc::new(vec![
                driver::driver_api::plan::LaunchStagePlan::default(),
                driver::driver_api::plan::LaunchStagePlan::default(),
            ]),
            // The shape that is coming: an adapter prologue and a
            // sampling epilogue.
            kinds: Arc::new(vec![
                super::super::runtime::stage_kind::PROLOGUE,
                super::super::runtime::stage_kind::EPILOGUE,
            ]),
        };
        let refusal = super::one_stage_only(&two).expect_err("two stages is refused");
        // The message names the COUNT, because a caller reading it has to
        // decide whether to split the program or wait for per-stage
        // preparation, and "more than one" does not tell them which.
        let text = format!("{refusal:?}");
        assert!(
            text.contains("2 stages"),
            "the refusal states how many: {text}"
        );
        assert!(
            text.contains("descriptors") || text.contains("scratch"),
            "and what would go wrong, not merely that something did: {text}"
        );

        // ONE stage is the served case — the shape every program in the
        // tree has, and the shape `ptir_shell.rs` fires end to end. Without
        // this the guard could refuse everything and still pass above.
        let one = Compiled {
            stages: Arc::new(vec![stage()]),
            plans: Arc::new(vec![driver::driver_api::plan::LaunchStagePlan::default()]),
            kinds: Arc::new(vec![super::super::runtime::stage_kind::EPILOGUE]),
        };
        super::one_stage_only(&one).expect("one stage is what runs today");
    }
}
