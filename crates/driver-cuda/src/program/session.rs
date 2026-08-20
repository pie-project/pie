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
use driver::driver_api::plan::LaunchStagePlan;
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

    /// Run one stage of a compiled program, end to end.
    ///
    /// `logits` is the readout buffer and `vocab` its row width; `row_of` says
    /// which row a lane samples. One [`Extents`] per lane: a grouped fire's
    /// members share a plan and cursors, differing only in how much each
    /// submitted. The buffer is the model fire's and is handed in, so a program
    /// that reads no intrinsic may pass a null base.
    ///
    /// Only one stage: `Prepared` is built from one plan — its value types size
    /// the scratch, its bindings index the lane table, its op count strides the
    /// params — so a second stage's regions would index the first stage's
    /// memory with nothing on the device to fault on. A multi-stage program is
    /// refused until each stage gets its own `Prepared` sharing one commit (the
    /// channel cursors advance per fire, not per stage).
    ///
    /// # Errors
    ///
    /// If the program has more than one stage, if the channel sets cannot be
    /// derived, if a copy fails, or if a region refuses to launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fire(
        &mut self,
        rings: &mut Rings,
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

        self.pull(rings, host, &sets, stream)?;
        // The control kernels index the registry's flat arrays, so map the
        // dense channel numbers to slots before gating.
        let (need_full, need_empty) = (self.global(&sets.need_full)?, self.global(&sets.need_empty)?);
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
        // WHAT THE PROGRAM IS. A sampler that publishes a constant is
        // indistinguishable from one that never read the logits, and the ops
        // are the only place the difference shows.
        if std::env::var_os("PIE_TRACE_VALUES").is_some() {
            eprintln!(
                "[plan] flags={:#x} ops={} values={} error={:?}",
                plan.flags,
                plan.ops.len(),
                plan.value_types.len(),
                plan.error
            );
            for (i, op) in plan.ops.iter().enumerate() {
                eprintln!(
                    "[op] {i} code={:#x} intrinsic={} results={}",
                    op.code, op.intrinsic, op.result_count
                );
            }
        }
        let mut prepared = Prepared::build(alloc, plan, &members, stream)?;
        let (base, vocab, row_stride) = logits;
        if base != 0 {
            prepared.bind_intrinsic(
                IntrinsicId::Logits,
                base,
                // Raw bf16, which is what the fire writes into the pin:
                // `deliver_logits` reads the buffer as bf16 and widens
                // `(bits as u32) << 16`, matching `m1_intrinsic_row_base` for
                // mode 1. F32 mode (wire byte 0) would stride four bytes per
                // logit through a two-byte buffer and select different tokens.
                crate::program::params::INTRINSIC_STORAGE_RAW_BF16,
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

/// The one-stage precondition, named so a test can reach it without a GPU.
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

    /// A two-stage program is refused before anything is prepared — the check
    /// is on the compiled stage count, which the caller (passing one plan)
    /// cannot see. No GPU is touched; the refusal precedes every device call.
    #[test]
    fn a_two_stage_program_is_refused_rather_than_run_against_one_prepared() {
        let stage = || Stage {
            signature_hash: 0,
            regions: Arc::new(Vec::new()),
        };
        let two = Compiled {
            stages: Arc::new(vec![stage(), stage()]),
            plans: Arc::new(vec![
                driver::driver_api::plan::LaunchStagePlan::default(),
                driver::driver_api::plan::LaunchStagePlan::default(),
            ]),
            // The coming shape: an adapter prologue and a sampling epilogue.
            kinds: Arc::new(vec![
                super::super::runtime::stage_kind::PROLOGUE,
                super::super::runtime::stage_kind::EPILOGUE,
            ]),
        };
        let refusal = super::one_stage_only(&two).expect_err("two stages is refused");
        // The message names the count, so a caller knows how many to split.
        let text = format!("{refusal:?}");
        assert!(
            text.contains("2 stages"),
            "the refusal states how many: {text}"
        );
        assert!(
            text.contains("descriptors") || text.contains("scratch"),
            "and what would go wrong, not merely that something did: {text}"
        );

        // One stage is the served case; without this the guard could refuse
        // everything and still pass above.
        let one = Compiled {
            stages: Arc::new(vec![stage()]),
            plans: Arc::new(vec![driver::driver_api::plan::LaunchStagePlan::default()]),
            kinds: Arc::new(vec![super::super::runtime::stage_kind::EPILOGUE]),
        };
        super::one_stage_only(&one).expect("one stage is what runs today");
    }
}
