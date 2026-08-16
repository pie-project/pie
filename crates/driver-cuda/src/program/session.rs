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
//! A program names a channel by INDEX into its instance's `channel_ids`, so
//! a session is that list — a map from the instance's dense channel index to
//! the driver-wide slot whose ring holds the cells. The map is per instance;
//! the RINGS are not, and that is the correction this module carries.
//!
//! They used to be. `Rings` was sized from the instance's own channels and
//! allocated per session, so two instances naming ONE channel got two rings
//! and the second read a zeroed cell where the first had published. That is
//! the bench's decode loop exactly — the prefill's epilogue puts its sampled
//! token on a channel the DECODE instance binds to `EmbedTokens` — and it is
//! why a chained frame could not be served. [`super::channel::Rings`] is now
//! one registry for the whole driver and this holds indices into it.
//!
//! The map cannot be rebuilt per fire, for the reason the rings could not be:
//! rebuilding would renumber, and a slot is the only thing tying a program's
//! channel to the cursors that record what a previous fire published.
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
use driver::tensor_ir::op::IntrinsicId;

use super::channel::{ChannelShape, Rings};
use super::channel::{HostChannel, StageChannels};
use super::launch_control;
use super::run::Control;
use super::run::Prepared;
use super::runtime::Compiled;
use crate::device::{Allocator, StreamRef};
use crate::error::{Error, Result};

/// Where one bound instance's channels live in the driver's ring registry,
/// and what a fire needs to know about them.
#[derive(Debug)]
pub struct Session {
    /// Dense channel index → registry slot, in the order the instance's
    /// `channel_ids` lists them.
    slots: Vec<u32>,
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
    /// Name an instance's channels: `slots[dense]` is the registry slot whose
    /// ring holds channel `dense`'s cells, and `shapes[dense]` its geometry.
    ///
    /// # Errors
    ///
    /// If the instance has no channels, or the two lists disagree in length —
    /// which would silently give some channel another's cell width.
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
        rings: &mut Rings,
        host: &mut [HostChannel],
        sets: &StageChannels,
        stream: StreamRef<'_>,
    ) -> Result<usize> {
        self.pull_channels(rings, host, &sets.need_full, stream)
    }

    /// [`Session::pull`], for a stated list of the instance's dense channels.
    ///
    /// Split out because the DESCRIPTOR ports need the same copy and are not a
    /// stage's channel set. It is the port channels a HOST WRITER feeds that
    /// this moves — the seed an instance is bound with reaches its ring at
    /// registration instead (`launch::ensure_sessions`), because a seed's ring
    /// does not exist until then and the mirror is not where the engine leaves
    /// one.
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
            // ONLY A PLANE THE ENGINE WRITES. See `HostChannel::role`: the
            // mirror has one head/tail pair for both directions, so taking
            // from a plane this driver also publishes into re-injects the
            // program's OWN output as an input — one extra cell per fire on
            // every loop-carried channel.
            if !plane.engine_writes() {
                continue;
            }
            // EVERYTHING the host has, not one cell: the device ring is as
            // deep as the host's, and a fire that pulled one would leave a
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

    /// Copy everything the stage published back into the host mirror, and
    /// consume the device cells that landed there.
    ///
    /// Returns how many cells moved.
    ///
    /// # The mirror is the CONSUMER of a reader channel
    ///
    /// A `chan_put` advances the device ring's TAIL and nothing advances its
    /// head: no stage takes an output channel, and the commit kernel is only
    /// handed what the stage named. So a reader channel's ring filled up after
    /// `capacity` fires and the epilogue's own put then failed readiness —
    /// which is a decode loop that produces `capacity` tokens and then stops,
    /// silently, with the fire reporting `NotReady`. Measured at 21 tokens on
    /// the bench's `out`, whose ring is 22.
    ///
    /// Consuming here says what is true: the value has crossed to the plane
    /// the engine reads, so the device cell is spent.
    ///
    /// # A refused publish is BACK-PRESSURE, not a drop
    ///
    /// This used to say a full host ring was "a DROPPED output rather than an
    /// error — the alternative is a fire that blocks on a reader". The
    /// alternative is not blocking: the device cell simply stays unconsumed,
    /// so the next fire's readiness declines and the fire is retried once the
    /// engine drains. Nothing waits and no token is lost, which is strictly
    /// better than dropping one, so the cell is consumed only when the
    /// publish actually took it.
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
            // The tails are ring-modular, so the count is the forward
            // distance rather than a subtraction — a fire that wrapped
            // would otherwise publish a negative number of cells, which
            // in `u32` is most of them.
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
                // ONLY A PLANE THE ENGINE READS — the mirror of `pull`'s rule
                // and for the same reason. Publishing into a device-only
                // channel's plane fills a ring nobody drains, and then the
                // driver reads its own writes back.
                if !plane.engine_reads() {
                    continue;
                }
                if !plane.publish(&wire) {
                    // The engine is behind. Leave the cell where it is: the
                    // next fire's readiness declines on this channel and
                    // retries, which is what makes the back-pressure above
                    // lossless.
                    break;
                }
                moved += 1;
                // NOT for a channel the stage also TAKES: the commit kernel
                // has already advanced that head, and a second advance here
                // would skip a cell nobody read.
                if !sets.taken.contains(&c) {
                    rings.consume_front(global, stream)?;
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
        // The control kernels index the registry's flat arrays, so the sets
        // cross as SLOTS. A stage names its channels densely, and handing the
        // dense number straight through would gate on whichever channel
        // happened to be registered at that position.
        let (need_full, need_empty) = (self.global(&sets.need_full)?, self.global(&sets.need_empty)?);
        if !launch_control::readiness(control, rings, &need_full, &need_empty, alloc, stream)? {
            // WHICH CHANNEL, and in which direction. A bare `NotReady` says a
            // fire waited and nothing else, and the two causes are opposite —
            // an input nobody filled, or an output nobody drained — so the
            // trace names the cursors that decided rather than the verdict.
            // The env check inline rather than `fire::launch::sg_trace`: that
            // module is `feature = "abi"` and this one is not, so naming it
            // would make the trace decide whether the program plane compiles.
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

        // The tails BEFORE, because the push below counts what this fire
        // added and the control kernels are what move them. Indexed DENSELY,
        // which is how `push` reads it back.
        let cursors = rings.cursors(stream)?;
        let before: Vec<u32> = self
            .slots
            .iter()
            .map(|&g| cursors.get(g as usize).map_or(0, |c| c.tail))
            .collect();

        // ONE LANE PER MEMBER. The lanes of a grouped fire are different
        // INSTANCES, so each carries its own dense→slot map; the registry
        // behind them is one, because a channel has one ring wherever it is
        // named from.
        let members: Vec<Lane<'_>> = lane_extents
            .iter()
            .map(|&extents| Lane {
                rings,
                slots: &self.slots,
                extents,
            })
            .collect();
        let mut prepared = Prepared::build(alloc, plan, &members, stream)?;
        let (base, vocab, row_stride) = logits;
        if base != 0 {
            prepared.bind_intrinsic(
                IntrinsicId::Logits,
                base,
                // RAW BF16, which is what the fire writes into the pin.
                //
                // `publish_seam_pins` sizes a named pin `rows * width * 4` and
                // says why: "the GDN seam pins are f32; llama-like's are bf16
                // and simply leave half the pin unread". `deliver_logits`
                // reads that same buffer as bf16 and widens it
                // `(bits as u32) << 16` on the way to the wire, which is the
                // same widening `m1_intrinsic_row_base` does on the device —
                // for `mode == 1`.
                //
                // This said `DType::F32`, whose wire byte is `0`, and mode `0`
                // strides FOUR bytes per logit through a two-byte buffer. See
                // `Prepared::bind_intrinsic`: the table is a STORAGE mode and
                // the two vocabularies agree on exactly that one value, which
                // is how it went unnoticed. Measured: the two modes select
                // different tokens from the same fire.
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
