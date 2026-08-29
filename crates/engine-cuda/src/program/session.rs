//! One bound instance: its rings, its per-stage buffers, and one fire.
//!
//! **THE DEVICE GATES AND THE DEVICE COMMITS; THE HOST PREDICTS** (alto
//! design §5, articles 3 and 4). A fire is a ticket table, then every stage's
//! regions, then one predicated bump — and the host reads no device word
//! anywhere in it.
//!
//! ```text
//! fire
//! ----
//! admit       the host pre-check, the Article 4 bridge (see below)   host
//! mint        predictions -> tickets, cell addresses, slot lists     host
//! pull        channel::pull_validate: seed the commit word, check    device
//!             every prediction against the live pinned counters,
//!             pull what the guest wrote into the device cells
//! launch      each stage's regions, in stage order, no sync between  device
//! bump        channel::commit_bump: the ONLY advance of durable      device
//!             ring state, and only if the commit word survived
//! publish     channel::scatter_publish: the committed cells into     device
//!             the guest's mapped pinned mirror — not a memcpy
//! settle      one synchronize, then the verdict off the PINNED       host
//!             commit word: never a device read
//! ```
//!
//! # What this replaced, and why the replacement is not optional
//!
//! What stood here ran readiness and commit on the HOST: a
//! `context.synchronize()` per stage, a four-byte D2H to read the stage's
//! commit slot, and a host advance of the cursors afterwards. The survey
//! names that inversion as the root of six of its ten violations — I1
//! (durable ring state advanced host-side), I2 (positions read back rather
//! than predicted), I3 (outcome classification through a device read), I5
//! (cells crossing by memcpy in both directions) — and the module that stood
//! here stated the condition under which the control kernels would have to
//! come back: *a fire path that does not synchronize before the next fire's
//! ports are read*. That is what alto is. So:
//!
//! * **the host owns a PREDICTION and the device owns the TRUTH.** A
//!   [`Cursor`] here is a monotone counter the host advances by COUNTING at
//!   mint — never by looking — and this fire's cell addresses are arithmetic
//!   on it. `channel::pull_validate` compares it against the live pinned
//!   words where the data is and clears the commit word if it is stale.
//! * **a refused fire is a DUMMY RUN, not a fire that did not happen.** Every
//!   stage launches; every stage's kernel reads the commit word first and
//!   early-returns when it is clear (`fused_block1.cuh`'s first three lines).
//!   Its puts went into the pending cell, the bump never moved the tail, and
//!   the next fire overwrites them. There is no rollback path to get wrong.
//! * **nothing crosses by memcpy.** A guest's cell is READ where the guest
//!   wrote it (the pinned mirror, inside `pull_validate`) and a pass's cell is
//!   WRITTEN where the guest will read it (`scatter_publish`).
//!
//! # The end-of-fire synchronize is still the law (F2b is where it goes)
//!
//! One `context.synchronize()` remains, at the end of the fire, and the
//! settlement below reads the pinned commit word after it. That is deliberate
//! for this wave: making settlement asynchronous needs the staging ring and
//! more than one frame in flight, and both are the next wave's. What has
//! already gone is everything the sync was load-bearing FOR — the per-stage
//! sync, the commit readback, and the host cursor advance — so the sync is
//! now a lifetime fence and not a control-flow one.
//!
//! # The Article 4 bridge, crossed (wave E)
//!
//! F2a named this the bridge: [`Session::blocked_channel`] asked, before a
//! fire, whether this instance's own declared per-channel requirement was met,
//! and the shell's `prepare` asked it again over every attached instance so a
//! blocked one could cross the contract as `Error::Exhausted` and be
//! re-offered. It was an approximation of static admission — one instance's
//! requirement rather than the whole frame's union — and wave E landed the
//! real thing: `runtime::pipeline::fire::validate_frame` walks a frame's steps
//! in slot order and proves device-only ring occupancy, host-writer staged
//! cells and reader-ring worst-case pressure against the channels' declared
//! capacities, before the frame is admitted.
//!
//! So the shell's gate is gone and this one is no longer an admission
//! decision. What is left of it is the STANDALONE fire's semantics: a program
//! fired on its own through `Shell::fire_program` answers [`Fired::Blocked`]
//! for the same channel [`engine::step`] blocks on, which is what the parity
//! test compares. At a model fire's BOUNDARY a block is not an answer at all —
//! an epilogue runs after the forward has written the lane's KV, so there is
//! nothing to replay — and `serve::committed_or` turns it into a [`Fault`]
//! naming the instance and the channel.
//!
//! Past that door the device decides, and a device refusal that is a
//! READINESS MISS is a loud [`Fault`] rather than a silent retry: the host
//! said this fire could commit and the ring disagreed, which means something
//! moved the cursors between the two. [`Session::settle`] is where that
//! sentence is written.
//!
//! **ONE COMMIT PER FIRE, NOT PER STAGE.** A program's stages are separate
//! programs joined only by channels: nothing flows between them in scratch,
//! every stage resolves its cell addresses from the SAME cursors, and a stage
//! that clears the commit word refuses the whole fire. They share one word —
//! the one `pull_validate` seeds, `commit_bump` reads and `scatter_publish`
//! reads — which is what makes the pass atomic.

use std::sync::Arc;

use engine::tensor_ir::registry::GeometryClass;
use engine::tensor_ir::container::HostRole;
use engine::tensor_ir::validate::Direction;
use engine::{ExecPlan, Extents};
use kernels_cuda::channel::{self, BumpLane, PublishLane, PullLane, Ticket};

use crate::device::{Buffer, Context, Pinned};
use crate::error::{Fault, Result};

use super::compile::Compiled;
use super::endpoint::Endpoint;
use super::launch::{
    ChannelShape, Cursor, Prepared, Rings, native_to_wire, record_bytes, slice_bytes,
    wire_to_native,
};
use super::ports::{self, Envelope};

/// What one fire produced.
///
/// Deliberately the shape of [`engine::StepOutcome`], because the parity test
/// compares the two directly: a device fire that blocks must block on the same
/// channel the host interpreter blocks on, and neither half may quietly turn a
/// refusal into a commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fired {
    /// Every stage ran and the cursors advanced.
    Committed,
    /// Nothing launched: this channel did not meet the program's declared
    /// requirement. The counterpart of [`engine::StepOutcome::Blocked`].
    Blocked(u32),
    /// A stage's kernel cleared its commit slot — a stale table ABI, a fault
    /// it raised, or a check it made itself. The cursors are left where they
    /// were, so the next fire sees the same inputs.
    Declined,
    /// The instance is unusable and stays so. The counterpart of
    /// [`engine::StepOutcome::Faulted`].
    Faulted(String),
}

/// One bound instance's device state.
///
/// The cursors are `u64` sequence numbers that never wrap, exactly as the host
/// half's [`ChannelState`](engine::ChannelState) keeps them; a ring position
/// is the residue. Keeping the same spelling is what makes a slot-for-slot
/// diff of the two halves mean anything.
#[derive(Debug)]
pub struct Session {
    rings: Rings,
    shapes: Vec<ChannelShape>,
    /// **THE PREDICTION, NOT THE TRUTH.** One monotone counter pair per
    /// channel, advanced by COUNTING when a fire is minted and rolled back
    /// when the device refuses it — never read back off the device. For a
    /// channel with a host end only the counter the ENGINE owns is kept here;
    /// the guest's own counter is its pinned word, and
    /// [`Session::cursors_now`] is where the two are read as one.
    cursors: Vec<Cursor>,
    /// One per stage plan, `None` for a stage with nothing to launch — the
    /// adapter prologue is exactly that, its one region being a sink the model
    /// fire reads out of the plan rather than a body anyone compiled.
    prepared: Vec<Option<Prepared>>,
    /// Which intrinsics have been pointed at a buffer, one bit per
    /// `IntrinsicId`. Tracked because the emitted kernel DEREFERENCES an
    /// unbound intrinsic's base — the side tables start at zero, and zero is
    /// address zero — so a program that reads the readout and was never handed
    /// one is an illegal access rather than a wrong answer. Refusing by name
    /// is the difference between a sentence and a poisoned CUDA context.
    bound: u64,
    poisoned: bool,
    fires: u64,
    /// **THE FIRE'S COMMIT PAIR, IN MAPPED PINNED MEMORY**: `[0]` the pass
    /// commit flag, `[1]` the kill word.
    ///
    /// Pinned rather than device-resident for one reason and it is invariant
    /// I3: the settlement classifies this fire's outcome by READING this word
    /// on the host, and a device buffer would make that a four-byte D2H after
    /// a synchronize — which is exactly the readback F2a exists to delete.
    /// Every writer of it is a kernel.
    commit: Pinned,
    /// This fire's ticket table, one entry per host-visible channel that
    /// makes a claim. Carved at bind: the fire path allocates nothing
    /// (article 7).
    tickets: Buffer,
    /// The one [`PullLane`] a fire is, and the one [`BumpLane`] and
    /// [`PublishLane`] it settles through. One lane because one instance is
    /// one pass with one commit; the kernels take tables because a grouped
    /// fire is many.
    pull_lane: Buffer,
    bump_lane: Buffer,
    publish_lane: Buffer,
    /// The two slot lists `commit_bump` walks: channels whose head advances
    /// and channels whose tail does.
    taken: Buffer,
    put: Buffer,
}

/// What one fire's mint decided, kept until the settlement reads it.
///
/// The whole of a fire's host-side state between the launch and the verdict —
/// which is small on purpose: everything else about the fire is already on
/// the stream as device data.
#[derive(Debug)]
struct Minted {
    /// The cursors as they stood BEFORE this fire, which the predictions roll
    /// back to when the device refuses.
    before: Vec<Cursor>,
    /// The tickets as they were staged, re-checked against the pinned words
    /// when the fire is refused to tell a readiness MISS (a contract
    /// violation) from a stage that declined.
    tickets: Vec<Ticket>,
    /// Channels whose head the bump advances.
    taken: Vec<u32>,
    /// Channels whose tail the bump advances.
    put: Vec<u32>,
}

impl Session {
    /// Allocate this instance's rings and every stage's fire-path buffers,
    /// then seed the channels the program declares seeds for.
    ///
    /// `seeds` are WIRE cells, one per `(channel, bytes)` pair — the same
    /// encoding [`engine::Registry::bind_instance`] takes, so an instance that
    /// already exists on the host half is adopted by handing over what its
    /// rings hold (see [`seeds_of`]) rather than by a second seeding rule.
    ///
    /// `extents` is what the program's symbolic value shapes resolve against.
    /// A guest program with no intrinsic resolves entirely from static dims
    /// and never reads it; one attached to a model fire is handed that fire's.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the compiled stages and the plans are not
    /// parallel, when a seed names a channel the instance does not carry or is
    /// not one cell wide, and whatever the allocations said.
    pub fn bind(
        compiled: &Compiled,
        plan: &ExecPlan,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        endpoints: Vec<Option<Arc<Endpoint>>>,
    ) -> Result<Session> {
        stages_and_plans_agree(compiled)?;
        if plan.package.channels.is_empty() {
            return Err(Fault::program(
                "program::session",
                "an instance with no channels: there is nothing for a fire to read \
                 or publish",
            ));
        }

        let shapes: Vec<ChannelShape> =
            plan.package.channels.iter().map(ChannelShape::of).collect();
        let rings = Rings::allocate(&shapes, endpoints)?;
        let cursors = vec![Cursor { head: 0, tail: 0 }; shapes.len()];

        // ── The fire's control-plane bytes, all of them, now (article 9).
        //    `commit` is pinned because the settlement reads it; the rest are
        //    device buffers the mint stages into, sized for the widest table
        //    this instance could ever hand a kernel — one entry per channel.
        let slots = shapes.len().max(1);
        let commit = Pinned::mapped(2 * size_of::<u32>())?;
        let session_commit = commit.device();

        let mut prepared = Vec::with_capacity(compiled.plans.len());
        for (index, stage_plan) in compiled.plans.iter().enumerate() {
            let launches = compiled
                .stages
                .get(index)
                .is_some_and(|stage| !stage.regions.is_empty());
            prepared.push(if launches {
                Some(Prepared::build(stage_plan, &shapes, extents, session_commit)?)
            } else {
                None
            });
        }

        let mut session = Session {
            rings,
            shapes,
            cursors,
            prepared,
            bound: 0,
            poisoned: false,
            fires: 0,
            commit,
            tickets: Buffer::zeroed(slots * size_of::<Ticket>())?,
            pull_lane: Buffer::zeroed(size_of::<PullLane>())?,
            bump_lane: Buffer::zeroed(size_of::<BumpLane>())?,
            publish_lane: Buffer::zeroed(size_of::<PublishLane>())?,
            taken: Buffer::zeroed(slots * size_of::<u32>())?,
            put: Buffer::zeroed(slots * size_of::<u32>())?,
        };
        for (channel, wire) in seeds {
            if !session.publish(*channel, wire)? {
                return Err(Fault::program(
                    "program::session",
                    format!(
                        "channel {channel}'s seed does not fit: its ring already holds \
                         a cell, so the seed would be the second value rather than the first"
                    ),
                ));
            }
        }
        // The registry's ring positions and full bytes, stated once, from
        // where the seeds left the cursors. Every later write to them is
        // `commit_bump`'s.
        let seeded = session.cursors_now();
        session.rings.seed_registry(&seeded)?;
        Ok(session)
    }

    /// How many channels this instance carries.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.shapes.len()
    }

    /// Channel `channel`'s geometry.
    #[must_use]
    pub fn shape(&self, channel: u32) -> Option<ChannelShape> {
        self.shapes.get(channel as usize).copied()
    }

    /// Channel `channel`'s cursors, as the two owners have them right now.
    #[must_use]
    pub fn cursor(&self, channel: u32) -> Option<Cursor> {
        let channel = channel as usize;
        let prediction = self.cursors.get(channel).copied()?;
        Some(self.merge(channel, prediction))
    }

    /// **EVERY CHANNEL'S TWO COUNTERS, FROM WHICHEVER OWNER KEEPS EACH.**
    ///
    /// A counter has exactly one writer and the channel's [`HostRole`] names
    /// it (`endpoint`'s header): on a channel the host WRITES the guest owns
    /// the tail and the engine owns the head, and on one the host READS it is
    /// the other way round. The engine's counter is the PREDICTION in
    /// [`Session::cursors`]; the guest's is its pinned word, read here where
    /// the guest wrote it — no device read in either case.
    ///
    /// This is the one place the two spellings meet, and everything that asks
    /// "where does this ring stand" — readiness, the ticket table, the cell
    /// addresses in the lane table, the descriptor ports — asks it here.
    fn cursors_now(&self) -> Vec<Cursor> {
        (0..self.shapes.len())
            .map(|channel| {
                let prediction = self.cursors[channel];
                self.merge(channel, prediction)
            })
            .collect()
    }

    fn merge(&self, channel: usize, prediction: Cursor) -> Cursor {
        match self.rings.endpoint(channel) {
            Some(endpoint) => Cursor {
                head: if endpoint.engine_owns_head() {
                    prediction.head
                } else {
                    endpoint.head()
                },
                tail: if endpoint.engine_owns_tail() {
                    prediction.tail
                } else {
                    endpoint.tail()
                },
            },
            None => prediction,
        }
    }

    /// **STATE A PREDICTION THE DEVICE WILL REFUSE** — for gates, and named
    /// so that no serving path could reach for it by accident.
    ///
    /// The whole point of alto's channel protocol is that the host's belief
    /// about a ring and the ring itself are separate things reconciled by a
    /// kernel, and the property that matters is what happens when they
    /// DISAGREE: the fire still launches, every stage early-returns on the
    /// commit word, the bump moves nothing, and nothing of the pass is
    /// observable afterwards. That property is invisible from the host in
    /// normal operation — a correct engine never disagrees with itself — so a
    /// gate has to be able to make it disagree on purpose.
    ///
    /// Shifts the engine-owned prediction of `channel` by `head` and `tail`.
    /// The guest's own counter is not touched: it is the pinned word, and the
    /// disagreement being tested is exactly the one between the two.
    ///
    /// **COMPILED ONLY WHERE A GATE CAN SEE IT** (alto wave P): behind
    /// `feature = "probe"`, which this crate defaults on and a serving binary
    /// drops with `default-features = false`. "Named so a serving path could
    /// not reach for it by accident" is a comment; this is the compiler
    /// saying it.
    #[cfg(any(test, feature = "probe"))]
    pub fn skew_prediction(&mut self, channel: u32, head: i64, tail: i64) {
        let shift = |counter: &mut u64, by: i64| {
            *counter = counter.saturating_add_signed(by);
        };
        if let Some(cursor) = self.cursors.get_mut(channel as usize) {
            shift(&mut cursor.head, head);
            shift(&mut cursor.tail, tail);
        }
    }

    /// **The device address of this instance's pass commit word.**
    ///
    /// The word `channel::pull_validate` seeds and every stage's kernel reads
    /// first — one per instance, living in the pinned commit pair, and the one
    /// place "did this pass commit?" is written down. Published because the
    /// recurrent fold is predicated on the same word: `channel::mask_from_commit`
    /// takes an array of these addresses and scatters each into the fold
    /// predicate its lane's scans read (alto design §6's change (a)).
    ///
    /// Read-only, and a POINTER rather than a value on purpose — the host
    /// reading it would be the round trip article 3 forbids.
    #[must_use]
    pub fn commit_word(&self) -> u64 {
        self.commit.device()
    }

    /// How many fires have committed on this instance.
    #[must_use]
    pub const fn fires(&self) -> u64 {
        self.fires
    }

    /// Whether this instance is unusable.
    #[must_use]
    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }

    /// How many unconsumed cells channel `channel` holds.
    #[must_use]
    pub fn depth(&self, channel: u32) -> u64 {
        self.cursor(channel)
            .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head))
    }

    /// Push one wire cell into channel `channel`, answering `false` when the
    /// ring has no room — back-pressure, not a drop.
    ///
    /// The host-side counterpart of [`engine::host_put`], and the only door a
    /// caller's bytes enter this plane through.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel or a cell of the wrong width,
    /// and whatever the copy said.
    pub fn publish(&mut self, channel: u32, wire: &[u8]) -> Result<bool> {
        let shape = self.shape_of(channel)?;
        if self.depth(channel) >= u64::from(shape.capacity) {
            return Ok(false);
        }
        let native = wire_to_native(shape.dtype, shape.numel, wire)?;
        let tail = self.cursor(channel).map_or(0, |cursor| cursor.tail);
        self.rings.write_cell(channel as usize, tail, &native)?;
        self.advance_tail(channel as usize);
        Ok(true)
    }

    /// Advance whichever storage owns channel `channel`'s tail by one.
    ///
    /// A guest-owned tail lives ONLY in the pinned word — that word is the
    /// counter, not a mirror of one. An engine-owned tail (a private channel,
    /// or a seed planted into a channel the host reads) lives in the
    /// prediction, and in the pinned word too when there is one, because
    /// `pull_validate` compares the two and a pair that drifted would refuse
    /// every fire.
    fn advance_tail(&mut self, channel: usize) {
        match self.rings.endpoint(channel) {
            Some(endpoint) if !endpoint.engine_owns_tail() => endpoint.bump_tail(),
            Some(endpoint) => {
                endpoint.bump_tail();
                self.cursors[channel].tail += 1;
            }
            None => self.cursors[channel].tail += 1,
        }
    }

    /// Advance whichever storage owns channel `channel`'s head by one. The
    /// mirror image of [`Session::advance_tail`].
    fn advance_head(&mut self, channel: usize) {
        match self.rings.endpoint(channel) {
            Some(endpoint) if !endpoint.engine_owns_head() => endpoint.bump_head(),
            Some(endpoint) => {
                endpoint.bump_head();
                self.cursors[channel].head += 1;
            }
            None => self.cursors[channel].head += 1,
        }
    }

    /// Take channel `channel`'s committed cell as wire bytes, advancing its
    /// head; `None` when the ring is empty.
    ///
    /// The counterpart of [`engine::host_take`].
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel, and whatever the copy said.
    pub fn take(&mut self, channel: u32) -> Result<Option<Vec<u8>>> {
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let shape = self.shape_of(channel)?;
        let head = self.cursor(channel).map_or(0, |cursor| cursor.head);
        let native = self.rings.read_cell(channel as usize, head)?;
        self.advance_head(channel as usize);
        Ok(Some(native_to_wire(shape.dtype, shape.numel, &native)?))
    }

    /// Channel `channel`'s cell at ring position `sequence`, as wire bytes,
    /// touching no cursor.
    ///
    /// **FOR DIFFING, NOT FOR SERVING.** The parity test reads every slot of
    /// every ring after every fire, because comparing only what was drained
    /// would miss a program that wrote the right value into the wrong slot.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel, and whatever the copy said.
    pub fn peek(&self, channel: u32, sequence: u64) -> Result<Vec<u8>> {
        let shape = self.shape_of(channel)?;
        let native = self.rings.read_cell(channel as usize, sequence)?;
        native_to_wire(shape.dtype, shape.numel, &native)
    }

    /// What this instance's descriptor ports hold right now.
    ///
    /// **THE COMMITTED FRONT, WHICH IS THE CELL THE GUEST'S OWN PASS TAKES.**
    /// A port's value for THIS fire is the cell at `head` — the same address
    /// [`Prepared::refresh`] publishes to the emitted kernel as
    /// `committed_cell` — so the shell's read and the guest's take are one
    /// value. Nothing is consumed: the pass's own commit advances `head` for
    /// every port [`Port::consumes`](engine::tensor_ir::registry::Port::consumes)
    /// names, and draining here as well would spend two cells per fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a port naming a channel this instance does not
    /// carry or holding a non-integer cell, and whatever the copy said.
    pub fn envelope(&self, plan: &ExecPlan, class: GeometryClass) -> Result<Envelope> {
        ports::resolve(plan, class, &self.rings, &self.cursors_now(), &self.shapes)
    }

    /// Point one intrinsic at a device buffer, for every stage of this
    /// instance.
    ///
    /// **EVERY STAGE, BECAUSE A PROGRAM IS ONE FIRE.** The side tables are
    /// per-stage buffers, but `logits` means the same buffer to a prologue and
    /// to an epilogue: binding one stage would leave the other reading address
    /// zero, which the emitted kernel dereferences.
    ///
    /// The binding SURVIVES a fire — [`Session::fire`] resets the pending
    /// flags and the scratch and nothing else (the commit word is the
    /// device's, seeded by `channel::pull_validate`) — so a caller
    /// that rebinds the same buffer every fire and one that binds it once
    /// behave the same. `width` is the row width (the vocabulary for logits),
    /// `row_stride` the ELEMENTS between rows, and `row_offset` which row this
    /// instance reads.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an intrinsic past the side tables' pitch, and
    /// whatever the copies said.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        context: &Context,
        intrinsic: engine::tensor_ir::op::IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        let stream = context.stream();
        for prepared in self.prepared.iter_mut().flatten() {
            prepared.bind_intrinsic(
                intrinsic, base, storage, width, row_stride, row_offset, stream,
            )?;
        }
        self.bound |= 1u64 << (intrinsic as u32);
        context.synchronize()
    }

    /// Run every stage of `compiled` as ONE fire: mint, pull-validate,
    /// launch, bump, publish, settle. The module header is the map.
    ///
    /// **A REFUSED FIRE IS NOT AN ERROR AND A BLOCKED ONE NEVER LAUNCHED.**
    /// A program whose inputs are not there is [`Fired::Blocked`] and is
    /// refused at the admission door with zero side effects; a program whose
    /// stages cleared the commit word is [`Fired::Declined`] and ran in full,
    /// publishing nothing. The one refusal that IS an error is a ticket the
    /// ring denied after the admission check approved it — see
    /// [`Session::settle`].
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the compiled stages and their plans are not
    /// parallel, or when a prediction the gate approved the ring denied; and
    /// whatever the launches, copies and the synchronize said.
    pub fn fire(
        &mut self,
        context: &Context,
        compiled: &Compiled,
        plan: &ExecPlan,
    ) -> Result<Fired> {
        if self.poisoned {
            return Ok(Fired::Faulted("instance is poisoned".to_string()));
        }
        stages_and_plans_agree(compiled)?;

        // AN UNBOUND INTRINSIC IS A NULL POINTER THE KERNEL DEREFERENCES, and
        // an illegal access poisons the CUDA context for the rest of the
        // process — every later call on every later shell answers 700. So the
        // one thing the host can check before the launch, it checks.
        if plan.needs_logits
            && self.bound & (1u64 << (engine::tensor_ir::op::IntrinsicId::Logits as u32)) == 0
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `logits` intrinsic and no buffer has been \
                 bound to it; the emitted kernel dereferences the side table's zero, \
                 which is address zero",
            ));
        }
        // THE DRAFT COLUMN GETS ITS OWN GUARD, BECAUSE IT IS ITS OWN BUFFER
        // (palo C3b). `needs_mtp_logits` was a flag nothing checked while
        // there was one rectangle to bind; now the shell binds `MtpLogits` at
        // the `mtp` export and a load whose plan declares none binds nothing,
        // so a program that reads drafts against a headless model would take
        // the same address-zero dereference the line above exists to prevent.
        if plan.needs_mtp_logits
            && self.bound & (1u64 << (engine::tensor_ir::op::IntrinsicId::MtpLogits as u32)) == 0
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `mtp_logits` intrinsic and no buffer has \
                 been bound to it; a model whose text declares no `mtp` export has \
                 no draft column for it to point at",
            ));
        }

        // ── THE STANDALONE FIRE'S OWN VERB (the module header's "Article 4
        //    bridge, crossed"). The program's declared per-channel
        //    requirement, in channel order, answering with the FIRST channel
        //    that fails — because `engine::step` does exactly that and the
        //    parity test compares the two answers channel for channel.
        //
        //    **IT IS NOT AN ADMISSION CHECK ANY MORE.** Static admission is
        //    the runtime's `validate_frame`, over the whole frame's union of
        //    rings, before a frame is admitted; the shell's own copy of this
        //    question — asked in `serve`'s prepare over every attachment, and
        //    crossing the contract as `Error::Exhausted` — is deleted with the
        //    sleep-retry loop that consumed it. At a boundary a block reaches
        //    `serve::committed_or` and is a fault, not a scheduling answer.
        //
        //    Past this door the host makes no more decisions about this fire.
        if let Some(blocked) = self.blocked_channel(plan) {
            return Ok(Fired::Blocked(blocked));
        }

        let stream = context.stream();
        let ctx = context.ctx();

        // ── MINT. Predictions to tickets, tickets to cell addresses; the
        //    predictions advance HERE, before anything has run, because that
        //    is what lets fire N+1 be built while fire N is still on the
        //    stream (survey §7 upgrade 1). A device refusal rolls them back
        //    at settle.
        let minted = match self.mint(plan) {
            Ok(minted) => minted,
            Err(why) => {
                self.poisoned = true;
                return Ok(Fired::Faulted(why));
            }
        };

        // ── PULL-VALIDATE. It seeds the fire's commit word — so a fire with
        //    no ticket at all still runs this, because the seeding is what
        //    every stage's early-return reads — then checks each prediction
        //    against the live pinned counters and pulls what the guest wrote
        //    into the device cells.
        channel::pull_validate(ctx, self.tickets.ptr(), self.pull_lane.ptr(), 1)?;

        // ── THE STAGES, UNCONDITIONALLY (survey §7 I4). Every stage launches
        //    whatever the pull decided: the emitted kernel's first three lines
        //    read the commit word and return when it is clear
        //    (`fused_block1.cuh`), so a refused fire costs what a running one
        //    does and writes only into the pending cell nobody can address.
        //    The kernel sequence is therefore a function of the program and
        //    not of this fire's ring state, which is article 5.
        //
        //    NO SYNCHRONIZE BETWEEN STAGES. One stream is the ordering: a
        //    stage that reads a cell the previous one wrote is a later launch
        //    on the same stream, and kernel completion is itself the release.
        for (index, stage) in compiled.stages.iter().enumerate() {
            let Some(prepared) = self.prepared.get_mut(index).and_then(Option::as_mut) else {
                // A stage with nothing to launch needs no buffers and no
                // launch. Not an empty case: the adapter prologue is exactly
                // this shape.
                continue;
            };
            prepared.refresh(&self.rings, &minted.before, stream)?;
            for region in stage.regions.iter() {
                prepared.launch_region(region, stream)?;
            }
        }

        // ── THE BUMP, which is the only advance of durable ring state, and
        //    then the publication, which is the only crossing outward. Both
        //    read the one commit word; both do nothing when it is clear.
        //    Enqueued in this order because the launch boundary between the
        //    cell write and the announcement is what orders payload before
        //    tail (`channels.cuh`'s ordering note).
        channel::commit_bump(ctx, self.bump_lane.ptr(), 1)?;
        channel::scatter_publish(ctx, self.tickets.ptr(), self.publish_lane.ptr(), 1)?;

        // ── THE ONE SYNCHRONIZE, and it is the last host touch in the fire.
        //    F2b is where it goes; what it no longer does is decide anything.
        context.synchronize()?;
        self.settle(&minted)
    }

    /// **THIS FIRE'S TICKETS, ITS SLOT LISTS AND ITS THREE LANES, STAGED.**
    ///
    /// The host's whole contribution to a fire's admission decision, and it is
    /// arithmetic on counters: what this fire believes each ring's head and
    /// tail are, what it intends to do at each, and where the cells and
    /// counters live. Nothing here reads a device word.
    ///
    /// Only a channel with a HOST END gets a ticket. A private channel's ring
    /// never leaves the device, so there is no prediction to be wrong about —
    /// its slot still joins the bump's lists, because the registry it moves is
    /// the device's durable state for every channel alike.
    ///
    /// The prediction advances here, and the arithmetic is the one
    /// `engine::program::step` commits with, transcribed:
    ///
    /// * a take advances the head, but only when the ring held something;
    /// * a put advances the tail, and overflows the ring rather than wrapping
    ///   past the head, which is an instance that can never be trusted again;
    /// * the capacity check counts the take's credit, so a loop-carried
    ///   channel — taken and put in one fire, which is every decode loop —
    ///   commits at capacity 1.
    ///
    /// A slot joins `taken`/`put` exactly when the prediction moved, so the
    /// device registry and the host's prediction advance in lockstep.
    fn mint(&mut self, plan: &ExecPlan) -> std::result::Result<Minted, String> {
        let before = self.cursors_now();
        let mut next = before.clone();
        let mut tickets: Vec<Ticket> = Vec::with_capacity(self.shapes.len());
        let mut taken: Vec<u32> = Vec::new();
        let mut put: Vec<u32> = Vec::new();

        for (channel, cursor) in before.iter().enumerate() {
            let slot = channel as u32;
            let shape = self.shapes[channel];
            if cursor.tail < cursor.head {
                return Err(format!("channel {channel}: tail precedes head at mint"));
            }
            let takes = plan.takes_channel(slot);
            let puts = plan.puts_channel(slot);
            // A `read` addresses the committed cell without consuming it, so
            // it is a claim about the head like a take is — and, on a channel
            // the host writes, it is what makes the pull happen. dev gates the
            // pull on HostWriter|Consume; a peek that did not set CONSUME
            // would read a cell nobody copied in.
            let addresses_head = takes || plan.reads_channel(slot);

            let mut used = cursor.tail - cursor.head;
            if takes && used != 0 {
                next[channel].head = cursor.head + 1;
                used -= 1;
                taken.push(slot);
            }
            if puts {
                if used >= u64::from(shape.capacity) {
                    return Err(format!(
                        "channel {channel}: put overflows capacity {} at commit",
                        shape.capacity
                    ));
                }
                next[channel].tail = cursor.tail + 1;
                put.push(slot);
            }

            let Some(endpoint) = self.rings.endpoint(channel) else {
                continue;
            };
            if !addresses_head && !puts {
                continue;
            }
            let mut flags = 0u32;
            if addresses_head {
                flags |= Ticket::CONSUME;
                if endpoint.role() == HostRole::Writer {
                    flags |= Ticket::HOST_WRITER;
                }
            }
            if puts {
                flags |= Ticket::PUBLISH;
                if endpoint.role() == HostRole::Reader {
                    flags |= Ticket::HOST_READER;
                }
            }
            // The same requirement the admission check just asked, restated
            // where the data is: `tail > head`, checked against the guest's
            // live counter rather than against the host's memory of it.
            if plan.requires_channel_input(slot) {
                flags |= Ticket::REQUIRE_INPUT;
            }
            if shape.dtype == engine::tensor_ir::DType::Bool {
                flags |= Ticket::PACKED_BOOL;
            }
            let cells = self
                .rings
                .cell_address(channel, 0)
                .map_err(|why| format!("channel {channel}: {why}"))?;
            tickets.push(Ticket {
                slot,
                flags,
                expected_head: if addresses_head {
                    cursor.head
                } else {
                    channel::NO_TICKET
                },
                expected_tail: if puts { cursor.tail } else { channel::NO_TICKET },
                words: endpoint.words_device(),
                mirror: endpoint.mirror_device(),
                cells,
                cap1: endpoint.cap1(),
                wire_bytes: endpoint.wire_bytes(),
                native_bytes: u32::try_from(shape.cell_bytes()).unwrap_or(u32::MAX),
            });
        }

        let count = u32::try_from(tickets.len()).unwrap_or(u32::MAX);
        let rings = self.rings.device();
        let commit = self.commit.device();
        let stage = |buffer: &mut Buffer, bytes: &[u8]| -> std::result::Result<(), String> {
            if bytes.is_empty() {
                return Ok(());
            }
            buffer.write(0, bytes).map_err(|why| format!("{why}"))
        };
        stage(&mut self.tickets, &slice_bytes(&tickets))?;
        stage(&mut self.taken, &slice_bytes(&taken))?;
        stage(&mut self.put, &slice_bytes(&put))?;
        stage(
            &mut self.pull_lane,
            &record_bytes(&PullLane {
                full: rings.full,
                pass_commit: commit,
                ticket_offset: 0,
                ticket_count: count,
                // The fire arrives with no reason of its own to refuse: the
                // admission check passed and no stage has run. Every later
                // clearing of this word is a device decision.
                initial_commit: 1,
                diagnose: 0,
            }),
        )?;
        stage(
            &mut self.bump_lane,
            &record_bytes(&rings.bump_lane(
                self.taken.ptr(),
                u32::try_from(taken.len()).unwrap_or(u32::MAX),
                self.put.ptr(),
                u32::try_from(put.len()).unwrap_or(u32::MAX),
                commit,
            )),
        )?;
        stage(
            &mut self.publish_lane,
            &record_bytes(&PublishLane {
                commit,
                ticket_offset: 0,
                ticket_count: count,
            }),
        )?;

        self.cursors = next;
        Ok(Minted {
            before,
            tickets,
            taken,
            put,
        })
    }

    /// **THE VERDICT, OFF THE PINNED COMMIT WORD** (survey §7 I3).
    ///
    /// One host load out of mapped memory a kernel wrote — no synchronize of
    /// its own, no device read, no four-byte D2H. What it decides:
    ///
    /// ```text
    /// committed   advance the engine's PINNED counters, one per slot the
    ///             bump moved, so the next fire's prediction and the guest's
    ///             live words still agree
    /// refused     roll the predictions back, then say which kind of refusal
    /// ```
    ///
    /// And the two kinds of refusal are not the same sentence. A stage
    /// clearing the word is [`Fired::Declined`] — the program refused itself,
    /// its cursors stay where they were, and the next fire sees the same
    /// inputs. A TICKET clearing it is a readiness miss after the admission
    /// check passed, which article 4 says cannot happen: something moved this
    /// instance's cursors between the gate and the launch. That is a loud
    /// fault and never a silent retry.
    ///
    /// Which one it was is asked by re-running the kernel's own comparison
    /// against the pinned words, on the host, after the fact — the words are
    /// the host's to read and the guest may have advanced one since the kernel
    /// looked, so this can only ever misreport a genuine race as a decline,
    /// never a decline as a violation.
    fn settle(&mut self, minted: &Minted) -> Result<Fired> {
        let word = self.commit.read(0, size_of::<u32>());
        let committed = u32::from_le_bytes([word[0], word[1], word[2], word[3]]) != 0;
        if committed {
            for slot in &minted.taken {
                if let Some(endpoint) = self.rings.endpoint(*slot as usize)
                    && endpoint.engine_owns_head()
                {
                    endpoint.bump_head();
                }
            }
            for slot in &minted.put {
                if let Some(endpoint) = self.rings.endpoint(*slot as usize)
                    && endpoint.engine_owns_tail()
                {
                    endpoint.bump_tail();
                }
            }
            self.fires += 1;
            return Ok(Fired::Committed);
        }

        self.cursors = minted.before.clone();
        if let Some(stale) = self.stale_ticket(&minted.tickets) {
            return Err(Fault::program(
                "program::session",
                format!(
                    "channel {stale}'s ring is not where this fire predicted it, and the \
                     admission check said it would be: the pass was refused on the device \
                     and nothing of it is observable, but a prediction the gate approved \
                     and the ring denied means something advanced this instance's cursors \
                     between the two (alto article 4: a surviving refusal is a contract \
                     violation, not a retry)"
                ),
            ));
        }
        Ok(Fired::Declined)
    }

    /// The first ticket whose claim the live pinned words do not bear out —
    /// `pull_validate`'s comparison, restated on the host for the sake of the
    /// error message and nothing else.
    fn stale_ticket(&self, tickets: &[Ticket]) -> Option<u32> {
        for ticket in tickets {
            let endpoint = self.rings.endpoint(ticket.slot as usize)?;
            let (head, tail) = (endpoint.head(), endpoint.tail());
            if ticket.flags & Ticket::CONSUME != 0 && head != ticket.expected_head {
                return Some(ticket.slot);
            }
            if ticket.flags & Ticket::REQUIRE_INPUT != 0 && tail <= head {
                return Some(ticket.slot);
            }
            if ticket.flags & Ticket::PUBLISH != 0 {
                let credit = u64::from(ticket.flags & Ticket::CONSUME != 0);
                if tail != ticket.expected_tail
                    || tail - head >= u64::from(ticket.cap1 - 1) + credit
                {
                    return Some(ticket.slot);
                }
            }
        }
        None
    }

    /// The first channel whose declared requirement a fire right now would
    /// not meet, or `None` when this instance is ready to fire.
    ///
    /// **THE QUESTION, ASKED WITHOUT FIRING.** [`Session::fire`] asks it for
    /// itself, and that is the whole of it for a program fired on its own.
    ///
    /// It used to be an admission gate as well: the attachment path asked it
    /// over every attached instance and refused the whole fire before anything
    /// launched, because an epilogue that discovered its rings were not ready
    /// AFTER the forward would leave a fire the caller cannot retry. Wave E
    /// moved that proof to where it can be made over the whole frame at once
    /// — `runtime::pipeline::fire::validate_frame` — and past it a block at a
    /// boundary is a contract violation `serve::committed_or` names, not a
    /// refusal to be re-offered.
    ///
    /// Same arithmetic, one implementation: [`Session::fire`] calls this.
    #[must_use]
    pub fn blocked_channel(&self, plan: &ExecPlan) -> Option<u32> {
        for channel in 0..self.shapes.len() {
            let readiness = plan
                .package
                .channels
                .get(channel)
                .and_then(|declared| declared.readiness);
            let live = self.depth(channel as u32);
            let capacity = u64::from(self.shapes[channel].capacity);
            let ready = match readiness {
                Some(Direction::NeedsFull) => live != 0,
                Some(Direction::NeedsEmpty) => live < capacity,
                None => true,
            };
            if !ready {
                return Some(channel as u32);
            }
        }
        None
    }

    fn shape_of(&self, channel: u32) -> Result<ChannelShape> {
        self.shapes.get(channel as usize).copied().ok_or_else(|| {
            Fault::program(
                "program::session",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }
}

/// The wire cells a host-half instance's rings hold, as [`Session::bind`]
/// takes them.
///
/// **"CURSORS INITIALISED FROM THE HOST-SIDE STATE, OR FRESH" IS ONE CALL.**
/// A fresh instance is `bind(.., &[], ..)`; an instance the host half already
/// carries — because the runtime ran its prologue there, or because a previous
/// shell owned it — is `bind(.., &seeds_of(interp, plan), ..)`. There is no
/// second seeding rule, and no way for the two to disagree about what a cell
/// is: both sides speak wire bytes.
///
/// Cells are returned oldest first, so republishing them reproduces the ring's
/// order. A channel whose ring is empty contributes nothing.
#[must_use]
pub fn seeds_of(interp: &engine::InterpInstance, plan: &ExecPlan) -> Vec<(u32, Vec<u8>)> {
    let mut seeds = Vec::new();
    for (channel, ring) in interp.channels.iter().enumerate() {
        let declared = match plan.package.channels.get(channel) {
            Some(declared) => declared,
            None => continue,
        };
        let dtype = engine::concrete_dtype(declared.dtype);
        let numel = declared
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        for sequence in ring.head()..ring.tail() {
            let mut wire = vec![0u8; engine::wire_cell_bytes(dtype, numel)];
            engine::encode_wire(&ring.decode_sequence(sequence), &mut wire);
            seeds.push((channel as u32, wire));
        }
    }
    seeds
}

/// The pairing every stage's [`Prepared`] depends on.
///
/// A fire builds one `Prepared` per LAUNCHING stage from `compiled.plans[i]`,
/// so the two arrays have to be parallel and each launching stage has to be
/// the one its plan describes. A plan that is off by one sizes a stage's
/// scratch from someone else's value types, indexes the lane table with
/// someone else's bindings and strides the params by someone else's op count —
/// none of which faults on the device.
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
/// Projected to signatures and "launches?" first, because a compiled region
/// owns a loaded cubin, and a precondition nobody can test on the host is a
/// precondition nobody tests.
fn stage_plans_are_parallel(stages: &[(u64, bool)], plans: &[u64]) -> Result<()> {
    if stages.len() != plans.len() {
        return Err(Fault::program(
            "program::session",
            format!(
                "this program has {} compiled stage(s) and {} plan(s): the fire pairs \
                 them by index to prepare each stage's own scratch, so it cannot tell \
                 which plan describes which stage",
                stages.len(),
                plans.len()
            ),
        ));
    }
    for (index, (&(signature, launches), &plan)) in stages.iter().zip(plans).enumerate() {
        // Only a LAUNCHING stage is prepared, so only a launching stage's
        // signature has to match. A stage with no regions carries nothing to
        // compare.
        if launches && signature != plan {
            return Err(Fault::program(
                "program::session",
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
    /// The served shape, and the one a "single stage only" guard used to
    /// refuse outright: an adapter prologue that launches nothing beside a
    /// sampling epilogue that does. Two stages, two plans, one fire.
    #[test]
    fn an_adapter_prologue_and_a_sampling_epilogue_are_one_fire() {
        super::stage_plans_are_parallel(&[(0xa11, false), (0xb22, true)], &[0xa11, 0xb22])
            .expect("the plans are parallel and the launching stage is its own");
    }

    /// Two stages that both launch are served too — each gets its own
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
        let text = format!("{refusal}");
        assert!(text.contains("2 compiled stage"), "names how many: {text}");
        assert!(text.contains("1 plan"), "and how many of the other: {text}");
    }

    /// A launching stage paired with somebody else's plan is refused, which is
    /// the failure nothing on the device would fault on.
    #[test]
    fn a_launching_stage_paired_with_the_wrong_plan_is_refused() {
        let refusal =
            super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xdead])
                .expect_err("stage 1 is not what plan 1 describes");
        let text = format!("{refusal}");
        assert!(text.contains("stage 1"), "names which: {text}");
        assert!(
            text.contains("scratch") || text.contains("descriptors"),
            "and what would go wrong: {text}"
        );
    }

    /// A stage that launches nothing is never prepared, so its signature is
    /// never compared — demanding a match would refuse the adapter prologue
    /// over a plan the fire does not read.
    #[test]
    fn a_stage_that_launches_nothing_is_not_compared() {
        super::stage_plans_are_parallel(&[(0xa11, false)], &[0xdead])
            .expect("nothing is prepared from it");
    }
}
