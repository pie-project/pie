//! One bound instance: rings, per-stage buffers, and one fire.

use std::sync::Arc;

use eta_exec::{ExecPlan, Extents};
use eta_ir::container::HostRole;
use eta_ir::registry::GeometryClass;
use eta_ir::validate::Direction;
use kernels_cuda::channel::{self, PublishLane, PullLane, SettleLane, Ticket};

use crate::device::Pinned;
use crate::error::{Fault, Result};

use super::compile::Compiled;
use super::endpoint::Endpoint;
use super::launch::{ChannelShape, Cursor, Prepared, Rings, native_to_wire, wire_to_native};
use super::wave::Wave;
use super::ports::{self, Envelope};

/// What one fire produced; mirrors [`eta_exec::StepOutcome`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fired {
    /// Every stage ran and the cursors advanced.
    Committed,
    /// This channel did not meet the program's declared requirement.
    Blocked(u32),
    /// A stage declined; cursors are unchanged.
    Declined,
    /// The instance is unusable and stays so.
    Faulted(String),
}

/// A fire's staged result, kept apart from [`Fired`] so nothing settles
/// what never launched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Launched {
    /// On the stream; owed a verdict from [`Session::settle_launched`].
    Airborne,
    /// Nothing launched; nothing owed.
    Refused(Fired),
}

/// An intrinsic bound to a buffer; row is per-lane, resolved when the lane
/// is taken.
#[derive(Clone, Copy, Debug)]
struct Intrinsic {
    id: eta_ir::op::IntrinsicId,
    base: u64,
    storage: u32,
    width: u32,
    row_stride: u32,
    row_offset: u32,
}

/// One bound instance's device state. Cursors are `u64` sequence numbers
/// that never wrap; a ring position is the residue.
#[derive(Debug)]
pub struct Session {
    rings: Rings,
    shapes: Vec<ChannelShape>,
    /// Predicted cursors, advanced at mint and rolled back on refusal. For a
    /// host-ended channel only the engine's own counter lives here; the
    /// guest's is its pinned word.
    cursors: Vec<Cursor>,
    /// What this instance's value shapes resolve against, fixed at bind.
    extents: Extents,
    /// Intrinsic bindings, indexed by `IntrinsicId as usize`; survive a fire.
    intrinsics: Vec<Option<Intrinsic>>,
    /// One bit per bound `IntrinsicId`. An unbound intrinsic's base is zero,
    /// which the kernel would dereference, so `stage` checks this first.
    bound: u64,
    poisoned: bool,
    fires: u64,
    /// Mapped pinned pair: `[0]` pass commit flag, `[1]` kill word.
    commit: Pinned,
    /// The one fire this session may have airborne. At most one, since a
    /// second mint would predict against unreconciled cursors.
    pending: Option<Minted>,
}

/// What one fire's mint decided, kept until settlement reads it.
#[derive(Debug)]
struct Minted {
    /// Cursors before this fire; rolled back to on device refusal.
    before: Vec<Cursor>,
    /// Tickets as staged, re-checked on refusal.
    tickets: Vec<Ticket>,
    /// Shared rings whose predicted head this fire advanced.
    shared_head: Vec<u32>,
    /// Same, for the predicted tail.
    shared_tail: Vec<u32>,
}

impl Session {
    /// Allocates rings and fire-path buffers, then seeds declared channels.
    /// `seeds` are wire cells per `(channel, bytes)`; `extents` is what
    /// symbolic value shapes resolve against.
    ///
    /// # Errors
    ///
    /// Stages/plans mismatch, or a seed for a channel this instance lacks
    /// or of the wrong width.
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

        // Pinned: settlement reads it on the host.
        let commit = Pinned::mapped(2 * size_of::<u32>())?;

        let mut session = Session {
            rings,
            shapes,
            cursors,
            extents,
            intrinsics: vec![None; super::launch::INTRINSIC_SLOTS],
            bound: 0,
            poisoned: false,
            fires: 0,
            commit,
            pending: None,
        };
        for (channel, wire) in seeds {
            // A shared ring is seeded once, by whichever attachment binds first.
            let shared = session
                .rings
                .endpoint(*channel as usize)
                .filter(|endpoint| endpoint.role() == HostRole::None);
            if let Some(endpoint) = shared
                && !endpoint.claim_seeding()
            {
                continue;
            }
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
        // Registry state as of the seeds; later writes are `commit_bump`'s.
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

    /// Every channel's two counters, from whichever owner keeps each.
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
            // A shared ring's truth is the endpoint's own predicted counters.
            Some(endpoint) if endpoint.role() == HostRole::None => {
                let (head, tail) = endpoint.predicted();
                Cursor { head, tail }
            }
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

    /// Test-only: skew `channel`'s prediction to force a device refusal.
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

    /// Device address of this instance's pass commit word.
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

    /// Push one wire cell into channel `channel`; `false` means back-pressure,
    /// not a drop.
    ///
    /// # Errors
    ///
    /// Unknown channel, wrong cell width, or whatever the copy said.
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

    /// Advance whichever storage owns channel `channel`'s head by one.
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
    /// # Errors
    ///
    /// Unknown channel, and whatever the copy said.
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
    /// touching no cursor. For the parity test, not for serving.
    ///
    /// # Errors
    ///
    /// Unknown channel, and whatever the copy said.
    pub fn peek(&self, channel: u32, sequence: u64) -> Result<Vec<u8>> {
        let shape = self.shape_of(channel)?;
        let native = self.rings.read_cell(channel as usize, sequence)?;
        native_to_wire(shape.dtype, shape.numel, &native)
    }

    /// What this instance's descriptor ports hold right now: the cell at
    /// `head`. Nothing is consumed here.
    ///
    /// # Errors
    ///
    /// A port names a channel this instance lacks or a non-integer cell.
    pub fn envelope(&self, plan: &ExecPlan, class: GeometryClass) -> Result<Envelope> {
        ports::resolve(plan, class, &self.rings, &self.cursors_now(), &self.shapes)
    }

    /// Point one intrinsic at a device buffer, for every stage of this
    /// instance; survives a fire. `width` is the row width, `row_stride` the
    /// elements between rows, `row_offset` the row this instance reads.
    ///
    /// # Errors
    ///
    /// An intrinsic past the side tables' pitch.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        intrinsic: eta_ir::op::IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        let slot = intrinsic as usize;
        let seat = self.intrinsics.get_mut(slot).ok_or_else(|| {
            Fault::program(
                "program::session",
                format!(
                    "intrinsic {slot} is past the pitch the side tables are indexed with"
                ),
            )
        })?;
        *seat = Some(Intrinsic {
            id: intrinsic,
            base,
            storage,
            width,
            row_stride,
            row_offset,
        });
        self.bound |= 1u64 << (intrinsic as u32);
        Ok(())
    }

    /// What this instance's value shapes resolve against.
    #[must_use]
    pub const fn extents(&self) -> Extents {
        self.extents
    }

    /// The enqueue half of a fire: mint, pull-validate, launch, bump,
    /// publish. At most one fire airborne at a time.
    ///
    /// # Errors
    ///
    /// As [`super::Plane::fire`], less whatever the launches said.
    pub fn stage(
        &mut self,
        compiled: &Compiled,
        plan: &ExecPlan,
        wave: &mut Wave,
    ) -> Result<Launched> {
        if self.pending.is_some() {
            return Err(Fault::program(
                "program::session",
                "this instance already has a fire airborne: a second mint would predict \
                 against cursors the first has not yet reconciled with the pinned words \
                 `pull_validate` reads, and the device would refuse whichever of the two \
                 it saw second",
            ));
        }
        if self.poisoned {
            return Ok(Launched::Refused(Fired::Faulted(
                "instance is poisoned".to_string(),
            )));
        }
        stages_and_plans_agree(compiled)?;

        // An unbound intrinsic is a null pointer the kernel dereferences.
        if plan.needs_logits
            && self.bound & (1u64 << (eta_ir::op::IntrinsicId::Logits as u32)) == 0
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `logits` intrinsic and no buffer has been \
                 bound to it; the emitted kernel dereferences the side table's zero, \
                 which is address zero",
            ));
        }
        // The draft column needs the same guard.
        if plan.needs_mtp_logits
            && self.bound & (1u64 << (eta_ir::op::IntrinsicId::MtpLogits as u32)) == 0
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `mtp_logits` intrinsic and no buffer has \
                 been bound to it; a model whose text declares no `mtp` export has \
                 no draft column for it to point at",
            ));
        }

        // The attention-score capture buffer needs the same guard.
        if plan.needs_attn_scores
            && self.bound & (1u64 << (eta_ir::op::IntrinsicId::AttnScore as u32)) == 0
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `attn_score` intrinsic and no buffer has been \
                 bound to it; a lane that did not ask to capture its attention has no \
                 block of the observability slab for it to point at",
            ));
        }

        // First failing channel wins, matching `eta_exec::step`'s ordering.
        if let Some(blocked) = self.blocked_channel(plan) {
            return Ok(Launched::Refused(Fired::Blocked(blocked)));
        }

        // Mint: predictions advance here, rolled back at settle on refusal.
        let minted = match self.mint(plan, wave) {
            Ok(minted) => minted,
            Err(why) => {
                self.poisoned = true;
                return Ok(Launched::Refused(Fired::Faulted(why)));
            }
        };

        self.pending = Some(minted);
        Ok(Launched::Airborne)
    }

    /// Take this fire's lane of the program's batch, in every stage: writes
    /// cell addresses, commit word, and intrinsic bindings into the row.
    ///
    /// # Errors
    ///
    /// A stage-local slot names a channel this instance lacks, or the batch
    /// has no lane left.
    pub fn take_lane(&mut self, stages: &mut [Option<Prepared>]) -> Result<()> {
        let Some(minted) = self.pending.as_ref() else {
            return Ok(());
        };
        let commit = self.commit.device();
        for prepared in stages.iter_mut().flatten() {
            let lane = prepared.stage_lane(&self.rings, &minted.before, commit)?;
            // Bind every stage: e.g. `logits` binds both prologue and epilogue.
            for intrinsic in self.intrinsics.iter().flatten() {
                prepared.bind_intrinsic(
                    lane,
                    intrinsic.id,
                    intrinsic.base,
                    intrinsic.storage,
                    intrinsic.width,
                    intrinsic.row_stride,
                    intrinsic.row_offset,
                )?;
            }
        }
        Ok(())
    }

    /// The verdict half: read the pinned commit word and reconcile
    /// predictions with it. Caller must ensure the kernels have run (e.g.
    /// after a synchronize). Answers [`Fired::Committed`] when nothing was
    /// airborne.
    ///
    /// # Errors
    ///
    /// As [`Session::settle`].
    pub fn settle_launched(&mut self) -> Result<Fired> {
        let Some(minted) = self.pending.take() else {
            return Ok(Fired::Committed);
        };
        self.settle(&minted)
    }

    /// Whether a fire of this session's is on the stream, unsettled.
    #[must_use]
    pub const fn airborne(&self) -> bool {
        self.pending.is_some()
    }

    /// The shared (device-only) rings this instance is attached to, keyed by
    /// endpoint identity.
    pub fn shared_rings(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.shapes.len()).filter_map(|channel| {
            self.rings
                .endpoint(channel)
                .filter(|endpoint| endpoint.role() == HostRole::None)
                .map(|endpoint| Arc::as_ptr(endpoint) as usize)
        })
    }

    /// This fire's tickets, slot lists, and three lanes, staged; touches no
    /// device word. Mirrors `eta_exec::step`'s commit arithmetic.
    fn mint(
        &mut self,
        plan: &ExecPlan,
        wave: &mut Wave,
    ) -> std::result::Result<Minted, String> {
        let before = self.cursors_now();
        let mut next = before.clone();
        let mut tickets: Vec<Ticket> = Vec::with_capacity(self.shapes.len());
        let mut taken: Vec<u32> = Vec::new();
        let mut put: Vec<u32> = Vec::new();
        // Shared rings' prediction is the endpoint's; `commit_bump` never sees them.
        let mut shared_head: Vec<u32> = Vec::new();
        let mut shared_tail: Vec<u32> = Vec::new();

        for (channel, cursor) in before.iter().enumerate() {
            let slot = channel as u32;
            let shape = self.shapes[channel];
            if cursor.tail < cursor.head {
                return Err(format!("channel {channel}: tail precedes head at mint"));
            }
            let takes = plan.takes_channel(slot);
            let puts = plan.puts_channel(slot);
            // A `read` claims the head like a take, without consuming.
            let addresses_head = takes || plan.reads_channel(slot);

            // A shared ring's durable state is its endpoint's.
            let shared = self
                .rings
                .endpoint(channel)
                .is_some_and(|endpoint| endpoint.role() == HostRole::None);

            let mut used = cursor.tail - cursor.head;
            let mut moved_head = false;
            let mut moved_tail = false;
            if takes && used != 0 {
                next[channel].head = cursor.head + 1;
                used -= 1;
                moved_head = true;
                if shared {
                    shared_head.push(slot);
                } else {
                    taken.push(slot);
                }
            }
            if puts {
                if used >= u64::from(shape.capacity) {
                    return Err(format!(
                        "channel {channel}: put overflows capacity {} at commit",
                        shape.capacity
                    ));
                }
                next[channel].tail = cursor.tail + 1;
                moved_tail = true;
                if shared {
                    shared_tail.push(slot);
                } else {
                    put.push(slot);
                }
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
                // A device-only ring also gets a pinned mirror written, so
                // ports read mapped memory instead of a blocking D2H copy.
                if matches!(endpoint.role(), HostRole::Reader | HostRole::None) {
                    flags |= Ticket::HOST_READER;
                }
            }
            if plan.requires_channel_input(slot) {
                flags |= Ticket::REQUIRE_INPUT;
            }
            // Bump the engine-owned counter of a slot whose prediction moved.
            if moved_head && endpoint.engine_owns_head() {
                flags |= Ticket::ADVANCE_HEAD;
            }
            if moved_tail && endpoint.engine_owns_tail() {
                flags |= Ticket::ADVANCE_TAIL;
            }
            // A shared ring's endpoint is native-width; packing bools would
            // write a cell an eighth the expected width.
            if shape.dtype == eta_ir::Dtype::Bool
                && endpoint.role() != HostRole::None
            {
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

        let rings = self.rings.device();
        let commit = self.commit.device();

        // One copy for the whole boundary: every lane appends to the same
        // six control lists; the wave fills in offsets once the arena base
        // is known.
        let _lane = wave.stage(
            &tickets,
            &taken,
            &put,
            PullLane {
                full: rings.full,
                pass_commit: commit,
                ticket_offset: 0,
                ticket_count: 0,
                // No reason yet to refuse: admission passed, no stage ran.
                initial_commit: 1,
                diagnose: 0,
            },
            rings.bump_lane(0, 0, 0, 0, commit),
            PublishLane {
                commit,
                ticket_offset: 0,
                ticket_count: 0,
            },
            SettleLane {
                commit,
                ticket_offset: 0,
                ticket_count: 0,
            },
        );

        self.cursors = next;
        // Shared rings' predictions advance on the endpoint too, so the next
        // mint (this attachment's or another's) counts what this fire minted.
        for slot in &shared_head {
            if let Some(endpoint) = self.rings.endpoint(*slot as usize) {
                endpoint.predict_head();
            }
        }
        for slot in &shared_tail {
            if let Some(endpoint) = self.rings.endpoint(*slot as usize) {
                endpoint.predict_tail();
            }
        }
        Ok(Minted {
            before,
            tickets,
            shared_head,
            shared_tail,
        })
    }

    /// The verdict, off the pinned commit word: one host load, no
    /// synchronize. On commit, counters were already advanced on the device.
    /// On refusal, predictions are rolled back; a stage clearing the word is
    /// [`Fired::Declined`], and a ticket mismatch after admission passed is
    /// a fault, never a silent retry.
    fn settle(&mut self, minted: &Minted) -> Result<Fired> {
        let word = self.commit.read(0, size_of::<u32>());
        let committed = u32::from_le_bytes([word[0], word[1], word[2], word[3]]) != 0;
        if committed {
            self.fires += 1;
            return Ok(Fired::Committed);
        }
        // A refused fire takes its shared predictions back.
        for slot in &minted.shared_head {
            if let Some(endpoint) = self.rings.endpoint(*slot as usize) {
                endpoint.unpredict_head(1);
            }
        }
        for slot in &minted.shared_tail {
            if let Some(endpoint) = self.rings.endpoint(*slot as usize) {
                endpoint.unpredict_tail(1);
            }
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
                     between the two (a surviving refusal is a contract \
                     violation, not a retry)"
                ),
            ));
        }
        Ok(Fired::Declined)
    }

    /// The first ticket whose claim the live pinned words do not bear out.
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
    /// not meet, or `None` when ready.
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
/// takes them. Returned oldest first, so republishing reproduces order.
#[must_use]
pub fn seeds_of(interp: &eta_exec::InterpInstance, plan: &ExecPlan) -> Vec<(u32, Vec<u8>)> {
    let mut seeds = Vec::new();
    for (channel, ring) in interp.channels.iter().enumerate() {
        let declared = match plan.package.channels.get(channel) {
            Some(declared) => declared,
            None => continue,
        };
        let dtype = eta_exec::concrete_dtype(declared.dtype);
        let numel = declared
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        for sequence in ring.head()..ring.tail() {
            let mut wire = vec![0u8; eta_exec::wire_cell_bytes(dtype, numel)];
            eta_exec::encode_wire(&ring.decode_sequence(sequence), &mut wire);
            seeds.push((channel as u32, wire));
        }
    }
    seeds
}

/// `compiled.stages` and `compiled.plans` must be parallel: a fire builds
/// one `Prepared` per launching stage from `compiled.plans[i]`.
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
        // Only a launching stage is prepared, so only its signature must match.
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

    /// A launching stage paired with the wrong plan is refused.
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

}
