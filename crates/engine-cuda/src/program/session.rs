use std::sync::Arc;

use eta_exec::{ExecPlan, Extents};
use eta_ir::container::HostRole;
use eta_ir::registry::{GeometryClass, Port};
use eta_ir::validate::Direction;
use kernels_cuda::channel::{self, PublishLane, PullLane, SettleLane, Ticket};

use crate::device::Pinned;
use crate::error::{Fault, Result};

use super::compile::Compiled;
use super::endpoint::Endpoint;
use super::launch::{ChannelShape, Cursor, Prepared, Rings, native_to_wire, wire_to_native};
use super::ports::{self, Envelope};
use super::wave::Wave;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fired {
    Committed,
    Blocked(u32),
    Declined,
    Faulted(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Launched {
    Airborne,
    Refused(Fired),
}

#[derive(Clone, Copy, Debug)]
struct Intrinsic {
    id: eta_ir::op::IntrinsicId,
    base: u64,
    storage: u32,
    width: u32,
    row_stride: u32,
    row_offset: u32,
}

#[derive(Debug)]
pub struct Session {
    rings: Rings,
    shapes: Vec<ChannelShape>,
    cursors: Vec<Cursor>,
    extents: Extents,
    intrinsics: Vec<Option<Intrinsic>>,
    bound: u64,
    poisoned: bool,
    fires: u64,
    commit: Pinned,
    pending: Option<Minted>,
    shadow: bool,
}

#[derive(Debug)]
struct Minted {
    before: Vec<Cursor>,
    tickets: Vec<Ticket>,
    shared_head: Vec<u32>,
    shared_tail: Vec<u32>,
}

impl Session {
    pub fn bind(
        compiled: &Compiled,
        plan: &ExecPlan,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        endpoints: Vec<Option<Arc<Endpoint>>>,
        shadow: bool,
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
            shadow,
        };
        for (channel, wire) in seeds {
            if shadow
                && let Some(endpoint) = session.rings.endpoint(*channel as usize)
                && endpoint.role() != HostRole::None
            {
                if endpoint.engine_owns_tail() {
                    session.cursors[*channel as usize].tail += 1;
                }
                continue;
            }
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
        let seeded = session.cursors_now();
        session.rings.seed_registry(&seeded)?;
        Ok(session)
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn shape(&self, channel: u32) -> Option<ChannelShape> {
        self.shapes.get(channel as usize).copied()
    }

    #[must_use]
    pub fn cell(&self, channel: usize, sequence: u64) -> Result<(u64, u64)> {
        let address = self.rings.cell_address(channel, sequence)?;
        let bytes = self.rings.shape_of(channel)?.cell_bytes() as u64;
        Ok((address, bytes))
    }

    pub fn feed_cell(&self, channel: u32) -> Result<Option<(u64, u64)>> {
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let head = self.cursor(channel).map_or(0, |cursor| cursor.head);
        let address = self.rings.feed_address(channel as usize, head)?;
        let bytes = self.rings.shape_of(channel as usize)?.cell_bytes() as u64;
        Ok(Some((address, bytes)))
    }

    pub fn cell_bytes(&self, channel: u32) -> Result<u64> {
        Ok(self.rings.shape_of(channel as usize)?.cell_bytes() as u64)
    }

    pub fn cursor(&self, channel: u32) -> Option<Cursor> {
        let channel = channel as usize;
        let prediction = self.cursors.get(channel).copied()?;
        Some(self.merge(channel, prediction))
    }

    #[must_use]
    pub fn predictions(&self) -> Vec<Cursor> {
        self.cursors_now()
    }

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

    #[must_use]
    pub fn commit_word(&self) -> u64 {
        self.commit.device()
    }

    #[must_use]
    pub const fn fires(&self) -> u64 {
        self.fires
    }

    #[must_use]
    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }

    #[must_use]
    pub fn depth(&self, channel: u32) -> u64 {
        self.cursor(channel)
            .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head))
    }

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

    pub fn peek(&self, channel: u32, sequence: u64) -> Result<Vec<u8>> {
        let shape = self.shape_of(channel)?;
        let native = self.rings.read_cell(channel as usize, sequence)?;
        native_to_wire(shape.dtype, shape.numel, &native)
    }

    pub fn envelope(&self, plan: &ExecPlan, class: GeometryClass) -> Result<Envelope> {
        ports::resolve(plan, class, &self.rings, &self.cursors_now(), &self.shapes)
    }

    #[must_use]
    pub fn token_device_source(&self, plan: &ExecPlan, class: GeometryClass) -> Option<(u64, u32)> {
        if !ports::resolves(class, Port::EmbedTokens) {
            return None;
        }
        let binding = plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedTokens && !binding.is_const)?;
        let channel = binding.channel as usize;
        let endpoint = self.rings.endpoint(channel)?;
        if endpoint.role() != HostRole::None {
            return None;
        }
        let base = endpoint.device_cells()?;
        let native = u32::try_from(self.shapes.get(channel)?.cell_bytes()).ok()?;
        let head = self.cursors_now().get(channel)?.head;
        let src = base + (head % u64::from(endpoint.cap1())) * u64::from(native);
        Some((src, native))
    }

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
                format!("intrinsic {slot} is past the pitch the side tables are indexed with"),
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

    #[must_use]
    pub const fn extents(&self) -> Extents {
        self.extents
    }

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

        for (intrinsic, name, why) in [
            (
                eta_ir::op::IntrinsicId::Logits,
                "logits",
                "the emitted kernel dereferences the side table's zero, which is address zero",
            ),
            (
                eta_ir::op::IntrinsicId::Velocity,
                "velocity",
                "the lane's reading plants no `velocity` export",
            ),
            (
                eta_ir::op::IntrinsicId::Hidden,
                "hidden",
                "the lane's reading plants no `hidden` export",
            ),
        ] {
            if plan.reads_intrinsic(intrinsic) && self.bound & (1u64 << (intrinsic as u32)) == 0 {
                return Err(Fault::program(
                    "program::session",
                    format!(
                        "this program reads the `{name}` intrinsic and no buffer has been bound \
                         to it; {why}"
                    ),
                ));
            }
        }
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

        if plan.needs_mtp_drafts
            && self.bound & (1u64 << (eta_ir::op::IntrinsicId::MtpDrafts as u32)) == 0
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `mtp_drafts` intrinsic and no buffer has \
                 been bound to it; a model whose text plants no `mtp.drafts` export \
                 has no token plane for it to point at",
            ));
        }

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

        if self.bound & (1u64 << (eta_ir::op::IntrinsicId::Pixels as u32)) == 0
            && plan
                .package
                .values
                .iter()
                .any(|value| value.intrinsic == Some(eta_ir::op::IntrinsicId::Pixels))
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `pixels` intrinsic and no buffer has been                  bound to it; a lane whose reading plants no `pixels` seam — or a                  fire that submitted no clip — has no pixel plane for it to point at",
            ));
        }

        if self.bound & (1u64 << (eta_ir::op::IntrinsicId::PeerVelocity as u32)) == 0
            && plan
                .package
                .values
                .iter()
                .any(|value| value.intrinsic == Some(eta_ir::op::IntrinsicId::PeerVelocity))
        {
            return Err(Fault::program(
                "program::session",
                "this program reads the `peer_velocity` intrinsic and no buffer has \
                 been bound to it; the pass named no peer group (`forward-pass.peer`), \
                 so there is no second denoising in this fire to guide with",
            ));
        }

        if let Some(blocked) = self.blocked_channel(plan) {
            return Ok(Launched::Refused(Fired::Blocked(blocked)));
        }

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

    pub fn take_lane(&mut self, stages: &mut [Option<Prepared>]) -> Result<()> {
        let Some(minted) = self.pending.as_ref() else {
            return Ok(());
        };
        let commit = self.commit.device();
        for prepared in stages.iter_mut().flatten() {
            let lane = prepared.stage_lane(&self.rings, &minted.before, commit)?;
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

    pub fn settle_launched(&mut self) -> Result<Fired> {
        let Some(minted) = self.pending.take() else {
            return Ok(Fired::Committed);
        };
        self.settle(&minted)
    }

    #[must_use]
    pub const fn airborne(&self) -> bool {
        self.pending.is_some()
    }

    pub fn shared_rings(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.shapes.len()).filter_map(|channel| {
            self.rings
                .endpoint(channel)
                .filter(|endpoint| endpoint.role() == HostRole::None)
                .map(|endpoint| Arc::as_ptr(endpoint) as usize)
        })
    }

    fn mint(&mut self, plan: &ExecPlan, wave: &mut Wave) -> std::result::Result<Minted, String> {
        let before = self.cursors_now();
        let mut next = before.clone();
        let mut tickets: Vec<Ticket> = Vec::with_capacity(self.shapes.len());
        let mut taken: Vec<u32> = Vec::new();
        let mut put: Vec<u32> = Vec::new();
        let mut shared_head: Vec<u32> = Vec::new();
        let mut shared_tail: Vec<u32> = Vec::new();

        for (channel, cursor) in before.iter().enumerate() {
            let slot = channel as u32;
            let shape = self.shapes[channel];

            if self.shadow
                && self
                    .rings
                    .endpoint(channel)
                    .is_some_and(|endpoint| endpoint.role() != HostRole::None)
            {
                continue;
            }

            if cursor.tail < cursor.head {
                return Err(format!("channel {channel}: tail precedes head at mint"));
            }
            let takes = plan.takes_channel(slot);
            let puts = plan.puts_channel(slot);
            let addresses_head = takes || plan.reads_channel(slot);

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
            if self.shadow && endpoint.role() != HostRole::None {
                flags |= Ticket::SHADOW;
            }
            if addresses_head {
                flags |= Ticket::CONSUME;
                if endpoint.role() == HostRole::Writer {
                    flags |= Ticket::HOST_WRITER;
                }
            }
            if puts {
                flags |= Ticket::PUBLISH;
                if matches!(endpoint.role(), HostRole::Reader | HostRole::None) {
                    flags |= Ticket::HOST_READER;
                }
            }
            if plan.requires_channel_input(slot) {
                flags |= Ticket::REQUIRE_INPUT;
            }
            if moved_head && endpoint.engine_owns_head() {
                flags |= Ticket::ADVANCE_HEAD;
            }
            if moved_tail && endpoint.engine_owns_tail() {
                flags |= Ticket::ADVANCE_TAIL;
            }
            if shape.dtype == eta_ir::Dtype::Bool && endpoint.role() != HostRole::None {
                flags |= Ticket::PACKED_BOOL;
            }
            if flags & Ticket::SHADOW != 0 {
                flags &= !(Ticket::REQUIRE_INPUT
                    | Ticket::HOST_READER
                    | Ticket::ADVANCE_HEAD
                    | Ticket::ADVANCE_TAIL);
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
                expected_tail: if puts {
                    cursor.tail
                } else {
                    channel::NO_TICKET
                },
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

        let _lane = wave.stage(
            &tickets,
            &taken,
            &put,
            PullLane {
                full: rings.full,
                pass_commit: commit,
                ticket_offset: 0,
                ticket_count: 0,
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

    fn settle(&mut self, minted: &Minted) -> Result<Fired> {
        let word = self.commit.read(0, size_of::<u32>());
        let committed = u32::from_le_bytes([word[0], word[1], word[2], word[3]]) != 0;
        if committed {
            self.fires += 1;
            return Ok(Fired::Committed);
        }
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

    fn stale_ticket(&self, tickets: &[Ticket]) -> Option<u32> {
        for ticket in tickets {
            if ticket.flags & Ticket::SHADOW != 0 {
                continue;
            }
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
