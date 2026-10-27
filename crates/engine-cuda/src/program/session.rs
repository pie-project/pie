use std::sync::Arc;

use engine::ports::Envelope;
use eta_exec::{ExecPlan, Extents};
use eta_ir::container::HostRole;
use eta_ir::registry::{GeometryClass, Port, PortMask, Stage as Attach};
use eta_ir::validate::Direction;
use kernels_cuda::runtime::channel::{self, PublishLane, PullLane, SettleLane, Ticket};

use crate::device::Pinned;
use crate::error::{Fault, Result};

use super::compile::Compiled;
use super::endpoint::Endpoint;
use super::launch::{ChannelShape, Cursor, Prepared, Rings, native_to_wire, wire_to_native};
use super::ports;
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
struct Output {
    slot: u32,
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
    /// Indexed by [`eta_ir::op::output_slot`] in the stage reading outputs.
    outputs: Vec<Option<Output>>,
    /// That stage's names, an output's index among them, and the slots one
    /// lane takes.
    output_names: Vec<String>,
    output_slots: u32,
    poisoned: bool,
    fires: u64,
    /// A commit slot (two words the device writes) for each fire airborne
    /// at once, reused as they settle.
    commits: Vec<Pinned>,
    /// The pool the commit slots return to.
    commit_key: Option<crate::device::held::pool::PoolKey>,
    /// The fires minted and not yet settled, oldest first: each mints
    /// against the cursors the ones ahead of it predicted.
    pending: std::collections::VecDeque<Minted>,
    /// A fire ahead of the airborne ones declined, so they were minted
    /// against cursors that did not happen and decline in turn.
    cascading: bool,
    shadow: bool,
    /// The descriptor ports the program computes on the device
    /// (`engine::InstanceBinding::device_resolved`): the ones this instance
    /// resolves for a decode envelope.
    device: PortMask,
    /// A decode envelope's device-computed mask is packed into the staged
    /// mask on the device and never read on the host ([`Self::resolve_on`]).
    mask_on_device: bool,
}

#[derive(Debug)]
struct Minted {
    slot: usize,
    before: Vec<Cursor>,
    next: Vec<Cursor>,
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
        stream: *mut core::ffi::c_void,
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
        let rings = Rings::allocate(&shapes, endpoints, stream)?;
        let cursors = vec![Cursor { head: 0, tail: 0 }; shapes.len()];

        let commit_key = crate::device::held::pool::commit_key(2 * size_of::<u32>());

        let reading = plan.package.plans.iter().find(|stage| {
            stage
                .ops
                .iter()
                .any(|op| op.tag == eta_ir::op::tags::OUTPUT)
        });
        let mut session = Session {
            rings,
            shapes,
            cursors,
            extents,
            outputs: vec![None; reading.map_or(0, |stage| stage.output_pitch as usize)],
            output_names: reading.map_or_else(Vec::new, |stage| stage.names.clone()),
            output_slots: reading.map_or(0, |stage| stage.output_slots),
            poisoned: false,
            fires: 0,
            commits: Vec::new(),
            commit_key,
            pending: std::collections::VecDeque::new(),
            cascading: false,
            shadow,
            device: PortMask::NONE,
            mask_on_device: false,
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
            if let Some(endpoint) = session.rings.endpoint(*channel as usize)
                && !endpoint.claim_seeding()
            {
                continue;
            }
            if !session.publish(*channel, wire, stream)? {
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
        session.rings.seed_registry(&seeded, stream)?;
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

    /// The commit word of the fire minted last.
    #[must_use]
    pub fn commit_word(&self) -> u64 {
        self.pending
            .back()
            .map_or(0, |minted| self.commits[minted.slot].device())
    }

    /// A commit slot no airborne fire holds, a new one when every slot is.
    fn free_commit(&mut self) -> Result<usize> {
        let held = |slot: usize| self.pending.iter().any(|minted| minted.slot == slot);
        if let Some(slot) = (0..self.commits.len()).find(|&slot| !held(slot)) {
            return Ok(slot);
        }
        let commit = match self
            .commit_key
            .and_then(|key| crate::device::held::pool::take(key, None))
        {
            Some((commit, _, _)) => commit,
            None => Pinned::mapped(2 * size_of::<u32>())?,
        };
        self.commits.push(commit);
        Ok(self.commits.len() - 1)
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

    /// Publishes `wire` into `channel`, its cell staged on `stream` ahead of
    /// the fire that reads it.
    pub fn publish(
        &mut self,
        channel: u32,
        wire: &[u8],
        stream: *mut core::ffi::c_void,
    ) -> Result<bool> {
        let shape = self.shape_of(channel)?;
        if self.depth(channel) >= u64::from(shape.capacity) {
            return Ok(false);
        }
        let native = wire_to_native(shape.dtype, shape.numel, wire)?;
        let tail = self.cursor(channel).map_or(0, |cursor| cursor.tail);
        self.rings
            .write_cell(channel as usize, tail, &native, stream)?;
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

    /// Takes the oldest cell a settled fire put; `None` while the next one is
    /// a prediction of a fire still airborne.
    pub fn take(&mut self, channel: u32) -> Result<Option<Vec<u8>>> {
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let shape = self.shape_of(channel)?;
        let head = self.cursor(channel).map_or(0, |cursor| cursor.head);
        let settled_tail = self.pending.front().map_or(u64::MAX, |minted| {
            minted
                .before
                .get(channel as usize)
                .map_or(0, |cursor| cursor.tail)
        });
        if head >= settled_tail {
            return Ok(None);
        }
        let native = self.rings.read_cell(channel as usize, head)?;
        self.advance_head(channel as usize);
        Ok(Some(native_to_wire(shape.dtype, shape.numel, &native)?))
    }

    /// Fixes, once at bind, which ports this instance resolves under
    /// `class` (the `device` ones of a decode envelope, every one of the
    /// wide class) and how it reads a decode envelope's mask. A mask the
    /// device computes is packed on the device, unless the load buffers
    /// recurrent state (`buffers_rs`): a buffered lane replays the slots its
    /// mask admits, a schedule the host plans, so there the mask is read on
    /// the host after the epilogue that writes it.
    pub fn resolve_on(
        &mut self,
        plan: &ExecPlan,
        class: GeometryClass,
        device: PortMask,
        buffers_rs: bool,
    ) {
        self.device = device;
        self.mask_on_device = !buffers_rs
            && class == GeometryClass::DecodeEnvelope
            && engine::ports::engine_resolves(class, device, Port::AttnMask)
            && self.port_device_source(plan, Port::AttnMask).is_some();
    }

    /// Whether this instance resolves `port` under `class`
    /// ([`engine::ports::engine_resolves`]).
    #[must_use]
    pub fn resolves(&self, class: GeometryClass, port: Port) -> bool {
        engine::ports::engine_resolves(class, self.device, port)
    }

    /// The ports the host resolves under `class`; a mask packed on the device
    /// ([`Self::mask_device_source`]) is not among them.
    pub fn envelope(&self, plan: &ExecPlan, class: GeometryClass) -> Result<Envelope> {
        ports::resolve(
            plan,
            class,
            self.device,
            &self.rings,
            &self.cursors_now(),
            &self.shapes,
            self.mask_on_device,
        )
    }

    /// The device cell a decode envelope's attention mask is packed from,
    /// `(address, bool bytes)`, where it is packed on the device: like its
    /// tokens, a mask the device computes never rides the host.
    #[must_use]
    pub fn mask_device_source(&self, plan: &ExecPlan) -> Option<(u64, u32)> {
        if !self.mask_on_device {
            return None;
        }
        self.port_device_source(plan, Port::AttnMask)
    }

    /// A decode envelope's device-computed mask this instance reads on the
    /// host ([`Self::resolve_on`]): a fire carrying it is never staged ahead
    /// of the epilogue that writes it.
    #[must_use]
    pub fn reads_mask_on_host(&self, class: GeometryClass) -> bool {
        class == GeometryClass::DecodeEnvelope
            && self.resolves(class, Port::AttnMask)
            && !self.mask_on_device
    }

    #[must_use]
    pub fn token_device_source(&self, plan: &ExecPlan, class: GeometryClass) -> Option<(u64, u32)> {
        if !self.resolves(class, Port::EmbedTokens) {
            return None;
        }
        self.port_device_source(plan, Port::EmbedTokens)
    }

    #[must_use]
    pub fn fold_device_source(&self, plan: &ExecPlan, class: GeometryClass) -> Option<(u64, u32)> {
        if !self.resolves(class, Port::Fold) {
            return None;
        }
        self.port_device_source(plan, Port::Fold)
    }

    pub fn port_device_source(&self, plan: &ExecPlan, port: Port) -> Option<(u64, u32)> {
        let binding = plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == port && !binding.is_const)?;
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

    /// Points the program's reads of output `name` for `lane` at `base`;
    /// an output the program does not read is let go.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_output(
        &mut self,
        name: &str,
        lane: u32,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        let Some(slot) = self.slot_of(name, lane) else {
            return Ok(());
        };
        if let Some(seat) = self.outputs.get_mut(slot as usize) {
            *seat = Some(Output {
                slot,
                base,
                storage,
                width,
                row_stride,
                row_offset,
            });
        }
        Ok(())
    }

    fn slot_of(&self, name: &str, lane: u32) -> Option<u32> {
        let index = self.output_names.iter().position(|named| named == name)?;
        Some(u32::try_from(index).ok()? + self.output_slots * lane)
    }

    fn bound(&self, name: &str, lane: u32) -> bool {
        self.slot_of(name, lane)
            .and_then(|slot| self.outputs.get(slot as usize))
            .is_some_and(Option::is_some)
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
        at: Attach,
    ) -> Result<Launched> {
        if self.poisoned {
            return Ok(Launched::Refused(Fired::Faulted(
                "instance is poisoned".to_string(),
            )));
        }
        stages_and_plans_agree(compiled)?;
        if at != Attach::Prologue {
            self.outputs_bound(plan)?;
        }
        if let Some(blocked) = self.blocked_at(plan, at) {
            return Ok(Launched::Refused(Fired::Blocked(blocked)));
        }

        let slot = self.free_commit()?;
        let minted = match self.mint(plan, wave, slot, at) {
            Ok(minted) => minted,
            Err(why) => {
                self.poisoned = true;
                return Ok(Launched::Refused(Fired::Faulted(why)));
            }
        };

        self.pending.push_back(minted);
        Ok(Launched::Airborne)
    }

    /// An unbound output would make the emitted kernel dereference the side table's zero.
    fn outputs_bound(&self, plan: &ExecPlan) -> Result<()> {
        for (name, lane) in plan.outputs() {
            if !self.bound(name, lane) {
                return Err(Fault::program(
                    "program::session",
                    format!(
                        "this program reads output `{name}` of its pass's lane {lane} and \
                         nothing has been bound to it: the pass has no such lane, or the \
                         lane's class writes no such output"
                    ),
                ));
            }
        }
        Ok(())
    }

    pub fn take_lane(&mut self, stages: Vec<&mut Prepared>) -> Result<()> {
        let Some(minted) = self.pending.back() else {
            return Ok(());
        };
        let commit = self.commits[minted.slot].device();
        for prepared in stages {
            let lane = prepared.stage_lane(&self.rings, &minted.before, commit)?;
            for output in self.outputs.iter().flatten() {
                prepared.bind_intrinsic(
                    lane,
                    output.slot,
                    output.base,
                    output.storage,
                    output.width,
                    output.row_stride,
                    output.row_offset,
                )?;
            }
        }
        Ok(())
    }

    /// Settles the oldest airborne fire.
    pub fn settle_launched(&mut self) -> Result<Fired> {
        let Some(minted) = self.pending.pop_front() else {
            return Ok(Fired::Committed);
        };
        let fired = self.settle(&minted);
        if self.pending.is_empty() {
            self.cascading = false;
        }
        fired
    }

    #[must_use]
    pub fn airborne(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn shared_rings(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.shapes.len()).filter_map(|channel| {
            self.rings
                .endpoint(channel)
                .filter(|endpoint| endpoint.role() == HostRole::None)
                .map(|endpoint| Arc::as_ptr(endpoint) as usize)
        })
    }

    /// The shared rings this instance's stages put.
    pub fn shared_puts<'p>(&'p self, plan: &'p ExecPlan) -> impl Iterator<Item = usize> + 'p {
        (0..self.shapes.len()).filter_map(move |channel| {
            if !plan.puts_channel(channel as u32, Attach::Prologue)
                && !plan.puts_channel(channel as u32, Attach::Epilogue)
            {
                return None;
            }
            self.rings
                .endpoint(channel)
                .filter(|endpoint| endpoint.role() == HostRole::None)
                .map(|endpoint| Arc::as_ptr(endpoint) as usize)
        })
    }

    fn host_resolved_channels(&self, plan: &ExecPlan, class: GeometryClass) -> Vec<u32> {
        let tokens_on_device = self.token_device_source(plan, class).is_some();
        let mask_on_device = self.mask_device_source(plan).is_some();
        plan.package
            .ports
            .iter()
            .filter(|binding| !binding.is_const && self.resolves(class, binding.port))
            .filter(|binding| !(binding.port == Port::EmbedTokens && tokens_on_device))
            .filter(|binding| !(binding.port == Port::AttnMask && mask_on_device))
            .map(|binding| binding.channel)
            .collect()
    }

    pub fn host_resolved_rings(&self, plan: &ExecPlan, class: GeometryClass) -> Vec<usize> {
        self.host_resolved_channels(plan, class)
            .into_iter()
            .filter_map(|channel| {
                self.rings
                    .endpoint(channel as usize)
                    .filter(|endpoint| endpoint.role() == HostRole::None)
                    .map(|endpoint| Arc::as_ptr(endpoint) as usize)
            })
            .collect()
    }

    #[must_use]
    pub fn resolves_prologue_put(&self, plan: &ExecPlan, class: GeometryClass) -> bool {
        self.host_resolved_channels(plan, class)
            .into_iter()
            .any(|channel| plan.puts_channel(channel, Attach::Prologue))
    }

    fn mint(
        &mut self,
        plan: &ExecPlan,
        wave: &mut Wave,
        slot: usize,
        at: Attach,
    ) -> std::result::Result<Minted, String> {
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
            let takes = plan.takes_channel(slot, at);
            let puts = plan.puts_channel(slot, at);
            let addresses_head = takes || plan.reads_channel(slot, at);

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
                let port_channel = plan
                    .package
                    .ports
                    .iter()
                    .any(|binding| !binding.is_const && binding.channel == slot);
                if endpoint.role() == HostRole::Reader
                    || (endpoint.role() == HostRole::None && port_channel)
                {
                    flags |= Ticket::HOST_READER;
                }
            }
            if plan.requires_channel_input(slot, at) {
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
        let commit = self.commits[slot].device();

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

        self.cursors = next.clone();
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
            slot,
            before,
            next,
            tickets,
            shared_head,
            shared_tail,
        })
    }

    fn settle(&mut self, minted: &Minted) -> Result<Fired> {
        let word = self.commits[minted.slot].read(0, size_of::<u32>());
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

        // Only this fire's own advance is undone: the fires minted after it
        // keep theirs, and settle (declining) after it.
        for ((cursor, before), next) in self
            .cursors
            .iter_mut()
            .zip(&minted.before)
            .zip(&minted.next)
        {
            cursor.head -= next.head - before.head;
            cursor.tail -= next.tail - before.tail;
        }
        if core::mem::replace(&mut self.cascading, !self.pending.is_empty()) {
            return Ok(Fired::Declined);
        }
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
        self.blocked_at(plan, Attach::Prologue)
            .or_else(|| self.blocked_at(plan, Attach::Epilogue))
    }

    #[must_use]
    pub fn blocked_at(&self, plan: &ExecPlan, at: Attach) -> Option<u32> {
        for channel in 0..self.shapes.len() {
            let readiness = plan.readiness(channel as u32, at);
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

/// The commit slot goes back to the pool for the next session.
impl Drop for Session {
    fn drop(&mut self) {
        if let Some(key) = self.commit_key {
            for commit in self.commits.drain(..) {
                crate::device::held::pool::give(key, (commit, Pinned::empty(), None));
            }
        }
    }
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
