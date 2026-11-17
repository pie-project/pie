//! An instance's tables, a channel's ring and an instance's adapter in the
//! shared pool (REDESIGN 2.6, `new-memory/STATE-ROLES.md` option A).
//!
//! An engine that states a fixed shape of role [`SlotRole::Instance`],
//! [`SlotRole::Channel`] or [`SlotRole::Adapter`] places a bound instance's
//! launch tables, a ring's cells and the adapter an instance lands in slots
//! of them, which the runtime takes from the pool on demand:
//! [`Engine::program_facts`] and [`Engine::ring_facts`] say how many, and
//! the binding ([`InstanceBinding::instance_slots`],
//! [`InstanceBinding::ring_slots`], [`InstanceBinding::adapter_slots`]) or
//! registration ([`ChannelRegistration::slots`]) names them. They are no
//! part of a record nor of a working set ([`super::shared`]).
//!
//! No such role is replayable ([`SlotRole::replayable`]): no token rebuilds
//! them, so nothing here drops one. Under pressure the policy may park them
//! with their process on the host tier, as it parks the process's kv: a
//! park copies each item to host slots and frees its device slots, an
//! unpark copies it into fresh device slots, and the engine is told the new
//! refs ([`Engine::rebind_slots`]) before the next fire reads them.
//!
//! A ring an instance holds itself is the instance's item. A registered
//! channel's ring is the channel's, shared by every instance bound to it,
//! and freed once the channel is closed and the last of them has closed
//! too. An adapter is keyed by its content ([`Engine::adapter_key`]): every
//! instance whose binding lands the same planes holds the one item, taken
//! at the first bind and freed at the last close; it parks only when every
//! instance holding it parks, and each holder is rebound to where it lands.
//!
//! [`Engine::adapter_key`]: ::engine::Engine::adapter_key
//!
//! A guest program's scratch is state too ([`::engine::GuestState`]): each
//! instance's, held while it is bound ([`InstanceBinding::scratch`]), and
//! its (program, extents) pair's, taken at the pair's first bind and given
//! back at its last close ([`InstanceBinding::pair`]), held as an adapter
//! is. Either lies in the `Instance` role ([`::engine::ScratchPlace`]): one
//! run of adjacent slots in a chunk where it fits a chunk's slots, else
//! pieces of [`::engine::GuestState::run`] adjacent whole chunks, each
//! inside one segment, so a value wider than a chunk lies whole in its
//! piece. The room is placed at the bind, never at a fire; a bind no free
//! run holds is refused by name ([`RolesError::GuestShort`], naming the run
//! and what was free). Their contents are dead between fires: a park gives
//! them back and copies nothing, an unpark takes a fresh run or fresh
//! runs of chunks, and the engine is told where ([`Rebind::Instance`])
//! before the next fire.
//!
//! An engine that states neither shape has no [`Roles`]: it keeps its
//! tables and rings itself, as before.
//!
//! [`InstanceBinding::scratch`]: ::engine::InstanceBinding::scratch
//! [`InstanceBinding::pair`]: ::engine::InstanceBinding::pair
//! [`Rebind::Instance`]: ::engine::Rebind::Instance
//!
//! [`Engine::program_facts`]: ::engine::Engine::program_facts
//! [`Engine::ring_facts`]: ::engine::Engine::ring_facts
//! [`Engine::rebind_slots`]: ::engine::Engine::rebind_slots
//! [`InstanceBinding::instance_slots`]: ::engine::InstanceBinding::instance_slots
//! [`InstanceBinding::ring_slots`]: ::engine::InstanceBinding::ring_slots
//! [`InstanceBinding::adapter_slots`]: ::engine::InstanceBinding::adapter_slots
//! [`ChannelRegistration::slots`]: ::engine::ChannelRegistration::slots

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use ::engine::chunk::{ShapeId as EngineShape, SlotRole};
use ::engine::{
    BoundInstance, ChannelId, ChannelRegistration, Engine, GuestState, InstanceBinding, InstanceId,
    ProgramFacts, ProgramId, Rebind, RingFacts, RingSlots, ScratchPlace,
};

use super::shared::programs::FireKey;
use super::shared::{FixedShape, RoleShapes, Shared};
use super::slots::SlotError;

/// Whose slots an item is: an instance's tables (and the rings it holds
/// itself), or a registered channel's ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Owner {
    Instance(InstanceId),
    Channel(ChannelId),
    /// An adapter, by its content key, shared by the instances that land it.
    Adapter(u128),
    /// A guest (program, extents) pair's state, shared by its instances
    /// ([`::engine::GuestState::per_pair`]).
    Pair(FireKey),
}

impl Owner {
    /// Whose slots `rebind` moves.
    #[must_use]
    pub fn of(rebind: &Rebind) -> Self {
        match rebind {
            Rebind::Instance { id, .. } => Owner::Instance(*id),
            Rebind::Channel { id, .. } => Owner::Channel(*id),
        }
    }
}

/// A host write the runtime issues into an instance's rings through the
/// engine: cells published into the rings the guest reads
/// ([`::engine::Engine::publish_channel`], `In`), or taken out of the ones
/// it writes ([`::engine::Engine::take_channel`], `Out`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pump {
    In,
    Out,
}

/// The process an item parks and unparks with.
pub type Holder = uuid::Uuid;

/// One engine's instance and channel slots.
pub type RolesHandle = Arc<Mutex<Roles>>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RolesError {
    #[error("the pool states no {0:?} shape, and the engine asks for its slots")]
    NoShape(SlotRole),
    #[error("the device pool has no room for {slots} {role:?} slots")]
    Short { role: SlotRole, slots: u32 },
    #[error("program {0} was never registered here")]
    Program(ProgramId),
    /// `holder` is the process it parked with: a bind of that process's
    /// waits for it to be resident again ([`bind_when_resident`]).
    #[error("channel {channel} is parked with its process")]
    Parked {
        channel: ChannelId,
        holder: Option<Holder>,
    },
    #[error("the adapter this binding lands is parked with the instances that hold it")]
    AdapterParked,
    #[error("{0}'s state is parked with the instances that hold it")]
    PairParked(FireKey),
    /// A bind found no device room for its items now, though the budget's
    /// state chunks hold them: it waits for room given back (a process
    /// completing, or parked), and binds again ([`bind_when_resident`]).
    #[error("{0}; the budget holds it, so the bind waits for room")]
    BindShort(Box<RolesError>),
    /// A parked process's items found no device room now, though the
    /// budget's state chunks hold them: the restore waits, queued, for room
    /// a completion or an eviction frees, and is never failed for it.
    #[error("{0}; the budget holds it, so the restore waits for room")]
    UnparkShort(Box<RolesError>),
    #[error("the device pool has no room for {what}, {bytes} bytes of Instance slots: {room}")]
    GuestShort {
        what: String,
        bytes: u64,
        room: GuestRoom,
    },
    /// No free run holds the guest state, and a segment still to commit
    /// within the budget would: the bind waits on the walk to commit one
    /// ([`grow_for_guest`]), as a kv admission short of room does, and is
    /// sent again once the committed prefix passes `prefix`. The pool
    /// counts what it is short ([`GUEST_WAITER`]) toward the walk's goal.
    #[error(
        "{what}, {bytes} bytes of Instance slots, waits on a segment commit: {room}; the \
         committed prefix is {prefix} segments"
    )]
    GuestGrow {
        what: String,
        bytes: u64,
        room: GuestRoom,
        run: u32,
        prefix: u32,
    },
    #[error(
        "no chunk has {slots} adjacent free {role:?} slots for an item that must lie in one run"
    )]
    NoRun { role: SlotRole, slots: u32 },
    #[error(
        "a {role:?} item must lie in one run of {slots} adjacent slots of {slot_bytes} bytes, \
         and a chunk holds {per_chunk} of them: its cell is larger than a chunk, which the \
         pool cannot place (new-memory/STATE-ROLES.md option D)"
    )]
    PastChunk {
        role: SlotRole,
        slots: u32,
        slot_bytes: u64,
        per_chunk: u32,
    },
}

impl RolesError {
    /// The device pool lacks room for it now: what a park, a completion or
    /// a commit may make, not a wrong ask.
    #[must_use]
    pub fn is_short(&self) -> bool {
        matches!(
            self,
            Self::Short { .. }
                | Self::NoRun { .. }
                | Self::GuestShort { .. }
                | Self::GuestGrow { .. }
        )
    }
}

/// The pool's waiter id for what guest binds are short
/// ([`super::slots::SlotPool::wait`]): counted toward the walk's goal while
/// a bind waits on a segment commit ([`RolesError::GuestGrow`]).
pub const GUEST_WAITER: u64 = u64::MAX - 1;

/// What a guest state asked of the pool, and what was free, where a bind
/// is refused or waits for want of it ([`RolesError::GuestShort`],
/// [`RolesError::GuestGrow`]); `committable`, the runs of that length (a
/// chunk, for a run of slots) the segments still to commit within the
/// budget hold ([`super::slots::SlotPool::committable_runs`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuestRoom {
    /// One run of `slots` adjacent `Instance` slots in a chunk.
    Slots { slots: u32, committable: u32 },
    /// `pieces` runs of `run` adjacent whole chunks, each inside one
    /// segment, and the free runs of that length and free chunks there
    /// were.
    Runs {
        run: u32,
        pieces: u32,
        free_runs: u32,
        free_chunks: u32,
        committable: u32,
    },
}

impl GuestRoom {
    fn committable(self) -> u32 {
        match self {
            Self::Slots { committable, .. } | Self::Runs { committable, .. } => committable,
        }
    }

    fn with_committable(self, runs: u32) -> Self {
        match self {
            Self::Slots { slots, .. } => Self::Slots {
                slots,
                committable: runs,
            },
            Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                ..
            } => Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                committable: runs,
            },
        }
    }
}

impl std::fmt::Display for GuestRoom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Slots { slots, .. } => write!(f, "no chunk has a run of {slots} free slots")?,
            Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                ..
            } => write!(
                f,
                "{pieces} runs of {run} adjacent chunks inside one segment asked, and \
                 {free_runs} such runs were free ({free_chunks} free chunks)"
            )?,
        }
        match self.committable() {
            0 => write!(f, ", and no segment is left to commit within the budget"),
            runs => write!(
                f,
                ", and the segments left to commit within the budget hold {runs} more"
            ),
        }
    }
}

/// Where an item's slots lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Device,
    Host,
    /// A park or unpark is copying it: neither side's slots are free.
    Moving,
}

/// One role's slots of an item, as refs of the tier the item lies on;
/// `contiguous`, adjacent in one chunk ([`::engine::Footprint::contiguous`]),
/// and taken so again on every tier it moves to. Guest scratch, whose
/// contents are dead between fires, is never copied and holds nothing off
/// the device (`refs` empty there), fresh room taken when it comes back:
/// `pieces` runs of `chunks` adjacent whole chunks, or (`run`) one run of
/// that many slots.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Part {
    role: SlotRole,
    refs: Vec<u32>,
    contiguous: bool,
    pieces: u32,
    chunks: u32,
    run: u32,
}

impl Part {
    /// Guest scratch, dead between fires.
    fn dead(&self) -> bool {
        self.pieces > 0 || self.run > 0
    }

    /// What it asks of the pool, as a refusal names it.
    fn room(&self) -> GuestRoom {
        if self.pieces > 0 {
            GuestRoom::Runs {
                run: self.chunks,
                pieces: self.pieces,
                free_runs: 0,
                free_chunks: 0,
                committable: 0,
            }
        } else {
            GuestRoom::Slots {
                slots: self.run,
                committable: 0,
            }
        }
    }

    /// The run of chunks a piece takes and the pieces: a run of slots
    /// takes one chunk.
    fn chunk_runs(&self) -> (u32, u32) {
        if self.pieces > 0 {
            (self.chunks, self.pieces)
        } else {
            (1, 1)
        }
    }
}

#[derive(Debug)]
struct Item {
    holder: Option<Holder>,
    tier: Tier,
    /// An instance: its tables' part, then each ring it holds itself. A
    /// channel: its ring's part. An adapter: its planes' part.
    parts: Vec<Part>,
    /// An instance: each own ring's channel index, in part order after the
    /// first.
    rings: Vec<u32>,
    /// An instance: the registered channels with slots it is bound to.
    channels: Vec<ChannelId>,
    /// An instance: the adapter it holds, by its key.
    adapter: Option<u128>,
    /// An instance: the guest pair whose state it shares.
    pair: Option<FireKey>,
    /// A channel or an adapter: the instances bound to it.
    attached: BTreeSet<InstanceId>,
    /// A channel: still registered. An instance: still bound.
    open: bool,
    /// Its slots moved and the engine has not been told yet.
    stale: bool,
}

impl Item {
    fn new(holder: Option<Holder>, parts: Vec<Part>) -> Self {
        Self {
            holder,
            tier: Tier::Device,
            parts,
            rings: Vec::new(),
            channels: Vec::new(),
            adapter: None,
            pair: None,
            attached: BTreeSet::new(),
            open: true,
            stale: false,
        }
    }

    fn refs(&self) -> Vec<u32> {
        self.parts
            .iter()
            .flat_map(|part| part.refs.iter().copied())
            .collect()
    }
}

/// The slots taken for an instance about to bind, before the engine names
/// it: [`Roles::bound`] keeps them, [`Roles::untake`] gives them back.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Taken {
    /// The tables' part, then each own ring's.
    parts: Vec<Part>,
    /// Each own ring's channel index, in part order after the first.
    rings: Vec<u32>,
    /// The registered channels with slots the binding names.
    channels: Vec<ChannelId>,
    /// The adapter the binding lands: its key, its slots, and whether they
    /// were taken for it (the first holder) or are another's.
    adapter: Option<(u128, Part, bool)>,
    /// The guest pair's state: its key, its chunks, and whether they were
    /// taken for this bind (the pair's first) or are its instances'.
    pair: Option<(FireKey, Part, bool)>,
    /// The slots of an `Instance` chunk: a whole part's chunks are its
    /// refs over it.
    chunk_slots: u32,
}

impl Taken {
    /// [`::engine::InstanceBinding::instance_slots`].
    #[must_use]
    pub fn instance_slots(&self) -> Vec<u32> {
        self.parts
            .first()
            .map_or_else(Vec::new, |part| part.refs.clone())
    }

    /// [`::engine::InstanceBinding::ring_slots`].
    #[must_use]
    pub fn ring_slots(&self) -> Vec<RingSlots> {
        own_rings(&self.rings, &self.parts)
    }

    /// [`::engine::InstanceBinding::adapter_slots`].
    #[must_use]
    pub fn adapter_slots(&self) -> Vec<u32> {
        self.adapter
            .as_ref()
            .map_or_else(Vec::new, |(_, part, _)| part.refs.clone())
    }

    /// [`::engine::InstanceBinding::scratch`].
    #[must_use]
    pub fn scratch(&self) -> ScratchPlace {
        scratch(&self.parts).map_or(ScratchPlace::None, |part| place(part, self.chunk_slots))
    }

    /// [`::engine::InstanceBinding::pair`].
    #[must_use]
    pub fn pair(&self) -> ScratchPlace {
        self.pair
            .as_ref()
            .map_or(ScratchPlace::None, |(_, part, _)| {
                place(part, self.chunk_slots)
            })
    }
}

/// An instance's scratch: its dead part.
fn scratch(parts: &[Part]) -> Option<&Part> {
    parts.iter().find(|part| part.dead())
}

/// Where a dead part lies: its run's first slot ref and slots, or each of
/// its runs of whole chunks by its first chunk, `chunk_slots` refs a chunk.
fn place(part: &Part, chunk_slots: u32) -> ScratchPlace {
    let Some(&first) = part.refs.first() else {
        return ScratchPlace::None;
    };
    if part.pieces == 0 {
        return ScratchPlace::Run {
            first,
            slots: u32::try_from(part.refs.len()).unwrap_or(u32::MAX),
        };
    }
    let spc = chunk_slots.max(1);
    ScratchPlace::Chunks {
        run: part.chunks,
        pieces: part
            .refs
            .iter()
            .step_by((spc * part.chunks.max(1)) as usize)
            .map(|&r| r / spc)
            .collect(),
    }
}

/// The slots taken for a channel about to register:
/// [`Roles::channel_registered`] keeps them, [`Roles::untake_channel`]
/// gives them back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TakenRing(Part);

impl TakenRing {
    /// [`::engine::ChannelRegistration::slots`].
    #[must_use]
    pub fn slots(&self) -> Vec<u32> {
        self.0.refs.clone()
    }
}

/// A park (`out`) or unpark of a process's items: each item's slots on the
/// tier it leaves, paired with the ones taken on the tier it goes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleMove {
    out: bool,
    items: Vec<(Owner, Vec<Part>)>,
    copies: Vec<RoleCopy>,
}

/// One role shape's slots a move copies: device slot `device[i]` and host
/// slot `host[i]`, whichever way the move goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleCopy {
    pub shape: EngineShape,
    pub device: Vec<u32>,
    pub host: Vec<u32>,
}

impl RoleMove {
    /// The copies the move takes, a role shape each.
    #[must_use]
    pub fn copies(&self) -> &[RoleCopy] {
        &self.copies
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct Roles {
    device: Shared,
    host: Option<Shared>,
    shapes: RoleShapes,
    programs: HashMap<ProgramId, ProgramFacts>,
    items: HashMap<Owner, Item>,
    /// The host writes held while their instance's slots are parked or
    /// not yet rebound, in the order they were asked.
    held: Vec<(InstanceId, Pump)>,
    /// The slots of an `Instance` chunk (1 where the pool states none).
    chunk_slots: u32,
    /// The guest (program, extents) pairs whose room has been stated
    /// ([`Self::take_instance`]): each states it once, at its first bind.
    stated: std::collections::HashSet<FireKey>,
    /// Told each time guest state's device room is given back
    /// ([`Self::room_watch`]).
    freed: Arc<Freed>,
}

/// Device room given back: a count of the gives, and their event.
#[derive(Debug, Default)]
pub struct Freed {
    gives: std::sync::atomic::AtomicU64,
    notify: tokio::sync::Notify,
}

/// A watch on device room given back from when it was taken: what a bind
/// short of room waits on ([`bind_when_resident`]).
#[derive(Debug, Clone)]
pub struct RoomWatch {
    freed: Arc<Freed>,
    seen: u64,
}

impl RoomWatch {
    /// Ends once device slots have been given back since the watch was
    /// taken: at once where some were already.
    pub async fn freed(self) {
        use std::sync::atomic::Ordering;
        loop {
            let notified = self.freed.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.freed.gives.load(Ordering::Acquire) != self.seen {
                return;
            }
            notified.await;
        }
    }
}

impl Roles {
    /// The roles of `device`'s pool, parked on `host`; `None` where the pool
    /// states neither shape.
    #[must_use]
    pub fn new(device: &Shared, host: Option<&Shared>) -> Option<Self> {
        let (shapes, chunk_slots) = {
            let pool = device.lock().unwrap();
            let shapes = pool.roles();
            let spc = shapes
                .of(SlotRole::Instance)
                .map_or(1, |shape| pool.chunk_slots(shape.pool));
            (shapes, spc)
        };
        if shapes.is_empty() {
            return None;
        }
        Some(Self {
            device: Arc::clone(device),
            host: host.cloned(),
            shapes,
            programs: HashMap::new(),
            items: HashMap::new(),
            held: Vec::new(),
            chunk_slots,
            stated: std::collections::HashSet::new(),
            freed: Arc::default(),
        })
    }

    /// Whether `holder` holds guest state on the device: a bind of its
    /// short of room would wait on room it holds itself.
    #[must_use]
    pub fn holds_guest_state(&self, holder: Holder) -> bool {
        self.items.values().any(|item| {
            item.holder == Some(holder)
                && item.tier == Tier::Device
                && item
                    .parts
                    .iter()
                    .any(|part| part.dead() && !part.refs.is_empty())
        })
    }

    /// A watch on device room given back from now on.
    #[must_use]
    pub fn room_watch(&self) -> RoomWatch {
        RoomWatch {
            seen: self.freed.gives.load(std::sync::atomic::Ordering::Acquire),
            freed: Arc::clone(&self.freed),
        }
    }

    fn shape(&self, role: SlotRole) -> Result<FixedShape, RolesError> {
        self.shapes.of(role).ok_or(RolesError::NoShape(role))
    }

    /// `slots` of `role` on `on`: any free ones, or (`contiguous`) one run
    /// of adjacent ones in a chunk, never split.
    fn take(
        &self,
        on: &Shared,
        role: SlotRole,
        slots: u32,
        contiguous: bool,
    ) -> Result<Part, RolesError> {
        let part = |refs| Part {
            role,
            refs,
            contiguous,
            pieces: 0,
            chunks: 0,
            run: 0,
        };
        if slots == 0 {
            return Ok(part(Vec::new()));
        }
        let shape = self.shape(role)?;
        let mut pool = on.lock().unwrap();
        if !contiguous {
            return pool
                .take_refs(shape.pool, u64::from(slots))
                .map(part)
                .ok_or(RolesError::Short { role, slots });
        }
        match pool.take_run(shape.pool, u64::from(slots)) {
            Ok(refs) => Ok(part(refs)),
            Err(SlotError::RunPastChunk { per_chunk, .. }) => Err(RolesError::PastChunk {
                role,
                slots,
                slot_bytes: shape.slot_bytes,
                per_chunk,
            }),
            Err(_) => Err(RolesError::NoRun { role, slots }),
        }
    }

    /// The dead part `bytes` of guest scratch take, pieces of `run`
    /// chunks each: one run of `Instance` slots in a chunk where a second
    /// state of its size fits the same chunk beside it, else whole runs of
    /// `run` adjacent chunks. A state two of which do not fit one chunk
    /// gains nothing from a run of slots, its chunk holding no other state
    /// of its program, and only fragments: what it leaves of its chunk
    /// takes some smaller item, which keeps the chunk from coming free, and
    /// a run of that many slots needs a chunk all but empty, which a pool
    /// under pressure seldom has though whole chunks free up as processes
    /// leave. In whole chunks it comes back to any free chunk.
    fn scratch_part(&self, bytes: u64, run: u32) -> Result<Part, RolesError> {
        let shape = self.shape(SlotRole::Instance)?;
        let slot = shape.slot_bytes.max(1);
        let run = run.max(1);
        let slots = bytes.div_ceil(slot);
        let two_share_a_chunk = slots.saturating_mul(2) <= u64::from(self.chunk_slots);
        let (pieces, slots) = if two_share_a_chunk {
            (0, u32::try_from(slots).unwrap_or(u32::MAX))
        } else {
            let chunk = self.device.lock().unwrap().pool().chunk_bytes().max(1);
            let piece = chunk.saturating_mul(u64::from(run));
            (u32::try_from(bytes.div_ceil(piece)).unwrap_or(u32::MAX), 0)
        };
        Ok(Part {
            role: SlotRole::Instance,
            refs: Vec::new(),
            contiguous: true,
            pieces,
            chunks: if pieces > 0 { run } else { 0 },
            run: slots,
        })
    }

    /// Room on the device for `bytes` of guest scratch (`what`) in pieces
    /// of `run` chunks ([`Self::scratch_part`]); all or none. Where no
    /// free run holds it and the segments still to commit within the
    /// budget would, the pool counts it short ([`GUEST_WAITER`]) and the
    /// bind waits on a commit ([`RolesError::GuestGrow`]); else it is
    /// refused by name, with what it asked and what was free.
    fn take_scratch(
        &self,
        bytes: u64,
        run: u32,
        what: impl Fn() -> String,
    ) -> Result<Part, RolesError> {
        let part = self.scratch_part(bytes, run)?;
        let shape = self.shape(SlotRole::Instance)?;
        let room = match self.retake(&part) {
            Ok(part) => {
                self.device.lock().unwrap().pool_mut().unwait(GUEST_WAITER);
                return Ok(part);
            }
            Err(room) => room,
        };
        let (chunks, pieces) = part.chunk_runs();
        let mut device = self.device.lock().unwrap();
        let pool = device.pool_mut();
        let free = match room {
            GuestRoom::Runs { free_runs, .. } => free_runs,
            GuestRoom::Slots { .. } => 0,
        };
        let committable = pool.committable_runs(chunks);
        let room = room.with_committable(committable);
        if free.saturating_add(committable) < pieces {
            pool.unwait(GUEST_WAITER);
            let short = RolesError::GuestShort {
                what: what(),
                bytes,
                room,
            };
            let state_chunks = u64::from(device.state_chunks_now());
            let asks = u64::from(chunks) * u64::from(pieces);
            return Err(if asks <= state_chunks {
                RolesError::BindShort(Box::new(short))
            } else {
                short
            });
        }
        let slots = u64::from(chunks) * u64::from(pieces) * u64::from(self.chunk_slots);
        pool.wait(GUEST_WAITER, shape.pool, slots);
        Err(RolesError::GuestGrow {
            what: what(),
            bytes,
            room,
            run: chunks,
            prefix: pool.prefix_segments(),
        })
    }

    /// Fresh device room for dead part `like`: its run of slots, or its
    /// runs of whole chunks, each inside one segment; all or none, else
    /// what it asked and what was free.
    fn retake(&self, like: &Part) -> Result<Part, GuestRoom> {
        let shape = self.shape(SlotRole::Instance).map_err(|_| like.room())?;
        let mut pool = self.device.lock().unwrap();
        let taken = if like.pieces > 0 {
            pool.take_chunk_runs(shape.pool, like.chunks, like.pieces)
        } else {
            pool.take_run(shape.pool, u64::from(like.run))
        };
        match taken {
            Ok(refs) => Ok(Part {
                refs,
                ..like.clone()
            }),
            Err(SlotError::NoChunkRun {
                run,
                pieces,
                free_runs,
                free_chunks,
            }) => Err(GuestRoom::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                committable: 0,
            }),
            Err(_) => Err(like.room()),
        }
    }

    /// The bytes dead part `part` holds on the device.
    fn dead_bytes(&self, part: &Part) -> u64 {
        if part.pieces > 0 {
            let chunk = self.device.lock().unwrap().pool().chunk_bytes();
            u64::from(part.pieces) * u64::from(part.chunks) * chunk
        } else {
            let slot = self
                .shapes
                .of(part.role)
                .map_or(0, |shape| shape.slot_bytes);
            u64::from(part.run) * slot
        }
    }

    /// The free runs of `run` adjacent `Instance` chunks inside one
    /// segment the device pool holds now: how many more pieces of a guest
    /// state that lies in runs of that length it could place.
    #[must_use]
    pub fn free_runs(&self, run: u32) -> u32 {
        self.device.lock().unwrap().pool().free_chunk_runs(run)
    }

    /// Gives `parts` back on `on`: a failed take's rollback, which frees no
    /// room a waiter lacked (a take is all or none).
    fn give(&self, on: &Shared, parts: &[Part]) {
        let mut pool = on.lock().unwrap();
        for part in parts.iter().filter(|part| !part.refs.is_empty()) {
            if let Some(shape) = self.shapes.of(part.role) {
                pool.give_refs(shape.pool, &part.refs);
            }
        }
    }

    /// Gives back `parts` a closed or parked item held on `on`: guest state
    /// given back on the device is the room a bind short of it waits on
    /// ([`Self::room_watch`]), and its waiters are told.
    fn give_freeing(&self, on: &Shared, parts: &[Part]) {
        self.give(on, parts);
        if Arc::ptr_eq(on, &self.device)
            && parts
                .iter()
                .any(|part| part.dead() && !part.refs.is_empty())
        {
            self.freed
                .gives
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            self.freed.notify.notify_waiters();
        }
    }

    /// Takes the slots of every part, all or none: a whole part's fresh
    /// chunks on the device, nothing off it (`device`: `on` is the device).
    fn take_parts(
        &self,
        on: &Shared,
        like: &[Part],
        device: bool,
    ) -> Result<Vec<Part>, RolesError> {
        let mut taken = Vec::with_capacity(like.len());
        for part in like {
            let took = match (part.dead(), device) {
                (false, _) => self.take(on, part.role, part.refs.len() as u32, part.contiguous),
                (true, true) => self.retake(part).map_err(|room| {
                    let (run, _) = part.chunk_runs();
                    let committable = self.device.lock().unwrap().pool().committable_runs(run);
                    RolesError::GuestShort {
                        what: "a parked guest state coming back".into(),
                        bytes: self.dead_bytes(part),
                        room: room.with_committable(committable),
                    }
                }),
                (true, false) => Ok(Part {
                    refs: Vec::new(),
                    ..part.clone()
                }),
            };
            match took {
                Ok(part) => taken.push(part),
                Err(error) => {
                    self.give(on, &taken);
                    return Err(error);
                }
            }
        }
        Ok(taken)
    }

    // ---- programs and instances ----

    /// The engine registered `program`, whose instances hold `facts`.
    pub fn program_registered(&mut self, program: ProgramId, facts: ProgramFacts) {
        self.programs.insert(program, facts);
    }

    /// Whether `program`'s facts are known here.
    #[must_use]
    pub fn knows(&self, program: ProgramId) -> bool {
        self.programs.contains_key(&program)
    }

    /// The slots an instance of `program` bound to `channels` takes:
    /// `footprint.slots` of the `Instance` shape (cells are packed whole, so
    /// the engine counts, not the bytes), for each ring of the program no
    /// registered channel covers (`registered`), its slots of the `Channel`
    /// shape, and for the adapter it lands (`adapter`, its key) the slots
    /// of the `Adapter` shape another instance already holds for that key,
    /// else fresh ones. A guest program's state ([`GuestState`], of the
    /// pair `pair`): the instance's scratch, whole `Instance` chunks, and
    /// the pair's chunks its other instances hold already, else fresh ones.
    pub fn take_instance(
        &mut self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
        adapter: Option<u128>,
        (pair, state): (FireKey, GuestState),
    ) -> Result<Taken, RolesError> {
        let facts = self
            .programs
            .get(&program)
            .cloned()
            .ok_or(RolesError::Program(program))?;
        let mut shared = Vec::new();
        for &id in channels {
            if let Some(item) = self.items.get(&Owner::Channel(id)) {
                if item.tier != Tier::Device {
                    return Err(RolesError::Parked {
                        channel: id,
                        holder: item.holder,
                    });
                }
                shared.push(id);
            }
        }
        let device = Arc::clone(&self.device);
        let mut taken = Taken {
            channels: shared,
            chunk_slots: self.chunk_slots,
            ..Taken::default()
        };
        let footprint = facts.instance;
        taken.parts.push(self.take(
            &device,
            SlotRole::Instance,
            footprint.slots,
            footprint.contiguous,
        )?);
        for ring in &facts.rings {
            let covered = channels
                .get(ring.channel as usize)
                .is_some_and(|&id| registered(id));
            if covered {
                continue;
            }
            let footprint = ring.footprint;
            match self.take(
                &device,
                SlotRole::Channel,
                footprint.slots,
                footprint.contiguous,
            ) {
                Ok(part) => {
                    taken.parts.push(part);
                    taken.rings.push(ring.channel);
                }
                Err(error) => {
                    self.give(&device, &taken.parts);
                    return Err(error);
                }
            }
        }
        let footprint = facts.adapter;
        if let Some(key) = adapter.filter(|_| footprint.slots > 0) {
            taken.adapter = match self.items.get(&Owner::Adapter(key)) {
                Some(item) if item.tier == Tier::Device => {
                    Some((key, item.parts[0].clone(), false))
                }
                Some(_) => {
                    self.give(&device, &taken.parts);
                    return Err(RolesError::AdapterParked);
                }
                None => match self.take(
                    &device,
                    SlotRole::Adapter,
                    footprint.slots,
                    footprint.contiguous,
                ) {
                    Ok(part) => Some((key, part, true)),
                    Err(error) => {
                        self.give(&device, &taken.parts);
                        return Err(error);
                    }
                },
            };
        }
        let run = state.chunks_a_piece();
        if (state.per_instance > 0 || state.per_pair > 0) && self.stated.insert(pair) {
            self.state_room(pair, state);
        }
        if state.per_instance > 0 {
            match self.take_scratch(state.per_instance, run, || {
                format!("an instance of {pair}'s scratch")
            }) {
                Ok(part) => taken.parts.push(part),
                Err(error) => {
                    self.untake(taken);
                    return Err(error);
                }
            }
        }
        if state.per_pair > 0 {
            taken.pair = match self.items.get(&Owner::Pair(pair)) {
                Some(item) if item.tier == Tier::Device => {
                    Some((pair, item.parts[0].clone(), false))
                }
                Some(_) => {
                    self.untake(taken);
                    return Err(RolesError::PairParked(pair));
                }
                None => {
                    match self.take_scratch(state.per_pair, run, || format!("{pair}'s state")) {
                        Ok(part) => Some((pair, part, true)),
                        Err(error) => {
                            self.untake(taken);
                            return Err(error);
                        }
                    }
                }
            };
        }
        Ok(taken)
    }

    /// States, once a pair, what its guest state asks of the pool and how
    /// many of it the pool holds free as it stands: the run of adjacent
    /// chunks a piece takes, the pieces an instance and the pair take, and
    /// the free runs of that length, so a refusal for want of a run is
    /// visible before it happens.
    fn state_room(&self, pair: FireKey, state: GuestState) {
        let run = state.chunks_a_piece();
        let pieces = |bytes: u64| self.scratch_part(bytes, run).map_or(0, |part| part.pieces);
        tracing::info!(
            %pair,
            per_instance = state.per_instance,
            per_pair = state.per_pair,
            run,
            instance_pieces = pieces(state.per_instance),
            pair_pieces = pieces(state.per_pair),
            free_runs = self.free_runs(run),
            "a guest program's room: pieces of `run` adjacent chunks, and the free runs of that length"
        );
    }

    /// The bind of `taken` failed: its slots go back.
    pub fn untake(&mut self, taken: Taken) {
        self.give(&self.device, &taken.parts);
        if let Some((_, part, true)) = taken.adapter {
            self.give(&self.device, &[part]);
        }
        if let Some((_, part, true)) = taken.pair {
            self.give(&self.device, &[part]);
        }
    }

    /// The engine bound `taken` as instance `id`, for `holder`.
    pub fn bound(&mut self, id: InstanceId, holder: Option<Holder>, taken: Taken) {
        for &channel in &taken.channels {
            if let Some(item) = self.items.get_mut(&Owner::Channel(channel)) {
                item.attached.insert(id);
            }
        }
        let mut item = Item::new(holder, taken.parts);
        item.rings = taken.rings;
        item.channels = taken.channels;
        if let Some((key, part, _)) = taken.adapter {
            let shared = self.items.entry(Owner::Adapter(key)).or_insert_with(|| {
                let mut adapter = Item::new(holder, vec![part]);
                // Held by its instances alone: freed at the last close.
                adapter.open = false;
                adapter
            });
            shared.attached.insert(id);
            item.adapter = Some(key);
        }
        if let Some((key, part, _)) = taken.pair {
            let shared = self.items.entry(Owner::Pair(key)).or_insert_with(|| {
                let mut pair = Item::new(holder, vec![part]);
                // Held by its instances alone: freed at the last close.
                pair.open = false;
                pair
            });
            shared.attached.insert(id);
            item.pair = Some(key);
        }
        if item.parts.iter().all(|part| part.refs.is_empty())
            && item.channels.is_empty()
            && item.adapter.is_none()
            && item.pair.is_none()
        {
            return;
        }
        self.items.insert(Owner::Instance(id), item);
    }

    /// The engine closed instance `id`: its slots go back, and a closed
    /// channel it was the last on with them.
    pub fn close_instance(&mut self, id: InstanceId) {
        let owner = Owner::Instance(id);
        self.held.retain(|&(instance, _)| instance != id);
        let Some(item) = self.items.get_mut(&owner) else {
            return;
        };
        item.open = false;
        let shared: Vec<Owner> = item
            .channels
            .iter()
            .map(|&channel| Owner::Channel(channel))
            .chain(item.adapter.map(Owner::Adapter))
            .chain(item.pair.map(Owner::Pair))
            .collect();
        self.free_if_done(owner);
        for owner in shared {
            if let Some(item) = self.items.get_mut(&owner) {
                item.attached.remove(&id);
            }
            self.free_if_done(owner);
        }
    }

    // ---- channels ----

    /// The slots a channel whose ring `facts` states takes: one run of
    /// adjacent slots where it says `contiguous`.
    pub fn take_channel(&mut self, facts: &RingFacts) -> Result<TakenRing, RolesError> {
        let footprint = facts.footprint;
        self.take(
            &self.device,
            SlotRole::Channel,
            footprint.slots,
            footprint.contiguous,
        )
        .map(TakenRing)
    }

    /// The channel's registration failed: its slots go back.
    pub fn untake_channel(&mut self, taken: TakenRing) {
        self.give(&self.device, &[taken.0]);
    }

    /// The engine registered channel `id` over `taken`, for `holder`.
    pub fn channel_registered(&mut self, id: ChannelId, holder: Option<Holder>, taken: TakenRing) {
        if taken.0.refs.is_empty() {
            return;
        }
        self.items
            .insert(Owner::Channel(id), Item::new(holder, vec![taken.0]));
    }

    /// The engine closed channel `id`: its ring goes back once no bound
    /// instance holds it.
    pub fn close_channel(&mut self, id: ChannelId) {
        let owner = Owner::Channel(id);
        if let Some(item) = self.items.get_mut(&owner) {
            item.open = false;
        }
        self.free_if_done(owner);
    }

    fn free_if_done(&mut self, owner: Owner) {
        let Some(item) = self.items.get(&owner) else {
            return;
        };
        if item.open || !item.attached.is_empty() || item.tier == Tier::Moving {
            return;
        }
        let item = self.items.remove(&owner).expect("present");
        let on = self.tier(item.tier == Tier::Host);
        if let Some(on) = on {
            self.give_freeing(&on, &item.parts);
        }
    }

    // ---- parking ----

    /// Whether `owner` parks with `holder`: an instance it holds, a channel
    /// it registered that only its instances are bound to, or an adapter
    /// every holder of which is `holder`'s or parked already.
    fn moves_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
        let instance = |id: &InstanceId| self.items.get(&Owner::Instance(*id));
        match owner {
            Owner::Instance(_) => item.holder == Some(holder) && item.open,
            Owner::Channel(_) => {
                item.holder == Some(holder)
                    && item.open
                    && item.attached.iter().all(|id| {
                        instance(id).is_some_and(|instance| instance.holder == Some(holder))
                    })
            }
            Owner::Adapter(_) | Owner::Pair(_) => {
                item.attached
                    .iter()
                    .any(|id| instance(id).is_some_and(|instance| instance.holder == Some(holder)))
                    && item.attached.iter().all(|id| {
                        instance(id).is_some_and(|instance| {
                            instance.holder == Some(holder) || instance.tier != Tier::Device
                        })
                    })
            }
        }
    }

    /// Whether `owner`, parked, unparks with `holder`: what it parked, or
    /// an adapter one of its instances holds.
    fn unparks_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
        match owner {
            Owner::Adapter(_) | Owner::Pair(_) => item.attached.iter().any(|id| {
                self.items
                    .get(&Owner::Instance(*id))
                    .is_some_and(|instance| instance.holder == Some(holder))
            }),
            _ => item.holder == Some(holder),
        }
    }

    /// Parks `holder`'s items on the host: host slots taken for each item
    /// that fits there, to copy into. An item the host lacks room for stays
    /// on the device; nothing is dropped. `None` when nothing moves.
    pub fn prepare_park(&mut self, holder: Holder) -> Option<RoleMove> {
        let host = self.host.clone()?;
        let owners: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                item.tier == Tier::Device && self.moves_with(**owner, item, holder)
            })
            .map(|(owner, _)| *owner)
            .collect();
        let mut items = Vec::new();
        for owner in owners {
            let parts = &self.items[&owner].parts;
            match self.take_parts(&host, parts, false) {
                Ok(to) => {
                    self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                    items.push((owner, to));
                }
                Err(error) => {
                    tracing::debug!(?owner, %error, "an item stays on the device: host short");
                }
            }
        }
        (!items.is_empty()).then(|| self.moving(true, items))
    }

    /// Unparks `holder`'s parked items: device slots taken for each, all or
    /// none. `None` when nothing is parked.
    pub fn prepare_unpark(&mut self, holder: Holder) -> Result<Option<RoleMove>, RolesError> {
        let owners: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                item.tier == Tier::Host && self.unparks_with(**owner, item, holder)
            })
            .map(|(owner, _)| *owner)
            .collect();
        let device = Arc::clone(&self.device);
        let mut items: Vec<(Owner, Vec<Part>)> = Vec::new();
        for owner in &owners {
            match self.take_parts(&device, &self.items[owner].parts, true) {
                Ok(to) => items.push((*owner, to)),
                Err(error) => {
                    for (_, parts) in &items {
                        self.give(&device, parts);
                    }
                    // Short of room the budget holds: the restore waits.
                    if error.is_short() && self.fits_budget(&owners) {
                        return Err(RolesError::UnparkShort(Box::new(error)));
                    }
                    return Err(error);
                }
            }
        }
        for (owner, _) in &items {
            self.items.get_mut(owner).expect("present").tier = Tier::Moving;
        }
        Ok((!items.is_empty()).then(|| self.moving(false, items)))
    }

    /// Whether the items of `owners` together fit the device pool's state
    /// chunks at all, whole chunks for each part: room a completion or an
    /// eviction can make.
    fn fits_budget(&self, owners: &[Owner]) -> bool {
        let device = self.device.lock().unwrap();
        let mut chunks = 0u64;
        for owner in owners {
            for part in &self.items[owner].parts {
                chunks += if part.dead() {
                    let (run, pieces) = part.chunk_runs();
                    u64::from(run) * u64::from(pieces)
                } else {
                    let per_chunk = self
                        .shapes
                        .of(part.role)
                        .and_then(|shape| device.pool().shape(shape.pool))
                        .map_or(1, |shape| shape.slots_per_chunk.max(1));
                    (part.refs.len() as u64).div_ceil(u64::from(per_chunk))
                };
            }
        }
        chunks <= u64::from(device.state_chunks_now())
    }

    /// The move of `items` (each item's slots to take), with the copies
    /// from the slots each holds now.
    fn moving(&self, out: bool, items: Vec<(Owner, Vec<Part>)>) -> RoleMove {
        let mut copies: Vec<RoleCopy> = Vec::new();
        for (owner, to) in &items {
            let from = &self.items[owner].parts;
            for (from, to) in from.iter().zip(to) {
                // Scratch: dead between fires, nothing to copy.
                if from.dead() {
                    continue;
                }
                let Some(shape) = self.shapes.of(from.role) else {
                    continue;
                };
                let (device, host) = if out {
                    (&from.refs, &to.refs)
                } else {
                    (&to.refs, &from.refs)
                };
                let copy = match copies.iter_mut().find(|copy| copy.shape == shape.id) {
                    Some(copy) => copy,
                    None => {
                        copies.push(RoleCopy {
                            shape: shape.id,
                            device: Vec::new(),
                            host: Vec::new(),
                        });
                        copies.last_mut().expect("pushed")
                    }
                };
                copy.device.extend_from_slice(device);
                copy.host.extend_from_slice(host);
            }
        }
        copies.retain(|copy| !copy.device.is_empty());
        RoleMove { out, items, copies }
    }

    /// The copies landed: each item lies at its new slots, and its old ones
    /// go back. An unparked item is stale until [`Self::rebound`].
    pub fn commit(&mut self, txn: RoleMove) {
        let left = self.tier(!txn.out);
        for (owner, parts) in txn.items {
            let Some(item) = self.items.get_mut(&owner) else {
                continue;
            };
            let old = std::mem::replace(&mut item.parts, parts);
            item.tier = if txn.out { Tier::Host } else { Tier::Device };
            item.stale = !txn.out;
            if let Some(left) = &left {
                self.give_freeing(left, &old);
            }
            self.free_if_done(owner);
        }
    }

    /// The move failed: each item stays where it was, and the slots taken
    /// for it go back.
    pub fn abort(&mut self, txn: RoleMove) {
        let taken_on = self.tier(txn.out);
        for (owner, parts) in txn.items {
            if let Some(on) = &taken_on {
                self.give(on, &parts);
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = if txn.out { Tier::Device } else { Tier::Host };
            }
            self.free_if_done(owner);
        }
    }

    /// The host tier (`host`), else the device.
    fn tier(&self, host: bool) -> Option<Shared> {
        if host {
            self.host.clone()
        } else {
            Some(Arc::clone(&self.device))
        }
    }

    /// `holder`'s items whose slots moved since the engine last named them:
    /// what [`::engine::Engine::rebind_slots`] tells it, channels first (an
    /// instance's tables may address their rings).
    #[must_use]
    pub fn stale(&self, holder: Holder) -> Vec<Rebind> {
        // An adapter that moved rebinds every instance holding it.
        let mut instances: BTreeSet<InstanceId> = BTreeSet::new();
        let mut channels: BTreeSet<ChannelId> = BTreeSet::new();
        for (owner, item) in &self.items {
            match *owner {
                Owner::Channel(id) if item.stale && item.holder == Some(holder) => {
                    channels.insert(id);
                }
                Owner::Instance(id) if item.stale && item.holder == Some(holder) => {
                    instances.insert(id);
                }
                // Its holders on the device; a parked one is rebound when it
                // unparks.
                Owner::Adapter(_) | Owner::Pair(_)
                    if item.stale && self.unparks_with(*owner, item, holder) =>
                {
                    instances.extend(item.attached.iter().copied().filter(|id| {
                        self.items
                            .get(&Owner::Instance(*id))
                            .is_some_and(|instance| instance.tier == Tier::Device)
                    }));
                }
                _ => {}
            }
        }
        channels
            .into_iter()
            .filter_map(|id| {
                let item = self.items.get(&Owner::Channel(id))?;
                Some(Rebind::Channel {
                    id,
                    slots: item.refs(),
                })
            })
            .chain(instances.into_iter().filter_map(|id| {
                let item = self.items.get(&Owner::Instance(id))?;
                Some(Rebind::Instance {
                    id,
                    instance: item.parts.first().map_or_else(Vec::new, |p| p.refs.clone()),
                    rings: own_rings(&item.rings, &item.parts),
                    adapter: self.adapter_refs(item),
                    scratch: scratch(&item.parts)
                        .map_or(ScratchPlace::None, |part| place(part, self.chunk_slots)),
                    pair: item
                        .pair
                        .and_then(|key| self.items.get(&Owner::Pair(key)))
                        .map_or(ScratchPlace::None, |pair| {
                            place(&pair.parts[0], self.chunk_slots)
                        }),
                })
            }))
            .collect()
    }

    /// The `Adapter` slots the instance `item` lands its adapter in.
    fn adapter_refs(&self, item: &Item) -> Vec<u32> {
        item.adapter
            .and_then(|key| self.items.get(&Owner::Adapter(key)))
            .map_or_else(Vec::new, Item::refs)
    }

    /// Whether a host write into `instance`'s rings must wait: its own
    /// slots, or a registered channel's it is bound to, are parked, moving,
    /// or back but not yet rebound.
    #[must_use]
    pub fn writes_held(&self, instance: InstanceId) -> bool {
        let away = |owner: Owner| {
            self.items
                .get(&owner)
                .is_some_and(|item| item.tier != Tier::Device || item.stale)
        };
        let Some(item) = self.items.get(&Owner::Instance(instance)) else {
            return false;
        };
        away(Owner::Instance(instance))
            || item
                .channels
                .iter()
                .any(|&channel| away(Owner::Channel(channel)))
    }

    /// Holds `pump` of `instance` until its slots are back and rebound.
    pub fn hold(&mut self, instance: InstanceId, pump: Pump) {
        if self.held.last() != Some(&(instance, pump)) {
            self.held.push((instance, pump));
        }
    }

    /// The held writes whose instances may write again, in the order they
    /// were held; the rest stay held.
    pub fn released(&mut self) -> Vec<(InstanceId, Pump)> {
        let held = std::mem::take(&mut self.held);
        let (keep, go): (Vec<_>, Vec<_>) = held
            .into_iter()
            .partition(|&(instance, _)| self.writes_held(instance));
        self.held = keep;
        go
    }

    /// The engine took `owner`'s new slots.
    pub fn rebound(&mut self, owner: Owner) {
        let (adapter, pair) = match self.items.get_mut(&owner) {
            Some(item) => {
                item.stale = false;
                (item.adapter, item.pair)
            }
            None => (None, None),
        };
        // An instance rebound names its adapter's and its pair's new slots
        // too.
        if let Some(item) = adapter.and_then(|key| self.items.get_mut(&Owner::Adapter(key))) {
            item.stale = false;
        }
        if let Some(item) = pair.and_then(|key| self.items.get_mut(&Owner::Pair(key))) {
            item.stale = false;
        }
    }

    /// Bytes of the items' slots on the device: state, none of it cache.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.items
            .values()
            .filter(|item| item.tier != Tier::Host)
            .flat_map(|item| &item.parts)
            .map(|part| {
                part.refs.len() as u64 * self.shapes.of(part.role).map_or(0, |s| s.slot_bytes)
            })
            .sum()
    }

    /// The slots `owner` holds now, in binding order; none where it holds
    /// none.
    #[cfg(test)]
    #[must_use]
    pub fn slots(&self, owner: Owner) -> Vec<u32> {
        self.items.get(&owner).map_or_else(Vec::new, Item::refs)
    }
}

/// Asks `engine` what an instance of `program` holds, where `roles` has
/// not yet: once a program, at its registration.
pub fn learn_program(
    roles: &RolesHandle,
    engine: &dyn Engine,
    program: ProgramId,
) -> ::engine::Result<()> {
    if roles.lock().unwrap().knows(program) {
        return Ok(());
    }
    let facts = engine.program_facts(program)?;
    roles.lock().unwrap().program_registered(program, facts);
    Ok(())
}

/// Registers channel `plan` by `register`, over the `Channel` slots
/// [`Engine::ring_facts`] states its ring takes ([`ChannelRegistration::slots`]),
/// held for `holder` until it closes. Without `roles`, `plan` as it is.
///
/// [`Engine::ring_facts`]: ::engine::Engine::ring_facts
/// [`ChannelRegistration::slots`]: ::engine::ChannelRegistration::slots
pub fn register_channel<T>(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
    holder: Option<Holder>,
    register: impl FnOnce(&mut dyn Engine, &ChannelRegistration) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let Some(roles) = roles else {
        return register(engine, plan);
    };
    let Some(facts) = engine.ring_facts(plan)? else {
        return register(engine, plan);
    };
    let taken = roles.lock().unwrap().take_channel(&facts)?;
    let plan = ChannelRegistration {
        slots: taken.slots(),
        ..plan.clone()
    };
    match register(engine, &plan) {
        Ok(registered) => {
            roles
                .lock()
                .unwrap()
                .channel_registered(plan.id, holder, taken);
            Ok(registered)
        }
        Err(error) => {
            roles.lock().unwrap().untake_channel(taken);
            Err(error)
        }
    }
}

/// Binds `binding` on `engine` over the `Instance` and `Channel` slots its
/// program's facts state, and the whole chunks of its guest state
/// ([`Engine::guest_state`], asked at every bind), held for `holder` until
/// the instance closes; a ring of a channel `registered` names is the
/// channel's. A pool short of them refuses the bind by name. Without
/// `roles`, `binding` as it is: the engine keeps it all itself.
///
/// [`Engine::guest_state`]: ::engine::Engine::guest_state
pub fn bind_instance(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    holder: Option<Holder>,
    registered: impl Fn(ChannelId) -> bool,
) -> anyhow::Result<BoundInstance> {
    let Some(roles) = roles else {
        return Ok(engine.bind_instance(binding)?);
    };
    learn_program(roles, engine, binding.program)?;
    let adapter = engine.adapter_key(binding)?;
    let state = engine.guest_state(binding.program, &binding.extents)?;
    let pair = FireKey {
        program: binding.program,
        extents: binding.extents,
    };
    let taken = roles.lock().unwrap().take_instance(
        binding.program,
        &binding.channels,
        registered,
        adapter,
        (pair, state),
    )?;
    let binding = InstanceBinding {
        instance_slots: taken.instance_slots(),
        ring_slots: taken.ring_slots(),
        adapter_slots: taken.adapter_slots(),
        scratch: taken.scratch(),
        pair: taken.pair(),
        ..binding.clone()
    };
    match engine.bind_instance(&binding) {
        Ok(bound) => {
            roles.lock().unwrap().bound(bound.id, holder, taken);
            Ok(bound)
        }
        Err(error) => {
            roles.lock().unwrap().untake(taken);
            Err(error.into())
        }
    }
}

/// Runs `bind`, a register+bind of process `pid`'s, until it lands or is
/// refused by name. Refused because a channel it binds is parked with
/// `pid` (suspended, its rings with it), it waits by `resident`, the wait a
/// fire takes for `pid` to be resident again (the planner's restore), and
/// binds again: a suspension holds a bind back as it holds a fire, and
/// never refuses it. A channel parks with `pid` only while `pid` is not
/// resident, so each bind again follows a restore. A wait that ends
/// otherwise (the process failed or ended) is the bind's refusal.
///
/// Short of a free run for its guest state where a segment still to
/// commit within the budget holds one ([`RolesError::GuestGrow`]), it
/// waits by `grown(prefix, run)` ([`grow_for_guest`]: the walk commits a
/// segment past the committed prefix `prefix`, as it does for a kv
/// admission short of room) and binds again; a wait that ends with no
/// segment left to commit is the bind's refusal.
///
/// This is not a retry: nothing is attempted again on the chance it now
/// fits, and nothing is degraded or made smaller. The bind is sent again
/// only after the planner has restored `pid`, or the walk has committed a
/// segment, the event the refusal named, exactly as a fire held back by a
/// suspension is sent after its restore.
pub async fn bind_when_resident<T, E, B, W, G>(
    pid: Holder,
    mut bind: impl FnMut() -> B,
    resident: impl Fn() -> W,
    grown: impl Fn(u32, u32) -> G,
    room: impl Fn() -> Option<RoomWatch>,
    suspend: impl Fn() -> bool,
) -> anyhow::Result<T>
where
    E: std::fmt::Display,
    B: std::future::Future<Output = anyhow::Result<T>>,
    W: std::future::Future<Output = Result<(), E>>,
    G: std::future::Future<Output = Result<(), String>>,
{
    loop {
        // Taken before the bind: room given back meanwhile is not lost.
        let watch = room();
        let error = match bind().await {
            Ok(bound) => return Ok(bound),
            Err(error) => error,
        };
        if matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::BindShort(_))
        ) {
            // Holding guest state of its own, it would wait on itself: it is
            // suspended, its state given back, and binds once restored.
            if suspend() {
                tracing::debug!(%pid, %error, "a bind short of room suspends its process");
                if let Err(why) = resident().await {
                    return Err(error.context(format!(
                        "its process was suspended for room and is not resident again: {why}"
                    )));
                }
                continue;
            }
            if let Some(watch) = watch {
                tracing::debug!(%pid, %error, "a bind waits for room given back");
                watch.freed().await;
                continue;
            }
        }
        if let Some(&RolesError::GuestGrow { prefix, run, .. }) = error.downcast_ref::<RolesError>()
        {
            tracing::debug!(%pid, %error, "a bind waits on a segment commit");
            if let Err(why) = grown(prefix, run).await {
                return Err(error.context(format!("refused: {why}")));
            }
            continue;
        }
        let parked_with_it = matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::Parked { holder: Some(holder), .. }) if *holder == pid
        );
        if !parked_with_it {
            return Err(error);
        }
        tracing::debug!(%pid, %error, "a bind waits for its process to be resident again");
        if let Err(why) = resident().await {
            return Err(error.context(format!(
                "its process was suspended and is not resident again: {why}"
            )));
        }
    }
}

/// Waits until the walk of `engine`'s pool `shared` commits a segment past
/// the committed prefix `prefix`, for a guest bind short of a free run of
/// `run` chunks ([`RolesError::GuestGrow`]): the walk is asked to look
/// ([`crate::store::shared::ask_walk`]) and the wait is on the event a
/// commit tells ([`crate::store::shared::rung_watch`]), never a clock; the
/// walk's goal counts what the bind is short ([`GUEST_WAITER`]), within the
/// governor's target and yield, as it counts a kv admission's. Refused
/// (`Err`) once no segment is left to commit within the budget, the
/// pool's count of it cleared.
pub async fn grow_for_guest(
    shared: &Shared,
    engine: usize,
    prefix: u32,
    run: u32,
) -> Result<(), String> {
    loop {
        // Armed before the look: a commit between the two is not lost.
        let watch = crate::store::shared::rung_watch(engine);
        tokio::pin!(watch);
        watch.as_mut().enable();
        {
            let mut pool = shared.lock().unwrap();
            if pool.pool().prefix_segments() > prefix {
                return Ok(());
            }
            if pool.pool().committable_runs(run) == 0 {
                pool.pool_mut().unwait(GUEST_WAITER);
                return Err(format!(
                    "no free run of {run} adjacent chunks, and no segment is left to commit \
                     within the budget"
                ));
            }
        }
        crate::store::shared::ask_walk(engine);
        watch.await;
    }
}

/// Runs `write`, a host write into `instance`'s rings, unless its slots
/// are away ([`Roles::writes_held`]): then it is held, to be replayed in
/// order once they are rebound ([`rebind`]).
pub fn pump(
    roles: Option<&RolesHandle>,
    instance: InstanceId,
    pump: Pump,
    write: impl FnOnce() -> ::engine::Result<()>,
) -> ::engine::Result<()> {
    if let Some(roles) = roles {
        let mut roles = roles.lock().unwrap();
        if roles.writes_held(instance) {
            roles.hold(instance, pump);
            return Ok(());
        }
    }
    write()
}

/// Tells `engine` where moved slots lie now (`rebind`), then replays, in
/// order, the host writes held for the instances that may write again.
/// Before any later fire: the lane runs it between frames.
pub fn rebind(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    rebind: Rebind,
    mut replay: impl FnMut(&mut dyn Engine, InstanceId, Pump) -> ::engine::Result<()>,
) -> ::engine::Result<()> {
    let owner = Owner::of(&rebind);
    engine.rebind_slots(rebind)?;
    let Some(roles) = roles else {
        return Ok(());
    };
    let released = {
        let mut roles = roles.lock().unwrap();
        roles.rebound(owner);
        roles.released()
    };
    for (instance, pump) in released {
        replay(engine, instance, pump)?;
    }
    Ok(())
}

/// An instance's own rings as the engine names them: channel `rings[k]` in
/// part `k + 1` (part 0 the tables).
fn own_rings(rings: &[u32], parts: &[Part]) -> Vec<RingSlots> {
    rings
        .iter()
        .zip(parts.iter().skip(1))
        .map(|(&channel, part)| RingSlots {
            channel,
            slots: part.refs.clone(),
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod guest_tests;
