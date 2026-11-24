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
//! refs ([`Engine::rebind_slots`]) before the next fire reads them. What
//! the host has no room for at the park stays on the device, parked, until
//! the device wants its room ([`Roles::prepare_park_down`]): then it goes
//! on down to the host, or, where the load states no host tier (unified
//! memory), to the disk tier's slot file, a slot of it a role slot
//! ([`DiskSlots`]).
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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use ::engine::chunk::{ShapeId as EngineShape, SlotRole};
use ::engine::{
    BoundInstance, ChannelId, ChannelRegistration, Engine, GuestState, InstanceBinding, InstanceId,
    ProgramFacts, ProgramId, Rebind, RingFacts, RingSlots, ScratchPlace,
};

use super::shared::programs::FireKey;
use super::shared::{FixedShape, RoleShapes, Shared};
use super::slots::SlotError;

mod tiers;
mod census;
mod room;
mod bind;
mod park;
mod budget;
mod txn;
mod rebind;
mod calls;
pub use calls::*;

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
    /// Its place in an order of owners: kind, then id.
    fn order(&self) -> (u8, u128, [u32; 5]) {
        match self {
            Owner::Instance(id) => (0, u128::from(id.0), [0; 5]),
            Owner::Channel(id) => (1, u128::from(id.0), [0; 5]),
            Owner::Adapter(key) => (2, *key, [0; 5]),
            Owner::Pair(key) => {
                let at = key.extents;
                (
                    3,
                    u128::from(key.program.0),
                    [at.row_count, at.token_count, at.sampled_rows, at.query_len, at.key_len],
                )
            }
        }
    }
}

impl PartialOrd for Owner {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Owner {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order().cmp(&other.order())
    }
}

// The items by owner: under test in owner order, so a seeded simulation's
// parks and unparks lay their slots the same on each run; in service they
// hash.
#[cfg(test)]
type Items = std::collections::BTreeMap<Owner, Item>;
#[cfg(not(test))]
type Items = HashMap<Owner, Item>;

impl Owner {
    /// Whose slots `rebind` moves.
    #[must_use]
    pub fn of(rebind: &Rebind) -> Self {
        match rebind {
            Rebind::Instance { id, .. } | Rebind::Parked { id, .. } => Owner::Instance(*id),
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
    /// The process binding is not resident (evicted, or on its way off or
    /// back onto the device): a bind of it waits for it to be resident
    /// again ([`bind_when_resident`]), as one on a channel parked with it
    /// does. Bound meanwhile, its instance stood on the device outside its
    /// process's park, never said parked to the engine, while the pair it
    /// joined went down with a later move of that process's and its run was
    /// taken by another (the simulation's seed 1116: a rescore's bind on a
    /// process evicted with its first program closed, holding no item the
    /// park could take).
    #[error("its process is parked or moving")]
    HolderParked(Holder),
    #[error("the adapter this binding lands is parked with the instances that hold it")]
    AdapterParked,
    /// The pair's state is on its way to or from the host with the
    /// instances that hold it: a bind of it waits for the move to land
    /// ([`Roles::room_watch`]), and an unpark is queued again.
    #[error("{0}'s state is moving with the instances that hold it")]
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

/// Whether `error` is a bind waiting for room for its guest state
/// ([`RolesError::BindShort`], [`RolesError::GuestGrow`]): nothing was
/// registered for it, and it binds again once room is there.
#[must_use]
pub fn waits_for_room(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<RolesError>(),
        Some(
            RolesError::BindShort(_) | RolesError::GuestGrow { .. }
        )
    )
}

impl RolesError {
    /// The room a bind short of it waits for (`BindShort`), as `(run,
    /// pieces)`: `pieces` free runs of `run` adjacent chunks. One free
    /// chunk holds a slot item, whatever its run of slots.
    #[must_use]
    pub fn room_runs(&self) -> Option<(u32, u32, u32)> {
        let Self::BindShort(short) = self else {
            return None;
        };
        match short.as_ref() {
            Self::GuestShort { room, .. } => {
                let (run, pieces) = room.chunk_runs();
                Some((run, pieces, room.singles()))
            }
            Self::Short { .. } | Self::NoRun { .. } => Some((1, 1, 0)),
            _ => None,
        }
    }

    /// The free runs of its need's length the pool held when an unpark was
    /// refused short of runs of whole chunks ([`Self::UnparkShort`]): the
    /// refusal's own view. `None` for a run of slots, or another refusal.
    #[must_use]
    pub fn unpark_free_runs(&self) -> Option<u32> {
        let Self::UnparkShort(short) = self else {
            return None;
        };
        match short.as_ref() {
            Self::GuestShort {
                room: GuestRoom::Runs { free_runs, .. },
                ..
            }
            | Self::GuestGrow {
                room: GuestRoom::Runs { free_runs, .. },
                ..
            } => Some(*free_runs),
            _ => None,
        }
    }

    /// What an unpark short of room ([`Self::UnparkShort`]) needs, as runs
    /// of whole chunks `(run, pieces)`: a guest state's runs, or one chunk
    /// for slots in a chunk (a run of Channel slots too).
    #[must_use]
    pub fn unpark_runs(&self) -> Option<(u32, u32)> {
        let Self::UnparkShort(short) = self else {
            return None;
        };
        match short.as_ref() {
            Self::GuestShort { room, .. } | Self::GuestGrow { room, .. } => Some(room.chunk_runs()),
            Self::Short { .. } | Self::NoRun { .. } => Some((1, 1)),
            _ => None,
        }
    }

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
    /// One run of `slots` adjacent `Instance` slots in a chunk; `chunks`,
    /// the fresh chunks the whole bind asks with it (each of its parts a
    /// run of slots, the part short first among them), at least one.
    Slots {
        slots: u32,
        committable: u32,
        chunks: u32,
    },
    /// `pieces` runs of `run` adjacent whole chunks, each inside one
    /// segment, and the free runs of that length and free chunks there
    /// were.
    Runs {
        run: u32,
        pieces: u32,
        free_runs: u32,
        free_chunks: u32,
        committable: u32,
        /// Single chunks the bind opens beside its runs: its tables, rings
        /// and adapter, each role's slots in chunks of their own. Asked as
        /// runs of `run`, one short chunk was a whole run more, and a bind
        /// that fits the device alone was refused as never fitting.
        singles: u32,
    },
}

impl GuestRoom {
    /// The runs of chunks it asks, `(run, pieces)`: a run of slots takes
    /// one chunk.
    #[must_use]
    pub fn chunk_runs(self) -> (u32, u32) {
        match self {
            Self::Slots { chunks, .. } => (1, chunks.max(1)),
            Self::Runs { run, pieces, .. } => (run, pieces),
        }
    }

    /// The single chunks it asks beside its runs.
    #[must_use]
    pub fn singles(self) -> u32 {
        match self {
            Self::Slots { .. } => 0,
            Self::Runs { singles, .. } => singles,
        }
    }

    fn committable(self) -> u32 {
        match self {
            Self::Slots { committable, .. } | Self::Runs { committable, .. } => committable,
        }
    }

    /// The same room, asking `whole` runs of its run where that is more
    /// than it named, and `beside` single chunks for the bind's tables,
    /// rings and adapter: what the whole bind takes, not the part that came
    /// up short first.
    fn asking(self, whole: u32, beside: u32) -> Self {
        match self {
            Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                committable,
                singles,
            } => Self::Runs {
                run,
                pieces: pieces.max(whole),
                free_runs,
                free_chunks,
                committable,
                singles: singles.max(beside),
            },
            // Each of the bind's parts a run of slots opening a chunk of its
            // own: the room asked is every one of them, never the part short
            // first alone (seed 16 of the simulation: an instance's and its
            // pair's runs each opening a chunk, refused as one, served one
            // chunk, short again, a spin).
            Self::Slots {
                slots,
                committable,
                chunks,
            } => Self::Slots {
                slots,
                committable,
                chunks: chunks.max(whole.saturating_add(beside)),
            },
        }
    }

    fn with_committable(self, runs: u32) -> Self {
        match self {
            Self::Slots { slots, chunks, .. } => Self::Slots {
                slots,
                committable: runs,
                chunks,
            },
            Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                singles,
                ..
            } => Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                committable: runs,
                singles,
            },
        }
    }
}

impl RolesError {
    /// A bind's refusal for want of room, asking `whole` runs: every guest
    /// part the bind takes fresh, so the room its wait (or its suspended
    /// process's restore) is served on holds the whole bind. Asked for the
    /// part short first, the restore came back once that stood, the bind
    /// took it for another part, came up short again and was suspended
    /// again (bind, suspend, restore, short).
    fn asking(self, whole: u32, beside: u32) -> Self {
        match self {
            Self::BindShort(short) => Self::BindShort(Box::new(short.asking(whole, beside))),
            Self::GuestShort { what, bytes, room } => Self::GuestShort {
                what,
                bytes,
                room: room.asking(whole, beside),
            },
            Self::GuestGrow {
                what,
                bytes,
                room,
                run,
                prefix,
            } => Self::GuestGrow {
                what,
                bytes,
                room: room.asking(whole, beside),
                run,
                prefix,
            },
            other => other,
        }
    }
}

impl std::fmt::Display for GuestRoom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Slots { slots, chunks, .. } => {
                write!(f, "no chunk has a run of {slots} free slots")?;
                if *chunks > 1 {
                    write!(f, " (the bind opens {chunks} chunks)")?;
                }
            }
            Self::Runs {
                run,
                pieces,
                free_runs,
                free_chunks,
                singles,
                ..
            } => {
                write!(
                    f,
                    "{pieces} runs of {run} adjacent chunks inside one segment asked, and \
                     {free_runs} such runs were free ({free_chunks} free chunks)"
                )?;
                if *singles > 0 {
                    write!(f, ", and {singles} single chunks beside them")?;
                }
            }
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
    /// In the disk tier's slot file, where no host tier holds it
    /// ([`Roles::prepare_park_down`]): its live parts' refs are slots of
    /// the file ([`DiskSlots`]).
    Disk,
    /// A park or unpark is copying it: neither side's slots are free.
    Moving,
}

/// One role's slots of an item, as refs of the tier the item lies on;
/// `contiguous`, adjacent in one chunk ([`::engine::Footprint::contiguous`]),
/// and taken so again on every tier it moves to. Guest scratch, whose
/// contents are dead between fires, is never copied and holds nothing off
/// the device (`refs` empty there), fresh room taken when it comes back:
/// `pieces` runs of `chunks` adjacent whole chunks, or (`run`) one run of
/// that many slots. A guest part holding resident bytes (`held`,
/// `engine::GuestState::resident`) is laid the same way and copied as a
/// live part is: down a tier at a park, back into fresh runs at an unpark.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Part {
    role: SlotRole,
    refs: Vec<u32>,
    contiguous: bool,
    pieces: u32,
    chunks: u32,
    run: u32,
    held: bool,
}

/// `parts` as a trace names them: each part's role and slot refs.
fn traced(parts: &[Part]) -> String {
    parts
        .iter()
        .map(|part| {
            let dead = if part.dead() { "(dead)" } else { "" };
            format!("{:?}{dead}{:?}", part.role, part.refs)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl Part {
    /// Guest scratch, dead between fires: given back at a park, never
    /// copied.
    fn dead(&self) -> bool {
        self.shaped() && !self.held
    }

    /// A guest part, laid on the device as runs of whole chunks or a run
    /// of slots ([`Roles::retake`]), dead or held.
    fn shaped(&self) -> bool {
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
                singles: 0,
            }
        } else {
            GuestRoom::Slots {
                slots: self.run,
                committable: 0,
                chunks: 1,
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
    /// Parked with no tier below to hold its live parts: they stayed where
    /// they lie, its dead parts (guest scratch, dead between fires) were
    /// given back, and its restore takes fresh ones in their class
    /// ([`Roles::prepare_unpark`]). Its live parts go on down once the
    /// device wants their room ([`Roles::prepare_park_down`]).
    shed: bool,
    /// Frames on their way through the engine that name it (an instance
    /// they attach, or the pair, channels and adapter one of those shares):
    /// let go meanwhile, it is given back only once the last of them has
    /// landed ([`Roles::landed`]), never under a fire still reading it.
    reads: u32,
    /// A pair or an adapter: binds taken on it and not yet bound or given
    /// back ([`Roles::take_instance`] to [`Roles::bound`] or
    /// [`Roles::untake`]). The taker is no member yet, so it is in no
    /// `attached`: unheld by this, the item would go with its last member's
    /// park (or be given back as unheld) while the bind is on its way, and
    /// the bind would name slots another owner takes next (`guest::claims`
    /// refusing an instance bound over a live pair's run).
    binds: u32,
    /// Which taking of its slots it lies in: a new one each time it is
    /// given fresh slots ([`Roles::next_generation`]). A pair's is the
    /// engine's to compare ([`::engine::InstanceBinding::pair_generation`]):
    /// slots freed, written by another owner and taken again at the same
    /// refs are new.
    generation: u64,
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
            shed: false,
            reads: 0,
            binds: 0,
            generation: 0,
        }
    }

    /// On the device and short of a dead part (guest scratch, a pair's
    /// state) its rebind must name: parked where it lies ([`Self::shed`]),
    /// or a dead part with no slots. Its unpark takes fresh ones in their
    /// class before its restore commits, the runs counted in its need
    /// ([`Roles::unpark_need`]): let through, the restore would commit and
    /// wait on a rebind no room is made for, its process Restoring until
    /// teardown.
    fn takes_fresh(&self) -> bool {
        self.tier == Tier::Device
            && (self.shed || self.parts.iter().any(|part| part.dead() && part.refs.is_empty()))
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
    /// The guest pair's state: its key, its chunks, whether they were
    /// taken for this bind (the pair's first) or are its instances', and
    /// the taking they are ([`Item::generation`]).
    pair: Option<(FireKey, Part, bool, u64)>,
    /// The slots of an `Instance` chunk: a whole part's chunks are its
    /// refs over it.
    chunk_slots: u32,
    /// The process whose bind took it ([`Roles::binds_on_way`]).
    holder: Option<Holder>,
    /// The bytes it took for itself, counted on its way in
    /// [`Roles::footprint`] until it is bound or given back.
    bytes: u64,
}

/// A bind's guest state taken before it registered anything
/// ([`Roles::take_guest`]): its scratch, and its pair's state with whether
/// it was taken for this bind. [`Roles::take_instance`] takes it into the
/// bind's [`Taken`]; [`Roles::untake_guest`] gives it back.
#[derive(Debug, Default)]
pub struct GuestTaken {
    scratch: Option<Part>,
    pair: Option<(FireKey, Part, bool, u64)>,
    /// Its pair came back from a park for this bind: fresh room, kept on
    /// the device whatever the bind's fate (the pair item's own).
    pair_retaken: bool,
}

impl GuestTaken {
    /// The runs of chunks its fresh parts took, `(run, pieces)` summed at
    /// the longest run: what a bind short of room for its tables beside
    /// them asks with its own chunk ([`Roles::bind_short`]).
    fn fresh_runs(&self) -> Option<(u32, u32)> {
        let parts: Vec<(u32, u32)> = self
            .scratch
            .iter()
            .chain(
                self.pair
                    .iter()
                    .filter(|(_, _, fresh, _)| *fresh || self.pair_retaken)
                    .map(|(_, part, ..)| part),
            )
            .map(Part::chunk_runs)
            .collect();
        let longest = parts.iter().map(|&(run, _)| run).max()?;
        Some((
            longest,
            parts
                .iter()
                .map(|&(run, pieces)| run.div_ceil(longest).saturating_mul(pieces))
                .sum(),
        ))
    }
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
            .map_or(ScratchPlace::None, |(_, part, _, _)| {
                place(part, self.chunk_slots)
            })
    }

    /// [`::engine::InstanceBinding::pair_generation`].
    #[must_use]
    pub fn pair_generation(&self) -> u64 {
        self.pair
            .as_ref()
            .map_or(0, |&(_, _, _, generation)| generation)
    }
}

/// An instance's scratch: its dead part.
fn scratch(parts: &[Part]) -> Option<&Part> {
    parts.iter().find(|part| part.shaped())
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
    /// Each item's tier off the device, in item order: the one it goes to
    /// (`out`) or comes from.
    below: Vec<Tier>,
    copies: Vec<RoleCopy>,
    /// Items whose live parts stay where they lie, each with its parts after
    /// the move: parking, its dead parts given back (no refs); unparking,
    /// its dead parts taken fresh. Nothing of them is copied.
    shed: Vec<(Owner, Vec<Part>)>,
}

/// One role shape's slots a move copies: device slot `device[i]` and host
/// slot `host[i]` (`disk`: slot `host[i]` of the disk tier's slot file),
/// whichever way the move goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleCopy {
    pub shape: EngineShape,
    pub device: Vec<u32>,
    pub host: Vec<u32>,
    pub disk: bool,
}

/// The disk tier's slot file, for the live parts of parked items no host
/// tier holds (a load on unified memory states none). A disk chunk is laid
/// out as the device's (CONTRACT-BC 22): a role shape's slots lie in a disk
/// chunk of their own, a slot's disk ref `chunk * slots a chunk + index`,
/// as the engine places it. The chunks are the kv store's disk slots, taken
/// whole and given back whole through it, counted against the disk's budget
/// with kv's.
///
/// Lock order: a taker holds its own lock (the roles', or the rs store's)
/// and then the kv store's, which in turn may take the shared pool's; no
/// path takes the roles' or the rs store's lock under the kv store's, and
/// the planner reads the file's room outside the rs store's lock.
pub trait DiskSlots: Send + Sync {
    /// `n` whole free disk chunks, as chunk ids; all or none.
    fn take_chunks(&self, n: usize) -> Option<Vec<u32>>;
    /// Chunks taken before, free again.
    fn give_chunks(&self, chunks: &[u32]);
    /// The whole chunks free, at most (the file's free slots a chunk's
    /// worth each): what a planner reads the disk's room by.
    fn free_chunks(&self) -> usize {
        0
    }
}

/// The disk tier a [`Roles`] parks on: its chunks, and each role shape's
/// disk chunks with the slots free in each.
#[derive(Clone)]
struct Disk {
    slots: Arc<dyn DiskSlots>,
    /// The slots of each role shape a disk chunk holds, as a device chunk:
    /// read once, so asking takes no lock (a caller may hold the device's).
    per_chunk: HashMap<SlotRole, u32>,
    chunks: Arc<std::sync::Mutex<HashMap<SlotRole, BTreeMap<u32, BTreeSet<u32>>>>>,
}

impl std::fmt::Debug for Disk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Disk").finish_non_exhaustive()
    }
}

/// Parked items' live parts moving down from the device
/// ([`Roles::prepare_park_down`]): the move of those the host has room for,
/// and the host chunks the rest would take.
#[derive(Debug, Default)]
pub struct ParkDown {
    pub txn: Option<RoleMove>,
    pub host_chunks: u64,
}

impl RoleMove {
    /// The copies the move takes, a role shape each.
    #[must_use]
    pub fn copies(&self) -> &[RoleCopy] {
        &self.copies
    }

    /// A park's instances, moved down or left where they lie
    /// ([`Item::shed`]): each is said to the engine as parked
    /// ([`Rebind::Parked`]) once the copies have landed and before the
    /// commit gives its device slots to anyone else; a move down of one
    /// parked where it lies says it again, its tables kept nowhere now.
    /// None for an unpark.
    #[must_use]
    pub fn parked(&self) -> Vec<Rebind> {
        if !self.out {
            return Vec::new();
        }
        let parked = |kept: bool| {
            move |(owner, _): &(Owner, Vec<Part>)| match *owner {
                Owner::Instance(id) => Some(Rebind::Parked { id, kept }),
                _ => None,
            }
        };
        self.items
            .iter()
            .filter_map(parked(false))
            .chain(self.shed.iter().filter_map(parked(true)))
            .collect()
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct Roles {
    /// Each served restore's reservation on the device pool, by holder
    /// ([`super::slots::SlotPool::reserve_chunks`]): its unpark and bind
    /// take from it, until it is released.
    claims: HashMap<Holder, u64>,
    /// The reservation the takes under way may take from: its holder's,
    /// set for the length of its unpark or bind.
    claim: Option<u64>,
    device: Shared,
    host: Option<Shared>,
    /// The disk tier, where the load keeps one ([`Self::set_disk`]).
    disk: Option<Disk>,
    shapes: RoleShapes,
    programs: HashMap<ProgramId, ProgramFacts>,
    items: Items,
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
    /// The process whose guest bind's need the pool counts toward the
    /// walk's goal ([`GUEST_WAITER`]): the last bind short of room that
    /// waits on a commit. Its need is no longer counted once it binds, is
    /// refused, or its process is gone ([`Self::leave_guest_wait`]).
    guest_waiter: Option<Holder>,
    /// The last generation handed out ([`Self::next_generation`]).
    generations: u64,
    /// Slots out of service for good ([`Self::abandon`]).
    leaked: u64,
    /// Binds taken and not yet bound or given back, by process
    /// ([`Self::binds_on_way`]): a park of the process waits for them.
    binding: HashMap<Holder, u32>,
    /// Told each time one of those lands ([`Self::bind_landed`]).
    bind_landed: Arc<tokio::sync::Notify>,
    /// The bytes the binds on their way took for themselves
    /// ([`Taken::bytes`]): held, in no item yet. By the process each is
    /// for, and together.
    on_way: (HashMap<Option<Holder>, u64>, u64),
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
    /// A generation not handed out before ([`Item::generation`]).
    fn next_generation(&mut self) -> u64 {
        self.generations += 1;
        self.generations
    }

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
            claims: HashMap::new(),
            claim: None,
            device: Arc::clone(device),
            host: host.cloned(),
            disk: None,
            shapes,
            programs: HashMap::new(),
            items: Items::new(),
            held: Vec::new(),
            chunk_slots,
            stated: std::collections::HashSet::new(),
            freed: Arc::default(),
            binding: HashMap::new(),
            bind_landed: Arc::default(),
            on_way: (HashMap::new(), 0),
            guest_waiter: None,
            generations: 0,
            leaked: 0,
        })
    }

    /// Parks on `disk` too, the disk tier's slot file: the live parts of
    /// every role lie there where no host tier holds them, in disk chunks of
    /// their own shape ([`DiskSlots`]).
    pub fn set_disk(&mut self, disk: Arc<dyn DiskSlots>) {
        let per_chunk = {
            let device = self.device.lock().unwrap();
            [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter]
                .into_iter()
                .filter_map(|role| {
                    let shape = self.shapes.of(role)?;
                    let per_chunk = device.pool().shape(shape.pool)?.slots_per_chunk;
                    (per_chunk > 0).then_some((role, per_chunk))
                })
                .collect()
        };
        self.disk = Some(Disk {
            slots: disk,
            per_chunk,
            chunks: Arc::default(),
        });
    }
}

#[cfg(test)]
mod test_support;
#[cfg(test)]
pub use test_support::{bind_instance, register_channel};

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod guest_tests;
