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
    /// `attached`: held only by this, the item went with its last member's
    /// park (or was given back as unheld) while the bind was on its way,
    /// and the bind named slots another owner took next (the L40S at
    /// 3 GiB on df488038b: a rescore's instance bound over a live pair's
    /// run, `guest::claims` refused).
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
    /// ([`Roles::unpark_need`]): let through, the restore committed and
    /// waited on a rebind no room was made for (Mac Pro at 2 GiB on
    /// 1e87ab171: InstanceId(5), its process Restoring until teardown).
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

    /// The slots of `role` a disk chunk holds, laid out as a device chunk,
    /// where the load keeps a disk tier.
    fn disk_per_chunk(&self, role: SlotRole) -> Option<u32> {
        self.disk.as_ref()?.per_chunk.get(&role).copied()
    }

    /// Whether the disk tier holds slots of `role`.
    fn disk_holds(&self, role: SlotRole) -> bool {
        self.disk_per_chunk(role).is_some()
    }

    /// Whether some tier below the device holds every live part of `parts`.
    fn lies_below(&self, parts: &[Part]) -> bool {
        self.host.is_some()
            || parts
                .iter()
                .filter(|part| !part.dead())
                .all(|part| self.disk_holds(part.role))
    }

    /// Disk slots for the live parts of `parts`, all or none: a dead part
    /// holds nothing off the device. Each role's slots come from its own
    /// disk chunks, a chunk opened whole where those are full.
    fn take_disk(&self, parts: &[Part]) -> Option<Vec<Part>> {
        let disk = self.disk.as_ref()?;
        if !parts
            .iter()
            .filter(|part| !part.dead())
            .all(|part| self.disk_holds(part.role))
        {
            return None;
        }
        let mut taken: Vec<Part> = Vec::with_capacity(parts.len());
        for part in parts {
            let refs = if part.dead() || part.refs.is_empty() {
                Vec::new()
            } else {
                let per_chunk = self.disk_per_chunk(part.role)?;
                match Self::take_disk_slots(disk, part.role, per_chunk, part.refs.len()) {
                    Some(refs) => refs,
                    None => {
                        self.give_on(Tier::Disk, &taken);
                        return None;
                    }
                }
            };
            taken.push(Part {
                refs,
                ..part.clone()
            });
        }
        Some(taken)
    }

    /// `n` free slots of `role` in its disk chunks, `per_chunk` a chunk,
    /// opening whole chunks for what they lack; all or none.
    fn take_disk_slots(disk: &Disk, role: SlotRole, per_chunk: u32, n: usize) -> Option<Vec<u32>> {
        let mut shapes = disk.chunks.lock().unwrap();
        let chunks = shapes.entry(role).or_default();
        let free: usize = chunks.values().map(BTreeSet::len).sum();
        if free < n {
            let opened = (n - free).div_ceil(per_chunk as usize);
            for chunk in disk.slots.take_chunks(opened)? {
                chunks.insert(chunk, (0..per_chunk).collect());
            }
        }
        let mut refs = Vec::with_capacity(n);
        for (&chunk, slots) in chunks.iter_mut() {
            while refs.len() < n {
                let Some(index) = slots.pop_first() else {
                    break;
                };
                refs.push(chunk * per_chunk + index);
            }
            if refs.len() == n {
                break;
            }
        }
        Some(refs)
    }

    /// `refs` of `role` free again in its disk chunks; a chunk wholly free
    /// goes back to the slot file.
    fn give_disk_slots(disk: &Disk, role: SlotRole, per_chunk: u32, refs: &[u32]) {
        let mut shapes = disk.chunks.lock().unwrap();
        let chunks = shapes.entry(role).or_default();
        let mut emptied = Vec::new();
        for &slot in refs {
            let (chunk, index) = (slot / per_chunk, slot % per_chunk);
            let slots = chunks.entry(chunk).or_default();
            slots.insert(index);
            if slots.len() == per_chunk as usize {
                emptied.push(chunk);
            }
        }
        for chunk in &emptied {
            chunks.remove(chunk);
        }
        emptied.sort_unstable();
        emptied.dedup();
        if !emptied.is_empty() {
            disk.slots.give_chunks(&emptied);
        }
    }

    /// Gives `parts` back on tier `on` (their refs that tier's).
    fn give_on(&self, on: Tier, parts: &[Part]) {
        match on {
            Tier::Disk => {
                let Some(disk) = &self.disk else {
                    return;
                };
                for part in parts.iter().filter(|part| !part.refs.is_empty()) {
                    if let Some(per_chunk) = self.disk_per_chunk(part.role) {
                        Self::give_disk_slots(disk, part.role, per_chunk, &part.refs);
                    }
                }
            }
            Tier::Host => {
                if let Some(host) = self.host.clone() {
                    self.give(&host, parts);
                }
            }
            Tier::Device | Tier::Moving => {
                let device = Arc::clone(&self.device);
                self.give_freeing(&device, parts);
            }
        }
    }

    /// Every live part's slot on the device, by item: what the pool must
    /// count held while the item names it (for tests: a holder naming a
    /// slot the pool counts free reads and writes another's state).
    #[cfg(test)]
    pub(crate) fn device_slots_named(&self) -> Vec<(String, super::slots::Slot)> {
        let device = self.device.lock().unwrap();
        let mut out = Vec::new();
        for (owner, item) in &self.items {
            if item.tier != Tier::Device {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.dead()) {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    out.push((
                        format!("{owner:?} {:?} shed={}", part.role, item.shed),
                        device.slot(shape.pool, slot_ref),
                    ));
                }
            }
        }
        out
    }

    /// Every item `holder` moves with, for a quiet wait's census: its owner,
    /// tier, whether shed, and each part's role, live or dead, and the
    /// physical chunks its refs lie in where it is on the device.
    #[must_use]
    pub fn guest_items_census(&self, holder: Holder) -> String {
        let device = self.device.lock().unwrap();
        let mut out = Vec::new();
        for (owner, item) in &self.items {
            if !self.moves_with(*owner, item, holder) {
                continue;
            }
            let parts: Vec<String> = item
                .parts
                .iter()
                .map(|part| {
                    let chunks: BTreeSet<u32> = match (item.tier == Tier::Device, self.shapes.of(part.role)) {
                        (true, Some(shape)) => part
                            .refs
                            .iter()
                            .filter_map(|&slot_ref| device.pool().physical(device.slot(shape.pool, slot_ref).chunk))
                            .collect(),
                        _ => BTreeSet::new(),
                    };
                    format!(
                        "{:?} {} refs={} chunks={chunks:?}",
                        part.role,
                        if part.dead() { "dead" } else { "live" },
                        part.refs.len()
                    )
                })
                .collect();
            out.push(format!("{owner:?} tier={:?} shed={} [{}]", item.tier, item.shed, parts.join("; ")));
        }
        out.join(" | ")
    }

    /// Per logical chunk, the slots each of `holders` holds there in guest
    /// items bound on the device: dead parts (scratch and pair state, given
    /// back at a park whatever lies below) and live ones (tables, rings),
    /// which a park takes below or, with no room there, leaves parked in
    /// place for a move down a tier to clear. Every one of them goes once
    /// its holder moves, so a chunk they fill is emptiable, as the pool's
    /// census counts it: counting the dead parts alone, a chunk holding a
    /// table never qualified for a run, and a long run (nine chunks, two
    /// of them) found no window, though moving its holders stood it.
    /// Parked items lying on the device are not here: they are the parked
    /// occupants (`parked_chunk_slots`).
    #[must_use]
    pub fn guest_chunk_slots(&self, holders: &[Holder]) -> Vec<(u32, Holder, u32)> {
        let device = self.device.lock().unwrap();
        let mut counts: BTreeMap<(u32, Holder), u32> = BTreeMap::new();
        for (owner, item) in &self.items {
            if item.tier != Tier::Device || item.shed {
                continue;
            }
            let Some(holder) = holders
                .iter()
                .copied()
                .find(|&holder| self.moves_with(*owner, item, holder))
            else {
                continue;
            };
            for part in &item.parts {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    let slot = device.slot(shape.pool, slot_ref);
                    *counts.entry((slot.chunk, holder)).or_default() += 1;
                }
            }
        }
        counts
            .into_iter()
            .map(|((chunk, holder), slots)| (chunk, holder, slots))
            .collect()
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

    /// Whether `holder` holds any part of a guest's state on the device,
    /// live, held or dead: what a park of it moves down a tier or gives
    /// back. A process with no kv or state of its own (a prompt not
    /// seated) holding only such parts is a victim whose eviction parks
    /// them ([`crate::planner`]'s eviction): read by dead parts alone, one
    /// holding resident bytes (a held part) "found nothing to move", its
    /// chunks never emptied (seed 3392).
    #[must_use]
    pub fn holds_device_parts(&self, holder: Holder) -> bool {
        self.items.values().any(|item| {
            item.holder == Some(holder)
                && item.tier == Tier::Device
                && item.parts.iter().any(|part| !part.refs.is_empty())
        })
    }

    /// Whether `pair`'s state is on its way to or from the host with the
    /// instances that hold it, and those instances with their holders: what
    /// a restore short of it ([`RolesError::PairParked`]) waits to see end.
    #[must_use]
    pub fn pair_moving(&self, pair: FireKey) -> Option<Vec<(InstanceId, Option<Holder>)>> {
        let item = self.items.get(&Owner::Pair(pair))?;
        (item.tier == Tier::Moving).then(|| {
            item.attached
                .iter()
                .map(|id| {
                    let holder = self
                        .items
                        .get(&Owner::Instance(*id))
                        .and_then(|instance| instance.holder);
                    (*id, holder)
                })
                .collect()
        })
    }

    /// The pair of one of `instances` on its way to or from the device
    /// (another holder's park or restore moving it): what an unpark that
    /// would leave them unbound waits to see end ([`Self::pair_moving`]).
    #[must_use]
    pub fn moving_pair_of(&self, instances: &[InstanceId]) -> Option<FireKey> {
        instances.iter().find_map(|id| {
            let key = self.items.get(&Owner::Instance(*id))?.pair?;
            (self.items.get(&Owner::Pair(key))?.tier == Tier::Moving).then_some(key)
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
            held: false,
        };
        if slots == 0 {
            return Ok(part(Vec::new()));
        }
        let shape = self.shape(role)?;
        let mut pool = on.lock().unwrap();
        let claim = if Arc::ptr_eq(on, &self.device) { self.claim } else { None };
        if !contiguous {
            return pool
                .claiming(claim, |pool| pool.take_refs(shape.pool, u64::from(slots)))
                .map(part)
                .ok_or(RolesError::Short { role, slots });
        }
        match pool.claiming(claim, |pool| pool.take_run(shape.pool, u64::from(slots))) {
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
            held: false,
        })
    }

    /// Room on the device for `bytes` of guest scratch (`what`) in pieces
    /// of `run` chunks ([`Self::scratch_part`]); all or none. Where no
    /// free run holds it and the segments still to commit within the
    /// budget would, the pool counts it short ([`GUEST_WAITER`]) and the
    /// bind waits on a commit ([`RolesError::GuestGrow`]); else it is
    /// refused by name, with what it asked and what was free.
    /// Counted short in the pool where `heads`, driving the walk.
    fn take_scratch(
        &self,
        bytes: u64,
        run: u32,
        what: impl Fn() -> String,
        heads: bool,
    ) -> Result<Part, RolesError> {
        let part = self.scratch_part(bytes, run)?;
        let shape = self.shape(SlotRole::Instance)?;
        let room = match self.retake(&part) {
            Ok(part) => {
                if heads {
                    let mut device = self.device.lock().unwrap();
                    device.pool_mut().unwait(GUEST_WAITER);
                    device.set_guest_head(None);
                }
                return Ok(part);
            }
            Err(room) => room,
        };
        let (chunks, pieces) = part.chunk_runs();
        let mut device = self.device.lock().unwrap();
        let free = match room {
            GuestRoom::Runs { free_runs, .. } => free_runs,
            GuestRoom::Slots { .. } => 0,
        };
        // Runs a commit could make, only where one can come: at the
        // governor's target the walk commits nothing, and a bind short of
        // room goes to the planner (BindShort) instead of waiting on it.
        let committable = if device.commit_can_come() {
            device.pool().committable_runs(chunks)
        } else {
            0
        };
        let pool = device.pool_mut();
        let room = room.with_committable(committable);
        if free.saturating_add(committable) < pieces {
            if heads {
                pool.unwait(GUEST_WAITER);
                device.set_guest_head(None);
            }
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
        let prefix = pool.prefix_segments();
        if heads {
            pool.wait(GUEST_WAITER, shape.pool, slots);
            // The walk commits until the head's runs stand free, not only
            // its slot count ([`SharedPool::walk_goal`]).
            device.set_guest_head(Some((chunks, pieces)));
        }
        Err(RolesError::GuestGrow {
            what: what(),
            bytes,
            room,
            run: chunks,
            prefix,
        })
    }

    /// Fresh device room for dead part `like`: its run of slots, or its
    /// runs of whole chunks, each inside one segment; all or none, else
    /// what it asked and what was free.
    fn retake(&self, like: &Part) -> Result<Part, GuestRoom> {
        let shape = self.shape(SlotRole::Instance).map_err(|_| like.room())?;
        let mut pool = self.device.lock().unwrap();
        let taken = pool.claiming(self.claim, |pool| {
            if like.pieces > 0 {
                pool.take_chunk_runs(shape.pool, like.chunks, like.pieces)
            } else {
                pool.take_run(shape.pool, u64::from(like.run))
            }
        });
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
                singles: 0,
            }),
            // Any other refusal (the budget, a run across a segment's end)
            // is said by name: read as the part's ask alone, its room's
            // zeros looked like a full device (Mac Pro at 2 GiB on
            // c10d7de5d: "free_runs: 0, free_chunks: 0, committable: 0" on a
            // device 310 of 365 pages free).
            Err(error) => {
                tracing::warn!(%error, run = like.chunks, pieces = like.pieces, "roles: a guest part's fresh room refused by the pool");
                Err(like.room())
            }
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

    /// Whether `pieces` free runs of `run` adjacent `Instance` chunks stand
    /// beside where `holder`'s unpark puts its parked dead parts back
    /// (guest scratch and pair state, given back at its park), each placed
    /// as the unpark's take would place it, in the unpark's order: a read
    /// of the device pool under its lock, nothing taken
    /// ([`super::slots::SlotPool::chunk_runs_beside`]), so the room its own
    /// park gave back never counts toward its bind.
    #[must_use]
    pub fn fits_beside_unpark(&self, holder: Holder, run: u32, pieces: u32) -> bool {
        use super::slots::OwnPiece;
        let Some(shape) = self.shapes.of(SlotRole::Instance) else {
            return true;
        };
        let mut own: Vec<OwnPiece> = Vec::new();
        for (owner, item) in &self.items {
            let parked = matches!(item.tier, Tier::Host | Tier::Disk) || item.takes_fresh();
            if !parked || !self.unparks_with(*owner, item, holder) {
                continue;
            }
            let below = matches!(item.tier, Tier::Host | Tier::Disk);
            // Its guest parts laid fresh: the dead ones given back, and a
            // held one coming up from a tier below.
            for part in item
                .parts
                .iter()
                .filter(|part| (part.dead() && part.refs.is_empty()) || (part.held && below))
            {
                own.push(if part.pieces > 0 {
                    OwnPiece::Chunks {
                        run: part.chunks,
                        pieces: part.pieces,
                    }
                } else {
                    OwnPiece::Slots {
                        shape: shape.pool,
                        slots: part.run,
                    }
                });
            }
        }
        // The unpark lays its whole-chunk runs first, the longest first,
        // then its runs of slots ([`Self::prepare_unpark`]).
        own.sort_by_key(|piece| match *piece {
            OwnPiece::Chunks { run, .. } => (0, std::cmp::Reverse(run)),
            OwnPiece::Slots { .. } | OwnPiece::Take { .. } => (1, std::cmp::Reverse(0)),
        });
        self.device
            .lock()
            .unwrap()
            .pool()
            .chunk_runs_beside(&own, run, pieces)
    }

    /// Every piece `holder`'s unpark takes on the device, in the order it
    /// takes them, as the pool's placement reads them
    /// ([`super::slots::OwnPiece`]): each dead part with no slots (given
    /// back at its park, or short of slots its rebind names) as its runs of
    /// whole chunks or its run of slots, and each live part of an item
    /// parked below the device as the take that brings it back (a run of
    /// slots where it is contiguous, single slots else); the whole-chunk
    /// runs first, the longest first, then the rest
    /// ([`Self::prepare_unpark`]).
    fn unpark_pieces(&self, holder: Holder) -> Vec<super::slots::OwnPiece> {
        use super::slots::OwnPiece;
        let (groups, order) = self.unpark_order(holder);
        order
            .into_iter()
            .filter_map(|(at, index)| {
                let part = &groups[at].2[index];
                let shape = self.shapes.of(part.role)?;
                // As [`Self::take_parts`] takes it on the device: a dead
                // part retaken fresh ([`Self::retake`]), a live one by
                // [`Self::take`].
                Some(if part.shaped() {
                    if part.pieces > 0 {
                        OwnPiece::Chunks {
                            run: part.chunks.max(1),
                            pieces: part.pieces,
                        }
                    } else {
                        OwnPiece::Slots {
                            shape: shape.pool,
                            slots: part.run,
                        }
                    }
                } else {
                    let slots = u32::try_from(part.refs.len()).unwrap_or(u32::MAX);
                    if part.contiguous {
                        OwnPiece::Slots {
                            shape: shape.pool,
                            slots,
                        }
                    } else {
                        OwnPiece::Take {
                            shape: shape.pool,
                            slots: u64::from(slots),
                        }
                    }
                })
            })
            .collect()
    }

    /// The items `holder`'s unpark brings back, each with whether it shed
    /// its dead parts at its park (it takes those fresh, its live parts
    /// where they lie), and the order its parts are taken in: its runs of
    /// whole chunks first, the longest first, then its runs of slots and
    /// its live parts, in the items' order. Taken in the items' order, a
    /// single chunk opened for a table at the lowest free chunk split the
    /// only stretch a run of two stood in, and the unpark was refused the
    /// run its need counted on (the L40S). What the unpark takes
    /// ([`Self::prepare_unpark`]) and what its placement reads
    /// ([`Self::unpark_pieces`]): one order.
    #[allow(clippy::type_complexity)]
    fn unpark_order(&self, holder: Holder) -> (Vec<(Owner, bool, Vec<Part>)>, Vec<(usize, usize)>) {
        // Items that shed their dead parts at the park take fresh ones in
        // their class, their live parts where they lie: nothing is copied.
        // So does one on the device with a dead part its rebind must name
        // and no slots for it ([`Item::takes_fresh`]): the restore commits
        // only with every part its rebinds name in place.
        let groups: Vec<(Owner, bool, Vec<Part>)> = self
            .items
            .iter()
            .filter(|(owner, item)| item.takes_fresh() && self.unparks_with(**owner, item, holder))
            .map(|(owner, item)| (*owner, true, item.parts.clone()))
            .chain(
                self.items
                    .iter()
                    .filter(|(owner, item)| {
                        matches!(item.tier, Tier::Host | Tier::Disk)
                            && self.unparks_with(**owner, item, holder)
                    })
                    .map(|(owner, item)| (*owner, false, item.parts.clone())),
            )
            .collect();
        let mut order: Vec<(usize, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(at, (_, shed, parts))| {
                parts
                    .iter()
                    .enumerate()
                    .filter(move |(_, part)| !*shed || (part.dead() && part.refs.is_empty()))
                    .map(move |(index, _)| (at, index))
            })
            .collect();
        order.sort_by_key(|&(at, index)| {
            let part = &groups[at].2[index];
            if part.shaped() && part.pieces > 0 {
                (0, std::cmp::Reverse(part.chunks))
            } else {
                (1, std::cmp::Reverse(0))
            }
        });
        (groups, order)
    }

    /// The runs of adjacent chunks `holder`'s unpark lays fresh, one entry
    /// per run length, the longest first, read from the pool's own
    /// placement of every piece it takes ([`Self::unpark_pieces`]): its
    /// runs of whole chunks, and the chunks its runs and takes of slots
    /// open beyond the partly filled ones, in single chunks. A run of 465
    /// slots asks a chunk wherever no partly filled chunk of its class holds
    /// it, whatever free slots other chunks hold: counted by free slots, it
    /// asked none, the need stood, and the unpark was refused for good (the
    /// L40S on 5aa45805b: need [(1, 1)], refused a run of 465 slots).
    /// Where the pool cannot place them all now, each run of slots and
    /// take asks a chunk of its own.
    #[must_use]
    pub fn unpark_need(&self, holder: Holder) -> Vec<(u32, u32)> {
        use super::slots::OwnPiece;
        let own = self.unpark_pieces(holder);
        let mut runs: BTreeMap<u32, u64> = BTreeMap::new();
        let mut whole = 0u32;
        for piece in &own {
            match *piece {
                OwnPiece::Chunks { run, pieces } => {
                    *runs.entry(run).or_default() += u64::from(pieces);
                    whole = whole.saturating_add(run.saturating_mul(pieces));
                }
                OwnPiece::Slots { .. } | OwnPiece::Take { .. } => {}
            }
        }
        // Where its placement does not stand now, its placement alone: a
        // part each a whole chunk was no placement (`SlotPool::opens_alone`).
        let device = self.device.lock().unwrap();
        let pool = device.pool();
        let singles = pool
            .opens_beside(&own)
            .unwrap_or_else(|| pool.opens_alone(&own))
            .saturating_sub(whole);
        drop(device);
        if singles > 0 {
            *runs.entry(1).or_default() += u64::from(singles);
        }
        runs.into_iter()
            .rev()
            .map(|(run, pieces)| (run, u32::try_from(pieces).unwrap_or(u32::MAX)))
            .collect()
    }

    /// `holder`'s whole restore placed by the takes themselves
    /// ([`super::slots::SlotPool::place`]): `takes` (its kv pages still to
    /// gather and the single slots its serve takes, by shape) first, then
    /// every piece its unpark takes, in the unpark's order. The pieces it
    /// asked, beside where they went: the one placement a restore is sent
    /// on, its moves planned by, and its pending bind read against.
    #[must_use]
    pub fn restore_placement(
        &self,
        holder: Holder,
        takes: &[(super::slots::ShapeId, u64)],
    ) -> (Vec<super::slots::OwnPiece>, super::slots::Placement) {
        use super::slots::OwnPiece;
        let own: Vec<OwnPiece> = takes
            .iter()
            .map(|&(shape, slots)| OwnPiece::Take { shape, slots })
            .chain(self.unpark_pieces(holder))
            .collect();
        let placement = self.device.lock().unwrap().pool().place(&own);
        (own, placement)
    }

    /// Holds for `holder`'s served restore the free chunks its unpark's
    /// pieces and then `bind` (a pending bind's runs) open as the takes
    /// place them ([`super::slots::SlotPool::place`]): no other take opens
    /// them until its unpark and bind have taken them or it is released.
    /// Whether its whole placement stands; held only where it does.
    pub fn reserve_restore(&mut self, holder: Holder, bind: &[(u32, u32)]) -> bool {
        use super::slots::OwnPiece;
        self.release_claim(holder);
        let own: Vec<OwnPiece> = self
            .unpark_pieces(holder)
            .into_iter()
            .chain(bind.iter().map(|&(run, pieces)| OwnPiece::Chunks { run, pieces }))
            .collect();
        let mut device = self.device.lock().unwrap();
        let placement = device.pool().place(&own);
        if !placement.stands() {
            return false;
        }
        let ticket = device.pool_mut().reserve_chunks(placement.opened.iter().copied());
        drop(device);
        self.claims.insert(holder, ticket);
        true
    }

    /// `own` placed for `holder` by the takes themselves
    /// ([`super::slots::SlotPool::place`]), what is held for its served
    /// restore open to them: the one placement its pending bind is checked
    /// by once its unpark has taken its own. Read as a count of free runs
    /// without its own reservation, a restore whose bind runs were held for
    /// it found none of them, every serve (Mac Pro g2 on bd4f4ffe7:
    /// PendingBindShort, 0 free runs of 7 where 5 stood).
    #[must_use]
    pub fn place_claimed(&self, holder: Holder, own: &[super::slots::OwnPiece]) -> super::slots::Placement {
        let ticket = self.claims.get(&holder).copied();
        self.device
            .lock()
            .unwrap()
            .pool_mut()
            .with_claim(ticket, |pool| pool.place(own))
    }

    /// `holder`'s reservation, its chunks not yet taken, goes back.
    pub fn release_claim(&mut self, holder: Holder) {
        if let Some(ticket) = self.claims.remove(&holder) {
            self.device.lock().unwrap().pool_mut().release_reservation(ticket);
        }
    }

    /// The chunks held for `holder`'s served restore.
    #[cfg(test)]
    #[must_use]
    pub fn claimed(&self, holder: Holder) -> Vec<u32> {
        self.claims
            .get(&holder)
            .map(|&ticket| self.device.lock().unwrap().pool().reserved(ticket))
            .unwrap_or_default()
    }

    /// What `holder`'s restore would take back, in one line for a stall
    /// census: each parked item (instance, channel, adapter, pair) with its
    /// tier, and each part by role and size (scratch runs of whole chunks,
    /// or a run of slots; dead between fires or live, with its refs), then
    /// whether `need` runs stand beside the dead parts as the pool stands
    /// ([`Self::fits_beside_unpark`]: the live parts are not placed there),
    /// the free runs of `need`'s length, and each role shape's free slots.
    #[must_use]
    pub fn unpark_census(&self, holder: Holder, need: Option<(u32, u32)>) -> String {
        let mut items = Vec::new();
        for (owner, item) in &self.items {
            let parked = matches!(item.tier, Tier::Host | Tier::Disk) || item.takes_fresh();
            if !parked || !self.unparks_with(*owner, item, holder) {
                continue;
            }
            let parts: Vec<String> = item
                .parts
                .iter()
                .map(|part| {
                    format!(
                        "{:?}:{}{}x{}c/{}s/refs{}",
                        part.role,
                        if part.dead() { "dead:" } else { "live:" },
                        part.pieces,
                        part.chunks,
                        part.run,
                        part.refs.len()
                    )
                })
                .collect();
            items.push(format!("{owner:?}@{:?}{}[{}]", item.tier, if item.shed { "(shed)" } else { "" }, parts.join(",")));
        }
        let beside = need.map(|(run, pieces)| self.fits_beside_unpark(holder, run, pieces));
        let free_runs = need.map(|(run, _)| self.free_runs(run));
        let shapes: Vec<String> = [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter]
            .into_iter()
            .filter_map(|role| {
                let shape = self.shapes.of(role)?;
                let pool = self.device.lock().unwrap();
                Some(format!("{role:?}:free_slots={}", pool.pool().available(shape.pool)))
            })
            .collect();
        format!(
            "parked=[{}] need={need:?} beside_dead_parts={beside:?} free_runs={free_runs:?} shapes=[{}]",
            items.join("; "),
            shapes.join(",")
        )
    }

    /// The free runs of `run` adjacent `Instance` chunks inside one
    /// segment the device pool holds now: how many more pieces of a guest
    /// state that lies in runs of that length it could place.
    #[must_use]
    pub fn free_runs(&self, run: u32) -> u32 {
        self.device.lock().unwrap().pool().free_chunk_runs(run)
    }

    /// Whether `pieces` free runs of `run` chunks stand, and `singles` free
    /// chunks beside them.
    #[must_use]
    pub fn room_free(&self, run: u32, pieces: u32, singles: u32) -> bool {
        let pool = self.device.lock().unwrap();
        let pool = pool.pool();
        pool.free_chunk_runs(run) >= pieces
            && pool.free_chunk_runs(1) >= run.max(1).saturating_mul(pieces).saturating_add(singles)
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

    /// Tells the binds waiting on room ([`Self::room_watch`]) to look again.
    fn tell_room(&self) {
        self.freed
            .gives
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.freed.notify.notify_waiters();
    }

    /// A frame naming `instances` (its attachments) was handed to the
    /// engine: each instance's item, and the pair, channels and adapter it
    /// shares, are read until it lands. Answers what it named, for
    /// [`Self::landed`]. A pair's batch tables are read by every grouped
    /// fire of its instances: given back at the pair's last close while
    /// one was in flight, their chunks went to another owner under the
    /// fire (Metal's 0xa0, FUSED_GEOMETRY_MISMATCH).
    pub fn reading(&mut self, instances: impl IntoIterator<Item = InstanceId>) -> Vec<Owner> {
        let mut named: Vec<Owner> = Vec::new();
        for id in instances {
            let Some(item) = self.items.get(&Owner::Instance(id)) else {
                continue;
            };
            named.push(Owner::Instance(id));
            named.extend(item.channels.iter().map(|&channel| Owner::Channel(channel)));
            named.extend(item.adapter.map(Owner::Adapter));
            named.extend(item.pair.map(Owner::Pair));
        }
        named.retain(|owner| self.items.contains_key(owner));
        for owner in &named {
            if let Some(item) = self.items.get_mut(owner) {
                item.reads += 1;
            }
        }
        named
    }

    /// The frame that named `named` ([`Self::reading`]) landed: what was
    /// let go under it goes back now.
    pub fn landed(&mut self, named: &[Owner]) {
        for &owner in named {
            if let Some(item) = self.items.get_mut(&owner) {
                item.reads = item.reads.saturating_sub(1);
            }
            self.free_if_done(owner);
        }
        self.release_unheld_pairs();
    }

    /// `holder`'s guest bind no longer waits: it bound, was refused, or
    /// its process is gone. Where the pool counts its need, it counts it no
    /// more, so the walk commits for no one gone.
    pub fn leave_guest_wait(&mut self, holder: Holder) {
        if self.guest_waiter != Some(holder) {
            return;
        }
        self.guest_waiter = None;
        let mut device = self.device.lock().unwrap();
        device.pool_mut().unwait(GUEST_WAITER);
        device.set_guest_head(None);
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
            self.tell_room();
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
            let took = if device && part.shaped() {
                // Laid as runs on the device, dead or held: its fresh runs,
                // a held part's bytes copied into them.
                self.retake(part).map_err(|room| {
                    let (run, _) = part.chunk_runs();
                    let committable = self.device.lock().unwrap().pool().committable_runs(run);
                    RolesError::GuestShort {
                        what: "a parked guest state coming back".into(),
                        bytes: self.dead_bytes(part),
                        room: room.with_committable(committable),
                    }
                })
            } else if part.dead() {
                Ok(Part {
                    refs: Vec::new(),
                    ..part.clone()
                })
            } else {
                // Below the device, its slots anywhere; what it is (held,
                // its runs' shape for the way back) goes with it.
                self.take(on, part.role, part.refs.len() as u32, part.contiguous)
                    .map(|taken| Part {
                        refs: taken.refs,
                        ..part.clone()
                    })
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
    /// else fresh ones; with them the guest state `guest`, taken before
    /// anything was registered for the bind ([`Self::take_guest`]), which
    /// goes back with the rest where the take fails.
    /// The fresh chunks the parts a bind of `program` takes beside its
    /// guest state may open: its tables, each ring of its own (a channel
    /// `registered` names holds its ring) and its adapter where none is on
    /// the device, as [`Self::take_instance`] takes them, each role's parts
    /// in as few chunks as their slots fill.
    fn instance_parts_fresh(
        &self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
        adapter: Option<u128>,
    ) -> u32 {
        let Some(facts) = self.programs.get(&program) else {
            return 0;
        };
        let rings: u32 = facts
            .rings
            .iter()
            .filter(|ring| {
                !channels
                    .get(ring.channel as usize)
                    .is_some_and(|&id| registered(id))
            })
            .map(|ring| ring.footprint.slots)
            .sum();
        let adapter = adapter
            .filter(|&key| {
                !matches!(self.items.get(&Owner::Adapter(key)), Some(item) if item.tier == Tier::Device)
            })
            .map_or(0, |_| facts.adapter.slots);
        let Ok(instance) = self.shape(SlotRole::Instance) else {
            return 0;
        };
        let chunk_bytes = u64::from(self.chunk_slots) * instance.slot_bytes;
        [
            (SlotRole::Instance, facts.instance.slots),
            (SlotRole::Channel, rings),
            (SlotRole::Adapter, adapter),
        ]
        .into_iter()
        .filter(|&(_, slots)| slots > 0)
        .map(|(role, slots)| {
            let bytes = self
                .shape(role)
                .map_or(chunk_bytes, |shape| u64::from(slots) * shape.slot_bytes);
            u32::try_from(bytes.div_ceil(chunk_bytes.max(1)).max(1)).unwrap_or(u32::MAX)
        })
        .sum()
    }

    pub fn take_instance(
        &mut self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
        adapter: Option<u128>,
        guest: GuestTaken,
    ) -> Result<Taken, (RolesError, u32)> {
        let Some(facts) = self.programs.get(&program).cloned() else {
            self.untake_guest(guest);
            return Err((RolesError::Program(program), 0));
        };
        let mut shared = Vec::new();
        for &id in channels {
            if let Some(item) = self.items.get(&Owner::Channel(id)) {
                if item.tier != Tier::Device {
                    let holder = item.holder;
                    self.untake_guest(guest);
                    return Err((
                        RolesError::Parked {
                            channel: id,
                            holder,
                        },
                        0,
                    ));
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
        match self.take(
            &device,
            SlotRole::Instance,
            footprint.slots,
            footprint.contiguous,
        ) {
            Ok(part) => taken.parts.push(part),
            Err(error) => {
                self.untake_guest(guest);
                return Err((error, 0));
            }
        }
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
                    let opened = u32::try_from(taken.parts.len()).unwrap_or(u32::MAX);
                    self.give(&device, &taken.parts);
                    self.untake_guest(guest);
                    return Err((error, opened));
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
                    self.untake_guest(guest);
                    return Err((RolesError::AdapterParked, 0));
                }
                None => match self.take(
                    &device,
                    SlotRole::Adapter,
                    footprint.slots,
                    footprint.contiguous,
                ) {
                    Ok(part) => Some((key, part, true)),
                    Err(error) => {
                        let opened = u32::try_from(taken.parts.len()).unwrap_or(u32::MAX);
                        self.give(&device, &taken.parts);
                        self.untake_guest(guest);
                        return Err((error, opened));
                    }
                },
            };
        }
        taken.parts.extend(guest.scratch);
        taken.pair = guest.pair;
        self.pin_shared(&taken, true);
        Ok(taken)
    }

    /// The pair and the adapter `taken` names that are items already: held
    /// by its bind (`pin`), or let go of at its bind or its give back.
    fn pin_shared(&mut self, taken: &Taken, pin: bool) {
        let owners = [
            taken.adapter.as_ref().map(|(key, ..)| Owner::Adapter(*key)),
            taken.pair.as_ref().map(|(key, ..)| Owner::Pair(*key)),
        ];
        for owner in owners.into_iter().flatten() {
            if let Some(item) = self.items.get_mut(&owner) {
                item.binds = if pin {
                    item.binds + 1
                } else {
                    item.binds.saturating_sub(1)
                };
            }
        }
    }

    /// A guest program's state for a bind of an instance on `channels`
    /// ([`GuestState`], of the pair `pair`): the instance's scratch, whole
    /// `Instance` chunks, and the pair's chunks its other instances hold
    /// already, else fresh ones. Taken first, before the bind registers
    /// anything: a bind short of it ([`RolesError::BindShort`],
    /// [`RolesError::GuestGrow`]) waits for room with nothing to undo. A
    /// channel it binds parked with its process refuses it first, as
    /// [`Self::take_instance`] would.
    /// As [`Self::take_guest_now`], for `holder`'s bind. Every bind takes
    /// room as it comes, none waiting behind another: a smaller one never
    /// takes a longer run's chunks ([`crate::store::slots`] size classes).
    /// The first bind short of room that waits on a commit has its need
    /// counted toward the walk's goal until it binds or leaves; one short
    /// meanwhile leaves that need as it is, and waits on the commit and the
    /// planner as it does (the planner asks the walk for its need too).
    pub fn take_guest(
        &mut self,
        channels: &[ChannelId],
        (pair, state): (FireKey, GuestState),
        holder: Option<Holder>,
        beside: u32,
    ) -> Result<GuestTaken, RolesError> {
        self.claim = holder.and_then(|holder| self.claims.get(&holder).copied());
        if let Some(ticket) = self.claim {
            tracing::debug!(
                ?holder,
                chunks = ?self.device.lock().unwrap().pool().reserved(ticket),
                "roles: a bind takes from what its restore held for it"
            );
        }
        let taken = self.take_guest_claimed(channels, (pair, state), holder, beside);
        self.claim = None;
        // Its bind took what was held for it: the rest goes back.
        if taken.is_ok()
            && let Some(holder) = holder
        {
            self.release_claim(holder);
        }
        taken
    }

    fn take_guest_claimed(
        &mut self,
        channels: &[ChannelId],
        (pair, state): (FireKey, GuestState),
        holder: Option<Holder>,
        beside: u32,
    ) -> Result<GuestTaken, RolesError> {
        let heads = holder.is_none_or(|holder| self.guest_waiter.is_none_or(|counted| counted == holder));
        let taken = self.take_guest_now(channels, (pair, state), heads, beside);
        if let Some(holder) = holder.filter(|_| heads) {
            match &taken {
                Err(RolesError::GuestGrow { .. }) => self.guest_waiter = Some(holder),
                _ if self.guest_waiter == Some(holder) => self.guest_waiter = None,
                _ => {}
            }
        }
        taken
    }

    fn take_guest_now(
        &mut self,
        channels: &[ChannelId],
        (pair, state): (FireKey, GuestState),
        heads: bool,
        beside: u32,
    ) -> Result<GuestTaken, RolesError> {
        // What segregating runs by length holds back: each class's partly
        // filled Instance chunks and their free bytes, at debug.
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(shape) = self.shape(SlotRole::Instance)
        {
            let device = self.device.lock().unwrap();
            let classes: Vec<(u32, usize, u64)> = device
                .pool()
                .partial_classes(shape.pool)
                .into_iter()
                .map(|(class, chunks, free)| (class, chunks, free * shape.slot_bytes))
                .collect();
            tracing::debug!(
                ?classes,
                "Instance chunks partly filled, by class: (run length, chunks, free bytes)"
            );
        }
        for &id in channels {
            if let Some(item) = self.items.get(&Owner::Channel(id))
                && item.tier != Tier::Device
            {
                return Err(RolesError::Parked {
                    channel: id,
                    holder: item.holder,
                });
            }
        }
        let mut taken = GuestTaken::default();
        let run = state.chunks_a_piece();
        // The instance's own bytes: its resident part and every scratch arm
        // (none taken per fire yet), placed at the bind.
        let per_instance = state.instance_bytes();
        if (per_instance > 0 || state.per_pair > 0) && self.stated.insert(pair) {
            self.state_room(pair, state);
        }
        // Every part the bind takes fresh (its scratch, and its pair's state
        // where the pair holds none on the device now), in runs of the
        // longest, and a single chunk for each role of its tables, rings
        // and adapter (`beside`): what a refusal for want of room asks
        // ([`RolesError::asking`]), so the room it waits on holds the bind.
        // Those parts found slots in chunks open when it was refused; once
        // its process was suspended and restored for the room asked, those
        // slots had gone, the bind was short again of the chunks they then
        // opened and its process went down a second time for one bind (the
        // simulation's field mix, seed 86: refused opening 2 chunks,
        // restored with 2, refused again opening 4).
        let pair_fresh = state.per_pair > 0
            && !matches!(
                self.items.get(&Owner::Pair(pair)),
                Some(item) if item.tier == Tier::Device && !item.shed
            );
        // Its pair's state is asked too where another holds it on the
        // device now: suspended for this room, the pair's holders may park
        // meanwhile, and the bind, restored with the room asked, lays the
        // pair fresh (the simulation's seed 380: asked a run of three,
        // restored with it, short of the pair's second run). Over, never
        // under: the room a pair joined in place leaves goes back with the
        // take.
        let _ = pair_fresh;
        let fresh: Vec<(u32, u32)> = [
            (per_instance > 0).then_some(per_instance),
            (state.per_pair > 0).then_some(state.per_pair),
        ]
        .into_iter()
        .flatten()
        .filter_map(|bytes| self.scratch_part(bytes, run).ok())
        .map(|part| part.chunk_runs())
        .collect();
        let longest = fresh.iter().map(|&(run, _)| run).max().unwrap_or(1).max(1);
        let whole: u32 = fresh
            .iter()
            .map(|&(run, pieces)| run.div_ceil(longest).saturating_mul(pieces))
            .sum::<u32>();
        if per_instance > 0 {
            match self.take_scratch(
                per_instance,
                run,
                || format!("an instance of {pair}'s scratch"),
                heads,
            ) {
                // Resident bytes beside the scratch: the part is held,
                // copied at a park and back at an unpark, never dead.
                Ok(part) => {
                    taken.scratch = Some(Part {
                        held: state.resident > 0,
                        ..part
                    });
                }
                Err(error) => {
                    self.untake_guest(taken);
                    return Err(error.asking(whole, beside));
                }
            }
        }
        if state.per_pair > 0 {
            taken.pair = match self.items.get(&Owner::Pair(pair)) {
                Some(item) if item.tier == Tier::Device && !item.shed => {
                    Some((pair, item.parts[0].clone(), false, item.generation))
                }
                // Parked with its instances: its contents are dead between
                // fires, so it comes back on its own, fresh room on the
                // device for this bind; its parked instances are rebound to
                // it as they unpark.
                // Parked where it lies, its dead part given back
                // ([`Item::shed`]), it comes back the same way: bound to its
                // emptied part, the bind named no place.
                Some(item)
                    if item.tier == Tier::Host || (item.tier == Tier::Device && item.shed) =>
                {
                    let like = item.parts[0].clone();
                    match self.retake(&like) {
                        Ok(part) => {
                            let generation = self.next_generation();
                            let item = self.items.get_mut(&Owner::Pair(pair)).expect("present");
                            item.parts = vec![part.clone()];
                            item.tier = Tier::Device;
                            item.shed = false;
                            item.generation = generation;
                            taken.pair_retaken = true;
                            Some((pair, part, false, generation))
                        }
                        Err(room) => {
                            let short = RolesError::GuestShort {
                                what: format!("{pair}'s state coming back"),
                                bytes: self.dead_bytes(&like),
                                room,
                            };
                            let fits = self.fits_budget(&[Owner::Pair(pair)]);
                            self.untake_guest(taken);
                            return Err(if fits {
                                RolesError::BindShort(Box::new(short))
                            } else {
                                short
                            }
                            .asking(whole, beside));
                        }
                    }
                }
                Some(_) => {
                    self.untake_guest(taken);
                    return Err(RolesError::PairParked(pair));
                }
                None => {
                    match self.take_scratch(
                        state.per_pair,
                        run,
                        || format!("{pair}'s state"),
                        heads,
                    ) {
                        // Its item from the take on, held by the binds on
                        // their way ([`Self::pin_shared`]) and then by its
                        // instances: a second bind of the pair taken before
                        // this one bound joins it. Taken fresh again, two
                        // binds of one pair each named a taking of its own,
                        // the engine laid the group at one under the other's
                        // member, and the first bound left the other's chunk
                        // held by no item for good (the simulation's field
                        // mix, seed 20: pair 3 at chunk 6 for instance 16, at
                        // chunk 19 for 15 and 17, at one generation; seed
                        // 115: a beam pair's chunk out of service from step 0).
                        Ok(part) => {
                            let generation = self.next_generation();
                            let mut item = Item::new(None, vec![part.clone()]);
                            // Held by its instances alone: freed at the last
                            // close, or with its last bind given back.
                            item.open = false;
                            item.generation = generation;
                            self.items.insert(Owner::Pair(pair), item);
                            Some((pair, part, true, generation))
                        }
                        Err(error) => {
                            self.untake_guest(taken);
                            return Err(error.asking(whole, beside));
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
            resident = state.resident,
            scratch = state.scratch,
            scratch_arms = state.scratch_arms,
            per_pair = state.per_pair,
            run,
            instance_pieces = pieces(state.instance_bytes()),
            pair_pieces = pieces(state.per_pair),
            free_runs = self.free_runs(run),
            "a guest program's room: pieces of `run` adjacent chunks, and the free runs of that length"
        );
    }

    /// A bind of `holder`'s was taken: it is on its way until bound or
    /// given back.
    fn bind_taken_for(&mut self, holder: Option<Holder>) {
        if let Some(holder) = holder {
            *self.binding.entry(holder).or_default() += 1;
        }
    }

    /// A bind of `holder`'s was bound or given back.
    fn bind_ended_for(&mut self, holder: Option<Holder>) {
        let Some(holder) = holder else {
            return;
        };
        if let Some(count) = self.binding.get_mut(&holder) {
            *count -= 1;
            if *count == 0 {
                self.binding.remove(&holder);
            }
        }
        self.bind_landed.notify_waiters();
    }

    /// Whether a bind of `holder`'s is on its way: taken, its slots and
    /// its pair held, and not yet bound by the engine or given back. A
    /// park of `holder` waits for it ([`Self::bind_landed`]): parked
    /// meanwhile, the instance it binds stood on the device outside its
    /// process's park, its pair given fresh slots at a restore the
    /// instance's bind never named, and those slots went to another
    /// program's bind (the simulation, a yield between a bind's take and
    /// its bind: `guest::claims` refused an instance over a live pair's
    /// run, the L40S's claims refusals at 3 GiB).
    #[must_use]
    pub fn binds_on_way(&self, holder: Holder) -> bool {
        self.binding.contains_key(&holder)
    }

    /// Told each time a bind on its way is bound or given back.
    #[must_use]
    pub fn bind_landed(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.bind_landed)
    }

    /// The bind of `taken` failed: its slots go back.
    pub fn untake(&mut self, taken: Taken) {
        self.off_way(&taken);
        self.bind_ended_for(taken.holder);
        self.pin_shared(&taken, false);
        let shared = [
            taken.adapter.as_ref().map(|(key, ..)| Owner::Adapter(*key)),
            taken.pair.as_ref().map(|(key, ..)| Owner::Pair(*key)),
        ];
        self.give(&self.device, &taken.parts);
        if let Some((_, part, true)) = taken.adapter {
            self.give(&self.device, &[part]);
        }
        // A pair taken fresh is an item from its take ([`Self::take_guest`]):
        // it goes below with its last bind, unless another joined it.
        // What the bind held and no one else holds goes now: a pair whose
        // members all parked while the bind was on its way, an adapter or
        // pair with none attached.
        for owner in shared.into_iter().flatten() {
            self.free_if_done(owner);
        }
        self.release_unheld_pairs();
    }

    /// A bind's or a registration's take short of slots (`Short`, `NoRun`):
    /// a free chunk of the budget holds any one item (a run past a chunk is
    /// `PastChunk`), so it waits for room as a guest state does
    /// ([`RolesError::BindShort`]). Anything else stands as it is.
    /// `guest`: the runs its guest state's fresh parts took; `opened`: the
    /// parts of its own (tables, rings, adapter) taken before the one
    /// short, each of which may have opened a chunk.
    fn bind_short(&self, error: RolesError, guest: Option<(u32, u32)>, opened: u32) -> RolesError {
        match error {
            RolesError::Short { .. } | RolesError::NoRun { .. }
                if self.device.lock().unwrap().state_chunks_now() > 0 =>
            {
                // Short of its tables' chunk beside the fresh chunks its guest
                // state took (and gave back): the room asked is all of them,
                // never the tables' one chunk alone (seed 46 of the
                // simulation on f152f9631: a scratch chunk and a pair chunk
                // taken, the tables short of a third, one chunk served, short
                // again, a spin). Runs of more than one chunk ask one more
                // run of their length for the tables: over, never under.
                let short = match guest.or((opened > 0).then_some((1, 0))) {
                    Some((run, pieces)) => RolesError::GuestShort {
                        what: format!(
                            "its tables, rings and adapter beside its guest state's {pieces} fresh \
                             runs ({error})"
                        ),
                        bytes: 0,
                        room: if run <= 1 {
                            GuestRoom::Slots {
                                slots: 0,
                                committable: 0,
                                chunks: pieces + opened + 1,
                            }
                        } else {
                            GuestRoom::Runs {
                                run,
                                pieces,
                                free_runs: 0,
                                free_chunks: 0,
                                committable: 0,
                                singles: opened + 1,
                            }
                        },
                    },
                    None => error,
                };
                RolesError::BindShort(Box::new(short))
            }
            error => error,
        }
    }

    /// A guest state taken for a bind that registered nothing, or failed
    /// before its instance's take: its chunks go back.
    pub fn untake_guest(&mut self, guest: GuestTaken) {
        if let Some(part) = &guest.scratch {
            self.give(&self.device, std::slice::from_ref(part));
        }
        // A pair it took fresh is an item from its take: gone with it
        // unless another bind joined it meanwhile.
        if let Some((key, _, true, _)) = guest.pair {
            self.free_if_done(Owner::Pair(key));
        }
    }

    /// The engine bound `taken` as instance `id`, for `holder`.
    pub fn bound(&mut self, id: InstanceId, holder: Option<Holder>, taken: Taken) {
        self.off_way(&taken);
        self.bind_ended_for(taken.holder);
        self.pin_shared(&taken, false);
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
            // A live member again: no longer parked where it lies, so no
            // move down takes it from under this instance.
            shared.shed = false;
            item.adapter = Some(key);
        }
        if let Some((key, part, _, generation)) = taken.pair {
            let shared = self.items.entry(Owner::Pair(key)).or_insert_with(|| {
                let mut pair = Item::new(holder, vec![part]);
                // Held by its instances alone: freed at the last close.
                pair.open = false;
                pair.generation = generation;
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
    /// instance holds it. Until then the channel is the engine's still
    /// (closing, `Engine::close_channel`): its ring parks and unparks with
    /// its instances' holder and is rebound like an open channel's
    /// ([`Roles::rebinds`]), and goes back only after the engine closed
    /// the last instance on it ([`Roles::close_instance`], which the lane
    /// runs once `Engine::close_instance` returned), so no slot the engine
    /// may still address is handed elsewhere.
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
        if item.open
            || !item.attached.is_empty()
            || item.tier == Tier::Moving
            || item.reads > 0
            || item.binds > 0
        {
            return;
        }
        let item = self.items.remove(&owner).expect("present");
        self.give_on(item.tier, &item.parts);
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
            // A pair's run is its batch's while any member is on the device
            // or on its way to or from it: a member mid-park (its move may
            // yet be taken back, the tier below full) or mid-unpark holds
            // it. Read as parked, the pair went with another holder's park,
            // the member's park was taken back, and the run, given back
            // under a resident member, was another program's fresh room
            // (the L40S at 3 GiB on c8077be0f: instance 371's pair run
            // bound by 372-374 of another program, CUDA 700 on its next
            // fire). The member's park landing gives it back
            // ([`Self::release_unheld_pairs`]).
            // A bind on its way holds it as a member does.
            Owner::Pair(_) if item.binds > 0 => false,
            Owner::Adapter(_) if item.binds > 0 => false,
            Owner::Pair(_) => {
                item.attached
                    .iter()
                    .any(|id| instance(id).is_some_and(|instance| instance.holder == Some(holder)))
                    && item.attached.iter().all(|id| {
                        instance(id).is_none_or(|instance| {
                            instance.holder == Some(holder) || Self::settled_parked(instance)
                        })
                    })
            }
            Owner::Adapter(_) => {
                item.attached
                    .iter()
                    .any(|id| instance(id).is_some_and(|instance| instance.holder == Some(holder)))
                    && item.attached.iter().all(|id| {
                        // Gone (closed) holds it no longer: read as held, a
                        // beam pair's scratch whose other copies had ended
                        // never moved with the last one, and its chunk stayed
                        // on the device for good (the L40S at 3 GiB on
                        // c10d7de5d: chunk 17, a pair of 73 dead refs, its
                        // evicted holder no candidate, the head one chunk short).
                        // Parked already and settled, as a pair's: a member
                        // on its way holds it (its park may yet be taken
                        // back), and a member's landing park parks it where
                        // it lies ([`Self::release_unheld_pairs`]).
                        instance(id).is_none_or(|instance| {
                            instance.holder == Some(holder) || Self::settled_parked(instance)
                        })
                    })
            }
        }
    }

    /// A member parked and settled: off the device, or parked where it
    /// lies ([`Item::shed`]); never one on its way (`Moving`).
    fn settled_parked(instance: &Item) -> bool {
        matches!(instance.tier, Tier::Host | Tier::Disk)
            || (instance.tier == Tier::Device && instance.shed)
    }

    /// Gives back the run of each pair on the device every member of which
    /// is parked and settled, nothing reading it: what a member's park
    /// left behind while another member was on its way
    /// ([`Self::moves_with`]). Its state is dead between fires; it parks
    /// as one with no slots below, and a member's restore takes it fresh.
    fn release_unheld_pairs(&mut self) {
        let settled = |item: &Item| {
            item.tier == Tier::Device
                && !item.shed
                && item.reads == 0
                && item.binds == 0
                && !item.attached.is_empty()
                && item.attached.iter().all(|id| {
                    self.items
                        .get(&Owner::Instance(*id))
                        .is_none_or(Self::settled_parked)
                })
        };
        // An adapter holds live parts: parked where it lies, its room a
        // move down a tier takes ([`Self::prepare_park_down`]), its rebind
        // at its first member's unpark.
        let adapters: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| matches!(owner, Owner::Adapter(_)) && settled(item))
            .map(|(owner, _)| *owner)
            .collect();
        for owner in adapters {
            self.items.get_mut(&owner).expect("present").shed = true;
        }
        let unheld: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                matches!(owner, Owner::Pair(_))
                    && item.tier == Tier::Device
                    && !item.shed
                    && item.reads == 0
                    && item.binds == 0
                    && item.parts.iter().all(Part::dead)
                    && !item.attached.is_empty()
                    && item.attached.iter().all(|id| {
                        self.items
                            .get(&Owner::Instance(*id))
                            .is_none_or(Self::settled_parked)
                    })
            })
            .map(|(owner, _)| *owner)
            .collect();
        for owner in unheld {
            let item = self.items.get_mut(&owner).expect("present");
            let emptied: Vec<Part> = item
                .parts
                .iter()
                .map(|part| Part {
                    refs: Vec::new(),
                    ..part.clone()
                })
                .collect();
            let old = std::mem::replace(&mut item.parts, emptied);
            item.tier = Tier::Host;
            item.stale = false;
            let device = Arc::clone(&self.device);
            self.give_freeing(&device, &old);
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

    /// Parks `holder`'s items: host slots taken for each item that fits
    /// there, to copy into. Dead parts (guest scratch and pair state, dead
    /// between fires) need no slots below and are given back whatever the
    /// tier below holds: an item that is all dead parks with no host slots
    /// at all, and one whose live parts the host lacks room for (or with no
    /// host tier) keeps them where they lie and sheds its dead parts
    /// ([`Item::shed`]). Nothing is dropped. `None` when nothing moves. The
    /// process's fires have landed (the planner quiesced it first): no fire
    /// reads the dead parts given back.
    pub fn prepare_park(&mut self, holder: Holder) -> Option<RoleMove> {
        let owners: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                item.tier == Tier::Device && !item.shed && self.moves_with(**owner, item, holder)
            })
            .map(|(owner, _)| *owner)
            .collect();
        let mut items = Vec::new();
        let mut below = Vec::new();
        let mut shed = Vec::new();
        for owner in owners {
            let parts = self.items[&owner].parts.clone();
            let emptied = |parts: &[Part]| -> Vec<Part> {
                parts
                    .iter()
                    .map(|part| {
                        if part.dead() {
                            Part {
                                refs: Vec::new(),
                                ..part.clone()
                            }
                        } else {
                            part.clone()
                        }
                    })
                    .collect()
            };
            // Its live parts to the host where it takes them, else to the
            // disk tier's slot file (no host tier on unified memory, or a
            // host with no room for them): left on the device there, a
            // process's tables and rings held their chunks until a move
            // down was asked for them, and none was (the M1: 379 opened
            // chunks of the Instance shape after 107 evictions).
            let moved = if parts.iter().all(Part::dead) {
                Some((emptied(&parts), Tier::Host))
            } else {
                let host = self.host.clone().and_then(|host| {
                    self.take_parts(&host, &parts, false)
                        .map_err(|error| {
                            tracing::debug!(?owner, %error, "an item's live parts find no host room");
                        })
                        .ok()
                });
                host.map(|to| (to, Tier::Host))
                    .or_else(|| self.take_disk(&parts).map(|to| (to, Tier::Disk)))
            };
            match moved {
                Some((to, tier)) => {
                    self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                    items.push((owner, to));
                    below.push(tier);
                }
                // Its live parts stay where they lie, parked all the same:
                // room a move down a tier makes once the device wants it
                // ([`Self::prepare_park_down`]).
                None => {
                    self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                    shed.push((owner, emptied(&parts)));
                }
            }
        }
        (!items.is_empty() || !shed.is_empty()).then(|| {
            let mut txn = self.moving(true, items, below);
            txn.shed = shed;
            txn
        })
    }

    /// Unparks `holder`'s parked items: device slots taken for each, all or
    /// none. `None` when nothing is parked.
    pub fn prepare_unpark(&mut self, holder: Holder) -> Result<Option<RoleMove>, RolesError> {
        self.claim = self.claims.get(&holder).copied();
        let unparked = self.prepare_unpark_claimed(holder);
        self.claim = None;
        unparked
    }

    fn prepare_unpark_claimed(&mut self, holder: Holder) -> Result<Option<RoleMove>, RolesError> {
        let (groups, order) = self.unpark_order(holder);
        // An instance whose pair is on its way back with another process's
        // unpark is rebound once that lands: until then its restore is
        // queued again, never told no room for its pair. One parked where
        // it lies (no tier below, [`Item::shed`]) waits the same: let
        // through, it came back with its pair away, so its restore named it
        // no rebind and its next fire ran on the places it gave back.
        if let Some(pair) = groups.iter().find_map(|(owner, _, _)| {
            let key = self.items[owner].pair?;
            (self.items.get(&Owner::Pair(key))?.tier == Tier::Moving).then_some(key)
        }) {
            return Err(RolesError::PairParked(pair));
        }
        let device = Arc::clone(&self.device);
        let mut back: Vec<Vec<Option<Part>>> =
            groups.iter().map(|(_, _, parts)| vec![None; parts.len()]).collect();
        for (at, index) in order {
            let part = &groups[at].2[index];
            match self.take_parts(&device, std::slice::from_ref(part), true) {
                Ok(mut taken) => back[at][index] = Some(taken.remove(0)),
                Err(error) => {
                    let taken: Vec<Part> = back.into_iter().flatten().flatten().collect();
                    self.give(&device, &taken);
                    // It held these parts on the device before its park:
                    // the budget holds them, and the restore waits for
                    // room, never failed for it.
                    if error.is_short() {
                        return Err(RolesError::UnparkShort(Box::new(error)));
                    }
                    return Err(error);
                }
            }
        }
        let mut items: Vec<(Owner, Vec<Part>)> = Vec::new();
        let mut shed: Vec<(Owner, Vec<Part>)> = Vec::new();
        for ((owner, is_shed, parts), taken) in groups.into_iter().zip(back) {
            let parts: Vec<Part> = parts
                .into_iter()
                .zip(taken)
                .map(|(part, taken)| taken.unwrap_or(part))
                .collect();
            if is_shed {
                shed.push((owner, parts));
            } else {
                items.push((owner, parts));
            }
        }
        let below: Vec<Tier> = items.iter().map(|(owner, _)| self.items[owner].tier).collect();
        for (owner, _) in items.iter().chain(&shed) {
            self.items.get_mut(owner).expect("present").tier = Tier::Moving;
        }
        Ok((!items.is_empty() || !shed.is_empty()).then(|| {
            let mut txn = self.moving(false, items, below);
            txn.shed = shed;
            txn
        }))
    }

    /// The process a parked item went with: an instance's or a channel's
    /// holder, an adapter's (or a pair's) first instance's.
    fn parked_with(&self, owner: Owner, item: &Item) -> Option<Holder> {
        match owner {
            Owner::Instance(_) | Owner::Channel(_) => item.holder,
            Owner::Adapter(_) | Owner::Pair(_) => item.attached.iter().find_map(|id| {
                self.items
                    .get(&Owner::Instance(*id))
                    .and_then(|instance| instance.holder)
            }),
        }
    }

    /// Parked items whose live parts stayed on the device for want of room
    /// below ([`Item::shed`]): no fire reads them until their process is
    /// restored, so they are room a move down a tier makes.
    fn parked_on_device(&self) -> impl Iterator<Item = (Owner, &Item, Holder)> {
        self.items.iter().filter_map(|(owner, item)| {
            (item.tier == Tier::Device && item.shed)
                .then(|| self.parked_with(*owner, item))
                .flatten()
                .map(|holder| (*owner, item, holder))
        })
    }

    /// Per logical chunk, the slots parked items' live parts hold on the
    /// device ([`Item::shed`]), by the process each parked with, `except`'s
    /// left out (its restore keeps them where they lie): the chunks moving
    /// them down a tier ([`Self::prepare_park_down`]) empties.
    #[must_use]
    pub fn parked_chunk_slots(&self, except: Holder) -> Vec<(u32, Holder, u32)> {
        let device = self.device.lock().unwrap();
        let mut counts: BTreeMap<(u32, Holder), u32> = BTreeMap::new();
        for (_, item, holder) in self.parked_on_device() {
            if holder == except {
                continue;
            }
            // Room a move makes only where some tier below takes it.
            if !self.lies_below(&item.parts) {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.dead()) {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    let slot = device.slot(shape.pool, slot_ref);
                    *counts.entry((slot.chunk, holder)).or_default() += 1;
                }
            }
        }
        counts
            .into_iter()
            .map(|((chunk, holder), slots)| (chunk, holder, slots))
            .collect()
    }

    /// Moves the items `holders` parked with whose live parts stayed on the
    /// device (no room below at their park, [`Item::shed`]) down a tier, as
    /// the load's tiers lie: to the host where it has room now, slots taken
    /// there for each that fits, to copy into, as a park takes them; where
    /// the load states no host tier (unified memory: its host is the
    /// device's own memory), or with `disk_too` once the host can make no
    /// room, to the disk tier's slot file ([`DiskSlots`]) for a role whose
    /// slot fits one of its slots. Committed, each lies below as if parked
    /// there, and its restore takes fresh device slots
    /// ([`Self::prepare_unpark`]). What the host has no room for, short of
    /// `disk_too`, is counted in host chunks: the room kv on the host
    /// moving on down to disk makes for it, older below newer.
    pub fn prepare_park_down(&mut self, holders: &[Holder], disk_too: bool) -> ParkDown {
        let mut owners: Vec<Owner> = self
            .parked_on_device()
            .filter(|(_, _, holder)| holders.contains(holder))
            .map(|(owner, ..)| owner)
            .collect();
        // Left behind: a shared item (a pair, an adapter) still on the
        // device that moves with one of these parked holders now, its other
        // holders parked or gone since that holder's park (which left it,
        // shared then). It goes down as a parked item does, its dead parts
        // given back (the L40S at 3 GiB on c10d7de5d: a beam pair of 73 dead
        // refs on chunk 17, its last holder evicted, the head one chunk
        // short for good).
        let behind: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                matches!(owner, Owner::Pair(_) | Owner::Adapter(_))
                    && item.tier == Tier::Device
                    && !item.shed
                    && holders.iter().any(|&holder| self.moves_with(**owner, item, holder))
            })
            .map(|(owner, _)| *owner)
            .collect();
        owners.extend(behind);
        let mut items = Vec::new();
        let mut below = Vec::new();
        let mut host_chunks = 0u64;
        for owner in owners {
            let parts = self.items[&owner].parts.clone();
            // Nothing live: nothing to copy, its dead parts given back.
            if parts.iter().all(Part::dead) {
                let emptied: Vec<Part> = parts
                    .iter()
                    .map(|part| Part {
                        refs: Vec::new(),
                        ..part.clone()
                    })
                    .collect();
                self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                items.push((owner, emptied));
                below.push(Tier::Host);
                continue;
            }
            let on_host = self
                .host
                .clone()
                .map(|host| (self.take_parts(&host, &parts, false).ok(), host));
            let to = match on_host {
                Some((Some(to), _)) => Some((to, Tier::Host)),
                Some((None, host)) if !disk_too => {
                    host_chunks += self.chunks_on(&host, &parts);
                    None
                }
                _ => self.take_disk(&parts).map(|to| (to, Tier::Disk)),
            };
            if let Some((to, tier)) = to {
                self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                items.push((owner, to));
                below.push(tier);
            }
        }
        ParkDown {
            txn: (!items.is_empty()).then(|| self.moving(true, items, below)),
            host_chunks,
        }
    }

    /// The whole chunks `part` takes on the device: a guest part its runs
    /// of chunks (a run of slots, one), any other part its slots in whole
    /// chunks of its shape.
    fn part_chunks(&self, device: &super::shared::SharedPool, part: &Part) -> u64 {
        if part.shaped() {
            let (run, pieces) = part.chunk_runs();
            u64::from(run) * u64::from(pieces)
        } else {
            let per_chunk = self
                .shapes
                .of(part.role)
                .and_then(|shape| device.pool().shape(shape.pool))
                .map_or(1, |shape| shape.slots_per_chunk.max(1));
            (part.refs.len() as u64).div_ceil(u64::from(per_chunk))
        }
    }

    /// The bytes `part` holds wherever it lies, one figure on every tier
    /// (a move of it changes no count): a guest part in runs of whole
    /// chunks those chunks, any other part (a run of guest slots, tables,
    /// a ring, an adapter) its slots, dead guest scratch included (given
    /// back at a park, taken fresh at the restore: the process's still).
    fn part_bytes(&self, device: &super::shared::SharedPool, part: &Part) -> u64 {
        let pool = device.pool();
        if part.pieces > 0 {
            return u64::from(part.chunks) * u64::from(part.pieces) * pool.chunk_bytes();
        }
        let slot_bytes = self
            .shapes
            .of(part.role)
            .and_then(|shape| pool.shape(shape.pool))
            .map_or(0, |shape| shape.slot_bytes);
        let slots = if part.run > 0 { part.run as usize } else { part.refs.len() };
        slots as u64 * slot_bytes
    }

    /// The bytes every item holds together, wherever each lies
    /// ([`Self::part_bytes`]), and those the binds on their way took: what
    /// the capacity gate counts guest state and tables by
    /// (`planner::PoolPort::capacity_census`). One figure on every tier: a
    /// move of an item changes no count.
    #[must_use]
    pub fn footprint(&self) -> u64 {
        let device = self.device.lock().unwrap();
        self.items
            .values()
            .flat_map(|item| &item.parts)
            .map(|part| self.part_bytes(&device, part))
            .sum::<u64>()
            + self.on_way.1
    }

    /// The bytes the items parked on the disk tier hold
    /// ([`Self::footprint`]'s share of them): what of the disk chunks they
    /// take is theirs.
    #[must_use]
    pub fn footprint_on_disk(&self) -> u64 {
        let device = self.device.lock().unwrap();
        self.items
            .values()
            .filter(|item| matches!(item.tier, Tier::Disk))
            .flat_map(|item| &item.parts)
            .map(|part| self.part_bytes(&device, part))
            .sum()
    }

    /// The bytes the roles' shapes hold open on the device and on the host
    /// tier that no slot uses: a chunk of small slots (tables, rings) or of
    /// runs of one class is room no other shape takes until it empties.
    /// No item's state: the capacity gate counts it beside the state it
    /// holds, as room that is not free (`planner::PoolPort::capacity_census`).
    #[must_use]
    pub fn slack(&self) -> u64 {
        let shapes: Vec<_> = [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter]
            .into_iter()
            .filter_map(|role| self.shapes.of(role).map(|shape| shape.pool))
            .collect();
        let device = self.device.lock().unwrap().pool().slack_of(&shapes);
        let host = self
            .host
            .as_ref()
            .map_or(0, |host| host.lock().unwrap().pool().slack_of(&shapes));
        device + host
    }

    /// [`Self::footprint`] of what `holder`'s binds took: its items, and
    /// its binds on their way.
    #[must_use]
    pub fn footprint_of(&self, holder: Holder) -> u64 {
        let device = self.device.lock().unwrap();
        self.items
            .values()
            .filter(|item| item.holder == Some(holder))
            .flat_map(|item| &item.parts)
            .map(|part| self.part_bytes(&device, part))
            .sum::<u64>()
            + self.on_way.0.get(&Some(holder)).copied().unwrap_or(0)
    }

    /// `taken` is off its way (bound or given back): its bytes are an
    /// item's now, or free.
    fn off_way(&mut self, taken: &Taken) {
        if let Some(bytes) = self.on_way.0.get_mut(&taken.holder) {
            *bytes = bytes.saturating_sub(taken.bytes);
            if *bytes == 0 {
                self.on_way.0.remove(&taken.holder);
            }
        }
        self.on_way.1 = self.on_way.1.saturating_sub(taken.bytes);
    }

    /// The bytes `taken` took for itself: its own parts, and an adapter's
    /// or a pair's taken for it (another's are their item's).
    fn taken_bytes(&self, taken: &Taken) -> u64 {
        let device = self.device.lock().unwrap();
        let fresh = |part: &Part, fresh: bool| if fresh { self.part_bytes(&device, part) } else { 0 };
        taken
            .parts
            .iter()
            .map(|part| self.part_bytes(&device, part))
            .sum::<u64>()
            + taken.adapter.as_ref().map_or(0, |(_, part, own)| fresh(part, *own))
            // A pair taken fresh is an item from its take: counted there.
    }

    /// `taken` is on its way: its bytes are counted until it is bound or
    /// given back.
    fn on_way_take(&mut self, taken: &mut Taken) {
        taken.bytes = self.taken_bytes(taken);
        *self.on_way.0.entry(taken.holder).or_default() += taken.bytes;
        self.on_way.1 += taken.bytes;
    }

    /// The whole chunks of `on` the live parts of `parts` take.
    fn chunks_on(&self, on: &Shared, parts: &[Part]) -> u64 {
        let pool = on.lock().unwrap();
        parts
            .iter()
            .filter(|part| !part.dead())
            .map(|part| {
                let per_chunk = self
                    .shapes
                    .of(part.role)
                    .and_then(|shape| pool.pool().shape(shape.pool))
                    .map_or(1, |shape| shape.slots_per_chunk.max(1));
                (part.refs.len() as u64).div_ceil(u64::from(per_chunk))
            })
            .sum()
    }

    /// Whether the items of `owners` together fit the device pool's state
    /// chunks at all, whole chunks for each part: room a completion or an
    /// eviction can make.
    fn fits_budget(&self, owners: &[Owner]) -> bool {
        let device = self.device.lock().unwrap();
        let chunks: u64 = owners
            .iter()
            .flat_map(|owner| &self.items[owner].parts)
            .map(|part| self.part_chunks(&device, part))
            .sum();
        chunks <= u64::from(device.state_chunks_now())
    }

    /// The move of `items` (each item's slots to take), with the copies
    /// from the slots each holds now: each item's tier off the device is
    /// `below`'s, in item order, the one it goes to (`out`), else the one
    /// it lies on now.
    fn moving(&self, out: bool, items: Vec<(Owner, Vec<Part>)>, below: Vec<Tier>) -> RoleMove {
        let mut copies: Vec<RoleCopy> = Vec::new();
        for ((owner, to), tier) in items.iter().zip(&below) {
            let disk = *tier == Tier::Disk;
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
                let copy = match copies
                    .iter_mut()
                    .find(|copy| copy.shape == shape.id && copy.disk == disk)
                {
                    Some(copy) => copy,
                    None => {
                        copies.push(RoleCopy {
                            shape: shape.id,
                            device: Vec::new(),
                            host: Vec::new(),
                            disk,
                        });
                        copies.last_mut().expect("pushed")
                    }
                };
                copy.device.extend_from_slice(device);
                copy.host.extend_from_slice(host);
            }
        }
        copies.retain(|copy| !copy.device.is_empty());
        RoleMove {
            out,
            items,
            below,
            copies,
            shed: Vec::new(),
        }
    }

    /// The copies landed: each item lies at its new slots, and its old ones
    /// go back. An unparked item is stale until [`Self::rebound`].
    pub fn commit(&mut self, txn: RoleMove) {
        // Back on the device in fresh slots: a taking of their own.
        let generation = self.next_generation();
        for (owner, parts) in txn.shed {
            let Some(item) = self.items.get_mut(&owner) else {
                continue;
            };
            let old = std::mem::replace(&mut item.parts, parts);
            item.tier = Tier::Device;
            item.shed = txn.out;
            if !txn.out {
                item.generation = generation;
            }
            // Back in place: the engine is told its fresh dead parts.
            item.stale = !txn.out;
            if txn.out {
                let dead: Vec<Part> = old.into_iter().filter(Part::dead).collect();
                let device = Arc::clone(&self.device);
                self.give_freeing(&device, &dead);
            }
            self.free_if_done(owner);
        }
        for ((owner, parts), below) in txn.items.into_iter().zip(txn.below) {
            let Some(item) = self.items.get_mut(&owner) else {
                continue;
            };
            let old = std::mem::replace(&mut item.parts, parts);
            item.tier = if txn.out { below } else { Tier::Device };
            // Moved whole: nothing of it stayed behind (a parked item whose
            // live parts went on down later, [`Self::prepare_park_down`]).
            item.shed = false;
            if !txn.out {
                item.generation = generation;
            }
            item.stale = !txn.out;
            self.give_on(if txn.out { Tier::Device } else { below }, &old);
            self.free_if_done(owner);
        }
        // A pair a member's park left behind, every member now parked.
        if txn.out {
            self.release_unheld_pairs();
        }
        // A bind waiting on a pair's move looks again.
        self.freed
            .gives
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.freed.notify.notify_waiters();
    }

    /// The move failed: each item stays where it was, and the slots taken
    /// for it go back.
    pub fn abort(&mut self, txn: RoleMove) {
        for (owner, parts) in txn.shed {
            if !txn.out {
                // Only the dead parts the unpark took fresh: one the item
                // held in place already ([`Item::takes_fresh`]) stays its.
                let had: Vec<Part> = self
                    .items
                    .get(&owner)
                    .map_or_else(Vec::new, |item| item.parts.clone());
                let dead: Vec<Part> = parts
                    .into_iter()
                    .zip(had.iter().map(Some).chain(std::iter::repeat(None)))
                    .filter(|(part, had)| {
                        part.dead() && had.is_none_or(|had| had.refs.is_empty())
                    })
                    .map(|(part, _)| part)
                    .collect();
                let device = Arc::clone(&self.device);
                self.give(&device, &dead);
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = Tier::Device;
            }
            self.free_if_done(owner);
        }
        for ((owner, parts), below) in txn.items.into_iter().zip(txn.below) {
            if txn.out {
                self.give_on(below, &parts);
            } else {
                let device = Arc::clone(&self.device);
                self.give(&device, &parts);
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = if txn.out { Tier::Device } else { below };
            }
            self.free_if_done(owner);
        }
        // What waits on a pair's move (a bind, a restore) looks again: the
        // move ended, here as at a commit.
        self.tell_room();
    }

    /// The move failed and its copies never settled (the engine could not
    /// say they landed): it ends all the same, each item where the move
    /// found it, so nothing waits on it moving, but the slots it took on
    /// the tier it was going to stay out of service for good, leaked by
    /// name: a late copy may still write them, so they are never given
    /// back nor reused. The slots it leaves still hold what they held (a
    /// copy only reads them). Answers the slots leaked.
    pub fn abandon(&mut self, txn: RoleMove) -> u64 {
        let mut leaked = 0u64;
        for (owner, parts) in txn.shed {
            if !txn.out {
                leaked += parts
                    .iter()
                    .filter(|part| part.dead())
                    .map(|part| part.refs.len() as u64)
                    .sum::<u64>();
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = Tier::Device;
            }
            self.free_if_done(owner);
        }
        for ((owner, parts), below) in txn.items.into_iter().zip(txn.below) {
            leaked += parts.iter().map(|part| part.refs.len() as u64).sum::<u64>();
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = if txn.out { Tier::Device } else { below };
            }
            self.free_if_done(owner);
        }
        self.leaked += leaked;
        // What waits on the move (a bind, a restore of a pair's other
        // holder) looks again: it no longer moves.
        self.tell_room();
        leaked
    }

    /// Every device slot the items hold in slots of their shape (tables,
    /// rings, channels, adapters: not guest runs of chunks), checked
    /// against the pool and each other: each held by one item, and handed
    /// out in the pool. What is said is a slot two items hold, or one an
    /// item holds that the pool counts free: the store's double give that
    /// handed one ref to a restore and a fresh bind together.
    #[cfg(test)]
    #[must_use]
    pub fn audit_device_slots(&self) -> Vec<String> {
        let device = self.device.lock().unwrap();
        let mut seen: HashMap<(SlotRole, u32), Owner> = HashMap::new();
        let mut said = Vec::new();
        for (owner, item) in &self.items {
            if item.tier != Tier::Device {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.shaped()) {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    if let Some(other) = seen.insert((part.role, slot_ref), *owner) {
                        said.push(format!(
                            "{:?} slot {slot_ref} held by {other:?} and {owner:?}",
                            part.role
                        ));
                    }
                    if !device.pool().is_held(shape.pool, device.slot(shape.pool, slot_ref)) {
                        said.push(format!(
                            "{:?} slot {slot_ref} held by {owner:?} is free in the pool",
                            part.role
                        ));
                    }
                }
            }
        }
        drop(device);
        // Nothing moving and no bind on its way: every slot the pool holds
        // out of a role's shape is one an item names (a fresh pair taken by
        // a bind no item recorded held its chunk for good).
        let settled = self.on_way.1 == 0
            && self.binding.is_empty()
            && self.items.values().all(|item| item.tier != Tier::Moving);
        if settled {
            for (role, used, named) in self.audit_held() {
                if used != named {
                    said.push(format!(
                        "{role:?}: the pool holds {used} slots out, the items name {named}"
                    ));
                }
            }
        }
        said
    }

    /// The device slots of each role's shape the pool holds out, beside the
    /// slots the items on the device (and moving) name: `(role, pool,
    /// items)`. Equal, nothing is held that no item names.
    #[cfg(test)]
    #[must_use]
    pub fn audit_held(&self) -> Vec<(SlotRole, u64, u64)> {
        let device = self.device.lock().unwrap();
        let pool = device.pool();
        let mut out = Vec::new();
        for role in [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter] {
            let Some(shape) = self.shapes.of(role) else {
                continue;
            };
            let used: u64 = (0..pool.total_chunks())
                .filter(|&chunk| pool.chunk_shape(chunk) == Some(shape.pool))
                .map(|chunk| u64::from(pool.chunk_used(chunk)))
                .sum();
            let named: u64 = self
                .items
                .values()
                .filter(|item| matches!(item.tier, Tier::Device | Tier::Moving))
                .flat_map(|item| &item.parts)
                .filter(|part| part.role == role)
                .map(|part| part.refs.len() as u64)
                .sum();
            out.push((role, used, named));
        }
        out
    }

    /// Role slots out of service for good: taken by moves whose copies an
    /// engine could not settle ([`Self::abandon`]).
    #[must_use]
    pub fn leaked(&self) -> u64 {
        self.leaked
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
                // Only what lies on the device now: one moved off since its
                // restore (parked again while its rebinds waited) names no
                // slots the engine may claim.
                Owner::Channel(id)
                    if item.stale && item.holder == Some(holder) && item.tier == Tier::Device =>
                {
                    channels.insert(id);
                }
                Owner::Instance(id)
                    if item.stale && item.holder == Some(holder) && item.tier == Tier::Device =>
                {
                    instances.insert(id);
                }
                // Its holders on the device; a parked one is rebound when it
                // unparks. One parked where it lies (no tier below took its
                // live parts, its dead ones given back: [`Item::shed`]) is
                // parked too: rebound now, it named no scratch at all ("a
                // guest item ... and its binding names []" on another
                // process's restore of the pair).
                Owner::Adapter(_) | Owner::Pair(_)
                    if item.stale && self.unparks_with(*owner, item, holder) =>
                {
                    instances.extend(item.attached.iter().copied().filter(|id| {
                        self.items
                            .get(&Owner::Instance(*id))
                            .is_some_and(|instance| instance.tier == Tier::Device && !instance.shed)
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
                // A guest part given back (its process parked where it lies,
                // [`Item::shed`]) lies nowhere: never named to the engine;
                // its own restore takes it back first, and rebinds it then.
                if item.shed || item.parts.iter().any(|part| part.dead() && part.refs.is_empty()) {
                    return None;
                }
                // Its pair not on the device yet: rebound once it lands,
                // never told the pair lies nowhere.
                if let Some(key) = item.pair
                    && self
                        .items
                        .get(&Owner::Pair(key))
                        .is_none_or(|pair| pair.tier != Tier::Device)
                {
                    return None;
                }
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
                    pair_generation: item
                        .pair
                        .and_then(|key| self.items.get(&Owner::Pair(key)))
                        .map_or(0, |pair| pair.generation),
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

    /// `holder`'s open instances not yet bound where they lie: off the
    /// device, moving, parked where they lie, or back but not rebound. A
    /// restore reports its process resident only once this is empty: a
    /// fire of an instance the engine still binds at the places it gave
    /// back at its park runs on room others hold since.
    #[must_use]
    pub fn unbound(&self, holder: Holder) -> Vec<InstanceId> {
        self.items
            .iter()
            .filter_map(|(owner, item)| match *owner {
                Owner::Instance(id)
                    if item.holder == Some(holder)
                        && item.open
                        && (item.tier != Tier::Device || item.stale || item.takes_fresh()) =>
                {
                    Some(id)
                }
                _ => None,
            })
            .collect()
    }

    /// `holder`'s open instances an unpark of `txn` (none: nothing to
    /// unpark) would leave unbound, its rebinds unable to name them: off
    /// the device or moving and not in it, short of a dead part it does
    /// not take ([`Item::takes_fresh`]), or holding a pair that is not on
    /// the device and not in it. A restore commits only once this is empty
    /// ([`Self::prepare_unpark`] then [`Self::commit`]): committed with
    /// one, it waited Restoring on a rebind nothing would make (Mac Pro at
    /// 2 GiB on 1e87ab171, InstanceId(5)).
    #[must_use]
    pub fn unbound_after(&self, holder: Holder, txn: Option<&RoleMove>) -> Vec<InstanceId> {
        let moved = |owner: Owner| {
            txn.is_some_and(|txn| {
                txn.items.iter().chain(&txn.shed).any(|(moved, _)| *moved == owner)
            })
        };
        self.items
            .iter()
            .filter_map(|(owner, item)| {
                let Owner::Instance(id) = *owner else {
                    return None;
                };
                if item.holder != Some(holder) || !item.open {
                    return None;
                }
                let placed = moved(*owner)
                    || (item.tier == Tier::Device && !item.shed && !item.takes_fresh());
                let pair = item.pair.is_none_or(|key| {
                    moved(Owner::Pair(key))
                        || self
                            .items
                            .get(&Owner::Pair(key))
                            .is_some_and(|pair| pair.tier == Tier::Device && !pair.takes_fresh())
                });
                (!placed || !pair).then_some(id)
            })
            .collect()
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

    /// Whether `owner` is still the engine's to rebind: an instance not
    /// closed, a channel some open instance or its own registration still
    /// holds. A restore reads its rebinds off the roles ([`Self::stale`])
    /// and the lane runs them in turn, closes between them: one the lane
    /// closed meanwhile is gone from the engine, and nothing reads its
    /// slots, so its rebind is cancelled, never sent.
    #[must_use]
    pub fn rebinds(&self, owner: Owner) -> bool {
        self.items.get(&owner).is_some_and(|item| match owner {
            Owner::Instance(_) => item.open,
            _ => true,
        })
    }

    /// Whether `rebind`, read off the roles before the lane ran it, still
    /// says where its item lies: a park of an instance not closed since; a
    /// rebind of an item that is on the device, not parked where it lies,
    /// at the slots it names. One whose item has moved since (parked again
    /// by an eviction or a bind's suspension while its restore waited on
    /// its rebinds, its device slots given back at that park's commit) is
    /// cancelled: sent, the engine claimed slots the store had handed out
    /// again, and the next bind there was refused (the L40S and the M1 on
    /// batch 3: instance slot 12537 rebound 4 ms after its park, the same
    /// refs, then bound to another instance 1 ms later). The move that
    /// brings it back names it anew.
    #[must_use]
    pub fn rebind_stands(&self, rebind: &Rebind) -> bool {
        let owner = Owner::of(rebind);
        if !self.rebinds(owner) {
            return false;
        }
        let Some(item) = self.items.get(&owner) else {
            return false;
        };
        match rebind {
            Rebind::Parked { .. } => true,
            Rebind::Instance { instance, .. } => {
                item.tier == Tier::Device
                    && !item.shed
                    && item.parts.first().map_or(instance.is_empty(), |part| part.refs == *instance)
            }
            Rebind::Channel { slots, .. } => item.tier == Tier::Device && item.refs() == *slots,
        }
    }

    /// Instances `ids` were said parked to the engine and their park was
    /// taken back: each on the device where it lay, not parked where it
    /// lies, is rebound at its slots ([`Self::stale`]) before its process
    /// fires again, so the engine never believes a park the runtime undid.
    pub fn park_undone(&mut self, ids: &[InstanceId]) {
        for id in ids {
            if let Some(item) = self.items.get_mut(&Owner::Instance(*id))
                && item.tier == Tier::Device
                && !item.shed
            {
                item.stale = true;
            }
        }
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
            .filter(|item| !matches!(item.tier, Tier::Host | Tier::Disk))
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
#[cfg(test)]
pub fn register_channel<T>(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
    holder: Option<Holder>,
    register: impl FnOnce(&mut dyn Engine, &ChannelRegistration) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let taken = take_ring(roles, engine, plan)?;
    register_taken(roles, engine, plan, holder, taken, register)
}

/// The slots a registration of `plan` takes for its ring, before anything
/// is registered: short of room a budget holds, [`RolesError::BindShort`].
/// `None` where no roles are kept or the engine keeps the ring itself.
pub fn take_ring(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
) -> anyhow::Result<Option<TakenRing>> {
    let Some(roles) = roles else {
        return Ok(None);
    };
    let Some(facts) = engine.ring_facts(plan)? else {
        return Ok(None);
    };
    let mut roles = roles.lock().unwrap();
    Ok(Some(
        roles
            .take_channel(&facts)
            .map_err(|error| roles.bind_short(error, None, 0))?,
    ))
}

/// The rings of every registration of `plans` ([`take_ring`]), all or
/// none.
pub fn take_rings(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plans: &[ChannelRegistration],
) -> anyhow::Result<Vec<Option<TakenRing>>> {
    let mut rings = Vec::with_capacity(plans.len());
    for plan in plans {
        match take_ring(roles, engine, plan) {
            Ok(ring) => rings.push(ring),
            Err(error) => {
                untake_rings(roles, rings);
                return Err(error);
            }
        }
    }
    Ok(rings)
}

/// The channels of `plans`, registered by the bind they come with, whose
/// rings the engine holds ([`Engine::holds_channel`]): with the channels
/// it holds registered already, what a bind's `registered` names
/// ([`take_bind`]). A channel the engine leaves to the runtime is each
/// instance's own ring, registered by this bind or not.
///
/// [`Engine::holds_channel`]: ::engine::Engine::holds_channel
#[must_use]
pub fn held_by(engine: &dyn Engine, plans: &[ChannelRegistration]) -> Vec<ChannelId> {
    plans
        .iter()
        .filter(|plan| engine.holds_channel(plan))
        .map(|plan| plan.id)
        .collect()
}

/// Gives back rings taken for registrations that will not happen.
pub fn untake_rings(roles: Option<&RolesHandle>, rings: Vec<Option<TakenRing>>) {
    if let Some(roles) = roles {
        let mut roles = roles.lock().unwrap();
        for ring in rings.into_iter().flatten() {
            roles.untake_channel(ring);
        }
    }
}

/// Registers `plan` over the ring [`take_ring`] took for it (`taken`).
pub fn register_taken<T>(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
    holder: Option<Holder>,
    taken: Option<TakenRing>,
    register: impl FnOnce(&mut dyn Engine, &ChannelRegistration) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let (Some(roles), Some(taken)) = (roles, taken) else {
        return register(engine, plan);
    };
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
#[cfg(test)]
pub fn bind_instance(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    holder: Option<Holder>,
    registered: impl Fn(ChannelId) -> bool,
) -> anyhow::Result<BoundInstance> {
    let taken = take_bind(roles, engine, binding, registered, holder, |_| true, |_, _| false)?;
    bind_taken(roles, engine, binding, holder, taken)
}

/// Everything a bind of `binding` takes of the pool, before the bind
/// registers anything: its guest state ([`Roles::take_guest`]), then its
/// tables, its own rings and its adapter ([`Roles::take_instance`]); a
/// ring of a channel `registered` names (registered with the engine, or
/// to be by this bind where the engine holds it: [`held_by`]) is the
/// channel's. Short of room a budget holds, the bind is
/// answered [`RolesError::BindShort`] and waits for it with nothing to
/// roll back. `None` where no roles are kept.
pub fn take_bind(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    registered: impl Fn(ChannelId) -> bool,
    holder: Option<Holder>,
    resident: impl Fn(Holder) -> bool,
    gated: impl Fn(Holder, u32) -> bool,
) -> anyhow::Result<Option<Taken>> {
    let Some(roles) = roles else {
        return Ok(None);
    };
    let roles_handle = roles;
    if let Some(holder) = holder.filter(|&holder| !resident(holder)) {
        return Err(RolesError::HolderParked(holder).into());
    }
    learn_program(roles, engine, binding.program)?;
    let state = engine.guest_state(binding.program, &binding.extents, binding.geometry)?;
    let adapter = engine.adapter_key(binding)?;
    let pair = FireKey {
        program: binding.program,
        extents: binding.extents,
    };
    let mut roles = roles.lock().unwrap();
    let free_before = roles.device.lock().unwrap().pool().free_chunks();
    let beside = roles.instance_parts_fresh(binding.program, &binding.channels, &registered, adapter);
    let guest = roles.take_guest(&binding.channels, (pair, state), holder, beside)?;
    let fresh = guest.fresh_runs();
    let mut taken = roles
        .take_instance(
            binding.program,
            &binding.channels,
            &registered,
            adapter,
            guest,
        )
        .map_err(|(error, opened)| roles.bind_short(error, fresh, opened))?;
    taken.holder = holder;
    roles.bind_taken_for(holder);
    roles.on_way_take(&mut taken);
    let added = free_before.saturating_sub(roles.device.lock().unwrap().pool().free_chunks());
    drop(roles);
    // What it added: the fresh chunks it opened, its new state
    // ([`Roles::footprint`]) and the room those chunks hold open past it
    // ([`Roles::slack`]); slots it took in chunks open already add state
    // and take as much slack, so none. Past what every tier keeps for the
    // finisher (`gated`), the take goes back and the bind waits in the
    // planner for the gate
    // ([`crate::planner::ResidencyPlanner::capacity_holds_chunks`]).
    if let Some(holder) = holder
        && added > 0
        && gated(holder, added)
    {
        roles_handle.lock().unwrap().untake(taken);
        // The room it waits for is the whole bind's, as a refusal for want
        // of room asks ([`Roles::take_guest`]): what it opened now (slots
        // its tables took in chunks open now) may be gone by the time its
        // process, suspended meanwhile, is restored with that room, and the
        // bind is short again (the simulation's field mix, seed 80: held
        // opening 1 chunk, restored with it, short of 4).
        // Runs of more than a chunk are asked as runs: asked as single
        // chunks, room was made in chunks that formed no run (the
        // simulation's seed 189: a victim moved for four single chunks, the
        // bind's run of two still short).
        let room = match fresh {
            Some((run, pieces)) if run > 1 => GuestRoom::Runs {
                run,
                pieces,
                free_runs: 0,
                free_chunks: 0,
                committable: 0,
                singles: beside,
            },
            fresh => GuestRoom::Slots {
                slots: 0,
                committable: 0,
                chunks: added.max(fresh.map_or(0, |(_, pieces)| pieces).saturating_add(beside)),
            },
        };
        return Err(RolesError::BindShort(Box::new(RolesError::GuestShort {
            what: format!(
                "its {added} new chunks of state, past what every tier keeps for the finisher"
            ),
            bytes: 0,
            room,
        }))
        .into());
    }
    // Read again past the take: a park that began since the first read
    // either saw this bind on its way and waits for it, or parked before
    // it was taken, and then the holder reads parked here and the take
    // goes back.
    if let Some(holder) = holder.filter(|&holder| !resident(holder)) {
        roles_handle.lock().unwrap().untake(taken);
        return Err(RolesError::HolderParked(holder).into());
    }
    Ok(Some(taken))
}

/// Binds `binding` over what [`take_bind`] took for it (`taken`): the
/// engine's bind, and what it took kept for `holder`, or given back where
/// the engine refuses it.
pub fn bind_taken(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    holder: Option<Holder>,
    taken: Option<Taken>,
) -> anyhow::Result<BoundInstance> {
    let (Some(roles), Some(taken)) = (roles, taken) else {
        return Ok(engine.bind_instance(binding)?);
    };
    let binding = InstanceBinding {
        instance_slots: taken.instance_slots(),
        ring_slots: taken.ring_slots(),
        adapter_slots: taken.adapter_slots(),
        scratch: taken.scratch(),
        pair: taken.pair(),
        pair_generation: taken.pair_generation(),
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
/// segment left to commit binds again too, short of room the planner then
/// makes ([`RolesError::BindShort`]), never refused for it.
///
/// This is not a retry: nothing is attempted again on the chance it now
/// fits, and nothing is degraded or made smaller. The bind is sent again
/// only after the planner has restored `pid`, or the walk has committed a
/// segment, the event the refusal named, exactly as a fire held back by a
/// suspension is sent after its restore.
///
/// However the wait ends (bound, refused, or dropped with its process),
/// its need is no longer counted (`leave`, [`Roles::leave_guest_wait`]).
///
/// Before any wait, its process's fires already submitted are settled
/// (`settle`): their leases on its working sets go back as they land, as
/// they do at any other wait of the process's ([`crate::inferlet::process::gate`]).
/// Waiting with them unsettled, the process was held between an eviction
/// that waits for those leases and a bind that waits on that eviction.
/// A wait for room its own eviction withdrew
/// ([`crate::planner::BindRoomWait::Withdrawn`]) waits for its restore.
#[allow(clippy::too_many_arguments)]
pub async fn bind_when_resident<T, E, B, W, G, P, S>(
    pid: Holder,
    mut bind: impl FnMut() -> B,
    resident: impl Fn() -> W,
    grown: impl Fn(u32, u32) -> G,
    room: impl Fn() -> Option<RoomWatch>,
    fits: impl Fn(u32, u32, u32) -> bool,
    suspend: impl Fn(u32, u32, u32) -> bool,
    planned: impl Fn(u32, u32, u32) -> P,
    leave: impl Fn(),
    waiting: impl Fn(bool),
    settle: impl Fn() -> S,
) -> anyhow::Result<T>
where
    E: std::fmt::Display,
    B: std::future::Future<Output = anyhow::Result<T>>,
    W: std::future::Future<Output = Result<(), E>>,
    G: std::future::Future<Output = Result<(), String>>,
    P: std::future::Future<Output = Result<crate::planner::BindRoomWait, String>>,
    S: std::future::Future<Output = ()>,
{
    /// Runs its `leave` on drop: the wait's every end, its process gone
    /// (the future dropped) too.
    struct Leaves<'a, F: Fn()>(&'a F);
    impl<F: Fn()> Drop for Leaves<'_, F> {
        fn drop(&mut self) {
            (self.0)();
        }
    }
    let _leaves = Leaves(&leave);
    // Waiting for room, the process yields to the planner as an idle one
    // does (`waiting`); past the wait, however it ends, it no longer does.
    let waits = || waiting(true);
    struct Ends<'a, F: Fn(bool)>(&'a F);
    impl<F: Fn(bool)> Drop for Ends<'_, F> {
        fn drop(&mut self) {
            (self.0)(false);
        }
    }
    let _ends = Ends(&waiting);
    // Suspended for this bind's room once and restored: short again, it
    // waits parked for room the planner makes, never suspended and evicted
    // again (mx-2: suspend, evict, restore, short, every ~25 ms, 222,924
    // evictions and no serve).
    let mut restored_for_room = false;
    loop {
        // Taken before the bind: room given back meanwhile is not lost.
        let watch = room();
        // Binding again, it is no victim while the bind is under way.
        waiting(false);
        let error = match bind().await {
            Ok(bound) => return Ok(bound),
            Err(error) => error,
        };
        // Every way on from here waits, but a refusal: what the process
        // has in flight lands and gives its leases back first.
        let waits_on = error.downcast_ref::<RolesError>().is_some_and(|error| {
            matches!(
                error,
                RolesError::PairParked(_) | RolesError::GuestGrow { .. }
            ) || error.room_runs().is_some()
                || matches!(error, RolesError::Parked { holder: Some(holder), .. } if *holder == pid)
                || matches!(error, RolesError::HolderParked(holder) if *holder == pid)
        });
        if waits_on {
            settle().await;
        }
        if let Some(RolesError::PairParked(key)) = error.downcast_ref::<RolesError>()
            && let Some(watch) = &watch
        {
            tracing::debug!(%pid, %key, "a bind waits for its pair's move to land");
            let freed = watch.clone().freed();
            crate::inferlet::process::said("bind: its pair's move", pid, async move {
                // Waiting on its pair's move: its owner is parked here.
                let _parked = crate::inferlet::process::parks_here();
                freed.await
            })
            .await;
            continue;
        }
        if let Some((run, pieces, singles)) = error
            .downcast_ref::<RolesError>()
            .and_then(RolesError::room_runs)
        {
            // Holding guest state of its own, it would wait on itself: it is
            // suspended, its state given back, and binds once restored with
            // room for this bind (`run`, `pieces`) beside its own.
            if !restored_for_room && suspend(run, pieces, singles) {
                tracing::debug!(%pid, %error, "a bind short of room suspends its process");
                // Waiting for its restore, not for room: its need is not
                // counted meanwhile.
                leave();
                if let Err(why) = resident().await {
                    return Err(error.context(format!(
                        "its process was suspended for room and is not resident again: {why}"
                    )));
                }
                restored_for_room = true;
                continue;
            }
            // Else it waits, queued in the planner, for room: given back
            // (a close or a park, `watch`), or made for it as for a
            // restore short of room for its roles (`planned`: idle cache
            // dropped first, then a resident parked by the planner's
            // victim choice).
            // Room given back wakes it only where its runs (`fits`) now
            // stand free: it is sent again on the room it waits for, never
            // on any give.
            tracing::debug!(%pid, %error, "a bind waits in the planner for room");
            waits();
            let freed = async {
                let Some(mut watch) = watch else {
                    return std::future::pending().await;
                };
                loop {
                    let next = room();
                    watch.freed().await;
                    if fits(run, pieces, singles) {
                        return;
                    }
                    match next {
                        Some(next) => watch = next,
                        None => return std::future::pending().await,
                    }
                }
            };
            tokio::select! {
                // In order, never at random: a seeded run takes the same branch.
                biased;
                () = freed => {}
                made = planned(run, pieces, singles) => match made {
                    Err(why) => {
                        return Err(error.context(format!("its wait for room ended: {why}")));
                    }
                    Ok(crate::planner::BindRoomWait::Served) => {}
                    Ok(crate::planner::BindRoomWait::Withdrawn) => {
                        tracing::debug!(%pid, "a bind's wait for room ends with its eviction");
                        leave();
                        waiting(false);
                        if let Err(why) = resident().await {
                            return Err(error.context(format!(
                                "its process was evicted and is not resident again: {why}"
                            )));
                        }
                    }
                },
            }
            continue;
        }
        if let Some(RolesError::GuestGrow {
            prefix, run, room: short, ..
        }) = error.downcast_ref::<RolesError>()
        {
            let (prefix, run) = (*prefix, *run);
            let (pieces, singles) = (short.chunk_runs().1, short.singles());
            tracing::debug!(%pid, %error, "a bind waits on a segment commit");
            waits();
            // A commit, room given back (its runs may stand free in it), or
            // the planner's serve: any ends the wait, and it binds again.
            // The planner is asked too, as for a bind short of room
            // (`planned`): where the governor's target stops the walk
            // committing after all, it is the only one that makes room
            // (idle cache dropped, a resident parked), and it parks no one
            // for a grow that is coming.
            let freed = async {
                match watch {
                    Some(watch) => watch.freed().await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                // In order, never at random: a seeded run takes the same branch.
                biased;
                // No segment left to commit within the budget by the time
                // it looked (the room the refusal counted was committed
                // for others, or the target came down): the room is the
                // planner's to make now, never the client's refusal. It
                // binds again, and a bind short of room then waits in the
                // planner (`BindShort`); only a need that can never fit
                // alone is refused (Mac Pro at 5 GiB on 1e87ab171: two
                // first-frame binds refused to the client "no segment is
                // left to commit" beside "the segments left ... hold 3
                // more", two reads at two instants).
                grew = grown(prefix, run) => {
                    if let Err(why) = grew {
                        tracing::debug!(%pid, %why, "a bind's commit will not come: the planner makes its room");
                    }
                }
                () = freed => {}
                made = planned(run, pieces, singles) => match made {
                    Err(why) => {
                        return Err(error.context(format!("its wait for room ended: {why}")));
                    }
                    Ok(crate::planner::BindRoomWait::Served) => {}
                    Ok(crate::planner::BindRoomWait::Withdrawn) => {
                        leave();
                        waiting(false);
                        if let Err(why) = resident().await {
                            return Err(error.context(format!(
                                "its process was evicted and is not resident again: {why}"
                            )));
                        }
                    }
                },
            }
            continue;
        }
        let parked_with_it = matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::Parked { holder: Some(holder), .. } | RolesError::HolderParked(holder))
                if *holder == pid
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
                pool.set_guest_head(None);
                return Err(format!(
                    "no free run of {run} adjacent chunks, and no segment is left to commit \
                     within the budget"
                ));
            }
        }
        crate::store::shared::ask_walk(engine);
        // Waiting on a segment's commit: its owner is parked here.
        let _parked = crate::inferlet::process::parks_here();
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
    // Run on the lane, after any close the lane ran since the restore read
    // its rebinds: a closed instance is the engine's no longer, and the
    // runtime never rebinds what it closed.
    if let Some(roles) = roles
        && !roles.lock().unwrap().rebind_stands(&rebind)
    {
        tracing::debug!(
            ?owner,
            "a rebind cancelled: its item closed or moved since it was read"
        );
        return Ok(());
    }
    let parked = matches!(rebind, Rebind::Parked { .. });
    engine.rebind_slots(rebind)?;
    // A park names no new slots: nothing is rebound, no held write goes.
    if parked {
        return Ok(());
    }
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
