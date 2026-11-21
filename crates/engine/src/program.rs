use serde::{Deserialize, Serialize};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::EmittedKernel;
use eta_ir::registry::GeometryClass;

use crate::channel::ChannelSeed;

mod term;

pub use term::{FirePair, GuestTerm};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct ProgramId(pub u64);

impl std::fmt::Display for ProgramId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InstanceId(pub u64);

impl std::fmt::Display for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramRegistration {
    pub program_hash: u64,
    pub emitted_kernels: Vec<EmittedKernel>,
    pub emitter_version: u32,
    pub region_analysis: Vec<RegionAnalysis>,
    pub launch: LaunchPackage,
    pub reference_ptir: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindExtents {
    pub row_count: u32,
    pub token_count: u32,
    /// The readout rows one instance reads (its rows per instance), what
    /// every engine checks its readout seats against: a beam group's
    /// (`parents`) instance reads its `b` beams' rows.
    pub sampled_rows: u32,
    pub query_len: u32,
    pub key_len: u32,
}

impl Default for BindExtents {
    fn default() -> BindExtents {
        BindExtents {
            row_count: 1,
            token_count: 1,
            sampled_rows: 1,
            query_len: 1,
            key_len: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InstanceBinding {
    pub program: ProgramId,
    pub channels: Vec<crate::channel::ChannelId>,
    pub seeds: Vec<ChannelSeed>,
    pub geometry: GeometryClass,
    pub extents: BindExtents,
    /// The descriptor ports the program computes on the device, which the
    /// engine resolves for a decode envelope; the runtime resolves its other
    /// ports on the host and states them on the lanes. Each port has one
    /// source ([`crate::ports::engine_resolves`]).
    #[serde(default)]
    pub device_resolved: eta_ir::registry::PortMask,
    /// The `Instance` slots the runtime took for this instance's tables, as
    /// many as [`ProgramFacts::instance`] states. Empty where the program's
    /// facts state none: the engine then places the tables itself.
    #[serde(default)]
    pub instance_slots: Vec<u32>,
    /// The `Channel` slots of each ring this instance holds itself, one
    /// entry per [`ProgramFacts::rings`] entry the binding does not cover
    /// with a registered channel in `channels`.
    #[serde(default)]
    pub ring_slots: Vec<RingSlots>,
    /// The `Adapter` slots the runtime took for the adapter this instance
    /// lands, as many as [`ProgramFacts::adapter`] states. Empty where the
    /// program binds none.
    #[serde(default)]
    pub adapter_slots: Vec<u32>,
    /// Where this instance's lanes' scratch lies ([`GuestState::scratch`]):
    /// dead between fires.
    #[serde(default)]
    pub scratch: ScratchPlace,
    /// Where its resident state lies ([`GuestState::resident`]): one run of
    /// adjacent `Instance` slots, live from the bind to the close, copied
    /// wherever a park moves the instance and laid again at the same bytes
    /// in the slots it comes back to ([`Rebind::Instance`]).
    #[serde(default)]
    pub resident: ScratchPlace,
    /// Where its pair's lies ([`GuestState::per_pair`]): the same on every
    /// instance of the pair, taken at the pair's first bind and given back
    /// at its last close.
    #[serde(default)]
    pub pair: ScratchPlace,
    /// Which taking of the pair's slots [`Self::pair`] names: the runtime
    /// renews it each time the pair's state is given fresh slots (its first
    /// bind, a return from the host), so slots freed, written by another
    /// owner and taken again at the same refs read as new. The engine lays
    /// the pair's state again wherever it differs from the one it laid, and
    /// not where the same live slots are bound again.
    #[serde(default)]
    pub pair_generation: u64,
}

/// What a guest program bound at given extents holds in the pool, as state
/// (REDESIGN §6): the scratch each bound instance keeps for its lanes, and
/// what the (program, extents) pair keeps once while any instance of it is
/// bound ([`crate::Engine::guest_state`]). Each lies in the `Instance` role,
/// taken at the bind and held while bound ([`InstanceBinding::scratch`],
/// [`InstanceBinding::resident`], [`InstanceBinding::pair`],
/// [`ScratchPlace`]). The scratch's and the pair's contents are dead between
/// fires: a park gives them back and an unpark takes fresh room, nothing
/// copied. The resident state is live: a park copies it wherever the
/// instance goes and an unpark copies it back into fresh slots. The engine
/// is told where each lies ([`Rebind::Instance`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestState {
    /// Bytes an instance holds from bind to unbind that a later fire reads
    /// (a beam group's page lists): one run of adjacent `Instance` slots, at
    /// most a chunk, taken at the bind, priced and moved as the instance's
    /// tables are ([`InstanceBinding::resident`]).
    #[serde(default)]
    pub resident: u64,
    /// Bytes ONE fire of the instance needs for its value scratch: dead outside that
    /// fire, never read by a later one.
    #[serde(default)]
    pub scratch: u64,
    /// Fires of one instance this engine may have in flight at once for the class
    /// being bound, each needing its own scratch: the arms (>= 1).
    #[serde(default = "one_arm")]
    pub scratch_arms: u32,
    /// The pair's tables, held once while any instance of it is bound (a layout
    /// cache: stays resident; unchanged).
    #[serde(default)]
    pub per_pair: u64,
    /// Chunks a piece takes (unchanged; applies to resident and to each scratch arm).
    /// A run of `run` adjacent chunks inside one segment, the least that
    /// holds the program's widest value (or its temporary area) whole, so
    /// each piece is addressed from its first chunk (0 or 1: a chunk). Its
    /// bytes past a chunk's are whole runs, `resident`, `scratch` and
    /// `per_pair` a multiple of `run` chunks.
    #[serde(default)]
    pub run: u32,
}

fn one_arm() -> u32 {
    1
}

impl Default for GuestState {
    fn default() -> Self {
        Self {
            resident: 0,
            scratch: 0,
            scratch_arms: 1,
            per_pair: 0,
            run: 0,
        }
    }
}

impl GuestState {
    /// Bytes an instance holds with every scratch arm held: [`Self::resident`]
    /// plus [`Self::scratch`] times [`Self::scratch_arms`] (at least one).
    #[must_use]
    pub fn instance_bytes(&self) -> u64 {
        self.resident
            .saturating_add(self.scratch.saturating_mul(u64::from(self.scratch_arms.max(1))))
    }

    /// The chunks a piece takes: [`Self::run`], at least one.
    #[must_use]
    pub fn chunks_a_piece(&self) -> u32 {
        self.run.max(1)
    }
}

/// Where a guest's scratch lies in the pool ([`GuestState`]): bytes that fit
/// in a chunk's `Instance` slots (its slots times their bytes) as one run of
/// adjacent slots in one chunk, else whole runs of adjacent chunks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScratchPlace {
    /// None held.
    #[default]
    None,
    /// `slots` adjacent `Instance` slots from slot ref `first`, in its chunk.
    Run { first: u32, slots: u32 },
    /// Pieces of `run` adjacent chunks inside one segment each
    /// ([`GuestState::run`]): each piece's first chunk, by logical chunk
    /// id. A piece is addressed from its first chunk's first byte; the
    /// chunks after it lie at the next physical chunks, which the pool
    /// keeps so (an evacuation moves a run whole).
    Chunks { run: u32, pieces: Vec<u32> },
}

/// What an instance of a registered program holds in the pool, beside the
/// fire's bytes (REDESIGN 2.6, `new-memory/STATE-ROLES.md`). The engine
/// counts slots with the slot bytes it stated for the `Instance` and
/// `Channel` shapes in [`crate::chunk::ChunkFacts`]; an engine that states
/// neither shape returns the default, and keeps the tables and rings itself.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramFacts {
    /// The instance's bind-time launch tables.
    pub instance: Footprint,
    /// The rings the program declares, by its channel index. A binding
    /// that names a registered channel for one holds no slots for it.
    pub rings: Vec<RingFacts>,
    /// The adapter an instance lands at bind (the planes its program's
    /// correction reads), in `Adapter` slots; none where it lands none.
    #[serde(default)]
    pub adapter: Footprint,
}

/// The lanes an engine lays a guest program's flight of `members` lanes
/// for, stated in [`FireFacts::laid`]: what a fire of `members` takes of
/// the span is the bytes of the lanes laid, not of the members in it, so
/// the term a rung backs counts these lanes.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum LaidLanes {
    /// The members themselves: each takes its own lane's bytes.
    #[default]
    Exact,
    /// The power of two over the members: the engine keeps one layout of a
    /// flight's lane tables (or one executable) for every count in a power
    /// of two, so 170 members are laid as 256.
    PowerOfTwo,
}

impl LaidLanes {
    /// The lanes laid for a flight of `members`.
    #[must_use]
    pub fn of(self, members: u64) -> u64 {
        match self {
            LaidLanes::Exact => members,
            // No member lays no lane (the power of two over 0 is 1).
            LaidLanes::PowerOfTwo if members == 0 => 0,
            LaidLanes::PowerOfTwo => members.checked_next_power_of_two().unwrap_or(u64::MAX),
        }
    }
}

/// What a fire holding lanes of a program bound at given extents needs
/// beside the model's own activation (REDESIGN 2.4, §6): the per-batch
/// tables and value scratch of the program's stage, sized by the binding's
/// [`BindExtents`] (the same program at 256 and 4096 rows differs 16x).
///
/// The facts are a pair's, a (program, extents) bound; the term is a
/// fire's. A fire of a rung of `r` tokens carries `min(r,
/// FireLimits::max_lanes)` members in all, split among the pairs open. A
/// pair given `m ≥ 1` of them (no more than its bound instances) is laid
/// as `l = min(laid.of(m), max_lanes)` lanes and needs `fixed + l ×
/// a_lane` bytes; a pair given none needs nothing, its `fixed` neither.
/// The runtime sets aside, at every rung, what the worst such fire takes
/// ([`GuestTerm`], [`crate::Engine::fire_facts`]), and an engine that
/// backs its own carve counts the same, from the same function. A program
/// registers after load, so `ChunkFacts` cannot state it.
///
/// `fixed` is charged to each pair a fire carries. An engine may charge to
/// it a table it lays once a fire whatever the pairs in it (CUDA: its
/// wave's rows): the term then counts that table once for every pair
/// carried where the fire lays it once, an upper bound over what the fire
/// lays by `(pairs carried − 1) ×` that table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireFacts {
    #[serde(default)]
    pub a_lane: u64,
    #[serde(default)]
    pub fixed: u64,
    /// The most lanes one fire of the program flies (CUDA: its flight, a
    /// power of two under the SM count); 0: no cap of its own, the fire's
    /// `FireLimits::max_lanes` only.
    #[serde(default)]
    pub max_lanes: u32,
    /// The lanes the engine lays for a flight's members. The engine states
    /// it and lays by it; the runtime sizes the term by it and has no rule
    /// of its own.
    #[serde(default)]
    pub laid: LaidLanes,
}

impl FireFacts {
    /// The lanes a fire carrying `members` instances of the program lays:
    /// [`Self::laid`] over them, capped by [`Self::max_lanes`] where it
    /// states one.
    #[must_use]
    pub fn lanes(&self, members: u64) -> u64 {
        let laid = self.laid.of(members);
        if self.max_lanes == 0 {
            laid
        } else {
            laid.min(u64::from(self.max_lanes))
        }
    }

    /// The bytes a fire of `members` instances needs for the program; a
    /// fire that carries none of it needs none, its `fixed` neither.
    #[must_use]
    pub fn bytes(&self, members: u64) -> u64 {
        if members == 0 {
            return 0;
        }
        self.fixed
            .saturating_add(self.lanes(members).saturating_mul(self.a_lane))
    }
}

/// Where the rows a fire reads out land ([`ReadoutFacts::landing`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadoutLanding {
    /// Inside the activation the rungs state already (CUDA: a slot of the
    /// compiled arena, carved with its rung): the runtime counts nothing
    /// for them. What an engine that states nothing is taken to say.
    #[default]
    InSpan,
    /// In the span beside the model's activation, `row_bytes` for each row
    /// a fire reads out: the fire's readout term, counted beside its guest
    /// term at every rung.
    Beside { row_bytes: u64 },
}

/// What the rows a fire reads out take, as the engine states them
/// ([`crate::Engine::readout_facts`]): a fire bound, never a load
/// allocation sized at a ceiling. The facts are the engine's, whatever
/// programs a fire carries: every fire reads out (a plain decode its last
/// row), so they are no (program, extents) pair's [`FireFacts`].
///
/// The rows one fire's sequence lanes read out (sampled rows: a decode's
/// last, a verify's block) are at most its tokens, at most the engine's
/// `FireLimits::max_readouts` where it states one, and at most
/// [`Self::store_rows`] where a fixed store holds them
/// ([`Self::fire_rows`]): the runtime cuts fires to that many. A float
/// lane's rows (a kv-less lane: an encoder's, an embedding's, every row it
/// embeds) are no sequence readouts: they land through [`Self::landing`]
/// whatever the store holds, at most the fire's tokens, and are never cut
/// or refused against the store (a cacheless encoder attends over all its
/// rows in one fire). The term counts, for a fire of a rung, the rows it
/// may read out of either kind ([`Self::row_bytes`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadoutFacts {
    #[serde(default)]
    pub landing: ReadoutLanding,
    /// The rows a fixed readout store holds (XLA: its ring): no fire's
    /// sequence lanes read out more, and a sequence lane that alone reads
    /// more is refused at its bind. Float lanes' rows are not held here.
    /// `None`: no store of the engine's bounds them.
    #[serde(default)]
    pub store_rows: Option<u32>,
}

impl ReadoutFacts {
    /// The bytes a readout row takes beside the model's activation: 0 where
    /// the rows land inside the span already.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        match self.landing {
            ReadoutLanding::InSpan => 0,
            ReadoutLanding::Beside { row_bytes } => row_bytes,
        }
    }

    /// The most rows one fire reads out, beside the engine's own ceiling
    /// `ceiling` (`FireLimits::max_readouts`, 0: none stated): the two's
    /// least, 0 where neither bounds them.
    #[must_use]
    pub fn fire_rows(&self, ceiling: u32) -> u32 {
        match (ceiling, self.store_rows) {
            (0, None) => 0,
            (0, Some(rows)) => rows.max(1),
            (ceiling, None) => ceiling,
            (ceiling, Some(rows)) => ceiling.min(rows.max(1)),
        }
    }
}

/// An item's bytes and the slots of its role that hold them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Footprint {
    pub bytes: u64,
    pub slots: u32,
    /// The slots must be one run (`new-memory/STATE-ROLES.md` option D):
    /// `slots` adjacent slots of one chunk, refs `first .. first + slots`
    /// in order, so the engine addresses the item as one span from its
    /// first slot rather than through its ref table. What a ring whose cell
    /// is larger than a slot states. The runtime refuses an item that
    /// needs more slots than a chunk holds, and moves a run whole, into
    /// another run, when it parks or unparks it. `false`: any slots, the
    /// item addressed slot by slot.
    #[serde(default)]
    pub contiguous: bool,
}

/// One ring: `capacity + 1` cells of `cell_bytes` each (padded to the
/// engine's stride) and the head / tail words, in `footprint.slots`
/// `Channel` slots. Cells are packed whole into a slot where one fits, so a
/// slot count is not always `ceil(bytes / slot_bytes)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingFacts {
    /// The program's channel index.
    pub channel: u32,
    pub cell_bytes: u64,
    pub capacity: u32,
    pub footprint: Footprint,
}

/// Pool slots the runtime moved, as [`crate::Engine::rebind_slots`] tells
/// them: it parked them on another tier and copied them back into fresh
/// refs of the same shapes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rebind {
    /// An instance's tables (`Instance` slots, as
    /// [`InstanceBinding::instance_slots`]) and the rings it holds itself
    /// (those no registered channel covers, in [`ProgramFacts::rings`]
    /// order, as [`InstanceBinding::ring_slots`]).
    Instance {
        id: InstanceId,
        instance: Vec<u32>,
        rings: Vec<RingSlots>,
        /// Its adapter's `Adapter` slots, as
        /// [`InstanceBinding::adapter_slots`].
        #[serde(default)]
        adapter: Vec<u32>,
        /// Its scratch, as [`InstanceBinding::scratch`].
        #[serde(default)]
        scratch: ScratchPlace,
        /// Its resident state's slots, as [`InstanceBinding::resident`],
        /// holding the bytes they held before the move.
        #[serde(default)]
        resident: ScratchPlace,
        /// Its pair's, as [`InstanceBinding::pair`].
        #[serde(default)]
        pair: ScratchPlace,
        /// The taking of the pair's slots it names, as
        /// [`InstanceBinding::pair_generation`].
        #[serde(default)]
        pair_generation: u64,
    },
    /// A registered channel's ring (`Channel` slots, as
    /// [`crate::channel::ChannelRegistration::slots`]).
    Channel {
        id: crate::channel::ChannelId,
        slots: Vec<u32>,
    },
}

/// The `Channel` slots the runtime took for one of an instance's rings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingSlots {
    /// The program's channel index ([`RingFacts::channel`]).
    pub channel: u32,
    pub slots: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundInstance {
    pub id: InstanceId,
    pub program: ProgramId,
    pub geometry: GeometryClass,
    /// The engine's standing guest bytes once this instance is bound
    /// ([`crate::Engine::standing_after`]): its total over every live
    /// instance, a set shared by instances counted once. 0 where the
    /// engine keeps none.
    #[serde(default)]
    pub standing_guest_bytes: u64,
}
