use serde::{Deserialize, Serialize};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::EmittedKernel;
use eta_ir::registry::GeometryClass;

use crate::channel::ChannelSeed;

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
    /// Where this instance's lanes' scratch lies
    /// ([`GuestState::per_instance`]).
    #[serde(default)]
    pub scratch: ScratchPlace,
    /// Where its pair's lies ([`GuestState::per_pair`]): the same on every
    /// instance of the pair, taken at the pair's first bind and given back
    /// at its last close.
    #[serde(default)]
    pub pair: ScratchPlace,
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

/// What a guest program bound at given extents holds in the pool, as state
/// (REDESIGN §6): the scratch each bound instance keeps for its lanes, and
/// what the (program, extents) pair keeps once while any instance of it is
/// bound ([`crate::Engine::guest_state`]). Each lies in the `Instance` role,
/// taken at the bind and held while bound ([`InstanceBinding::scratch`],
/// [`InstanceBinding::pair`], [`ScratchPlace`]); no fire takes anything of
/// the activation span for a program. Their contents are dead between
/// fires: a park gives them back and an unpark takes fresh room, nothing
/// copied, the engine told where ([`Rebind::Instance`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestState {
    /// Bytes each bound instance holds for its lanes' scratch.
    #[serde(default)]
    pub per_instance: u64,
    /// Bytes the pair holds once while any instance of it is bound.
    #[serde(default)]
    pub per_pair: u64,
}

/// Where a guest's scratch lies in the pool ([`GuestState`]): bytes that fit
/// in a chunk's `Instance` slots (its slots times their bytes) as one run of
/// adjacent slots in one chunk, else whole chunks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScratchPlace {
    /// None held.
    #[default]
    None,
    /// `slots` adjacent `Instance` slots from slot ref `first`, in its chunk.
    Run { first: u32, slots: u32 },
    /// Whole chunks, by logical chunk id.
    Chunks(Vec<u32>),
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
/// row), so they are no (program, extents) pair's [`GuestState`].
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
        /// Its pair's, as [`InstanceBinding::pair`].
        #[serde(default)]
        pair: ScratchPlace,
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
