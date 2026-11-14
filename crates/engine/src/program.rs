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

/// What a fire holding lanes of a program bound at given extents needs
/// beside the model's own activation (REDESIGN 2.4, §6): the per-batch
/// tables and value scratch of the program's stage, sized by the binding's
/// [`BindExtents`] (the same program at 256 and 4096 rows differs 16x). A
/// fire of a rung of `r` tokens carries `l = min(r, FireLimits::max_lanes,
/// max_lanes)` lanes of it and needs `fixed + l × a_lane` bytes for each
/// (program, extents) bound that can be in it; the runtime folds the sum
/// into the rung's activation ([`crate::Engine::fire_facts`]). A program
/// registers after load, so `ChunkFacts` cannot state it.
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
}

impl FireFacts {
    /// The lanes of the program a fire carrying `lanes` lanes flies:
    /// capped by [`Self::max_lanes`] where it states one.
    #[must_use]
    pub fn lanes(&self, lanes: u64) -> u64 {
        if self.max_lanes == 0 {
            lanes
        } else {
            lanes.min(u64::from(self.max_lanes))
        }
    }

    /// The bytes a fire of `lanes` lanes needs for the program.
    #[must_use]
    pub fn bytes(&self, lanes: u64) -> u64 {
        self.fixed
            .saturating_add(self.lanes(lanes).saturating_mul(self.a_lane))
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
}
