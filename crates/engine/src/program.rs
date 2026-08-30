//! Guest-program registration — what a caller states, and what the engine
//! answers.
//!
//! # What is here, and what moved
//!
//! The [`LaunchPackage`] and its whole lineage — the value table, the channels
//! and ports, the per-stage op DAGs, the per-stage plans, the emitted kernels,
//! the region analysis — used to be declared in this file. They are the
//! **compiler's output artifact**, and they live with the compiler now
//! ([`eta_compiler::codegen::launch`], plus [`eta_compiler::codegen::program`]
//! and [`eta_compiler::codegen::cuda::region_analysis`] for the two halves
//! that belong beside the walk and the analysis that fill them). The
//! `LaunchPackage` header carries the reasoning that used to be this one's,
//! including what "purify" meant and the `Direction::of` / `Direction::from_wire`
//! bug that is the argument for it; if you came here for that, follow the link.
//!
//! That move deleted an inverted edge. `eta-compiler` depended on this crate
//! solely to describe its own output, and five of the types it borrowed back —
//! `LibraryOp`, `RegionKind`, `KernelKind`, `EmittedKernel`, `RegionAnalysis` —
//! were second declarations of types it already had, joined by `match`es and
//! struct copies whose only job was to cross the crate line. The edge runs the
//! other way now, which is the direction a contract and a producer actually
//! stand in.
//!
//! # What stayed
//!
//! The four nouns below are **trait method arguments**, not compiler output:
//! [`ProgramRegistration`] and [`InstanceBinding`] are what a caller states to
//! [`Engine::register_program`](crate::engine::Engine::register_program) and
//! [`Engine::bind_instance`](crate::engine::Engine::bind_instance);
//! [`BoundInstance`] is what the engine answers; [`BindExtents`] is the
//! resolution a binding carries. Nothing produces them but the caller, so
//! nothing else can own them.

use serde::{Deserialize, Serialize};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::EmittedKernel;
use eta_compiler::plan::SymbolicExtent;
use eta_ir::registry::GeometryClass;

use crate::channel::ChannelSeed;

/// A registered program's id, minted by the engine.
pub type ProgramId = u64;

/// A bound instance's id, minted by the engine.
pub type InstanceId = u64;

/// Everything a program registration states.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramRegistration {
    /// The program's identity — FNV-1a over its canonical container bytes.
    pub program_hash: u64,
    /// The kernels the compiler emitted.
    pub emitted_kernels: Vec<EmittedKernel>,
    /// Which emitter version produced them. An engine's compiled-artifact cache
    /// keys on it.
    pub emitter_version: u32,
    /// What the backend's region analysis found.
    pub region_analysis: Vec<RegionAnalysis>,
    /// The package to execute.
    pub launch: LaunchPackage,
    /// The canonical container bytes, for an engine that runs the reference
    /// interpreter beside the device and diffs the two.
    pub reference_ptir: Vec<u8>,
}

/// What a bound instance's symbolic value shapes resolve against.
///
/// **A GUESS ZERO-FILLS SILENTLY, SO IT IS STATED** (Build log 15). A stage
/// plan's value types are written in
/// [`Dimension::Symbolic`](eta_compiler::plan::Dimension::Symbolic) over the
/// seven [`SymbolicExtent`]s; an engine carves each stage's fire-path buffers at
/// BIND time, from these numbers, and a buffer carved for one row when the fire
/// hands it four leaves three rows of zeroes that no launch faults on. So the
/// caller states them, and the one that matters at a model fire's boundary is
/// [`BindExtents::sampled_rows`] — how many readout rows the epilogue reads,
/// which is the fire's [`Readout`](crate::fire::Readout) and nothing the
/// engine can infer from the package.
///
/// [`BindExtents::default`] is every extent ONE, which is what a program that
/// resolves entirely from static dims reads (it never reads these at all) and
/// what a `Readout::Last` lane hands an epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindExtents {
    /// The request's readable KV extent.
    pub kv_len: u32,
    /// How many pages its page list holds.
    pub page_count: u32,
    /// How many rows the value has.
    pub row_count: u32,
    /// How many tokens the fire carries.
    pub token_count: u32,
    /// How many rows the epilogue reads out.
    pub sampled_rows: u32,
    /// The query length.
    pub query_len: u32,
    /// The key length.
    pub key_len: u32,
}

impl Default for BindExtents {
    fn default() -> BindExtents {
        BindExtents {
            kv_len: 1,
            page_count: 1,
            row_count: 1,
            token_count: 1,
            sampled_rows: 1,
            query_len: 1,
            key_len: 1,
        }
    }
}

impl BindExtents {
    /// What `role` resolves to.
    #[must_use]
    pub const fn get(&self, role: SymbolicExtent) -> u32 {
        match role {
            SymbolicExtent::KvLen => self.kv_len,
            SymbolicExtent::PageCount => self.page_count,
            SymbolicExtent::RowCount => self.row_count,
            SymbolicExtent::TokenCount => self.token_count,
            SymbolicExtent::SampledRows => self.sampled_rows,
            SymbolicExtent::QueryLen => self.query_len,
            SymbolicExtent::KeyLen => self.key_len,
        }
    }
}

/// Everything an instance binding states.
///
/// Was `InstanceBindingPlan`, which carried three more fields —
/// `driver_id: usize`, `pacing_wait_id: u64` and `requested_instance_id` — that
/// were the runtime's bookkeeping travelling through the engine so it could
/// come back unchanged. The engine mints the id; the runtime keeps its own
/// tables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceBinding {
    /// Which program to instantiate.
    pub program: ProgramId,
    /// The channels this instance binds, in the package's declaration order.
    pub channels: Vec<crate::channel::ChannelId>,
    /// The values its seeded channels start holding.
    pub seeds: Vec<ChannelSeed>,
    /// How much of the fire geometry this instance's descriptor resolves on
    /// the device.
    pub geometry: GeometryClass,
    /// What this instance's symbolic value shapes resolve against. See
    /// [`BindExtents`].
    pub extents: BindExtents,
}

impl Default for InstanceBinding {
    /// A binding of nothing, at every extent one — the shape a program with
    /// no symbolic axis is bound in. Written out rather than derived because
    /// a derived [`BindExtents`] would be every extent ZERO, and a zero
    /// extent carves a zero-row buffer that no launch faults on.
    fn default() -> InstanceBinding {
        InstanceBinding {
            program: 0,
            channels: Vec::new(),
            seeds: Vec::new(),
            geometry: GeometryClass::default(),
            extents: BindExtents::default(),
        }
    }
}

/// A bound instance, as the engine answers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundInstance {
    /// The instance's id.
    pub id: InstanceId,
    /// The program it instantiates.
    pub program: ProgramId,
    /// The geometry class it was bound in — the engine's acknowledgement,
    /// which a caller compares against what it asked for.
    pub geometry: GeometryClass,
}
