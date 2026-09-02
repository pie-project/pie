//! Guest-program registration — what a caller states, and what the engine
//! answers. `ProgramRegistration`/`InstanceBinding` are what a caller states,
//! `BoundInstance` is what the engine answers, `BindExtents` is the
//! resolution a binding carries.

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
/// Stated by the caller rather than inferred: a wrong guess zero-fills
/// silently (e.g. a buffer carved for one row when the fire hands it four
/// leaves three rows of undetected zeroes). Default is every extent one.
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
    /// Written out rather than derived: a derived `BindExtents` would be
    /// every extent zero, and a zero extent carves a buffer no launch faults
    /// on.
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
