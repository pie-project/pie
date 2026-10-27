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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundInstance {
    pub id: InstanceId,
    pub program: ProgramId,
    pub geometry: GeometryClass,
}
