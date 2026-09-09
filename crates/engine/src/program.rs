use serde::{Deserialize, Serialize};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::EmittedKernel;
use eta_compiler::plan::SymbolicExtent;
use eta_ir::registry::GeometryClass;

use crate::channel::ChannelSeed;

pub type ProgramId = u64;

pub type InstanceId = u64;

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
    pub kv_len: u32,
    pub page_count: u32,
    pub row_count: u32,
    pub token_count: u32,
    pub sampled_rows: u32,
    pub query_len: u32,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InstanceBinding {
    pub program: ProgramId,
    pub channels: Vec<crate::channel::ChannelId>,
    pub seeds: Vec<ChannelSeed>,
    pub geometry: GeometryClass,
    pub extents: BindExtents,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundInstance {
    pub id: InstanceId,
    pub program: ProgramId,
    pub geometry: GeometryClass,
}
