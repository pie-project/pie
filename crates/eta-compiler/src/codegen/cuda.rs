//! The CUDA emitter: the shared SIMT text under NVRTC's dialect, where a
//! warp is the 32-lane chunk.

pub use super::simt::{
    emitted_order, fused, is_matmul_region, is_order_region, is_scan_region, matmul, order,
    region_analysis, runtime, sampler, scan, second_party_region_supported, singleton,
    spent_values, validate, validate_generated_region,
};

use super::simt::{self, Dialect};
use crate::codegen::error::EmitError;
use crate::plan::{CompiledStage, Region};
use alloc::string::String;

pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 66;

/// The line a region carries where it holds the exact cut's state, which on
/// CUDA sits in [`CUT_STATE_BYTES`] of the launch's dynamic shared memory
/// (NVRTC's static shared memory stops at 48 KB).
pub const CUT_STATE_MARK: &str = "// m1: the exact cut's state\n";
/// The chunk header's `M1_CUT_STATE_BYTES`.
pub const CUT_STATE_BYTES: u32 = 34_816;

/// How a lane's blocks meet on a CUDA device: L2-coherent words behind the
/// device-scope fences.
const CUDA_LANE: &str = "
#define M1_LANE_PAUSE() __nanosleep(32)
#define M1_LANE_LOAD(p) (*reinterpret_cast<volatile m1_u32*>(p))
#define M1_LANE_PUT32(p, v) (*reinterpret_cast<volatile m1_u32*>(p) = (m1_u32)(v))
#define M1_LANE_PUT64(p, v) (*reinterpret_cast<volatile m1_u64*>(p) = (m1_u64)(v))
#define M1_LANE_GET32(p) (*reinterpret_cast<volatile m1_u32*>(p))
#define M1_LANE_GET64(p) (*reinterpret_cast<volatile m1_u64*>(p))
#define M1_LANE_RELEASE() __threadfence()
#define M1_LANE_ACQUIRE() __threadfence()
#define M1_LANE_FADD(p, v) atomicAdd((p), (v))
";

pub const DIALECT: Dialect = Dialect {
    chunk: include_str!("../../runtime/cuda/ptir_m1_chunk.cuh"),
    lane: CUDA_LANE,
    cut_mark: CUT_STATE_MARK,
};

pub fn emit_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    simt::emit_region(DIALECT, entry_name, stage, region)
}

pub fn emit_fused_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    simt::emit_fused_region(DIALECT, entry_name, stage, region)
}

pub fn emit_order_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    simt::emit_order_region(DIALECT, entry_name, stage, region)
}

pub fn emit_scan_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    simt::emit_scan_region(DIALECT, entry_name, stage, region)
}

pub fn emit_singleton_region(entry_name: &str, op_tag: u8) -> Result<String, EmitError> {
    simt::emit_singleton_region(DIALECT, entry_name, op_tag)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_chunk_header_gives_the_cut_state_what_the_launcher_does() {
        let line = alloc::format!("#define M1_CUT_STATE_BYTES {}u", super::CUT_STATE_BYTES);
        assert!(super::DIALECT.chunk.contains(&line), "{line}");
    }
}
