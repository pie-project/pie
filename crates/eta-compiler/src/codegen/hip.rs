//! The HIP emitter: the shared SIMT text under hiprtc's dialect, where a
//! 64-lane wavefront holds two 32-lane chunks.

use super::simt::{self, Dialect};
use crate::codegen::error::EmitError;
use crate::plan::{CompiledStage, Region};
use alloc::string::String;

/// The HIP emitter shares the CUDA one's text and moves with it.
pub const HIP_GENERATED_EMITTER_VERSION: u16 = 18;

/// How a lane's blocks meet on CDNA: the exchange words at device scope
/// (each XCD's L2 written through and read past, never written back or
/// invalidated whole), a barrier that waits for the stores it publishes.
const HIP_LANE: &str = "
#define M1_LANE_PAUSE() __builtin_amdgcn_s_sleep(1)
#define M1_LANE_LOAD(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_PUT32(p, v) __hip_atomic_store((p), (m1_u32)(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_PUT64(p, v) __hip_atomic_store((p), (m1_u64)(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_GET32(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_GET64(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_RELEASE() asm volatile(\"s_waitcnt vmcnt(0)\" ::: \"memory\")
#define M1_LANE_ACQUIRE() asm volatile(\"\" ::: \"memory\")
#define M1_LANE_FADD(p, v) unsafeAtomicAdd((p), (v))
";

pub const DIALECT: Dialect = Dialect {
    chunk: include_str!("../../runtime/hip/ptir_m1_chunk.hip"),
    lane: HIP_LANE,
    cut_mark: "",
};

pub fn emit_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    simt::emit_region(DIALECT, entry_name, stage, region)
}
