//! CUDA kernel emitters: the only producer of Pie's generated CUDA. Emission
//! is a pure function of the plan, so the same stage emits the same bytes
//! every time and `compiler/tests/golden-cuda/` pins them. [`runtime`] is
//! the embedded runtime template, [`validate`] the region ABI checks,
//! [`singleton`]/[`fused`]/[`order`]/[`scan`] the kernel shapes.

pub mod fused;
pub mod order;
pub mod region_analysis;
pub mod runtime;
pub mod scan;
pub mod singleton;
pub mod validate;

pub use fused::emit_fused_region;
pub use order::{emit_order_region, is_order_region};
pub use runtime::singleton_runtime_source;
pub use scan::{emit_scan_region, is_scan_region};
pub use singleton::emit_singleton_region;
pub use validate::{second_party_region_supported, validate_generated_region};

use crate::codegen::error::EmitError;
use crate::plan::{CompiledStage, Region};
use alloc::string::String;

/// `kCudaGeneratedEmitterVersion` — bumped whenever emitted CUDA changes, so
/// the engine's compile cache keys on it. Also keys the negative tier
/// (cached `Deterministic` compile failures), so a bump is how a stale
/// refusal is forgotten.
pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 28;

/// The kernel this backend compiles for one region of a stage: the single
/// place that decides which emitter a region goes through.
///
/// A `RegionKind::Library` region is not automatically refused: the plan
/// recognising a library dataflow doesn't mean this backend has a kernel for
/// it. `top_k`, `sort_desc` and `cumsum`/`cumprod` have one; `matmul` does
/// not and falls through to [`emit_fused_region`]. A `SecondParty` region
/// falls through too and is refused there, since it names a kernel the
/// shell launches itself.
///
/// # Errors
///
/// Whatever the chosen emitter refuses; see [`EmitError`].
pub fn emit_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if is_order_region(stage, region) {
        return emit_order_region(entry_name, stage, region);
    }
    if is_scan_region(stage, region) {
        return emit_scan_region(entry_name, stage, region);
    }
    emit_fused_region(entry_name, stage, region)
}
