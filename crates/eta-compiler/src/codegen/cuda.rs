//! # CUDA kernel emitters
//!
//! The only producer of Pie's generated CUDA. Emission is a pure function of
//! the plan -- no device-architecture inputs -- so the same stage emits the
//! same bytes every time and `compiler/tests/golden-cuda/` pins them; nothing
//! can re-derive a golden, so a diff is a decision. [`runtime`] is the
//! embedded runtime template, [`validate`] the region ABI checks the fused
//! emitter gates on, [`singleton`]/[`fused`]/[`order`]/[`scan`] the kernel
//! shapes.

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
/// the engine's compile cache keys on it.
///
/// It keys the NEGATIVE tier too (`engine::cache_identity` folds it into the
/// program key `engine_cuda::program::Cache` remembers a `Deterministic`
/// failure under), which is what makes a bump the way a *refusal* is
/// forgotten. Without one, a shell that had already answered "this guest
/// program does not compile here" for a ranking or scanning program would keep
/// answering it from memory for the life of the process.
pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 25;

/// The kernel this backend compiles for one region of a stage.
///
/// **THE ONE DOOR.** `emit_program` writes exactly one `KernelKind::Fused`
/// entry per region and `engine_cuda::program::compile`'s `build_stage` reads
/// exactly that slot for every region it does not skip, so "which emitter does
/// this region go through" is a question with a single answer and it is
/// answered here. Metal makes the same choice in `codegen::program` across
/// three kernel families; CUDA has one family, so the choice belongs beside
/// the emitters rather than in the walk above them — and a test asking whether
/// a planned region reaches the engine as something launchable
/// (`cuda_every_region_runs`) asks this function, not the three below it.
///
/// A `RegionKind::Library` region is NOT automatically refused: the plan
/// saying it recognised a library dataflow is not the same as this backend
/// having a kernel for it. `top_k`, `sort_desc` and `cumsum`/`cumprod` have
/// one; `matmul` does not, and falls through to [`emit_fused_region`], whose
/// validator names the boundary op it found. A `SecondParty` region falls
/// through too and is refused there — correctly, because it is a NAME the
/// shell launches itself rather than a body anything here could write, and
/// `build_stage` skips it before it ever reads this slot.
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
