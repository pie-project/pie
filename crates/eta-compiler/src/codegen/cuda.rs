pub mod fused;
pub mod order;
pub mod region_analysis;
pub mod runtime;
pub mod sampler;
pub mod scan;
pub mod singleton;
mod stream;
pub use stream::spent_values;
pub mod validate;

pub use fused::{emit_fused_region, emit_fused_region_in};
pub use order::{emit_order_region, emit_order_region_in, is_order_region};
pub use runtime::{Dialect, runtime_source, singleton_runtime_source};
pub use scan::{emit_scan_region, emit_scan_region_in, is_scan_region};
pub use singleton::{emit_singleton_region, emit_singleton_region_in};
pub use validate::{second_party_region_supported, validate_generated_region};

use crate::codegen::error::EmitError;
use crate::plan::{CompiledStage, Region};
use alloc::string::String;

pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 50;

/// The HIP emitter shares the CUDA one's text and moves with it.
pub const HIP_GENERATED_EMITTER_VERSION: u16 = 10;

pub fn emit_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    emit_region_in(Dialect::Cuda, entry_name, stage, region)
}

pub fn emit_region_in(
    dialect: Dialect,
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if is_order_region(stage, region) {
        return emit_order_region_in(dialect, entry_name, stage, region);
    }
    if is_scan_region(stage, region) {
        return emit_scan_region_in(dialect, entry_name, stage, region);
    }
    if sampler::is_sampler_region(stage, region) {
        return sampler::emit_sampler_region_in(dialect, entry_name, stage, region);
    }
    emit_fused_region_in(dialect, entry_name, stage, region)
}
