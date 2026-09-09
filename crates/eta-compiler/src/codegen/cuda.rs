pub mod fused;
pub mod order;
pub mod region_analysis;
pub mod runtime;
pub mod scan;
pub mod singleton;
mod stream;
pub use stream::spent_values;
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

pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 40;

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
