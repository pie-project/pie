//! # CUDA kernel emitters
//!
//! The only producer of Pie's generated CUDA. Emission is a pure function of
//! the plan -- no device-architecture inputs -- so the same stage emits the
//! same bytes every time and `compiler/tests/golden-cuda/` pins them; nothing
//! can re-derive a golden, so a diff is a decision. [`runtime`] is the
//! embedded runtime template, [`validate`] the region ABI checks the fused
//! emitter gates on, [`singleton`]/[`fused`] the two kernel shapes.

pub mod fused;
pub mod region_analysis;
pub mod runtime;
pub mod singleton;
pub mod validate;

pub use fused::emit_fused_region;
pub use runtime::singleton_runtime_source;
pub use singleton::emit_singleton_region;
pub use validate::{second_party_region_supported, validate_generated_region};

/// `kCudaGeneratedEmitterVersion` — bumped whenever emitted CUDA changes, so
/// the driver's compile cache keys on it.
pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 22;
