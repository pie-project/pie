//! # CUDA kernel emitters
//!
//! The only producer of Pie's generated CUDA. Emission is a pure function of
//! the plan — no device-architecture inputs — so the same stage emits the same
//! bytes every time, and `compiler/tests/golden-cuda/` pins them.
//!
//! Those goldens started as a dump of an in-driver C++ emitter that no longer
//! exists, which is why they are formatted as a foreign dump and why
//! regenerating one is guarded. They are now the contract itself rather than a
//! transcript of one: nothing can re-derive them, so a diff is a decision to be
//! justified, not a comparison to be re-run.
//!
//! ## Modules
//!
//! * [`runtime`] — the embedded CUDA runtime template.
//! * [`validate`] — the region ABI checks the fused emitter gates on.
//! * [`singleton`] — the one-op-per-launch kernel.
//! * [`fused`] — the whole-region kernel.
//! * [`region_analysis`] — the per-region bind gates and intrinsic side-table
//!   analysis this backend's driver would otherwise re-derive.

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
pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 21;
