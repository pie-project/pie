//! # CUDA kernel emitters
//!
//! A direct port of `driver/cuda/src/pipeline/generated/` (namespace
//! `pie_cuda_driver::pipeline::generated`). The C++ original is the oracle:
//! every function here emits the same bytes for the same input, and
//! `compiler/tests/golden-cuda/` is the checked-in proof (see
//! `compiler/tests/oracle/README.md`).
//!
//! Like the Metal port, the differences from the C++ are mechanical: the
//! `GeneratedKernelSource` out-struct becomes [`Result`], and the runtime
//! template is embedded with `include_str!` instead of being rebuilt from a
//! string literal on every call.
//!
//! ## Modules
//!
//! * [`runtime`] — the embedded CUDA runtime template.
//! * [`validate`] — the region ABI checks the fused emitter gates on.
//! * [`singleton`] — the one-op-per-launch kernel.
//! * [`fused`] — the whole-region kernel.

pub mod fused;
pub mod runtime;
pub mod singleton;
pub mod validate;

pub use fused::emit_fused_region;
pub use runtime::singleton_runtime_source;
pub use singleton::emit_singleton_region;
pub use validate::{second_party_region_supported, validate_generated_region};

/// `kCudaGeneratedEmitterVersion` — bumped whenever emitted CUDA changes, so
/// the driver's compile cache keys on it.
pub const CUDA_GENERATED_EMITTER_VERSION: u16 = 19;
