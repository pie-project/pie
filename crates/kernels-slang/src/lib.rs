//! One kernel tree for Vulkan and WebGPU, in Slang. A unit under `kernels/`
//! is one algorithm; its `// pie:instantiate` lines name its entrypoints, and
//! build.rs compiles each to SPIR-V and WGSL as the features ask. Buffers hold
//! bf16 two to a word (`prim/bf16.slang`) and scalars sit in a `PIE_PARAMS`
//! block (`prim/params.slang`), so one source reads the same on both.

pub mod attn;
pub mod elemwise;
pub mod encode;
pub mod error;
pub mod gather;
pub mod linear;
pub mod sampling;
pub mod sources;
pub mod tensor;

pub use encode::{ABSENT, Arg, ArgValue, Ctx, Encode, Fire, Geometry, Grid, Tier};
pub use error::Error;
pub use sources::{CENSUS, SPIRV, WGSL, key, spirv, wgsl};
pub use tensor::{Bank, KvPool, RaggedTensor, Tensor};
