//! Cerebras kernels: op entries that emit CSL.
//!
//! There is no compiler between an op graph and the wafer, so this crate is
//! both the kernel library and the placement layer: a fire is traced into one
//! [`program::Program`] (a PE rectangle's CSL source plus the host manifest
//! that says which exported symbol holds which tensor), each entry appends a
//! phase to it, and the engine compiles and runs the whole program.
//!
//! Entries follow the kernels-wgpu surface (module path, name, arguments) and
//! the numerics rules of `.wiki/xla/kernels.md`: compute in f32, round at the
//! write, refuse rather than approximate.

pub mod attn;
pub mod csl;
pub mod cx;
pub mod elemwise;
pub mod error;
pub mod layout;
pub mod library;
pub mod linear;
pub mod program;
pub mod tensor;

pub use attn::{DecodePlan, PrefillPlan};
pub use cx::{Ctx, Cx, Emit, Env, View, expect, shaped};
pub use error::Error;
pub use program::{Buf, Manifest, Placement, Program, Symbol};
pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};
