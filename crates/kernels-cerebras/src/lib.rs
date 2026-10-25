//! Cerebras kernels: op entries that emit device work.
//!
//! There is no compiler between an op graph and the wafer: an entry writes
//! the CSL calls of its op into a [`phase::Phase`] (the kernel library's
//! functions, by [`library::Kernel`] id, over the buffers it reads and
//! writes, and how they spread over PEs), through the engine's
//! [`phase::Sink`]. The engine owns where data lives and when it runs: it
//! places the operands, assembles a fire's phases into programs, compiles
//! and runs them.
//!
//! Entries follow the kernels-slang surface (module path, name, arguments) and
//! the numerics rules of `.wiki/xla/kernels.md`: compute in f32, round at the
//! write, refuse rather than approximate.

pub mod attn;
pub mod conv;
pub mod csl;
pub mod cx;
pub mod elemwise;
pub mod error;
pub mod gather;
pub mod library;
pub mod linear;
pub mod phase;
pub mod plan;
pub mod sampling;
pub mod ssm;
pub mod tensor;

pub use attn::{DecodePlan, PrefillPlan};
pub use cx::{Ctx, Cx, Emit, expect, shaped};
pub use error::Error;
pub use phase::{Buf, Sink};
pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};
