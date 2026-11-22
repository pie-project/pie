//! Cerebras kernels: the wafer program's compiler. It plans a model's trace onto the wafer, builds
//! the program (floor, links, kernels, weight bindings), links it with the SDK's toolchain and
//! cuts the checkpoint into the tiles it places, written as an [`compile::artifact::Artifact`]
//! the engine loads and runs. The CSL sources are [`sources`]; the hand-written kernels, by
//! family ([`attn`], [`conv`], [`linear`], [`ssm`]), register in [`kernel`]; the compiler is
//! [`compile`]. The engine names no model.

pub mod attn;
pub mod compile;
pub mod conv;
pub mod kernel;
pub mod linear;
pub mod sources;
pub mod ssm;
pub mod tuning;
