//! nemotron_h's shape, re-exported.
//!
//! The SEMANTIC facts moved to [`super::super::spec`], ungated, because
//! a catalog row is written in these words under every aspect — `chat`
//! asks which template speaks for a checkpoint, `contract` asks how to
//! author it, `forward` asks what to trace — and a shape that only
//! existed behind `#[cfg(feature = "forward")]` could be named by
//! exactly one of the three.
//!
//! The re-export is here so the traced text below and every fixture that
//! spells `forward::facts::NemotronHFacts` keep compiling unchanged.
//!
//! Nothing stayed: this family has no per-backend binding facts of its
//! own, unlike qwen3_5's `Qwen35CudaFacts`. When it grows one — a
//! workspace ceiling, an env gate — this is where it goes, because those
//! name kernels and kernels belong to the aspect that has them.

pub use super::super::spec::{
    NemotronAttnFacts, NemotronHFacts, NemotronLayerKind, NemotronMambaFacts, NemotronMoeFacts,
};
