//! glm5's per-backend binding facts — of which this generation has
//! none, yet.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a row is written in it, and
//! a row has to exist under every aspect — `chat`, `contract` and
//! `forward` all read it, and only one of those compiles the tracer).
//!
//! What would live here is what a deployment BOUND rather than what the
//! model IS — kimi-k2's `KimiCudaFacts` is the shape of it: one fused
//! latent GEMM instead of two, a YaRN rope the config asked for. glm5's
//! text reads neither, so this file is the re-export that keeps
//! `forward`'s spelling of the facts working unchanged.

/// The shape, re-exported so a declaration reaches its facts and the
/// words they are stated in from one place.
pub use super::super::spec::{Glm5DsaFacts, Glm5Facts, Glm5MlaFacts, Glm5MoeFacts};
