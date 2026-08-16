//! `llama_like` -- the shape a dozen generations share: llama 2/3,
//! mistral, qwen 2/3, phi-3 and olmo 2/3 are one forward pass parameterized
//! by facts, so it lives under `shared/` rather than in any one generation.
//! [`forward`] is that pass, [`spec`] the numbers a checkpoint has (ungated:
//! a catalog row is written in them), [`project`] its three projections.

pub mod forward;

pub mod spec;

pub mod project;

#[cfg(feature = "contract")]
pub mod contract;

/// The lineage's tensor names, in pie's vocabulary and in llama.cpp's.
#[cfg(feature = "contract")]
pub mod import;
