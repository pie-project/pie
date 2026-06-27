//! # `sampling-edsl` — runtime builder EDSL for the Pie Sampling IR (Lane L5)
//!
//! Author sampling programs as typed array graphs that lower to the canonical
//! `pie-sampling-ir` (PSIR v4) bytecode. The authoring surface is the runtime,
//! length-erased [`DynValue`] handle (inferlets learn the vocab at run time);
//! ready-made programs live in [`program`] and the legacy-`Sampler` sugar in
//! [`sugar`].
//!
//! ```
//! use sampling_edsl::prelude::*;
//!
//! let (built, keys) = sampling_edsl::program::grammar(32_000).unwrap();
//! let lowered = built.lower();
//! assert_eq!(lowered.outputs, vec![OutputKind::Token]);
//! let _ = keys.mask; // bind a per-step mask tensor under this key each fire
//! ```
//!
//! ## Layering
//! - [`DynValue`](dynamic::DynValue) — runtime SSA handle (shape + dtype as data).
//! - [`Graph`](builder::Graph) — the builder: declare inputs, compose, mark
//!   outputs, [`build`](builder::Graph::build) → [`Built`](builder::Built), then
//!   [`lower`](builder::Built::lower) → [`LoweredProgram`](builder::LoweredProgram).
//! - [`program`] / [`sugar`] — mirostat, grammar, spec-verify (greedy + lossless),
//!   and the `Sampler`→IR lowering. RNG is Model B (`Op::Rng{stream}`, ambient seed).
//!
//! The [`ir`] module re-exports the canonical `pie-sampling-ir` types.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod builder;
pub mod dynamic;
pub mod ir;
pub mod program;
pub mod standard;
pub mod sugar;

pub use builder::{BuildError, Built, Graph, HostInputDecl, LoweredProgram, OutputKind};
pub use dynamic::{DynValue, dselect};
pub use standard::{StandardSampler, StdParamKeys, build_standard};
pub use sugar::{SamplerSpec, build_sampler, lower_sampler};

// IR enums an author touches at the surface.
pub use ir::{DType, Readiness};

/// Glob-import surface for inferlet authors.
pub mod prelude {
    pub use crate::builder::{Graph, OutputKind};
    pub use crate::dynamic::{DynValue, dselect};
    pub use crate::ir::{DType, Readiness};
    pub use crate::program::{
        grammar, grammar_sampled, mirostat, spec_verify_greedy, spec_verify_lossless,
    };
    pub use crate::sugar::{SamplerSpec, build_sampler, lower_sampler};
}
