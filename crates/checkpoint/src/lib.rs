//! Runtime-owned planner for turning a checkpoint into resident bytes.
//!
//! `plan = compile(source_facts, program, target)`. None of the three inputs
//! is a model's name: the engine states what it needs as a contract over the
//! checkpoint's byte space, this crate reads the checkpoint's own metadata,
//! and the target carries the numbers a device measured.
//!
//! This crate knows nothing of model semantics (no family names, no layer
//! structure); `contract::materialize` synthesizes what to produce from the
//! checkpoint alone. It also calls no GPU, links no device runtime, and
//! wants no toolkit — the whole crate is `dtype`, `half`, `ztensor`, `serde`
//! and `thiserror`. What is left of the device here is vocabulary:
//! [`executor::arena::ArenaBacking`] is the seam a consumer supplies, and
//! [`plan::passes::tile`]'s `CUDA_*` constants are the words a plan carries
//! to whoever launches them.

pub mod codec;
/// Releasing a source's bytes as the import reads them, and the ledger that
/// says which ranges it is allowed to. Outside [`executor`] because two of its
/// three read sites are `pie model import`'s own.
pub mod consume;
pub mod contract;
pub mod dump;
pub mod error;
pub mod executor;
pub mod extent;
pub mod file;
pub mod plan;
/// The `pie.serving/1` format layer: the definitions a serving artifact's
/// reader and its writer both spell their agreement in. Deliberately outside
/// [`file`], because it touches nothing.
pub mod serving;
mod term;

pub use term::spec_of_term;
#[cfg(feature = "testkit")]
pub mod testkit;
pub mod types;
pub mod verify;
