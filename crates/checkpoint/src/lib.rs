//! Runtime-owned planner for turning a checkpoint into resident bytes:
//! `plan = compile(source_facts, program, target)`. Knows no model
//! semantics and links no device runtime; the device is present only as
//! vocabulary ([`executor::arena::ArenaBacking`] is the seam a consumer
//! supplies, [`plan::passes::tile`]'s `CUDA_*` constants the words a plan
//! carries).

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
/// The `pie.serving/1` format layer: what a serving artifact's reader and
/// writer agree on. Touches no file.
pub mod serving;
mod term;

pub use term::{spec_of_term, term_of};
#[cfg(feature = "testkit")]
pub mod testkit;
pub mod types;
pub mod verify;
