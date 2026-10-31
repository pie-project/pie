//! The spatial engine: the model forward runs as a resident wafer program on
//! a CS-3 (or the fabric simulator), driven from this process ([`wafer`]),
//! and everything else stays as the cerebras engine has it: guest (eta)
//! programs in the host interpreter (or the wafer's eta VM), channels, the
//! runtime's frames.
//!
//! The engine keeps every lane's cache as the tokens that made it (by slot,
//! and by page for `copy_kv`). A one-sequence program holds one sequence's
//! KV cache: before a lane's rows run the engine brings it to that lane,
//! kept as far as it agrees and the rest replayed, so rejected speculative
//! rows, forks, shared prefixes and several lanes a fire (run one after
//! another) all answer as a device cache would, slowly where lanes
//! alternate. Attention there is causal over a lane's own history only: a
//! lane whose mask says otherwise is refused, as is a program reading
//! attention scores.
//!
//! A batched program owns the KV cache as PIE's pages: it runs a fire's
//! lanes together, each with its KV write slots, RoPE positions, pages and
//! mask, and needs no replay: `copy_kv` is a page copy, a truncate a smaller
//! slot. A lane whose pages the program never got (a fire that named another
//! lane's pages) is re-prefilled into them from its own tokens first.
//!
//! Selected by `[cerebras.spatial]` in the engine config. The program runs
//! on the `[cerebras]` platform (`cmaddr`, else the simulator), its weights
//! read through PIE's checkpoint path (the load's contract over its
//! checkpoint, by the model's parameter names); `ckpt` names a Hugging Face
//! directory to read instead (`model.safetensors`, its shards, or
//! `weights.npz`, with `config.json`). The program is one of:
//!
//! - `[cerebras.spatial.serve]`: PIE builds the model's planned program
//!   ([`build`], compiled once, then cached);
//! - `program = "dir"`: an already compiled output directory.
//!
//! The code is layered (`PLAN.md` beside this module: the architecture, and
//! how to add a design, a component or a validated shape):
//!
//! - [`model`]: [`model::ModelSpec`] read off the trace;
//! - [`plan`], [`shapes`], [`build`]: the plan, the hardware-validated
//!   shapes, the build of a [`program::ProgramSpec`];
//! - [`program`], [`component`]: the typed program and its components'
//!   host halves and ABI ([`serve2d`], [`serve8b`], [`serve8b_region`]: the
//!   designs' tile layouts and route rules);
//! - [`device`], [`wafer`]: the [`device::Device`] the host drives, and the
//!   driver;
//! - `engine`: PIE's `Engine` over it ([`Spatial`]).

pub mod build;
pub mod ckpt;
pub mod codegen;
pub mod component;
mod config;
pub mod device;
mod engine;
#[cfg(test)]
mod golden;
pub mod guest4;
pub mod manifest;
pub mod model;
pub mod paged;
pub mod plan;
pub mod program;
pub mod serve2d;
pub mod serve8b;
pub mod serve8b_region;
pub mod shapes;
pub mod wafer;

pub use config::SpatialBoot;
pub use engine::Spatial;
