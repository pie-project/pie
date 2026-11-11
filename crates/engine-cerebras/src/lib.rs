//! The Cerebras engine: the model forward runs as a resident wafer program on
//! a CS-3 (or the fabric simulator), driven from this process ([`wafer`]),
//! and everything else on the host: guest (eta) programs in the host
//! interpreter ([`guest`]), channels, the runtime's frames. The SDK runtime
//! binding is [`sdk`].
//!
//! The program is compiled by kernels-cerebras (the plan from the model's
//! trace, the build, the link) and carries its [`artifact`]: the host's map
//! of the wafer, every copy onto it, every PE's route words. No model is
//! named here.
//!
//! The program holds one sequence's KV cache. The engine keeps every lane's
//! cache as the tokens that made it (by slot, and by page for `copy_kv`):
//! before a lane's rows run the program's cache is brought to that lane,
//! kept as far as it agrees and the rest replayed, so rejected speculative
//! rows, forks, shared prefixes and several lanes a fire (run one after
//! another) all answer as a device cache would, slowly where lanes
//! alternate. Attention there is causal over a lane's own history only: a
//! lane whose mask says otherwise is refused, as is a program reading
//! attention scores.
//!
//! A load plans the model's trace, builds (or finds in the cache) the program and cuts (or
//! finds) the checkpoint's tiles for it ([`prepare`]); the program runs on the configured
//! system (`cmaddr`, else the simulator). [`Config`] is the whole configuration.
//!
//! - [`artifact`]: the program on a [`device::Device`], placed and driven;
//! - [`wafer`]: that model on a thread of its own;
//! - [`prepare`]: the plan, the program and the tiles of a load, cached;
//! - `backend`: PIE's `Engine` over it ([`CerebrasEngine`]).

pub mod artifact;
mod backend;
pub mod config;
pub mod device;
mod facts;
pub mod guest;
pub mod prepare;
pub mod sdk;
pub mod wafer;

use std::path::Path;

use checkpoint::contract::ModelContract;
use poem_ir::Trace;

pub use backend::CerebrasEngine;
pub use config::Config;

/// The runtime's import contract for a trace over a checkpoint.
pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

/// The engine the `[cerebras]` boot table describes: the model forward as a wafer program,
/// guests on the host.
pub fn open_engine(
    config_bytes: &[u8],
    contract_for: ContractFor,
) -> Result<Box<dyn engine::Engine>, String> {
    Ok(Box::new(CerebrasEngine::new(
        Config::parse(config_bytes)?,
        contract_for,
    )))
}

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
