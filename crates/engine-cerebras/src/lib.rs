//! The Cerebras engine: the model forward runs as a resident wafer program on a CS-3 (or the
//! fabric simulator), driven from this process ([`wafer`]); guest programs, channels and frames
//! run on the host ([`guest`]). The program holds one sequence's cache; the engine keeps every
//! lane's as tokens and replays into it as lanes alternate. A load plans the trace and builds
//! and cuts what it needs ([`prepare`]); [`Config`] is the whole configuration.

pub mod artifact;
mod backend;
pub mod chip;
pub mod config;
pub mod device;
mod facts;
pub mod guest;
pub mod prepare;
pub mod reading;
pub mod sdk;
pub mod units;
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

/// The fusions this backend implements: none.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
