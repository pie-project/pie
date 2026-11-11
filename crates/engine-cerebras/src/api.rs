//! What the engine shares with the runtime: how it maps a checkpoint onto a trace, and how its
//! refusals read as engine errors.

use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::error::Error;
use poem_ir::Trace;

/// The runtime's import contract for a trace over a checkpoint.
pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

pub(crate) fn refusal(refusal: crate::program::Refusal) -> Error {
    match refusal {
        crate::program::Refusal::Closed { what, id } => Error::Closed { what, id },
        crate::program::Refusal::Program(why) => Error::Program(why),
    }
}

pub(crate) fn port_error(error: engine::ports::PortError) -> Error {
    Error::Program(format!("`program::ports` refused: {}", error.0))
}
