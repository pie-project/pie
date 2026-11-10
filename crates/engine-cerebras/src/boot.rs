//! The `[cerebras]` boot table: the engine it opens.

use crate::api::ContractFor;
use crate::spatial::{Config, Spatial};

/// The engine the boot config describes: the model forward as a wafer program, guests on the
/// host.
pub fn open_engine(
    config_bytes: &[u8],
    contract_for: ContractFor,
) -> Result<Box<dyn engine::Engine>, String> {
    Ok(Box::new(Spatial::new(
        Config::parse(config_bytes)?,
        contract_for,
    )))
}
