mod capability;
pub use crate::capability::Capability;

pub mod module;
pub use crate::module::{MODULES, code, embedded};

pub mod runtime;

pub mod plane;
pub mod points;
pub mod views;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod ssm;

#[must_use]
pub fn entrypoints() -> Vec<String> {
    module::CENSUS.iter().map(|n| (*n).to_owned()).collect()
}
