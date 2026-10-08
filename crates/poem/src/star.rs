//! A model package in Starlark: the layout a deployment's weights take, the
//! forward its rows run and the formats its checkpoints are read in, each
//! evaluated with only the builtins its stage may use.

#![allow(clippy::too_many_arguments)]

mod bind;
mod formats;
mod forward;
mod generative;
mod layout;
mod manifest;
mod ops;
mod package;
mod run;
pub mod values;

pub use layout::with_env;
pub use manifest::{Manifest, Model, Published};
pub use package::{API, ATTRIBUTE, LIBRARY, Package, Stage};
pub use run::Deploy;
