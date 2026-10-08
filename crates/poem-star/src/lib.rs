//! A model package in Starlark: the layout a deployment's weights take, the
//! forward its rows run and the formats its checkpoints are read in, each
//! evaluated with only the builtins its stage may use.

#![allow(clippy::too_many_arguments)]

mod formats;
mod forward;
mod layout;
mod package;
mod run;
pub mod values;

pub use package::{Package, Stage};
pub use run::Deploy;
