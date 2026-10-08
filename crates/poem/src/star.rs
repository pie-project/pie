//! A model package in Starlark: the layout a deployment's weights take, the
//! forward its rows run and the formats its checkpoints are read in, each
//! evaluated with only the builtins its stage may use.

#![allow(clippy::too_many_arguments)]

mod bind;
mod formats;
mod forward;
mod layout;
mod manifest;
mod ops;
mod package;
mod run;
pub mod values;

pub use manifest::{Manifest, Model};
pub use package::{API, ATTRIBUTE, Package, Stage};
pub use run::Deploy;
