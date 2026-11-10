//! A model package in Starlark: the layout a deployment's weights take, the
//! forward its rows run and the formats its checkpoints are read in, each
//! evaluated with only the builtins its stage may use.

#![allow(clippy::too_many_arguments)]

mod adopt;
mod bind;
mod formats;
mod forward;
mod generative;
mod layout;
mod manifest;
mod ops;
mod ops_by_hand;
mod package;
mod paths;
mod run;
mod values;

pub use adopt::{ADOPT, Adoption, Refusal};
pub use manifest::{Manifest, Model, Published, Template, Tokenizer, dtype_of, precision_of};
pub use package::Package;
pub use paths::renamed;
pub use run::{Deploy, ImageSpec, Recognized};
pub use values::{precision_word, word};
