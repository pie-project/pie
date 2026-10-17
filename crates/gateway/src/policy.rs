//! The router plane of PLEX (`plex:policy@1.0.0`).
//!
//! A process here is a launch, named by its turn's request id.

mod manifest;
mod program;
mod service;

pub use manifest::{GateFail, Manifest, ManifestError};
pub use program::{Commit, Fault, LoadError, Plane, Program, Table};
pub use service::Router;

pub mod wit {
    pub use super::program::bindings::plex::policy::types::*;
}

pub const CAPABILITIES: &[&str] = &["router.hold", "placement.pd", "placement.migrate"];
