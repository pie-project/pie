//! The cluster plane of PLEX (`plex:policy@1.0.0`): where programs and their
//! state live.
//!
//! A process here is a launch, named by its turn's request id.

mod manifest;
mod program;
mod service;

pub use manifest::{Manifest, ManifestError};
pub use program::{Commit, Fault, LoadError, Plane, Program, StateCommit, Table};
pub use service::Cluster;

pub mod wit {
    pub use super::program::bindings::plex::policy::types::*;
}

pub const CAPABILITIES: &[&str] = &[
    "cluster.hold",
    "cluster.scale",
    "placement.executors",
    "placement.migrate",
    "state.replicate",
    "state.retain",
    "state.transfer-shares",
    "state.accept",
];
