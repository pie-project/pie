//! The cluster plane of PLEX (`plex:policy@1.0.0`): where programs and their
//! state live. Its programs run here, once for the cluster; the gateways
//! send what happens to their processes and apply what the programs decide
//! about them.

mod host;
mod manifest;
mod program;

pub use host::Host;
pub use manifest::{Manifest, ManifestError};
pub use program::{Commit, Fault, LoadError, Plane, Program, StateCommit, Table};

pub mod wit {
    pub use super::program::bindings::plex::policy::types::*;
}

pub const CAPABILITIES: &[&str] = &[
    "cluster.hold",
    "cluster.scale",
    "kv.persist",
    "placement.executors",
    "placement.migrate",
    "state.replicate",
    "state.retain",
    "state.transfer-shares",
    "state.accept",
];
