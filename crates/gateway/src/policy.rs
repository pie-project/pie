//! The cluster plane of PLEX, as a gateway takes part in it: its programs
//! run in the controller; the gateway tells them what happens to its
//! launches and applies what they decide.
//!
//! A process here is a launch, named by its turn's request id.

mod service;

pub use service::{By, Cluster, Started};
