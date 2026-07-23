//! Deliberately small JSON contract for the PLEX policy proof of concept.
//!
//! The Wasm ABI carries explicit invocation/output records whose fields are
//! JSON strings. This crate defines only the package manifest and host-side
//! validation/fill rules. Request state, candidates, and context remain
//! ordinary [`serde_json::Value`] dictionaries.

#![forbid(unsafe_code)]

mod manifest;
mod operation;
pub mod v0_6;

pub type Document = serde_json::Value;

pub use manifest::{ContractVersion, Manifest, ManifestValidationError, PolicyLimits};
pub use operation::{
    AdmissionDecision, DecisionValidationError, Operation, SelectedEviction, SelectedService,
    rank_route, select_evictions, select_schedule, validate_admit, validate_request_scope,
};
