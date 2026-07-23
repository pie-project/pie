//! Canonical PLEX policy contracts.

#![forbid(unsafe_code)]

mod manifest;
mod operation;
pub mod v0_6;

pub type Document = serde_json::Value;

pub mod v0_5 {
    pub use crate::manifest::{ContractVersion, Manifest, ManifestValidationError, PolicyLimits};
    pub use crate::operation::{
        AdmissionDecision, DecisionValidationError, Operation, SelectedEviction, SelectedService,
        rank_route, select_evictions, select_schedule, validate_admit, validate_request_scope,
    };
}

pub use v0_6::*;
