//! Frozen semantic contract for `pie:plex@0.6.0`.
//!
//! The active runtime remains on v0.5 until its state model, generated WIT
//! bindings, and adapters migrate in later phases.

mod manifest;
mod mechanics;
mod types;
mod validate;

pub use manifest::{
    ContractVersion, Manifest, ManifestValidationError, PolicyLimits, SchemaKind, SchemaRequirement,
};
pub use mechanics::{
    MechanicId, MechanicKind, STANDARD_MECHANICS, StandardMechanic, standard_mechanic,
};
pub use types::*;
pub use validate::{
    ContractValidationError, validate_admit_context, validate_admit_plan, validate_cache_context,
    validate_cache_plan, validate_feedback_context, validate_group_transition,
    validate_policy_error, validate_policy_state, validate_request_continuation,
    validate_request_transition, validate_route_context, validate_route_plan,
    validate_schedule_context, validate_schedule_plan, validate_state_update,
};

pub const PACKAGE_FORMAT_VERSION: u16 = 6;
