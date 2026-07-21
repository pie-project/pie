//! Engine-neutral contract for the PLEX policy kernel.
//!
//! This crate contains only policy vocabulary and validation. It deliberately
//! does not depend on Pie gateway, engine, scheduler, store, worker, inferlet,
//! or Wasmtime implementation types.
//!
//! # Compatibility
//!
//! PLEX `0.x` contracts require an exact [`ContractVersion`] match. WIT records,
//! variants, and exported operation signatures are closed: changing their shape
//! requires a new contract version and world. Open vocabularies instead use
//! versioned [`Symbol`] names resolved at attachment, so hosts can add fields,
//! events, maps, and capabilities without changing the component type. Missing
//! required symbols reject attachment; optional symbols resolve explicitly to
//! absence and are never approximated.

#![forbid(unsafe_code)]

pub mod capability;
pub mod ids;
pub mod manifest;
pub mod map;
pub mod metadata;
pub mod operation;
pub mod record;
pub mod value;

pub use capability::{
    CapabilityDeclaration, DependencyRequirement, EventDeclaration, InvocationMode, Symbol,
    SymbolError,
};
pub use ids::{
    CapabilityHandle, DeliveryId, EventHandle, FactHandle, GenerationId, LogicalRequestId,
    MapHandle, MetadataHandle,
};
pub use manifest::{ContractVersion, Manifest, ManifestValidationError, PolicyLimits};
pub use map::{
    MapClass, MapDeclaration, MapKey, MapKeyError, MapKeyType, MapMutation,
    MapMutationValidationError, MapPersistence, MapSchema, Revision,
};
pub use metadata::{
    FactDeclaration, FieldLocation, FieldUse, MetadataDeclaration, MetadataScope, Provenance,
};
pub use operation::{
    AdmissionDecision, AdmissionInput, AdmissionOutput, CandidateCharge, CandidateSet,
    DecisionValidationError, DenseScores, EvictionCause, EvictionInput, FeedbackAck,
    FeedbackAcknowledgement, FeedbackBatch, FeedbackOutput, FeedbackSubject, InvocationCause,
    LinkSet, Operation, OperationInputError, PlacementCandidate, PlacementCause, PlacementInput,
    RequestContext, ResidentUnit, ScheduleCause, ScheduleInput, SelectedEviction, SelectedService,
    ServiceCandidate, ServiceCapacity, ServiceDecision, ServicePlan, TerminalOutcome,
    rank_placements, select_evictions, select_service, validate_admission_output,
    validate_dense_scores, validate_feedback_output, validate_service_plan,
};
pub use record::{
    ColumnValueError, ColumnValues, FactColumn, FieldSchema, LinkedRecordSchema, MetadataColumn,
    RecordBatch, RecordValidationError,
};
pub use value::{TypedValue, ValueError, ValueType};
