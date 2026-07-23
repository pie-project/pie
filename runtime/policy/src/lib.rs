//! Isolated JSON proof-of-concept host for PLEX policy components.

#![forbid(unsafe_code)]

mod bindings;
mod bindings_v0_6;
mod context;
mod engine;
mod engine_api;
mod engine_api_v0_6;
mod error;
mod host;
mod lifecycle;
mod lifecycle_v0_6;
mod package;
mod package_format;
mod package_format_v0_6;
mod package_v0_6;
mod protocol;
mod protocol_v0_6;
mod registry;
mod registry_v0_6;
mod replay;
mod replay_v0_6;
mod state_store;
mod state_store_v0_6;
mod wire_v0_6;

pub use engine::{PolicyEngine, PolicyEngineConfig};
pub use engine_api::{
    ENGINE_API_VERSION as ENGINE_API_VERSION_V0_5, PlexError as PlexErrorV0_5,
    PlexRuntime as PlexRuntimeV0_5,
};
pub use engine_api_v0_6::{ENGINE_API_VERSION_V0_6, PlexErrorV0_6, PlexRuntimeV0_6};
pub use engine_api_v0_6::{
    ENGINE_API_VERSION_V0_6 as ENGINE_API_VERSION, PlexErrorV0_6 as PlexError,
    PlexRuntimeV0_6 as PlexRuntime,
};
pub use error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
pub use host::{
    DictionaryQueryHandler, QueryError, QueryHandler, RejectingQueryHandler, StagedAction,
};
pub use lifecycle::{LifecycleHost, PlacementOutcome};
pub use lifecycle_v0_6::{LifecycleEventV0_6, LifecycleHostV0_6, LifecycleOutcomeV0_6};
pub use package::{AttachedPolicy, PreparedPolicyResult};
pub use package_format::{
    PackageError as PackageErrorV0_5, PackageLimits, PolicyPackage as PolicyPackageV0_5,
};
pub use package_format_v0_6::{PackageErrorV0_6, PolicyPackageV0_6};
pub use package_format_v0_6::{
    PackageErrorV0_6 as PackageError, PolicyPackageV0_6 as PolicyPackage,
};
pub use package_v0_6::{AttachedPolicyV0_6, AttachmentErrorV0_6, PreparedPolicyResultV0_6};
pub use pie_plex::Document;
pub use protocol::ProtocolError;
pub use protocol_v0_6::{
    CacheEpisodeTrackerV0_6, NormalizedPlanV0_6, NormalizedRouteAssignmentV0_6,
    NormalizedRoutePlanV0_6, OperationContextV0_6, OperationPlanV0_6, OpportunityTrackerV0_6,
    ProtocolErrorV0_6, ProtocolLimitsV0_6, ReplayRecordV0_6, TraceOrderV0_6, normalized_plan_v0_6,
    replay_record_v0_6, snapshot_ref_v0_6, validate_context_v0_6, validate_output_v0_6,
    validate_snapshot_context_v0_6, working_set_v0_6,
};
pub use registry::{AttachmentRegistry, AttachmentSnapshot, RegistryError};
pub use registry_v0_6::{
    AttachmentRegistryV0_6, AttachmentSnapshotV0_6, HostSupportV0_6, RegistryErrorV0_6,
    SchemaKeyV0_6,
};
pub use replay::{
    ReplayCommand, ReplayDivergence, ReplayError, ReplayOutcome, ReplayReport, ReplayRunner,
    ReplaySetupError, ReplayTrace,
};
pub use replay_v0_6::{ReplayErrorV0_6, ReplayReportV0_6, ReplayRunnerV0_6};
pub use state_store::{
    FeedbackCommit, InMemoryPolicyStateBackend, PolicyStateBackend, RequestStateUpdate,
    StateBackendError, StateSnapshot, StateUpdates,
};
pub use state_store_v0_6::{
    FeedbackCommitV0_6, InMemoryPolicyStateBackendV0_6, PolicyStateBackendV0_6,
    StateBackendErrorV0_6, StateMetricsV0_6, StateScopeV0_6, StateSnapshotV0_6,
    TerminalCleanupV0_6, TerminalGroupV0_6, TerminalRequestV0_6, WorkingSetV0_6,
};
