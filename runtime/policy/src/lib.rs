//! Isolated JSON proof-of-concept host for PLEX policy components.

#![forbid(unsafe_code)]

mod bindings;
mod context;
mod engine;
mod engine_api;
mod error;
mod host;
mod lifecycle;
mod package;
mod package_format;
mod protocol;
mod registry;
mod replay;
mod state_store;

pub use engine::{PolicyEngine, PolicyEngineConfig};
pub use engine_api::{ENGINE_API_VERSION, PlexError, PlexRuntime};
pub use error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
pub use host::{
    DictionaryQueryHandler, QueryError, QueryHandler, RejectingQueryHandler, StagedAction,
};
pub use lifecycle::{LifecycleHost, PlacementOutcome};
pub use package::{AttachedPolicy, PreparedPolicyResult};
pub use package_format::{PackageError, PackageLimits, PolicyPackage};
pub use pie_plex::Document;
pub use protocol::ProtocolError;
pub use registry::{AttachmentRegistry, AttachmentSnapshot, RegistryError};
pub use replay::{
    ReplayCommand, ReplayDivergence, ReplayError, ReplayOutcome, ReplayReport, ReplayRunner,
    ReplaySetupError, ReplayTrace,
};
pub use state_store::{
    FeedbackCommit, InMemoryPolicyStateBackend, PolicyStateBackend, RequestStateUpdate,
    StateBackendError, StateSnapshot, StateUpdates,
};
