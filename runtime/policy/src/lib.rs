//! Isolated host for PLEX operator-policy components.

#![forbid(unsafe_code)]

mod bindings;
mod context;
mod convert;
mod engine;
mod error;
mod link;
mod maps;
mod package;
mod package_format;
mod registry;
mod replay;
mod telemetry;

pub use engine::{PolicyEngine, PolicyEngineConfig};
pub use error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
pub use link::{AttachmentResolution, CapabilityCatalog, CatalogError, LinkError};
pub use maps::{
    Clock, CommitResult, DedupLimits, FeedbackStart, InvocationTransaction, ManualClock,
    MapAccessError, MapStore, MapStoreError, PrepareError, PreparedTransaction, StateTransferError,
    SystemClock,
};
pub use package::{AttachedPolicy, InvocationMetrics, PreparedDecision};
pub use package_format::{PackageError, PackageLimits, PolicyPackage};
pub use registry::{AttachedDecision, AttachmentRegistry, AttachmentSnapshot, RegistryError};
pub use replay::{
    Enactment, ReplayCommand, ReplayDivergence, ReplayError, ReplayOutcome, ReplayReport,
    ReplayRunner, ReplaySetupError, ReplayTrace,
};
pub use telemetry::TelemetryRecord;
