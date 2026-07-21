//! Isolated JSON proof-of-concept host for PLEX policy components.

#![forbid(unsafe_code)]

mod bindings;
mod context;
mod engine;
mod error;
mod lifecycle;
mod package;
mod package_format;
mod protocol;
mod registry;
mod replay;
mod request_store;

pub use engine::{PolicyEngine, PolicyEngineConfig};
pub use error::{AttachmentError, Invocation, InvocationFailure, InvocationFailureKind};
pub use lifecycle::{LifecycleHost, PlacementOutcome};
pub use package::AttachedPolicy;
pub use package_format::{PackageError, PackageLimits, PolicyPackage};
pub use protocol::{JsonResponse, ProtocolError};
pub use registry::{AttachmentRegistry, AttachmentSnapshot, RegistryError};
pub use replay::{
    ReplayCommand, ReplayDivergence, ReplayError, ReplayOutcome, ReplayReport, ReplayRunner,
    ReplaySetupError, ReplayTrace,
};
pub use request_store::{CanonicalRequestStore, RequestStoreError};
