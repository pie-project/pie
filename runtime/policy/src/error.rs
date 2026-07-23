use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AttachmentError {
    #[error("invalid policy engine configuration: {0}")]
    EngineConfig(String),
    #[error(transparent)]
    Manifest(#[from] pie_plex::v0_5::ManifestValidationError),
    #[error("manifest limit {field} requests {requested}; host maximum is {maximum}")]
    HostLimit {
        field: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("failed to compile policy component: {0}")]
    Compile(String),
    #[error("failed to link policy component: {0}")]
    Link(String),
    #[error("failed to instantiate policy component within declared limits: {0}")]
    Instantiate(String),
    #[error("policy engine has no free invocation slot")]
    EngineSaturated,
    #[error("policy package is invalid")]
    Package(#[source] crate::package_format::PackageError),
    #[error("PLEX policy component imports unsupported interface {0}")]
    UnsupportedImport(String),
    #[error("policy component does not import required interface {0}")]
    MissingRequiredImport(String),
    #[error("policy component exports unsupported interface {0}")]
    UnsupportedExport(String),
    #[error("policy component does not export pie:plex/policy@0.5.0")]
    MissingPolicyExport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InvocationFailureKind {
    InvalidInput,
    Instantiation,
    PolicyFallback,
    Trap,
    FuelExhausted,
    DeadlineExceeded,
    HostSaturated,
    Query,
    ActionValidation,
    StateConflict,
    BackendFailure,
    InvalidOutput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationFailure {
    pub kind: InvocationFailureKind,
    pub message: String,
}

impl InvocationFailure {
    pub(crate) fn new(kind: InvocationFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Invocation<T> {
    Success(T),
    Unavailable,
    FallbackRequired(InvocationFailure),
}
