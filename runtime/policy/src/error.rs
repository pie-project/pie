use pie_plex::{DecisionValidationError, ManifestValidationError, OperationInputError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AttachmentError {
    #[error("invalid policy engine configuration: {0}")]
    EngineConfig(String),
    #[error(transparent)]
    Manifest(#[from] ManifestValidationError),
    #[error("component contains {actual} bytes; host maximum is {maximum}")]
    ComponentTooLarge { actual: usize, maximum: usize },
    #[error("manifest limit {field} requests {requested}; host maximum is {maximum}")]
    HostLimit {
        field: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("failed to compile policy component")]
    Compile(String),
    #[error("failed to link policy component")]
    Link(String),
    #[error("failed to instantiate policy component within declared limits")]
    Instantiate(String),
    #[error("policy engine has no free invocation slot")]
    EngineSaturated,
    #[error("policy requirements do not match the host catalog")]
    Resolve(#[source] crate::link::LinkError),
    #[error("failed to initialize policy maps")]
    Maps(#[source] crate::maps::MapStoreError),
    #[error("policy package is invalid")]
    Package(#[source] crate::package_format::PackageError),
    #[error("policy component imports unsupported interface {0}")]
    UnsupportedImport(String),
    #[error("policy component exports unsupported interface {0}")]
    UnsupportedExport(String),
    #[error("policy component does not export pie:plex/policy@0.1.0")]
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
    MapLimitExceeded,
    HostSaturated,
    TransactionConflict,
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

    pub(crate) fn input(error: OperationInputError) -> Self {
        Self::new(InvocationFailureKind::InvalidInput, error.to_string())
    }

    pub(crate) fn output(error: DecisionValidationError) -> Self {
        Self::new(InvocationFailureKind::InvalidOutput, error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Invocation<T> {
    Success(T),
    Unavailable,
    FallbackRequired(InvocationFailure),
}
