use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::mechanics::{MechanicId, valid_versioned_name};
use super::types::Operation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractVersion {
    pub major: u16,
    pub minor: u16,
}

impl ContractVersion {
    pub const V0_6: Self = Self { major: 0, minor: 6 };
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLimits {
    pub memory_bytes: u64,
    pub fuel: u64,
    pub deadline_ms: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub host_calls: u32,
    pub host_call_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchemaKind {
    Fact,
    ActionInput,
    ActionFeedback,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaRequirement {
    pub kind: SchemaKind,
    pub id: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub contract: ContractVersion,
    pub package_name: String,
    pub package_version: String,
    pub implements: BTreeSet<Operation>,
    pub requires: BTreeSet<MechanicId>,
    pub optional: BTreeSet<MechanicId>,
    pub schemas: BTreeSet<SchemaRequirement>,
    pub limits: PolicyLimits,
}

impl Manifest {
    pub fn validate(&self) -> Result<(), ManifestValidationError> {
        if self.contract != ContractVersion::V0_6 {
            return Err(ManifestValidationError::UnsupportedContract {
                actual: self.contract,
                expected: ContractVersion::V0_6,
            });
        }
        validate_package_name(&self.package_name)?;
        validate_package_version(&self.package_version)?;
        if self.implements.is_empty() {
            return Err(ManifestValidationError::NoOperations);
        }
        for mechanic in self.requires.iter().chain(&self.optional) {
            if !valid_versioned_name(mechanic.as_str()) {
                return Err(ManifestValidationError::InvalidMechanic(mechanic.0.clone()));
            }
        }
        if let Some(mechanic) = self.requires.intersection(&self.optional).next() {
            return Err(ManifestValidationError::MechanicOverlap(mechanic.0.clone()));
        }
        let mut schema_ids = BTreeSet::new();
        for schema in &self.schemas {
            if !valid_versioned_name(&schema.id) {
                return Err(ManifestValidationError::InvalidSchemaId(schema.id.clone()));
            }
            if !schema_ids.insert((schema.kind, schema.id.as_str())) {
                return Err(ManifestValidationError::DuplicateSchemaRequirement {
                    kind: schema.kind,
                    id: schema.id.clone(),
                });
            }
        }
        for (name, value) in [
            ("memory_bytes", self.limits.memory_bytes),
            ("fuel", self.limits.fuel),
            ("deadline_ms", self.limits.deadline_ms),
            ("input_bytes", self.limits.input_bytes),
            ("output_bytes", self.limits.output_bytes),
            ("host_calls", u64::from(self.limits.host_calls)),
            ("host_call_bytes", self.limits.host_call_bytes),
        ] {
            if value == 0 {
                return Err(ManifestValidationError::ZeroLimit(name));
            }
        }
        Ok(())
    }
}

fn validate_package_name(name: &str) -> Result<(), ManifestValidationError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ManifestValidationError::InvalidPackageName);
    }
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return Err(ManifestValidationError::InvalidPackageName);
    };
    if !first.is_ascii_alphanumeric()
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(ManifestValidationError::InvalidPackageName);
    }
    Ok(())
}

fn validate_package_version(version: &str) -> Result<(), ManifestValidationError> {
    if version.len() > 32 {
        return Err(ManifestValidationError::InvalidPackageVersion);
    }
    let mut components = version.split('.');
    let valid = (0..3).all(|_| {
        components
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    }) && components.next().is_none();
    if !valid {
        return Err(ManifestValidationError::InvalidPackageVersion);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ManifestValidationError {
    #[error("contract version {actual:?} is unsupported; expected {expected:?}")]
    UnsupportedContract {
        actual: ContractVersion,
        expected: ContractVersion,
    },
    #[error(
        "package name must be 1-64 ASCII alphanumeric, '-' or '_' bytes and start alphanumeric"
    )]
    InvalidPackageName,
    #[error("package version must contain exactly three numeric components")]
    InvalidPackageVersion,
    #[error("manifest must implement at least one operation")]
    NoOperations,
    #[error("mechanic ID {0:?} is not a valid versioned name")]
    InvalidMechanic(String),
    #[error("mechanic {0:?} cannot be both required and optional")]
    MechanicOverlap(String),
    #[error("schema ID {0:?} is not a valid versioned name")]
    InvalidSchemaId(String),
    #[error("schema requirement {kind:?}/{id:?} is declared more than once")]
    DuplicateSchemaRequirement { kind: SchemaKind, id: String },
    #[error("limit {0} must be non-zero")]
    ZeroLimit(&'static str),
}
