use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Operation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractVersion {
    pub major: u16,
    pub minor: u16,
}

impl ContractVersion {
    pub const V0_2: Self = Self { major: 0, minor: 2 };
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLimits {
    pub memory_bytes: u64,
    pub fuel: u64,
    pub deadline_ms: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub contract: ContractVersion,
    pub package_name: String,
    pub package_version: String,
    pub operations: BTreeSet<Operation>,
    pub limits: PolicyLimits,
}

impl Manifest {
    pub fn validate(&self) -> Result<(), ManifestValidationError> {
        if self.contract != ContractVersion::V0_2 {
            return Err(ManifestValidationError::UnsupportedContract {
                actual: self.contract,
                expected: ContractVersion::V0_2,
            });
        }
        validate_package_name(&self.package_name)?;
        validate_package_version(&self.package_version)?;
        if self.operations.is_empty() {
            return Err(ManifestValidationError::NoOperations);
        }
        for (name, value) in [
            ("memory_bytes", self.limits.memory_bytes),
            ("fuel", self.limits.fuel),
            ("deadline_ms", self.limits.deadline_ms),
            ("input_bytes", self.limits.input_bytes),
            ("output_bytes", self.limits.output_bytes),
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
    #[error("manifest must own at least one operation")]
    NoOperations,
    #[error("limit {0} must be non-zero")]
    ZeroLimit(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            contract: ContractVersion::V0_2,
            package_name: "json-policy".into(),
            package_version: "0.2.0".into(),
            operations: BTreeSet::from([Operation::Route]),
            limits: PolicyLimits {
                memory_bytes: 1 << 20,
                fuel: 100_000,
                deadline_ms: 20,
                input_bytes: 1 << 16,
                output_bytes: 1 << 16,
            },
        }
    }

    #[test]
    fn accepts_minimal_json_manifest() {
        manifest().validate().unwrap();
    }

    #[test]
    fn rejects_old_contract_and_unknown_fields() {
        let mut old = manifest();
        old.contract = ContractVersion { major: 0, minor: 1 };
        assert!(matches!(
            old.validate(),
            Err(ManifestValidationError::UnsupportedContract { .. })
        ));

        let mut encoded = serde_json::to_value(manifest()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .insert("maps".into(), serde_json::json!([]));
        assert!(serde_json::from_value::<Manifest>(encoded).is_err());
    }
}
