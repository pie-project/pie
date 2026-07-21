use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capability::{CapabilityDeclaration, EventDeclaration, InvocationMode, Symbol};
use crate::map::{MapClass, MapDeclaration};
use crate::metadata::{FactDeclaration, FieldLocation, FieldUse, MetadataDeclaration};
use crate::operation::Operation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractVersion {
    pub major: u16,
    pub minor: u16,
}

impl ContractVersion {
    pub const V0_1: Self = Self { major: 0, minor: 1 };
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLimits {
    pub memory_bytes: u64,
    pub fuel: u64,
    pub deadline_ms: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub map_calls: u32,
    pub map_bytes: u64,
    pub staged_mutations: u32,
    pub feedback_records: u32,
    pub telemetry_records: u32,
    pub telemetry_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub contract: ContractVersion,
    pub package_name: String,
    pub package_version: String,
    pub operations: BTreeSet<Operation>,
    pub invocation_mode: InvocationMode,
    pub capabilities: Vec<CapabilityDeclaration>,
    pub facts: Vec<FactDeclaration>,
    pub metadata: Vec<MetadataDeclaration>,
    pub events: Vec<EventDeclaration>,
    pub maps: Vec<MapDeclaration>,
    pub limits: PolicyLimits,
}

impl Manifest {
    pub fn validate(&self) -> Result<(), ManifestValidationError> {
        if self.contract != ContractVersion::V0_1 {
            return Err(ManifestValidationError::UnsupportedContract {
                actual: self.contract,
                expected: ContractVersion::V0_1,
            });
        }
        validate_package_name(&self.package_name)?;
        validate_package_version(&self.package_version)?;
        if self.operations.is_empty() {
            return Err(ManifestValidationError::NoOperations);
        }
        validate_limits(&self.limits)?;

        validate_symbols(
            "capability",
            self.capabilities.iter().map(|item| &item.name),
        )?;
        validate_symbols("fact", self.facts.iter().map(|item| &item.name))?;
        validate_symbols("metadata", self.metadata.iter().map(|item| &item.name))?;
        validate_symbols("event", self.events.iter().map(|item| &item.name))?;
        validate_symbols("map", self.maps.iter().map(|item| &item.name))?;

        for declaration in &self.facts {
            if declaration.max_value_bytes == 0
                || declaration.max_value_bytes < declaration.value_type.minimum_payload_bytes()
            {
                return Err(ManifestValidationError::InvalidFieldLimit(
                    declaration.name.clone(),
                ));
            }
            validate_field_uses(&declaration.name, &declaration.uses, &self.operations)?;
        }
        for declaration in &self.metadata {
            if declaration.max_value_bytes == 0
                || declaration.max_value_bytes < declaration.value_type.minimum_payload_bytes()
            {
                return Err(ManifestValidationError::InvalidFieldLimit(
                    declaration.name.clone(),
                ));
            }
            validate_field_uses(&declaration.name, &declaration.uses, &self.operations)?;
        }

        fn validate_field_uses(
            name: &Symbol,
            uses: &BTreeSet<FieldUse>,
            operations: &BTreeSet<Operation>,
        ) -> Result<(), ManifestValidationError> {
            if uses.is_empty() {
                return Err(ManifestValidationError::FieldWithoutUse(name.clone()));
            }
            for field_use in uses {
                if !operations.contains(&field_use.operation) {
                    return Err(ManifestValidationError::FieldUseWithoutOperation {
                        name: name.clone(),
                        operation: field_use.operation,
                    });
                }
                let valid = matches!(
                    (field_use.operation, field_use.location),
                    (Operation::Admit, FieldLocation::Request)
                        | (Operation::Route, FieldLocation::Request)
                        | (Operation::Route, FieldLocation::Candidate)
                        | (Operation::Schedule, FieldLocation::Candidate)
                        | (Operation::Evict, FieldLocation::Candidate)
                        | (Operation::Feedback, FieldLocation::Feedback)
                );
                if !valid {
                    return Err(ManifestValidationError::InvalidFieldUse {
                        name: name.clone(),
                        operation: field_use.operation,
                        location: field_use.location,
                    });
                }
            }
            Ok(())
        }
        for declaration in &self.maps {
            let schema = &declaration.schema;
            if schema.max_entries == 0
                || schema.max_key_bytes == 0
                || schema.max_value_bytes == 0
                || schema.max_key_bytes < schema.key_type.minimum_payload_bytes()
                || schema.max_value_bytes < schema.value_type.minimum_payload_bytes()
            {
                return Err(ManifestValidationError::InvalidMapBounds(
                    declaration.name.clone(),
                ));
            }
            if schema.default_ttl_ms == Some(0) {
                return Err(ManifestValidationError::ZeroMapTtl(
                    declaration.name.clone(),
                ));
            }
            if schema.max_ttl_ms == Some(0)
                || schema.default_ttl_ms.is_some()
                    && schema.max_ttl_ms.is_none_or(|maximum| {
                        schema
                            .default_ttl_ms
                            .is_some_and(|default| default > maximum)
                    })
            {
                return Err(ManifestValidationError::InvalidMapTtlBounds(
                    declaration.name.clone(),
                ));
            }
            if matches!(
                declaration.class,
                MapClass::PolicyOwned {
                    persistence: crate::map::MapPersistence::Pinned
                }
            ) {
                // The contract can describe pinned state, but the first runtime
                // must reject live writable transfer until its drain protocol is
                // implemented. Keeping the declaration here avoids an ABI change.
            }
        }
        if !self.events.is_empty() && !self.operations.contains(&Operation::Feedback) {
            return Err(ManifestValidationError::EventsWithoutFeedback);
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

fn validate_limits(limits: &PolicyLimits) -> Result<(), ManifestValidationError> {
    let zero = if limits.memory_bytes == 0 {
        Some("memory_bytes")
    } else if limits.fuel == 0 {
        Some("fuel")
    } else if limits.deadline_ms == 0 {
        Some("deadline_ms")
    } else if limits.input_bytes == 0 {
        Some("input_bytes")
    } else if limits.output_bytes == 0 {
        Some("output_bytes")
    } else if limits.map_calls == 0 {
        Some("map_calls")
    } else if limits.map_bytes == 0 {
        Some("map_bytes")
    } else if limits.staged_mutations == 0 {
        Some("staged_mutations")
    } else if limits.feedback_records == 0 {
        Some("feedback_records")
    } else if limits.telemetry_records != 0 && limits.telemetry_bytes == 0 {
        Some("telemetry_bytes")
    } else {
        None
    };
    if let Some(field) = zero {
        return Err(ManifestValidationError::ZeroLimit(field));
    }
    Ok(())
}

fn validate_symbols<'a>(
    kind: &'static str,
    symbols: impl Iterator<Item = &'a Symbol>,
) -> Result<(), ManifestValidationError> {
    let mut seen = HashSet::new();
    for symbol in symbols {
        symbol
            .validate()
            .map_err(|source| ManifestValidationError::InvalidSymbol {
                kind,
                symbol: symbol.clone(),
                source,
            })?;
        if !seen.insert(symbol) {
            return Err(ManifestValidationError::DuplicateSymbol {
                kind,
                symbol: symbol.clone(),
            });
        }
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
    #[error("field declaration {0} has an impossible byte limit")]
    InvalidFieldLimit(Symbol),
    #[error("field declaration {0} is not used by any operation")]
    FieldWithoutUse(Symbol),
    #[error("field declaration {name} names unimplemented operation {operation:?}")]
    FieldUseWithoutOperation { name: Symbol, operation: Operation },
    #[error("field declaration {name} cannot be used at {location:?} for operation {operation:?}")]
    InvalidFieldUse {
        name: Symbol,
        operation: Operation,
        location: FieldLocation,
    },
    #[error("map declaration {0} has impossible entry, key, or value bounds")]
    InvalidMapBounds(Symbol),
    #[error("map declaration {0} has a zero TTL")]
    ZeroMapTtl(Symbol),
    #[error("map declaration {0} has inconsistent TTL bounds")]
    InvalidMapTtlBounds(Symbol),
    #[error("manifest subscribes to events but does not own feedback")]
    EventsWithoutFeedback,
    #[error("invalid {kind} symbol {symbol}: {source}")]
    InvalidSymbol {
        kind: &'static str,
        symbol: Symbol,
        source: crate::capability::SymbolError,
    },
    #[error("duplicate {kind} symbol {symbol}")]
    DuplicateSymbol { kind: &'static str, symbol: Symbol },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{DependencyRequirement, EventDeclaration};
    use crate::map::{MapKeyType, MapPersistence, MapSchema};
    use crate::value::ValueType;

    fn limits() -> PolicyLimits {
        PolicyLimits {
            memory_bytes: 1 << 20,
            fuel: 100_000,
            deadline_ms: 10,
            input_bytes: 1 << 16,
            output_bytes: 1 << 16,
            map_calls: 64,
            map_bytes: 1 << 14,
            staged_mutations: 16,
            feedback_records: 64,
            telemetry_records: 0,
            telemetry_bytes: 0,
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            contract: ContractVersion::V0_1,
            package_name: "attained-service".into(),
            package_version: "0.1.0".into(),
            operations: [Operation::Schedule, Operation::Feedback]
                .into_iter()
                .collect(),
            invocation_mode: InvocationMode::SetDependent,
            capabilities: Vec::new(),
            facts: Vec::new(),
            metadata: Vec::new(),
            events: vec![EventDeclaration {
                name: Symbol::new("pie.progress@1").unwrap(),
                requirement: DependencyRequirement::Required,
            }],
            maps: vec![MapDeclaration {
                name: Symbol::new("policy.accounting@1").unwrap(),
                class: MapClass::PolicyOwned {
                    persistence: MapPersistence::Attachment,
                },
                schema: MapSchema {
                    key_type: MapKeyType::Bytes,
                    value_type: ValueType::U64,
                    max_entries: 128,
                    max_key_bytes: 16,
                    max_value_bytes: 8,
                    default_ttl_ms: None,
                    max_ttl_ms: None,
                },
            }],
            limits: limits(),
        }
    }

    #[test]
    fn accepts_minimal_schedule_feedback_manifest() {
        manifest().validate().unwrap();
    }

    #[test]
    fn rejects_duplicate_symbols_after_deserialization() {
        let mut manifest = manifest();
        manifest.events.push(manifest.events[0].clone());
        assert!(matches!(
            manifest.validate(),
            Err(ManifestValidationError::DuplicateSymbol { kind: "event", .. })
        ));
    }

    #[test]
    fn rejects_event_subscription_without_feedback_owner() {
        let mut manifest = manifest();
        manifest.operations.remove(&Operation::Feedback);
        assert_eq!(
            manifest.validate(),
            Err(ManifestValidationError::EventsWithoutFeedback)
        );
    }

    #[test]
    fn rejects_zero_limits_and_map_bounds() {
        let mut zero_limit = manifest();
        zero_limit.limits.fuel = 0;
        assert_eq!(
            zero_limit.validate(),
            Err(ManifestValidationError::ZeroLimit("fuel"))
        );

        let mut zero_map = manifest();
        zero_map.maps[0].schema.max_entries = 0;
        assert!(matches!(
            zero_map.validate(),
            Err(ManifestValidationError::InvalidMapBounds(_))
        ));
    }

    #[test]
    fn rejects_unbounded_package_version_identity() {
        let mut manifest = manifest();
        manifest.package_version = format!("{}.0.0", "1".repeat(33));
        assert_eq!(
            manifest.validate(),
            Err(ManifestValidationError::InvalidPackageVersion)
        );
    }

    #[test]
    fn rejects_unscoped_and_invalid_field_uses() {
        let mut unscoped = manifest();
        unscoped.facts.push(FactDeclaration {
            name: Symbol::new("pie.unscoped@1").unwrap(),
            value_type: ValueType::U64,
            requirement: DependencyRequirement::Required,
            max_value_bytes: 8,
            uses: BTreeSet::new(),
        });
        assert!(matches!(
            unscoped.validate(),
            Err(ManifestValidationError::FieldWithoutUse(_))
        ));

        let mut invalid = manifest();
        invalid.facts.push(FactDeclaration {
            name: Symbol::new("pie.invalid-location@1").unwrap(),
            value_type: ValueType::U64,
            requirement: DependencyRequirement::Required,
            max_value_bytes: 8,
            uses: BTreeSet::from([FieldUse {
                operation: Operation::Schedule,
                location: FieldLocation::Request,
            }]),
        });
        assert!(matches!(
            invalid.validate(),
            Err(ManifestValidationError::InvalidFieldUse { .. })
        ));
    }

    #[test]
    fn manifest_json_round_trip_preserves_contract() {
        let manifest = manifest();
        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: Manifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, manifest);
        decoded.validate().unwrap();
    }

    #[test]
    fn manifest_json_rejects_unknown_fields() {
        let mut encoded = serde_json::to_value(manifest()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .insert("future-authority".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<Manifest>(encoded).is_err());
    }
}
