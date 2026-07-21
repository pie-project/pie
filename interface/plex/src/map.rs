use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capability::{DependencyRequirement, Symbol};
use crate::ids::MapHandle;
use crate::value::{TypedValue, ValueType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MapKeyType {
    Bool,
    I64,
    U64,
    String,
    Bytes,
}

impl MapKeyType {
    pub const fn minimum_payload_bytes(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I64 | Self::U64 => 8,
            Self::String | Self::Bytes => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "kebab-case")]
pub enum MapKey {
    Bool(bool),
    I64(i64),
    U64(u64),
    String(String),
    Bytes(Vec<u8>),
}

impl MapKey {
    pub const fn key_type(&self) -> MapKeyType {
        match self {
            Self::Bool(_) => MapKeyType::Bool,
            Self::I64(_) => MapKeyType::I64,
            Self::U64(_) => MapKeyType::U64,
            Self::String(_) => MapKeyType::String,
            Self::Bytes(_) => MapKeyType::Bytes,
        }
    }

    pub fn payload_len(&self) -> usize {
        match self {
            Self::Bool(_) => 1,
            Self::I64(_) | Self::U64(_) => 8,
            Self::String(value) => value.len(),
            Self::Bytes(value) => value.len(),
        }
    }

    pub fn into_value(self) -> TypedValue {
        match self {
            Self::Bool(value) => TypedValue::Bool(value),
            Self::I64(value) => TypedValue::I64(value),
            Self::U64(value) => TypedValue::U64(value),
            Self::String(value) => TypedValue::String(value),
            Self::Bytes(value) => TypedValue::Bytes(value),
        }
    }
}

impl TryFrom<TypedValue> for MapKey {
    type Error = MapKeyError;

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        match value {
            TypedValue::Bool(value) => Ok(Self::Bool(value)),
            TypedValue::I64(value) => Ok(Self::I64(value)),
            TypedValue::U64(value) => Ok(Self::U64(value)),
            TypedValue::String(value) => Ok(Self::String(value)),
            TypedValue::Bytes(value) => Ok(Self::Bytes(value)),
            TypedValue::F64(_) => Err(MapKeyError::FloatUnsupported),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MapKeyError {
    #[error("floating-point values cannot be map keys")]
    FloatUnsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MapPersistence {
    Attachment,
    Pinned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "owner", rename_all = "kebab-case")]
pub enum MapClass {
    External { requirement: DependencyRequirement },
    PolicyOwned { persistence: MapPersistence },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapSchema {
    pub key_type: MapKeyType,
    pub value_type: ValueType,
    pub max_entries: u32,
    pub max_key_bytes: u32,
    pub max_value_bytes: u32,
    pub default_ttl_ms: Option<u64>,
    pub max_ttl_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapDeclaration {
    pub name: Symbol,
    pub class: MapClass,
    pub schema: MapSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case")]
pub enum MapMutation {
    Upsert {
        map: MapHandle,
        key: MapKey,
        value: TypedValue,
        ttl_ms: Option<u64>,
    },
    AddI64 {
        map: MapHandle,
        key: MapKey,
        delta: i64,
        ttl_ms: Option<u64>,
    },
    AddU64 {
        map: MapHandle,
        key: MapKey,
        delta: u64,
        ttl_ms: Option<u64>,
    },
    Delete {
        map: MapHandle,
        key: MapKey,
    },
}

impl MapMutation {
    pub const fn map(&self) -> MapHandle {
        match self {
            Self::Upsert { map, .. }
            | Self::AddI64 { map, .. }
            | Self::AddU64 { map, .. }
            | Self::Delete { map, .. } => *map,
        }
    }

    pub fn key(&self) -> &MapKey {
        match self {
            Self::Upsert { key, .. }
            | Self::AddI64 { key, .. }
            | Self::AddU64 { key, .. }
            | Self::Delete { key, .. } => key,
        }
    }

    pub const fn ttl_ms(&self) -> Option<u64> {
        match self {
            Self::Upsert { ttl_ms, .. }
            | Self::AddI64 { ttl_ms, .. }
            | Self::AddU64 { ttl_ms, .. } => *ttl_ms,
            Self::Delete { .. } => None,
        }
    }

    pub fn charged_bytes(&self) -> Result<usize, MapMutationValidationError> {
        // Mutation tag + map handle + key tag + key payload.
        let mut total = 9usize
            .checked_add(self.key().payload_len())
            .ok_or(MapMutationValidationError::SizeOverflow)?;
        match self {
            Self::Upsert { value, ttl_ms, .. } => {
                total = total
                    .checked_add(1 + value.payload_len())
                    .ok_or(MapMutationValidationError::SizeOverflow)?;
                total = total
                    .checked_add(1 + usize::from(ttl_ms.is_some()) * 8)
                    .ok_or(MapMutationValidationError::SizeOverflow)?;
            }
            Self::AddI64 { ttl_ms, .. } | Self::AddU64 { ttl_ms, .. } => {
                total = total
                    .checked_add(8 + 1 + usize::from(ttl_ms.is_some()) * 8)
                    .ok_or(MapMutationValidationError::SizeOverflow)?;
            }
            Self::Delete { .. } => {}
        }
        Ok(total)
    }

    pub fn validate_against(
        &self,
        declaration: &MapDeclaration,
    ) -> Result<(), MapMutationValidationError> {
        if matches!(declaration.class, MapClass::External { .. }) {
            return Err(MapMutationValidationError::ExternalMap);
        }

        let schema = &declaration.schema;
        if self.key().key_type() != schema.key_type {
            return Err(MapMutationValidationError::KeyType {
                expected: schema.key_type,
                actual: self.key().key_type(),
            });
        }
        let key_bytes = self.key().payload_len();
        if key_bytes > schema.max_key_bytes as usize {
            return Err(MapMutationValidationError::KeyTooLarge {
                actual: key_bytes,
                maximum: schema.max_key_bytes as usize,
            });
        }
        validate_ttl(self.ttl_ms(), schema)?;

        match self {
            Self::Upsert { value, .. } => {
                value
                    .validate()
                    .map_err(|_| MapMutationValidationError::NonFiniteValue)?;
                if value.value_type() != schema.value_type {
                    return Err(MapMutationValidationError::ValueType {
                        expected: schema.value_type,
                        actual: value.value_type(),
                    });
                }
                if value.payload_len() > schema.max_value_bytes as usize {
                    return Err(MapMutationValidationError::ValueTooLarge {
                        actual: value.payload_len(),
                        maximum: schema.max_value_bytes as usize,
                    });
                }
            }
            Self::AddI64 { .. } if schema.value_type != ValueType::I64 => {
                return Err(MapMutationValidationError::AddType {
                    operation: "add-i64",
                    actual: schema.value_type,
                });
            }
            Self::AddU64 { .. } if schema.value_type != ValueType::U64 => {
                return Err(MapMutationValidationError::AddType {
                    operation: "add-u64",
                    actual: schema.value_type,
                });
            }
            Self::AddI64 { .. } | Self::AddU64 { .. } | Self::Delete { .. } => {}
        }
        Ok(())
    }
}

fn validate_ttl(ttl_ms: Option<u64>, schema: &MapSchema) -> Result<(), MapMutationValidationError> {
    let Some(ttl_ms) = ttl_ms else {
        return Ok(());
    };
    if ttl_ms == 0 {
        return Err(MapMutationValidationError::ZeroTtl);
    }
    let Some(maximum) = schema.max_ttl_ms else {
        return Err(MapMutationValidationError::TtlUnsupported);
    };
    if ttl_ms > maximum {
        return Err(MapMutationValidationError::TtlTooLarge {
            actual: ttl_ms,
            maximum,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MapMutationValidationError {
    #[error("external maps are read-only")]
    ExternalMap,
    #[error("map key has type {actual:?}; expected {expected:?}")]
    KeyType {
        expected: MapKeyType,
        actual: MapKeyType,
    },
    #[error("map key contains {actual} bytes; maximum is {maximum}")]
    KeyTooLarge { actual: usize, maximum: usize },
    #[error("map value has type {actual:?}; expected {expected:?}")]
    ValueType {
        expected: ValueType,
        actual: ValueType,
    },
    #[error("map value contains {actual} bytes; maximum is {maximum}")]
    ValueTooLarge { actual: usize, maximum: usize },
    #[error("map value contains a non-finite float")]
    NonFiniteValue,
    #[error("{operation} requires a matching numeric map, got {actual:?}")]
    AddType {
        operation: &'static str,
        actual: ValueType,
    },
    #[error("TTL must be non-zero")]
    ZeroTtl,
    #[error("map does not permit per-entry TTLs")]
    TtlUnsupported,
    #[error("TTL {actual}ms exceeds maximum {maximum}ms")]
    TtlTooLarge { actual: u64, maximum: u64 },
    #[error("mutation charge overflowed usize")]
    SizeOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Revision(u64);

impl Revision {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_keys_round_trip_through_typed_values() {
        let keys = [
            MapKey::Bool(true),
            MapKey::I64(-1),
            MapKey::U64(1),
            MapKey::String("plex".into()),
            MapKey::Bytes(vec![1, 2, 3]),
        ];
        for key in keys {
            let expected = key.clone();
            assert_eq!(MapKey::try_from(key.into_value()).unwrap(), expected);
        }
    }

    #[test]
    fn floating_point_map_keys_are_rejected() {
        assert_eq!(
            MapKey::try_from(TypedValue::F64(1.0)),
            Err(MapKeyError::FloatUnsupported)
        );
    }

    fn policy_map(value_type: ValueType) -> MapDeclaration {
        MapDeclaration {
            name: Symbol::new("policy.state@1").unwrap(),
            class: MapClass::PolicyOwned {
                persistence: MapPersistence::Attachment,
            },
            schema: MapSchema {
                key_type: MapKeyType::Bytes,
                value_type,
                max_entries: 8,
                max_key_bytes: 4,
                max_value_bytes: 8,
                default_ttl_ms: None,
                max_ttl_ms: Some(100),
            },
        }
    }

    #[test]
    fn validates_mutation_schema_and_ttl() {
        let mutation = MapMutation::Upsert {
            map: MapHandle::new(0),
            key: MapKey::Bytes(vec![1, 2]),
            value: TypedValue::U64(4),
            ttl_ms: Some(50),
        };
        mutation
            .validate_against(&policy_map(ValueType::U64))
            .unwrap();

        let wrong_type = policy_map(ValueType::String);
        assert!(matches!(
            mutation.validate_against(&wrong_type),
            Err(MapMutationValidationError::ValueType { .. })
        ));

        let too_long = MapMutation::Upsert {
            map: MapHandle::new(0),
            key: MapKey::Bytes(vec![1, 2, 3, 4, 5]),
            value: TypedValue::U64(4),
            ttl_ms: None,
        };
        assert!(matches!(
            too_long.validate_against(&policy_map(ValueType::U64)),
            Err(MapMutationValidationError::KeyTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_writes_to_external_maps() {
        let external = MapDeclaration {
            name: Symbol::new("operator.config@1").unwrap(),
            class: MapClass::External {
                requirement: DependencyRequirement::Required,
            },
            schema: policy_map(ValueType::U64).schema,
        };
        let mutation = MapMutation::Delete {
            map: MapHandle::new(0),
            key: MapKey::Bytes(vec![1]),
        };
        assert_eq!(
            mutation.validate_against(&external),
            Err(MapMutationValidationError::ExternalMap)
        );
    }
}
