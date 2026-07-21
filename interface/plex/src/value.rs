use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueType {
    Bool,
    I64,
    U64,
    F64,
    String,
    Bytes,
}

impl ValueType {
    pub const fn minimum_payload_bytes(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::String | Self::Bytes => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "kebab-case")]
pub enum TypedValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
}

impl TypedValue {
    pub const fn value_type(&self) -> ValueType {
        match self {
            Self::Bool(_) => ValueType::Bool,
            Self::I64(_) => ValueType::I64,
            Self::U64(_) => ValueType::U64,
            Self::F64(_) => ValueType::F64,
            Self::String(_) => ValueType::String,
            Self::Bytes(_) => ValueType::Bytes,
        }
    }

    pub fn validate(&self) -> Result<(), ValueError> {
        if let Self::F64(value) = self
            && !value.is_finite()
        {
            return Err(ValueError::NonFiniteFloat);
        }
        Ok(())
    }

    /// Payload bytes used for invocation and map quotas.
    pub fn payload_len(&self) -> usize {
        match self {
            Self::Bool(_) => 1,
            Self::I64(_) | Self::U64(_) | Self::F64(_) => 8,
            Self::String(value) => value.len(),
            Self::Bytes(value) => value.len(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ValueError {
    #[error("floating-point values must be finite")]
    NonFiniteFloat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_types_and_payload_lengths() {
        let cases = [
            (TypedValue::Bool(true), ValueType::Bool, 1),
            (TypedValue::I64(-4), ValueType::I64, 8),
            (TypedValue::U64(4), ValueType::U64, 8),
            (TypedValue::F64(0.25), ValueType::F64, 8),
            (TypedValue::String("plex".to_owned()), ValueType::String, 4),
            (TypedValue::Bytes(vec![1, 2, 3]), ValueType::Bytes, 3),
        ];

        for (value, expected_type, expected_len) in cases {
            assert_eq!(value.value_type(), expected_type);
            assert_eq!(value.payload_len(), expected_len);
            value.validate().unwrap();
        }
    }

    #[test]
    fn rejects_all_non_finite_floats() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                TypedValue::F64(value).validate(),
                Err(ValueError::NonFiniteFloat)
            );
        }
    }
}
