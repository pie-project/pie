use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{FactHandle, MetadataHandle};
use crate::value::ValueType;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "values", rename_all = "kebab-case")]
pub enum ColumnValues {
    Bool(Vec<Option<bool>>),
    I64(Vec<Option<i64>>),
    U64(Vec<Option<u64>>),
    F64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Bytes(Vec<Option<Vec<u8>>>),
}

impl ColumnValues {
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

    pub fn len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::String(values) => values.len(),
            Self::Bytes(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn validate_finite(&self) -> Result<(), ColumnValueError> {
        if let Self::F64(values) = self
            && let Some((row, _)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| value.is_some_and(|value| !value.is_finite()))
        {
            return Err(ColumnValueError::NonFiniteFloat { row });
        }
        Ok(())
    }

    fn validate_cells(
        &self,
        required_values: bool,
        max_value_bytes: usize,
    ) -> Result<(), ColumnValueError> {
        self.validate_finite()?;
        match self {
            Self::Bool(values) => validate_fixed_cells(values, required_values, max_value_bytes, 1),
            Self::I64(values) => validate_fixed_cells(values, required_values, max_value_bytes, 8),
            Self::U64(values) => validate_fixed_cells(values, required_values, max_value_bytes, 8),
            Self::F64(values) => validate_fixed_cells(values, required_values, max_value_bytes, 8),
            Self::String(values) => {
                validate_variable_cells(values, required_values, max_value_bytes, String::len)
            }
            Self::Bytes(values) => {
                validate_variable_cells(values, required_values, max_value_bytes, Vec::len)
            }
        }
    }

    fn charged_bytes(&self) -> Result<usize, RecordValidationError> {
        // Variant discriminant + list length.
        let mut total = 5usize;
        match self {
            Self::Bool(values) => {
                for value in values {
                    total = checked_add(total, 1 + usize::from(value.is_some()))?;
                }
            }
            Self::I64(values) => total = charge_fixed_options(total, values, 8)?,
            Self::U64(values) => total = charge_fixed_options(total, values, 8)?,
            Self::F64(values) => total = charge_fixed_options(total, values, 8)?,
            Self::String(values) => {
                for value in values {
                    total = checked_add(total, 1)?;
                    if let Some(value) = value {
                        total = checked_add(total, 4)?;
                        total = checked_add(total, value.len())?;
                    }
                }
            }
            Self::Bytes(values) => {
                for value in values {
                    total = checked_add(total, 1)?;
                    if let Some(value) = value {
                        total = checked_add(total, 4)?;
                        total = checked_add(total, value.len())?;
                    }
                }
            }
        }
        Ok(total)
    }
}

fn validate_fixed_cells<T>(
    values: &[Option<T>],
    required_values: bool,
    maximum: usize,
    actual: usize,
) -> Result<(), ColumnValueError> {
    for (row, value) in values.iter().enumerate() {
        if value.is_none() && required_values {
            return Err(ColumnValueError::MissingRequiredValue { row });
        }
        if value.is_some() && actual > maximum {
            return Err(ColumnValueError::ValueTooLarge {
                row,
                actual,
                maximum,
            });
        }
    }
    Ok(())
}

fn validate_variable_cells<T>(
    values: &[Option<T>],
    required_values: bool,
    maximum: usize,
    len: impl Fn(&T) -> usize,
) -> Result<(), ColumnValueError> {
    for (row, value) in values.iter().enumerate() {
        let Some(value) = value else {
            if required_values {
                return Err(ColumnValueError::MissingRequiredValue { row });
            }
            continue;
        };
        let actual = len(value);
        if actual > maximum {
            return Err(ColumnValueError::ValueTooLarge {
                row,
                actual,
                maximum,
            });
        }
    }
    Ok(())
}

fn charge_fixed_options<T>(
    mut total: usize,
    values: &[Option<T>],
    width: usize,
) -> Result<usize, RecordValidationError> {
    for value in values {
        total = checked_add(total, 1)?;
        if value.is_some() {
            total = checked_add(total, width)?;
        }
    }
    Ok(total)
}

fn checked_add(lhs: usize, rhs: usize) -> Result<usize, RecordValidationError> {
    lhs.checked_add(rhs)
        .ok_or(RecordValidationError::SizeOverflow)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactColumn {
    pub handle: FactHandle,
    pub values: ColumnValues,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataColumn {
    pub handle: MetadataHandle,
    pub values: ColumnValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldSchema {
    pub value_type: ValueType,
    pub required_column: bool,
    /// Facts normally require a value for every row. Metadata normally permits
    /// absence even when the metadata schema itself was required at attachment.
    pub required_values: bool,
    pub max_value_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LinkedRecordSchema {
    pub facts: BTreeMap<FactHandle, FieldSchema>,
    pub metadata: BTreeMap<MetadataHandle, FieldSchema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordBatch {
    pub rows: u32,
    pub facts: Vec<FactColumn>,
    pub metadata: Vec<MetadataColumn>,
}

impl RecordBatch {
    pub const fn empty(rows: u32) -> Self {
        Self {
            rows,
            facts: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Deterministic quota charge for the WIT-shaped record batch.
    ///
    /// This is a contract charge, not an estimate of Wasmtime's in-memory
    /// representation. It includes record/list/option/variant tags and payloads.
    pub fn charged_bytes(&self) -> Result<usize, RecordValidationError> {
        // rows + fact-list length + metadata-list length.
        let mut total = 12usize;
        for column in &self.facts {
            total = checked_add(total, 4)?;
            total = checked_add(total, column.values.charged_bytes()?)?;
        }
        for column in &self.metadata {
            total = checked_add(total, 4)?;
            total = checked_add(total, column.values.charged_bytes()?)?;
        }
        Ok(total)
    }

    pub fn validate(&self, max_charged_bytes: usize) -> Result<(), RecordValidationError> {
        let rows =
            usize::try_from(self.rows).map_err(|_| RecordValidationError::RowCountOverflow)?;
        let mut facts = HashSet::with_capacity(self.facts.len());
        let mut metadata = HashSet::with_capacity(self.metadata.len());

        for column in &self.facts {
            if !facts.insert(column.handle) {
                return Err(RecordValidationError::DuplicateFactHandle(column.handle));
            }
            validate_column_length(rows, column.values.len(), FieldKind::Fact(column.handle))?;
            column.values.validate_finite().map_err(|source| {
                RecordValidationError::InvalidFact {
                    handle: column.handle,
                    source,
                }
            })?;
        }
        for column in &self.metadata {
            if !metadata.insert(column.handle) {
                return Err(RecordValidationError::DuplicateMetadataHandle(
                    column.handle,
                ));
            }
            validate_column_length(
                rows,
                column.values.len(),
                FieldKind::Metadata(column.handle),
            )?;
            column.values.validate_finite().map_err(|source| {
                RecordValidationError::InvalidMetadata {
                    handle: column.handle,
                    source,
                }
            })?;
        }

        let actual = self.charged_bytes()?;
        if actual > max_charged_bytes {
            return Err(RecordValidationError::PayloadTooLarge {
                actual,
                maximum: max_charged_bytes,
            });
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        schema: &LinkedRecordSchema,
        max_charged_bytes: usize,
    ) -> Result<(), RecordValidationError> {
        self.validate(max_charged_bytes)?;

        let present_facts: HashSet<_> = self.facts.iter().map(|column| column.handle).collect();
        let present_metadata: HashSet<_> =
            self.metadata.iter().map(|column| column.handle).collect();

        for column in &self.facts {
            let field = schema
                .facts
                .get(&column.handle)
                .ok_or(RecordValidationError::UnknownFactHandle(column.handle))?;
            validate_schema(&column.values, *field, FieldKind::Fact(column.handle))?;
        }
        for column in &self.metadata {
            let field = schema
                .metadata
                .get(&column.handle)
                .ok_or(RecordValidationError::UnknownMetadataHandle(column.handle))?;
            validate_schema(&column.values, *field, FieldKind::Metadata(column.handle))?;
        }

        if let Some(handle) = schema.facts.iter().find_map(|(handle, field)| {
            (field.required_column && !present_facts.contains(handle)).then_some(handle)
        }) {
            return Err(RecordValidationError::MissingFactColumn(*handle));
        }
        if let Some(handle) = schema.metadata.iter().find_map(|(handle, field)| {
            (field.required_column && !present_metadata.contains(handle)).then_some(handle)
        }) {
            return Err(RecordValidationError::MissingMetadataColumn(*handle));
        }
        Ok(())
    }
}

fn validate_column_length(
    expected: usize,
    actual: usize,
    field: FieldKind,
) -> Result<(), RecordValidationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(match field {
            FieldKind::Fact(handle) => RecordValidationError::FactColumnLength {
                handle,
                expected,
                actual,
            },
            FieldKind::Metadata(handle) => RecordValidationError::MetadataColumnLength {
                handle,
                expected,
                actual,
            },
        })
    }
}

fn validate_schema(
    values: &ColumnValues,
    schema: FieldSchema,
    field: FieldKind,
) -> Result<(), RecordValidationError> {
    if values.value_type() != schema.value_type {
        return Err(match field {
            FieldKind::Fact(handle) => RecordValidationError::FactType {
                handle,
                expected: schema.value_type,
                actual: values.value_type(),
            },
            FieldKind::Metadata(handle) => RecordValidationError::MetadataType {
                handle,
                expected: schema.value_type,
                actual: values.value_type(),
            },
        });
    }
    values
        .validate_cells(
            schema.required_values,
            usize::try_from(schema.max_value_bytes).unwrap_or(usize::MAX),
        )
        .map_err(|source| match field {
            FieldKind::Fact(handle) => RecordValidationError::InvalidFact { handle, source },
            FieldKind::Metadata(handle) => {
                RecordValidationError::InvalidMetadata { handle, source }
            }
        })
}

#[derive(Debug, Clone, Copy)]
enum FieldKind {
    Fact(FactHandle),
    Metadata(MetadataHandle),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ColumnValueError {
    #[error("row {row} is missing a required value")]
    MissingRequiredValue { row: usize },
    #[error("row {row} contains {actual} bytes; maximum is {maximum}")]
    ValueTooLarge {
        row: usize,
        actual: usize,
        maximum: usize,
    },
    #[error("row {row} contains a non-finite float")]
    NonFiniteFloat { row: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RecordValidationError {
    #[error("record row count does not fit in usize")]
    RowCountOverflow,
    #[error("fact handle {0:?} appears more than once")]
    DuplicateFactHandle(FactHandle),
    #[error("metadata handle {0:?} appears more than once")]
    DuplicateMetadataHandle(MetadataHandle),
    #[error("fact handle {handle:?} has {actual} values; expected {expected}")]
    FactColumnLength {
        handle: FactHandle,
        expected: usize,
        actual: usize,
    },
    #[error("metadata handle {handle:?} has {actual} values; expected {expected}")]
    MetadataColumnLength {
        handle: MetadataHandle,
        expected: usize,
        actual: usize,
    },
    #[error("fact handle {0:?} is not linked")]
    UnknownFactHandle(FactHandle),
    #[error("metadata handle {0:?} is not linked")]
    UnknownMetadataHandle(MetadataHandle),
    #[error("linked fact handle {0:?} has no column")]
    MissingFactColumn(FactHandle),
    #[error("linked metadata handle {0:?} has no column")]
    MissingMetadataColumn(MetadataHandle),
    #[error("fact handle {handle:?} has type {actual:?}; expected {expected:?}")]
    FactType {
        handle: FactHandle,
        expected: ValueType,
        actual: ValueType,
    },
    #[error("metadata handle {handle:?} has type {actual:?}; expected {expected:?}")]
    MetadataType {
        handle: MetadataHandle,
        expected: ValueType,
        actual: ValueType,
    },
    #[error("fact handle {handle:?} is invalid: {source}")]
    InvalidFact {
        handle: FactHandle,
        source: ColumnValueError,
    },
    #[error("metadata handle {handle:?} is invalid: {source}")]
    InvalidMetadata {
        handle: MetadataHandle,
        source: ColumnValueError,
    },
    #[error("record payload size overflowed usize")]
    SizeOverflow,
    #[error("record payload charge is {actual} bytes; maximum is {maximum}")]
    PayloadTooLarge { actual: usize, maximum: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(handle: u32, values: ColumnValues) -> FactColumn {
        FactColumn {
            handle: FactHandle::new(handle),
            values,
        }
    }

    fn metadata(handle: u32, values: ColumnValues) -> MetadataColumn {
        MetadataColumn {
            handle: MetadataHandle::new(handle),
            values,
        }
    }

    #[test]
    fn validates_distinct_fact_and_metadata_namespaces() {
        let batch = RecordBatch {
            rows: 2,
            facts: vec![fact(0, ColumnValues::U64(vec![Some(1), Some(2)]))],
            metadata: vec![metadata(
                0,
                ColumnValues::String(vec![Some("a".into()), None]),
            )],
        };
        batch.validate(1024).unwrap();
    }

    #[test]
    fn validates_linked_schema_type_presence_and_bounds() {
        let batch = RecordBatch {
            rows: 2,
            facts: vec![fact(0, ColumnValues::U64(vec![Some(1), Some(2)]))],
            metadata: vec![metadata(
                0,
                ColumnValues::String(vec![Some("abc".into()), None]),
            )],
        };
        let schema = LinkedRecordSchema {
            facts: BTreeMap::from([(
                FactHandle::new(0),
                FieldSchema {
                    value_type: ValueType::U64,
                    required_column: true,
                    required_values: true,
                    max_value_bytes: 8,
                },
            )]),
            metadata: BTreeMap::from([(
                MetadataHandle::new(0),
                FieldSchema {
                    value_type: ValueType::String,
                    required_column: false,
                    required_values: false,
                    max_value_bytes: 3,
                },
            )]),
        };
        batch.validate_against(&schema, 1024).unwrap();

        let mut wrong_type = schema.clone();
        wrong_type
            .metadata
            .get_mut(&MetadataHandle::new(0))
            .unwrap()
            .value_type = ValueType::Bytes;
        assert!(matches!(
            batch.validate_against(&wrong_type, 1024),
            Err(RecordValidationError::MetadataType { .. })
        ));

        let mut too_small = schema;
        too_small
            .metadata
            .get_mut(&MetadataHandle::new(0))
            .unwrap()
            .max_value_bytes = 2;
        assert!(matches!(
            batch.validate_against(&too_small, 1024),
            Err(RecordValidationError::InvalidMetadata {
                source: ColumnValueError::ValueTooLarge { .. },
                ..
            })
        ));
    }

    #[test]
    fn rejects_missing_required_fact_value() {
        let batch = RecordBatch {
            rows: 1,
            facts: vec![fact(0, ColumnValues::U64(vec![None]))],
            metadata: Vec::new(),
        };
        let schema = LinkedRecordSchema {
            facts: BTreeMap::from([(
                FactHandle::new(0),
                FieldSchema {
                    value_type: ValueType::U64,
                    required_column: true,
                    required_values: true,
                    max_value_bytes: 8,
                },
            )]),
            metadata: BTreeMap::new(),
        };
        assert!(matches!(
            batch.validate_against(&schema, 1024),
            Err(RecordValidationError::InvalidFact {
                source: ColumnValueError::MissingRequiredValue { row: 0 },
                ..
            })
        ));
    }

    #[test]
    fn optional_columns_may_be_absent() {
        let batch = RecordBatch::empty(2);
        let schema = LinkedRecordSchema {
            facts: BTreeMap::new(),
            metadata: BTreeMap::from([(
                MetadataHandle::new(0),
                FieldSchema {
                    value_type: ValueType::String,
                    required_column: false,
                    required_values: false,
                    max_value_bytes: 32,
                },
            )]),
        };
        batch.validate_against(&schema, 1024).unwrap();
    }

    #[test]
    fn rejects_wrong_length_duplicate_and_non_finite_columns() {
        let wrong_length = RecordBatch {
            rows: 2,
            facts: vec![fact(0, ColumnValues::Bool(vec![Some(true)]))],
            metadata: Vec::new(),
        };
        assert!(matches!(
            wrong_length.validate(usize::MAX),
            Err(RecordValidationError::FactColumnLength { .. })
        ));

        let duplicate = RecordBatch {
            rows: 1,
            facts: vec![
                fact(0, ColumnValues::Bool(vec![Some(true)])),
                fact(0, ColumnValues::Bool(vec![Some(false)])),
            ],
            metadata: Vec::new(),
        };
        assert_eq!(
            duplicate.validate(usize::MAX),
            Err(RecordValidationError::DuplicateFactHandle(FactHandle::new(
                0
            )))
        );

        let non_finite = RecordBatch {
            rows: 1,
            facts: vec![fact(0, ColumnValues::F64(vec![Some(f64::NAN)]))],
            metadata: Vec::new(),
        };
        assert!(matches!(
            non_finite.validate(usize::MAX),
            Err(RecordValidationError::InvalidFact {
                source: ColumnValueError::NonFiniteFloat { row: 0 },
                ..
            })
        ));
    }

    #[test]
    fn enforces_deterministic_charge_limit() {
        let batch = RecordBatch {
            rows: 1,
            facts: Vec::new(),
            metadata: vec![metadata(0, ColumnValues::String(vec![Some("plex".into())]))],
        };
        let charged = batch.charged_bytes().unwrap();
        assert_eq!(
            batch.validate(charged - 1),
            Err(RecordValidationError::PayloadTooLarge {
                actual: charged,
                maximum: charged - 1,
            })
        );
    }
}
