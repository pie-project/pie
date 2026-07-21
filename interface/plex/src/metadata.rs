use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::capability::{DependencyRequirement, Symbol};
use crate::operation::Operation;
use crate::value::ValueType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Provenance {
    Fact,
    Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MetadataScope {
    LogicalRequest,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldLocation {
    Request,
    Candidate,
    Feedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FieldUse {
    pub operation: Operation,
    pub location: FieldLocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactDeclaration {
    pub name: Symbol,
    pub value_type: ValueType,
    pub requirement: DependencyRequirement,
    pub max_value_bytes: u32,
    pub uses: BTreeSet<FieldUse>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataDeclaration {
    pub name: Symbol,
    pub value_type: ValueType,
    pub scope: MetadataScope,
    pub requirement: DependencyRequirement,
    pub max_value_bytes: u32,
    pub uses: BTreeSet<FieldUse>,
}
