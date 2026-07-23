use serde::{Deserialize, Serialize};

use super::types::Operation;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MechanicId(pub String);

impl MechanicId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for MechanicId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for MechanicId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MechanicKind {
    Guarantee,
    Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StandardMechanic {
    pub id: &'static str,
    pub kind: MechanicKind,
    pub method: Option<&'static str>,
    pub request_schema: Option<&'static str>,
    pub feedback_schema: Option<&'static str>,
    pub operations: &'static [Operation],
}

pub const STANDARD_MECHANICS: &[StandardMechanic] = &[
    StandardMechanic {
        id: "schedule.atomic-enqueue@1",
        kind: MechanicKind::Guarantee,
        method: None,
        request_schema: None,
        feedback_schema: None,
        operations: &[Operation::Schedule],
    },
    StandardMechanic {
        id: "request.cancel@1",
        kind: MechanicKind::Action,
        method: Some("pie.request.cancel@1"),
        request_schema: Some("../../schema/0.6/actions/request-cancel.schema.json"),
        feedback_schema: Some("../../schema/0.6/actions/action-feedback.schema.json"),
        operations: &[Operation::Schedule, Operation::Feedback],
    },
    StandardMechanic {
        id: "group.cancel@1",
        kind: MechanicKind::Action,
        method: Some("pie.group.cancel@1"),
        request_schema: Some("../../schema/0.6/actions/group-cancel.schema.json"),
        feedback_schema: Some("../../schema/0.6/actions/action-feedback.schema.json"),
        operations: &[Operation::Schedule, Operation::Feedback],
    },
    StandardMechanic {
        id: "cache.prefetch@1",
        kind: MechanicKind::Action,
        method: Some("pie.cache.prefetch@1"),
        request_schema: Some("../../schema/0.6/actions/cache-prefetch.schema.json"),
        feedback_schema: Some("../../schema/0.6/actions/action-feedback.schema.json"),
        operations: &[Operation::Cache, Operation::Schedule],
    },
    StandardMechanic {
        id: "cache.swap@1",
        kind: MechanicKind::Action,
        method: Some("pie.cache.swap@1"),
        request_schema: Some("../../schema/0.6/actions/cache-swap.schema.json"),
        feedback_schema: Some("../../schema/0.6/actions/action-feedback.schema.json"),
        operations: &[Operation::Cache],
    },
    StandardMechanic {
        id: "request.rebalance@1",
        kind: MechanicKind::Action,
        method: Some("pie.request.rebalance@1"),
        request_schema: Some("../../schema/0.6/actions/request-rebalance.schema.json"),
        feedback_schema: Some("../../schema/0.6/actions/action-feedback.schema.json"),
        operations: &[Operation::Route, Operation::Schedule, Operation::Feedback],
    },
];

pub fn standard_mechanic(id: &str) -> Option<&'static StandardMechanic> {
    STANDARD_MECHANICS.iter().find(|mechanic| mechanic.id == id)
}

pub(crate) fn valid_versioned_name(value: &str) -> bool {
    let Some((name, version)) = value.rsplit_once('@') else {
        return false;
    };
    if name.is_empty()
        || name.len() > 128
        || !name.bytes().enumerate().all(|(index, byte)| match byte {
            b'a'..=b'z' => true,
            b'0'..=b'9' | b'.' | b'_' | b'-' => index > 0,
            _ => false,
        })
    {
        return false;
    }
    !version.is_empty()
        && !version.starts_with('0')
        && version.bytes().all(|byte| byte.is_ascii_digit())
}
