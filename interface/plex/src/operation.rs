use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Operation {
    Route,
    Admit,
    Schedule,
    Evict,
    Feedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdmissionDecision {
    Accept,
    Defer,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedService {
    pub candidate_index: usize,
    pub token_budget: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedEviction {
    pub candidate_index: usize,
    pub size_bytes: u64,
}

const SCOPE_NAMESPACES: [&str; 3] = ["facts", "fields", "scratch"];

pub fn validate_global_scope(global: &Value) -> Result<(), DecisionValidationError> {
    validate_scope(global, "global").map(|_| ())
}

pub fn validate_request_scope(
    logical_request_id: &str,
    request: &Value,
) -> Result<(), DecisionValidationError> {
    let object = validate_scope(request, "request")?;
    let facts = object["facts"]
        .as_object()
        .expect("validated request facts");
    if facts.get("logical_request_id").and_then(Value::as_str) != Some(logical_request_id)
        || logical_request_id.is_empty()
        || facts.get("generation_id").and_then(Value::as_u64).is_none()
    {
        return Err(DecisionValidationError::InvalidRequestIdentity);
    }
    Ok(())
}

pub fn validate_state_envelope(input: &Value) -> Result<(), DecisionValidationError> {
    let input = input
        .as_object()
        .ok_or(DecisionValidationError::EnvelopeNotObject)?;
    validate_global_scope(
        input
            .get("global")
            .ok_or(DecisionValidationError::MissingField("global"))?,
    )?;
    let requests = input
        .get("requests")
        .and_then(Value::as_object)
        .ok_or(DecisionValidationError::RequestsNotObject)?;
    for (logical_request_id, request) in requests {
        validate_request_scope(logical_request_id, request)?;
    }
    Ok(())
}

fn validate_scope<'a>(
    scope: &'a Value,
    name: &'static str,
) -> Result<&'a serde_json::Map<String, Value>, DecisionValidationError> {
    let object = scope
        .as_object()
        .ok_or(DecisionValidationError::ScopeNotObject(name))?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = SCOPE_NAMESPACES.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(DecisionValidationError::InvalidScopeNamespaces(name));
    }
    for namespace in SCOPE_NAMESPACES {
        if !object.get(namespace).is_some_and(Value::is_object) {
            return Err(DecisionValidationError::NamespaceNotObject {
                scope: name,
                namespace,
            });
        }
    }
    Ok(object)
}

pub fn validate_admit(result: &Value) -> Result<AdmissionDecision, DecisionValidationError> {
    let decision = result
        .as_object()
        .and_then(|result| result.get("decision"))
        .and_then(Value::as_str)
        .ok_or(DecisionValidationError::MissingField("decision"))?;
    match decision {
        "accept" => Ok(AdmissionDecision::Accept),
        "defer" => Ok(AdmissionDecision::Defer),
        "reject" => Ok(AdmissionDecision::Reject),
        _ => Err(DecisionValidationError::InvalidAdmissionDecision),
    }
}

pub fn rank_route(
    result: &Value,
    candidate_count: usize,
) -> Result<Vec<usize>, DecisionValidationError> {
    let scores = scores(result, "scores", candidate_count)?;
    Ok(stable_order(&scores, true))
}

pub fn select_schedule(
    input: &Value,
    result: &Value,
) -> Result<Vec<SelectedService>, DecisionValidationError> {
    let runnable = input
        .get("runnable")
        .and_then(Value::as_array)
        .ok_or(DecisionValidationError::MissingField("runnable"))?;
    let capacity = input
        .get("capacity")
        .and_then(Value::as_object)
        .ok_or(DecisionValidationError::MissingField("capacity"))?;
    let max_selected = usize_value(capacity.get("max_selected"), "capacity.max_selected")?;
    let max_total_tokens = u32_value(
        capacity.get("max_total_tokens"),
        "capacity.max_total_tokens",
    )?;
    let host_max = u32_value(
        capacity.get("max_token_budget"),
        "capacity.max_token_budget",
    )?;
    let token_budget_capability = input
        .pointer("/context/capabilities/token_budget")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let decisions = result
        .get("decisions")
        .and_then(Value::as_array)
        .ok_or(DecisionValidationError::MissingField("decisions"))?;
    if decisions.len() != runnable.len() {
        return Err(DecisionValidationError::DenseLength {
            expected: runnable.len(),
            actual: decisions.len(),
        });
    }

    let mut scores = Vec::with_capacity(decisions.len());
    let mut budgets = Vec::with_capacity(decisions.len());
    for (index, (decision, candidate)) in decisions.iter().zip(runnable).enumerate() {
        let decision = decision
            .as_object()
            .ok_or(DecisionValidationError::DecisionNotObject(index))?;
        scores.push(finite_number(decision.get("score"), "decisions[].score")?);
        let candidate_max = u32_value(
            candidate.get("max_token_budget"),
            "runnable[].max_token_budget",
        )?
        .min(host_max);
        let budget = match decision.get("token_budget") {
            None | Some(Value::Null) => candidate_max,
            Some(value) => {
                if !token_budget_capability {
                    return Err(DecisionValidationError::TokenBudgetUnsupported(index));
                }
                let budget = u32_value(Some(value), "decisions[].token_budget")?;
                if budget > candidate_max {
                    return Err(DecisionValidationError::TokenBudgetTooLarge {
                        index,
                        actual: budget,
                        maximum: candidate_max,
                    });
                }
                budget
            }
        };
        budgets.push(budget);
    }

    let mut remaining = max_total_tokens;
    let mut selected = Vec::with_capacity(max_selected.min(runnable.len()));
    for index in stable_order(&scores, true) {
        if selected.len() == max_selected || remaining == 0 {
            break;
        }
        let budget = budgets[index].min(remaining);
        if budget == 0 {
            continue;
        }
        selected.push(SelectedService {
            candidate_index: index,
            token_budget: budget,
        });
        remaining -= budget;
    }
    Ok(selected)
}

pub fn select_evictions(
    input: &Value,
    result: &Value,
) -> Result<Vec<SelectedEviction>, DecisionValidationError> {
    let resident = input
        .get("resident")
        .and_then(Value::as_array)
        .ok_or(DecisionValidationError::MissingField("resident"))?;
    let bytes_needed = input
        .get("bytes_needed")
        .and_then(Value::as_u64)
        .ok_or(DecisionValidationError::MissingField("bytes_needed"))?;
    let scores = scores(result, "scores", resident.len())?;
    let mut freed = 0u64;
    let mut selected = Vec::new();
    for index in stable_order(&scores, false) {
        if freed >= bytes_needed {
            break;
        }
        let size_bytes = resident[index]
            .get("size_bytes")
            .and_then(Value::as_u64)
            .ok_or(DecisionValidationError::MissingField(
                "resident[].size_bytes",
            ))?;
        freed = freed.saturating_add(size_bytes);
        selected.push(SelectedEviction {
            candidate_index: index,
            size_bytes,
        });
    }
    Ok(selected)
}

fn scores(
    result: &Value,
    field: &'static str,
    expected: usize,
) -> Result<Vec<f64>, DecisionValidationError> {
    let values = result
        .get(field)
        .and_then(Value::as_array)
        .ok_or(DecisionValidationError::MissingField(field))?;
    if values.len() != expected {
        return Err(DecisionValidationError::DenseLength {
            expected,
            actual: values.len(),
        });
    }
    values
        .iter()
        .map(|value| finite_number(Some(value), field))
        .collect()
}

fn finite_number(
    value: Option<&Value>,
    field: &'static str,
) -> Result<f64, DecisionValidationError> {
    let value = value
        .and_then(Value::as_f64)
        .ok_or(DecisionValidationError::InvalidNumber(field))?;
    if !value.is_finite() {
        return Err(DecisionValidationError::NonFiniteScore);
    }
    Ok(value)
}

fn usize_value(
    value: Option<&Value>,
    field: &'static str,
) -> Result<usize, DecisionValidationError> {
    let value = value
        .and_then(Value::as_u64)
        .ok_or(DecisionValidationError::InvalidInteger(field))?;
    usize::try_from(value).map_err(|_| DecisionValidationError::InvalidInteger(field))
}

fn u32_value(value: Option<&Value>, field: &'static str) -> Result<u32, DecisionValidationError> {
    let value = value
        .and_then(Value::as_u64)
        .ok_or(DecisionValidationError::InvalidInteger(field))?;
    u32::try_from(value).map_err(|_| DecisionValidationError::InvalidInteger(field))
}

fn stable_order(scores: &[f64], descending: bool) -> Vec<usize> {
    let mut order: Vec<_> = (0..scores.len()).collect();
    order.sort_by(|left, right| {
        let ordering = if descending {
            scores[*right].partial_cmp(&scores[*left])
        } else {
            scores[*left].partial_cmp(&scores[*right])
        };
        ordering
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DecisionValidationError {
    #[error("operation envelope must be a JSON object")]
    EnvelopeNotObject,
    #[error("{0} scope must be a JSON object")]
    ScopeNotObject(&'static str),
    #[error("{0} scope must contain exactly facts, fields, and scratch")]
    InvalidScopeNamespaces(&'static str),
    #[error("{scope}.{namespace} must be a JSON object")]
    NamespaceNotObject {
        scope: &'static str,
        namespace: &'static str,
    },
    #[error("requests must be a JSON object")]
    RequestsNotObject,
    #[error(
        "request-map key must equal non-empty facts.logical_request_id and generation_id must be unsigned"
    )]
    InvalidRequestIdentity,
    #[error("missing or invalid JSON field {0}")]
    MissingField(&'static str),
    #[error("JSON field {0} must be a finite number")]
    InvalidNumber(&'static str),
    #[error("JSON field {0} must be an in-range unsigned integer")]
    InvalidInteger(&'static str),
    #[error("dense result contains {actual} entries; expected {expected}")]
    DenseLength { expected: usize, actual: usize },
    #[error("score must be finite")]
    NonFiniteScore,
    #[error("schedule decision {0} must be a JSON object")]
    DecisionNotObject(usize),
    #[error("schedule decision {0} returned a token budget without capability")]
    TokenBudgetUnsupported(usize),
    #[error("schedule decision {index} returned budget {actual}; maximum is {maximum}")]
    TokenBudgetTooLarge {
        index: usize,
        actual: u32,
        maximum: u32,
    },
    #[error("admission decision must be accept, defer, or reject")]
    InvalidAdmissionDecision,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn validates_global_and_request_scopes() {
        let global = json!({"facts": {}, "fields": {}, "scratch": {}});
        let request = json!({
            "facts": {"logical_request_id": "L", "generation_id": 0},
            "fields": {"body": {}, "metadata": {}},
            "scratch": {}
        });
        validate_global_scope(&global).unwrap();
        validate_request_scope("L", &request).unwrap();
        validate_state_envelope(&json!({
            "global": global,
            "requests": {"L": request}
        }))
        .unwrap();
        assert!(validate_request_scope("M", &request).is_err());
        assert!(
            validate_global_scope(&json!({"facts": {}, "fields": {}, "scratch": {}, "extra": {}}))
                .is_err()
        );
    }

    #[test]
    fn route_is_descending_and_stable() {
        assert_eq!(
            rank_route(&json!({"scores": [1.0, 3.0, 3.0]}), 3).unwrap(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn schedule_fill_enforces_capacity_and_budgets() {
        let input = json!({
            "runnable": [
                {"max_token_budget": 8},
                {"max_token_budget": 8}
            ],
            "capacity": {
                "max_selected": 2,
                "max_total_tokens": 9,
                "max_token_budget": 8
            },
            "context": {"capabilities": {"token_budget": true}}
        });
        let selected = select_schedule(
            &input,
            &json!({"decisions": [
                {"score": 2.0, "token_budget": 8},
                {"score": 1.0}
            ]}),
        )
        .unwrap();
        assert_eq!(
            selected,
            vec![
                SelectedService {
                    candidate_index: 0,
                    token_budget: 8
                },
                SelectedService {
                    candidate_index: 1,
                    token_budget: 1
                }
            ]
        );
    }

    #[test]
    fn eviction_is_low_retention_first() {
        let selected = select_evictions(
            &json!({
                "bytes_needed": 6,
                "resident": [
                    {"size_bytes": 4},
                    {"size_bytes": 3},
                    {"size_bytes": 8}
                ]
            }),
            &json!({"scores": [2.0, 1.0, 3.0]}),
        )
        .unwrap();
        assert_eq!(
            selected,
            vec![
                SelectedEviction {
                    candidate_index: 1,
                    size_bytes: 3
                },
                SelectedEviction {
                    candidate_index: 0,
                    size_bytes: 4
                }
            ]
        );
    }
}
