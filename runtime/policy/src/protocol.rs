use std::collections::{BTreeMap, BTreeSet};

use pie_plex::{Document, Operation};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::state_store::{RequestStateUpdate, StateSnapshot, StateUpdates};

pub(crate) fn parse_result(output: &str) -> Result<Document, ProtocolError> {
    let result: Document = serde_json::from_str(output).map_err(ProtocolError::MalformedResult)?;
    let object = result.as_object().ok_or(ProtocolError::ResultNotObject)?;
    if object.contains_key("input") && object.contains_key("result") {
        return Err(ProtocolError::LegacyResponseWrapper);
    }
    Ok(result)
}

pub(crate) fn parse_state_updates(
    updates_json: &str,
    snapshot: &StateSnapshot,
) -> Result<StateUpdates, ProtocolError> {
    let updates: Document =
        serde_json::from_str(updates_json).map_err(ProtocolError::MalformedStateUpdates)?;
    let updates = updates
        .as_object()
        .ok_or(ProtocolError::StateUpdatesNotObject)?;
    for key in updates.keys() {
        if !matches!(key.as_str(), "shared" | "requests") {
            return Err(ProtocolError::InvalidStateUpdateNamespace(key.clone()));
        }
    }

    let shared = updates
        .get("shared")
        .map(|shared| {
            if !shared.is_object() {
                return Err(ProtocolError::SharedUpdateNotObject);
            }
            Ok(shared.clone())
        })
        .transpose()?;

    let mut requests = BTreeMap::new();
    if let Some(request_updates) = updates.get("requests") {
        let request_updates = request_updates
            .as_object()
            .ok_or(ProtocolError::RequestUpdatesNotObject)?;
        for (logical_request_id, request_update) in request_updates {
            if !snapshot.requests.contains_key(logical_request_id) {
                return Err(ProtocolError::UnknownRequestUpdate(
                    logical_request_id.clone(),
                ));
            }
            let request_update = request_update
                .as_object()
                .ok_or_else(|| ProtocolError::RequestUpdateNotObject(logical_request_id.clone()))?;
            let actual = request_update
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            let expected = ["fields", "scratch"].into_iter().collect::<BTreeSet<_>>();
            if actual != expected {
                return Err(ProtocolError::InvalidRequestUpdateNamespaces(
                    logical_request_id.clone(),
                ));
            }
            let fields = request_update["fields"]
                .as_object()
                .ok_or_else(|| ProtocolError::RequestFieldsNotObject(logical_request_id.clone()))?;
            let scratch = request_update["scratch"].as_object().ok_or_else(|| {
                ProtocolError::RequestScratchNotObject(logical_request_id.clone())
            })?;
            requests.insert(
                logical_request_id.clone(),
                RequestStateUpdate {
                    fields: Value::Object(fields.clone()),
                    scratch: Value::Object(scratch.clone()),
                },
            );
        }
    }
    Ok(StateUpdates { shared, requests })
}

pub(crate) fn validate_context(
    operation: Operation,
    context: &Document,
) -> Result<BTreeSet<String>, ProtocolError> {
    let object = context.as_object().ok_or(ProtocolError::ContextNotObject)?;
    if object.contains_key("shared") || object.contains_key("requests") {
        return Err(ProtocolError::PersistentStateInContext);
    }
    referenced_request_ids(operation, context)
}

pub(crate) fn referenced_request_ids(
    operation: Operation,
    context: &Document,
) -> Result<BTreeSet<String>, ProtocolError> {
    let object = context.as_object().ok_or(ProtocolError::ContextNotObject)?;
    let mut request_ids = BTreeSet::new();
    match operation {
        Operation::Route => {
            cause(object)?;
            request_ids.insert(request_id(object.get("request_id"), "request_id")?);
            hook_context(object)?;
            for candidate in array(object, "candidates")? {
                let candidate = json_object(Some(candidate), "candidates[]")?;
                string(candidate.get("id"), "candidates[].id")?;
                json_object(candidate.get("facts"), "candidates[].facts")?;
            }
        }
        Operation::Admit => {
            cause(object)?;
            request_ids.insert(request_id(object.get("request_id"), "request_id")?);
            hook_context(object)?;
            let target = json_object(object.get("target"), "target")?;
            string(target.get("id"), "target.id")?;
            json_object(target.get("facts"), "target.facts")?;
        }
        Operation::Schedule => {
            cause(object)?;
            hook_context(object)?;
            for candidate in array(object, "runnable")? {
                let candidate = json_object(Some(candidate), "runnable[]")?;
                request_ids.insert(request_id(
                    candidate.get("request_id"),
                    "runnable[].request_id",
                )?);
                json_object(candidate.get("facts"), "runnable[].facts")?;
                unsigned(
                    candidate.get("max_token_budget"),
                    "runnable[].max_token_budget",
                )?;
            }
            let capacity = json_object(object.get("capacity"), "capacity")?;
            for field in ["max_selected", "max_total_tokens", "max_token_budget"] {
                unsigned(capacity.get(field), "capacity[]")?;
            }
        }
        Operation::Evict => {
            cause(object)?;
            hook_context(object)?;
            unsigned(object.get("bytes_needed"), "bytes_needed")?;
            for unit in array(object, "resident")? {
                let unit = json_object(Some(unit), "resident[]")?;
                string(unit.get("id"), "resident[].id")?;
                unsigned(unit.get("size_bytes"), "resident[].size_bytes")?;
                json_object(unit.get("facts"), "resident[].facts")?;
                match unit.get("request_id") {
                    Some(Value::Null) => {}
                    value => {
                        request_ids.insert(request_id(value, "resident[].request_id")?);
                    }
                }
            }
        }
        Operation::Feedback => {
            hook_context(object)?;
            let delivery = string(object.get("delivery_id"), "delivery_id")?;
            if delivery.is_empty() {
                return Err(ProtocolError::EmptyDeliveryId);
            }
            for record in array(object, "records")? {
                let record = json_object(Some(record), "records[]")?;
                string(record.get("event"), "records[].event")?;
                request_ids.insert(request_id(
                    record.get("request_id"),
                    "records[].request_id",
                )?);
                json_object(record.get("facts"), "records[].facts")?;
            }
        }
    }
    Ok(request_ids)
}

fn cause(object: &Map<String, Value>) -> Result<(), ProtocolError> {
    string(object.get("cause"), "cause").map(|_| ())
}

fn hook_context(object: &Map<String, Value>) -> Result<(), ProtocolError> {
    let context = json_object(object.get("context"), "context")?;
    json_object(context.get("capabilities"), "context.capabilities")?;
    Ok(())
}

fn request_id(value: Option<&Value>, field: &'static str) -> Result<String, ProtocolError> {
    let value = string(value, field)?;
    if value.is_empty() {
        return Err(ProtocolError::EmptyRequestId(field));
    }
    Ok(value.to_owned())
}

fn array<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a Vec<Value>, ProtocolError> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or(ProtocolError::MissingField(field))
}

fn json_object<'a>(
    value: Option<&'a Value>,
    field: &'static str,
) -> Result<&'a Map<String, Value>, ProtocolError> {
    value
        .and_then(Value::as_object)
        .ok_or(ProtocolError::MissingField(field))
}

fn string<'a>(value: Option<&'a Value>, field: &'static str) -> Result<&'a str, ProtocolError> {
    value
        .and_then(Value::as_str)
        .ok_or(ProtocolError::MissingField(field))
}

fn unsigned(value: Option<&Value>, field: &'static str) -> Result<u64, ProtocolError> {
    value
        .and_then(Value::as_u64)
        .ok_or(ProtocolError::MissingField(field))
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("policy context must be a top-level JSON object")]
    ContextNotObject,
    #[error("policy context must not contain shared or requests state")]
    PersistentStateInContext,
    #[error("policy result is malformed JSON")]
    MalformedResult(#[source] serde_json::Error),
    #[error("policy result must be a top-level JSON object")]
    ResultNotObject,
    #[error("policy result must not use the removed input/result response wrapper")]
    LegacyResponseWrapper,
    #[error("state updates are malformed JSON")]
    MalformedStateUpdates(#[source] serde_json::Error),
    #[error("state updates must be a top-level JSON object")]
    StateUpdatesNotObject,
    #[error("state updates contain unsupported namespace {0}")]
    InvalidStateUpdateNamespace(String),
    #[error("shared state update must be a JSON object")]
    SharedUpdateNotObject,
    #[error("request updates must be a JSON object")]
    RequestUpdatesNotObject,
    #[error("state update references request {0} outside the invocation working set")]
    UnknownRequestUpdate(String),
    #[error("state update for request {0} must be a JSON object")]
    RequestUpdateNotObject(String),
    #[error("state update for request {0} must contain exactly fields and scratch")]
    InvalidRequestUpdateNamespaces(String),
    #[error("fields update for request {0} must be a JSON object")]
    RequestFieldsNotObject(String),
    #[error("scratch update for request {0} must be a JSON object")]
    RequestScratchNotObject(String),
    #[error("missing or invalid field {0}")]
    MissingField(&'static str),
    #[error("request ID in {0} must not be empty")]
    EmptyRequestId(&'static str),
    #[error("feedback delivery ID must not be empty")]
    EmptyDeliveryId,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(id: &str) -> Value {
        json!({
            "facts": {"logical_request_id": id, "generation_id": 0},
            "fields": {"body": {"prompt": "hello"}, "metadata": {}},
            "scratch": {}
        })
    }

    fn snapshot() -> StateSnapshot {
        StateSnapshot {
            shared: json!({}),
            requests: BTreeMap::from([("L".into(), request("L"))]),
            shared_revision: 0,
            request_revisions: BTreeMap::from([("L".into(), 0)]),
        }
    }

    #[test]
    fn validates_transient_context_and_unique_working_set() {
        let context = json!({
            "delivery_id": "d",
            "records": [
                {"event": "progress", "request_id": "L", "facts": {}},
                {"event": "tool-boundary", "request_id": "L", "facts": {}}
            ],
            "context": {"capabilities": {}}
        });
        assert_eq!(
            validate_context(Operation::Feedback, &context).unwrap(),
            BTreeSet::from(["L".to_owned()])
        );
        let mut invalid = context;
        invalid["shared"] = json!({});
        assert!(matches!(
            validate_context(Operation::Feedback, &invalid),
            Err(ProtocolError::PersistentStateInContext)
        ));
    }

    #[test]
    fn parses_only_mutable_working_set_updates() {
        let updates = parse_state_updates(
            r#"{
                "shared":{"calls":1},
                "requests":{"L":{"fields":{"body":{},"metadata":{}},"scratch":{"calls":1}}}
            }"#,
            &snapshot(),
        )
        .unwrap();
        assert_eq!(updates.shared, Some(json!({"calls": 1})));
        assert_eq!(updates.requests["L"].scratch["calls"], 1);

        assert!(matches!(
            parse_state_updates(
                r#"{"requests":{"L":{"facts":{},"fields":{},"scratch":{}}}}"#,
                &snapshot()
            ),
            Err(ProtocolError::InvalidRequestUpdateNamespaces(_))
        ));
        assert!(matches!(
            parse_state_updates(
                r#"{"requests":{"M":{"fields":{},"scratch":{}}}}"#,
                &snapshot()
            ),
            Err(ProtocolError::UnknownRequestUpdate(_))
        ));
    }

    #[test]
    fn parses_result_without_input_wrapper() {
        assert_eq!(
            parse_result(r#"{"decision":"accept"}"#).unwrap(),
            json!({"decision": "accept"})
        );
        assert!(matches!(
            parse_result(r#"{"input":{},"result":{}}"#),
            Err(ProtocolError::LegacyResponseWrapper)
        ));
        assert!(matches!(
            parse_result("[]"),
            Err(ProtocolError::ResultNotObject)
        ));
    }
}
