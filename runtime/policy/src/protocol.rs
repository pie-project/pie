use std::collections::BTreeSet;

use pie_plex::{Document, Operation, validate_state_envelope};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonResponse {
    pub input: Document,
    pub result: Document,
    #[serde(default)]
    pub duplicate_feedback: bool,
}

impl JsonResponse {
    pub(crate) fn duplicate_feedback(input: Document, result: Document) -> Self {
        Self {
            input,
            result,
            duplicate_feedback: true,
        }
    }
}

pub(crate) fn parse_response(
    output: &str,
    original: &Document,
    operation: Operation,
) -> Result<JsonResponse, ProtocolError> {
    let response: Value = serde_json::from_str(output).map_err(ProtocolError::MalformedResponse)?;
    let response = response
        .as_object()
        .ok_or(ProtocolError::ResponseNotObject)?;
    let mutated = response
        .get("input")
        .cloned()
        .ok_or(ProtocolError::MissingResponseField("input"))?;
    let result = response
        .get("result")
        .cloned()
        .ok_or(ProtocolError::MissingResponseField("result"))?;
    if !result.is_object() {
        return Err(ProtocolError::ResultNotObject);
    }
    let input = validate_mutation(operation, original, mutated)?;
    Ok(JsonResponse {
        input,
        result,
        duplicate_feedback: false,
    })
}

pub(crate) fn validate_input(operation: Operation, input: &Document) -> Result<(), ProtocolError> {
    let referenced = referenced_request_ids(operation, input)?;
    validate_state_envelope(input)
        .map_err(|error| ProtocolError::InvalidState(error.to_string()))?;
    let exposed = input["requests"]
        .as_object()
        .expect("validated requests")
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if referenced != exposed {
        return Err(ProtocolError::RequestSetMismatch {
            referenced,
            exposed,
        });
    }
    Ok(())
}

pub(crate) fn referenced_request_ids(
    operation: Operation,
    input: &Document,
) -> Result<BTreeSet<String>, ProtocolError> {
    let object = input.as_object().ok_or(ProtocolError::InputNotObject)?;
    let mut request_ids = BTreeSet::new();
    match operation {
        Operation::Route => {
            cause(object)?;
            request_ids.insert(request_id(object.get("request_id"), "request_id")?);
            context(object)?;
            for candidate in array(object, "candidates")? {
                let candidate = json_object(Some(candidate), "candidates[]")?;
                string(candidate.get("id"), "candidates[].id")?;
                json_object(candidate.get("facts"), "candidates[].facts")?;
            }
        }
        Operation::Admit => {
            cause(object)?;
            request_ids.insert(request_id(object.get("request_id"), "request_id")?);
            context(object)?;
            let target = json_object(object.get("target"), "target")?;
            string(target.get("id"), "target.id")?;
            json_object(target.get("facts"), "target.facts")?;
        }
        Operation::Schedule => {
            cause(object)?;
            context(object)?;
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
            context(object)?;
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
            context(object)?;
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

fn validate_mutation(
    operation: Operation,
    original: &Document,
    mutated: Document,
) -> Result<Document, ProtocolError> {
    validate_input(operation, original)?;
    validate_input(operation, &mutated)?;
    let original = original.as_object().expect("validated object");
    let mutated_object = mutated
        .as_object()
        .ok_or(ProtocolError::MutatedInputNotObject)?;

    if original.keys().collect::<BTreeSet<_>>() != mutated_object.keys().collect::<BTreeSet<_>>() {
        return Err(ProtocolError::ReadOnlyContextChanged);
    }
    for (field, value) in original {
        if !matches!(field.as_str(), "global" | "requests")
            && mutated_object.get(field) != Some(value)
        {
            return Err(ProtocolError::ReadOnlyContextChanged);
        }
    }

    if original["global"]["facts"] != mutated_object["global"]["facts"] {
        return Err(ProtocolError::GlobalFactsChanged);
    }
    let original_requests = original["requests"]
        .as_object()
        .expect("validated requests");
    let mutated_requests = mutated_object["requests"]
        .as_object()
        .expect("validated requests");
    if original_requests.keys().collect::<BTreeSet<_>>()
        != mutated_requests.keys().collect::<BTreeSet<_>>()
    {
        return Err(ProtocolError::RequestMapChanged);
    }
    for (logical_request_id, request) in original_requests {
        if request["facts"] != mutated_requests[logical_request_id]["facts"] {
            return Err(ProtocolError::RequestFactsChanged(
                logical_request_id.clone(),
            ));
        }
    }
    Ok(mutated)
}

fn cause(object: &Map<String, Value>) -> Result<(), ProtocolError> {
    string(object.get("cause"), "cause").map(|_| ())
}

fn context(object: &Map<String, Value>) -> Result<(), ProtocolError> {
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
    #[error("policy input must be a top-level JSON object")]
    InputNotObject,
    #[error("policy response is malformed JSON")]
    MalformedResponse(#[source] serde_json::Error),
    #[error("policy response must be a top-level JSON object")]
    ResponseNotObject,
    #[error("policy response is missing {0}")]
    MissingResponseField(&'static str),
    #[error("policy result must be a JSON object")]
    ResultNotObject,
    #[error("policy returned input must be a JSON object")]
    MutatedInputNotObject,
    #[error("missing or invalid field {0}")]
    MissingField(&'static str),
    #[error("request ID in {0} must not be empty")]
    EmptyRequestId(&'static str),
    #[error("feedback delivery ID must not be empty")]
    EmptyDeliveryId,
    #[error("invalid global/request state: {0}")]
    InvalidState(String),
    #[error("referenced request IDs {referenced:?} do not match exposed request IDs {exposed:?}")]
    RequestSetMismatch {
        referenced: BTreeSet<String>,
        exposed: BTreeSet<String>,
    },
    #[error("policy changed read-only invocation context")]
    ReadOnlyContextChanged,
    #[error("policy changed host-owned global facts")]
    GlobalFactsChanged,
    #[error("policy added or removed a request map")]
    RequestMapChanged,
    #[error("policy changed host-owned facts for request {0}")]
    RequestFactsChanged(String),
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

    fn route_input() -> Value {
        json!({
            "global": {"facts": {}, "fields": {}, "scratch": {}},
            "requests": {"L": request("L")},
            "cause": "generation-arrival",
            "request_id": "L",
            "candidates": [{"id": "a", "facts": {"queue_depth": 1}}],
            "context": {"capabilities": {}}
        })
    }

    #[test]
    fn commits_only_fields_and_scratch() {
        let original = route_input();
        let mut mutated = original.clone();
        mutated["global"]["scratch"]["routes"] = json!(1);
        mutated["requests"]["L"]["fields"]["body"]["prompt"] = json!("rewritten");
        mutated["requests"]["L"]["scratch"]["routed"] = json!(true);
        let response = json!({"input": mutated, "result": {"scores": [1.0]}});
        let response = parse_response(&response.to_string(), &original, Operation::Route).unwrap();
        assert_eq!(response.input["global"]["scratch"]["routes"], 1);
        assert_eq!(
            response.input["requests"]["L"]["fields"]["body"]["prompt"],
            "rewritten"
        );
    }

    #[test]
    fn rejects_fact_context_and_request_set_mutation() {
        let original = route_input();

        let mut facts = original.clone();
        facts["requests"]["L"]["facts"]["generation_id"] = json!(1);
        assert!(matches!(
            validate_mutation(Operation::Route, &original, facts),
            Err(ProtocolError::RequestFactsChanged(_))
        ));

        let mut candidate = original.clone();
        candidate["candidates"][0]["facts"]["queue_depth"] = json!(999);
        assert!(matches!(
            validate_mutation(Operation::Route, &original, candidate),
            Err(ProtocolError::ReadOnlyContextChanged)
        ));

        let mut requests = original.clone();
        requests["requests"]["M"] = request("M");
        assert!(matches!(
            validate_mutation(Operation::Route, &original, requests),
            Err(ProtocolError::RequestSetMismatch { .. })
        ));
    }

    #[test]
    fn feedback_references_one_shared_request_map() {
        let input = json!({
            "global": {"facts": {}, "fields": {}, "scratch": {}},
            "requests": {"L": request("L")},
            "delivery_id": "d",
            "records": [
                {"event": "progress", "request_id": "L", "facts": {}},
                {"event": "tool-boundary", "request_id": "L", "facts": {}}
            ],
            "context": {"capabilities": {}}
        });
        validate_input(Operation::Feedback, &input).unwrap();
        assert_eq!(
            referenced_request_ids(Operation::Feedback, &input).unwrap(),
            BTreeSet::from(["L".to_owned()])
        );
    }
}
