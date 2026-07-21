use pie_plex::{Document, Operation, validate_request};
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
    pub(crate) fn unchanged_feedback(input: Document, result: Document) -> Self {
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
    let object = input.as_object().ok_or(ProtocolError::InputNotObject)?;
    match operation {
        Operation::Route => {
            cause(object)?;
            request(object.get("request"))?;
            context(object)?;
            let candidates = array(object, "candidates")?;
            for candidate in candidates {
                candidate_object(candidate, "candidates[]")?;
                string(candidate.get("id"), "candidates[].id")?;
                json_object(candidate.get("facts"), "candidates[].facts")?;
            }
        }
        Operation::Admit => {
            cause(object)?;
            request(object.get("request"))?;
            context(object)?;
            let target = object
                .get("target")
                .and_then(Value::as_object)
                .ok_or(ProtocolError::MissingField("target"))?;
            string(target.get("id"), "target.id")?;
            json_object(target.get("facts"), "target.facts")?;
        }
        Operation::Schedule => {
            cause(object)?;
            context(object)?;
            let runnable = array(object, "runnable")?;
            for candidate in runnable {
                candidate_object(candidate, "runnable[]")?;
                request(candidate.get("request"))?;
                json_object(candidate.get("facts"), "runnable[].facts")?;
                unsigned(
                    candidate.get("max_token_budget"),
                    "runnable[].max_token_budget",
                )?;
            }
            let capacity = object
                .get("capacity")
                .and_then(Value::as_object)
                .ok_or(ProtocolError::MissingField("capacity"))?;
            for field in ["max_selected", "max_total_tokens", "max_token_budget"] {
                unsigned(capacity.get(field), "capacity[]")?;
            }
        }
        Operation::Evict => {
            cause(object)?;
            context(object)?;
            unsigned(object.get("bytes_needed"), "bytes_needed")?;
            let resident = array(object, "resident")?;
            for unit in resident {
                candidate_object(unit, "resident[]")?;
                string(unit.get("id"), "resident[].id")?;
                unsigned(unit.get("size_bytes"), "resident[].size_bytes")?;
                json_object(unit.get("facts"), "resident[].facts")?;
                match unit.get("request") {
                    Some(Value::Null) => {}
                    value => request(value)?,
                }
            }
        }
        Operation::Feedback => {
            context(object)?;
            let delivery = string(object.get("delivery_id"), "delivery_id")?;
            if delivery.is_empty() {
                return Err(ProtocolError::EmptyDeliveryId);
            }
            let records = array(object, "records")?;
            for record in records {
                candidate_object(record, "records[]")?;
                string(record.get("event"), "records[].event")?;
                request(record.get("request"))?;
                json_object(record.get("facts"), "records[].facts")?;
            }
        }
    }
    Ok(())
}

fn validate_mutation(
    operation: Operation,
    original: &Document,
    mutated: Document,
) -> Result<Document, ProtocolError> {
    validate_input(operation, original)?;
    let original = original.as_object().expect("validated object");
    let mut mutated = mutated
        .as_object()
        .cloned()
        .ok_or(ProtocolError::MutatedInputNotObject)?;

    match operation {
        Operation::Route => {
            let request = mutated_request(original.get("request"), mutated.get("request"))?;
            validate_candidate_ids(
                array(original, "candidates")?,
                mutated
                    .get("candidates")
                    .and_then(Value::as_array)
                    .ok_or(ProtocolError::CandidateListChanged)?,
                "id",
            )?;
            mutated.insert("request".into(), request);
            restore(&mut mutated, original, "cause");
            restore(&mut mutated, original, "candidates");
            restore(&mut mutated, original, "context");
        }
        Operation::Admit => {
            let request = mutated_request(original.get("request"), mutated.get("request"))?;
            let original_target = original["target"].as_object().expect("validated target");
            let mutated_target = mutated
                .get("target")
                .and_then(Value::as_object)
                .ok_or(ProtocolError::CandidateListChanged)?;
            if original_target.get("id") != mutated_target.get("id") {
                return Err(ProtocolError::CandidateIdentityChanged);
            }
            mutated.insert("request".into(), request);
            restore(&mut mutated, original, "cause");
            restore(&mut mutated, original, "target");
            restore(&mut mutated, original, "context");
        }
        Operation::Schedule => {
            let original_runnable = array(original, "runnable")?;
            let mutated_runnable = mutated
                .get("runnable")
                .and_then(Value::as_array)
                .ok_or(ProtocolError::CandidateListChanged)?;
            if original_runnable.len() != mutated_runnable.len() {
                return Err(ProtocolError::CandidateListChanged);
            }
            let mut cleaned = Vec::with_capacity(original_runnable.len());
            for (original_candidate, mutated_candidate) in
                original_runnable.iter().zip(mutated_runnable)
            {
                let original_candidate = candidate_object(original_candidate, "runnable[]")?;
                let mutated_candidate = candidate_object(mutated_candidate, "runnable[]")?;
                if original_candidate.get("max_token_budget")
                    != mutated_candidate.get("max_token_budget")
                {
                    return Err(ProtocolError::CandidateIdentityChanged);
                }
                let request = mutated_request(
                    original_candidate.get("request"),
                    mutated_candidate.get("request"),
                )?;
                let mut candidate = original_candidate.clone();
                candidate.insert("request".into(), request);
                cleaned.push(Value::Object(candidate));
            }
            mutated.insert("runnable".into(), Value::Array(cleaned));
            restore(&mut mutated, original, "cause");
            restore(&mut mutated, original, "capacity");
            restore(&mut mutated, original, "context");
        }
        Operation::Evict => {
            let original_resident = array(original, "resident")?;
            let mutated_resident = mutated
                .get("resident")
                .and_then(Value::as_array)
                .ok_or(ProtocolError::CandidateListChanged)?;
            validate_candidate_ids(original_resident, mutated_resident, "id")?;
            let mut cleaned = Vec::with_capacity(original_resident.len());
            for (original_unit, mutated_unit) in original_resident.iter().zip(mutated_resident) {
                let original_unit = candidate_object(original_unit, "resident[]")?;
                let mutated_unit = candidate_object(mutated_unit, "resident[]")?;
                if original_unit.get("size_bytes") != mutated_unit.get("size_bytes") {
                    return Err(ProtocolError::CandidateIdentityChanged);
                }
                let mut unit = original_unit.clone();
                if !original_unit.get("request").is_some_and(Value::is_null) {
                    let request =
                        mutated_request(original_unit.get("request"), mutated_unit.get("request"))?;
                    unit.insert("request".into(), request);
                }
                cleaned.push(Value::Object(unit));
            }
            mutated.insert("resident".into(), Value::Array(cleaned));
            restore(&mut mutated, original, "cause");
            restore(&mut mutated, original, "bytes_needed");
            restore(&mut mutated, original, "context");
        }
        Operation::Feedback => {
            if original.get("delivery_id") != mutated.get("delivery_id") {
                return Err(ProtocolError::DeliveryIdChanged);
            }
            let original_records = array(original, "records")?;
            let mutated_records = mutated
                .get("records")
                .and_then(Value::as_array)
                .ok_or(ProtocolError::RecordListChanged)?;
            if original_records.len() != mutated_records.len() {
                return Err(ProtocolError::RecordListChanged);
            }
            let mut cleaned = Vec::with_capacity(original_records.len());
            for (original_record, mutated_record) in original_records.iter().zip(mutated_records) {
                let original_record = candidate_object(original_record, "records[]")?;
                let mutated_record = candidate_object(mutated_record, "records[]")?;
                if original_record.get("event") != mutated_record.get("event") {
                    return Err(ProtocolError::RecordListChanged);
                }
                let request = mutated_request(
                    original_record.get("request"),
                    mutated_record.get("request"),
                )?;
                let mut record = original_record.clone();
                record.insert("request".into(), request);
                cleaned.push(Value::Object(record));
            }
            mutated.insert("records".into(), Value::Array(cleaned));
            restore(&mut mutated, original, "delivery_id");
            restore(&mut mutated, original, "context");
        }
    }

    let cleaned = Value::Object(mutated);
    validate_input(operation, &cleaned)?;
    Ok(cleaned)
}

fn mutated_request(
    original: Option<&Value>,
    mutated: Option<&Value>,
) -> Result<Value, ProtocolError> {
    let original = original.ok_or(ProtocolError::MissingField("request"))?;
    let mutated = mutated.ok_or(ProtocolError::MissingField("request"))?;
    validate_request(original).map_err(|error| ProtocolError::InvalidRequest(error.to_string()))?;
    validate_request(mutated).map_err(|error| ProtocolError::InvalidRequest(error.to_string()))?;
    if original.get("identity") != mutated.get("identity") {
        return Err(ProtocolError::IdentityChanged);
    }
    Ok(mutated.clone())
}

fn validate_candidate_ids(
    original: &[Value],
    mutated: &[Value],
    field: &'static str,
) -> Result<(), ProtocolError> {
    if original.len() != mutated.len() {
        return Err(ProtocolError::CandidateListChanged);
    }
    for (original, mutated) in original.iter().zip(mutated) {
        if original.get(field) != mutated.get(field) {
            return Err(ProtocolError::CandidateIdentityChanged);
        }
    }
    Ok(())
}

fn restore(mutated: &mut Map<String, Value>, original: &Map<String, Value>, field: &str) {
    if let Some(value) = original.get(field) {
        mutated.insert(field.to_owned(), value.clone());
    }
}

fn cause(object: &Map<String, Value>) -> Result<(), ProtocolError> {
    string(object.get("cause"), "cause").map(|_| ())
}

fn context(object: &Map<String, Value>) -> Result<(), ProtocolError> {
    let context = object
        .get("context")
        .and_then(Value::as_object)
        .ok_or(ProtocolError::MissingField("context"))?;
    json_object(context.get("config"), "context.config")?;
    Ok(())
}

fn request(value: Option<&Value>) -> Result<(), ProtocolError> {
    let value = value.ok_or(ProtocolError::MissingField("request"))?;
    validate_request(value).map_err(|error| ProtocolError::InvalidRequest(error.to_string()))
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

fn candidate_object<'a>(
    value: &'a Value,
    field: &'static str,
) -> Result<&'a Map<String, Value>, ProtocolError> {
    value.as_object().ok_or(ProtocolError::MissingField(field))
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
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("policy changed host-owned request identity")]
    IdentityChanged,
    #[error("policy changed candidate count or order")]
    CandidateListChanged,
    #[error("policy changed a host-owned candidate identity")]
    CandidateIdentityChanged,
    #[error("policy changed feedback record count or event order")]
    RecordListChanged,
    #[error("policy changed host-owned feedback delivery ID")]
    DeliveryIdChanged,
    #[error("feedback delivery ID must not be empty")]
    EmptyDeliveryId,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(id: &str, generation: u64) -> Value {
        json!({
            "identity": {
                "logical_request_id": id,
                "generation_id": generation
            },
            "body": {"prompt": "hello"},
            "metadata": {},
            "state": {}
        })
    }

    #[test]
    fn preserves_request_mutation_but_restores_route_candidates() {
        let original = json!({
            "cause": "generation-arrival",
            "request": request("L", 0),
            "candidates": [{"id": "a", "facts": {"queue_depth": 1}}],
            "context": {"config": {}}
        });
        let mut mutated = original.clone();
        mutated["request"]["state"]["routed"] = json!(true);
        mutated["candidates"][0]["facts"]["queue_depth"] = json!(999);
        let response = json!({"input": mutated, "result": {"scores": [1.0]}});
        let response = parse_response(&response.to_string(), &original, Operation::Route).unwrap();
        assert_eq!(response.input["request"]["state"]["routed"], true);
        assert_eq!(response.input["candidates"][0]["facts"]["queue_depth"], 1);
    }

    #[test]
    fn rejects_identity_and_candidate_mutation() {
        let original = json!({
            "cause": "generation-arrival",
            "request": request("L", 0),
            "candidates": [{"id": "a", "facts": {}}],
            "context": {"config": {}}
        });
        let mut identity = original.clone();
        identity["request"]["identity"]["generation_id"] = json!(1);
        assert!(matches!(
            validate_mutation(Operation::Route, &original, identity),
            Err(ProtocolError::IdentityChanged)
        ));
        let mut candidate = original.clone();
        candidate["candidates"][0]["id"] = json!("b");
        assert!(matches!(
            validate_mutation(Operation::Route, &original, candidate),
            Err(ProtocolError::CandidateIdentityChanged)
        ));
    }
}
