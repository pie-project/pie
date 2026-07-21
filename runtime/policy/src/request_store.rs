use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pie_plex::{Document, Operation, validate_request};
use serde_json::{Map, Value, json};
use thiserror::Error;

#[derive(Clone, Default)]
pub struct CanonicalRequestStore {
    inner: Arc<Mutex<BTreeMap<String, Document>>>,
}

impl CanonicalRequestStore {
    pub fn create(
        &self,
        logical_request_id: impl Into<String>,
        body: Document,
        metadata: Document,
    ) -> Result<Document, RequestStoreError> {
        let logical_request_id = logical_request_id.into();
        if logical_request_id.is_empty() {
            return Err(RequestStoreError::EmptyLogicalRequestId);
        }
        require_object(&body, "body")?;
        require_object(&metadata, "metadata")?;
        let request = json!({
            "identity": {
                "logical_request_id": logical_request_id,
                "generation_id": 0
            },
            "body": body,
            "metadata": metadata,
            "state": {}
        });
        let mut requests = self.inner.lock().unwrap();
        let id = logical_id(&request)?.to_owned();
        if requests.insert(id.clone(), request.clone()).is_some() {
            return Err(RequestStoreError::AlreadyExists(id));
        }
        Ok(request)
    }

    pub fn continuation(
        &self,
        logical_request_id: &str,
        body: Document,
        continuation_metadata: Document,
    ) -> Result<Document, RequestStoreError> {
        require_object(&body, "body")?;
        let continuation_metadata = continuation_metadata
            .as_object()
            .ok_or(RequestStoreError::FieldNotObject("metadata"))?;
        let mut requests = self.inner.lock().unwrap();
        let previous = requests
            .get(logical_request_id)
            .cloned()
            .ok_or_else(|| RequestStoreError::NotFound(logical_request_id.to_owned()))?;
        let generation = previous
            .pointer("/identity/generation_id")
            .and_then(Value::as_u64)
            .and_then(|generation| generation.checked_add(1))
            .ok_or(RequestStoreError::GenerationExhausted)?;
        let mut metadata = previous["metadata"]
            .as_object()
            .cloned()
            .ok_or(RequestStoreError::FieldNotObject("metadata"))?;
        for (key, value) in continuation_metadata {
            metadata.insert(key.clone(), value.clone());
        }
        let request = json!({
            "identity": {
                "logical_request_id": logical_request_id,
                "generation_id": generation
            },
            "body": body,
            "metadata": metadata,
            "state": previous["state"].clone()
        });
        requests.insert(logical_request_id.to_owned(), request.clone());
        Ok(request)
    }

    pub fn get(&self, logical_request_id: &str) -> Result<Document, RequestStoreError> {
        self.inner
            .lock()
            .unwrap()
            .get(logical_request_id)
            .cloned()
            .ok_or_else(|| RequestStoreError::NotFound(logical_request_id.to_owned()))
    }

    pub fn apply_operation(
        &self,
        operation: Operation,
        validated_input: &Document,
    ) -> Result<(), RequestStoreError> {
        let documents = operation_requests(operation, validated_input)?;
        let mut requests = self.inner.lock().unwrap();
        let mut next = requests.clone();
        for request in documents {
            validate_request(request)
                .map_err(|error| RequestStoreError::InvalidRequest(error.to_string()))?;
            let id = logical_id(request)?.to_owned();
            let current = next
                .get(&id)
                .ok_or_else(|| RequestStoreError::NotFound(id.clone()))?;
            if current.get("identity") != request.get("identity") {
                return Err(RequestStoreError::StaleIdentity(id));
            }
            next.insert(id, request.clone());
        }
        *requests = next;
        Ok(())
    }

    pub fn remove(&self, logical_request_id: &str) -> Result<Document, RequestStoreError> {
        self.inner
            .lock()
            .unwrap()
            .remove(logical_request_id)
            .ok_or_else(|| RequestStoreError::NotFound(logical_request_id.to_owned()))
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub(crate) fn operation_requests(
    operation: Operation,
    input: &Document,
) -> Result<Vec<&Document>, RequestStoreError> {
    match operation {
        Operation::Route | Operation::Admit => Ok(vec![
            input
                .get("request")
                .ok_or(RequestStoreError::MissingRequest)?,
        ]),
        Operation::Schedule => input
            .get("runnable")
            .and_then(Value::as_array)
            .ok_or(RequestStoreError::MissingRequest)?
            .iter()
            .map(|candidate| {
                candidate
                    .get("request")
                    .ok_or(RequestStoreError::MissingRequest)
            })
            .collect(),
        Operation::Evict => input
            .get("resident")
            .and_then(Value::as_array)
            .ok_or(RequestStoreError::MissingRequest)?
            .iter()
            .filter_map(|unit| match unit.get("request") {
                Some(Value::Null) | None => None,
                Some(request) => Some(Ok(request)),
            })
            .collect(),
        Operation::Feedback => input
            .get("records")
            .and_then(Value::as_array)
            .ok_or(RequestStoreError::MissingRequest)?
            .iter()
            .map(|record| {
                record
                    .get("request")
                    .ok_or(RequestStoreError::MissingRequest)
            })
            .collect(),
    }
}

fn logical_id(request: &Document) -> Result<&str, RequestStoreError> {
    request
        .pointer("/identity/logical_request_id")
        .and_then(Value::as_str)
        .ok_or(RequestStoreError::InvalidIdentity)
}

fn require_object<'a>(
    value: &'a Document,
    field: &'static str,
) -> Result<&'a Map<String, Value>, RequestStoreError> {
    value
        .as_object()
        .ok_or(RequestStoreError::FieldNotObject(field))
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RequestStoreError {
    #[error("logical request ID must not be empty")]
    EmptyLogicalRequestId,
    #[error("request field {0} must be a JSON object")]
    FieldNotObject(&'static str),
    #[error("logical request {0} already exists")]
    AlreadyExists(String),
    #[error("logical request {0} was not found")]
    NotFound(String),
    #[error("logical request generation counter exhausted")]
    GenerationExhausted,
    #[error("request identity is invalid")]
    InvalidIdentity,
    #[error("request mutation has stale identity for {0}")]
    StaleIdentity(String),
    #[error("operation input is missing an associated request")]
    MissingRequest,
    #[error("request is invalid: {0}")]
    InvalidRequest(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_preserves_state_and_shallow_merges_metadata() {
        let store = CanonicalRequestStore::default();
        let mut request = store
            .create(
                "L",
                json!({"prompt": "first"}),
                json!({"keep": 1, "replace": 1}),
            )
            .unwrap();
        request["state"]["served"] = json!(8);
        store
            .apply_operation(
                Operation::Route,
                &json!({
                    "request": request,
                    "cause": "generation-arrival",
                    "candidates": [],
                    "context": {"config": {}}
                }),
            )
            .unwrap();
        let continuation = store
            .continuation(
                "L",
                json!({"prompt": "second"}),
                json!({"replace": 2, "new": 3}),
            )
            .unwrap();
        assert_eq!(continuation["identity"]["generation_id"], 1);
        assert_eq!(
            continuation["metadata"],
            json!({"keep": 1, "replace": 2, "new": 3})
        );
        assert_eq!(continuation["state"]["served"], 8);
    }
}
