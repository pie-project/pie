use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pie_plex::{Document, Operation, validate_global_scope, validate_state_envelope};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::protocol::{referenced_request_ids, validate_input};

const GENERATION_LOCAL_FACTS: [&str; 1] = ["current_target"];

#[derive(Clone)]
pub struct PolicyStateStore {
    inner: Arc<Mutex<PolicyState>>,
}

#[derive(Clone)]
struct PolicyState {
    global: Document,
    requests: BTreeMap<String, Document>,
    feedback_deliveries: BTreeMap<String, Document>,
}

impl Default for PolicyStateStore {
    fn default() -> Self {
        Self::new(json!({})).expect("empty global facts are valid")
    }
}

impl PolicyStateStore {
    pub fn new(global_facts: Document) -> Result<Self, StateStoreError> {
        require_object(&global_facts, "global facts")?;
        let global = json!({
            "facts": global_facts,
            "fields": {},
            "scratch": {}
        });
        validate_global_scope(&global)
            .map_err(|error| StateStoreError::InvalidScope(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(PolicyState {
                global,
                requests: BTreeMap::new(),
                feedback_deliveries: BTreeMap::new(),
            })),
        })
    }

    pub fn read_global(&self) -> Document {
        self.inner.lock().unwrap().global.clone()
    }

    pub fn reset_global(&self) {
        let mut state = self.inner.lock().unwrap();
        state.global["fields"] = json!({});
        state.global["scratch"] = json!({});
    }

    pub fn replace_global_facts(&self, facts: Document) -> Result<(), StateStoreError> {
        require_object(&facts, "global facts")?;
        self.inner.lock().unwrap().global["facts"] = facts;
        Ok(())
    }

    pub fn merge_global_facts(&self, facts: Document) -> Result<(), StateStoreError> {
        let facts = require_object(&facts, "global facts")?;
        let mut state = self.inner.lock().unwrap();
        let current = state.global["facts"]
            .as_object_mut()
            .expect("global facts are canonical");
        merge(current, facts);
        Ok(())
    }

    pub fn create_request(
        &self,
        logical_request_id: impl Into<String>,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateStoreError> {
        let logical_request_id = logical_request_id.into();
        if logical_request_id.is_empty() {
            return Err(StateStoreError::EmptyLogicalRequestId);
        }
        require_object(&body, "request body")?;
        require_object(&metadata, "request metadata")?;
        let request = json!({
            "facts": {
                "logical_request_id": logical_request_id,
                "generation_id": 0
            },
            "fields": {
                "body": body,
                "metadata": metadata
            },
            "scratch": {}
        });
        let id = request["facts"]["logical_request_id"]
            .as_str()
            .expect("constructed request ID")
            .to_owned();
        let mut state = self.inner.lock().unwrap();
        if state.requests.contains_key(&id) {
            return Err(StateStoreError::AlreadyExists(id));
        }
        state.requests.insert(id, request.clone());
        Ok(request)
    }

    pub fn continue_request(
        &self,
        logical_request_id: &str,
        body: Document,
        continuation_metadata: Document,
    ) -> Result<Document, StateStoreError> {
        require_object(&body, "request body")?;
        let continuation_metadata = require_object(&continuation_metadata, "request metadata")?;
        let mut state = self.inner.lock().unwrap();
        let previous = state
            .requests
            .get(logical_request_id)
            .cloned()
            .ok_or_else(|| StateStoreError::NotFound(logical_request_id.to_owned()))?;
        let generation = previous
            .pointer("/facts/generation_id")
            .and_then(Value::as_u64)
            .and_then(|generation| generation.checked_add(1))
            .ok_or(StateStoreError::GenerationExhausted)?;
        let mut request = previous;
        let facts = request["facts"]
            .as_object_mut()
            .expect("canonical request facts");
        facts.insert("generation_id".into(), json!(generation));
        for key in GENERATION_LOCAL_FACTS {
            facts.remove(key);
        }
        let fields = request["fields"]
            .as_object_mut()
            .expect("canonical request fields");
        fields.insert("body".into(), body);
        let metadata = fields
            .entry("metadata")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or(StateStoreError::FieldNotObject("request fields.metadata"))?;
        merge(metadata, continuation_metadata);
        state
            .requests
            .insert(logical_request_id.to_owned(), request.clone());
        Ok(request)
    }

    pub fn read_request(&self, logical_request_id: &str) -> Result<Document, StateStoreError> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .get(logical_request_id)
            .cloned()
            .ok_or_else(|| StateStoreError::NotFound(logical_request_id.to_owned()))
    }

    pub fn remove_request(&self, logical_request_id: &str) -> Result<Document, StateStoreError> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .remove(logical_request_id)
            .ok_or_else(|| StateStoreError::NotFound(logical_request_id.to_owned()))
    }

    pub fn request_count(&self) -> usize {
        self.inner.lock().unwrap().requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.request_count() == 0
    }

    pub fn merge_request_facts(
        &self,
        logical_request_id: &str,
        facts: Document,
    ) -> Result<(), StateStoreError> {
        let facts = require_object(&facts, "request facts")?;
        let mut state = self.inner.lock().unwrap();
        let request = state
            .requests
            .get_mut(logical_request_id)
            .ok_or_else(|| StateStoreError::NotFound(logical_request_id.to_owned()))?;
        let current = request["facts"]
            .as_object_mut()
            .expect("canonical request facts");
        for identity in ["logical_request_id", "generation_id"] {
            if let Some(value) = facts.get(identity)
                && current.get(identity) != Some(value)
            {
                return Err(StateStoreError::HostIdentityMutation(identity));
            }
        }
        merge(current, facts);
        Ok(())
    }

    pub fn record_enacted_placement(
        &self,
        logical_request_id: &str,
        target_id: impl Into<String>,
    ) -> Result<(), StateStoreError> {
        let target_id = target_id.into();
        if target_id.is_empty() {
            return Err(StateStoreError::EmptyTargetId);
        }
        self.merge_request_facts(logical_request_id, json!({"previous_target": target_id}))
    }

    pub(crate) fn hydrate(
        &self,
        operation: Operation,
        mut input: Document,
    ) -> Result<Document, StateStoreError> {
        let object = input
            .as_object_mut()
            .ok_or(StateStoreError::InputNotObject)?;
        if object.contains_key("global") || object.contains_key("requests") {
            return Err(StateStoreError::StateAlreadyProvided);
        }
        let request_ids = referenced_request_ids(operation, &Value::Object(object.clone()))
            .map_err(|error| StateStoreError::InvalidInput(error.to_string()))?;
        let state = self.inner.lock().unwrap();
        let mut requests = Map::new();
        for logical_request_id in request_ids {
            let request = state
                .requests
                .get(&logical_request_id)
                .cloned()
                .ok_or_else(|| StateStoreError::NotFound(logical_request_id.clone()))?;
            requests.insert(logical_request_id, request);
        }
        object.insert("global".into(), state.global.clone());
        object.insert("requests".into(), Value::Object(requests));
        drop(state);
        validate_input(operation, &input)
            .map_err(|error| StateStoreError::InvalidInput(error.to_string()))?;
        Ok(input)
    }

    pub(crate) fn commit_policy_mutation(
        &self,
        original: &Document,
        mutated: &Document,
        feedback: Option<(&str, &Document, usize)>,
        terminal_logical_ids: &[String],
    ) -> Result<(), StateStoreError> {
        validate_state_envelope(original)
            .map_err(|error| StateStoreError::InvalidScope(error.to_string()))?;
        validate_state_envelope(mutated)
            .map_err(|error| StateStoreError::InvalidScope(error.to_string()))?;
        let original_requests = original["requests"]
            .as_object()
            .expect("validated original requests");
        let mutated_requests = mutated["requests"]
            .as_object()
            .expect("validated mutated requests");

        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        for logical_request_id in terminal_logical_ids {
            if !next.requests.contains_key(logical_request_id) {
                return Err(StateStoreError::NotFound(logical_request_id.clone()));
            }
        }
        if next.global != original["global"] {
            return Err(StateStoreError::StaleState("global".into()));
        }
        next.global["fields"] = mutated["global"]["fields"].clone();
        next.global["scratch"] = mutated["global"]["scratch"].clone();

        for (logical_request_id, original_request) in original_requests {
            let current = next
                .requests
                .get_mut(logical_request_id)
                .ok_or_else(|| StateStoreError::NotFound(logical_request_id.clone()))?;
            if *current != *original_request {
                return Err(StateStoreError::StaleState(logical_request_id.clone()));
            }
            let mutated_request = &mutated_requests[logical_request_id];
            current["fields"] = mutated_request["fields"].clone();
            current["scratch"] = mutated_request["scratch"].clone();
        }

        if let Some((delivery_id, result, maximum)) = feedback {
            if next.feedback_deliveries.contains_key(delivery_id) {
                return Err(StateStoreError::DuplicateFeedback(delivery_id.to_owned()));
            }
            if next.feedback_deliveries.len() >= maximum {
                return Err(StateStoreError::FeedbackLedgerFull(maximum));
            }
            next.feedback_deliveries
                .insert(delivery_id.to_owned(), result.clone());
        }
        for logical_request_id in terminal_logical_ids {
            next.requests.remove(logical_request_id);
        }
        *state = next;
        Ok(())
    }

    pub(crate) fn feedback_result(&self, delivery_id: &str) -> Option<Document> {
        self.inner
            .lock()
            .unwrap()
            .feedback_deliveries
            .get(delivery_id)
            .cloned()
    }
}

fn merge(target: &mut Map<String, Value>, source: &Map<String, Value>) {
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

fn require_object<'a>(
    value: &'a Document,
    field: &'static str,
) -> Result<&'a Map<String, Value>, StateStoreError> {
    value
        .as_object()
        .ok_or(StateStoreError::FieldNotObject(field))
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateStoreError {
    #[error("logical request ID must not be empty")]
    EmptyLogicalRequestId,
    #[error("{0} must be a JSON object")]
    FieldNotObject(&'static str),
    #[error("logical request {0} already exists")]
    AlreadyExists(String),
    #[error("logical request {0} was not found")]
    NotFound(String),
    #[error("logical request generation counter exhausted")]
    GenerationExhausted,
    #[error("host fact update cannot change {0}")]
    HostIdentityMutation(&'static str),
    #[error("target ID must not be empty")]
    EmptyTargetId,
    #[error("operation input must be a JSON object")]
    InputNotObject,
    #[error("operation input must not provide global or requests state")]
    StateAlreadyProvided,
    #[error("operation input is invalid: {0}")]
    InvalidInput(String),
    #[error("state scope is invalid: {0}")]
    InvalidScope(String),
    #[error("canonical state changed while invoking policy for {0}")]
    StaleState(String),
    #[error("feedback delivery {0} is already committed")]
    DuplicateFeedback(String),
    #[error("feedback delivery ledger reached its limit of {0}")]
    FeedbackLedgerFull(usize),
}

#[cfg(test)]
mod tests {
    use pie_plex::Operation;
    use serde_json::json;

    use super::*;

    #[test]
    fn continuation_preserves_scopes_and_updates_host_facts() {
        let store = PolicyStateStore::new(json!({"config": {"weight": 2}})).unwrap();
        let mut request = store
            .create_request(
                "L",
                json!({"prompt": "first"}),
                json!({"keep": 1, "replace": 1}),
            )
            .unwrap();
        request["scratch"]["served"] = json!(8);
        request["fields"]["custom"] = json!("preserved");
        store
            .commit_policy_mutation(
                &json!({
                    "global": store.read_global(),
                    "requests": {"L": store.read_request("L").unwrap()}
                }),
                &json!({
                    "global": store.read_global(),
                    "requests": {"L": request}
                }),
                None,
                &[],
            )
            .unwrap();
        store.record_enacted_placement("L", "node-a").unwrap();
        store
            .merge_request_facts("L", json!({"current_target": "node-a"}))
            .unwrap();

        let continuation = store
            .continue_request(
                "L",
                json!({"prompt": "second"}),
                json!({"replace": 2, "new": 3}),
            )
            .unwrap();
        assert_eq!(continuation["facts"]["generation_id"], 1);
        assert_eq!(continuation["facts"]["previous_target"], "node-a");
        assert!(continuation["facts"].get("current_target").is_none());
        assert_eq!(
            continuation["fields"]["metadata"],
            json!({"keep": 1, "replace": 2, "new": 3})
        );
        assert_eq!(continuation["fields"]["custom"], "preserved");
        assert_eq!(continuation["scratch"]["served"], 8);
    }

    #[test]
    fn hydration_exposes_each_referenced_request_once() {
        let store = PolicyStateStore::default();
        store.create_request("L", json!({}), json!({})).unwrap();
        let input = store
            .hydrate(
                Operation::Feedback,
                json!({
                    "delivery_id": "d",
                    "records": [
                        {"event": "progress", "request_id": "L", "facts": {}},
                        {"event": "tool-boundary", "request_id": "L", "facts": {}}
                    ],
                    "context": {"capabilities": {}}
                }),
            )
            .unwrap();
        assert_eq!(input["requests"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn host_reset_and_fact_ownership_are_explicit() {
        let store = PolicyStateStore::default();
        store.create_request("L", json!({}), json!({})).unwrap();
        assert!(matches!(
            store.merge_request_facts("L", json!({"generation_id": 9})),
            Err(StateStoreError::HostIdentityMutation("generation_id"))
        ));
        store.merge_global_facts(json!({"model": "m"})).unwrap();
        assert_eq!(store.read_global()["facts"]["model"], "m");
        store.reset_global();
        assert_eq!(store.read_global()["fields"], json!({}));
        assert_eq!(store.read_global()["scratch"], json!({}));
        assert_eq!(store.read_global()["facts"]["model"], "m");
    }
}
