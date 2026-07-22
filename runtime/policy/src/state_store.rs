use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use pie_plex::{Document, validate_request_scope};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

const GENERATION_LOCAL_FACTS: [&str; 1] = ["current_target"];

pub trait PolicyStateBackend: Send + Sync + 'static {
    fn load(&self, request_ids: &BTreeSet<String>) -> Result<StateSnapshot, StateBackendError>;

    fn commit(
        &self,
        snapshot: &StateSnapshot,
        updates: &StateUpdates,
        feedback: Option<&FeedbackCommit>,
        terminal_requests: &[String],
    ) -> Result<(), StateBackendError>;

    fn create_request(
        &self,
        logical_request_id: String,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateBackendError>;

    fn continue_request(
        &self,
        logical_request_id: &str,
        body: Document,
        continuation_metadata: Document,
    ) -> Result<Document, StateBackendError>;

    fn read_shared(&self) -> Result<Document, StateBackendError>;

    fn replace_shared(&self, shared: Document) -> Result<(), StateBackendError>;

    fn read_request(&self, logical_request_id: &str) -> Result<Document, StateBackendError>;

    fn remove_request(&self, logical_request_id: &str) -> Result<Document, StateBackendError>;

    fn request_count(&self) -> Result<usize, StateBackendError>;

    fn merge_request_facts(
        &self,
        logical_request_id: &str,
        facts: Document,
    ) -> Result<(), StateBackendError>;

    fn replace_request_fields(
        &self,
        logical_request_id: &str,
        fields: Document,
    ) -> Result<(), StateBackendError>;

    fn record_enacted_placement(
        &self,
        logical_request_id: &str,
        target_id: String,
    ) -> Result<(), StateBackendError>;

    fn feedback_result(&self, delivery_id: &str) -> Result<Option<Document>, StateBackendError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshot {
    pub shared: Document,
    pub requests: BTreeMap<String, Document>,
    pub(crate) shared_revision: u64,
    pub(crate) request_revisions: BTreeMap<String, u64>,
}

impl StateSnapshot {
    pub fn from_parts(
        shared: Document,
        requests: BTreeMap<String, Document>,
        shared_revision: u64,
        request_revisions: BTreeMap<String, u64>,
    ) -> Result<Self, StateBackendError> {
        let snapshot = Self {
            shared,
            requests,
            shared_revision,
            request_revisions,
        };
        validate_snapshot(&snapshot)?;
        Ok(snapshot)
    }

    pub fn document(&self) -> Document {
        let requests = self
            .requests
            .iter()
            .map(|(id, request)| (id.clone(), request.clone()))
            .collect::<Map<_, _>>();
        json!({
            "shared": self.shared,
            "requests": requests
        })
    }

    pub fn shared_revision(&self) -> u64 {
        self.shared_revision
    }

    pub fn request_revisions(&self) -> &BTreeMap<String, u64> {
        &self.request_revisions
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StateUpdates {
    pub shared: Option<Document>,
    pub requests: BTreeMap<String, RequestStateUpdate>,
}

impl StateUpdates {
    pub fn is_empty(&self) -> bool {
        self.shared.is_none() && self.requests.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestStateUpdate {
    pub fields: Document,
    pub scratch: Document,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackCommit {
    pub delivery_id: String,
    pub result: Document,
    pub maximum_deliveries: usize,
}

#[derive(Clone)]
pub struct InMemoryPolicyStateBackend {
    inner: Arc<Mutex<PolicyState>>,
}

#[derive(Clone)]
struct PolicyState {
    shared: VersionedDocument,
    requests: BTreeMap<String, VersionedDocument>,
    feedback_deliveries: BTreeMap<String, Document>,
}

#[derive(Clone)]
struct VersionedDocument {
    value: Document,
    revision: u64,
}

impl Default for InMemoryPolicyStateBackend {
    fn default() -> Self {
        Self::new(json!({})).expect("empty shared state is valid")
    }
}

impl InMemoryPolicyStateBackend {
    pub fn new(shared: Document) -> Result<Self, StateBackendError> {
        require_object(&shared, "shared")?;
        Ok(Self {
            inner: Arc::new(Mutex::new(PolicyState {
                shared: VersionedDocument {
                    value: shared,
                    revision: 0,
                },
                requests: BTreeMap::new(),
                feedback_deliveries: BTreeMap::new(),
            })),
        })
    }

    pub fn shared_backend(self) -> Arc<dyn PolicyStateBackend> {
        Arc::new(self)
    }
}

impl PolicyStateBackend for InMemoryPolicyStateBackend {
    fn load(&self, request_ids: &BTreeSet<String>) -> Result<StateSnapshot, StateBackendError> {
        let state = self.inner.lock().unwrap();
        let mut requests = BTreeMap::new();
        let mut request_revisions = BTreeMap::new();
        for logical_request_id in request_ids {
            let request = state
                .requests
                .get(logical_request_id)
                .ok_or_else(|| StateBackendError::NotFound(logical_request_id.clone()))?;
            requests.insert(logical_request_id.clone(), request.value.clone());
            request_revisions.insert(logical_request_id.clone(), request.revision);
        }
        Ok(StateSnapshot {
            shared: state.shared.value.clone(),
            requests,
            shared_revision: state.shared.revision,
            request_revisions,
        })
    }

    fn commit(
        &self,
        snapshot: &StateSnapshot,
        updates: &StateUpdates,
        feedback: Option<&FeedbackCommit>,
        terminal_requests: &[String],
    ) -> Result<(), StateBackendError> {
        validate_snapshot(snapshot)?;
        validate_updates(snapshot, updates)?;
        let terminal = terminal_requests.iter().collect::<BTreeSet<_>>();
        if terminal.len() != terminal_requests.len() {
            return Err(StateBackendError::DuplicateTerminalRequest);
        }
        for logical_request_id in terminal_requests {
            if !snapshot.requests.contains_key(logical_request_id) {
                return Err(StateBackendError::UnknownRequestUpdate(
                    logical_request_id.clone(),
                ));
            }
        }

        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        if next.shared.revision != snapshot.shared_revision {
            return Err(StateBackendError::RevisionConflict("shared".into()));
        }
        for (logical_request_id, expected_revision) in &snapshot.request_revisions {
            let current = next
                .requests
                .get(logical_request_id)
                .ok_or_else(|| StateBackendError::RevisionConflict(logical_request_id.clone()))?;
            if current.revision != *expected_revision {
                return Err(StateBackendError::RevisionConflict(
                    logical_request_id.clone(),
                ));
            }
        }

        if let Some(shared) = &updates.shared {
            next.shared.value = shared.clone();
            next.shared.revision = next_revision(next.shared.revision)?;
        }
        for (logical_request_id, update) in &updates.requests {
            let request = next
                .requests
                .get_mut(logical_request_id)
                .ok_or_else(|| StateBackendError::NotFound(logical_request_id.clone()))?;
            request.value["fields"] = update.fields.clone();
            request.value["scratch"] = update.scratch.clone();
            request.revision = next_revision(request.revision)?;
        }

        if let Some(feedback) = feedback {
            if feedback.delivery_id.is_empty() {
                return Err(StateBackendError::EmptyFeedbackDeliveryId);
            }
            if next.feedback_deliveries.contains_key(&feedback.delivery_id) {
                return Err(StateBackendError::DuplicateFeedback(
                    feedback.delivery_id.clone(),
                ));
            }
            if next.feedback_deliveries.len() >= feedback.maximum_deliveries {
                return Err(StateBackendError::FeedbackLedgerFull(
                    feedback.maximum_deliveries,
                ));
            }
            next.feedback_deliveries
                .insert(feedback.delivery_id.clone(), feedback.result.clone());
        }
        for logical_request_id in terminal_requests {
            next.requests.remove(logical_request_id);
        }
        *state = next;
        Ok(())
    }

    fn create_request(
        &self,
        logical_request_id: String,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateBackendError> {
        if logical_request_id.is_empty() {
            return Err(StateBackendError::EmptyLogicalRequestId);
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
            return Err(StateBackendError::AlreadyExists(id));
        }
        state.requests.insert(
            id,
            VersionedDocument {
                value: request.clone(),
                revision: 0,
            },
        );
        Ok(request)
    }

    fn continue_request(
        &self,
        logical_request_id: &str,
        body: Document,
        continuation_metadata: Document,
    ) -> Result<Document, StateBackendError> {
        require_object(&body, "request body")?;
        let continuation_metadata = require_object(&continuation_metadata, "request metadata")?;
        let mut state = self.inner.lock().unwrap();
        let stored = state
            .requests
            .get_mut(logical_request_id)
            .ok_or_else(|| StateBackendError::NotFound(logical_request_id.to_owned()))?;
        let revision = next_revision(stored.revision)?;
        let mut request = stored.value.clone();
        let generation = stored
            .value
            .pointer("/facts/generation_id")
            .and_then(Value::as_u64)
            .and_then(|generation| generation.checked_add(1))
            .ok_or(StateBackendError::GenerationExhausted)?;
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
            .ok_or(StateBackendError::FieldNotObject("request fields.metadata"))?;
        merge(metadata, continuation_metadata);
        stored.value = request;
        stored.revision = revision;
        Ok(stored.value.clone())
    }

    fn read_shared(&self) -> Result<Document, StateBackendError> {
        Ok(self.inner.lock().unwrap().shared.value.clone())
    }

    fn replace_shared(&self, shared: Document) -> Result<(), StateBackendError> {
        require_object(&shared, "shared")?;
        let mut state = self.inner.lock().unwrap();
        let revision = next_revision(state.shared.revision)?;
        state.shared.value = shared;
        state.shared.revision = revision;
        Ok(())
    }

    fn read_request(&self, logical_request_id: &str) -> Result<Document, StateBackendError> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .get(logical_request_id)
            .map(|request| request.value.clone())
            .ok_or_else(|| StateBackendError::NotFound(logical_request_id.to_owned()))
    }

    fn remove_request(&self, logical_request_id: &str) -> Result<Document, StateBackendError> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .remove(logical_request_id)
            .map(|request| request.value)
            .ok_or_else(|| StateBackendError::NotFound(logical_request_id.to_owned()))
    }

    fn request_count(&self) -> Result<usize, StateBackendError> {
        Ok(self.inner.lock().unwrap().requests.len())
    }

    fn merge_request_facts(
        &self,
        logical_request_id: &str,
        facts: Document,
    ) -> Result<(), StateBackendError> {
        let facts = require_object(&facts, "request facts")?;
        let mut state = self.inner.lock().unwrap();
        let request = state
            .requests
            .get_mut(logical_request_id)
            .ok_or_else(|| StateBackendError::NotFound(logical_request_id.to_owned()))?;
        let revision = next_revision(request.revision)?;
        let current = request.value["facts"]
            .as_object_mut()
            .expect("canonical request facts");
        for identity in ["logical_request_id", "generation_id"] {
            if let Some(value) = facts.get(identity)
                && current.get(identity) != Some(value)
            {
                return Err(StateBackendError::HostIdentityMutation(identity));
            }
        }
        merge(current, facts);
        request.revision = revision;
        Ok(())
    }

    fn replace_request_fields(
        &self,
        logical_request_id: &str,
        fields: Document,
    ) -> Result<(), StateBackendError> {
        require_object(&fields, "request fields")?;
        let mut state = self.inner.lock().unwrap();
        let request = state
            .requests
            .get_mut(logical_request_id)
            .ok_or_else(|| StateBackendError::NotFound(logical_request_id.to_owned()))?;
        let revision = next_revision(request.revision)?;
        request.value["fields"] = fields;
        request.revision = revision;
        Ok(())
    }

    fn record_enacted_placement(
        &self,
        logical_request_id: &str,
        target_id: String,
    ) -> Result<(), StateBackendError> {
        if target_id.is_empty() {
            return Err(StateBackendError::EmptyTargetId);
        }
        self.merge_request_facts(logical_request_id, json!({"previous_target": target_id}))
    }

    fn feedback_result(&self, delivery_id: &str) -> Result<Option<Document>, StateBackendError> {
        Ok(self
            .inner
            .lock()
            .unwrap()
            .feedback_deliveries
            .get(delivery_id)
            .cloned())
    }
}

fn validate_snapshot(snapshot: &StateSnapshot) -> Result<(), StateBackendError> {
    require_object(&snapshot.shared, "shared")?;
    if snapshot.requests.keys().collect::<BTreeSet<_>>()
        != snapshot.request_revisions.keys().collect::<BTreeSet<_>>()
    {
        return Err(StateBackendError::InvalidSnapshot(
            "request revisions do not match the working set".into(),
        ));
    }
    for (logical_request_id, request) in &snapshot.requests {
        validate_request_scope(logical_request_id, request)
            .map_err(|error| StateBackendError::InvalidSnapshot(error.to_string()))?;
    }
    Ok(())
}

fn validate_updates(
    snapshot: &StateSnapshot,
    updates: &StateUpdates,
) -> Result<(), StateBackendError> {
    if let Some(shared) = &updates.shared {
        require_object(shared, "shared update")?;
    }
    for (logical_request_id, update) in &updates.requests {
        if !snapshot.requests.contains_key(logical_request_id) {
            return Err(StateBackendError::UnknownRequestUpdate(
                logical_request_id.clone(),
            ));
        }
        require_object(&update.fields, "request fields update")?;
        require_object(&update.scratch, "request scratch update")?;
    }
    Ok(())
}

fn next_revision(revision: u64) -> Result<u64, StateBackendError> {
    revision
        .checked_add(1)
        .ok_or(StateBackendError::RevisionExhausted)
}

fn merge(target: &mut Map<String, Value>, source: &Map<String, Value>) {
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

fn require_object<'a>(
    value: &'a Document,
    field: &'static str,
) -> Result<&'a Map<String, Value>, StateBackendError> {
    value
        .as_object()
        .ok_or(StateBackendError::FieldNotObject(field))
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateBackendError {
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
    #[error("state snapshot is invalid: {0}")]
    InvalidSnapshot(String),
    #[error("state update references request {0} outside the invocation working set")]
    UnknownRequestUpdate(String),
    #[error("terminal request list contains a duplicate")]
    DuplicateTerminalRequest,
    #[error("state revision changed while invoking policy for {0}")]
    RevisionConflict(String),
    #[error("state revision counter exhausted")]
    RevisionExhausted,
    #[error("feedback delivery ID must not be empty")]
    EmptyFeedbackDeliveryId,
    #[error("feedback delivery {0} is already committed")]
    DuplicateFeedback(String),
    #[error("feedback delivery ledger reached its limit of {0}")]
    FeedbackLedgerFull(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_preserves_mutable_state_and_updates_host_facts() {
        let backend = InMemoryPolicyStateBackend::default();
        backend
            .create_request(
                "L".into(),
                json!({"prompt": "first"}),
                json!({"keep": 1, "replace": 1}),
            )
            .unwrap();
        let snapshot = backend.load(&BTreeSet::from(["L".into()])).unwrap();
        backend
            .commit(
                &snapshot,
                &StateUpdates {
                    shared: Some(json!({"tenant": {"served": 8}})),
                    requests: BTreeMap::from([(
                        "L".into(),
                        RequestStateUpdate {
                            fields: json!({
                                "body": {"prompt": "first"},
                                "metadata": {"keep": 1, "replace": 1},
                                "custom": "preserved"
                            }),
                            scratch: json!({"served": 8}),
                        },
                    )]),
                },
                None,
                &[],
            )
            .unwrap();
        backend
            .record_enacted_placement("L", "node-a".into())
            .unwrap();
        backend
            .merge_request_facts("L", json!({"current_target": "node-a"}))
            .unwrap();

        let continuation = backend
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
        assert_eq!(
            backend.read_shared().unwrap(),
            json!({"tenant": {"served": 8}})
        );
    }

    #[test]
    fn stale_snapshot_commits_neither_state_nor_feedback() {
        let backend = InMemoryPolicyStateBackend::default();
        backend
            .create_request("L".into(), json!({}), json!({}))
            .unwrap();
        let snapshot = backend.load(&BTreeSet::from(["L".into()])).unwrap();
        backend.replace_shared(json!({"revision": 1})).unwrap();

        let feedback = FeedbackCommit {
            delivery_id: "d".into(),
            result: json!({}),
            maximum_deliveries: 8,
        };
        assert!(matches!(
            backend.commit(
                &snapshot,
                &StateUpdates {
                    shared: Some(json!({"lost": true})),
                    requests: BTreeMap::new()
                },
                Some(&feedback),
                &[]
            ),
            Err(StateBackendError::RevisionConflict(_))
        ));
        assert_eq!(backend.read_shared().unwrap(), json!({"revision": 1}));
        assert_eq!(backend.feedback_result("d").unwrap(), None);
    }

    #[test]
    fn feedback_dedup_and_terminal_removal_are_atomic() {
        let backend = InMemoryPolicyStateBackend::default();
        backend
            .create_request("L".into(), json!({}), json!({}))
            .unwrap();
        let snapshot = backend.load(&BTreeSet::from(["L".into()])).unwrap();
        let feedback = FeedbackCommit {
            delivery_id: "d".into(),
            result: json!({"ok": true}),
            maximum_deliveries: 8,
        };
        backend
            .commit(
                &snapshot,
                &StateUpdates::default(),
                Some(&feedback),
                &["L".into()],
            )
            .unwrap();
        assert_eq!(
            backend.feedback_result("d").unwrap(),
            Some(json!({"ok": true}))
        );
        assert!(matches!(
            backend.read_request("L"),
            Err(StateBackendError::NotFound(_))
        ));
    }
}
