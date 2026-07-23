use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pie_plex::Document;
use pie_plex::v0_6::{
    DeliveryId, GroupId, GroupLimits, GroupState, GroupStatus, PolicyState, PrincipalId, RequestId,
    RequestRef, RequestState, RequestStatus, StateUpdate, validate_group_transition,
    validate_policy_state, validate_request_continuation, validate_request_transition,
    validate_state_update,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

const GENERATION_LOCAL_FACTS: [&str; 1] = ["current_target"];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkingSetV0_6 {
    pub request_ids: BTreeSet<RequestId>,
    pub group_ids: BTreeSet<GroupId>,
}

impl WorkingSetV0_6 {
    pub fn with_request(mut self, request_id: impl Into<RequestId>) -> Self {
        self.request_ids.insert(request_id.into());
        self
    }

    pub fn with_group(mut self, group_id: impl Into<GroupId>) -> Self {
        self.group_ids.insert(group_id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshotV0_6 {
    pub state: PolicyState,
    pub(crate) shared_revision: u64,
    pub(crate) group_revisions: BTreeMap<GroupId, u64>,
    pub(crate) request_revisions: BTreeMap<RequestId, u64>,
}

impl StateSnapshotV0_6 {
    pub fn from_parts(
        state: PolicyState,
        shared_revision: u64,
        group_revisions: BTreeMap<GroupId, u64>,
        request_revisions: BTreeMap<RequestId, u64>,
    ) -> Result<Self, StateBackendErrorV0_6> {
        let snapshot = Self {
            state,
            shared_revision,
            group_revisions,
            request_revisions,
        };
        validate_snapshot(&snapshot)?;
        Ok(snapshot)
    }

    pub fn shared_revision(&self) -> u64 {
        self.shared_revision
    }

    pub fn group_revisions(&self) -> &BTreeMap<GroupId, u64> {
        &self.group_revisions
    }

    pub fn request_revisions(&self) -> &BTreeMap<RequestId, u64> {
        &self.request_revisions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackCommitV0_6 {
    pub delivery_id: DeliveryId,
    pub result: Document,
    pub maximum_deliveries: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalRequestV0_6 {
    pub request_id: RequestId,
    pub status: RequestStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalGroupV0_6 {
    pub group_id: GroupId,
    pub status: GroupStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalCleanupV0_6 {
    pub requests: Vec<TerminalRequestV0_6>,
    pub groups: Vec<TerminalGroupV0_6>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateScopeV0_6 {
    Shared,
    Group,
    Request,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateMetricsV0_6 {
    pub loads: u64,
    pub auto_joined_groups: u64,
    pub commit_attempts: u64,
    pub commits: u64,
    pub shared_conflicts: u64,
    pub group_conflicts: u64,
    pub request_conflicts: u64,
}

pub trait PolicyStateBackendV0_6: Send + Sync + 'static {
    fn load(
        &self,
        working_set: &WorkingSetV0_6,
    ) -> Result<StateSnapshotV0_6, StateBackendErrorV0_6>;

    fn commit(
        &self,
        snapshot: &StateSnapshotV0_6,
        updates: &StateUpdate,
        feedback: Option<&FeedbackCommitV0_6>,
        cleanup: &TerminalCleanupV0_6,
    ) -> Result<(), StateBackendErrorV0_6>;

    fn create_group(
        &self,
        group_id: GroupId,
        principal_id: PrincipalId,
        limits: GroupLimits,
        facts: Document,
    ) -> Result<GroupState, StateBackendErrorV0_6>;

    fn transition_group(
        &self,
        group_id: &GroupId,
        status: GroupStatus,
    ) -> Result<GroupState, StateBackendErrorV0_6>;

    fn create_request(
        &self,
        request_id: RequestId,
        principal_id: PrincipalId,
        group_id: Option<GroupId>,
        fields: Document,
        facts: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6>;

    fn continue_request(
        &self,
        request_id: &RequestId,
        fields: Document,
        facts: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6>;

    fn transition_request(
        &self,
        request_id: &RequestId,
        status: RequestStatus,
    ) -> Result<RequestState, StateBackendErrorV0_6>;

    fn merge_group_facts(
        &self,
        group_id: &GroupId,
        facts: Document,
    ) -> Result<GroupState, StateBackendErrorV0_6>;

    fn merge_request_facts(
        &self,
        request_id: &RequestId,
        facts: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6>;

    fn replace_request_fields(
        &self,
        request_id: &RequestId,
        fields: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6>;

    fn read_shared(&self) -> Result<Document, StateBackendErrorV0_6>;

    fn replace_shared(&self, shared: Document) -> Result<(), StateBackendErrorV0_6>;

    fn read_group(&self, group_id: &GroupId) -> Result<GroupState, StateBackendErrorV0_6>;

    fn read_request(&self, request_id: &RequestId) -> Result<RequestState, StateBackendErrorV0_6>;

    fn group_count(&self) -> Result<usize, StateBackendErrorV0_6>;

    fn request_count(&self) -> Result<usize, StateBackendErrorV0_6>;

    fn feedback_result(
        &self,
        delivery_id: &DeliveryId,
    ) -> Result<Option<Document>, StateBackendErrorV0_6>;

    fn metrics(&self) -> StateMetricsV0_6;
}

#[derive(Clone)]
pub struct InMemoryPolicyStateBackendV0_6 {
    inner: Arc<Mutex<BackendState>>,
    metrics: Arc<StateMetrics>,
}

#[derive(Default)]
struct StateMetrics {
    loads: AtomicU64,
    auto_joined_groups: AtomicU64,
    commit_attempts: AtomicU64,
    commits: AtomicU64,
    shared_conflicts: AtomicU64,
    group_conflicts: AtomicU64,
    request_conflicts: AtomicU64,
}

#[derive(Clone)]
struct BackendState {
    shared: Versioned<Document>,
    groups: BTreeMap<GroupId, Versioned<GroupState>>,
    requests: BTreeMap<RequestId, Versioned<RequestState>>,
    feedback_deliveries: BTreeMap<DeliveryId, Document>,
}

#[derive(Clone)]
struct Versioned<T> {
    value: T,
    revision: u64,
}

impl Default for InMemoryPolicyStateBackendV0_6 {
    fn default() -> Self {
        Self::new(json!({})).expect("empty shared state is valid")
    }
}

impl InMemoryPolicyStateBackendV0_6 {
    pub fn new(shared: Document) -> Result<Self, StateBackendErrorV0_6> {
        require_object(&shared, "shared")?;
        Ok(Self {
            inner: Arc::new(Mutex::new(BackendState {
                shared: Versioned {
                    value: shared,
                    revision: 0,
                },
                groups: BTreeMap::new(),
                requests: BTreeMap::new(),
                feedback_deliveries: BTreeMap::new(),
            })),
            metrics: Arc::new(StateMetrics::default()),
        })
    }

    pub fn shared_backend(self) -> Arc<dyn PolicyStateBackendV0_6> {
        Arc::new(self)
    }
}

impl PolicyStateBackendV0_6 for InMemoryPolicyStateBackendV0_6 {
    fn load(
        &self,
        working_set: &WorkingSetV0_6,
    ) -> Result<StateSnapshotV0_6, StateBackendErrorV0_6> {
        self.metrics.loads.fetch_add(1, Ordering::Relaxed);
        let state = self.inner.lock().unwrap();
        let mut request_states = BTreeMap::new();
        let mut request_revisions = BTreeMap::new();
        let mut group_ids = working_set.group_ids.clone();
        for request_id in &working_set.request_ids {
            let request = state
                .requests
                .get(request_id)
                .ok_or_else(|| StateBackendErrorV0_6::RequestNotFound(request_id.clone()))?;
            if let Some(group_id) = &request.value.request.group_id {
                group_ids.insert(group_id.clone());
            }
            self.metrics.auto_joined_groups.fetch_add(
                u64::try_from(group_ids.len().saturating_sub(working_set.group_ids.len()))
                    .unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            request_states.insert(request_id.clone(), request.value.clone());
            request_revisions.insert(request_id.clone(), request.revision);
        }

        let mut group_states = BTreeMap::new();
        let mut group_revisions = BTreeMap::new();
        for group_id in group_ids {
            let group = state
                .groups
                .get(&group_id)
                .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(group_id.clone()))?;
            group_states.insert(group_id.clone(), group.value.clone());
            group_revisions.insert(group_id, group.revision);
        }

        StateSnapshotV0_6::from_parts(
            PolicyState {
                shared: state.shared.value.clone(),
                groups: group_states.into_values().collect(),
                requests: request_states.into_values().collect(),
            },
            state.shared.revision,
            group_revisions,
            request_revisions,
        )
    }

    fn commit(
        &self,
        snapshot: &StateSnapshotV0_6,
        updates: &StateUpdate,
        feedback: Option<&FeedbackCommitV0_6>,
        cleanup: &TerminalCleanupV0_6,
    ) -> Result<(), StateBackendErrorV0_6> {
        self.metrics.commit_attempts.fetch_add(1, Ordering::Relaxed);
        validate_snapshot(snapshot)?;
        validate_state_update(&snapshot.state, updates)
            .map_err(|error| StateBackendErrorV0_6::InvalidUpdate(error.to_string()))?;
        validate_cleanup(snapshot, updates, cleanup)?;
        if let Some(feedback) = feedback {
            require_object(&feedback.result, "feedback result")?;
            if feedback.delivery_id.as_str().is_empty() {
                return Err(StateBackendErrorV0_6::EmptyFeedbackDeliveryId);
            }
            if feedback.maximum_deliveries == 0 {
                return Err(StateBackendErrorV0_6::ZeroFeedbackLedgerLimit);
            }
        }

        let mut state = self.inner.lock().unwrap();
        if let Some(feedback) = feedback {
            if state
                .feedback_deliveries
                .contains_key(&feedback.delivery_id)
            {
                return Err(StateBackendErrorV0_6::DuplicateFeedback(
                    feedback.delivery_id.clone(),
                ));
            }
        }
        if let Err(error) = compare_revisions(&state, snapshot) {
            self.record_conflict(&error);
            return Err(error);
        }

        let mut next = state.clone();
        let mut changed_groups = BTreeSet::new();
        let mut changed_requests = BTreeSet::new();

        if let Some(shared) = &updates.shared {
            next.shared.value = shared.clone();
            next.shared.revision = next_revision(next.shared.revision)?;
        }
        for group_update in &updates.groups {
            let group = next.groups.get_mut(&group_update.group_id).ok_or_else(|| {
                StateBackendErrorV0_6::GroupNotFound(group_update.group_id.clone())
            })?;
            group.value.scratch = group_update.scratch.clone();
            changed_groups.insert(group_update.group_id.clone());
        }
        for request_update in &updates.requests {
            let request = next
                .requests
                .get_mut(&request_update.request_id)
                .ok_or_else(|| {
                    StateBackendErrorV0_6::RequestNotFound(request_update.request_id.clone())
                })?;
            if let Some(fields) = &request_update.fields {
                request.value.fields = fields.clone();
            }
            if let Some(scratch) = &request_update.scratch {
                request.value.scratch = scratch.clone();
            }
            changed_requests.insert(request_update.request_id.clone());
        }

        for terminal in &cleanup.requests {
            let request = next.requests.remove(&terminal.request_id).ok_or_else(|| {
                StateBackendErrorV0_6::RequestNotFound(terminal.request_id.clone())
            })?;
            validate_request_transition(Some(request.value.status), terminal.status)
                .map_err(|error| StateBackendErrorV0_6::InvalidLifecycle(error.to_string()))?;
            if let Some(group_id) = request.value.request.group_id {
                let group = next
                    .groups
                    .get_mut(&group_id)
                    .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(group_id.clone()))?;
                group.value.member_count =
                    group.value.member_count.checked_sub(1).ok_or_else(|| {
                        StateBackendErrorV0_6::MemberCountUnderflow(group_id.clone())
                    })?;
                changed_groups.insert(group_id);
            }
            changed_requests.remove(&terminal.request_id);
        }

        for terminal in &cleanup.groups {
            let group = next
                .groups
                .get_mut(&terminal.group_id)
                .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(terminal.group_id.clone()))?;
            validate_group_transition(Some(group.value.status), terminal.status)
                .map_err(|error| StateBackendErrorV0_6::InvalidLifecycle(error.to_string()))?;
            group.value.status = terminal.status;
            changed_groups.insert(terminal.group_id.clone());
        }

        let removable_groups = next
            .groups
            .iter()
            .filter_map(|(group_id, group)| {
                (group.value.status != GroupStatus::Open && group.value.member_count == 0)
                    .then_some(group_id.clone())
            })
            .collect::<Vec<_>>();
        for group_id in removable_groups {
            next.groups.remove(&group_id);
            changed_groups.remove(&group_id);
        }

        for group_id in changed_groups {
            let group = next
                .groups
                .get_mut(&group_id)
                .expect("changed group remains present");
            group.revision = next_revision(group.revision)?;
        }
        for request_id in changed_requests {
            let request = next
                .requests
                .get_mut(&request_id)
                .expect("changed request remains present");
            request.revision = next_revision(request.revision)?;
        }

        if let Some(feedback) = feedback {
            if next.feedback_deliveries.len() >= feedback.maximum_deliveries {
                return Err(StateBackendErrorV0_6::FeedbackLedgerFull(
                    feedback.maximum_deliveries,
                ));
            }
            next.feedback_deliveries
                .insert(feedback.delivery_id.clone(), feedback.result.clone());
        }
        validate_backend_state(&next)?;
        *state = next;
        self.metrics.commits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn create_group(
        &self,
        group_id: GroupId,
        principal_id: PrincipalId,
        limits: GroupLimits,
        facts: Document,
    ) -> Result<GroupState, StateBackendErrorV0_6> {
        let facts = canonical_group_facts(group_id.clone(), principal_id.clone(), facts)?;
        let group = GroupState {
            group_id: group_id.clone(),
            principal_id,
            status: GroupStatus::Open,
            limits,
            member_count: 0,
            facts,
            scratch: json!({}),
        };
        validate_policy_state(&PolicyState {
            shared: json!({}),
            groups: vec![group.clone()],
            requests: Vec::new(),
        })
        .map_err(|error| StateBackendErrorV0_6::InvalidState(error.to_string()))?;

        let mut state = self.inner.lock().unwrap();
        if let Some(existing) = state.groups.get(&group_id) {
            if existing.value.principal_id == group.principal_id
                && existing.value.limits == group.limits
                && document_contains(&existing.value.facts, &group.facts)
            {
                return Ok(existing.value.clone());
            }
            return Err(StateBackendErrorV0_6::GroupAlreadyExists(group_id));
        }
        state.groups.insert(
            group_id,
            Versioned {
                value: group.clone(),
                revision: 0,
            },
        );
        Ok(group)
    }

    fn transition_group(
        &self,
        group_id: &GroupId,
        status: GroupStatus,
    ) -> Result<GroupState, StateBackendErrorV0_6> {
        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        let group = next
            .groups
            .get_mut(group_id)
            .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(group_id.clone()))?;
        validate_group_transition(Some(group.value.status), status)
            .map_err(|error| StateBackendErrorV0_6::InvalidLifecycle(error.to_string()))?;
        if group.value.status == status {
            return Ok(group.value.clone());
        }
        group.value.status = status;
        group.revision = next_revision(group.revision)?;
        let result = group.value.clone();
        if status != GroupStatus::Open && group.value.member_count == 0 {
            next.groups.remove(group_id);
        }
        validate_backend_state(&next)?;
        *state = next;
        Ok(result)
    }

    fn create_request(
        &self,
        request_id: RequestId,
        principal_id: PrincipalId,
        group_id: Option<GroupId>,
        fields: Document,
        facts: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6> {
        require_object(&fields, "request fields")?;
        let facts = canonical_request_facts(
            request_id.clone(),
            principal_id.clone(),
            group_id.clone(),
            0,
            facts,
        )?;
        let request = RequestState {
            request: RequestRef {
                request_id: request_id.clone(),
                generation_id: 0,
                group_id: group_id.clone(),
                principal_id: principal_id.clone(),
            },
            status: RequestStatus::Pending,
            facts,
            fields,
            scratch: json!({}),
        };

        let mut state = self.inner.lock().unwrap();
        if let Some(existing) = state.requests.get(&request_id) {
            if existing.value.request.principal_id == principal_id
                && existing.value.request.group_id == group_id
            {
                return Ok(existing.value.clone());
            }
            return Err(StateBackendErrorV0_6::RequestAlreadyExists(request_id));
        }
        let mut next = state.clone();
        if let Some(group_id) = &group_id {
            let group = next
                .groups
                .get_mut(group_id)
                .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(group_id.clone()))?;
            if group.value.status != GroupStatus::Open {
                return Err(StateBackendErrorV0_6::TerminalGroup(group_id.clone()));
            }
            if group.value.principal_id != principal_id {
                return Err(StateBackendErrorV0_6::PrincipalMismatch {
                    request_id,
                    group_id: group_id.clone(),
                });
            }
            if group.value.member_count >= group.value.limits.max_members {
                return Err(StateBackendErrorV0_6::GroupMemberLimit(group_id.clone()));
            }
            group.value.member_count += 1;
            group.revision = next_revision(group.revision)?;
        }
        next.requests.insert(
            request.request.request_id.clone(),
            Versioned {
                value: request.clone(),
                revision: 0,
            },
        );
        validate_backend_state(&next)?;
        *state = next;
        Ok(request)
    }

    fn continue_request(
        &self,
        request_id: &RequestId,
        fields: Document,
        facts: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6> {
        require_object(&fields, "request fields")?;
        let fact_updates = require_object(&facts, "request facts")?;
        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        let request = next
            .requests
            .get_mut(request_id)
            .ok_or_else(|| StateBackendErrorV0_6::RequestNotFound(request_id.clone()))?;
        if !matches!(
            request.value.status,
            RequestStatus::Admitted | RequestStatus::Active | RequestStatus::Paused
        ) {
            return Err(StateBackendErrorV0_6::ContinuationStatus(
                request.value.status,
            ));
        }
        let requested_generation = fact_updates
            .get("generation_id")
            .and_then(Value::as_u64)
            .ok_or(StateBackendErrorV0_6::MissingContinuationGeneration)?;
        if requested_generation == request.value.request.generation_id {
            validate_identity_fact_updates(fact_updates, &request.value.request)?;
            return Ok(request.value.clone());
        }
        let next_generation = request
            .value
            .request
            .generation_id
            .checked_add(1)
            .ok_or(StateBackendErrorV0_6::GenerationExhausted)?;
        if requested_generation != next_generation {
            return Err(StateBackendErrorV0_6::InvalidContinuationGeneration {
                expected: next_generation,
                actual: requested_generation,
            });
        }
        let next_ref = RequestRef {
            request_id: request.value.request.request_id.clone(),
            generation_id: next_generation,
            group_id: request.value.request.group_id.clone(),
            principal_id: request.value.request.principal_id.clone(),
        };
        validate_request_continuation(&request.value.request, &next_ref)
            .map_err(|error| StateBackendErrorV0_6::InvalidLifecycle(error.to_string()))?;
        validate_identity_fact_updates(fact_updates, &next_ref)?;
        let request_facts = request
            .value
            .facts
            .as_object_mut()
            .expect("validated request facts");
        for key in GENERATION_LOCAL_FACTS {
            request_facts.remove(key);
        }
        merge(request_facts, fact_updates);
        request_facts.insert("generation_id".into(), json!(next_generation));
        request.value.request = next_ref;
        request.value.fields = fields;
        request.revision = next_revision(request.revision)?;
        let result = request.value.clone();
        validate_backend_state(&next)?;
        *state = next;
        Ok(result)
    }

    fn transition_request(
        &self,
        request_id: &RequestId,
        status: RequestStatus,
    ) -> Result<RequestState, StateBackendErrorV0_6> {
        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        let request = next
            .requests
            .get_mut(request_id)
            .ok_or_else(|| StateBackendErrorV0_6::RequestNotFound(request_id.clone()))?;
        validate_request_transition(Some(request.value.status), status)
            .map_err(|error| StateBackendErrorV0_6::InvalidLifecycle(error.to_string()))?;
        if request.value.status == status {
            return Ok(request.value.clone());
        }
        request.value.status = status;
        request.revision = next_revision(request.revision)?;
        let result = request.value.clone();
        validate_backend_state(&next)?;
        *state = next;
        Ok(result)
    }

    fn merge_group_facts(
        &self,
        group_id: &GroupId,
        facts: Document,
    ) -> Result<GroupState, StateBackendErrorV0_6> {
        let facts = require_object(&facts, "group facts")?;
        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        let group = next
            .groups
            .get_mut(group_id)
            .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(group_id.clone()))?;
        validate_group_identity_fact_updates(facts, &group.value)?;
        merge(
            group
                .value
                .facts
                .as_object_mut()
                .expect("validated group facts"),
            facts,
        );
        group.revision = next_revision(group.revision)?;
        let result = group.value.clone();
        validate_backend_state(&next)?;
        *state = next;
        Ok(result)
    }

    fn merge_request_facts(
        &self,
        request_id: &RequestId,
        facts: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6> {
        let facts = require_object(&facts, "request facts")?;
        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        let request = next
            .requests
            .get_mut(request_id)
            .ok_or_else(|| StateBackendErrorV0_6::RequestNotFound(request_id.clone()))?;
        validate_identity_fact_updates(facts, &request.value.request)?;
        merge(
            request
                .value
                .facts
                .as_object_mut()
                .expect("validated request facts"),
            facts,
        );
        request.revision = next_revision(request.revision)?;
        let result = request.value.clone();
        validate_backend_state(&next)?;
        *state = next;
        Ok(result)
    }

    fn replace_request_fields(
        &self,
        request_id: &RequestId,
        fields: Document,
    ) -> Result<RequestState, StateBackendErrorV0_6> {
        require_object(&fields, "request fields")?;
        let mut state = self.inner.lock().unwrap();
        let mut next = state.clone();
        let request = next
            .requests
            .get_mut(request_id)
            .ok_or_else(|| StateBackendErrorV0_6::RequestNotFound(request_id.clone()))?;
        request.value.fields = fields;
        request.revision = next_revision(request.revision)?;
        let result = request.value.clone();
        validate_backend_state(&next)?;
        *state = next;
        Ok(result)
    }

    fn read_shared(&self) -> Result<Document, StateBackendErrorV0_6> {
        Ok(self.inner.lock().unwrap().shared.value.clone())
    }

    fn replace_shared(&self, shared: Document) -> Result<(), StateBackendErrorV0_6> {
        require_object(&shared, "shared")?;
        let mut state = self.inner.lock().unwrap();
        state.shared.value = shared;
        state.shared.revision = next_revision(state.shared.revision)?;
        Ok(())
    }

    fn read_group(&self, group_id: &GroupId) -> Result<GroupState, StateBackendErrorV0_6> {
        self.inner
            .lock()
            .unwrap()
            .groups
            .get(group_id)
            .map(|group| group.value.clone())
            .ok_or_else(|| StateBackendErrorV0_6::GroupNotFound(group_id.clone()))
    }

    fn read_request(&self, request_id: &RequestId) -> Result<RequestState, StateBackendErrorV0_6> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .get(request_id)
            .map(|request| request.value.clone())
            .ok_or_else(|| StateBackendErrorV0_6::RequestNotFound(request_id.clone()))
    }

    fn group_count(&self) -> Result<usize, StateBackendErrorV0_6> {
        Ok(self.inner.lock().unwrap().groups.len())
    }

    fn request_count(&self) -> Result<usize, StateBackendErrorV0_6> {
        Ok(self.inner.lock().unwrap().requests.len())
    }

    fn feedback_result(
        &self,
        delivery_id: &DeliveryId,
    ) -> Result<Option<Document>, StateBackendErrorV0_6> {
        Ok(self
            .inner
            .lock()
            .unwrap()
            .feedback_deliveries
            .get(delivery_id)
            .cloned())
    }

    fn metrics(&self) -> StateMetricsV0_6 {
        StateMetricsV0_6 {
            loads: self.metrics.loads.load(Ordering::Relaxed),
            auto_joined_groups: self.metrics.auto_joined_groups.load(Ordering::Relaxed),
            commit_attempts: self.metrics.commit_attempts.load(Ordering::Relaxed),
            commits: self.metrics.commits.load(Ordering::Relaxed),
            shared_conflicts: self.metrics.shared_conflicts.load(Ordering::Relaxed),
            group_conflicts: self.metrics.group_conflicts.load(Ordering::Relaxed),
            request_conflicts: self.metrics.request_conflicts.load(Ordering::Relaxed),
        }
    }
}

impl InMemoryPolicyStateBackendV0_6 {
    fn record_conflict(&self, error: &StateBackendErrorV0_6) {
        let StateBackendErrorV0_6::RevisionConflict { scope, .. } = error else {
            return;
        };
        let metric = match scope {
            StateScopeV0_6::Shared => &self.metrics.shared_conflicts,
            StateScopeV0_6::Group => &self.metrics.group_conflicts,
            StateScopeV0_6::Request => &self.metrics.request_conflicts,
        };
        metric.fetch_add(1, Ordering::Relaxed);
    }
}

fn canonical_group_facts(
    group_id: GroupId,
    principal_id: PrincipalId,
    facts: Document,
) -> Result<Document, StateBackendErrorV0_6> {
    let mut facts = require_object(&facts, "group facts")?.clone();
    for (key, expected) in [
        ("group_id", Value::String(group_id.0)),
        ("principal_id", Value::String(principal_id.0)),
    ] {
        if let Some(actual) = facts.get(key)
            && actual != &expected
        {
            return Err(StateBackendErrorV0_6::HostIdentityMutation(key));
        }
        facts.insert(key.into(), expected);
    }
    Ok(Value::Object(facts))
}

fn canonical_request_facts(
    request_id: RequestId,
    principal_id: PrincipalId,
    group_id: Option<GroupId>,
    generation_id: u64,
    facts: Document,
) -> Result<Document, StateBackendErrorV0_6> {
    let mut facts = require_object(&facts, "request facts")?.clone();
    let identities = [
        ("request_id", Value::String(request_id.0)),
        ("principal_id", Value::String(principal_id.0)),
        (
            "group_id",
            group_id.map_or(Value::Null, |group_id| Value::String(group_id.0)),
        ),
        ("generation_id", json!(generation_id)),
    ];
    for (key, expected) in identities {
        if let Some(actual) = facts.get(key)
            && actual != &expected
        {
            return Err(StateBackendErrorV0_6::HostIdentityMutation(key));
        }
        facts.insert(key.into(), expected);
    }
    Ok(Value::Object(facts))
}

fn validate_group_identity_fact_updates(
    facts: &Map<String, Value>,
    group: &GroupState,
) -> Result<(), StateBackendErrorV0_6> {
    for (key, expected) in [
        ("group_id", Value::String(group.group_id.0.clone())),
        ("principal_id", Value::String(group.principal_id.0.clone())),
    ] {
        if let Some(actual) = facts.get(key)
            && actual != &expected
        {
            return Err(StateBackendErrorV0_6::HostIdentityMutation(key));
        }
    }
    Ok(())
}

fn validate_identity_fact_updates(
    facts: &Map<String, Value>,
    request: &RequestRef,
) -> Result<(), StateBackendErrorV0_6> {
    let identities = [
        ("request_id", Value::String(request.request_id.0.clone())),
        (
            "principal_id",
            Value::String(request.principal_id.0.clone()),
        ),
        (
            "group_id",
            request
                .group_id
                .as_ref()
                .map_or(Value::Null, |group_id| Value::String(group_id.0.clone())),
        ),
        ("generation_id", json!(request.generation_id)),
    ];
    for (key, expected) in identities {
        if let Some(actual) = facts.get(key)
            && actual != &expected
        {
            return Err(StateBackendErrorV0_6::HostIdentityMutation(key));
        }
    }
    Ok(())
}

fn validate_snapshot(snapshot: &StateSnapshotV0_6) -> Result<(), StateBackendErrorV0_6> {
    validate_policy_state(&snapshot.state)
        .map_err(|error| StateBackendErrorV0_6::InvalidSnapshot(error.to_string()))?;
    let group_ids = snapshot
        .state
        .groups
        .iter()
        .map(|group| &group.group_id)
        .collect::<BTreeSet<_>>();
    if group_ids != snapshot.group_revisions.keys().collect::<BTreeSet<_>>() {
        return Err(StateBackendErrorV0_6::InvalidSnapshot(
            "group revisions do not match the working set".into(),
        ));
    }
    let request_ids = snapshot
        .state
        .requests
        .iter()
        .map(|request| &request.request.request_id)
        .collect::<BTreeSet<_>>();
    if request_ids != snapshot.request_revisions.keys().collect::<BTreeSet<_>>() {
        return Err(StateBackendErrorV0_6::InvalidSnapshot(
            "request revisions do not match the working set".into(),
        ));
    }
    Ok(())
}

fn validate_cleanup(
    snapshot: &StateSnapshotV0_6,
    updates: &StateUpdate,
    cleanup: &TerminalCleanupV0_6,
) -> Result<(), StateBackendErrorV0_6> {
    let request_ids = snapshot
        .state
        .requests
        .iter()
        .map(|request| &request.request.request_id)
        .collect::<BTreeSet<_>>();
    let group_ids = snapshot
        .state
        .groups
        .iter()
        .map(|group| &group.group_id)
        .collect::<BTreeSet<_>>();
    let mut terminal_requests = BTreeSet::new();
    for terminal in &cleanup.requests {
        if !terminal.status.is_terminal() {
            return Err(StateBackendErrorV0_6::NonTerminalRequestCleanup(
                terminal.request_id.clone(),
            ));
        }
        if !request_ids.contains(&terminal.request_id) {
            return Err(StateBackendErrorV0_6::RequestOutsideSnapshot(
                terminal.request_id.clone(),
            ));
        }
        if !terminal_requests.insert(&terminal.request_id) {
            return Err(StateBackendErrorV0_6::DuplicateTerminalRequest(
                terminal.request_id.clone(),
            ));
        }
    }
    if let Some(update) = updates
        .requests
        .iter()
        .find(|update| terminal_requests.contains(&update.request_id))
    {
        return Err(StateBackendErrorV0_6::TerminalRequestUpdate(
            update.request_id.clone(),
        ));
    }

    let mut terminal_groups = BTreeSet::new();
    for terminal in &cleanup.groups {
        if terminal.status == GroupStatus::Open {
            return Err(StateBackendErrorV0_6::NonTerminalGroupCleanup(
                terminal.group_id.clone(),
            ));
        }
        if !group_ids.contains(&terminal.group_id) {
            return Err(StateBackendErrorV0_6::GroupOutsideSnapshot(
                terminal.group_id.clone(),
            ));
        }
        if !terminal_groups.insert(&terminal.group_id) {
            return Err(StateBackendErrorV0_6::DuplicateTerminalGroup(
                terminal.group_id.clone(),
            ));
        }
    }
    Ok(())
}

fn compare_revisions(
    state: &BackendState,
    snapshot: &StateSnapshotV0_6,
) -> Result<(), StateBackendErrorV0_6> {
    if state.shared.revision != snapshot.shared_revision {
        return Err(StateBackendErrorV0_6::RevisionConflict {
            scope: StateScopeV0_6::Shared,
            id: "shared".into(),
        });
    }
    for (group_id, expected) in &snapshot.group_revisions {
        let current =
            state
                .groups
                .get(group_id)
                .ok_or_else(|| StateBackendErrorV0_6::RevisionConflict {
                    scope: StateScopeV0_6::Group,
                    id: group_id.0.clone(),
                })?;
        if current.revision != *expected {
            return Err(StateBackendErrorV0_6::RevisionConflict {
                scope: StateScopeV0_6::Group,
                id: group_id.0.clone(),
            });
        }
    }
    for (request_id, expected) in &snapshot.request_revisions {
        let current = state.requests.get(request_id).ok_or_else(|| {
            StateBackendErrorV0_6::RevisionConflict {
                scope: StateScopeV0_6::Request,
                id: request_id.0.clone(),
            }
        })?;
        if current.revision != *expected {
            return Err(StateBackendErrorV0_6::RevisionConflict {
                scope: StateScopeV0_6::Request,
                id: request_id.0.clone(),
            });
        }
    }
    Ok(())
}

fn validate_backend_state(state: &BackendState) -> Result<(), StateBackendErrorV0_6> {
    let policy_state = PolicyState {
        shared: state.shared.value.clone(),
        groups: state
            .groups
            .values()
            .map(|group| group.value.clone())
            .collect(),
        requests: state
            .requests
            .values()
            .map(|request| request.value.clone())
            .collect(),
    };
    validate_policy_state(&policy_state)
        .map_err(|error| StateBackendErrorV0_6::InvalidState(error.to_string()))?;
    let mut actual_members = BTreeMap::<GroupId, u32>::new();
    for request in state.requests.values() {
        if let Some(group_id) = &request.value.request.group_id {
            let members = actual_members.entry(group_id.clone()).or_default();
            *members = members
                .checked_add(1)
                .ok_or_else(|| StateBackendErrorV0_6::MemberCountOverflow(group_id.clone()))?;
        }
    }
    for (group_id, group) in &state.groups {
        let actual = actual_members.get(group_id).copied().unwrap_or(0);
        if group.value.member_count != actual {
            return Err(StateBackendErrorV0_6::MemberCountMismatch {
                group_id: group_id.clone(),
                recorded: group.value.member_count,
                actual,
            });
        }
    }
    Ok(())
}

fn next_revision(revision: u64) -> Result<u64, StateBackendErrorV0_6> {
    revision
        .checked_add(1)
        .ok_or(StateBackendErrorV0_6::RevisionExhausted)
}

fn merge(target: &mut Map<String, Value>, source: &Map<String, Value>) {
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

fn document_contains(actual: &Document, expected: &Document) -> bool {
    let (Some(actual), Some(expected)) = (actual.as_object(), expected.as_object()) else {
        return false;
    };
    expected
        .iter()
        .all(|(key, value)| actual.get(key) == Some(value))
}

fn require_object<'a>(
    value: &'a Document,
    field: &'static str,
) -> Result<&'a Map<String, Value>, StateBackendErrorV0_6> {
    value
        .as_object()
        .ok_or(StateBackendErrorV0_6::FieldNotObject(field))
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateBackendErrorV0_6 {
    #[error("{0} must be a JSON object")]
    FieldNotObject(&'static str),
    #[error("group {0:?} already exists")]
    GroupAlreadyExists(GroupId),
    #[error("request {0:?} already exists")]
    RequestAlreadyExists(RequestId),
    #[error("group {0:?} was not found")]
    GroupNotFound(GroupId),
    #[error("request {0:?} was not found")]
    RequestNotFound(RequestId),
    #[error("request {request_id:?} principal does not own group {group_id:?}")]
    PrincipalMismatch {
        request_id: RequestId,
        group_id: GroupId,
    },
    #[error("group {0:?} is terminal")]
    TerminalGroup(GroupId),
    #[error("group {0:?} reached its member limit")]
    GroupMemberLimit(GroupId),
    #[error("request status {0:?} cannot continue to another generation")]
    ContinuationStatus(RequestStatus),
    #[error("continuation facts must contain generation_id")]
    MissingContinuationGeneration,
    #[error("continuation generation must be {expected}, got {actual}")]
    InvalidContinuationGeneration { expected: u64, actual: u64 },
    #[error("request generation counter exhausted")]
    GenerationExhausted,
    #[error("host fact update cannot change {0}")]
    HostIdentityMutation(&'static str),
    #[error("state snapshot is invalid: {0}")]
    InvalidSnapshot(String),
    #[error("policy state is invalid: {0}")]
    InvalidState(String),
    #[error("state update is invalid: {0}")]
    InvalidUpdate(String),
    #[error("lifecycle transition is invalid: {0}")]
    InvalidLifecycle(String),
    #[error("request {0:?} is outside the invocation snapshot")]
    RequestOutsideSnapshot(RequestId),
    #[error("group {0:?} is outside the invocation snapshot")]
    GroupOutsideSnapshot(GroupId),
    #[error("terminal request {0:?} appears more than once")]
    DuplicateTerminalRequest(RequestId),
    #[error("terminal group {0:?} appears more than once")]
    DuplicateTerminalGroup(GroupId),
    #[error("request {0:?} cleanup requires a terminal status")]
    NonTerminalRequestCleanup(RequestId),
    #[error("group {0:?} cleanup requires a terminal status")]
    NonTerminalGroupCleanup(GroupId),
    #[error("terminal request {0:?} cannot also receive a state update")]
    TerminalRequestUpdate(RequestId),
    #[error("group {0:?} member count underflowed")]
    MemberCountUnderflow(GroupId),
    #[error("group {0:?} member count overflowed")]
    MemberCountOverflow(GroupId),
    #[error("group {group_id:?} records {recorded} members but backend contains {actual}")]
    MemberCountMismatch {
        group_id: GroupId,
        recorded: u32,
        actual: u32,
    },
    #[error("state revision changed while invoking policy for {scope:?} {id}")]
    RevisionConflict { scope: StateScopeV0_6, id: String },
    #[error("state revision counter exhausted")]
    RevisionExhausted,
    #[error("feedback delivery ID must not be empty")]
    EmptyFeedbackDeliveryId,
    #[error("feedback ledger limit must be positive")]
    ZeroFeedbackLedgerLimit,
    #[error("feedback delivery {0:?} is already committed")]
    DuplicateFeedback(DeliveryId),
    #[error("feedback delivery ledger reached its limit of {0}")]
    FeedbackLedgerFull(usize),
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::thread;

    use pie_plex::v0_6::{GroupStateUpdate, RequestStateUpdate};

    use super::*;

    fn limits() -> GroupLimits {
        GroupLimits {
            max_members: 4,
            max_scratch_bytes: 1024,
        }
    }

    fn create_group(backend: &InMemoryPolicyStateBackendV0_6, id: &str, principal: &str) {
        backend
            .create_group(
                id.into(),
                principal.into(),
                limits(),
                json!({"kind": "agent"}),
            )
            .unwrap();
    }

    fn create_request(
        backend: &InMemoryPolicyStateBackendV0_6,
        id: &str,
        principal: &str,
        group: Option<&str>,
    ) {
        backend
            .create_request(
                id.into(),
                principal.into(),
                group.map(Into::into),
                json!({"body": {}, "metadata": {}}),
                json!({}),
            )
            .unwrap();
    }

    fn activate_request(backend: &InMemoryPolicyStateBackendV0_6, id: &str) {
        backend
            .transition_request(&id.into(), RequestStatus::Admitted)
            .unwrap();
        backend
            .transition_request(&id.into(), RequestStatus::Active)
            .unwrap();
    }

    #[test]
    fn group_outlives_one_child_and_is_removed_after_the_last() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_group(&backend, "G", "tenant");
        create_request(&backend, "A", "tenant", Some("G"));
        create_request(&backend, "B", "tenant", Some("G"));
        activate_request(&backend, "A");
        activate_request(&backend, "B");
        let snapshot = backend
            .load(
                &WorkingSetV0_6::default()
                    .with_request("A")
                    .with_request("B"),
            )
            .unwrap();
        backend
            .commit(
                &snapshot,
                &StateUpdate {
                    shared: None,
                    groups: Vec::new(),
                    requests: Vec::new(),
                },
                None,
                &TerminalCleanupV0_6 {
                    requests: vec![TerminalRequestV0_6 {
                        request_id: "A".into(),
                        status: RequestStatus::Completed,
                    }],
                    groups: vec![TerminalGroupV0_6 {
                        group_id: "G".into(),
                        status: GroupStatus::Closed,
                    }],
                },
            )
            .unwrap();
        assert_eq!(backend.read_group(&"G".into()).unwrap().member_count, 1);
        assert!(matches!(
            backend.read_request(&"A".into()),
            Err(StateBackendErrorV0_6::RequestNotFound(_))
        ));

        let snapshot = backend
            .load(&WorkingSetV0_6::default().with_request("B"))
            .unwrap();
        backend
            .commit(
                &snapshot,
                &StateUpdate {
                    shared: None,
                    groups: Vec::new(),
                    requests: Vec::new(),
                },
                None,
                &TerminalCleanupV0_6 {
                    requests: vec![TerminalRequestV0_6 {
                        request_id: "B".into(),
                        status: RequestStatus::Completed,
                    }],
                    groups: Vec::new(),
                },
            )
            .unwrap();
        assert!(matches!(
            backend.read_group(&"G".into()),
            Err(StateBackendErrorV0_6::GroupNotFound(_))
        ));
    }

    #[test]
    fn copied_group_id_cannot_forge_membership() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_group(&backend, "G", "tenant-a");
        assert!(matches!(
            backend.create_request(
                "A".into(),
                "tenant-b".into(),
                Some("G".into()),
                json!({}),
                json!({"group_id": "G"})
            ),
            Err(StateBackendErrorV0_6::PrincipalMismatch { .. })
        ));
    }

    #[test]
    fn lifecycle_create_and_continue_are_idempotent() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_group(&backend, "G", "tenant");
        create_group(&backend, "G", "tenant");
        create_request(&backend, "A", "tenant", Some("G"));
        create_request(&backend, "A", "tenant", Some("G"));
        assert_eq!(backend.read_group(&"G".into()).unwrap().member_count, 1);

        backend
            .transition_request(&"A".into(), RequestStatus::Admitted)
            .unwrap();
        let before = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        backend
            .transition_request(&"A".into(), RequestStatus::Admitted)
            .unwrap();
        let after = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        assert_eq!(before.request_revisions, after.request_revisions);

        let fields = json!({"body": {"prompt": "next"}, "metadata": {}});
        let facts = json!({"generation_id": 1});
        backend
            .continue_request(&"A".into(), fields.clone(), facts.clone())
            .unwrap();
        let before = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        let repeated = backend
            .continue_request(&"A".into(), fields, facts)
            .unwrap();
        let after = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        assert_eq!(repeated.request.generation_id, 1);
        assert_eq!(before.request_revisions, after.request_revisions);
    }

    #[test]
    fn terminal_group_rejects_new_requests_and_membership_is_immutable() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_group(&backend, "G", "tenant");
        create_request(&backend, "A", "tenant", Some("G"));
        activate_request(&backend, "A");
        backend
            .transition_group(&"G".into(), GroupStatus::Closed)
            .unwrap();
        assert!(matches!(
            backend.create_request(
                "B".into(),
                "tenant".into(),
                Some("G".into()),
                json!({}),
                json!({})
            ),
            Err(StateBackendErrorV0_6::TerminalGroup(_))
        ));
        assert!(matches!(
            backend.continue_request(
                &"A".into(),
                json!({}),
                json!({"group_id": "other", "generation_id": 1})
            ),
            Err(StateBackendErrorV0_6::HostIdentityMutation("group_id"))
        ));
    }

    #[test]
    fn group_scratch_quota_is_enforced() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        backend
            .create_group(
                "G".into(),
                "tenant".into(),
                GroupLimits {
                    max_members: 1,
                    max_scratch_bytes: 16,
                },
                json!({}),
            )
            .unwrap();
        let snapshot = backend
            .load(&WorkingSetV0_6::default().with_group("G"))
            .unwrap();
        assert!(matches!(
            backend.commit(
                &snapshot,
                &StateUpdate {
                    shared: None,
                    groups: vec![GroupStateUpdate {
                        group_id: "G".into(),
                        scratch: json!({"too_large": "xxxxxxxxxxxxxxxx"})
                    }],
                    requests: Vec::new()
                },
                None,
                &TerminalCleanupV0_6::default()
            ),
            Err(StateBackendErrorV0_6::InvalidUpdate(_))
        ));
    }

    #[test]
    fn concurrent_child_feedback_conflicts_instead_of_losing_accounting() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_group(&backend, "G", "tenant");
        create_request(&backend, "A", "tenant", Some("G"));
        create_request(&backend, "B", "tenant", Some("G"));
        let snapshot_a = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        let snapshot_b = backend
            .load(&WorkingSetV0_6::default().with_request("B"))
            .unwrap();
        let update = |service| StateUpdate {
            shared: None,
            groups: vec![GroupStateUpdate {
                group_id: "G".into(),
                scratch: json!({"service": service}),
            }],
            requests: Vec::new(),
        };
        backend
            .commit(
                &snapshot_a,
                &update(1),
                None,
                &TerminalCleanupV0_6::default(),
            )
            .unwrap();
        assert!(matches!(
            backend.commit(
                &snapshot_b,
                &update(1),
                None,
                &TerminalCleanupV0_6::default()
            ),
            Err(StateBackendErrorV0_6::RevisionConflict {
                scope: StateScopeV0_6::Group,
                ..
            })
        ));
        let retry = backend
            .load(&WorkingSetV0_6::default().with_request("B"))
            .unwrap();
        backend
            .commit(&retry, &update(2), None, &TerminalCleanupV0_6::default())
            .unwrap();
        assert_eq!(
            backend.read_group(&"G".into()).unwrap().scratch["service"],
            2
        );
        let metrics = backend.metrics();
        assert_eq!(metrics.group_conflicts, 1);
        assert!(metrics.auto_joined_groups >= 3);
    }

    #[test]
    fn unrelated_groups_commit_without_conflict() {
        let backend = Arc::new(InMemoryPolicyStateBackendV0_6::default());
        create_group(&backend, "A", "tenant-a");
        create_group(&backend, "B", "tenant-b");
        let snapshot_a = backend
            .load(&WorkingSetV0_6::default().with_group("A"))
            .unwrap();
        let snapshot_b = backend
            .load(&WorkingSetV0_6::default().with_group("B"))
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for (snapshot, group_id) in [(snapshot_a, "A"), (snapshot_b, "B")] {
            let backend = backend.clone();
            let barrier = barrier.clone();
            workers.push(thread::spawn(move || {
                barrier.wait();
                backend.commit(
                    &snapshot,
                    &StateUpdate {
                        shared: None,
                        groups: vec![GroupStateUpdate {
                            group_id: group_id.into(),
                            scratch: json!({"updated": true}),
                        }],
                        requests: Vec::new(),
                    },
                    None,
                    &TerminalCleanupV0_6::default(),
                )
            }));
        }
        barrier.wait();
        for worker in workers {
            worker.join().unwrap().unwrap();
        }
        let metrics = backend.metrics();
        assert_eq!(metrics.group_conflicts, 0);
        assert_eq!(metrics.commits, 2);
    }

    #[test]
    fn feedback_dedup_and_cleanup_commit_together() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_request(&backend, "A", "tenant", None);
        activate_request(&backend, "A");
        let snapshot = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        let feedback = FeedbackCommitV0_6 {
            delivery_id: "delivery".into(),
            result: json!({"ok": true}),
            maximum_deliveries: 8,
        };
        backend
            .commit(
                &snapshot,
                &StateUpdate {
                    shared: Some(json!({"completed": 1})),
                    groups: Vec::new(),
                    requests: Vec::new(),
                },
                Some(&feedback),
                &TerminalCleanupV0_6 {
                    requests: vec![TerminalRequestV0_6 {
                        request_id: "A".into(),
                        status: RequestStatus::Completed,
                    }],
                    groups: Vec::new(),
                },
            )
            .unwrap();
        assert_eq!(
            backend.feedback_result(&"delivery".into()).unwrap(),
            Some(json!({"ok": true}))
        );
        assert!(matches!(
            backend.read_request(&"A".into()),
            Err(StateBackendErrorV0_6::RequestNotFound(_))
        ));
        assert!(matches!(
            backend.commit(
                &snapshot,
                &StateUpdate {
                    shared: None,
                    groups: Vec::new(),
                    requests: Vec::new(),
                },
                Some(&feedback),
                &TerminalCleanupV0_6::default()
            ),
            Err(StateBackendErrorV0_6::DuplicateFeedback(_))
        ));
    }

    #[test]
    fn continuation_preserves_group_and_policy_state() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        create_group(&backend, "G", "tenant");
        create_request(&backend, "A", "tenant", Some("G"));
        activate_request(&backend, "A");
        let snapshot = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        backend
            .commit(
                &snapshot,
                &StateUpdate {
                    shared: None,
                    groups: vec![GroupStateUpdate {
                        group_id: "G".into(),
                        scratch: json!({"service": 4}),
                    }],
                    requests: vec![RequestStateUpdate {
                        request_id: "A".into(),
                        fields: None,
                        scratch: Some(json!({"attempts": 1})),
                    }],
                },
                None,
                &TerminalCleanupV0_6::default(),
            )
            .unwrap();
        backend
            .merge_request_facts(&"A".into(), json!({"current_target": "node-a"}))
            .unwrap();
        let continued = backend
            .continue_request(
                &"A".into(),
                json!({"body": {"prompt": "next"}, "metadata": {}}),
                json!({"generation_id": 1, "previous_target": "node-a"}),
            )
            .unwrap();
        assert_eq!(continued.request.generation_id, 1);
        assert_eq!(continued.request.group_id, Some("G".into()));
        assert_eq!(continued.scratch["attempts"], 1);
        assert!(continued.facts.get("current_target").is_none());
        assert_eq!(
            backend.read_group(&"G".into()).unwrap().scratch["service"],
            4
        );
    }
}
