use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use pie_plex::v0_6::{
    AdmitContext, AdmitPlan, Beneficiary, CacheContext, CachePlan, FeedbackContext,
    FeedbackSubject, GroupId, Operation, OpportunityId, PolicyState, RequestId, RouteContext,
    RouteDecision, RoutePlan, ScheduleContext, SchedulePlan, SnapshotRef, StateUpdate,
    validate_admit_context, validate_admit_plan, validate_cache_context, validate_cache_plan,
    validate_feedback_context, validate_route_context, validate_route_plan,
    validate_schedule_context, validate_schedule_plan, validate_state_update,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

use crate::{StateSnapshotV0_6, WorkingSetV0_6};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "context", rename_all = "kebab-case")]
pub enum OperationContextV0_6 {
    Admit(AdmitContext),
    Route(RouteContext),
    Schedule(ScheduleContext),
    Cache(CacheContext),
    Feedback(FeedbackContext),
}

impl OperationContextV0_6 {
    pub fn operation(&self) -> Operation {
        match self {
            Self::Admit(_) => Operation::Admit,
            Self::Route(_) => Operation::Route,
            Self::Schedule(_) => Operation::Schedule,
            Self::Cache(_) => Operation::Cache,
            Self::Feedback(_) => Operation::Feedback,
        }
    }

    fn opportunity_id(&self) -> Option<&OpportunityId> {
        match self {
            Self::Admit(context) => Some(&context.meta.opportunity_id),
            Self::Route(context) => Some(&context.meta.opportunity_id),
            Self::Schedule(context) => Some(&context.meta.opportunity_id),
            Self::Cache(context) => Some(&context.meta.opportunity_id),
            Self::Feedback(_) => None,
        }
    }

    fn attempt(&self) -> Option<u32> {
        match self {
            Self::Admit(context) => Some(context.meta.attempt),
            Self::Route(context) => Some(context.meta.attempt),
            Self::Schedule(context) => Some(context.meta.attempt),
            Self::Cache(context) => Some(context.meta.attempt),
            Self::Feedback(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "plan", rename_all = "kebab-case")]
pub enum OperationPlanV0_6 {
    Admit(AdmitPlan),
    Route(RoutePlan),
    Schedule(SchedulePlan),
    Cache(CachePlan),
    Feedback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedRouteAssignmentV0_6 {
    pub request_index: u32,
    pub edge_index: u32,
    pub target_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedRoutePlanV0_6 {
    pub assignments: Vec<NormalizedRouteAssignmentV0_6>,
    pub deferred: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "plan", rename_all = "kebab-case")]
pub enum NormalizedPlanV0_6 {
    Admit(AdmitPlan),
    Route(NormalizedRoutePlanV0_6),
    Schedule(SchedulePlan),
    Cache(CachePlan),
    Feedback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceOrderV0_6 {
    pub request_ids: Vec<RequestId>,
    pub group_ids: Vec<GroupId>,
    pub target_ids: Vec<String>,
    pub cache_object_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayRecordV0_6 {
    pub operation: Operation,
    pub opportunity_id: Option<OpportunityId>,
    pub delivery_id: Option<String>,
    pub attempt: Option<u32>,
    pub snapshot: Option<SnapshotRef>,
    pub order: TraceOrderV0_6,
    pub normalized_plan: NormalizedPlanV0_6,
    pub state_update: StateUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolLimitsV0_6 {
    pub max_context_bytes: usize,
    pub max_state_update_bytes: usize,
    pub max_requests: usize,
    pub max_groups: usize,
    pub max_targets: usize,
    pub max_route_edges: usize,
    pub max_schedule_selections: usize,
    pub max_cache_objects: usize,
    pub max_beneficiaries_per_object: usize,
    pub max_feedback_records: usize,
    pub max_mechanics: usize,
    pub max_tracked_opportunities: usize,
    pub max_cache_episodes: usize,
}

impl Default for ProtocolLimitsV0_6 {
    fn default() -> Self {
        Self {
            max_context_bytes: 4 * 1024 * 1024,
            max_state_update_bytes: 4 * 1024 * 1024,
            max_requests: 256,
            max_groups: 256,
            max_targets: 256,
            max_route_edges: 65_536,
            max_schedule_selections: 256,
            max_cache_objects: 4_096,
            max_beneficiaries_per_object: 64,
            max_feedback_records: 4_096,
            max_mechanics: 64,
            max_tracked_opportunities: 4_096,
            max_cache_episodes: 1_024,
        }
    }
}

pub fn working_set_v0_6(
    context: &OperationContextV0_6,
    limits: ProtocolLimitsV0_6,
) -> Result<WorkingSetV0_6, ProtocolErrorV0_6> {
    validate_context_bounds(context, limits)?;
    let mut working_set = WorkingSetV0_6::default();
    match context {
        OperationContextV0_6::Admit(context) => {
            for candidate in &context.candidates {
                insert_request_ref(&mut working_set, &candidate.request);
            }
        }
        OperationContextV0_6::Route(context) => {
            for request in &context.requests {
                insert_request_ref(&mut working_set, &request.request);
            }
        }
        OperationContextV0_6::Schedule(context) => {
            for candidate in &context.runnable {
                insert_request_ref(&mut working_set, &candidate.request);
            }
        }
        OperationContextV0_6::Cache(context) => {
            for object in context
                .resident
                .iter()
                .map(|resident| &resident.object)
                .chain(&context.prospective)
            {
                for beneficiary in &object.beneficiaries {
                    match beneficiary {
                        Beneficiary::Request(request_id) => {
                            working_set.request_ids.insert(request_id.clone());
                        }
                        Beneficiary::Group(group_id) => {
                            working_set.group_ids.insert(group_id.clone());
                        }
                    }
                }
            }
        }
        OperationContextV0_6::Feedback(context) => {
            for record in &context.records {
                match &record.subject {
                    FeedbackSubject::Request(request_id) => {
                        working_set.request_ids.insert(request_id.clone());
                    }
                    FeedbackSubject::WorkGroup(group_id) => {
                        working_set.group_ids.insert(group_id.clone());
                    }
                    _ => {}
                }
            }
        }
    }
    if working_set.request_ids.len() > limits.max_requests {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "working-set requests",
            actual: working_set.request_ids.len(),
            maximum: limits.max_requests,
        });
    }
    if working_set.group_ids.len() > limits.max_groups {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "working-set groups",
            actual: working_set.group_ids.len(),
            maximum: limits.max_groups,
        });
    }
    Ok(working_set)
}

pub fn validate_context_v0_6(
    context: &OperationContextV0_6,
    state: &PolicyState,
    limits: ProtocolLimitsV0_6,
) -> Result<(), ProtocolErrorV0_6> {
    validate_context_bounds(context, limits)?;
    let validation = match context {
        OperationContextV0_6::Admit(context) => validate_admit_context(state, context),
        OperationContextV0_6::Route(context) => validate_route_context(state, context),
        OperationContextV0_6::Schedule(context) => validate_schedule_context(state, context),
        OperationContextV0_6::Cache(context) => validate_cache_context(context),
        OperationContextV0_6::Feedback(context) => validate_feedback_context(context),
    };
    validation.map_err(|error| ProtocolErrorV0_6::InvalidContext(error.to_string()))
}

pub fn validate_snapshot_context_v0_6(
    context: &OperationContextV0_6,
    snapshot: &StateSnapshotV0_6,
    limits: ProtocolLimitsV0_6,
) -> Result<(), ProtocolErrorV0_6> {
    validate_context_v0_6(context, &snapshot.state, limits)?;
    let expected = snapshot_ref_v0_6(snapshot)?;
    let actual = match context {
        OperationContextV0_6::Admit(context) => Some(&context.meta.snapshot),
        OperationContextV0_6::Route(context) => Some(&context.meta.snapshot),
        OperationContextV0_6::Schedule(context) => Some(&context.meta.snapshot),
        OperationContextV0_6::Cache(context) => Some(&context.meta.snapshot),
        OperationContextV0_6::Feedback(_) => None,
    };
    if actual.is_some_and(|actual| actual != &expected) {
        return Err(ProtocolErrorV0_6::SnapshotMismatch {
            expected,
            actual: actual.cloned().expect("checked Some"),
        });
    }
    Ok(())
}

pub fn validate_output_v0_6(
    context: &OperationContextV0_6,
    plan: &OperationPlanV0_6,
    state: &PolicyState,
    update: &StateUpdate,
    limits: ProtocolLimitsV0_6,
) -> Result<NormalizedPlanV0_6, ProtocolErrorV0_6> {
    let update_bytes = serde_json::to_vec(update)
        .map_err(ProtocolErrorV0_6::Encode)?
        .len();
    if update_bytes > limits.max_state_update_bytes {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "state-update bytes",
            actual: update_bytes,
            maximum: limits.max_state_update_bytes,
        });
    }
    validate_state_update(state, update)
        .map_err(|error| ProtocolErrorV0_6::InvalidStateUpdate(error.to_string()))?;
    match (context, plan) {
        (OperationContextV0_6::Admit(context), OperationPlanV0_6::Admit(plan)) => {
            validate_admit_plan(context, plan)
                .map_err(|error| ProtocolErrorV0_6::InvalidPlan(error.to_string()))?;
        }
        (OperationContextV0_6::Route(context), OperationPlanV0_6::Route(plan)) => {
            validate_route_plan(context, plan)
                .map_err(|error| ProtocolErrorV0_6::InvalidPlan(error.to_string()))?;
        }
        (OperationContextV0_6::Schedule(context), OperationPlanV0_6::Schedule(plan)) => {
            if plan.selections.len() > limits.max_schedule_selections {
                return Err(ProtocolErrorV0_6::LimitExceeded {
                    field: "schedule selections",
                    actual: plan.selections.len(),
                    maximum: limits.max_schedule_selections,
                });
            }
            validate_schedule_plan(context, plan)
                .map_err(|error| ProtocolErrorV0_6::InvalidPlan(error.to_string()))?;
        }
        (OperationContextV0_6::Cache(context), OperationPlanV0_6::Cache(plan)) => {
            validate_cache_plan(context, plan)
                .map_err(|error| ProtocolErrorV0_6::InvalidPlan(error.to_string()))?;
        }
        (OperationContextV0_6::Feedback(_), OperationPlanV0_6::Feedback) => {}
        _ => {
            return Err(ProtocolErrorV0_6::OperationMismatch {
                context: context.operation(),
                plan: plan.operation(),
            });
        }
    }
    normalized_plan_v0_6(context, plan)
}

impl OperationPlanV0_6 {
    pub fn operation(&self) -> Operation {
        match self {
            Self::Admit(_) => Operation::Admit,
            Self::Route(_) => Operation::Route,
            Self::Schedule(_) => Operation::Schedule,
            Self::Cache(_) => Operation::Cache,
            Self::Feedback => Operation::Feedback,
        }
    }
}

pub fn normalized_plan_v0_6(
    context: &OperationContextV0_6,
    plan: &OperationPlanV0_6,
) -> Result<NormalizedPlanV0_6, ProtocolErrorV0_6> {
    match (context, plan) {
        (OperationContextV0_6::Admit(_), OperationPlanV0_6::Admit(plan)) => {
            Ok(NormalizedPlanV0_6::Admit(plan.clone()))
        }
        (OperationContextV0_6::Route(context), OperationPlanV0_6::Route(plan)) => {
            let mut assignments = Vec::new();
            let mut deferred = Vec::new();
            for (request_index, decision) in plan.decisions.iter().enumerate() {
                match decision {
                    RouteDecision::Assign(edge_index) => {
                        let edge = context.feasible_edges.get(*edge_index as usize).ok_or(
                            ProtocolErrorV0_6::InvalidPlan(
                                "route edge index is out of range".into(),
                            ),
                        )?;
                        assignments.push(NormalizedRouteAssignmentV0_6 {
                            request_index: u32::try_from(request_index).map_err(|_| {
                                ProtocolErrorV0_6::IndexOverflow("route request index")
                            })?,
                            edge_index: *edge_index,
                            target_index: edge.target_index,
                        });
                    }
                    RouteDecision::Defer => {
                        deferred.push(u32::try_from(request_index).map_err(|_| {
                            ProtocolErrorV0_6::IndexOverflow("route request index")
                        })?);
                    }
                }
            }
            Ok(NormalizedPlanV0_6::Route(NormalizedRoutePlanV0_6 {
                assignments,
                deferred,
            }))
        }
        (OperationContextV0_6::Schedule(_), OperationPlanV0_6::Schedule(plan)) => {
            Ok(NormalizedPlanV0_6::Schedule(plan.clone()))
        }
        (OperationContextV0_6::Cache(_), OperationPlanV0_6::Cache(plan)) => {
            Ok(NormalizedPlanV0_6::Cache(plan.clone()))
        }
        (OperationContextV0_6::Feedback(_), OperationPlanV0_6::Feedback) => {
            Ok(NormalizedPlanV0_6::Feedback)
        }
        _ => Err(ProtocolErrorV0_6::OperationMismatch {
            context: context.operation(),
            plan: plan.operation(),
        }),
    }
}

pub fn snapshot_ref_v0_6(snapshot: &StateSnapshotV0_6) -> Result<SnapshotRef, ProtocolErrorV0_6> {
    let payload = json!({
        "shared_revision": snapshot.shared_revision(),
        "group_revisions": snapshot.group_revisions(),
        "request_revisions": snapshot.request_revisions(),
    });
    let bytes = serde_json::to_vec(&payload).map_err(ProtocolErrorV0_6::Encode)?;
    let digest = blake3::hash(&bytes);
    let revision = snapshot
        .group_revisions()
        .values()
        .chain(snapshot.request_revisions().values())
        .copied()
        .fold(snapshot.shared_revision(), u64::max);
    Ok(SnapshotRef {
        id: digest.to_hex().to_string().into(),
        revision,
    })
}

pub fn replay_record_v0_6(
    context: &OperationContextV0_6,
    normalized_plan: NormalizedPlanV0_6,
    state_update: StateUpdate,
) -> ReplayRecordV0_6 {
    let (opportunity_id, delivery_id, attempt, snapshot) = match context {
        OperationContextV0_6::Admit(context) => (
            Some(context.meta.opportunity_id.clone()),
            None,
            Some(context.meta.attempt),
            Some(context.meta.snapshot.clone()),
        ),
        OperationContextV0_6::Route(context) => (
            Some(context.meta.opportunity_id.clone()),
            None,
            Some(context.meta.attempt),
            Some(context.meta.snapshot.clone()),
        ),
        OperationContextV0_6::Schedule(context) => (
            Some(context.meta.opportunity_id.clone()),
            None,
            Some(context.meta.attempt),
            Some(context.meta.snapshot.clone()),
        ),
        OperationContextV0_6::Cache(context) => (
            Some(context.meta.opportunity_id.clone()),
            None,
            Some(context.meta.attempt),
            Some(context.meta.snapshot.clone()),
        ),
        OperationContextV0_6::Feedback(context) => {
            (None, Some(context.delivery_id.0.clone()), None, None)
        }
    };
    ReplayRecordV0_6 {
        operation: context.operation(),
        opportunity_id,
        delivery_id,
        attempt,
        snapshot,
        order: trace_order(context),
        normalized_plan,
        state_update,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpportunityStatus {
    Active,
    Completed,
}

#[derive(Debug, Clone)]
struct OpportunityEntry {
    fingerprint: [u8; 32],
    last_attempt: u32,
    status: OpportunityStatus,
}

pub struct OpportunityTrackerV0_6 {
    maximum: usize,
    entries: Mutex<BTreeMap<OpportunityId, OpportunityEntry>>,
}

impl OpportunityTrackerV0_6 {
    pub fn new(maximum: usize) -> Result<Self, ProtocolErrorV0_6> {
        if maximum == 0 {
            return Err(ProtocolErrorV0_6::ZeroTrackerLimit("opportunity"));
        }
        Ok(Self {
            maximum,
            entries: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn begin(&self, context: &OperationContextV0_6) -> Result<(), ProtocolErrorV0_6> {
        let Some(opportunity_id) = context.opportunity_id() else {
            return Ok(());
        };
        let attempt = context.attempt().expect("decision context has attempt");
        let fingerprint = opportunity_fingerprint(context)?;
        let mut entries = self.entries.lock().unwrap();
        match entries.get_mut(opportunity_id) {
            None => {
                if attempt != 0 {
                    return Err(ProtocolErrorV0_6::FirstAttemptNotZero(attempt));
                }
                if entries.len() >= self.maximum {
                    return Err(ProtocolErrorV0_6::TrackerFull {
                        tracker: "opportunity",
                        maximum: self.maximum,
                    });
                }
                entries.insert(
                    opportunity_id.clone(),
                    OpportunityEntry {
                        fingerprint,
                        last_attempt: 0,
                        status: OpportunityStatus::Active,
                    },
                );
            }
            Some(entry) => {
                if entry.status == OpportunityStatus::Completed {
                    return Err(ProtocolErrorV0_6::CompletedOpportunityReused(
                        opportunity_id.clone(),
                    ));
                }
                if entry.fingerprint != fingerprint {
                    return Err(ProtocolErrorV0_6::OpportunityBoundaryChanged(
                        opportunity_id.clone(),
                    ));
                }
                let expected = entry
                    .last_attempt
                    .checked_add(1)
                    .ok_or(ProtocolErrorV0_6::AttemptExhausted)?;
                if attempt != expected {
                    return Err(ProtocolErrorV0_6::UnexpectedAttempt {
                        opportunity_id: opportunity_id.clone(),
                        expected,
                        actual: attempt,
                    });
                }
                entry.last_attempt = attempt;
            }
        }
        Ok(())
    }

    pub fn complete(&self, opportunity_id: &OpportunityId) -> Result<(), ProtocolErrorV0_6> {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries
            .get_mut(opportunity_id)
            .ok_or_else(|| ProtocolErrorV0_6::UnknownOpportunity(opportunity_id.clone()))?;
        entry.status = OpportunityStatus::Completed;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CacheEpisodeEntry {
    max_iterations: u32,
    last_iteration: u32,
    last_opportunity_id: OpportunityId,
    completed: bool,
}

pub struct CacheEpisodeTrackerV0_6 {
    maximum: usize,
    entries: Mutex<BTreeMap<String, CacheEpisodeEntry>>,
}

impl CacheEpisodeTrackerV0_6 {
    pub fn new(maximum: usize) -> Result<Self, ProtocolErrorV0_6> {
        if maximum == 0 {
            return Err(ProtocolErrorV0_6::ZeroTrackerLimit("cache episode"));
        }
        Ok(Self {
            maximum,
            entries: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn observe(&self, context: &CacheContext) -> Result<(), ProtocolErrorV0_6> {
        let Some(episode) = &context.episode else {
            return Ok(());
        };
        let mut entries = self.entries.lock().unwrap();
        match entries.get_mut(episode.episode_id.as_str()) {
            None => {
                if episode.iteration != 0 {
                    return Err(ProtocolErrorV0_6::FirstEpisodeIterationNotZero(
                        episode.iteration,
                    ));
                }
                if entries.len() >= self.maximum {
                    return Err(ProtocolErrorV0_6::TrackerFull {
                        tracker: "cache episode",
                        maximum: self.maximum,
                    });
                }
                entries.insert(
                    episode.episode_id.0.clone(),
                    CacheEpisodeEntry {
                        max_iterations: episode.max_iterations,
                        last_iteration: 0,
                        last_opportunity_id: context.meta.opportunity_id.clone(),
                        completed: false,
                    },
                );
            }
            Some(entry) => {
                if entry.completed {
                    return Err(ProtocolErrorV0_6::CompletedCacheEpisodeReused(
                        episode.episode_id.0.clone(),
                    ));
                }
                if entry.max_iterations != episode.max_iterations {
                    return Err(ProtocolErrorV0_6::CacheEpisodeBoundChanged(
                        episode.episode_id.0.clone(),
                    ));
                }
                if entry.last_opportunity_id == context.meta.opportunity_id {
                    if entry.last_iteration != episode.iteration {
                        return Err(ProtocolErrorV0_6::CacheEpisodeIteration {
                            episode_id: episode.episode_id.0.clone(),
                            expected: entry.last_iteration,
                            actual: episode.iteration,
                        });
                    }
                } else {
                    let expected = entry
                        .last_iteration
                        .checked_add(1)
                        .ok_or(ProtocolErrorV0_6::AttemptExhausted)?;
                    if episode.iteration != expected {
                        return Err(ProtocolErrorV0_6::CacheEpisodeIteration {
                            episode_id: episode.episode_id.0.clone(),
                            expected,
                            actual: episode.iteration,
                        });
                    }
                    entry.last_iteration = episode.iteration;
                    entry.last_opportunity_id = context.meta.opportunity_id.clone();
                }
            }
        }
        Ok(())
    }

    pub fn complete(&self, episode_id: &str) -> Result<(), ProtocolErrorV0_6> {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries
            .get_mut(episode_id)
            .ok_or_else(|| ProtocolErrorV0_6::UnknownCacheEpisode(episode_id.into()))?;
        entry.completed = true;
        Ok(())
    }
}

fn validate_context_bounds(
    context: &OperationContextV0_6,
    limits: ProtocolLimitsV0_6,
) -> Result<(), ProtocolErrorV0_6> {
    let context_bytes = serde_json::to_vec(context)
        .map_err(ProtocolErrorV0_6::Encode)?
        .len();
    if context_bytes > limits.max_context_bytes {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "context bytes",
            actual: context_bytes,
            maximum: limits.max_context_bytes,
        });
    }
    let (requests, groups, mechanics) = match context {
        OperationContextV0_6::Admit(context) => (
            context.candidates.len(),
            referenced_groups(
                context
                    .candidates
                    .iter()
                    .map(|candidate| candidate.request.group_id.as_ref()),
            ),
            context.meta.mechanics.len(),
        ),
        OperationContextV0_6::Route(context) => {
            if context.targets.len() > limits.max_targets {
                return Err(ProtocolErrorV0_6::LimitExceeded {
                    field: "route targets",
                    actual: context.targets.len(),
                    maximum: limits.max_targets,
                });
            }
            if context.feasible_edges.len() > limits.max_route_edges {
                return Err(ProtocolErrorV0_6::LimitExceeded {
                    field: "route edges",
                    actual: context.feasible_edges.len(),
                    maximum: limits.max_route_edges,
                });
            }
            (
                context.requests.len(),
                referenced_groups(
                    context
                        .requests
                        .iter()
                        .map(|request| request.request.group_id.as_ref()),
                ),
                context.meta.mechanics.len(),
            )
        }
        OperationContextV0_6::Schedule(context) => (
            context.runnable.len(),
            referenced_groups(
                context
                    .runnable
                    .iter()
                    .map(|candidate| candidate.request.group_id.as_ref()),
            ),
            context.meta.mechanics.len(),
        ),
        OperationContextV0_6::Cache(context) => {
            let objects = context.resident.len() + context.prospective.len();
            if objects > limits.max_cache_objects {
                return Err(ProtocolErrorV0_6::LimitExceeded {
                    field: "cache objects",
                    actual: objects,
                    maximum: limits.max_cache_objects,
                });
            }
            for object in context
                .resident
                .iter()
                .map(|resident| &resident.object)
                .chain(&context.prospective)
            {
                if object.beneficiaries.len() > limits.max_beneficiaries_per_object {
                    return Err(ProtocolErrorV0_6::LimitExceeded {
                        field: "cache beneficiaries",
                        actual: object.beneficiaries.len(),
                        maximum: limits.max_beneficiaries_per_object,
                    });
                }
            }
            (0, 0, context.meta.mechanics.len())
        }
        OperationContextV0_6::Feedback(context) => {
            if context.records.len() > limits.max_feedback_records {
                return Err(ProtocolErrorV0_6::LimitExceeded {
                    field: "feedback records",
                    actual: context.records.len(),
                    maximum: limits.max_feedback_records,
                });
            }
            (0, 0, 0)
        }
    };
    if requests > limits.max_requests {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "context requests",
            actual: requests,
            maximum: limits.max_requests,
        });
    }
    if groups > limits.max_groups {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "context groups",
            actual: groups,
            maximum: limits.max_groups,
        });
    }
    if mechanics > limits.max_mechanics {
        return Err(ProtocolErrorV0_6::LimitExceeded {
            field: "negotiated mechanics",
            actual: mechanics,
            maximum: limits.max_mechanics,
        });
    }
    Ok(())
}

fn insert_request_ref(working_set: &mut WorkingSetV0_6, request: &pie_plex::v0_6::RequestRef) {
    working_set.request_ids.insert(request.request_id.clone());
    if let Some(group_id) = &request.group_id {
        working_set.group_ids.insert(group_id.clone());
    }
}

fn referenced_groups<'a>(groups: impl IntoIterator<Item = Option<&'a GroupId>>) -> usize {
    groups.into_iter().flatten().collect::<BTreeSet<_>>().len()
}

fn opportunity_fingerprint(context: &OperationContextV0_6) -> Result<[u8; 32], ProtocolErrorV0_6> {
    let mut value = serde_json::to_value(context).map_err(ProtocolErrorV0_6::Encode)?;
    if let Some(meta) = value
        .get_mut("context")
        .and_then(|context| context.get_mut("meta"))
        .and_then(Value::as_object_mut)
    {
        meta.insert("attempt".into(), json!(0));
        meta.insert(
            "snapshot".into(),
            json!({"id": "normalized", "revision": 0}),
        );
    }
    let bytes = serde_json::to_vec(&value).map_err(ProtocolErrorV0_6::Encode)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

fn trace_order(context: &OperationContextV0_6) -> TraceOrderV0_6 {
    let mut request_ids = Vec::new();
    let mut group_ids = Vec::new();
    let mut target_ids = Vec::new();
    let mut cache_object_ids = Vec::new();
    match context {
        OperationContextV0_6::Admit(context) => {
            for candidate in &context.candidates {
                request_ids.push(candidate.request.request_id.clone());
                if let Some(group_id) = &candidate.request.group_id {
                    group_ids.push(group_id.clone());
                }
            }
        }
        OperationContextV0_6::Route(context) => {
            for request in &context.requests {
                request_ids.push(request.request.request_id.clone());
                if let Some(group_id) = &request.request.group_id {
                    group_ids.push(group_id.clone());
                }
            }
            target_ids.extend(
                context
                    .targets
                    .iter()
                    .map(|target| target.target_id.0.clone()),
            );
        }
        OperationContextV0_6::Schedule(context) => {
            for candidate in &context.runnable {
                request_ids.push(candidate.request.request_id.clone());
                if let Some(group_id) = &candidate.request.group_id {
                    group_ids.push(group_id.clone());
                }
            }
        }
        OperationContextV0_6::Cache(context) => {
            cache_object_ids.extend(
                context
                    .resident
                    .iter()
                    .map(|resident| resident.object.object_id.0.clone())
                    .chain(
                        context
                            .prospective
                            .iter()
                            .map(|object| object.object_id.0.clone()),
                    ),
            );
        }
        OperationContextV0_6::Feedback(context) => {
            for record in &context.records {
                match &record.subject {
                    FeedbackSubject::Request(request_id) => request_ids.push(request_id.clone()),
                    FeedbackSubject::WorkGroup(group_id) => group_ids.push(group_id.clone()),
                    _ => {}
                }
            }
        }
    }
    TraceOrderV0_6 {
        request_ids,
        group_ids,
        target_ids,
        cache_object_ids,
    }
}

#[derive(Debug, Error)]
pub enum ProtocolErrorV0_6 {
    #[error("failed to encode typed protocol data")]
    Encode(#[source] serde_json::Error),
    #[error("{field} contains {actual}; maximum is {maximum}")]
    LimitExceeded {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("policy context is invalid: {0}")]
    InvalidContext(String),
    #[error("policy plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("policy state update is invalid: {0}")]
    InvalidStateUpdate(String),
    #[error("context snapshot {actual:?} does not match loaded snapshot {expected:?}")]
    SnapshotMismatch {
        expected: SnapshotRef,
        actual: SnapshotRef,
    },
    #[error("context operation {context:?} does not match plan operation {plan:?}")]
    OperationMismatch { context: Operation, plan: Operation },
    #[error("{0} exceeds the u32 wire index range")]
    IndexOverflow(&'static str),
    #[error("first opportunity attempt must be zero, got {0}")]
    FirstAttemptNotZero(u32),
    #[error("opportunity attempt counter exhausted")]
    AttemptExhausted,
    #[error("{tracker} tracker reached its limit of {maximum}")]
    TrackerFull {
        tracker: &'static str,
        maximum: usize,
    },
    #[error("tracker limit for {0} must be positive")]
    ZeroTrackerLimit(&'static str),
    #[error("completed opportunity {0:?} cannot be reused")]
    CompletedOpportunityReused(OpportunityId),
    #[error("opportunity {0:?} changed its bounded decision boundary during retry")]
    OpportunityBoundaryChanged(OpportunityId),
    #[error("opportunity {opportunity_id:?} expected attempt {expected}, got {actual}")]
    UnexpectedAttempt {
        opportunity_id: OpportunityId,
        expected: u32,
        actual: u32,
    },
    #[error("opportunity {0:?} is not tracked")]
    UnknownOpportunity(OpportunityId),
    #[error("first cache episode iteration must be zero, got {0}")]
    FirstEpisodeIterationNotZero(u32),
    #[error("completed cache episode {0:?} cannot be reused")]
    CompletedCacheEpisodeReused(String),
    #[error("cache episode {0:?} changed its maximum iteration bound")]
    CacheEpisodeBoundChanged(String),
    #[error("cache episode {episode_id:?} expected iteration {expected}, got {actual}")]
    CacheEpisodeIteration {
        episode_id: String,
        expected: u32,
        actual: u32,
    },
    #[error("cache episode {0:?} is not tracked")]
    UnknownCacheEpisode(String),
}

#[cfg(test)]
mod tests {
    use pie_plex::v0_6::{
        AdmissionCandidate, AdmissionCapacity, AdmissionDecision, AdmitCause, CacheAdmission,
        CacheCapacity, CacheCause, CacheEpisode, CacheObject, DecisionMeta, GroupLimits,
        GroupState, GroupStatus, RequestRef, RequestState, RequestStatus, ResourceLimit,
        RouteCause, RouteEdge, RouteRequest, RouteTarget, ScheduleCandidate, ScheduleCapacity,
        ScheduleCause, ScheduleSelection,
    };
    use serde_json::json;

    use crate::{InMemoryPolicyStateBackendV0_6, PolicyStateBackendV0_6};

    use super::*;

    fn request_ref(id: &str, group: Option<&str>) -> RequestRef {
        RequestRef {
            request_id: id.into(),
            generation_id: 0,
            group_id: group.map(Into::into),
            principal_id: "tenant".into(),
        }
    }

    fn state(status: RequestStatus) -> PolicyState {
        PolicyState {
            shared: json!({}),
            groups: vec![GroupState {
                group_id: "G".into(),
                principal_id: "tenant".into(),
                status: GroupStatus::Open,
                limits: GroupLimits {
                    max_members: 8,
                    max_scratch_bytes: 1024,
                },
                member_count: 2,
                facts: json!({"group_id": "G", "principal_id": "tenant"}),
                scratch: json!({}),
            }],
            requests: ["A", "B"]
                .into_iter()
                .map(|id| RequestState {
                    request: request_ref(id, Some("G")),
                    status,
                    facts: json!({
                        "request_id": id,
                        "generation_id": 0,
                        "group_id": "G",
                        "principal_id": "tenant"
                    }),
                    fields: json!({}),
                    scratch: json!({}),
                })
                .collect(),
        }
    }

    fn meta(id: &str, attempt: u32) -> DecisionMeta {
        DecisionMeta {
            opportunity_id: id.into(),
            snapshot: SnapshotRef {
                id: "snapshot".into(),
                revision: attempt as u64,
            },
            attempt,
            mechanics: Vec::new(),
        }
    }

    #[test]
    fn working_set_includes_trusted_groups_and_cache_beneficiaries() {
        let context = OperationContextV0_6::Cache(CacheContext {
            meta: meta("cache", 0),
            cause: CacheCause::Insertion,
            resident: Vec::new(),
            prospective: vec![CacheObject {
                object_id: "object".into(),
                size_bytes: 1,
                beneficiaries: vec![
                    Beneficiary::Request("A".into()),
                    Beneficiary::Group("G".into()),
                ],
                beneficiary_count: 2,
                facts: json!({}),
            }],
            capacity: CacheCapacity {
                max_bytes: 1,
                fixed_bytes: 0,
                facts: json!({}),
            },
            episode: None,
        });
        let working_set = working_set_v0_6(&context, ProtocolLimitsV0_6::default()).unwrap();
        assert_eq!(working_set.request_ids, BTreeSet::from(["A".into()]));
        assert_eq!(working_set.group_ids, BTreeSet::from(["G".into()]));
    }

    #[test]
    fn joint_route_and_atomic_selection_are_validated_directly() {
        let route = RouteContext {
            meta: meta("route", 0),
            cause: RouteCause::Admission,
            requests: vec![
                RouteRequest {
                    request: request_ref("A", Some("G")),
                    facts: json!({}),
                },
                RouteRequest {
                    request: request_ref("B", Some("G")),
                    facts: json!({}),
                },
            ],
            targets: ["X", "Y"]
                .into_iter()
                .map(|id| RouteTarget {
                    target_id: id.into(),
                    max_assignments: 1,
                    capacity: Vec::new(),
                    revision: 1,
                    facts: json!({}),
                })
                .collect(),
            feasible_edges: vec![
                RouteEdge {
                    request_index: 0,
                    target_index: 0,
                    demand: Vec::new(),
                    facts: json!({"utility": 10}),
                },
                RouteEdge {
                    request_index: 0,
                    target_index: 1,
                    demand: Vec::new(),
                    facts: json!({"utility": 9}),
                },
                RouteEdge {
                    request_index: 1,
                    target_index: 0,
                    demand: Vec::new(),
                    facts: json!({"utility": 8}),
                },
                RouteEdge {
                    request_index: 1,
                    target_index: 1,
                    demand: Vec::new(),
                    facts: json!({"utility": 0}),
                },
            ],
        };
        let context = OperationContextV0_6::Route(route);
        let admitted_state = state(RequestStatus::Admitted);
        validate_context_v0_6(&context, &admitted_state, ProtocolLimitsV0_6::default()).unwrap();
        let normalized = validate_output_v0_6(
            &context,
            &OperationPlanV0_6::Route(RoutePlan {
                decisions: vec![RouteDecision::Assign(1), RouteDecision::Assign(2)],
            }),
            &admitted_state,
            &StateUpdate {
                shared: None,
                groups: Vec::new(),
                requests: Vec::new(),
            },
            ProtocolLimitsV0_6::default(),
        )
        .unwrap();
        assert!(matches!(
            normalized,
            NormalizedPlanV0_6::Route(NormalizedRoutePlanV0_6 {
                assignments,
                deferred
            }) if assignments.len() == 2 && deferred.is_empty()
        ));

        let schedule = OperationContextV0_6::Schedule(ScheduleContext {
            meta: meta("schedule", 0),
            cause: ScheduleCause::CapacityChanged,
            runnable: ["A", "B"]
                .into_iter()
                .map(|id| ScheduleCandidate {
                    request: request_ref(id, Some("G")),
                    max_token_budget: 4,
                    facts: json!({}),
                })
                .collect(),
            capacity: ScheduleCapacity {
                max_selections: 1,
                max_requests: 2,
                max_total_tokens: 8,
                facts: json!({}),
            },
        });
        let active = state(RequestStatus::Active);
        validate_output_v0_6(
            &schedule,
            &OperationPlanV0_6::Schedule(SchedulePlan {
                selections: vec![ScheduleSelection {
                    requests: vec![0, 1],
                    token_budgets: vec![4, 4],
                }],
            }),
            &active,
            &StateUpdate {
                shared: None,
                groups: Vec::new(),
                requests: Vec::new(),
            },
            ProtocolLimitsV0_6::default(),
        )
        .unwrap();
    }

    #[test]
    fn cache_bypass_and_episode_order_are_explicit() {
        let tracker = CacheEpisodeTrackerV0_6::new(4).unwrap();
        let context = |opportunity: &str, iteration| CacheContext {
            meta: meta(opportunity, 0),
            cause: CacheCause::DependencyProgress,
            resident: Vec::new(),
            prospective: vec![CacheObject {
                object_id: format!("object-{iteration}").into(),
                size_bytes: 1,
                beneficiaries: Vec::new(),
                beneficiary_count: 0,
                facts: json!({}),
            }],
            capacity: CacheCapacity {
                max_bytes: 1,
                fixed_bytes: 0,
                facts: json!({}),
            },
            episode: Some(CacheEpisode {
                episode_id: "episode".into(),
                iteration,
                max_iterations: 3,
            }),
        };
        tracker.observe(&context("op-0", 0)).unwrap();
        tracker.observe(&context("op-0", 0)).unwrap();
        tracker.observe(&context("op-1", 1)).unwrap();
        assert!(matches!(
            tracker.observe(&context("op-2", 1)),
            Err(ProtocolErrorV0_6::CacheEpisodeIteration { .. })
        ));

        let insertion = OperationContextV0_6::Cache(CacheContext {
            meta: meta("insert", 0),
            cause: CacheCause::Insertion,
            resident: Vec::new(),
            prospective: vec![CacheObject {
                object_id: "prospective".into(),
                size_bytes: 1,
                beneficiaries: Vec::new(),
                beneficiary_count: 0,
                facts: json!({}),
            }],
            capacity: CacheCapacity {
                max_bytes: 10,
                fixed_bytes: 0,
                facts: json!({}),
            },
            episode: None,
        });
        validate_output_v0_6(
            &insertion,
            &OperationPlanV0_6::Cache(CachePlan {
                admissions: vec![CacheAdmission::Bypass],
                reclaim: Vec::new(),
            }),
            &PolicyState {
                shared: json!({}),
                groups: Vec::new(),
                requests: Vec::new(),
            },
            &StateUpdate {
                shared: None,
                groups: Vec::new(),
                requests: Vec::new(),
            },
            ProtocolLimitsV0_6::default(),
        )
        .unwrap();
    }

    #[test]
    fn opportunity_retries_preserve_boundary_and_order() {
        let tracker = OpportunityTrackerV0_6::new(8).unwrap();
        let context = |attempt, candidates: Vec<&str>| {
            OperationContextV0_6::Admit(AdmitContext {
                meta: meta("admit", attempt),
                cause: AdmitCause::Retry,
                candidates: candidates
                    .into_iter()
                    .map(|id| AdmissionCandidate {
                        request: request_ref(id, Some("G")),
                        demand: Vec::new(),
                        facts: json!({}),
                    })
                    .collect(),
                capacity: AdmissionCapacity {
                    max_accepted: 2,
                    limits: vec![ResourceLimit {
                        name: "kv".into(),
                        unit: "bytes".into(),
                        maximum: 0,
                    }],
                    facts: json!({}),
                },
            })
        };
        tracker.begin(&context(0, vec!["A", "B"])).unwrap();
        tracker.begin(&context(1, vec!["A", "B"])).unwrap();
        assert!(matches!(
            tracker.begin(&context(2, vec!["B", "A"])),
            Err(ProtocolErrorV0_6::OpportunityBoundaryChanged(_))
        ));
        tracker.complete(&"admit".into()).unwrap();
        assert!(matches!(
            tracker.begin(&context(2, vec!["A", "B"])),
            Err(ProtocolErrorV0_6::CompletedOpportunityReused(_))
        ));
    }

    #[test]
    fn snapshot_reference_covers_all_scope_revisions() {
        let backend = InMemoryPolicyStateBackendV0_6::default();
        backend
            .create_group(
                "G".into(),
                "tenant".into(),
                GroupLimits {
                    max_members: 1,
                    max_scratch_bytes: 1024,
                },
                json!({}),
            )
            .unwrap();
        backend
            .create_request(
                "A".into(),
                "tenant".into(),
                Some("G".into()),
                json!({}),
                json!({}),
            )
            .unwrap();
        let snapshot = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        let first = snapshot_ref_v0_6(&snapshot).unwrap();
        let context = OperationContextV0_6::Admit(AdmitContext {
            meta: DecisionMeta {
                opportunity_id: "admit".into(),
                snapshot: first.clone(),
                attempt: 0,
                mechanics: Vec::new(),
            },
            cause: AdmitCause::Arrival,
            candidates: vec![AdmissionCandidate {
                request: snapshot.state.requests[0].request.clone(),
                demand: Vec::new(),
                facts: json!({}),
            }],
            capacity: AdmissionCapacity {
                max_accepted: 1,
                limits: Vec::new(),
                facts: json!({}),
            },
        });
        validate_snapshot_context_v0_6(&context, &snapshot, ProtocolLimitsV0_6::default()).unwrap();
        backend
            .merge_group_facts(&"G".into(), json!({"load": 1}))
            .unwrap();
        let snapshot = backend
            .load(&WorkingSetV0_6::default().with_request("A"))
            .unwrap();
        let second = snapshot_ref_v0_6(&snapshot).unwrap();
        assert_ne!(first, second);
        assert!(matches!(
            validate_snapshot_context_v0_6(&context, &snapshot, ProtocolLimitsV0_6::default()),
            Err(ProtocolErrorV0_6::SnapshotMismatch { .. })
        ));
    }

    #[test]
    fn protocol_limits_reject_oversized_sets() {
        let context = OperationContextV0_6::Admit(AdmitContext {
            meta: meta("admit", 0),
            cause: AdmitCause::Arrival,
            candidates: vec![
                AdmissionCandidate {
                    request: request_ref("A", Some("G")),
                    demand: Vec::new(),
                    facts: json!({}),
                },
                AdmissionCandidate {
                    request: request_ref("B", Some("G")),
                    demand: Vec::new(),
                    facts: json!({}),
                },
            ],
            capacity: AdmissionCapacity {
                max_accepted: 2,
                limits: Vec::new(),
                facts: json!({}),
            },
        });
        assert!(matches!(
            working_set_v0_6(
                &context,
                ProtocolLimitsV0_6 {
                    max_requests: 1,
                    ..ProtocolLimitsV0_6::default()
                }
            ),
            Err(ProtocolErrorV0_6::LimitExceeded {
                field: "context requests",
                ..
            })
        ));
    }

    #[test]
    fn replay_record_preserves_input_order() {
        let context = OperationContextV0_6::Admit(AdmitContext {
            meta: meta("admit", 0),
            cause: AdmitCause::Arrival,
            candidates: ["B", "A"]
                .into_iter()
                .map(|id| AdmissionCandidate {
                    request: request_ref(id, Some("G")),
                    demand: Vec::new(),
                    facts: json!({}),
                })
                .collect(),
            capacity: AdmissionCapacity {
                max_accepted: 2,
                limits: Vec::new(),
                facts: json!({}),
            },
        });
        let plan = AdmitPlan {
            decisions: vec![AdmissionDecision::Accept, AdmissionDecision::Defer],
        };
        let record = replay_record_v0_6(
            &context,
            NormalizedPlanV0_6::Admit(plan),
            StateUpdate {
                shared: None,
                groups: Vec::new(),
                requests: Vec::new(),
            },
        );
        assert_eq!(
            record.order.request_ids,
            vec![RequestId::from("B"), RequestId::from("A")]
        );
    }
}
