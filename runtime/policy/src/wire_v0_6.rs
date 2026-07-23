use pie_plex::Document;
use pie_plex::v0_6 as model;
use serde_json::Value;
use thiserror::Error;

use crate::bindings_v0_6::exports::pie::plex::policy;
use crate::bindings_v0_6::exports::pie::plex::policy as wire;

pub(crate) fn admit_invocation(
    context: &model::AdmitContext,
    state: &model::PolicyState,
) -> Result<policy::AdmitInvocation, WireErrorV0_6> {
    Ok(policy::AdmitInvocation {
        context: admit_context(context)?,
        state: policy_state(state)?,
    })
}

pub(crate) fn route_invocation(
    context: &model::RouteContext,
    state: &model::PolicyState,
) -> Result<policy::RouteInvocation, WireErrorV0_6> {
    Ok(policy::RouteInvocation {
        context: route_context(context)?,
        state: policy_state(state)?,
    })
}

pub(crate) fn schedule_invocation(
    context: &model::ScheduleContext,
    state: &model::PolicyState,
) -> Result<policy::ScheduleInvocation, WireErrorV0_6> {
    Ok(policy::ScheduleInvocation {
        context: schedule_context(context)?,
        state: policy_state(state)?,
    })
}

pub(crate) fn cache_invocation(
    context: &model::CacheContext,
    state: &model::PolicyState,
) -> Result<policy::CacheInvocation, WireErrorV0_6> {
    Ok(policy::CacheInvocation {
        context: cache_context(context)?,
        state: policy_state(state)?,
    })
}

pub(crate) fn feedback_invocation(
    context: &model::FeedbackContext,
    state: &model::PolicyState,
) -> Result<policy::FeedbackInvocation, WireErrorV0_6> {
    Ok(policy::FeedbackInvocation {
        context: feedback_context(context)?,
        state: policy_state(state)?,
    })
}

pub(crate) fn admit_output(
    output: policy::AdmitOutput,
) -> Result<(model::AdmitPlan, model::StateUpdate), WireErrorV0_6> {
    Ok((
        model::AdmitPlan {
            decisions: output
                .plan
                .decisions
                .into_iter()
                .map(admission_decision_from_wire)
                .collect(),
        },
        state_update_from_wire(output.state_update)?,
    ))
}

pub(crate) fn route_output(
    output: policy::RouteOutput,
) -> Result<(model::RoutePlan, model::StateUpdate), WireErrorV0_6> {
    Ok((
        model::RoutePlan {
            decisions: output
                .plan
                .decisions
                .into_iter()
                .map(route_decision_from_wire)
                .collect(),
        },
        state_update_from_wire(output.state_update)?,
    ))
}

pub(crate) fn schedule_output(
    output: policy::ScheduleOutput,
) -> Result<(model::SchedulePlan, model::StateUpdate), WireErrorV0_6> {
    Ok((
        model::SchedulePlan {
            selections: output
                .plan
                .selections
                .into_iter()
                .map(|selection| model::ScheduleSelection {
                    requests: selection.requests,
                    token_budgets: selection.token_budgets,
                })
                .collect(),
        },
        state_update_from_wire(output.state_update)?,
    ))
}

pub(crate) fn cache_output(
    output: policy::CacheOutput,
) -> Result<(model::CachePlan, model::StateUpdate), WireErrorV0_6> {
    Ok((
        model::CachePlan {
            admissions: output
                .plan
                .admissions
                .into_iter()
                .map(cache_admission_from_wire)
                .collect(),
            reclaim: output.plan.reclaim,
        },
        state_update_from_wire(output.state_update)?,
    ))
}

pub(crate) fn feedback_output(
    output: policy::FeedbackOutput,
) -> Result<model::StateUpdate, WireErrorV0_6> {
    state_update_from_wire(output.state_update)
}

pub(crate) fn policy_error_from_wire(
    error: wire::PolicyError,
) -> Result<model::PolicyError, WireErrorV0_6> {
    let error = model::PolicyError {
        code: error.code,
        message: error.message,
        details: document_from_wire("policy error details", error.details)?,
    };
    model::validate_policy_error(&error)
        .map_err(|error| WireErrorV0_6::InvalidPolicyError(error.to_string()))?;
    Ok(error)
}

fn policy_state(state: &model::PolicyState) -> Result<wire::PolicyState, WireErrorV0_6> {
    Ok(wire::PolicyState {
        shared: document_to_wire("state shared", &state.shared)?,
        groups: state
            .groups
            .iter()
            .map(group_state)
            .collect::<Result<_, _>>()?,
        requests: state
            .requests
            .iter()
            .map(request_state)
            .collect::<Result<_, _>>()?,
    })
}

fn group_state(state: &model::GroupState) -> Result<wire::GroupState, WireErrorV0_6> {
    Ok(wire::GroupState {
        group_id: state.group_id.0.clone(),
        principal_id: state.principal_id.0.clone(),
        status: group_status_to_wire(state.status),
        limits: wire::GroupLimits {
            max_members: state.limits.max_members,
            max_scratch_bytes: state.limits.max_scratch_bytes,
        },
        member_count: state.member_count,
        facts: document_to_wire("group facts", &state.facts)?,
        scratch: document_to_wire("group scratch", &state.scratch)?,
    })
}

fn request_state(state: &model::RequestState) -> Result<wire::RequestState, WireErrorV0_6> {
    Ok(wire::RequestState {
        request: request_ref(&state.request),
        status: request_status_to_wire(state.status),
        facts: document_to_wire("request facts", &state.facts)?,
        fields: document_to_wire("request fields", &state.fields)?,
        scratch: document_to_wire("request scratch", &state.scratch)?,
    })
}

fn request_ref(request: &model::RequestRef) -> wire::RequestRef {
    wire::RequestRef {
        request_id: request.request_id.0.clone(),
        generation_id: request.generation_id,
        group_id: request.group_id.as_ref().map(|group| group.0.clone()),
        principal_id: request.principal_id.0.clone(),
    }
}

fn decision_meta(meta: &model::DecisionMeta) -> wire::DecisionMeta {
    wire::DecisionMeta {
        opportunity_id: meta.opportunity_id.0.clone(),
        snapshot: wire::SnapshotRef {
            id: meta.snapshot.id.0.clone(),
            revision: meta.snapshot.revision,
        },
        attempt: meta.attempt,
        mechanics: meta
            .mechanics
            .iter()
            .map(|mechanic| mechanic.0.clone())
            .collect(),
    }
}

fn resource_amount(amount: &model::ResourceAmount) -> wire::ResourceAmount {
    wire::ResourceAmount {
        name: amount.name.clone(),
        unit: amount.unit.clone(),
        amount: amount.amount,
    }
}

fn resource_limit(limit: &model::ResourceLimit) -> wire::ResourceLimit {
    wire::ResourceLimit {
        name: limit.name.clone(),
        unit: limit.unit.clone(),
        maximum: limit.maximum,
    }
}

fn admit_context(context: &model::AdmitContext) -> Result<wire::AdmitContext, WireErrorV0_6> {
    Ok(wire::AdmitContext {
        meta: decision_meta(&context.meta),
        cause: match context.cause {
            model::AdmitCause::Arrival => wire::AdmitCause::Arrival,
            model::AdmitCause::Retry => wire::AdmitCause::Retry,
            model::AdmitCause::CapacityChanged => wire::AdmitCause::CapacityChanged,
        },
        candidates: context
            .candidates
            .iter()
            .map(|candidate| {
                Ok(wire::AdmissionCandidate {
                    request: request_ref(&candidate.request),
                    demand: candidate.demand.iter().map(resource_amount).collect(),
                    facts: document_to_wire("admission candidate facts", &candidate.facts)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
        capacity: wire::AdmissionCapacity {
            max_accepted: context.capacity.max_accepted,
            limits: context.capacity.limits.iter().map(resource_limit).collect(),
            facts: document_to_wire("admission capacity facts", &context.capacity.facts)?,
        },
    })
}

fn route_context(context: &model::RouteContext) -> Result<wire::RouteContext, WireErrorV0_6> {
    Ok(wire::RouteContext {
        meta: decision_meta(&context.meta),
        cause: match context.cause {
            model::RouteCause::Admission => wire::RouteCause::Admission,
            model::RouteCause::Retry => wire::RouteCause::Retry,
            model::RouteCause::Rebalance => wire::RouteCause::Rebalance,
            model::RouteCause::TargetChanged => wire::RouteCause::TargetChanged,
        },
        requests: context
            .requests
            .iter()
            .map(|request| {
                Ok(wire::RouteRequest {
                    request: request_ref(&request.request),
                    facts: document_to_wire("route request facts", &request.facts)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
        targets: context
            .targets
            .iter()
            .map(|target| {
                Ok(wire::RouteTarget {
                    target_id: target.target_id.0.clone(),
                    max_assignments: target.max_assignments,
                    capacity: target.capacity.iter().map(resource_limit).collect(),
                    revision: target.revision,
                    facts: document_to_wire("route target facts", &target.facts)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
        feasible_edges: context
            .feasible_edges
            .iter()
            .map(|edge| {
                Ok(wire::RouteEdge {
                    request_index: edge.request_index,
                    target_index: edge.target_index,
                    demand: edge.demand.iter().map(resource_amount).collect(),
                    facts: document_to_wire("route edge facts", &edge.facts)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
    })
}

fn schedule_context(
    context: &model::ScheduleContext,
) -> Result<wire::ScheduleContext, WireErrorV0_6> {
    Ok(wire::ScheduleContext {
        meta: decision_meta(&context.meta),
        cause: match context.cause {
            model::ScheduleCause::Arrival => wire::ScheduleCause::Arrival,
            model::ScheduleCause::Completion => wire::ScheduleCause::Completion,
            model::ScheduleCause::CapacityChanged => wire::ScheduleCause::CapacityChanged,
            model::ScheduleCause::Timer => wire::ScheduleCause::Timer,
            model::ScheduleCause::Feedback => wire::ScheduleCause::Feedback,
        },
        runnable: context
            .runnable
            .iter()
            .map(|candidate| {
                Ok(wire::ScheduleCandidate {
                    request: request_ref(&candidate.request),
                    max_token_budget: candidate.max_token_budget,
                    facts: document_to_wire("schedule candidate facts", &candidate.facts)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
        capacity: wire::ScheduleCapacity {
            max_selections: context.capacity.max_selections,
            max_requests: context.capacity.max_requests,
            max_total_tokens: context.capacity.max_total_tokens,
            facts: document_to_wire("schedule capacity facts", &context.capacity.facts)?,
        },
    })
}

fn cache_context(context: &model::CacheContext) -> Result<wire::CacheContext, WireErrorV0_6> {
    Ok(wire::CacheContext {
        meta: decision_meta(&context.meta),
        cause: match context.cause {
            model::CacheCause::Insertion => wire::CacheCause::Insertion,
            model::CacheCause::Pressure => wire::CacheCause::Pressure,
            model::CacheCause::Expiry => wire::CacheCause::Expiry,
            model::CacheCause::DependencyProgress => wire::CacheCause::DependencyProgress,
        },
        resident: context
            .resident
            .iter()
            .map(|resident| {
                Ok(wire::ResidentCacheObject {
                    object: cache_object(&resident.object)?,
                    reclaimable: resident.reclaimable,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
        prospective: context
            .prospective
            .iter()
            .map(cache_object)
            .collect::<Result<_, _>>()?,
        capacity: wire::CacheCapacity {
            max_bytes: context.capacity.max_bytes,
            fixed_bytes: context.capacity.fixed_bytes,
            facts: document_to_wire("cache capacity facts", &context.capacity.facts)?,
        },
        episode: context.episode.as_ref().map(|episode| wire::CacheEpisode {
            episode_id: episode.episode_id.0.clone(),
            iteration: episode.iteration,
            max_iterations: episode.max_iterations,
        }),
    })
}

fn cache_object(object: &model::CacheObject) -> Result<wire::CacheObject, WireErrorV0_6> {
    Ok(wire::CacheObject {
        object_id: object.object_id.0.clone(),
        size_bytes: object.size_bytes,
        beneficiaries: object
            .beneficiaries
            .iter()
            .map(|beneficiary| match beneficiary {
                model::Beneficiary::Request(request) => {
                    wire::Beneficiary::Request(request.0.clone())
                }
                model::Beneficiary::Group(group) => wire::Beneficiary::Group(group.0.clone()),
            })
            .collect(),
        beneficiary_count: object.beneficiary_count,
        facts: document_to_wire("cache object facts", &object.facts)?,
    })
}

fn feedback_context(
    context: &model::FeedbackContext,
) -> Result<wire::FeedbackContext, WireErrorV0_6> {
    Ok(wire::FeedbackContext {
        delivery_id: context.delivery_id.0.clone(),
        records: context
            .records
            .iter()
            .map(|record| {
                Ok(wire::FeedbackRecord {
                    subject: feedback_subject(&record.subject),
                    outcome: match record.outcome {
                        model::OutcomeKind::Progress => wire::OutcomeKind::Progress,
                        model::OutcomeKind::Completed => wire::OutcomeKind::Completed,
                        model::OutcomeKind::Failed => wire::OutcomeKind::Failed,
                        model::OutcomeKind::Cancelled => wire::OutcomeKind::Cancelled,
                        model::OutcomeKind::Expired => wire::OutcomeKind::Expired,
                        model::OutcomeKind::ActionSucceeded => wire::OutcomeKind::ActionSucceeded,
                        model::OutcomeKind::ActionFailed => wire::OutcomeKind::ActionFailed,
                    },
                    facts: document_to_wire("feedback facts", &record.facts)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
    })
}

fn feedback_subject(subject: &model::FeedbackSubject) -> wire::FeedbackSubject {
    match subject {
        model::FeedbackSubject::Request(request) => {
            wire::FeedbackSubject::Request(request.0.clone())
        }
        model::FeedbackSubject::WorkGroup(group) => {
            wire::FeedbackSubject::WorkGroup(group.0.clone())
        }
        model::FeedbackSubject::CacheObject(object) => {
            wire::FeedbackSubject::CacheObject(object.0.clone())
        }
        model::FeedbackSubject::RouteAssignment(subject) => {
            wire::FeedbackSubject::RouteAssignment(wire::RouteAssignmentSubject {
                opportunity_id: subject.opportunity_id.0.clone(),
                request_index: subject.request_index,
            })
        }
        model::FeedbackSubject::ScheduleSelection(subject) => {
            wire::FeedbackSubject::ScheduleSelection(wire::ScheduleSelectionSubject {
                opportunity_id: subject.opportunity_id.0.clone(),
                selection_index: subject.selection_index,
            })
        }
        model::FeedbackSubject::Action(action) => wire::FeedbackSubject::Action(action.0),
    }
}

fn state_update_from_wire(update: wire::StateUpdate) -> Result<model::StateUpdate, WireErrorV0_6> {
    Ok(model::StateUpdate {
        shared: update
            .shared
            .map(|document| document_from_wire("shared update", document))
            .transpose()?,
        groups: update
            .groups
            .into_iter()
            .map(|group| {
                Ok(model::GroupStateUpdate {
                    group_id: group.group_id.into(),
                    scratch: document_from_wire("group scratch update", group.scratch)?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
        requests: update
            .requests
            .into_iter()
            .map(|request| {
                Ok(model::RequestStateUpdate {
                    request_id: request.request_id.into(),
                    fields: request
                        .fields
                        .map(|document| document_from_wire("request fields update", document))
                        .transpose()?,
                    scratch: request
                        .scratch
                        .map(|document| document_from_wire("request scratch update", document))
                        .transpose()?,
                })
            })
            .collect::<Result<_, WireErrorV0_6>>()?,
    })
}

fn admission_decision_from_wire(decision: wire::AdmissionDecision) -> model::AdmissionDecision {
    match decision {
        wire::AdmissionDecision::Accept => model::AdmissionDecision::Accept,
        wire::AdmissionDecision::Defer => model::AdmissionDecision::Defer,
        wire::AdmissionDecision::Reject => model::AdmissionDecision::Reject,
    }
}

fn route_decision_from_wire(decision: wire::RouteDecision) -> model::RouteDecision {
    match decision {
        wire::RouteDecision::Assign(edge) => model::RouteDecision::Assign(edge),
        wire::RouteDecision::Defer => model::RouteDecision::Defer,
    }
}

fn cache_admission_from_wire(admission: wire::CacheAdmission) -> model::CacheAdmission {
    match admission {
        wire::CacheAdmission::Cache => model::CacheAdmission::Cache,
        wire::CacheAdmission::Bypass => model::CacheAdmission::Bypass,
    }
}

fn group_status_to_wire(status: model::GroupStatus) -> wire::GroupStatus {
    match status {
        model::GroupStatus::Open => wire::GroupStatus::Open,
        model::GroupStatus::Closed => wire::GroupStatus::Closed,
        model::GroupStatus::Cancelled => wire::GroupStatus::Cancelled,
        model::GroupStatus::Expired => wire::GroupStatus::Expired,
    }
}

fn request_status_to_wire(status: model::RequestStatus) -> wire::RequestStatus {
    match status {
        model::RequestStatus::Pending => wire::RequestStatus::Pending,
        model::RequestStatus::Admitted => wire::RequestStatus::Admitted,
        model::RequestStatus::Active => wire::RequestStatus::Active,
        model::RequestStatus::Paused => wire::RequestStatus::Paused,
        model::RequestStatus::Completed => wire::RequestStatus::Completed,
        model::RequestStatus::Failed => wire::RequestStatus::Failed,
        model::RequestStatus::Cancelled => wire::RequestStatus::Cancelled,
        model::RequestStatus::Expired => wire::RequestStatus::Expired,
        model::RequestStatus::Rejected => wire::RequestStatus::Rejected,
    }
}

fn document_to_wire(field: &'static str, document: &Document) -> Result<String, WireErrorV0_6> {
    if !document.is_object() {
        return Err(WireErrorV0_6::DocumentNotObject(field));
    }
    serde_json::to_string(document).map_err(WireErrorV0_6::EncodeDocument)
}

fn document_from_wire(field: &'static str, document: String) -> Result<Document, WireErrorV0_6> {
    let value: Value = serde_json::from_str(&document)
        .map_err(|source| WireErrorV0_6::DecodeDocument { field, source })?;
    if !value.is_object() {
        return Err(WireErrorV0_6::DocumentNotObject(field));
    }
    Ok(value)
}

#[derive(Debug, Error)]
pub(crate) enum WireErrorV0_6 {
    #[error("{0} must be a JSON object")]
    DocumentNotObject(&'static str),
    #[error("failed to encode document")]
    EncodeDocument(#[source] serde_json::Error),
    #[error("failed to decode {field}")]
    DecodeDocument {
        field: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("guest returned an invalid policy error: {0}")]
    InvalidPolicyError(String),
}
