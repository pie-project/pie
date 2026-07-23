use pie_plex::Document;
use pie_plex::v0_6 as model;
use serde_json::Value;

use crate::exports::pie::plex::policy;
use crate::exports::pie::plex::policy as wire;

pub(crate) fn admit_invocation_from_wire(
    input: policy::AdmitInvocation,
) -> Result<(model::AdmitContext, model::PolicyState), wire::PolicyError> {
    let state = policy_state_from_wire(input.state)?;
    let context = admit_context_from_wire(input.context)?;
    model::validate_admit_context(&state, &context)
        .map_err(|error| wire_error("invalid-admit-context", error.to_string()))?;
    Ok((context, state))
}

pub(crate) fn route_invocation_from_wire(
    input: policy::RouteInvocation,
) -> Result<(model::RouteContext, model::PolicyState), wire::PolicyError> {
    let state = policy_state_from_wire(input.state)?;
    let context = route_context_from_wire(input.context)?;
    model::validate_route_context(&state, &context)
        .map_err(|error| wire_error("invalid-route-context", error.to_string()))?;
    Ok((context, state))
}

pub(crate) fn schedule_invocation_from_wire(
    input: policy::ScheduleInvocation,
) -> Result<(model::ScheduleContext, model::PolicyState), wire::PolicyError> {
    let state = policy_state_from_wire(input.state)?;
    let context = schedule_context_from_wire(input.context)?;
    model::validate_schedule_context(&state, &context)
        .map_err(|error| wire_error("invalid-schedule-context", error.to_string()))?;
    Ok((context, state))
}

pub(crate) fn cache_invocation_from_wire(
    input: policy::CacheInvocation,
) -> Result<(model::CacheContext, model::PolicyState), wire::PolicyError> {
    let state = policy_state_from_wire(input.state)?;
    let context = cache_context_from_wire(input.context)?;
    model::validate_cache_context(&context)
        .map_err(|error| wire_error("invalid-cache-context", error.to_string()))?;
    Ok((context, state))
}

pub(crate) fn feedback_invocation_from_wire(
    input: policy::FeedbackInvocation,
) -> Result<(model::FeedbackContext, model::PolicyState), wire::PolicyError> {
    let state = policy_state_from_wire(input.state)?;
    let context = feedback_context_from_wire(input.context)?;
    model::validate_feedback_context(&context)
        .map_err(|error| wire_error("invalid-feedback-context", error.to_string()))?;
    Ok((context, state))
}

pub(crate) fn admit_output_to_wire(
    plan: model::AdmitPlan,
    state_update: model::StateUpdate,
) -> Result<policy::AdmitOutput, wire::PolicyError> {
    Ok(policy::AdmitOutput {
        plan: wire::AdmitPlan {
            decisions: plan
                .decisions
                .into_iter()
                .map(admission_decision_to_wire)
                .collect(),
        },
        state_update: state_update_to_wire(state_update)?,
    })
}

pub(crate) fn route_output_to_wire(
    plan: model::RoutePlan,
    state_update: model::StateUpdate,
) -> Result<policy::RouteOutput, wire::PolicyError> {
    Ok(policy::RouteOutput {
        plan: wire::RoutePlan {
            decisions: plan
                .decisions
                .into_iter()
                .map(route_decision_to_wire)
                .collect(),
        },
        state_update: state_update_to_wire(state_update)?,
    })
}

pub(crate) fn schedule_output_to_wire(
    plan: model::SchedulePlan,
    state_update: model::StateUpdate,
) -> Result<policy::ScheduleOutput, wire::PolicyError> {
    Ok(policy::ScheduleOutput {
        plan: wire::SchedulePlan {
            selections: plan
                .selections
                .into_iter()
                .map(|selection| wire::ScheduleSelection {
                    requests: selection.requests,
                    token_budgets: selection.token_budgets,
                })
                .collect(),
        },
        state_update: state_update_to_wire(state_update)?,
    })
}

pub(crate) fn cache_output_to_wire(
    plan: model::CachePlan,
    state_update: model::StateUpdate,
) -> Result<policy::CacheOutput, wire::PolicyError> {
    Ok(policy::CacheOutput {
        plan: wire::CachePlan {
            admissions: plan
                .admissions
                .into_iter()
                .map(cache_admission_to_wire)
                .collect(),
            reclaim: plan.reclaim,
        },
        state_update: state_update_to_wire(state_update)?,
    })
}

pub(crate) fn feedback_output_to_wire(
    state_update: model::StateUpdate,
) -> Result<policy::FeedbackOutput, wire::PolicyError> {
    Ok(policy::FeedbackOutput {
        state_update: state_update_to_wire(state_update)?,
    })
}

pub(crate) fn policy_error_to_wire(error: model::PolicyError) -> wire::PolicyError {
    let fallback = wire::PolicyError {
        code: "invalid-policy-error".into(),
        message: "policy returned an invalid error".into(),
        details: "{}".into(),
    };
    if model::validate_policy_error(&error).is_err() {
        return fallback;
    }
    match document_to_wire("policy error details", &error.details) {
        Ok(details) => wire::PolicyError {
            code: error.code,
            message: error.message,
            details,
        },
        Err(_) => fallback,
    }
}

fn policy_state_from_wire(
    state: wire::PolicyState,
) -> Result<model::PolicyState, wire::PolicyError> {
    let state = model::PolicyState {
        shared: document_from_wire("state shared", state.shared)?,
        groups: state
            .groups
            .into_iter()
            .map(group_state_from_wire)
            .collect::<Result<_, _>>()?,
        requests: state
            .requests
            .into_iter()
            .map(request_state_from_wire)
            .collect::<Result<_, _>>()?,
    };
    model::validate_policy_state(&state)
        .map_err(|error| wire_error("invalid-policy-state", error.to_string()))?;
    Ok(state)
}

fn group_state_from_wire(state: wire::GroupState) -> Result<model::GroupState, wire::PolicyError> {
    Ok(model::GroupState {
        group_id: state.group_id.into(),
        principal_id: state.principal_id.into(),
        status: group_status_from_wire(state.status),
        limits: model::GroupLimits {
            max_members: state.limits.max_members,
            max_scratch_bytes: state.limits.max_scratch_bytes,
        },
        member_count: state.member_count,
        facts: document_from_wire("group facts", state.facts)?,
        scratch: document_from_wire("group scratch", state.scratch)?,
    })
}

fn request_state_from_wire(
    state: wire::RequestState,
) -> Result<model::RequestState, wire::PolicyError> {
    Ok(model::RequestState {
        request: request_ref_from_wire(state.request),
        status: request_status_from_wire(state.status),
        facts: document_from_wire("request facts", state.facts)?,
        fields: document_from_wire("request fields", state.fields)?,
        scratch: document_from_wire("request scratch", state.scratch)?,
    })
}

fn request_ref_from_wire(request: wire::RequestRef) -> model::RequestRef {
    model::RequestRef {
        request_id: request.request_id.into(),
        generation_id: request.generation_id,
        group_id: request.group_id.map(Into::into),
        principal_id: request.principal_id.into(),
    }
}

fn decision_meta_from_wire(meta: wire::DecisionMeta) -> model::DecisionMeta {
    model::DecisionMeta {
        opportunity_id: meta.opportunity_id.into(),
        snapshot: model::SnapshotRef {
            id: meta.snapshot.id.into(),
            revision: meta.snapshot.revision,
        },
        attempt: meta.attempt,
        mechanics: meta.mechanics.into_iter().map(Into::into).collect(),
    }
}

fn resource_amount_from_wire(amount: wire::ResourceAmount) -> model::ResourceAmount {
    model::ResourceAmount {
        name: amount.name,
        unit: amount.unit,
        amount: amount.amount,
    }
}

fn resource_limit_from_wire(limit: wire::ResourceLimit) -> model::ResourceLimit {
    model::ResourceLimit {
        name: limit.name,
        unit: limit.unit,
        maximum: limit.maximum,
    }
}

fn admit_context_from_wire(
    context: wire::AdmitContext,
) -> Result<model::AdmitContext, wire::PolicyError> {
    Ok(model::AdmitContext {
        meta: decision_meta_from_wire(context.meta),
        cause: match context.cause {
            wire::AdmitCause::Arrival => model::AdmitCause::Arrival,
            wire::AdmitCause::Retry => model::AdmitCause::Retry,
            wire::AdmitCause::CapacityChanged => model::AdmitCause::CapacityChanged,
        },
        candidates: context
            .candidates
            .into_iter()
            .map(|candidate| {
                Ok(model::AdmissionCandidate {
                    request: request_ref_from_wire(candidate.request),
                    demand: candidate
                        .demand
                        .into_iter()
                        .map(resource_amount_from_wire)
                        .collect(),
                    facts: document_from_wire("admission candidate facts", candidate.facts)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
        capacity: model::AdmissionCapacity {
            max_accepted: context.capacity.max_accepted,
            limits: context
                .capacity
                .limits
                .into_iter()
                .map(resource_limit_from_wire)
                .collect(),
            facts: document_from_wire("admission capacity facts", context.capacity.facts)?,
        },
    })
}

fn route_context_from_wire(
    context: wire::RouteContext,
) -> Result<model::RouteContext, wire::PolicyError> {
    Ok(model::RouteContext {
        meta: decision_meta_from_wire(context.meta),
        cause: match context.cause {
            wire::RouteCause::Admission => model::RouteCause::Admission,
            wire::RouteCause::Retry => model::RouteCause::Retry,
            wire::RouteCause::Rebalance => model::RouteCause::Rebalance,
            wire::RouteCause::TargetChanged => model::RouteCause::TargetChanged,
        },
        requests: context
            .requests
            .into_iter()
            .map(|request| {
                Ok(model::RouteRequest {
                    request: request_ref_from_wire(request.request),
                    facts: document_from_wire("route request facts", request.facts)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
        targets: context
            .targets
            .into_iter()
            .map(|target| {
                Ok(model::RouteTarget {
                    target_id: target.target_id.into(),
                    max_assignments: target.max_assignments,
                    capacity: target
                        .capacity
                        .into_iter()
                        .map(resource_limit_from_wire)
                        .collect(),
                    revision: target.revision,
                    facts: document_from_wire("route target facts", target.facts)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
        feasible_edges: context
            .feasible_edges
            .into_iter()
            .map(|edge| {
                Ok(model::RouteEdge {
                    request_index: edge.request_index,
                    target_index: edge.target_index,
                    demand: edge
                        .demand
                        .into_iter()
                        .map(resource_amount_from_wire)
                        .collect(),
                    facts: document_from_wire("route edge facts", edge.facts)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
    })
}

fn schedule_context_from_wire(
    context: wire::ScheduleContext,
) -> Result<model::ScheduleContext, wire::PolicyError> {
    Ok(model::ScheduleContext {
        meta: decision_meta_from_wire(context.meta),
        cause: match context.cause {
            wire::ScheduleCause::Arrival => model::ScheduleCause::Arrival,
            wire::ScheduleCause::Completion => model::ScheduleCause::Completion,
            wire::ScheduleCause::CapacityChanged => model::ScheduleCause::CapacityChanged,
            wire::ScheduleCause::Timer => model::ScheduleCause::Timer,
            wire::ScheduleCause::Feedback => model::ScheduleCause::Feedback,
        },
        runnable: context
            .runnable
            .into_iter()
            .map(|candidate| {
                Ok(model::ScheduleCandidate {
                    request: request_ref_from_wire(candidate.request),
                    max_token_budget: candidate.max_token_budget,
                    facts: document_from_wire("schedule candidate facts", candidate.facts)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
        capacity: model::ScheduleCapacity {
            max_selections: context.capacity.max_selections,
            max_requests: context.capacity.max_requests,
            max_total_tokens: context.capacity.max_total_tokens,
            facts: document_from_wire("schedule capacity facts", context.capacity.facts)?,
        },
    })
}

fn cache_context_from_wire(
    context: wire::CacheContext,
) -> Result<model::CacheContext, wire::PolicyError> {
    Ok(model::CacheContext {
        meta: decision_meta_from_wire(context.meta),
        cause: match context.cause {
            wire::CacheCause::Insertion => model::CacheCause::Insertion,
            wire::CacheCause::Pressure => model::CacheCause::Pressure,
            wire::CacheCause::Expiry => model::CacheCause::Expiry,
            wire::CacheCause::DependencyProgress => model::CacheCause::DependencyProgress,
        },
        resident: context
            .resident
            .into_iter()
            .map(|resident| {
                Ok(model::ResidentCacheObject {
                    object: cache_object_from_wire(resident.object)?,
                    reclaimable: resident.reclaimable,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
        prospective: context
            .prospective
            .into_iter()
            .map(cache_object_from_wire)
            .collect::<Result<_, _>>()?,
        capacity: model::CacheCapacity {
            max_bytes: context.capacity.max_bytes,
            fixed_bytes: context.capacity.fixed_bytes,
            facts: document_from_wire("cache capacity facts", context.capacity.facts)?,
        },
        episode: context.episode.map(|episode| model::CacheEpisode {
            episode_id: episode.episode_id.into(),
            iteration: episode.iteration,
            max_iterations: episode.max_iterations,
        }),
    })
}

fn cache_object_from_wire(
    object: wire::CacheObject,
) -> Result<model::CacheObject, wire::PolicyError> {
    Ok(model::CacheObject {
        object_id: object.object_id.into(),
        size_bytes: object.size_bytes,
        beneficiaries: object
            .beneficiaries
            .into_iter()
            .map(|beneficiary| match beneficiary {
                wire::Beneficiary::Request(request) => model::Beneficiary::Request(request.into()),
                wire::Beneficiary::Group(group) => model::Beneficiary::Group(group.into()),
            })
            .collect(),
        beneficiary_count: object.beneficiary_count,
        facts: document_from_wire("cache object facts", object.facts)?,
    })
}

fn feedback_context_from_wire(
    context: wire::FeedbackContext,
) -> Result<model::FeedbackContext, wire::PolicyError> {
    Ok(model::FeedbackContext {
        delivery_id: context.delivery_id.into(),
        records: context
            .records
            .into_iter()
            .map(|record| {
                Ok(model::FeedbackRecord {
                    subject: feedback_subject_from_wire(record.subject),
                    outcome: match record.outcome {
                        wire::OutcomeKind::Progress => model::OutcomeKind::Progress,
                        wire::OutcomeKind::Completed => model::OutcomeKind::Completed,
                        wire::OutcomeKind::Failed => model::OutcomeKind::Failed,
                        wire::OutcomeKind::Cancelled => model::OutcomeKind::Cancelled,
                        wire::OutcomeKind::Expired => model::OutcomeKind::Expired,
                        wire::OutcomeKind::ActionSucceeded => model::OutcomeKind::ActionSucceeded,
                        wire::OutcomeKind::ActionFailed => model::OutcomeKind::ActionFailed,
                    },
                    facts: document_from_wire("feedback facts", record.facts)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
    })
}

fn feedback_subject_from_wire(subject: wire::FeedbackSubject) -> model::FeedbackSubject {
    match subject {
        wire::FeedbackSubject::Request(request) => model::FeedbackSubject::Request(request.into()),
        wire::FeedbackSubject::WorkGroup(group) => model::FeedbackSubject::WorkGroup(group.into()),
        wire::FeedbackSubject::CacheObject(object) => {
            model::FeedbackSubject::CacheObject(object.into())
        }
        wire::FeedbackSubject::RouteAssignment(subject) => {
            model::FeedbackSubject::RouteAssignment(model::RouteAssignmentSubject {
                opportunity_id: subject.opportunity_id.into(),
                request_index: subject.request_index,
            })
        }
        wire::FeedbackSubject::ScheduleSelection(subject) => {
            model::FeedbackSubject::ScheduleSelection(model::ScheduleSelectionSubject {
                opportunity_id: subject.opportunity_id.into(),
                selection_index: subject.selection_index,
            })
        }
        wire::FeedbackSubject::Action(action) => {
            model::FeedbackSubject::Action(model::ActionId(action))
        }
    }
}

fn state_update_to_wire(
    update: model::StateUpdate,
) -> Result<wire::StateUpdate, wire::PolicyError> {
    Ok(wire::StateUpdate {
        shared: update
            .shared
            .map(|document| document_to_wire("shared update", &document))
            .transpose()?,
        groups: update
            .groups
            .into_iter()
            .map(|group| {
                Ok(wire::GroupStateUpdate {
                    group_id: group.group_id.0,
                    scratch: document_to_wire("group scratch update", &group.scratch)?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
        requests: update
            .requests
            .into_iter()
            .map(|request| {
                Ok(wire::RequestStateUpdate {
                    request_id: request.request_id.0,
                    fields: request
                        .fields
                        .map(|document| document_to_wire("request fields update", &document))
                        .transpose()?,
                    scratch: request
                        .scratch
                        .map(|document| document_to_wire("request scratch update", &document))
                        .transpose()?,
                })
            })
            .collect::<Result<_, wire::PolicyError>>()?,
    })
}

fn admission_decision_to_wire(decision: model::AdmissionDecision) -> wire::AdmissionDecision {
    match decision {
        model::AdmissionDecision::Accept => wire::AdmissionDecision::Accept,
        model::AdmissionDecision::Defer => wire::AdmissionDecision::Defer,
        model::AdmissionDecision::Reject => wire::AdmissionDecision::Reject,
    }
}

fn route_decision_to_wire(decision: model::RouteDecision) -> wire::RouteDecision {
    match decision {
        model::RouteDecision::Assign(edge) => wire::RouteDecision::Assign(edge),
        model::RouteDecision::Defer => wire::RouteDecision::Defer,
    }
}

fn cache_admission_to_wire(admission: model::CacheAdmission) -> wire::CacheAdmission {
    match admission {
        model::CacheAdmission::Cache => wire::CacheAdmission::Cache,
        model::CacheAdmission::Bypass => wire::CacheAdmission::Bypass,
    }
}

fn group_status_from_wire(status: wire::GroupStatus) -> model::GroupStatus {
    match status {
        wire::GroupStatus::Open => model::GroupStatus::Open,
        wire::GroupStatus::Closed => model::GroupStatus::Closed,
        wire::GroupStatus::Cancelled => model::GroupStatus::Cancelled,
        wire::GroupStatus::Expired => model::GroupStatus::Expired,
    }
}

fn request_status_from_wire(status: wire::RequestStatus) -> model::RequestStatus {
    match status {
        wire::RequestStatus::Pending => model::RequestStatus::Pending,
        wire::RequestStatus::Admitted => model::RequestStatus::Admitted,
        wire::RequestStatus::Active => model::RequestStatus::Active,
        wire::RequestStatus::Paused => model::RequestStatus::Paused,
        wire::RequestStatus::Completed => model::RequestStatus::Completed,
        wire::RequestStatus::Failed => model::RequestStatus::Failed,
        wire::RequestStatus::Cancelled => model::RequestStatus::Cancelled,
        wire::RequestStatus::Expired => model::RequestStatus::Expired,
        wire::RequestStatus::Rejected => model::RequestStatus::Rejected,
    }
}

fn document_from_wire(
    field: &'static str,
    document: String,
) -> Result<Document, wire::PolicyError> {
    let value: Value = serde_json::from_str(&document)
        .map_err(|error| wire_error("document-decode", format!("{field}: {error}")))?;
    if !value.is_object() {
        return Err(wire_error(
            "document-not-object",
            format!("{field} must be a JSON object"),
        ));
    }
    Ok(value)
}

fn document_to_wire(field: &'static str, document: &Document) -> Result<String, wire::PolicyError> {
    if !document.is_object() {
        return Err(wire_error(
            "document-not-object",
            format!("{field} must be a JSON object"),
        ));
    }
    serde_json::to_string(document)
        .map_err(|error| wire_error("document-encode", format!("{field}: {error}")))
}

fn wire_error(code: impl Into<String>, message: impl Into<String>) -> wire::PolicyError {
    wire::PolicyError {
        code: code.into(),
        message: message.into(),
        details: "{}".into(),
    }
}
