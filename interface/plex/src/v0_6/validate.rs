use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::Document;

use super::mechanics::valid_versioned_name;
use super::types::*;

type ResourceKey = (String, String);

pub fn validate_group_transition(
    from: Option<GroupStatus>,
    to: GroupStatus,
) -> Result<(), ContractValidationError> {
    let valid = match from {
        None => to == GroupStatus::Open,
        Some(GroupStatus::Open) => matches!(
            to,
            GroupStatus::Open | GroupStatus::Closed | GroupStatus::Cancelled | GroupStatus::Expired
        ),
        Some(status) => to == status,
    };
    if !valid {
        return Err(ContractValidationError::InvalidGroupTransition { from, to });
    }
    Ok(())
}

pub fn validate_request_transition(
    from: Option<RequestStatus>,
    to: RequestStatus,
) -> Result<(), ContractValidationError> {
    let valid = match from {
        None => to == RequestStatus::Pending,
        Some(RequestStatus::Pending) => matches!(
            to,
            RequestStatus::Pending
                | RequestStatus::Admitted
                | RequestStatus::Rejected
                | RequestStatus::Expired
        ),
        Some(RequestStatus::Admitted) => matches!(
            to,
            RequestStatus::Admitted
                | RequestStatus::Active
                | RequestStatus::Failed
                | RequestStatus::Cancelled
                | RequestStatus::Expired
        ),
        Some(RequestStatus::Active) => matches!(
            to,
            RequestStatus::Active
                | RequestStatus::Paused
                | RequestStatus::Completed
                | RequestStatus::Failed
                | RequestStatus::Cancelled
                | RequestStatus::Expired
        ),
        Some(RequestStatus::Paused) => matches!(
            to,
            RequestStatus::Paused
                | RequestStatus::Active
                | RequestStatus::Completed
                | RequestStatus::Failed
                | RequestStatus::Cancelled
                | RequestStatus::Expired
        ),
        Some(status) => to == status,
    };
    if !valid {
        return Err(ContractValidationError::InvalidRequestTransition { from, to });
    }
    Ok(())
}

pub fn validate_request_continuation(
    previous: &RequestRef,
    next: &RequestRef,
) -> Result<(), ContractValidationError> {
    validate_request_ref(previous)?;
    validate_request_ref(next)?;
    let expected_generation = previous.generation_id.checked_add(1).ok_or(
        ContractValidationError::ArithmeticOverflow("request generation"),
    )?;
    if previous.request_id != next.request_id
        || previous.group_id != next.group_id
        || previous.principal_id != next.principal_id
        || next.generation_id != expected_generation
    {
        return Err(ContractValidationError::InvalidContinuation {
            request_id: previous.request_id.0.clone(),
            expected_generation,
            actual_generation: next.generation_id,
        });
    }
    Ok(())
}

pub fn validate_policy_error(error: &PolicyError) -> Result<(), ContractValidationError> {
    if error.code.is_empty()
        || error.code.len() > 64
        || !error
            .code
            .bytes()
            .enumerate()
            .all(|(index, byte)| match byte {
                b'a'..=b'z' => true,
                b'0'..=b'9' | b'.' | b'_' | b'-' => index > 0,
                _ => false,
            })
    {
        return Err(ContractValidationError::InvalidPolicyErrorCode(
            error.code.clone(),
        ));
    }
    if error.message.len() > 1024 {
        return Err(ContractValidationError::PolicyErrorMessageTooLong(
            error.message.len(),
        ));
    }
    validate_document("policy_error.details", &error.details)
}

pub fn validate_policy_state(state: &PolicyState) -> Result<(), ContractValidationError> {
    validate_document("state.shared", &state.shared)?;

    let mut groups = BTreeMap::new();
    for group in &state.groups {
        validate_id("group_id", group.group_id.as_str())?;
        validate_id("principal_id", group.principal_id.as_str())?;
        validate_document("state.groups[].facts", &group.facts)?;
        validate_document("state.groups[].scratch", &group.scratch)?;
        if group.limits.max_members == 0 || group.limits.max_scratch_bytes == 0 {
            return Err(ContractValidationError::ZeroGroupLimit(
                group.group_id.0.clone(),
            ));
        }
        if group.member_count > group.limits.max_members {
            return Err(ContractValidationError::GroupMemberLimit {
                group_id: group.group_id.0.clone(),
                actual: group.member_count,
                maximum: group.limits.max_members,
            });
        }
        validate_group_scratch_size(group, &group.scratch)?;
        if groups.insert(group.group_id.as_str(), group).is_some() {
            return Err(ContractValidationError::DuplicateId {
                kind: "group",
                id: group.group_id.0.clone(),
            });
        }
        if let Some(fact_id) = group.facts.get("group_id") {
            if fact_id.as_str() != Some(group.group_id.as_str()) {
                return Err(ContractValidationError::IdentityFactMismatch {
                    kind: "group",
                    id: group.group_id.0.clone(),
                });
            }
        }
        if let Some(principal_id) = group.facts.get("principal_id") {
            if principal_id.as_str() != Some(group.principal_id.as_str()) {
                return Err(ContractValidationError::IdentityFactMismatch {
                    kind: "group principal",
                    id: group.group_id.0.clone(),
                });
            }
        }
    }

    let mut requests = BTreeMap::new();
    for request in &state.requests {
        validate_request_ref(&request.request)?;
        validate_document("state.requests[].facts", &request.facts)?;
        validate_document("state.requests[].fields", &request.fields)?;
        validate_document("state.requests[].scratch", &request.scratch)?;
        if requests
            .insert(request.request.request_id.as_str(), request)
            .is_some()
        {
            return Err(ContractValidationError::DuplicateId {
                kind: "request",
                id: request.request.request_id.0.clone(),
            });
        }
        if let Some(group_id) = &request.request.group_id {
            let Some(group) = groups.get(group_id.as_str()) else {
                return Err(ContractValidationError::MissingScope {
                    kind: "group",
                    id: group_id.0.clone(),
                });
            };
            if group.principal_id != request.request.principal_id {
                return Err(ContractValidationError::PrincipalMismatch {
                    request_id: request.request.request_id.0.clone(),
                    group_id: group.group_id.0.clone(),
                });
            }
        }
        validate_request_identity_facts(request)?;
    }
    Ok(())
}

pub fn validate_state_update(
    state: &PolicyState,
    update: &StateUpdate,
) -> Result<(), ContractValidationError> {
    validate_policy_state(state)?;
    if let Some(shared) = &update.shared {
        validate_document("state_update.shared", shared)?;
    }

    let group_ids = state
        .groups
        .iter()
        .map(|group| (group.group_id.as_str(), group))
        .collect::<BTreeMap<_, _>>();
    let mut changed_groups = BTreeSet::new();
    for group in &update.groups {
        validate_id("group_id", group.group_id.as_str())?;
        let Some(current) = group_ids.get(group.group_id.as_str()) else {
            return Err(ContractValidationError::MissingScope {
                kind: "group",
                id: group.group_id.0.clone(),
            });
        };
        if !changed_groups.insert(group.group_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "group update",
                id: group.group_id.0.clone(),
            });
        }
        validate_document("state_update.groups[].scratch", &group.scratch)?;
        validate_group_scratch_size(current, &group.scratch)?;
    }

    let request_ids = state
        .requests
        .iter()
        .map(|request| request.request.request_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut changed_requests = BTreeSet::new();
    for request in &update.requests {
        validate_id("request_id", request.request_id.as_str())?;
        if !request_ids.contains(request.request_id.as_str()) {
            return Err(ContractValidationError::MissingScope {
                kind: "request",
                id: request.request_id.0.clone(),
            });
        }
        if !changed_requests.insert(request.request_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "request update",
                id: request.request_id.0.clone(),
            });
        }
        if request.fields.is_none() && request.scratch.is_none() {
            return Err(ContractValidationError::EmptyRequestUpdate(
                request.request_id.0.clone(),
            ));
        }
        if let Some(fields) = &request.fields {
            validate_document("state_update.requests[].fields", fields)?;
        }
        if let Some(scratch) = &request.scratch {
            validate_document("state_update.requests[].scratch", scratch)?;
        }
    }
    Ok(())
}

pub fn validate_admit_context(
    state: &PolicyState,
    context: &AdmitContext,
) -> Result<(), ContractValidationError> {
    let requests = state_requests(state)?;
    let groups = state_groups(state);
    validate_meta(&context.meta)?;
    validate_document("admit.capacity.facts", &context.capacity.facts)?;
    let limits = validate_limits("admit.capacity.limits", &context.capacity.limits)?;
    let mut candidates = BTreeSet::new();
    for candidate in &context.candidates {
        validate_document("admit.candidates[].facts", &candidate.facts)?;
        validate_demands("admit.candidates[].demand", &candidate.demand, &limits)?;
        validate_context_request(&candidate.request, &requests, &[RequestStatus::Pending])?;
        if let Some(group_id) = &candidate.request.group_id {
            let group = groups
                .get(group_id.as_str())
                .expect("policy state validation checked group membership");
            if group.status != GroupStatus::Open {
                return Err(ContractValidationError::AdmissionIntoTerminalGroup {
                    group_id: group_id.0.clone(),
                    request_id: candidate.request.request_id.0.clone(),
                });
            }
        }
        if !candidates.insert(candidate.request.request_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "admission candidate",
                id: candidate.request.request_id.0.clone(),
            });
        }
    }
    Ok(())
}

pub fn validate_admit_plan(
    context: &AdmitContext,
    plan: &AdmitPlan,
) -> Result<(), ContractValidationError> {
    if plan.decisions.len() != context.candidates.len() {
        return Err(ContractValidationError::DenseLength {
            field: "admit.decisions",
            expected: context.candidates.len(),
            actual: plan.decisions.len(),
        });
    }
    let limits = validate_limits("admit.capacity.limits", &context.capacity.limits)?;
    let mut accepted = 0u32;
    let mut totals = BTreeMap::new();
    for (candidate, decision) in context.candidates.iter().zip(&plan.decisions) {
        if *decision != AdmissionDecision::Accept {
            continue;
        }
        accepted = accepted
            .checked_add(1)
            .ok_or(ContractValidationError::ArithmeticOverflow(
                "admit accepted count",
            ))?;
        add_demands(&mut totals, &candidate.demand, "admit demand")?;
    }
    if accepted > context.capacity.max_accepted {
        return Err(ContractValidationError::CountCapacityExceeded {
            field: "admit.max_accepted",
            actual: u64::from(accepted),
            maximum: u64::from(context.capacity.max_accepted),
        });
    }
    validate_totals(&totals, &limits)
}

pub fn validate_route_context(
    state: &PolicyState,
    context: &RouteContext,
) -> Result<(), ContractValidationError> {
    let requests = state_requests(state)?;
    validate_meta(&context.meta)?;
    let mut request_ids = BTreeSet::new();
    for request in &context.requests {
        validate_document("route.requests[].facts", &request.facts)?;
        validate_context_request(
            &request.request,
            &requests,
            &[
                RequestStatus::Admitted,
                RequestStatus::Active,
                RequestStatus::Paused,
            ],
        )?;
        if !request_ids.insert(request.request.request_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "route request",
                id: request.request.request_id.0.clone(),
            });
        }
    }

    let mut targets = BTreeSet::new();
    let mut target_limits = Vec::with_capacity(context.targets.len());
    for target in &context.targets {
        validate_id("target_id", target.target_id.as_str())?;
        validate_document("route.targets[].facts", &target.facts)?;
        if !targets.insert(target.target_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "route target",
                id: target.target_id.0.clone(),
            });
        }
        target_limits.push(validate_limits(
            "route.targets[].capacity",
            &target.capacity,
        )?);
    }

    let mut edges = BTreeSet::new();
    for edge in &context.feasible_edges {
        let request_index = index(
            edge.request_index,
            context.requests.len(),
            "route edge request",
        )?;
        let target_index = index(
            edge.target_index,
            context.targets.len(),
            "route edge target",
        )?;
        validate_document("route.feasible_edges[].facts", &edge.facts)?;
        validate_demands(
            "route.feasible_edges[].demand",
            &edge.demand,
            &target_limits[target_index],
        )?;
        if !edges.insert((request_index, target_index)) {
            return Err(ContractValidationError::DuplicateEdge {
                request_index,
                target_index,
            });
        }
    }
    Ok(())
}

pub fn validate_route_plan(
    context: &RouteContext,
    plan: &RoutePlan,
) -> Result<(), ContractValidationError> {
    if plan.decisions.len() != context.requests.len() {
        return Err(ContractValidationError::DenseLength {
            field: "route.decisions",
            expected: context.requests.len(),
            actual: plan.decisions.len(),
        });
    }
    let target_limits = context
        .targets
        .iter()
        .map(|target| validate_limits("route.targets[].capacity", &target.capacity))
        .collect::<Result<Vec<_>, _>>()?;
    let mut target_counts = vec![0u32; context.targets.len()];
    let mut target_totals = vec![BTreeMap::new(); context.targets.len()];
    for (request_index, decision) in plan.decisions.iter().enumerate() {
        let RouteDecision::Assign(edge_index) = decision else {
            continue;
        };
        let edge_index = index(
            *edge_index,
            context.feasible_edges.len(),
            "route decision edge",
        )?;
        let edge = &context.feasible_edges[edge_index];
        if usize::try_from(edge.request_index).ok() != Some(request_index) {
            return Err(ContractValidationError::RouteEdgeRequestMismatch {
                request_index,
                edge_index,
            });
        }
        let target_index = index(
            edge.target_index,
            context.targets.len(),
            "route edge target",
        )?;
        target_counts[target_index] = target_counts[target_index].checked_add(1).ok_or(
            ContractValidationError::ArithmeticOverflow("route assignment count"),
        )?;
        add_demands(
            &mut target_totals[target_index],
            &edge.demand,
            "route demand",
        )?;
    }
    for (target_index, target) in context.targets.iter().enumerate() {
        if target_counts[target_index] > target.max_assignments {
            return Err(ContractValidationError::CountCapacityExceeded {
                field: "route.target.max_assignments",
                actual: u64::from(target_counts[target_index]),
                maximum: u64::from(target.max_assignments),
            });
        }
        validate_totals(&target_totals[target_index], &target_limits[target_index])?;
    }
    Ok(())
}

pub fn validate_schedule_context(
    state: &PolicyState,
    context: &ScheduleContext,
) -> Result<(), ContractValidationError> {
    let requests = state_requests(state)?;
    validate_meta(&context.meta)?;
    validate_document("schedule.capacity.facts", &context.capacity.facts)?;
    let mut runnable = BTreeSet::new();
    for candidate in &context.runnable {
        validate_document("schedule.runnable[].facts", &candidate.facts)?;
        validate_context_request(&candidate.request, &requests, &[RequestStatus::Active])?;
        if candidate.max_token_budget == 0 {
            return Err(ContractValidationError::ZeroTokenBudget);
        }
        if !runnable.insert(candidate.request.request_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "schedule candidate",
                id: candidate.request.request_id.0.clone(),
            });
        }
    }
    Ok(())
}

pub fn validate_schedule_plan(
    context: &ScheduleContext,
    plan: &SchedulePlan,
) -> Result<(), ContractValidationError> {
    if plan.selections.len()
        > usize::try_from(context.capacity.max_selections).unwrap_or(usize::MAX)
    {
        return Err(ContractValidationError::CountCapacityExceeded {
            field: "schedule.max_selections",
            actual: plan.selections.len() as u64,
            maximum: u64::from(context.capacity.max_selections),
        });
    }
    let mut selected = BTreeSet::new();
    let mut selected_count = 0u32;
    let mut total_tokens = 0u64;
    for (selection_index, selection) in plan.selections.iter().enumerate() {
        if selection.requests.is_empty() {
            return Err(ContractValidationError::EmptySelection(selection_index));
        }
        if selection.requests.len() != selection.token_budgets.len() {
            return Err(ContractValidationError::SelectionLength {
                selection: selection_index,
                requests: selection.requests.len(),
                budgets: selection.token_budgets.len(),
            });
        }
        for (&request_index, &budget) in selection.requests.iter().zip(&selection.token_budgets) {
            let request_index = index(
                request_index,
                context.runnable.len(),
                "schedule selection request",
            )?;
            if !selected.insert(request_index) {
                return Err(ContractValidationError::DuplicateSelectionRequest(
                    request_index,
                ));
            }
            if budget == 0 || budget > context.runnable[request_index].max_token_budget {
                return Err(ContractValidationError::InvalidTokenBudget {
                    request_index,
                    actual: budget,
                    maximum: context.runnable[request_index].max_token_budget,
                });
            }
            selected_count = selected_count.checked_add(1).ok_or(
                ContractValidationError::ArithmeticOverflow("schedule request count"),
            )?;
            total_tokens = total_tokens.checked_add(u64::from(budget)).ok_or(
                ContractValidationError::ArithmeticOverflow("schedule token budget"),
            )?;
        }
    }
    if selected_count > context.capacity.max_requests {
        return Err(ContractValidationError::CountCapacityExceeded {
            field: "schedule.max_requests",
            actual: u64::from(selected_count),
            maximum: u64::from(context.capacity.max_requests),
        });
    }
    if total_tokens > context.capacity.max_total_tokens {
        return Err(ContractValidationError::CountCapacityExceeded {
            field: "schedule.max_total_tokens",
            actual: total_tokens,
            maximum: context.capacity.max_total_tokens,
        });
    }
    Ok(())
}

pub fn validate_cache_context(context: &CacheContext) -> Result<(), ContractValidationError> {
    validate_meta(&context.meta)?;
    validate_document("cache.capacity.facts", &context.capacity.facts)?;
    if context.capacity.fixed_bytes > context.capacity.max_bytes {
        return Err(ContractValidationError::CountCapacityExceeded {
            field: "cache.fixed_bytes",
            actual: context.capacity.fixed_bytes,
            maximum: context.capacity.max_bytes,
        });
    }
    match (&context.cause, &context.episode) {
        (CacheCause::DependencyProgress, None) => {
            return Err(ContractValidationError::MissingCacheEpisode);
        }
        (_, Some(episode)) => {
            validate_id("episode_id", episode.episode_id.as_str())?;
            if episode.max_iterations == 0 || episode.iteration >= episode.max_iterations {
                return Err(ContractValidationError::InvalidCacheEpisode {
                    iteration: episode.iteration,
                    maximum: episode.max_iterations,
                });
            }
        }
        _ => {}
    }

    let mut object_ids = BTreeSet::new();
    for resident in &context.resident {
        validate_cache_object(&resident.object)?;
        if !object_ids.insert(resident.object.object_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "cache object",
                id: resident.object.object_id.0.clone(),
            });
        }
    }
    for prospective in &context.prospective {
        validate_cache_object(prospective)?;
        if !object_ids.insert(prospective.object_id.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "cache object",
                id: prospective.object_id.0.clone(),
            });
        }
    }
    Ok(())
}

pub fn validate_cache_plan(
    context: &CacheContext,
    plan: &CachePlan,
) -> Result<(), ContractValidationError> {
    if plan.admissions.len() != context.prospective.len() {
        return Err(ContractValidationError::DenseLength {
            field: "cache.admissions",
            expected: context.prospective.len(),
            actual: plan.admissions.len(),
        });
    }
    let mut reclaimed = BTreeSet::new();
    for &resident_index in &plan.reclaim {
        let resident_index = index(
            resident_index,
            context.resident.len(),
            "cache reclaim resident",
        )?;
        if !reclaimed.insert(resident_index) {
            return Err(ContractValidationError::DuplicateReclaim(resident_index));
        }
        if !context.resident[resident_index].reclaimable {
            return Err(ContractValidationError::ResidentNotReclaimable(
                resident_index,
            ));
        }
    }

    let mut retained = context.capacity.fixed_bytes;
    for (resident_index, resident) in context.resident.iter().enumerate() {
        if !reclaimed.contains(&resident_index) {
            retained = retained.checked_add(resident.object.size_bytes).ok_or(
                ContractValidationError::ArithmeticOverflow("cache resident bytes"),
            )?;
        }
    }
    for (prospective, admission) in context.prospective.iter().zip(&plan.admissions) {
        if *admission == CacheAdmission::Cache {
            retained = retained.checked_add(prospective.size_bytes).ok_or(
                ContractValidationError::ArithmeticOverflow("cache prospective bytes"),
            )?;
        }
    }
    if retained > context.capacity.max_bytes {
        return Err(ContractValidationError::CountCapacityExceeded {
            field: "cache.max_bytes",
            actual: retained,
            maximum: context.capacity.max_bytes,
        });
    }
    Ok(())
}

pub fn validate_feedback_context(context: &FeedbackContext) -> Result<(), ContractValidationError> {
    validate_id("delivery_id", context.delivery_id.as_str())?;
    if context.records.is_empty() {
        return Err(ContractValidationError::EmptyFeedback);
    }
    for record in &context.records {
        validate_document("feedback.records[].facts", &record.facts)?;
        let action_subject = matches!(&record.subject, FeedbackSubject::Action(_));
        let action_outcome = matches!(
            record.outcome,
            OutcomeKind::ActionSucceeded | OutcomeKind::ActionFailed
        );
        if action_subject {
            if !matches!(
                record.outcome,
                OutcomeKind::Progress | OutcomeKind::ActionSucceeded | OutcomeKind::ActionFailed
            ) {
                return Err(ContractValidationError::FeedbackOutcomeMismatch);
            }
        } else if action_outcome {
            return Err(ContractValidationError::FeedbackOutcomeMismatch);
        }
        match &record.subject {
            FeedbackSubject::Request(id) => validate_id("request_id", id.as_str())?,
            FeedbackSubject::WorkGroup(id) => validate_id("group_id", id.as_str())?,
            FeedbackSubject::CacheObject(id) => validate_id("cache_object_id", id.as_str())?,
            FeedbackSubject::RouteAssignment(subject) => {
                validate_id("opportunity_id", subject.opportunity_id.as_str())?;
            }
            FeedbackSubject::ScheduleSelection(subject) => {
                validate_id("opportunity_id", subject.opportunity_id.as_str())?;
            }
            FeedbackSubject::Action(_) => {}
        }
    }
    Ok(())
}

fn state_requests<'a>(
    state: &'a PolicyState,
) -> Result<BTreeMap<&'a str, &'a RequestState>, ContractValidationError> {
    validate_policy_state(state)?;
    Ok(state
        .requests
        .iter()
        .map(|request| (request.request.request_id.as_str(), request))
        .collect())
}

fn state_groups(state: &PolicyState) -> BTreeMap<&str, &GroupState> {
    state
        .groups
        .iter()
        .map(|group| (group.group_id.as_str(), group))
        .collect()
}

fn validate_context_request(
    reference: &RequestRef,
    requests: &BTreeMap<&str, &RequestState>,
    allowed: &[RequestStatus],
) -> Result<(), ContractValidationError> {
    validate_request_ref(reference)?;
    let Some(state) = requests.get(reference.request_id.as_str()) else {
        return Err(ContractValidationError::MissingScope {
            kind: "request",
            id: reference.request_id.0.clone(),
        });
    };
    if state.request != *reference {
        return Err(ContractValidationError::RequestReferenceMismatch(
            reference.request_id.0.clone(),
        ));
    }
    if !allowed.contains(&state.status) {
        return Err(ContractValidationError::InvalidRequestStatus {
            request_id: reference.request_id.0.clone(),
            actual: state.status,
        });
    }
    Ok(())
}

fn validate_request_ref(reference: &RequestRef) -> Result<(), ContractValidationError> {
    validate_id("request_id", reference.request_id.as_str())?;
    validate_id("principal_id", reference.principal_id.as_str())?;
    if let Some(group_id) = &reference.group_id {
        validate_id("group_id", group_id.as_str())?;
    }
    Ok(())
}

fn validate_request_identity_facts(request: &RequestState) -> Result<(), ContractValidationError> {
    if let Some(fact_id) = request.facts.get("request_id") {
        if fact_id.as_str() != Some(request.request.request_id.as_str()) {
            return Err(ContractValidationError::IdentityFactMismatch {
                kind: "request",
                id: request.request.request_id.0.clone(),
            });
        }
    }
    if let Some(generation) = request.facts.get("generation_id") {
        if generation.as_u64() != Some(request.request.generation_id) {
            return Err(ContractValidationError::IdentityFactMismatch {
                kind: "request generation",
                id: request.request.request_id.0.clone(),
            });
        }
    }
    if let Some(group) = request.facts.get("group_id") {
        let expected = request.request.group_id.as_ref().map(GroupId::as_str);
        if !(group.is_null() && expected.is_none()) && group.as_str() != expected {
            return Err(ContractValidationError::IdentityFactMismatch {
                kind: "request group",
                id: request.request.request_id.0.clone(),
            });
        }
    }
    if let Some(principal) = request.facts.get("principal_id") {
        if principal.as_str() != Some(request.request.principal_id.as_str()) {
            return Err(ContractValidationError::IdentityFactMismatch {
                kind: "request principal",
                id: request.request.request_id.0.clone(),
            });
        }
    }
    Ok(())
}

fn validate_meta(meta: &DecisionMeta) -> Result<(), ContractValidationError> {
    validate_id("opportunity_id", meta.opportunity_id.as_str())?;
    validate_id("snapshot_id", meta.snapshot.id.as_str())?;
    let mut mechanics = BTreeSet::new();
    for mechanic in &meta.mechanics {
        if !valid_versioned_name(mechanic.as_str()) {
            return Err(ContractValidationError::InvalidVersionedName(
                mechanic.0.clone(),
            ));
        }
        if !mechanics.insert(mechanic.as_str()) {
            return Err(ContractValidationError::DuplicateId {
                kind: "negotiated mechanic",
                id: mechanic.0.clone(),
            });
        }
    }
    Ok(())
}

fn validate_cache_object(object: &CacheObject) -> Result<(), ContractValidationError> {
    validate_id("cache_object_id", object.object_id.as_str())?;
    validate_document("cache.object.facts", &object.facts)?;
    if usize::try_from(object.beneficiary_count).unwrap_or(usize::MAX) < object.beneficiaries.len()
    {
        return Err(ContractValidationError::BeneficiaryCount {
            object_id: object.object_id.0.clone(),
            listed: object.beneficiaries.len(),
            total: object.beneficiary_count,
        });
    }
    let mut beneficiaries = BTreeSet::new();
    for beneficiary in &object.beneficiaries {
        match beneficiary {
            Beneficiary::Request(id) => validate_id("request_id", id.as_str())?,
            Beneficiary::Group(id) => validate_id("group_id", id.as_str())?,
        }
        if !beneficiaries.insert(beneficiary) {
            return Err(ContractValidationError::DuplicateBeneficiary(
                object.object_id.0.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_group_scratch_size(
    group: &GroupState,
    scratch: &Document,
) -> Result<(), ContractValidationError> {
    let actual = serde_json::to_vec(scratch)
        .expect("serde_json::Value serialization cannot fail")
        .len() as u64;
    if actual > group.limits.max_scratch_bytes {
        return Err(ContractValidationError::GroupScratchLimit {
            group_id: group.group_id.0.clone(),
            actual,
            maximum: group.limits.max_scratch_bytes,
        });
    }
    Ok(())
}

fn validate_limits(
    path: &'static str,
    limits: &[ResourceLimit],
) -> Result<BTreeMap<ResourceKey, u64>, ContractValidationError> {
    let mut result = BTreeMap::new();
    for limit in limits {
        validate_resource_atom(&limit.name)?;
        validate_resource_atom(&limit.unit)?;
        let key = (limit.name.clone(), limit.unit.clone());
        if result.insert(key.clone(), limit.maximum).is_some() {
            return Err(ContractValidationError::DuplicateResource {
                path,
                name: key.0,
                unit: key.1,
            });
        }
    }
    Ok(result)
}

fn validate_demands(
    path: &'static str,
    demands: &[ResourceAmount],
    limits: &BTreeMap<ResourceKey, u64>,
) -> Result<(), ContractValidationError> {
    let mut seen = BTreeSet::new();
    for demand in demands {
        validate_resource_atom(&demand.name)?;
        validate_resource_atom(&demand.unit)?;
        let key = (demand.name.clone(), demand.unit.clone());
        if !seen.insert(key.clone()) {
            return Err(ContractValidationError::DuplicateResource {
                path,
                name: key.0,
                unit: key.1,
            });
        }
        if !limits.contains_key(&key) {
            return Err(ContractValidationError::MissingResourceLimit {
                name: key.0,
                unit: key.1,
            });
        }
    }
    Ok(())
}

fn add_demands(
    totals: &mut BTreeMap<ResourceKey, u64>,
    demands: &[ResourceAmount],
    field: &'static str,
) -> Result<(), ContractValidationError> {
    for demand in demands {
        let entry = totals
            .entry((demand.name.clone(), demand.unit.clone()))
            .or_default();
        *entry = entry
            .checked_add(demand.amount)
            .ok_or(ContractValidationError::ArithmeticOverflow(field))?;
    }
    Ok(())
}

fn validate_totals(
    totals: &BTreeMap<ResourceKey, u64>,
    limits: &BTreeMap<ResourceKey, u64>,
) -> Result<(), ContractValidationError> {
    for ((name, unit), actual) in totals {
        let Some(maximum) = limits.get(&(name.clone(), unit.clone())) else {
            return Err(ContractValidationError::MissingResourceLimit {
                name: name.clone(),
                unit: unit.clone(),
            });
        };
        if actual > maximum {
            return Err(ContractValidationError::ResourceCapacityExceeded {
                name: name.clone(),
                unit: unit.clone(),
                actual: *actual,
                maximum: *maximum,
            });
        }
    }
    Ok(())
}

fn validate_resource_atom(value: &str) -> Result<(), ContractValidationError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().enumerate().all(|(index, byte)| match byte {
            b'a'..=b'z' => true,
            b'0'..=b'9' | b'.' | b'_' | b'-' => index > 0,
            _ => false,
        })
    {
        return Err(ContractValidationError::InvalidResourceAtom(
            value.to_owned(),
        ));
    }
    Ok(())
}

fn validate_document(
    path: &'static str,
    document: &Document,
) -> Result<(), ContractValidationError> {
    if !document.is_object() {
        return Err(ContractValidationError::DocumentNotObject(path));
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), ContractValidationError> {
    if value.is_empty() || value.len() > 128 {
        return Err(ContractValidationError::InvalidId {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn index(value: u32, len: usize, field: &'static str) -> Result<usize, ContractValidationError> {
    let value = usize::try_from(value).map_err(|_| ContractValidationError::IndexOutOfRange {
        field,
        index: usize::MAX,
        len,
    })?;
    if value >= len {
        return Err(ContractValidationError::IndexOutOfRange {
            field,
            index: value,
            len,
        });
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ContractValidationError {
    #[error("{0} must be a JSON object")]
    DocumentNotObject(&'static str),
    #[error("{kind} ID {value:?} must contain 1-128 UTF-8 bytes")]
    InvalidId { kind: &'static str, value: String },
    #[error("duplicate {kind} ID {id:?}")]
    DuplicateId { kind: &'static str, id: String },
    #[error("missing {kind} state for {id:?}")]
    MissingScope { kind: &'static str, id: String },
    #[error("{kind} identity facts do not match trusted identity {id:?}")]
    IdentityFactMismatch { kind: &'static str, id: String },
    #[error("pending request {request_id:?} cannot enter terminal group {group_id:?}")]
    AdmissionIntoTerminalGroup {
        group_id: String,
        request_id: String,
    },
    #[error("request reference for {0:?} does not match its trusted state")]
    RequestReferenceMismatch(String),
    #[error("request {request_id:?} principal does not own group {group_id:?}")]
    PrincipalMismatch {
        request_id: String,
        group_id: String,
    },
    #[error("request {request_id:?} has invalid lifecycle status {actual:?}")]
    InvalidRequestStatus {
        request_id: String,
        actual: RequestStatus,
    },
    #[error("request update for {0:?} changes neither fields nor scratch")]
    EmptyRequestUpdate(String),
    #[error("group {0:?} limits must be non-zero")]
    ZeroGroupLimit(String),
    #[error("group {group_id:?} has {actual} members; maximum is {maximum}")]
    GroupMemberLimit {
        group_id: String,
        actual: u32,
        maximum: u32,
    },
    #[error("group {group_id:?} scratch uses {actual} bytes; maximum is {maximum}")]
    GroupScratchLimit {
        group_id: String,
        actual: u64,
        maximum: u64,
    },
    #[error("{0:?} is not a valid versioned name")]
    InvalidVersionedName(String),
    #[error("resource atom {0:?} is invalid")]
    InvalidResourceAtom(String),
    #[error("duplicate resource {name:?}/{unit:?} in {path}")]
    DuplicateResource {
        path: &'static str,
        name: String,
        unit: String,
    },
    #[error("resource demand {name:?}/{unit:?} has no matching limit")]
    MissingResourceLimit { name: String, unit: String },
    #[error("resource {name:?}/{unit:?} uses {actual}, exceeding capacity {maximum}")]
    ResourceCapacityExceeded {
        name: String,
        unit: String,
        actual: u64,
        maximum: u64,
    },
    #[error("{field} uses {actual}, exceeding capacity {maximum}")]
    CountCapacityExceeded {
        field: &'static str,
        actual: u64,
        maximum: u64,
    },
    #[error("{field} has {actual} entries; expected {expected}")]
    DenseLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} index {index} is outside 0..{len}")]
    IndexOutOfRange {
        field: &'static str,
        index: usize,
        len: usize,
    },
    #[error("duplicate feasible edge request {request_index} -> target {target_index}")]
    DuplicateEdge {
        request_index: usize,
        target_index: usize,
    },
    #[error("route request {request_index} selected edge {edge_index} for another request")]
    RouteEdgeRequestMismatch {
        request_index: usize,
        edge_index: usize,
    },
    #[error("schedule candidate maximum token budget must be positive")]
    ZeroTokenBudget,
    #[error("schedule selection {0} is empty")]
    EmptySelection(usize),
    #[error("schedule selection {selection} has {requests} requests and {budgets} token budgets")]
    SelectionLength {
        selection: usize,
        requests: usize,
        budgets: usize,
    },
    #[error("schedule request index {0} appears in more than one selection")]
    DuplicateSelectionRequest(usize),
    #[error("schedule request {request_index} uses token budget {actual}; maximum is {maximum}")]
    InvalidTokenBudget {
        request_index: usize,
        actual: u32,
        maximum: u32,
    },
    #[error("dependency-progress cache context requires an episode")]
    MissingCacheEpisode,
    #[error("cache episode iteration {iteration} must be below positive maximum {maximum}")]
    InvalidCacheEpisode { iteration: u32, maximum: u32 },
    #[error("cache object {object_id:?} lists {listed} beneficiaries but total is {total}")]
    BeneficiaryCount {
        object_id: String,
        listed: usize,
        total: u32,
    },
    #[error("cache object {0:?} lists a beneficiary more than once")]
    DuplicateBeneficiary(String),
    #[error("resident cache index {0} appears more than once in reclaim order")]
    DuplicateReclaim(usize),
    #[error("resident cache index {0} is not reclaimable")]
    ResidentNotReclaimable(usize),
    #[error("feedback delivery must contain at least one record")]
    EmptyFeedback,
    #[error("feedback subject and outcome kinds are incompatible")]
    FeedbackOutcomeMismatch,
    #[error("unsigned arithmetic overflow while validating {0}")]
    ArithmeticOverflow(&'static str),
    #[error("group lifecycle transition {from:?} -> {to:?} is invalid")]
    InvalidGroupTransition {
        from: Option<GroupStatus>,
        to: GroupStatus,
    },
    #[error("request lifecycle transition {from:?} -> {to:?} is invalid")]
    InvalidRequestTransition {
        from: Option<RequestStatus>,
        to: RequestStatus,
    },
    #[error(
        "request {request_id:?} continuation expected generation {expected_generation}, got {actual_generation}"
    )]
    InvalidContinuation {
        request_id: String,
        expected_generation: u64,
        actual_generation: u64,
    },
    #[error("policy error code {0:?} is invalid")]
    InvalidPolicyErrorCode(String),
    #[error("policy error message contains {0} bytes; maximum is 1024")]
    PolicyErrorMessageTooLong(usize),
}
