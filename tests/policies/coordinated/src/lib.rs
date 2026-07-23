use plex::serde_json::json;
use plex::{
    AdmissionDecision, AdmitContext, AdmitPlan, CacheAdmission, CacheContext, CachePlan,
    FeedbackContext, FeedbackSubject, Host, OutcomeKind, Policy, RouteContext, RouteDecision,
    RoutePlan, ScheduleContext, SchedulePlan, ScheduleSelection, State,
};

struct Coordinated;

impl Policy for Coordinated {
    fn admit(ctx: &AdmitContext, state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut decisions = Vec::with_capacity(ctx.candidates.len());
        for candidate in &ctx.candidates {
            let queue_depth = candidate.facts["queue_depth"].as_u64().unwrap_or(0);
            let decision = if queue_depth < 80 {
                AdmissionDecision::Accept
            } else if queue_depth < 100 {
                AdmissionDecision::Defer
            } else {
                AdmissionDecision::Reject
            };
            let request = state.request_mut(candidate.request.request_id.as_str())?;
            request.scratch["admission_count"] =
                json!(request.scratch["admission_count"].as_u64().unwrap_or(0) + 1);
            request.fields["last_hook"] = json!("admit");
            decisions.push(decision);
        }
        state.shared["admit_calls"] = json!(state.shared["admit_calls"].as_u64().unwrap_or(0) + 1);
        Ok(AdmitPlan { decisions })
    }

    fn route(ctx: &RouteContext, state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let query_bias = if ctx
            .requests
            .first()
            .is_some_and(|request| request.facts["query"].as_bool() == Some(true))
        {
            host.query_raw(
                "pie.cluster.capacity@1",
                &json!({"model": "example-model"}),
            )?["queue_bias"]
                .as_u64()
                .unwrap_or(0)
        } else {
            0
        };
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let edge = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| edge.request_index as usize == request_index)
                .min_by_key(|(_, edge)| {
                    edge.facts["queue_depth"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_add(query_bias)
                })
                .map(|(index, _)| index as u32);
            decisions.push(edge.map_or(RouteDecision::Defer, RouteDecision::Assign));
            let request = state.request_mut(request.request.request_id.as_str())?;
            request.scratch["route_count"] =
                json!(request.scratch["route_count"].as_u64().unwrap_or(0) + 1);
            request.fields["last_hook"] = json!("route");
            if ctx.requests[request_index].facts["rebalance"].as_bool() == Some(true)
                && let Some(target) = ctx.targets.first()
            {
                host.rebalance_request(
                    ctx.requests[request_index].request.request_id.as_str(),
                    target.target_id.as_str(),
                    &format!("rebalance-{request_index}"),
                )?;
            }
            if ctx.requests[request_index].facts["invalid_cancel"].as_bool() == Some(true) {
                host.cancel_request(
                    ctx.requests[request_index].request.request_id.as_str(),
                    &format!("invalid-cancel-{request_index}"),
                    None,
                )?;
            }
        }
        state.shared["route_calls"] = json!(state.shared["route_calls"].as_u64().unwrap_or(0) + 1);
        Ok(RoutePlan { decisions })
    }

    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let budget = u64::from(candidate.max_token_budget).min(remaining_tokens) as u32;
            if budget == 0 {
                continue;
            }
            selections.push(ScheduleSelection {
                requests: vec![index as u32],
                token_budgets: vec![budget],
            });
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
            let request = state.request_mut(candidate.request.request_id.as_str())?;
            request.scratch["schedule_calls"] =
                json!(request.scratch["schedule_calls"].as_u64().unwrap_or(0) + 1);
            request.fields["last_hook"] = json!("schedule");
            if candidate.facts["cancel_request"].as_bool() == Some(true) {
                let key = format!("cancel-{index}");
                let first =
                    host.cancel_request(candidate.request.request_id.as_str(), &key, None)?;
                let duplicate =
                    host.cancel_request(candidate.request.request_id.as_str(), &key, None)?;
                debug_assert_eq!(first, duplicate);
            }
            if candidate.facts["cancel_group"].as_bool() == Some(true)
                && let Some(group_id) = &candidate.request.group_id
            {
                host.cancel_group(
                    group_id.as_str(),
                    "live-requests",
                    &format!("cancel-group-{index}"),
                )?;
            }
        }
        Ok(SchedulePlan { selections })
    }

    fn cache(ctx: &CacheContext, state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let resident_bytes = ctx
            .resident
            .iter()
            .map(|resident| resident.object.size_bytes)
            .sum::<u64>()
            .saturating_add(ctx.capacity.fixed_bytes);
        let mut remaining = ctx.capacity.max_bytes.saturating_sub(resident_bytes);
        let admissions = ctx
            .prospective
            .iter()
            .map(|object| {
                if object.size_bytes <= remaining && object.facts["cache"].as_bool().unwrap_or(true)
                {
                    remaining -= object.size_bytes;
                    CacheAdmission::Cache
                } else {
                    CacheAdmission::Bypass
                }
            })
            .collect();
        for resident in &ctx.resident {
            for beneficiary in &resident.object.beneficiaries {
                if let plex::Beneficiary::Request(request_id) = beneficiary
                    && let Ok(request) = state.request_mut(request_id.as_str())
                {
                    request.scratch["cache_checks"] =
                        json!(request.scratch["cache_checks"].as_u64().unwrap_or(0) + 1);
                }
            }
        }
        for object in &ctx.prospective {
            if object.facts["prefetch"].as_bool() == Some(true) {
                host.prefetch_cache(
                    object.object_id.as_str(),
                    None,
                    &format!("prefetch-{}", object.object_id.as_str()),
                )?;
            }
            if object.facts["swap"].as_bool() == Some(true) {
                host.swap_cache(
                    object.object_id.as_str(),
                    "cpu",
                    &format!("swap-{}", object.object_id.as_str()),
                )?;
            }
        }
        Ok(CachePlan {
            admissions,
            reclaim: Vec::new(),
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            match &record.subject {
                FeedbackSubject::Request(request_id) => {
                    if !matches!(
                        record.outcome,
                        OutcomeKind::Completed
                            | OutcomeKind::Failed
                            | OutcomeKind::Cancelled
                            | OutcomeKind::Expired
                    ) {
                        let request = state.request_mut(request_id.as_str())?;
                        request.scratch["feedback_records"] =
                            json!(request.scratch["feedback_records"].as_u64().unwrap_or(0) + 1);
                    }
                }
                FeedbackSubject::WorkGroup(group_id) => {
                    let group = state.group_mut(group_id.as_str())?;
                    group.scratch["feedback_records"] =
                        json!(group.scratch["feedback_records"].as_u64().unwrap_or(0) + 1);
                }
                _ => {}
            }
        }
        state.shared["feedback_records"] = json!(
            state.shared["feedback_records"].as_u64().unwrap_or(0) + ctx.records.len() as u64
        );
        Ok(())
    }
}

plex::export_policy!(Coordinated);
