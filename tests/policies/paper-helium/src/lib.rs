//! Helium execution of validated whole-DAG/TRT schedule plans.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use plex::serde_json::json;
use plex::{
    CacheAdmission, CacheContext, CachePlan, FeedbackContext, FeedbackSubject, Host, OutcomeKind,
    Policy, RouteContext, RouteDecision, RoutePlan, ScheduleContext, SchedulePlan,
    ScheduleSelection, State,
};

struct Helium;

impl Policy for Helium {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut capacity = HeliumCapacity::new(ctx);
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let planned = request.facts["planned_worker_id"].as_str();
            let selected = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index && capacity.can_assign(ctx, edge)
                })
                .min_by_key(|(edge_index, edge)| {
                    let target = ctx.targets[edge.target_index as usize].target_id.as_str();
                    (
                        planned.is_some_and(|planned| planned != target),
                        edge.facts["planned_rank"].as_u64().unwrap_or(u64::MAX),
                        edge.facts["estimated_token_steps"]
                            .as_u64()
                            .unwrap_or(u64::MAX),
                        *edge_index,
                    )
                });
            if let Some((edge_index, edge)) = selected {
                capacity.assign(ctx, edge);
                decisions.push(RouteDecision::Assign(edge_index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }

    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut ready = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            let workflow = helium_workflow(candidate);
            let worker = candidate.facts["worker_id"].as_str().unwrap_or("default");
            let segment = candidate.facts["segment_index"].as_u64().unwrap_or(0);
            let cursor = state.shared["helium_cursor"][&workflow][worker]
                .as_u64()
                .unwrap_or(0);
            let dependency_ready = candidate.facts["dependency_ready"]
                .as_bool()
                .or_else(|| candidate.facts["ready"].as_bool())
                .unwrap_or(false);
            let now_step = candidate.facts["now_token_step"].as_u64().unwrap_or(0);
            let precedence_ready =
                candidate.facts["precedence_ready_at"].as_u64().unwrap_or(0) <= now_step;
            if segment == cursor && dependency_ready && precedence_ready {
                ready.push(index);
            }
        }
        let forced = if ready.is_empty() {
            ctx.runnable
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    let workflow = helium_workflow(candidate);
                    let worker = candidate.facts["worker_id"].as_str().unwrap_or("default");
                    candidate.facts["segment_index"].as_u64().unwrap_or(0)
                        == state.shared["helium_cursor"][&workflow][worker]
                            .as_u64()
                            .unwrap_or(0)
                })
                .min_by_key(|(index, candidate)| {
                    (
                        candidate.facts["earliest_start"]
                            .as_u64()
                            .unwrap_or(u64::MAX),
                        *index,
                    )
                })
                .map(|(index, _)| index)
        } else {
            None
        };
        let mut order = if let Some(forced) = forced {
            vec![forced]
        } else {
            ready
        };
        order.sort_by_key(|index| {
            let candidate = &ctx.runnable[*index];
            (
                candidate.facts["worker_id"].as_str().unwrap_or("default"),
                candidate.facts["sequence_path"].as_str().unwrap_or(""),
                Reverse(
                    candidate.facts["critical_path_depth"]
                        .as_u64()
                        .or_else(|| candidate.facts["dependency_depth"].as_u64())
                        .unwrap_or(0),
                ),
                candidate.facts["earliest_start"]
                    .as_u64()
                    .unwrap_or(u64::MAX),
                *index,
            )
        });
        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        for index in order {
            if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let budget =
                u64::from(ctx.runnable[index].max_token_budget).min(remaining_tokens) as u32;
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
        }
        Ok(SchedulePlan { selections })
    }

    fn cache(ctx: &CacheContext, state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let mut admissions = Vec::with_capacity(ctx.prospective.len());
        for (index, object) in ctx.prospective.iter().enumerate() {
            if object.facts["proactive_warm"].as_bool().unwrap_or(false) {
                let action = host.prefetch_cache(
                    object.object_id.as_str(),
                    object.facts["worker_id"].as_str(),
                    &format!("helium-warm-{}-{index}", ctx.meta.opportunity_id.as_str()),
                )?;
                state.shared["helium_actions"][&action.0.to_string()] = json!({
                    "object_id": object.object_id.as_str()
                });
                admissions.push(CacheAdmission::Cache);
            } else {
                admissions.push(CacheAdmission::Bypass);
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
                FeedbackSubject::Request(_) if record.outcome == OutcomeKind::Completed => {
                    if let (Some(workflow), Some(worker), Some(segment)) = (
                        record.facts["workflow_id"].as_str(),
                        record.facts["worker_id"].as_str(),
                        record.facts["segment_index"].as_u64(),
                    ) {
                        state.shared["helium_cursor"][workflow][worker] = json!(
                            state.shared["helium_cursor"][workflow][worker]
                                .as_u64()
                                .unwrap_or(0)
                                .max(segment.saturating_add(1))
                        );
                    }
                }
                FeedbackSubject::Action(action_id)
                    if matches!(
                        record.outcome,
                        OutcomeKind::ActionSucceeded | OutcomeKind::ActionFailed
                    ) =>
                {
                    let key = action_id.0.to_string();
                    let action = state.shared["helium_actions"][&key].clone();
                    if let Some(object_id) = action["object_id"].as_str() {
                        state.shared["helium_warm"][object_id] =
                            json!(record.outcome == OutcomeKind::ActionSucceeded);
                    }
                    if let Some(actions) = state.shared["helium_actions"].as_object_mut() {
                        actions.remove(&key);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn helium_workflow(candidate: &plex::ScheduleCandidate) -> String {
    candidate.facts["workflow_id"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            candidate
                .request
                .group_id
                .as_ref()
                .map(|group| group.as_str().to_owned())
        })
        .unwrap_or_else(|| candidate.request.request_id.as_str().to_owned())
}

struct HeliumCapacity {
    assignments: Vec<u32>,
    remaining: Vec<BTreeMap<(String, String), u64>>,
}

impl HeliumCapacity {
    fn new(ctx: &RouteContext) -> Self {
        Self {
            assignments: vec![0; ctx.targets.len()],
            remaining: ctx
                .targets
                .iter()
                .map(|target| {
                    target
                        .capacity
                        .iter()
                        .map(|limit| ((limit.name.clone(), limit.unit.clone()), limit.maximum))
                        .collect()
                })
                .collect(),
        }
    }

    fn can_assign(&self, ctx: &RouteContext, edge: &plex::RouteEdge) -> bool {
        let target = edge.target_index as usize;
        if self.assignments[target] >= ctx.targets[target].max_assignments {
            return false;
        }
        if self.remaining[target].is_empty() {
            return true;
        }
        edge.demand.iter().all(|amount| {
            self.remaining[target]
                .get(&(amount.name.clone(), amount.unit.clone()))
                .is_some_and(|remaining| amount.amount <= *remaining)
        })
    }

    fn assign(&mut self, ctx: &RouteContext, edge: &plex::RouteEdge) {
        let target = edge.target_index as usize;
        self.assignments[target] = self.assignments[target].saturating_add(1);
        for amount in &edge.demand {
            if let Some(remaining) =
                self.remaining[target].get_mut(&(amount.name.clone(), amount.unit.clone()))
            {
                *remaining = remaining.saturating_sub(amount.amount);
            }
        }
        debug_assert!(self.assignments[target] <= ctx.targets[target].max_assignments);
    }
}

plex::export_policy!(Helium);
