//! Preble E2 global routing and cache-hit weighted local scheduling.

use std::cmp::Reverse;
use std::collections::{BTreeMap, VecDeque};

use plex::serde_json::json;
use plex::{
    CacheAdmission, CacheContext, CachePlan, FeedbackContext, FeedbackSubject, Host, OutcomeKind,
    Policy, RouteContext, RouteDecision, RoutePlan, ScheduleContext, SchedulePlan,
    ScheduleSelection, State,
};

struct Preble;

impl Policy for Preble {
    fn route(ctx: &RouteContext, state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut capacity = PrebleCapacity::new(ctx);
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        let mut dead_load = ctx
            .targets
            .iter()
            .map(|target| {
                let target_id = target.target_id.as_str();
                if let Some(observed) = target.facts["rolling_load_us"].as_u64() {
                    state.shared["preble_load_us"][target_id] = json!(observed);
                }
                state.shared["preble_load_us"][target_id]
                    .as_u64()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let mut pending = Vec::new();

        for (request_index, request) in ctx.requests.iter().enumerate() {
            let edges = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index && capacity.can_assign(ctx, edge)
                })
                .collect::<Vec<_>>();
            if edges.is_empty() {
                decisions.push(RouteDecision::Defer);
                continue;
            }
            let longest = edges
                .iter()
                .map(|(_, edge)| edge.facts["cached_tokens"].as_u64().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let missed = request.facts["uncached_tokens"]
                .as_u64()
                .unwrap_or_else(|| {
                    request.facts["prompt_tokens"]
                        .as_u64()
                        .unwrap_or(longest)
                        .saturating_sub(longest)
                });
            let exploit = missed < longest;
            let selected = if exploit {
                let longest_edges = edges
                    .iter()
                    .copied()
                    .filter(|(_, edge)| {
                        edge.facts["cached_tokens"].as_u64().unwrap_or(0) == longest
                    })
                    .collect::<Vec<_>>();
                let base = longest_edges
                    .iter()
                    .copied()
                    .min_by_key(|(edge_index, edge)| (preble_cost(edge), *edge_index));
                redirect_preble_exploit(request, &edges, base, &dead_load)
            } else {
                let imbalance_threshold = request.facts["decoder_imbalance_threshold_ppm"]
                    .as_u64()
                    .unwrap_or(1_500_000);
                let ratio = edges.iter().copied().max_by_key(|(edge_index, edge)| {
                    (preble_decoder_ratio(ctx, edge), Reverse(*edge_index))
                });
                if ratio
                    .is_some_and(|(_, edge)| preble_decoder_ratio(ctx, edge) > imbalance_threshold)
                {
                    ratio
                } else {
                    edges
                        .iter()
                        .copied()
                        .min_by_key(|(edge_index, edge)| (preble_cost(edge), *edge_index))
                }
            };

            if let Some((edge_index, edge)) = selected {
                let target_index = edge.target_index as usize;
                let target_id = ctx.targets[target_index].target_id.as_str();
                let load_delta = edge.facts["assignment_load_us"]
                    .as_u64()
                    .unwrap_or_else(|| preble_cost(edge));
                capacity.assign(ctx, edge);
                dead_load[target_index] = dead_load[target_index].saturating_add(load_delta);
                pending.push(json!({
                    "request_index": request_index,
                    "target_id": target_id,
                    "load_delta_us": load_delta
                }));
                decisions.push(RouteDecision::Assign(edge_index as u32));

                if request.facts["prefix_queue_time_doubled"]
                    .as_bool()
                    .unwrap_or(false)
                    && let (Some(object_id), Some(replica_target)) = (
                        request.facts["prefix_object_id"].as_str(),
                        request.facts["replica_target_id"].as_str(),
                    )
                {
                    state.shared["preble_replication_intents"][object_id] = json!(replica_target);
                }
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        state.shared["preble_route_pending"][ctx.meta.opportunity_id.as_str()] = json!(pending);
        trim_preble_pending(state);
        Ok(RoutePlan { decisions })
    }

    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        let mut queues = BTreeMap::<u32, VecDeque<usize>>::new();

        for (index, candidate) in ctx.runnable.iter().enumerate() {
            let queued = candidate.facts["queue_member"]
                .as_bool()
                .unwrap_or_else(|| candidate.facts["scheduler_state"].as_str() != Some("running"));
            if !queued {
                if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0 {
                    continue;
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
                continue;
            }
            let groups = candidate.facts["priority_groups"]
                .as_u64()
                .unwrap_or(10)
                .clamp(1, 100) as u32;
            let hit_ratio = candidate.facts["prefix_hit_ratio_ppm"]
                .as_u64()
                .unwrap_or(0)
                .min(1_000_000);
            let priority =
                ((hit_ratio.saturating_mul(u64::from(groups)) / 1_000_000) as u32).clamp(1, groups);
            queues.entry(priority).or_default().push_back(index);
        }

        let mut groups = queues.keys().copied().collect::<Vec<_>>();
        groups.sort_by_key(|priority| Reverse(*priority));
        if !groups.is_empty() {
            let cursor = state.shared["preble_group_cursor"].as_u64().unwrap_or(0) as usize;
            let group_count = groups.len();
            groups.rotate_left(cursor % group_count);
            state.shared["preble_group_cursor"] = json!(cursor.saturating_add(1));
        }
        while remaining_selections > 0
            && remaining_requests > 0
            && remaining_tokens > 0
            && queues.values().any(|queue| !queue.is_empty())
        {
            let mut progressed = false;
            for priority in &groups {
                for _ in 0..*priority {
                    if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0
                    {
                        break;
                    }
                    let Some(index) = queues.get_mut(priority).and_then(VecDeque::pop_front) else {
                        break;
                    };
                    let budget = u64::from(ctx.runnable[index].max_token_budget)
                        .min(remaining_tokens) as u32;
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
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        Ok(SchedulePlan { selections })
    }

    fn cache(ctx: &CacheContext, state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let mut admissions = Vec::with_capacity(ctx.prospective.len());
        for (index, object) in ctx.prospective.iter().enumerate() {
            let target = object.facts["replica_target_id"].as_str().or_else(|| {
                state.shared["preble_replication_intents"][object.object_id.as_str()].as_str()
            });
            if let Some(target) = target {
                host.prefetch_cache(
                    object.object_id.as_str(),
                    Some(target),
                    &format!(
                        "preble-replicate-{}-{index}",
                        ctx.meta.opportunity_id.as_str()
                    ),
                )?;
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
                FeedbackSubject::RouteAssignment(subject)
                    if record.outcome == OutcomeKind::Progress =>
                {
                    let opportunity = subject.opportunity_id.as_str();
                    let request_index = u64::from(subject.request_index);
                    let pending = state.shared["preble_route_pending"][opportunity]
                        .as_array()
                        .and_then(|entries| {
                            entries.iter().find(|entry| {
                                entry["request_index"].as_u64() == Some(request_index)
                            })
                        })
                        .cloned();
                    let Some(pending) = pending else {
                        continue;
                    };
                    if record.facts["status"].as_str() != Some("not-enacted") {
                        let target = pending["target_id"].as_str().unwrap_or("default");
                        state.shared["preble_load_us"][target] = json!(
                            state.shared["preble_load_us"][target]
                                .as_u64()
                                .unwrap_or(0)
                                .saturating_add(pending["load_delta_us"].as_u64().unwrap_or(0))
                        );
                    }
                    remove_preble_pending(state, opportunity, request_index);
                }
                FeedbackSubject::Request(_)
                    if matches!(
                        record.outcome,
                        OutcomeKind::Progress | OutcomeKind::Completed
                    ) =>
                {
                    if let (Some(target), Some(completed)) = (
                        record.facts["target_id"].as_str(),
                        record.facts["completed_load_us"].as_u64(),
                    ) {
                        state.shared["preble_load_us"][target] = json!(
                            state.shared["preble_load_us"][target]
                                .as_u64()
                                .unwrap_or(0)
                                .saturating_sub(completed)
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn preble_cost(edge: &plex::RouteEdge) -> u64 {
    edge.facts["load_cost"]
        .as_u64()
        .unwrap_or(0)
        .saturating_add(edge.facts["eviction_cost"].as_u64().unwrap_or(0))
        .saturating_add(edge.facts["miss_prefill_cost"].as_u64().unwrap_or(0))
}

fn preble_decoder_ratio(ctx: &RouteContext, edge: &plex::RouteEdge) -> u64 {
    edge.facts["decoder_ratio_ppm"]
        .as_u64()
        .or_else(|| ctx.targets[edge.target_index as usize].facts["decoder_ratio_ppm"].as_u64())
        .unwrap_or(0)
}

fn redirect_preble_exploit<'a>(
    request: &plex::RouteRequest,
    edges: &[(usize, &'a plex::RouteEdge)],
    base: Option<(usize, &'a plex::RouteEdge)>,
    load: &[u64],
) -> Option<(usize, &'a plex::RouteEdge)> {
    let Some(base) = base else {
        return None;
    };
    let threshold = request.facts["balance_threshold_ppm"]
        .as_u64()
        .unwrap_or(2_000_000);
    let heaviest = load
        .iter()
        .enumerate()
        .max_by_key(|(_, value)| *value)
        .map(|(index, value)| (index, *value));
    let lightest = load
        .iter()
        .enumerate()
        .min_by_key(|(_, value)| *value)
        .map(|(index, value)| (index, *value));
    let (Some((heavy_index, heavy)), Some((light_index, light))) = (heaviest, lightest) else {
        return Some(base);
    };
    let overloaded = heavy_index == base.1.target_index as usize
        && (light == 0
            || u128::from(heavy).saturating_mul(1_000_000)
                > u128::from(light).saturating_mul(u128::from(threshold)));
    if !overloaded {
        return Some(base);
    }
    edges
        .iter()
        .copied()
        .filter(|(_, edge)| edge.target_index as usize == light_index)
        .min_by_key(|(edge_index, edge)| (preble_cost(edge), *edge_index))
        .or(Some(base))
}

struct PrebleCapacity {
    assignments: Vec<u32>,
    remaining: Vec<BTreeMap<(String, String), u64>>,
}

impl PrebleCapacity {
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
        let mut demand = BTreeMap::<(String, String), u64>::new();
        for amount in &edge.demand {
            let key = (amount.name.clone(), amount.unit.clone());
            demand
                .entry(key)
                .and_modify(|value| *value = value.saturating_add(amount.amount))
                .or_insert(amount.amount);
        }
        demand.into_iter().all(|(key, amount)| {
            self.remaining[target]
                .get(&key)
                .is_some_and(|remaining| amount <= *remaining)
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

fn trim_preble_pending(state: &mut State) {
    const MAX_PENDING: usize = 64;
    let Some(pending) = state.shared["preble_route_pending"].as_object_mut() else {
        return;
    };
    if pending.len() <= MAX_PENDING {
        return;
    }
    let mut opportunities = pending.keys().cloned().collect::<Vec<_>>();
    opportunities.sort();
    for opportunity in opportunities.into_iter().take(pending.len() - MAX_PENDING) {
        pending.remove(&opportunity);
    }
}

fn remove_preble_pending(state: &mut State, opportunity: &str, request_index: u64) {
    if let Some(entries) = state.shared["preble_route_pending"][opportunity].as_array_mut() {
        entries.retain(|entry| entry["request_index"].as_u64() != Some(request_index));
    }
    if state.shared["preble_route_pending"][opportunity]
        .as_array()
        .is_some_and(Vec::is_empty)
        && let Some(pending) = state.shared["preble_route_pending"].as_object_mut()
    {
        pending.remove(opportunity);
    }
}

plex::export_policy!(Preble);
