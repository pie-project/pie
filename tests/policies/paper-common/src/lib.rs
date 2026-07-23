#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use plex::serde_json::json;
use plex::{
    AdmissionDecision, AdmitContext, AdmitPlan, CacheAdmission, CacheContext, CachePlan,
    FeedbackContext, FeedbackSubject, Host, OutcomeKind, Policy, RouteContext, RouteDecision,
    RoutePlan, ScheduleContext, SchedulePlan, ScheduleSelection, State,
};

pub struct Vtc;

impl Policy for Vtc {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let previous_active = state.shared["vtc_active"]
            .as_array()
            .map(|clients| {
                clients
                    .iter()
                    .filter_map(|client| client.as_str().map(str::to_owned))
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        let clients = ctx
            .runnable
            .iter()
            .map(|candidate| {
                candidate.facts["client_id"]
                    .as_str()
                    .unwrap_or("default")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let active_minimum = clients
            .iter()
            .filter(|client| previous_active.contains(*client))
            .filter_map(|client| state.shared["vtc"][client].as_u64())
            .min()
            .or_else(|| {
                clients
                    .iter()
                    .filter_map(|client| state.shared["vtc"][client].as_u64())
                    .min()
            })
            .unwrap_or(0);
        let mut counters = BTreeMap::new();
        for client in &clients {
            let mut counter = state.shared["vtc"][client]
                .as_u64()
                .unwrap_or(active_minimum);
            if !previous_active.contains(client) {
                counter = counter.max(active_minimum);
            }
            state.shared["vtc"][client] = json!(counter);
            counters.insert(client.clone(), counter);
        }
        state.shared["vtc_active"] = json!(clients.iter().cloned().collect::<BTreeSet<_>>());

        let mut available = (0..ctx.runnable.len()).collect::<BTreeSet<_>>();
        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        while !available.is_empty()
            && remaining_selections > 0
            && remaining_requests > 0
            && remaining_tokens > 0
        {
            let index = *available
                .iter()
                .min_by_key(|&&index| (&counters[&clients[index]], index))
                .expect("available set is non-empty");
            available.remove(&index);
            let budget =
                u64::from(ctx.runnable[index].max_token_budget).min(remaining_tokens) as u32;
            if budget == 0 {
                continue;
            }
            selections.push(ScheduleSelection {
                requests: vec![index as u32],
                token_budgets: vec![budget],
            });
            let input_weight = ctx.runnable[index].facts["input_weight"]
                .as_u64()
                .unwrap_or(1);
            let input_charge = ctx.runnable[index].facts["dispatch_input_tokens"]
                .as_u64()
                .unwrap_or(0)
                .saturating_mul(input_weight);
            let client = &clients[index];
            counters
                .entry(client.clone())
                .and_modify(|counter| *counter = counter.saturating_add(input_charge));
            state.shared["vtc"][client] = json!(counters[client]);
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        Ok(SchedulePlan { selections })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            if record.outcome != OutcomeKind::Progress
                || !matches!(record.subject, FeedbackSubject::Request(_))
            {
                continue;
            }
            let client = record.facts["client_id"].as_str().unwrap_or("default");
            let output_weight = record.facts["output_weight"].as_u64().unwrap_or(1);
            let charge = record.facts["output_tokens"]
                .as_u64()
                .unwrap_or(0)
                .saturating_mul(output_weight);
            let previous = state.shared["vtc"][client].as_u64().unwrap_or(0);
            state.shared["vtc"][client] = json!(previous.saturating_add(charge));
        }
        Ok(())
    }
}

pub struct LMetric;

impl Policy for LMetric {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let decisions = ctx
            .requests
            .iter()
            .enumerate()
            .map(|(request_index, _)| {
                ctx.feasible_edges
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| edge.request_index as usize == request_index)
                    .filter(|edge| {
                        let target = &ctx.targets[edge.1.target_index as usize];
                        !target.facts["hotspot_confirmed"].as_bool().unwrap_or(false)
                    })
                    .min_by_key(|(_, edge)| {
                        edge.facts["new_prefill_tokens"]
                            .as_u64()
                            .unwrap_or(u64::MAX)
                            .saturating_mul(
                                edge.facts["current_batch_size"]
                                    .as_u64()
                                    .unwrap_or(u64::MAX)
                                    .saturating_add(1),
                            )
                    })
                    .map_or(RouteDecision::Defer, |(index, _)| {
                        RouteDecision::Assign(index as u32)
                    })
            })
            .collect();
        Ok(RoutePlan { decisions })
    }
}

pub struct FairServe;

impl Policy for FairServe {
    fn admit(ctx: &AdmitContext, state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut order = (0..ctx.candidates.len())
            .filter(|&index| {
                let candidate = &ctx.candidates[index];
                !candidate.facts["kv_overloaded"].as_bool().unwrap_or(false)
                    || candidate.facts["interaction_in_progress"]
                        .as_bool()
                        .unwrap_or(false)
                    || (candidate.facts["user_rpm_remaining"].as_u64().unwrap_or(0) > 0
                        && candidate.facts["app_rpm_remaining"].as_u64().unwrap_or(0) > 0)
            })
            .collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.candidates[index];
            let application = candidate.facts["application_id"]
                .as_str()
                .or_else(|| candidate.facts["client_id"].as_str())
                .unwrap_or("default");
            let weight = candidate.facts["weight"].as_u64().unwrap_or(1).max(1);
            let service = state.shared["fairserve"][application].as_u64().unwrap_or(0) / weight;
            (
                !candidate.facts["interaction_in_progress"]
                    .as_bool()
                    .unwrap_or(false),
                service,
                index,
            )
        });
        let accepted = order
            .into_iter()
            .take(ctx.capacity.max_accepted as usize)
            .collect::<BTreeSet<_>>();
        Ok(AdmitPlan {
            decisions: (0..ctx.candidates.len())
                .map(|index| {
                    if accepted.contains(&index) {
                        AdmissionDecision::Accept
                    } else {
                        AdmissionDecision::Defer
                    }
                })
                .collect(),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let minimum = ctx
            .runnable
            .iter()
            .filter_map(|candidate| {
                let application = candidate.facts["application_id"]
                    .as_str()
                    .or_else(|| candidate.facts["client_id"].as_str())
                    .unwrap_or("default");
                state.shared["fairserve"][application].as_u64()
            })
            .min()
            .unwrap_or(0);
        for candidate in &ctx.runnable {
            let application = candidate.facts["application_id"]
                .as_str()
                .or_else(|| candidate.facts["client_id"].as_str())
                .unwrap_or("default");
            if state.shared["fairserve"][application].is_null() {
                state.shared["fairserve"][application] = json!(minimum);
            }
        }
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let application = candidate.facts["application_id"]
                .as_str()
                .or_else(|| candidate.facts["client_id"].as_str())
                .unwrap_or("default");
            let weight = candidate.facts["weight"].as_u64().unwrap_or(1).max(1);
            (
                !candidate.facts["interaction_in_progress"]
                    .as_bool()
                    .unwrap_or(false),
                state.shared["fairserve"][application].as_u64().unwrap_or(0) / weight,
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            if record.outcome != OutcomeKind::Progress
                || !matches!(record.subject, FeedbackSubject::Request(_))
            {
                continue;
            }
            let application = record.facts["application_id"]
                .as_str()
                .or_else(|| record.facts["client_id"].as_str())
                .unwrap_or("default");
            let input_weight = record.facts["input_weight"].as_u64().unwrap_or(1);
            let system_weight = record.facts["system_weight"].as_u64().unwrap_or(1);
            let output_weight = record.facts["output_weight"].as_u64().unwrap_or(1);
            let service = record.facts["input_tokens"]
                .as_u64()
                .unwrap_or(0)
                .saturating_mul(input_weight)
                .saturating_add(
                    record.facts["system_tokens"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_mul(system_weight),
                )
                .saturating_add(
                    record.facts["output_tokens"]
                        .as_u64()
                        .unwrap_or_else(|| record.facts["service_tokens"].as_u64().unwrap_or(0))
                        .saturating_mul(output_weight),
                );
            state.shared["fairserve"][application] = json!(
                state.shared["fairserve"][application]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(service)
            );
        }
        Ok(())
    }
}

pub struct Marconi;

impl Policy for Marconi {
    fn cache(ctx: &CacheContext, _state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        #[derive(Clone, Copy)]
        enum Kind {
            Resident(u32),
            Prospective(u32),
        }

        let mut fixed_bytes = ctx.capacity.fixed_bytes;
        let mut candidates = Vec::new();
        for (index, resident) in ctx.resident.iter().enumerate() {
            if resident.reclaimable {
                candidates.push((
                    value(&resident.object),
                    resident.object.size_bytes,
                    Kind::Resident(index as u32),
                ));
            } else {
                fixed_bytes = fixed_bytes.saturating_add(resident.object.size_bytes);
            }
        }
        for (index, object) in ctx.prospective.iter().enumerate() {
            candidates.push((
                value(object),
                object.size_bytes,
                Kind::Prospective(index as u32),
            ));
        }
        candidates.sort_by_key(|(value, size, kind)| {
            let tie = match kind {
                Kind::Resident(index) | Kind::Prospective(index) => *index,
            };
            (Reverse(*value), *size, tie)
        });
        let mut remaining = ctx.capacity.max_bytes.saturating_sub(fixed_bytes);
        let mut retained_residents = BTreeSet::new();
        let mut admitted = BTreeSet::new();
        for (_, size, kind) in candidates {
            if size > remaining {
                continue;
            }
            remaining -= size;
            match kind {
                Kind::Resident(index) => {
                    retained_residents.insert(index);
                }
                Kind::Prospective(index) => {
                    admitted.insert(index);
                }
            }
        }
        Ok(CachePlan {
            admissions: (0..ctx.prospective.len())
                .map(|index| {
                    if admitted.contains(&(index as u32)) {
                        CacheAdmission::Cache
                    } else {
                        CacheAdmission::Bypass
                    }
                })
                .collect(),
            reclaim: ctx
                .resident
                .iter()
                .enumerate()
                .filter(|(index, resident)| {
                    resident.reclaimable && !retained_residents.contains(&(*index as u32))
                })
                .map(|(index, _)| index as u32)
                .collect(),
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        state.shared["marconi_feedback_records"] = json!(
            state.shared["marconi_feedback_records"]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(ctx.records.len() as u64)
        );
        Ok(())
    }
}

pub struct RagCache;

impl Policy for RagCache {
    fn cache(ctx: &CacheContext, _state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        let mut ordered = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| {
                resident.reclaimable && resident.object.facts["leaf"].as_bool().unwrap_or(false)
            })
            .map(|(index, resident)| {
                let frequency = resident.object.facts["frequency"].as_u64().unwrap_or(1);
                let cost = resident.object.facts["recompute_cost"]
                    .as_u64()
                    .unwrap_or(0);
                let age = resident.object.facts["age"].as_u64().unwrap_or(0);
                (
                    (
                        age.saturating_add(
                            cost.saturating_mul(frequency) / resident.object.size_bytes.max(1),
                        ),
                        index,
                    ),
                    index as u32,
                )
            })
            .collect::<Vec<_>>();
        ordered.sort_by_key(|entry| entry.0);
        let admissions = vec![CacheAdmission::Cache; ctx.prospective.len()];
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                ordered.into_iter().map(|(_, index)| index),
            ),
            admissions,
        })
    }
}

pub struct Dlpm;

impl Policy for Dlpm {
    fn route(ctx: &RouteContext, state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut target_counts = vec![0u32; ctx.targets.len()];
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let client = request.facts["client_id"].as_str().unwrap_or("default");
            let client_deficit = state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0);
            let has_positive_worker = ctx.feasible_edges.iter().any(|edge| {
                edge.request_index as usize == request_index
                    && edge.facts["worker_deficit"]
                        .as_i64()
                        .unwrap_or(client_deficit)
                        > 0
            });
            let selected = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                        && (!has_positive_worker
                            || edge.facts["worker_deficit"]
                                .as_i64()
                                .unwrap_or(client_deficit)
                                > 0)
                })
                .max_by_key(|(_, edge)| {
                    if has_positive_worker {
                        (
                            edge.facts["cached_tokens"].as_i64().unwrap_or(0),
                            -edge.facts["load"].as_i64().unwrap_or(0),
                        )
                    } else {
                        (
                            -edge.facts["load"].as_i64().unwrap_or(0),
                            edge.facts["cached_tokens"].as_i64().unwrap_or(0),
                        )
                    }
                });
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                decisions.push(RouteDecision::Assign(index as u32));
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
        let clients = ctx
            .runnable
            .iter()
            .map(|candidate| {
                candidate.facts["client_id"]
                    .as_str()
                    .unwrap_or("default")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        if !clients
            .iter()
            .any(|client| state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0) > 0)
        {
            for (candidate, client) in ctx.runnable.iter().zip(&clients) {
                let quantum = candidate.facts["quantum"].as_i64().unwrap_or(1).max(1);
                state.shared["dlpm_deficit"][client] = json!(
                    state.shared["dlpm_deficit"][client]
                        .as_i64()
                        .unwrap_or(0)
                        .saturating_add(quantum)
                );
            }
        }
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            (
                Reverse(
                    ctx.runnable[index].facts["cached_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                ),
                index,
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
            let client = &clients[index];
            if state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0) <= 0 {
                continue;
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
            let extend = ctx.runnable[index].facts["extend_tokens"]
                .as_i64()
                .unwrap_or(i64::from(budget));
            state.shared["dlpm_deficit"][client] = json!(
                state.shared["dlpm_deficit"][client]
                    .as_i64()
                    .unwrap_or(0)
                    .saturating_sub(extend)
            );
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        Ok(SchedulePlan { selections })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            if record.outcome != OutcomeKind::Progress
                || !matches!(record.subject, FeedbackSubject::Request(_))
            {
                continue;
            }
            let client = record.facts["client_id"].as_str().unwrap_or("default");
            let service = record.facts["output_tokens"]
                .as_i64()
                .unwrap_or_else(|| record.facts["service_tokens"].as_i64().unwrap_or(0));
            state.shared["dlpm_deficit"][client] = json!(
                state.shared["dlpm_deficit"][client]
                    .as_i64()
                    .unwrap_or(0)
                    .saturating_sub(service)
            );
        }
        Ok(())
    }
}

pub struct InferCept;

impl Policy for InferCept {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            (
                !ctx.runnable[index].facts["resuming"]
                    .as_bool()
                    .unwrap_or(false),
                ctx.runnable[index].facts["expected_waste_tokens"]
                    .as_u64()
                    .unwrap_or(u64::MAX),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn cache(ctx: &CacheContext, _state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.reclaimable)
            .map(|(index, resident)| {
                (
                    resident.object.facts["expected_reuse_ms"]
                        .as_u64()
                        .unwrap_or(u64::MAX),
                    index as u32,
                )
            })
            .collect::<Vec<_>>();
        reclaim.sort_by_key(|entry| *entry);
        for resident in &ctx.resident {
            if resident.object.facts["swap"].as_bool() == Some(true) {
                host.swap_cache(
                    resident.object.object_id.as_str(),
                    "cpu",
                    &format!("infercept-{}", resident.object.object_id.as_str()),
                )?;
            }
        }
        let admissions = ctx
            .prospective
            .iter()
            .map(|object| {
                if object.facts["expected_reuse_ms"]
                    .as_u64()
                    .unwrap_or(u64::MAX)
                    < object.facts["recompute_ms"].as_u64().unwrap_or(0)
                {
                    CacheAdmission::Cache
                } else {
                    CacheAdmission::Bypass
                }
            })
            .collect::<Vec<_>>();
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                reclaim.into_iter().map(|(_, index)| index),
            ),
            admissions,
        })
    }
}

pub struct Peek;

impl Policy for Peek {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        state.shared["peek_pending"] = json!(ctx.runnable.len());
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let fairness_lane = ctx.runnable[index].facts["waiting_ms"]
                .as_u64()
                .unwrap_or(0)
                >= ctx.runnable[index].facts["fairness_threshold_ms"]
                    .as_u64()
                    .unwrap_or(u64::MAX);
            (
                !fairness_lane,
                Reverse(
                    ctx.runnable[index].facts["demand_depth"]
                        .as_u64()
                        .unwrap_or(0),
                ),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn cache(ctx: &CacheContext, state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        let pending = state.shared["peek_pending"].as_u64().unwrap_or(0);
        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.reclaimable)
            .map(|(index, resident)| {
                (
                    resident.object.facts["pending_demand_depth"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_add(pending),
                    index as u32,
                )
            })
            .collect::<Vec<_>>();
        reclaim.sort_by_key(|entry| *entry);
        let admissions = vec![CacheAdmission::Cache; ctx.prospective.len()];
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                reclaim.into_iter().map(|(_, index)| index),
            ),
            admissions,
        })
    }
}

pub struct Qlm;

impl Policy for Qlm {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut remaining = ctx.capacity.max_accepted;
        Ok(AdmitPlan {
            decisions: ctx
                .candidates
                .iter()
                .map(|candidate| {
                    if remaining > 0
                        && candidate.facts["estimated_wait_ms"].as_u64().unwrap_or(0)
                            <= candidate.facts["slo_ms"].as_u64().unwrap_or(u64::MAX)
                    {
                        remaining -= 1;
                        AdmissionDecision::Accept
                    } else {
                        AdmissionDecision::Defer
                    }
                })
                .collect(),
        })
    }

    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Ok(RoutePlan {
            decisions: min_edge_by(ctx, |edge| {
                edge.facts["estimated_wait_ms"].as_u64().unwrap_or(u64::MAX)
            }),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let group = candidate
                .request
                .group_id
                .as_ref()
                .and_then(|group_id| state.group(group_id.as_str()).ok());
            (
                candidate.facts["virtual_wait"]
                    .as_u64()
                    .or_else(|| group.and_then(|group| group.facts()["virtual_wait"].as_u64()))
                    .or_else(|| group.and_then(|group| group.scratch["virtual_wait"].as_u64()))
                    .unwrap_or(0),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let Some(virtual_wait) = record.facts["virtual_wait"].as_u64() else {
                continue;
            };
            match &record.subject {
                FeedbackSubject::WorkGroup(group_id) => {
                    state.group_mut(group_id.as_str())?.scratch["virtual_wait"] =
                        json!(virtual_wait);
                }
                FeedbackSubject::Request(request_id) => {
                    state.request_mut(request_id.as_str())?.scratch["virtual_wait"] =
                        json!(virtual_wait);
                }
                _ => {}
            }
        }
        count_feedback(state, "qlm_feedback_records", ctx.records.len());
        Ok(())
    }
}

pub struct SlosServe;

impl Policy for SlosServe {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut remaining = ctx.capacity.max_accepted;
        Ok(AdmitPlan {
            decisions: ctx
                .candidates
                .iter()
                .map(|candidate| {
                    if remaining > 0
                        && candidate.facts["predicted_total_ms"]
                            .as_u64()
                            .unwrap_or(u64::MAX)
                            <= candidate.facts["slo_ms"].as_u64().unwrap_or(0)
                    {
                        remaining -= 1;
                        AdmissionDecision::Accept
                    } else {
                        AdmissionDecision::Defer
                    }
                })
                .collect(),
        })
    }

    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Ok(RoutePlan {
            decisions: min_edge_by(ctx, |edge| {
                edge.facts["stage_latency_ms"].as_u64().unwrap_or(u64::MAX)
            }),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            (
                ctx.runnable[index].facts["slack_ms"]
                    .as_i64()
                    .unwrap_or(i64::MAX),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }
}

pub struct Dynasor;

impl Policy for Dynasor {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            if candidate.facts["confidence_ppm"].as_u64().unwrap_or(0)
                >= candidate.facts["stop_threshold_ppm"]
                    .as_u64()
                    .unwrap_or(u64::MAX)
            {
                host.cancel_request(
                    candidate.request.request_id.as_str(),
                    &format!("dynasor-{index}"),
                    Some("progress threshold reached"),
                )?;
            } else {
                order.push(index);
            }
        }
        order.sort_by_key(|&index| {
            (
                Reverse(
                    ctx.runnable[index].facts["progress_ppm"]
                        .as_u64()
                        .unwrap_or(0),
                ),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        count_feedback(state, "dynasor_feedback_records", ctx.records.len());
        Ok(())
    }
}

pub struct Justitia;

impl Policy for Justitia {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let completed = candidate.facts["completed_branches"]
                .as_u64()
                .or_else(|| {
                    candidate
                        .request
                        .group_id
                        .as_ref()
                        .and_then(|group_id| state.group(group_id.as_str()).ok())
                        .and_then(|group| group.scratch["completed_branches"].as_u64())
                })
                .unwrap_or(0);
            (completed, index)
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            if record.outcome != OutcomeKind::Completed {
                continue;
            }
            let FeedbackSubject::WorkGroup(group_id) = &record.subject else {
                continue;
            };
            let group = state.group_mut(group_id.as_str())?;
            group.scratch["completed_branches"] =
                json!(group.scratch["completed_branches"].as_u64().unwrap_or(0) + 1);
        }
        Ok(())
    }
}

pub struct Chameleon;

impl Policy for Chameleon {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut remaining = ctx.capacity.max_accepted;
        Ok(AdmitPlan {
            decisions: ctx
                .candidates
                .iter()
                .map(|candidate| {
                    let size = candidate.facts["weighted_size"].as_u64().unwrap_or(1);
                    if remaining > 0 && size <= candidate.facts["queue_quota"].as_u64().unwrap_or(0)
                    {
                        remaining -= 1;
                        AdmissionDecision::Accept
                    } else {
                        AdmissionDecision::Defer
                    }
                })
                .collect(),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            (
                ctx.runnable[index].facts["queue_class"]
                    .as_u64()
                    .unwrap_or(0),
                Reverse(
                    ctx.runnable[index].facts["waiting_ms"]
                        .as_u64()
                        .unwrap_or(0),
                ),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn cache(ctx: &CacheContext, _state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        let admissions = ctx
            .prospective
            .iter()
            .map(|object| {
                if object.facts["adapter_hot"].as_bool().unwrap_or(false) {
                    CacheAdmission::Cache
                } else {
                    CacheAdmission::Bypass
                }
            })
            .collect::<Vec<_>>();
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                ctx.resident
                    .iter()
                    .enumerate()
                    .filter(|(_, resident)| resident.reclaimable)
                    .map(|(index, _)| index as u32),
            ),
            admissions,
        })
    }
}

pub struct HotPrefix;

impl Policy for HotPrefix {
    fn cache(ctx: &CacheContext, state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let threshold = ctx.capacity.facts["hot_threshold"].as_u64().unwrap_or(1);
        let mut admissions = Vec::new();
        for object in &ctx.prospective {
            let hotness = state.shared["hotprefix"][object.object_id.as_str()]
                .as_u64()
                .unwrap_or_else(|| object.facts["hotness"].as_u64().unwrap_or(0));
            if hotness >= threshold {
                admissions.push(CacheAdmission::Cache);
                host.prefetch_cache(
                    object.object_id.as_str(),
                    None,
                    &format!("hotprefix-{}", object.object_id.as_str()),
                )?;
            } else {
                admissions.push(CacheAdmission::Bypass);
            }
        }
        let ordered = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| {
                resident.reclaimable
                    && state.shared["hotprefix"][resident.object.object_id.as_str()]
                        .as_u64()
                        .unwrap_or(0)
                        < threshold
            })
            .map(|(index, _)| index as u32);
        Ok(CachePlan {
            reclaim: reclaim_prefix(ctx, &admissions, ordered),
            admissions,
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let FeedbackSubject::CacheObject(object_id) = &record.subject else {
                continue;
            };
            let delta = record.facts["reuse_count"].as_u64().unwrap_or(0);
            state.shared["hotprefix"][object_id.as_str()] = json!(
                state.shared["hotprefix"][object_id.as_str()]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(delta)
            );
        }
        Ok(())
    }
}

pub struct Pard;

impl Policy for Pard {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            let projected = candidate.facts["upstream_elapsed_ms"]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(candidate.facts["current_queue_ms"].as_u64().unwrap_or(0))
                .saturating_add(
                    candidate.facts["current_execution_ms"]
                        .as_u64()
                        .unwrap_or(0),
                )
                .saturating_add(candidate.facts["downstream_queue_ms"].as_u64().unwrap_or(0))
                .saturating_add(
                    candidate.facts["downstream_execution_ms"]
                        .as_u64()
                        .unwrap_or(0),
                )
                .saturating_add(
                    candidate.facts["downstream_batch_wait_p10_ms"]
                        .as_u64()
                        .unwrap_or(0),
                );
            if projected > candidate.facts["deadline_ms"].as_u64().unwrap_or(u64::MAX) {
                host.cancel_request(
                    candidate.request.request_id.as_str(),
                    &format!("pard-{index}"),
                    Some("projected deadline miss"),
                )?;
            } else {
                order.push(index);
            }
        }
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        count_feedback(state, "pard_feedback_records", ctx.records.len());
        Ok(())
    }
}

pub struct BranchRegulation;

impl Policy for BranchRegulation {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut accepted_by_group = std::collections::BTreeMap::<String, u64>::new();
        let mut remaining = ctx.capacity.max_accepted;
        Ok(AdmitPlan {
            decisions: ctx
                .candidates
                .iter()
                .map(|candidate| {
                    let group = candidate
                        .request
                        .group_id
                        .as_ref()
                        .map(|group| group.0.clone())
                        .unwrap_or_else(|| candidate.request.request_id.0.clone());
                    let accepted = accepted_by_group.entry(group).or_default();
                    let limit = candidate.facts["branch_limit"].as_u64().unwrap_or(1);
                    if remaining > 0
                        && *accepted < limit
                        && candidate.facts["batch_interference"].as_u64().unwrap_or(0)
                            <= candidate.facts["interference_limit"]
                                .as_u64()
                                .unwrap_or(u64::MAX)
                    {
                        *accepted += 1;
                        remaining -= 1;
                        AdmissionDecision::Accept
                    } else {
                        AdmissionDecision::Defer
                    }
                })
                .collect(),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            if candidate.facts["excess_branch"].as_bool() == Some(true) {
                continue;
            } else {
                order.push(index);
            }
        }
        Ok(select_singletons(ctx, order))
    }
}

pub struct DualMap;

impl Policy for DualMap {
    fn route(ctx: &RouteContext, _state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        let mut target_counts = vec![0u32; ctx.targets.len()];
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let mut candidates = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                        && edge.facts["hash_candidate"].as_bool().unwrap_or(true)
                })
                .collect::<Vec<_>>();
            candidates
                .sort_by_key(|(_, edge)| edge.facts["hash_choice"].as_u64().unwrap_or(u64::MAX));
            candidates.truncate(2);
            let affinity = candidates.iter().copied().max_by_key(|(_, edge)| {
                (
                    edge.facts["prefix_hit_tokens"].as_u64().unwrap_or(0),
                    Reverse(edge.facts["predicted_ttft_ms"].as_u64().unwrap_or(u64::MAX)),
                )
            });
            let selected = affinity
                .filter(|(_, edge)| {
                    edge.facts["predicted_ttft_ms"].as_u64().unwrap_or(u64::MAX)
                        <= request.facts["slo_ms"].as_u64().unwrap_or(u64::MAX)
                })
                .or_else(|| {
                    candidates.iter().copied().min_by_key(|(_, edge)| {
                        edge.facts["predicted_ttft_ms"].as_u64().unwrap_or(u64::MAX)
                    })
                });
            if let Some((index, edge)) = selected {
                let target = &ctx.targets[edge.target_index as usize];
                target_counts[edge.target_index as usize] += 1;
                if request.facts["hotspot"].as_bool() == Some(true) {
                    host.rebalance_request(
                        request.request.request_id.as_str(),
                        target.target_id.as_str(),
                        &format!("dualmap-{request_index}"),
                    )?;
                }
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }
}

pub struct Llumnix;

impl Policy for Llumnix {
    fn route(ctx: &RouteContext, _state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let mut decisions = Vec::new();
        let mut target_counts = vec![0u32; ctx.targets.len()];
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let selected = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .max_by_key(|(_, edge)| {
                    let target = &ctx.targets[edge.target_index as usize];
                    target.facts["memory_capacity"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_sub(edge.facts["virtual_usage"].as_u64().unwrap_or(u64::MAX))
                        / target.facts["batch_size"].as_u64().unwrap_or(1).max(1)
                });
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                if request.facts["live_reschedule"].as_bool() == Some(true) {
                    host.rebalance_request(
                        request.request.request_id.as_str(),
                        ctx.targets[edge.target_index as usize].target_id.as_str(),
                        &format!("llumnix-{request_index}"),
                    )?;
                }
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        state.shared["llumnix_feedback"] = json!(
            state.shared["llumnix_feedback"]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(ctx.records.len() as u64)
        );
        Ok(())
    }
}

pub struct SMetric;

impl Policy for SMetric {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut decisions = Vec::new();
        let mut target_counts = vec![0u32; ctx.targets.len()];
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let followup = request.request.generation_id > 0;
            let candidates = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .collect::<Vec<_>>();
            let selected = if followup {
                let affinity = candidates.iter().copied().max_by_key(|(_, edge)| {
                    (
                        edge.facts["cache_affinity"].as_u64().unwrap_or(0),
                        Reverse(edge.facts["load"].as_u64().unwrap_or(u64::MAX)),
                    )
                });
                let mean_load = if candidates.is_empty() {
                    0
                } else {
                    candidates
                        .iter()
                        .map(|(_, edge)| edge.facts["load"].as_u64().unwrap_or(0))
                        .sum::<u64>()
                        / candidates.len() as u64
                };
                affinity
                    .filter(|(_, edge)| {
                        let overload_ppm =
                            request.facts["overload_ppm"].as_u64().unwrap_or(1_000_000);
                        let hit_ratio_ppm =
                            request.facts["hit_ratio_ppm"].as_u64().unwrap_or(1_000_000);
                        edge.facts["load"].as_u64().unwrap_or(u64::MAX)
                            <= mean_load.saturating_mul(overload_ppm) / 1_000_000
                            && edge.facts["cache_affinity"].as_u64().unwrap_or(0)
                                > edge.facts["estimated_history_hit"]
                                    .as_u64()
                                    .unwrap_or(0)
                                    .saturating_mul(hit_ratio_ppm)
                                    / 1_000_000
                    })
                    .or_else(|| {
                        candidates
                            .iter()
                            .copied()
                            .min_by_key(|(_, edge)| edge.facts["load"].as_u64().unwrap_or(u64::MAX))
                    })
            } else {
                candidates
                    .iter()
                    .copied()
                    .min_by_key(|(_, edge)| edge.facts["load"].as_u64().unwrap_or(u64::MAX))
            };
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }
}

pub struct ThunderAgent;

impl Policy for ThunderAgent {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            if candidate.facts["tool_failed"].as_bool() == Some(true) {
                host.cancel_request(
                    candidate.request.request_id.as_str(),
                    &format!("thunder-cancel-{index}"),
                    Some("tool resource failed"),
                )?;
                continue;
            }
            if let Some(target) = candidate.facts["migrate_target"].as_str() {
                host.rebalance_request(
                    candidate.request.request_id.as_str(),
                    target,
                    &format!("thunder-migrate-{index}"),
                )?;
            }
            if candidate.facts["tool_ready"].as_bool().unwrap_or(true) {
                order.push(index);
            }
        }
        Ok(select_singletons(ctx, order))
    }

    fn cache(ctx: &CacheContext, _state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        let admissions = ctx
            .prospective
            .iter()
            .map(|object| {
                if object.facts["program_live"].as_bool().unwrap_or(false) {
                    CacheAdmission::Cache
                } else {
                    CacheAdmission::Bypass
                }
            })
            .collect::<Vec<_>>();
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                ctx.resident
                    .iter()
                    .enumerate()
                    .filter(|(_, resident)| {
                        resident.reclaimable
                            && !resident.object.facts["program_live"]
                                .as_bool()
                                .unwrap_or(false)
                    })
                    .map(|(index, _)| index as u32),
            ),
            admissions,
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        count_feedback(state, "thunderagent_feedback_records", ctx.records.len());
        Ok(())
    }
}

pub struct Pythia;

impl Policy for Pythia {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Ok(RoutePlan {
            decisions: min_edge_by(ctx, |edge| {
                edge.facts["lookahead_cost"].as_u64().unwrap_or(u64::MAX)
            }),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            (
                ctx.runnable[index].facts["workflow_rank"]
                    .as_u64()
                    .unwrap_or(u64::MAX),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn cache(ctx: &CacheContext, _state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.reclaimable)
            .map(|(index, resident)| {
                (
                    Reverse(
                        resident.object.facts["next_use_step"]
                            .as_u64()
                            .unwrap_or(u64::MAX),
                    ),
                    index as u32,
                )
            })
            .collect::<Vec<_>>();
        reclaim.sort_by_key(|entry| *entry);
        for object in &ctx.prospective {
            if object.facts["prefetch"].as_bool() == Some(true) {
                host.prefetch_cache(
                    object.object_id.as_str(),
                    None,
                    &format!("pythia-{}", object.object_id.as_str()),
                )?;
            }
        }
        let admissions = vec![CacheAdmission::Cache; ctx.prospective.len()];
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                reclaim.into_iter().map(|(_, index)| index),
            ),
            admissions,
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        count_feedback(state, "pythia_feedback_records", ctx.records.len());
        Ok(())
    }
}

pub struct GoodServe;

impl Policy for GoodServe {
    fn route(ctx: &RouteContext, _state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let mut decisions = Vec::new();
        let mut target_counts = vec![0u32; ctx.targets.len()];
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let candidates = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .collect::<Vec<_>>();
            let deadline = request.facts["deadline_ms"].as_u64().unwrap_or(u64::MAX);
            let predicted = |edge: &plex::RouteEdge| {
                let input_tokens = request.facts["input_tokens"].as_u64().unwrap_or(0);
                let cached_tokens = edge.facts["cached_tokens"].as_u64().unwrap_or(0);
                edge.facts["queue_ms"]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(
                        edge.facts["prefill_ms_per_token"]
                            .as_u64()
                            .unwrap_or(0)
                            .saturating_mul(input_tokens.saturating_sub(cached_tokens)),
                    )
                    .saturating_add(
                        edge.facts["decode_ms_per_token"]
                            .as_u64()
                            .unwrap_or(0)
                            .saturating_mul(
                                request.facts["predicted_output_tokens"]
                                    .as_u64()
                                    .unwrap_or(0),
                            ),
                    )
            };
            let selected = candidates
                .iter()
                .copied()
                .filter(|(_, edge)| predicted(edge) <= deadline)
                .min_by_key(|(_, edge)| {
                    (
                        edge.facts["capability_rank"].as_u64().unwrap_or(u64::MAX),
                        edge.facts["cost"].as_u64().unwrap_or(u64::MAX),
                    )
                })
                .or_else(|| {
                    candidates.iter().copied().max_by_key(|(_, edge)| {
                        (
                            edge.facts["capability_rank"].as_u64().unwrap_or(0),
                            Reverse(predicted(edge)),
                        )
                    })
                });
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                if request.facts["risk_ppm"].as_u64().unwrap_or(0)
                    > request.facts["migration_threshold_ppm"]
                        .as_u64()
                        .unwrap_or(u64::MAX)
                {
                    host.rebalance_request(
                        request.request.request_id.as_str(),
                        ctx.targets[edge.target_index as usize].target_id.as_str(),
                        &format!("goodserve-{request_index}"),
                    )?;
                }
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        count_feedback(state, "goodserve_feedback_records", ctx.records.len());
        Ok(())
    }
}

pub struct ConServe;

impl Policy for ConServe {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut target_counts = vec![0u32; ctx.targets.len()];
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let followup = request.request.generation_id > 0;
            let selected = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .filter(|(_, edge)| {
                    let target = &ctx.targets[edge.target_index as usize];
                    if followup {
                        request.facts["bound_target_id"]
                            .as_str()
                            .is_none_or(|bound| target.target_id.as_str() == bound)
                            && !target.facts["prefiller"].as_bool().unwrap_or(false)
                    } else {
                        target.facts["prefiller"].as_bool().unwrap_or(false)
                    }
                })
                .min_by_key(|(_, edge)| {
                    if followup {
                        edge.facts["active_kv_bytes"].as_u64().unwrap_or(u64::MAX)
                    } else {
                        0
                    }
                });
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }
}

pub struct Parrot;

impl Policy for Parrot {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Ok(RoutePlan {
            decisions: min_edge_by(ctx, |edge| {
                edge.facts["dependency_distance"]
                    .as_u64()
                    .unwrap_or(u64::MAX)
            }),
        })
    }

    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let order = ctx
            .runnable
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.facts["dependency_ready"]
                    .as_bool()
                    .unwrap_or(false)
            })
            .map(|(index, _)| index)
            .collect();
        Ok(select_singletons(ctx, order))
    }
}

pub struct Saga;

impl Policy for Saga {
    fn route(ctx: &RouteContext, _state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let mut decisions = Vec::new();
        let mut target_counts = vec![0u32; ctx.targets.len()];
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let selected = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .max_by_key(|(_, edge)| {
                    (
                        edge.facts["cache_locality"].as_u64().unwrap_or(0),
                        Reverse(edge.facts["load"].as_u64().unwrap_or(u64::MAX)),
                    )
                });
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                if request.facts["steal"].as_bool() == Some(true) {
                    host.rebalance_request(
                        request.request.request_id.as_str(),
                        ctx.targets[edge.target_index as usize].target_id.as_str(),
                        &format!("saga-{request_index}"),
                    )?;
                }
                decisions.push(RouteDecision::Assign(index as u32));
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
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let group = candidate
                .request
                .group_id
                .as_ref()
                .and_then(|group_id| state.group(group_id.as_str()).ok());
            candidate.facts["group_service"]
                .as_u64()
                .or_else(|| group.and_then(|group| group.facts()["service"].as_u64()))
                .or_else(|| group.and_then(|group| group.scratch["service"].as_u64()))
                .unwrap_or(0)
        });
        Ok(select_singletons(ctx, order))
    }

    fn cache(ctx: &CacheContext, _state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        let admissions = ctx
            .prospective
            .iter()
            .map(|object| {
                if object.facts["workflow_ttl_ms"].as_u64().unwrap_or(0) > 0 {
                    CacheAdmission::Cache
                } else {
                    CacheAdmission::Bypass
                }
            })
            .collect::<Vec<_>>();
        Ok(CachePlan {
            reclaim: reclaim_prefix(
                ctx,
                &admissions,
                ctx.resident
                    .iter()
                    .enumerate()
                    .filter(|(_, resident)| {
                        resident.reclaimable
                            && resident.object.facts["workflow_ttl_ms"]
                                .as_u64()
                                .unwrap_or(0)
                                == 0
                    })
                    .map(|(index, _)| index as u32),
            ),
            admissions,
        })
    }
}

pub struct RouteBalance;

impl Policy for RouteBalance {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut request_order = (0..ctx.requests.len()).collect::<Vec<_>>();
        request_order.sort_by_key(|&index| {
            Reverse(
                ctx.requests[index].facts["predicted_output_tokens"]
                    .as_u64()
                    .unwrap_or(0),
            )
        });
        let mut decisions = vec![RouteDecision::Defer; ctx.requests.len()];
        let mut target_counts = vec![0u32; ctx.targets.len()];
        let mut target_load = ctx
            .targets
            .iter()
            .map(|target| target.facts["queued_tokens"].as_u64().unwrap_or(0))
            .collect::<Vec<_>>();
        for request_index in request_order {
            let request = &ctx.requests[request_index];
            let output_tokens = request.facts["predicted_output_tokens"]
                .as_u64()
                .unwrap_or(0);
            let cost_budget = request.facts["cost_budget"].as_u64().unwrap_or(u64::MAX);
            let candidates = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                        && edge.facts["cost"].as_u64().unwrap_or(u64::MAX) <= cost_budget
                })
                .map(|(edge_index, edge)| {
                    let target_index = edge.target_index as usize;
                    let latency = edge.facts["latency_ms"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_add(
                            edge.facts["decode_ms_per_token"]
                                .as_u64()
                                .unwrap_or(0)
                                .saturating_mul(target_load[target_index]),
                        );
                    (
                        edge_index,
                        edge,
                        edge.facts["quality_ppm"].as_u64().unwrap_or(0),
                        edge.facts["cost"].as_u64().unwrap_or(0),
                        latency,
                    )
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                continue;
            }
            let quality_min = candidates.iter().map(|entry| entry.2).min().unwrap_or(0);
            let quality_max = candidates.iter().map(|entry| entry.2).max().unwrap_or(0);
            let cost_min = candidates.iter().map(|entry| entry.3).min().unwrap_or(0);
            let cost_max = candidates.iter().map(|entry| entry.3).max().unwrap_or(0);
            let latency_min = candidates.iter().map(|entry| entry.4).min().unwrap_or(0);
            let latency_max = candidates.iter().map(|entry| entry.4).max().unwrap_or(0);
            let quality_weight = request.facts["quality_weight_ppm"]
                .as_u64()
                .unwrap_or(333_334);
            let cost_weight = request.facts["cost_weight_ppm"].as_u64().unwrap_or(333_333);
            let latency_weight = request.facts["latency_weight_ppm"]
                .as_u64()
                .unwrap_or(333_333);
            let selected = candidates.into_iter().max_by_key(|entry| {
                quality_weight
                    .saturating_mul(normalize(entry.2, quality_min, quality_max, true))
                    .saturating_add(
                        cost_weight.saturating_mul(normalize(entry.3, cost_min, cost_max, false)),
                    )
                    .saturating_add(latency_weight.saturating_mul(normalize(
                        entry.4,
                        latency_min,
                        latency_max,
                        false,
                    )))
            });
            if let Some((edge_index, edge, _, _, _)) = selected {
                let target_index = edge.target_index as usize;
                target_counts[target_index] += 1;
                target_load[target_index] = target_load[target_index].saturating_add(output_tokens);
                decisions[request_index] = RouteDecision::Assign(edge_index as u32);
            }
        }
        Ok(RoutePlan { decisions })
    }
}

fn select_singletons(ctx: &ScheduleContext, order: Vec<usize>) -> SchedulePlan {
    let mut remaining_selections = ctx.capacity.max_selections;
    let mut remaining_requests = ctx.capacity.max_requests;
    let mut remaining_tokens = ctx.capacity.max_total_tokens;
    let mut selections = Vec::new();
    for index in order {
        if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0 {
            break;
        }
        let budget = u64::from(ctx.runnable[index].max_token_budget).min(remaining_tokens) as u32;
        selections.push(ScheduleSelection {
            requests: vec![index as u32],
            token_budgets: vec![budget],
        });
        remaining_selections -= 1;
        remaining_requests -= 1;
        remaining_tokens -= u64::from(budget);
    }
    SchedulePlan { selections }
}

fn value(object: &plex::CacheObject) -> u64 {
    let reuse = object.facts["reuse_probability_ppm"].as_u64().unwrap_or(0);
    let flops = object.facts["recompute_flops"].as_u64().unwrap_or(0);
    reuse.saturating_mul(flops) / object.size_bytes.max(1)
}

fn count_feedback(state: &mut State, key: &str, records: usize) {
    state.shared[key] = json!(
        state.shared[key]
            .as_u64()
            .unwrap_or(0)
            .saturating_add(records as u64)
    );
}

fn reclaim_prefix(
    ctx: &CacheContext,
    admissions: &[CacheAdmission],
    ordered: impl IntoIterator<Item = u32>,
) -> Vec<u32> {
    let used = ctx
        .resident
        .iter()
        .fold(ctx.capacity.fixed_bytes, |total, resident| {
            total.saturating_add(resident.object.size_bytes)
        })
        .saturating_add(
            ctx.prospective
                .iter()
                .zip(admissions)
                .filter(|(_, admission)| **admission == CacheAdmission::Cache)
                .fold(0u64, |total, (object, _)| {
                    total.saturating_add(object.size_bytes)
                }),
        );
    let required = used.saturating_sub(ctx.capacity.max_bytes);
    let mut freed = 0u64;
    let mut reclaim = Vec::new();
    for index in ordered {
        if freed >= required {
            break;
        }
        let Some(resident) = ctx.resident.get(index as usize) else {
            continue;
        };
        if !resident.reclaimable {
            continue;
        }
        freed = freed.saturating_add(resident.object.size_bytes);
        reclaim.push(index);
    }
    reclaim
}

fn min_edge_by(ctx: &RouteContext, metric: impl Fn(&plex::RouteEdge) -> u64) -> Vec<RouteDecision> {
    let mut target_counts = vec![0u32; ctx.targets.len()];
    let mut decisions = Vec::with_capacity(ctx.requests.len());
    for request_index in 0..ctx.requests.len() {
        let selected = ctx
            .feasible_edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                edge.request_index as usize == request_index
                    && target_counts[edge.target_index as usize]
                        < ctx.targets[edge.target_index as usize].max_assignments
            })
            .min_by_key(|(_, edge)| metric(edge));
        if let Some((index, edge)) = selected {
            target_counts[edge.target_index as usize] += 1;
            decisions.push(RouteDecision::Assign(index as u32));
        } else {
            decisions.push(RouteDecision::Defer);
        }
    }
    decisions
}

fn normalize(value: u64, minimum: u64, maximum: u64, higher_is_better: bool) -> u64 {
    if maximum == minimum {
        return 1_000_000;
    }
    let numerator = if higher_is_better {
        value.saturating_sub(minimum)
    } else {
        maximum.saturating_sub(value)
    };
    numerator.saturating_mul(1_000_000) / maximum.saturating_sub(minimum)
}
