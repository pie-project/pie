#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::collections::BTreeSet;

use plex::serde_json::json;
use plex::{
    AdmitContext, AdmitPlan, AdmissionDecision, CacheAdmission, CacheContext, CachePlan,
    FeedbackContext, Host, Policy, RouteContext, RouteDecision, RoutePlan, ScheduleContext,
    SchedulePlan, ScheduleSelection, State,
};

pub struct Vtc;

impl Policy for Vtc {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let client = ctx.runnable[index].facts["client_id"]
                .as_str()
                .unwrap_or("default");
            (
                state.shared["vtc"][client].as_u64().unwrap_or(0),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(
        ctx: &FeedbackContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<()> {
        for record in &ctx.records {
            let client = record.facts["client_id"].as_str().unwrap_or("default");
            let charge = record.facts["input_tokens"].as_u64().unwrap_or(0)
                + record.facts["output_tokens"].as_u64().unwrap_or(0);
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
                        !target.facts["hotspot"].as_bool().unwrap_or(false)
                    })
                    .min_by_key(|(_, edge)| {
                        edge.facts["new_prefill_tokens"]
                            .as_u64()
                            .unwrap_or(u64::MAX)
                            .saturating_mul(
                                edge.facts["current_batch_size"]
                                    .as_u64()
                                    .unwrap_or(u64::MAX),
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
        let mut order = (0..ctx.candidates.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.candidates[index];
            let client = candidate.facts["client_id"].as_str().unwrap_or("default");
            let weight = candidate.facts["weight"].as_u64().unwrap_or(1).max(1);
            let service = state.shared["fairserve"][client].as_u64().unwrap_or(0) / weight;
            let interference = candidate.facts["interference_cost"].as_u64().unwrap_or(0);
            (service.saturating_add(interference), index)
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
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let client = candidate.facts["client_id"].as_str().unwrap_or("default");
            let weight = candidate.facts["weight"].as_u64().unwrap_or(1).max(1);
            (
                state.shared["fairserve"][client].as_u64().unwrap_or(0) / weight,
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(
        ctx: &FeedbackContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<()> {
        for record in &ctx.records {
            let client = record.facts["client_id"].as_str().unwrap_or("default");
            let service = record.facts["service_tokens"].as_u64().unwrap_or(0);
            state.shared["fairserve"][client] = json!(
                state.shared["fairserve"][client]
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

    fn feedback(
        ctx: &FeedbackContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<()> {
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
        let victim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| {
                resident.reclaimable
                    && resident.object.facts["leaf"].as_bool().unwrap_or(false)
            })
            .min_by_key(|(index, resident)| {
                let frequency = resident.object.facts["frequency"].as_u64().unwrap_or(1);
                let cost = resident.object.facts["recompute_cost"].as_u64().unwrap_or(0);
                let age = resident.object.facts["age"].as_u64().unwrap_or(0);
                (
                    age.saturating_add(cost.saturating_mul(frequency))
                        / resident.object.size_bytes.max(1),
                    *index,
                )
            })
            .map(|(index, _)| index as u32);
        Ok(CachePlan {
            admissions: vec![CacheAdmission::Cache; ctx.prospective.len()],
            reclaim: victim.into_iter().collect(),
        })
    }
}

fn select_singletons(ctx: &ScheduleContext, order: Vec<usize>) -> SchedulePlan {
    let mut remaining_requests = ctx.capacity.max_requests;
    let mut remaining_tokens = ctx.capacity.max_total_tokens;
    let mut selections = Vec::new();
    for index in order {
        if remaining_requests == 0 || remaining_tokens == 0 {
            break;
        }
        let budget = u64::from(ctx.runnable[index].max_token_budget).min(remaining_tokens) as u32;
        selections.push(ScheduleSelection {
            requests: vec![index as u32],
            token_budgets: vec![budget],
        });
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
