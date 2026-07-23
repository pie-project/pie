#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::collections::BTreeSet;

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
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let client = ctx.runnable[index].facts["client_id"]
                .as_str()
                .unwrap_or("default");
            (state.shared["vtc"][client].as_u64().unwrap_or(0), index)
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
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

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
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
        let victim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| {
                resident.reclaimable && resident.object.facts["leaf"].as_bool().unwrap_or(false)
            })
            .min_by_key(|(index, resident)| {
                let frequency = resident.object.facts["frequency"].as_u64().unwrap_or(1);
                let cost = resident.object.facts["recompute_cost"]
                    .as_u64()
                    .unwrap_or(0);
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

pub struct Dlpm;

impl Policy for Dlpm {
    fn route(ctx: &RouteContext, state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let decisions = ctx
            .requests
            .iter()
            .enumerate()
            .map(|(request_index, request)| {
                let client = request.facts["client_id"].as_str().unwrap_or("default");
                let deficit = state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0);
                ctx.feasible_edges
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| edge.request_index as usize == request_index)
                    .max_by_key(|(_, edge)| {
                        if deficit >= 0 {
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
                    })
                    .map_or(RouteDecision::Defer, |(index, _)| {
                        RouteDecision::Assign(index as u32)
                    })
            })
            .collect();
        Ok(RoutePlan { decisions })
    }

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
                Reverse(state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0)),
                Reverse(
                    ctx.runnable[index].facts["cached_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                ),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let client = record.facts["client_id"].as_str().unwrap_or("default");
            let quantum = record.facts["quantum"].as_i64().unwrap_or(0);
            let service = record.facts["service_tokens"].as_i64().unwrap_or(0);
            state.shared["dlpm_deficit"][client] = json!(
                state.shared["dlpm_deficit"][client]
                    .as_i64()
                    .unwrap_or(0)
                    .saturating_add(quantum)
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
        Ok(CachePlan {
            admissions: ctx
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
                .collect(),
            reclaim: reclaim.into_iter().map(|(_, index)| index).collect(),
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
        Ok(CachePlan {
            admissions: vec![CacheAdmission::Cache; ctx.prospective.len()],
            reclaim: reclaim.into_iter().map(|(_, index)| index).collect(),
        })
    }
}

pub struct Qlm;

impl Policy for Qlm {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        Ok(AdmitPlan {
            decisions: ctx
                .candidates
                .iter()
                .map(|candidate| {
                    if candidate.facts["estimated_wait_ms"].as_u64().unwrap_or(0)
                        <= candidate.facts["slo_ms"].as_u64().unwrap_or(u64::MAX)
                    {
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
            let group = ctx.runnable[index]
                .request
                .group_id
                .as_ref()
                .and_then(|group_id| state.group(group_id.as_str()).ok());
            (
                group
                    .and_then(|group| group.scratch["virtual_wait"].as_u64())
                    .unwrap_or(0),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }
}

pub struct SlosServe;

impl Policy for SlosServe {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        Ok(AdmitPlan {
            decisions: ctx
                .candidates
                .iter()
                .map(|candidate| {
                    if candidate.facts["predicted_total_ms"]
                        .as_u64()
                        .unwrap_or(u64::MAX)
                        <= candidate.facts["slo_ms"].as_u64().unwrap_or(0)
                    {
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
            let completed = ctx.runnable[index]
                .request
                .group_id
                .as_ref()
                .and_then(|group_id| state.group(group_id.as_str()).ok())
                .and_then(|group| group.scratch["completed_branches"].as_u64())
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
        Ok(CachePlan {
            admissions: ctx
                .prospective
                .iter()
                .map(|object| {
                    if object.facts["adapter_hot"].as_bool().unwrap_or(false) {
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
                .filter(|(_, resident)| resident.reclaimable)
                .map(|(index, _)| index as u32)
                .collect(),
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
        Ok(CachePlan {
            admissions,
            reclaim: ctx
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
                .map(|(index, _)| index as u32)
                .collect(),
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
                .saturating_add(candidate.facts["downstream_p95_ms"].as_u64().unwrap_or(0));
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
}

pub struct BranchRegulation;

impl Policy for BranchRegulation {
    fn admit(ctx: &AdmitContext, _state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let mut accepted_by_group = std::collections::BTreeMap::<String, u64>::new();
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
                    if *accepted < limit
                        && candidate.facts["batch_interference"].as_u64().unwrap_or(0)
                            <= candidate.facts["interference_limit"]
                                .as_u64()
                                .unwrap_or(u64::MAX)
                    {
                        *accepted += 1;
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
        host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = Vec::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            if candidate.facts["excess_branch"].as_bool() == Some(true) {
                host.cancel_request(
                    candidate.request.request_id.as_str(),
                    &format!("branch-regulation-{index}"),
                    Some("branch concurrency limit"),
                )?;
            } else {
                order.push(index);
            }
        }
        Ok(select_singletons(ctx, order))
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

fn min_edge_by(ctx: &RouteContext, metric: impl Fn(&plex::RouteEdge) -> u64) -> Vec<RouteDecision> {
    ctx.requests
        .iter()
        .enumerate()
        .map(|(request_index, _)| {
            ctx.feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| edge.request_index as usize == request_index)
                .min_by_key(|(_, edge)| metric(edge))
                .map_or(RouteDecision::Defer, |(index, _)| {
                    RouteDecision::Assign(index as u32)
                })
        })
        .collect()
}
