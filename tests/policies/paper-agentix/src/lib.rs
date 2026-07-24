//! Agentix PLAS and ATLAS program-aware scheduling.

use plex::serde_json::json;
use plex::{
    FeedbackContext, FeedbackSubject, Host, OutcomeKind, Policy, ScheduleContext, SchedulePlan,
    ScheduleSelection, State,
};

struct Agentix;

impl Policy for Agentix {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut ranked = Vec::with_capacity(ctx.runnable.len());
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            let mode = candidate.facts["agentix_mode"].as_str().unwrap_or("plas");
            let request_id = candidate.request.request_id.as_str();
            let group_id = candidate.request.group_id.as_ref();
            let (program_service, program_wait) = group_id
                .and_then(|id| state.group(id.as_str()).ok())
                .map(|group| {
                    (
                        group.scratch["agentix_service_us"]
                            .as_u64()
                            .or_else(|| group.scratch["observed_service_us"].as_u64())
                            .unwrap_or(0),
                        group.scratch["agentix_wait_us"].as_u64().unwrap_or(0),
                    )
                })
                .unwrap_or((0, 0));
            let initialized = state.request(request_id)?.scratch["agentix_initialized"]
                .as_bool()
                .unwrap_or(false);
            if !initialized {
                let inherited_service = if mode == "atlas" {
                    candidate.facts["critical_path_service_us"]
                        .as_u64()
                        .unwrap_or(program_service)
                } else {
                    program_service
                };
                let level = queue_level(&candidate.facts, inherited_service);
                let request = state.request_mut(request_id)?;
                request.scratch["agentix_initialized"] = json!(true);
                request.scratch["agentix_mode"] = json!(mode);
                request.scratch["agentix_inherited_service_us"] = json!(inherited_service);
                request.scratch["agentix_queue_level"] = json!(level);
                request.scratch["agentix_remaining_quantum_us"] =
                    json!(queue_quantum(&candidate.facts, level));
                request.scratch["agentix_model_time_us"] = json!(0);
                request.scratch["agentix_wait_reset_us"] = json!(0);
                request.scratch["agentix_promoted"] = json!(false);
            }

            let call_wait_us = candidate.facts["call_wait_us"]
                .as_u64()
                .or_else(|| {
                    candidate.facts["call_wait_ms"]
                        .as_u64()
                        .map(|value| value.saturating_mul(1000))
                })
                .unwrap_or_else(|| {
                    candidate.facts["waiting_ms"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_mul(1000)
                });
            let program_wait_us = candidate.facts["program_wait_us"]
                .as_u64()
                .or_else(|| {
                    candidate.facts["program_wait_ms"]
                        .as_u64()
                        .map(|value| value.saturating_mul(1000))
                })
                .unwrap_or(program_wait);
            let request = state.request(request_id)?;
            let wait_reset = request.scratch["agentix_wait_reset_us"]
                .as_u64()
                .unwrap_or(0);
            let effective_call_wait = call_wait_us.saturating_sub(wait_reset);
            let model_time = request.scratch["agentix_model_time_us"]
                .as_u64()
                .unwrap_or(0);
            let starvation_ratio_ppm = candidate.facts["starvation_ratio_ppm"]
                .as_u64()
                .unwrap_or(4_000_000);
            let total_wait = program_wait_us.saturating_add(effective_call_wait);
            let total_service = program_service.saturating_add(model_time);
            let already_promoted = request.scratch["agentix_promoted"]
                .as_bool()
                .unwrap_or(false);
            let starved = !already_promoted
                && total_wait > 0
                && (total_service == 0
                    || u128::from(total_wait).saturating_mul(1_000_000)
                        >= u128::from(total_service)
                            .saturating_mul(u128::from(starvation_ratio_ppm)));
            if starved {
                let request = state.request_mut(request_id)?;
                request.scratch["agentix_queue_level"] = json!(0);
                request.scratch["agentix_remaining_quantum_us"] =
                    json!(queue_quantum(&candidate.facts, 0));
                request.scratch["agentix_model_time_us"] = json!(0);
                request.scratch["agentix_wait_reset_us"] = json!(call_wait_us);
                request.scratch["agentix_promoted"] = json!(true);
                request.scratch["agentix_promotions"] = json!(
                    request.scratch["agentix_promotions"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_add(1)
                );
            }
            let request = state.request_mut(request_id)?;
            request.scratch["agentix_last_call_wait_us"] = json!(call_wait_us);
            let level = request.scratch["agentix_queue_level"].as_u64().unwrap_or(0);
            ranked.push((
                level,
                candidate.facts["call_arrival"]
                    .as_u64()
                    .unwrap_or(index as u64),
                index,
            ));
        }
        ranked.sort_unstable();

        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        for (_, _, index) in ranked {
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

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let FeedbackSubject::Request(request_id) = &record.subject else {
                continue;
            };
            let request_id = request_id.as_str();
            if record.outcome == OutcomeKind::Progress {
                let service = record.facts["service_us"].as_u64().unwrap_or(0);
                let request = state.request_mut(request_id)?;
                advance_queue(request, service, &record.facts);
                if let Some(call_wait_us) = feedback_wait_us(&record.facts) {
                    request.scratch["agentix_last_call_wait_us"] = json!(call_wait_us);
                }
                continue;
            }
            if record.outcome != OutcomeKind::Completed {
                continue;
            }

            let (group_id, mode, inherited_service, model_time, call_wait, wait_reset) = {
                let request = state.request_mut(request_id)?;
                let extra_service = record.facts["service_us"].as_u64().unwrap_or(0);
                if extra_service > 0 {
                    advance_queue(request, extra_service, &record.facts);
                }
                (
                    request.reference().group_id.clone(),
                    request.scratch["agentix_mode"]
                        .as_str()
                        .unwrap_or("plas")
                        .to_owned(),
                    request.scratch["agentix_inherited_service_us"]
                        .as_u64()
                        .unwrap_or(0),
                    request.scratch["agentix_model_time_us"]
                        .as_u64()
                        .unwrap_or(0),
                    feedback_wait_us(&record.facts)
                        .or_else(|| request.scratch["agentix_last_call_wait_us"].as_u64())
                        .unwrap_or(0),
                    request.scratch["agentix_wait_reset_us"]
                        .as_u64()
                        .unwrap_or(0),
                )
            };
            if let Some(group_id) = group_id {
                let group = state.group_mut(group_id.as_str())?;
                let completed_service = inherited_service.saturating_add(model_time);
                group.scratch["agentix_service_us"] = json!(
                    group.scratch["agentix_service_us"]
                        .as_u64()
                        .or_else(|| group.scratch["observed_service_us"].as_u64())
                        .unwrap_or(0)
                        .max(completed_service)
                );
                let completed_wait = call_wait.saturating_sub(wait_reset);
                let prior_wait = group.scratch["agentix_wait_us"].as_u64().unwrap_or(0);
                group.scratch["agentix_wait_us"] = json!(if mode == "atlas" {
                    prior_wait.max(completed_wait)
                } else {
                    prior_wait.saturating_add(completed_wait)
                });
                group.scratch["agentix_completed_calls"] = json!(
                    group.scratch["agentix_completed_calls"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_add(1)
                );
            }
        }
        Ok(())
    }
}

fn queue_level(facts: &plex::Document, service: u64) -> u64 {
    if let Some(bounds) = facts["queue_bounds_us"].as_array() {
        return bounds
            .iter()
            .filter_map(|bound| bound.as_u64())
            .take_while(|bound| service >= *bound)
            .count() as u64;
    }
    let levels = queue_levels(facts);
    let base = facts["base_quantum_us"].as_u64().unwrap_or(1000).max(1);
    let mut level = 0u64;
    let mut threshold = base;
    while level + 1 < levels && service >= threshold {
        level += 1;
        threshold = threshold.saturating_mul(2);
    }
    level
}

fn queue_levels(facts: &plex::Document) -> u64 {
    facts["queue_quanta_us"]
        .as_array()
        .map(|values| values.len().max(1) as u64)
        .or_else(|| {
            facts["queue_bounds_us"]
                .as_array()
                .map(|values| values.len().saturating_add(1) as u64)
        })
        .unwrap_or_else(|| facts["queue_levels"].as_u64().unwrap_or(8).clamp(1, 63))
}

fn queue_quantum(facts: &plex::Document, level: u64) -> u64 {
    if let Some(quanta) = facts["queue_quanta_us"].as_array() {
        return quanta
            .get(level as usize)
            .and_then(|value| value.as_u64())
            .or_else(|| quanta.last().and_then(|value| value.as_u64()))
            .unwrap_or(1)
            .max(1);
    }
    facts["base_quantum_us"]
        .as_u64()
        .unwrap_or(1000)
        .max(1)
        .checked_shl(level.min(63) as u32)
        .unwrap_or(u64::MAX)
        .max(1)
}

fn advance_queue(request: &mut plex::Request, mut service: u64, facts: &plex::Document) {
    let delivered_service = service;
    let levels = queue_levels(facts);
    let mut level = request.scratch["agentix_queue_level"]
        .as_u64()
        .unwrap_or(0)
        .min(levels - 1);
    let mut remaining = request.scratch["agentix_remaining_quantum_us"]
        .as_u64()
        .unwrap_or_else(|| queue_quantum(facts, level))
        .max(1);
    while service >= remaining {
        service -= remaining;
        if level + 1 < levels {
            level += 1;
        }
        remaining = queue_quantum(facts, level);
        if service == 0 {
            break;
        }
    }
    remaining = remaining.saturating_sub(service);
    request.scratch["agentix_queue_level"] = json!(level);
    request.scratch["agentix_remaining_quantum_us"] = json!(remaining);
    request.scratch["agentix_model_time_us"] = json!(
        request.scratch["agentix_model_time_us"]
            .as_u64()
            .unwrap_or(0)
            .saturating_add(delivered_service)
    );
    request.scratch["agentix_promoted"] = json!(false);
}

fn feedback_wait_us(facts: &plex::Document) -> Option<u64> {
    facts["call_wait_us"].as_u64().or_else(|| {
        facts["call_wait_ms"]
            .as_u64()
            .map(|value| value.saturating_mul(1000))
    })
}

plex::export_policy!(Agentix);
