//! Continuum empirical-CDF TTL retention and TTL-aware program scheduling.

use std::cmp::Reverse;
use std::collections::BTreeSet;

use plex::serde_json::json;
use plex::{
    CacheAdmission, CacheContext, CachePlan, FeedbackContext, FeedbackSubject, Host, OutcomeKind,
    Policy, ScheduleContext, SchedulePlan, ScheduleSelection, State,
};

struct Continuum;

impl Policy for Continuum {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let now_ms = ctx.capacity.facts["now_ms"]
            .as_u64()
            .or_else(|| {
                ctx.runnable
                    .iter()
                    .find_map(|candidate| candidate.facts["now_ms"].as_u64())
            })
            .unwrap_or(0);
        let queued_programs = ctx
            .runnable
            .iter()
            .map(|candidate| continuum_program(&candidate.request, &candidate.facts))
            .collect::<BTreeSet<_>>();
        expire_continuum_pins(state, now_ms, &queued_programs);
        if ctx.capacity.facts["pin_break_required"]
            .as_bool()
            .unwrap_or(false)
        {
            break_latest_continuum_pin(state);
        }

        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let program = continuum_program(&candidate.request, &candidate.facts);
            (
                !candidate.facts["preempted"].as_bool().unwrap_or(false),
                !continuum_pin_active(state, &program, now_ms),
                candidate.facts["program_arrival"]
                    .as_u64()
                    .unwrap_or(u64::MAX),
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

    fn cache(ctx: &CacheContext, state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        let now_ms = ctx.capacity.facts["now_ms"].as_u64().unwrap_or(0);
        let queued_programs = ctx.capacity.facts["queued_programs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|program| program.as_str().map(str::to_owned))
            .collect::<BTreeSet<_>>();
        expire_continuum_pins(state, now_ms, &queued_programs);
        let admissions = vec![CacheAdmission::Cache; ctx.prospective.len()];
        let required = cache_required_bytes(ctx, &admissions);
        let mut unpinned = Vec::new();
        let mut pinned = Vec::new();
        for (index, resident) in ctx.resident.iter().enumerate() {
            if !resident.reclaimable {
                continue;
            }
            let program = resident
                .object
                .beneficiaries
                .iter()
                .find_map(|beneficiary| {
                    let plex::Beneficiary::Request(request_id) = beneficiary else {
                        return None;
                    };
                    state
                        .request(request_id.as_str())
                        .ok()
                        .map(|request| continuum_program(request.reference(), request.facts()))
                })
                .unwrap_or_else(|| {
                    resident.object.facts["program_id"]
                        .as_str()
                        .unwrap_or("default")
                        .to_owned()
                });
            let arrival = state.shared["continuum_pins"][&program]["program_arrival"]
                .as_u64()
                .unwrap_or(0);
            if continuum_pin_active(state, &program, now_ms) {
                pinned.push((Reverse(arrival), index as u32));
            } else {
                unpinned.push(index as u32);
            }
        }
        pinned.sort_by_key(|entry| *entry);
        let mut ordered = unpinned;
        if ctx.capacity.facts["pin_break_required"]
            .as_bool()
            .unwrap_or(false)
            || ordered
                .iter()
                .map(|index| ctx.resident[*index as usize].object.size_bytes)
                .sum::<u64>()
                < required
        {
            ordered.extend(pinned.into_iter().map(|(_, index)| index));
        }
        Ok(CachePlan {
            reclaim: reclaim_required(ctx, required, ordered),
            admissions,
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let FeedbackSubject::Request(request_id) = &record.subject else {
                continue;
            };
            let request = state.request(request_id.as_str())?;
            let program = continuum_program(request.reference(), request.facts());
            if let (Some(tool), Some(duration)) = (
                record.facts["tool_id"].as_str(),
                record.facts["tool_duration_ms"].as_u64(),
            ) {
                record_continuum_tool(state, tool, duration);
            }
            if record.facts["program_resumed"].as_bool().unwrap_or(false) {
                let observed = state.shared["continuum_pins"][&program]
                    .as_object()
                    .and_then(|pin| {
                        Some((
                            pin.get("tool_id")?.as_str()?.to_owned(),
                            pin.get("finished_at_ms")?.as_u64()?,
                            record.facts["now_ms"].as_u64()?,
                        ))
                    });
                if let Some((tool, finished_at, now_ms)) = observed {
                    record_continuum_tool(state, &tool, now_ms.saturating_sub(finished_at));
                }
                remove_continuum_pin(state, &program);
            }
            if record.outcome != OutcomeKind::Completed {
                continue;
            }
            if record.facts["program_finished"].as_bool().unwrap_or(false) {
                remove_continuum_pin(state, &program);
                continue;
            }
            let Some(tool) = record.facts["next_tool_id"].as_str() else {
                continue;
            };
            let now_ms = record.facts["now_ms"].as_u64().unwrap_or(0);
            let ttl_ms = continuum_ttl(state, tool, &record.facts);
            if ttl_ms == 0 {
                remove_continuum_pin(state, &program);
                continue;
            }
            state.shared["continuum_pins"][&program] = json!({
                "request_id": request_id.as_str(),
                "tool_id": tool,
                "finished_at_ms": now_ms,
                "expires_at_ms": now_ms.saturating_add(ttl_ms),
                "ttl_ms": ttl_ms,
                "program_arrival": record.facts["program_arrival"].as_u64().unwrap_or(0)
            });
        }
        Ok(())
    }
}

fn continuum_program(request: &plex::RequestRef, facts: &plex::Document) -> String {
    facts["program_id"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            request
                .group_id
                .as_ref()
                .map(|group| group.as_str().to_owned())
        })
        .unwrap_or_else(|| request.request_id.as_str().to_owned())
}

fn continuum_ttl(state: &State, tool: &str, facts: &plex::Document) -> u64 {
    let threshold = facts["history_threshold"].as_u64().unwrap_or(100) as usize;
    let specific = continuum_history(&state.shared["continuum_tool_history"][tool]);
    let global = continuum_history(&state.shared["continuum_global_history"]);
    let samples = if specific.len() > threshold {
        specific
    } else if global.len() > threshold {
        global
    } else {
        return facts["default_ttl_ms"].as_u64().unwrap_or(0);
    };
    let average_wait = facts["average_wait_ms"].as_u64().unwrap_or(0);
    let eta_ppm = facts["memoryfulness_ppm"].as_i64().unwrap_or(1_000_000);
    let reload_ms = facts["prefill_reload_ms"].as_u64().unwrap_or(0);
    let benefit = if eta_ppm >= 0 {
        reload_ms.saturating_add(
            u128::from(average_wait)
                .saturating_mul(eta_ppm as u128)
                .checked_div(1_000_000)
                .unwrap_or(0)
                .min(u128::from(u64::MAX)) as u64,
        )
    } else {
        reload_ms.saturating_sub(
            u128::from(average_wait)
                .saturating_mul(eta_ppm.unsigned_abs() as u128)
                .checked_div(1_000_000)
                .unwrap_or(0)
                .min(u128::from(u64::MAX)) as u64,
        )
    };
    let mut candidates = samples.clone();
    candidates.push(0);
    candidates.sort_unstable();
    candidates.dedup();
    candidates
        .into_iter()
        .max_by_key(|candidate| {
            let finished = samples
                .iter()
                .filter(|sample| **sample <= *candidate)
                .count() as u128;
            let reward = finished
                .saturating_mul(u128::from(benefit))
                .checked_div(samples.len() as u128)
                .unwrap_or(0);
            (
                reward.saturating_sub(u128::from(*candidate)),
                Reverse(*candidate),
            )
        })
        .unwrap_or(0)
}

fn record_continuum_tool(state: &mut State, tool: &str, duration: u64) {
    append_bounded(
        &mut state.shared["continuum_tool_history"][tool],
        duration,
        512,
    );
    append_bounded(
        &mut state.shared["continuum_global_history"],
        duration,
        2048,
    );
}

fn append_bounded(value: &mut plex::Document, item: u64, limit: usize) {
    let mut entries = value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_u64())
        .collect::<Vec<_>>();
    entries.push(item);
    if entries.len() > limit {
        entries.drain(..entries.len() - limit);
    }
    *value = json!(entries);
}

fn continuum_history(value: &plex::Document) -> Vec<u64> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_u64())
        .collect()
}

fn continuum_pin_active(state: &State, program: &str, now_ms: u64) -> bool {
    state.shared["continuum_pins"][program]["expires_at_ms"]
        .as_u64()
        .is_some_and(|expires| now_ms <= expires)
}

fn expire_continuum_pins(state: &mut State, now_ms: u64, queued: &BTreeSet<String>) {
    let expired = state.shared["continuum_pins"]
        .as_object()
        .into_iter()
        .flat_map(|pins| pins.iter())
        .filter(|(program, pin)| {
            pin["expires_at_ms"].as_u64().unwrap_or(0) < now_ms && !queued.contains(*program)
        })
        .map(|(program, _)| program.clone())
        .collect::<Vec<_>>();
    for program in expired {
        remove_continuum_pin(state, &program);
    }
}

fn break_latest_continuum_pin(state: &mut State) {
    let victim = state.shared["continuum_pins"]
        .as_object()
        .into_iter()
        .flat_map(|pins| pins.iter())
        .max_by_key(|(_, pin)| pin["program_arrival"].as_u64().unwrap_or(0))
        .map(|(program, _)| program.clone());
    if let Some(victim) = victim {
        remove_continuum_pin(state, &victim);
    }
}

fn remove_continuum_pin(state: &mut State, program: &str) {
    if let Some(pins) = state.shared["continuum_pins"].as_object_mut() {
        pins.remove(program);
    }
}

fn cache_required_bytes(ctx: &CacheContext, admissions: &[CacheAdmission]) -> u64 {
    ctx.resident
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
        )
        .saturating_sub(ctx.capacity.max_bytes)
}

fn reclaim_required(
    ctx: &CacheContext,
    required: u64,
    ordered: impl IntoIterator<Item = u32>,
) -> Vec<u32> {
    let mut freed = 0u64;
    ordered
        .into_iter()
        .take_while(|index| {
            if freed >= required {
                return false;
            }
            freed = freed.saturating_add(ctx.resident[*index as usize].object.size_bytes);
            true
        })
        .collect()
}

plex::export_policy!(Continuum);
