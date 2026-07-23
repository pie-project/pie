//! Continuum TTL-aware group-FCFS scheduling and cache retention.

use plex::serde_json::json;
use plex::{
    CacheAdmission, CacheContext, CachePlan, FeedbackContext, FeedbackSubject, Host, Policy,
    ScheduleContext, SchedulePlan, ScheduleSelection, State,
};

struct Continuum;

impl Policy for Continuum {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let pinned = state
                .request(candidate.request.request_id.as_str())
                .ok()
                .and_then(|request| request.scratch["ttl_active"].as_bool())
                .unwrap_or(false);
            (
                !candidate.facts["preempted"].as_bool().unwrap_or(false),
                !pinned,
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
        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.reclaimable)
            .map(|(index, resident)| {
                let pinned = resident.object.beneficiaries.iter().any(|beneficiary| {
                    let plex::Beneficiary::Request(request_id) = beneficiary else {
                        return false;
                    };
                    state
                        .request(request_id.as_str())
                        .ok()
                        .and_then(|request| request.scratch["ttl_active"].as_bool())
                        .unwrap_or(false)
                });
                (
                    pinned,
                    resident.object.facts["reload_cost"].as_u64().unwrap_or(0),
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
                reclaim.into_iter().map(|(_, _, index)| index),
            ),
            admissions,
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let FeedbackSubject::Request(request_id) = &record.subject else {
                continue;
            };
            if state.request(request_id.as_str()).is_err() {
                continue;
            }
            let request = state.request_mut(request_id.as_str())?;
            if let Some(ttl_ms) = record.facts["ttl_ms"].as_u64() {
                request.scratch["ttl_active"] = json!(ttl_ms != 0);
            }
            if record.facts["ttl_expired"].as_bool() == Some(true) {
                request.scratch["ttl_active"] = json!(false);
            }
        }
        Ok(())
    }
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
        let resident = &ctx.resident[index as usize];
        freed = freed.saturating_add(resident.object.size_bytes);
        reclaim.push(index);
    }
    reclaim
}

plex::export_policy!(Continuum);
