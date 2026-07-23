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
        Ok(CachePlan {
            admissions: vec![CacheAdmission::Cache; ctx.prospective.len()],
            reclaim: reclaim
                .into_iter()
                .filter(|(pinned, _, _)| !pinned)
                .map(|(_, _, index)| index)
                .collect(),
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

plex::export_policy!(Continuum);
