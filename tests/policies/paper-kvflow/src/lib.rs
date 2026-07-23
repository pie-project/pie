//! KVFlow cache-loading-aware scheduling and varying-suffix-first reclaim.

use plex::{
    CacheAdmission, CacheContext, CachePlan, Host, Policy, ScheduleContext, SchedulePlan,
    ScheduleSelection, State,
};

struct KvFlow;

impl Policy for KvFlow {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            (
                !ctx.runnable[index].facts["cache_ready"]
                    .as_bool()
                    .unwrap_or(false),
                index,
            )
        });
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        for index in order {
            if remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let budget =
                u64::from(ctx.runnable[index].max_token_budget).min(remaining_tokens) as u32;
            selections.push(ScheduleSelection {
                requests: vec![index as u32],
                token_budgets: vec![budget],
            });
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        Ok(SchedulePlan { selections })
    }

    fn cache(ctx: &CacheContext, _state: &mut State, host: &Host) -> plex::Result<CachePlan> {
        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.reclaimable)
            .map(|(index, resident)| {
                (
                    resident.object.facts["fixed_prefix"]
                        .as_bool()
                        .unwrap_or(false),
                    std::cmp::Reverse(
                        resident.object.facts["steps_to_execution"]
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
                    &format!("kvflow-{}", object.object_id.as_str()),
                )?;
            }
        }
        Ok(CachePlan {
            admissions: vec![CacheAdmission::Cache; ctx.prospective.len()],
            reclaim: reclaim.into_iter().map(|(_, _, index)| index).collect(),
        })
    }
}

plex::export_policy!(KvFlow);
