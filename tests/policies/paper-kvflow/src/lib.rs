//! KVFlow workflow-aware eviction, prefetch, and transfer-state scheduling.

use std::cmp::Reverse;

use plex::serde_json::json;
use plex::{
    CacheAdmission, CacheContext, CachePlan, FeedbackContext, FeedbackSubject, Host, OutcomeKind,
    Policy, ScheduleContext, SchedulePlan, ScheduleSelection, State,
};

struct KvFlow;

impl Policy for KvFlow {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let order = (0..ctx.runnable.len())
            .filter(|&index| kvflow_request_ready(&ctx.runnable[index].facts, state))
            .collect::<Vec<_>>();
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
        for resident in &ctx.resident {
            sync_kvflow_object(state, &resident.object);
        }
        for object in &ctx.prospective {
            sync_kvflow_object(state, object);
        }

        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| {
                let status = kvflow_status(state, resident.object.object_id.as_str());
                resident.reclaimable && status != "loading" && status != "offloading"
            })
            .map(|(index, resident)| {
                (
                    resident.object.facts["fixed_prefix"]
                        .as_bool()
                        .unwrap_or(false),
                    Reverse(kvflow_steps(&resident.object.facts)),
                    index as u32,
                )
            })
            .collect::<Vec<_>>();
        reclaim.sort_by_key(|entry| *entry);

        let prefetch_horizon = ctx.capacity.facts["prefetch_horizon_steps"]
            .as_u64()
            .unwrap_or(1);
        let prefetch_limit = ctx.capacity.facts["max_concurrent_prefetches"]
            .as_u64()
            .unwrap_or(u64::MAX);
        let mut prefetched = 0u64;
        for object in ctx
            .prospective
            .iter()
            .chain(ctx.resident.iter().map(|resident| &resident.object))
        {
            let object_id = object.object_id.as_str();
            let should_prefetch = object.facts["prefetch"].as_bool().unwrap_or(false)
                || (kvflow_status(state, object_id) == "cpu"
                    && kvflow_steps(&object.facts) <= prefetch_horizon);
            if should_prefetch && prefetched < prefetch_limit {
                let action = host.prefetch_cache(
                    object_id,
                    object.facts["target_id"].as_str(),
                    &format!(
                        "kvflow-prefetch-{}-{}",
                        ctx.meta.opportunity_id.as_str(),
                        object_id
                    ),
                )?;
                state.shared["kvflow_actions"][&action.0.to_string()] = json!({
                    "object_id": object_id,
                    "target_state": "gpu"
                });
                state.shared["kvflow_objects"][object_id]["status"] = json!("loading");
                prefetched += 1;
            }
        }
        let admissions = ctx
            .prospective
            .iter()
            .map(|object| {
                if object.facts["admit"].as_bool().unwrap_or(true) {
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
                reclaim.into_iter().map(|(_, _, index)| index),
            ),
            admissions,
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            match &record.subject {
                FeedbackSubject::Action(action_id)
                    if matches!(
                        record.outcome,
                        OutcomeKind::ActionSucceeded | OutcomeKind::ActionFailed
                    ) =>
                {
                    let action_key = action_id.0.to_string();
                    let action = state.shared["kvflow_actions"][&action_key].clone();
                    let Some(object_id) = action["object_id"].as_str() else {
                        continue;
                    };
                    state.shared["kvflow_objects"][object_id]["status"] =
                        json!(if record.outcome == OutcomeKind::ActionSucceeded {
                            action["target_state"].as_str().unwrap_or("gpu")
                        } else {
                            "cpu"
                        });
                    if let Some(actions) = state.shared["kvflow_actions"].as_object_mut() {
                        actions.remove(&action_key);
                    }
                }
                FeedbackSubject::CacheObject(object_id) => {
                    if let Some(status) = record.facts["status"].as_str() {
                        state.shared["kvflow_objects"][object_id.as_str()]["status"] =
                            json!(status);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn kvflow_request_ready(facts: &plex::Document, state: &State) -> bool {
    if let Some(required) = facts["required_objects"].as_array() {
        return required.iter().all(|object| {
            object
                .as_str()
                .is_some_and(|object| kvflow_status(state, object) == "gpu")
        });
    }
    facts["cache_ready"].as_bool().unwrap_or(false)
}

fn sync_kvflow_object(state: &mut State, object: &plex::CacheObject) {
    let object_id = object.object_id.as_str();
    if let Some(status) = object.facts["cache_state"].as_str() {
        state.shared["kvflow_objects"][object_id]["status"] = json!(status);
    } else if state.shared["kvflow_objects"][object_id]["status"].is_null() {
        state.shared["kvflow_objects"][object_id]["status"] =
            json!(if object.facts["loading"].as_bool().unwrap_or(false) {
                "loading"
            } else if object.facts["offloading"].as_bool().unwrap_or(false) {
                "offloading"
            } else if object.facts["cpu_backup"].as_bool().unwrap_or(false) {
                "cpu"
            } else {
                "gpu"
            });
    }
    state.shared["kvflow_objects"][object_id]["steps_to_execution"] =
        json!(kvflow_steps(&object.facts));
}

fn kvflow_status<'a>(state: &'a State, object_id: &str) -> &'a str {
    state.shared["kvflow_objects"][object_id]["status"]
        .as_str()
        .unwrap_or("gpu")
}

fn kvflow_steps(facts: &plex::Document) -> u64 {
    facts["beneficiary_steps"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|step| step.as_u64())
        .min()
        .or_else(|| facts["steps_to_execution"].as_u64())
        .unwrap_or(u64::MAX)
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

plex::export_policy!(KvFlow);
