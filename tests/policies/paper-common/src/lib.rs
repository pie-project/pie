#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
        let previous_queued = state.shared["vtc_queued_clients"]
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
        let queued = ctx
            .runnable
            .iter()
            .map(|candidate| {
                candidate.facts["queue_member"]
                    .as_bool()
                    .unwrap_or_else(|| {
                        candidate.facts["scheduler_state"].as_str() != Some("running")
                    })
            })
            .collect::<Vec<_>>();
        let mut current_members = BTreeMap::<String, Vec<String>>::new();
        for ((candidate, client), queued) in ctx.runnable.iter().zip(&clients).zip(&queued) {
            if *queued {
                current_members
                    .entry(client.clone())
                    .or_default()
                    .push(vtc_request_key(&candidate.request));
            }
        }
        let current_clients = current_members.keys().cloned().collect::<BTreeSet<_>>();
        let previous_members = state.shared["vtc_queue_members"]
            .as_object()
            .map(|members| {
                members
                    .iter()
                    .map(|(client, requests)| {
                        (
                            client.clone(),
                            requests
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(|request| request.as_str().map(str::to_owned))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        for client in previous_members
            .keys()
            .chain(current_members.keys())
            .cloned()
            .collect::<BTreeSet<_>>()
        {
            if previous_members.get(&client) != current_members.get(&client) {
                let version = state.shared["vtc_queue_versions"][&client]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(1);
                state.shared["vtc_queue_versions"][&client] = json!(version);
            }
        }
        state.shared["vtc_queue_members"] = json!(current_members);
        if current_clients.is_empty() && previous_queued.len() == 1 {
            state.shared["vtc_last_client"] =
                json!(previous_queued.iter().next().expect("one previous client"));
        }

        let mut arrival_order = Vec::new();
        for (client, queued) in clients.iter().zip(&queued) {
            if *queued && !arrival_order.contains(client) {
                arrival_order.push(client.clone());
            }
        }
        let became_active = arrival_order
            .iter()
            .map(|client| {
                let explicit = ctx
                    .runnable
                    .iter()
                    .zip(&clients)
                    .filter(|(_, candidate_client)| *candidate_client == client)
                    .find_map(|(candidate, _)| candidate.facts["client_became_active"].as_bool());
                (
                    client.clone(),
                    explicit.unwrap_or(!previous_queued.contains(client)),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut counters = BTreeMap::new();
        let mut active_for_lift = current_clients
            .iter()
            .filter(|client| !became_active[*client])
            .cloned()
            .collect::<BTreeSet<_>>();
        for client in &current_clients {
            counters.insert(
                client.clone(),
                state.shared["vtc"][client].as_u64().unwrap_or(0),
            );
        }
        for client in arrival_order {
            let mut counter = counters[&client];
            if became_active[&client] {
                let floor = active_for_lift
                    .iter()
                    .filter_map(|active| counters.get(active).copied())
                    .min()
                    .or_else(|| {
                        state.shared["vtc_last_client"]
                            .as_str()
                            .and_then(|last| state.shared["vtc"][last].as_u64())
                    })
                    .unwrap_or(0);
                counter = counter.max(floor);
                counters.insert(client.clone(), counter);
            }
            state.shared["vtc"][&client] = json!(counter);
            active_for_lift.insert(client);
        }
        state.shared["vtc_queued_clients"] = json!(current_clients);

        let mut queues = BTreeMap::<String, VecDeque<usize>>::new();
        for (index, (client, queued)) in clients.iter().zip(&queued).enumerate() {
            if *queued {
                queues.entry(client.clone()).or_default().push_back(index);
            }
        }
        let mut active_clients = queues.keys().cloned().collect::<BTreeSet<_>>();
        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        let mut pending = Vec::new();

        for (index, queued) in queued.iter().enumerate() {
            if *queued
                || remaining_selections == 0
                || remaining_requests == 0
                || remaining_tokens == 0
            {
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
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }

        while !active_clients.is_empty()
            && remaining_selections > 0
            && remaining_requests > 0
            && remaining_tokens > 0
        {
            let client = active_clients
                .iter()
                .min_by(|left, right| {
                    counters[*left]
                        .cmp(&counters[*right])
                        .then_with(|| left.cmp(right))
                })
                .expect("active client set is non-empty")
                .clone();
            let index = queues
                .get_mut(&client)
                .and_then(VecDeque::pop_front)
                .expect("active client queue is non-empty");
            let client_queue_empty = queues[&client].is_empty();
            if client_queue_empty {
                active_clients.remove(&client);
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
            let input_weight = ctx.runnable[index].facts["input_weight"]
                .as_u64()
                .unwrap_or(1);
            let fair_weight_ppm = ctx.runnable[index].facts["fair_weight_ppm"]
                .as_u64()
                .unwrap_or(1_000_000)
                .max(1);
            let input_price = ctx.runnable[index].facts["input_price"]
                .as_u64()
                .unwrap_or(1);
            let input_charge = vtc_charge(
                ctx.runnable[index].facts["dispatch_input_tokens"]
                    .as_u64()
                    .unwrap_or(0),
                input_weight,
                input_price,
                fair_weight_ppm,
            );
            counters
                .entry(client.clone())
                .and_modify(|counter| *counter = counter.saturating_add(input_charge));
            pending.push(json!({
                "selection_index": selections.len() - 1,
                "client_id": client,
                "request_key": vtc_request_key(&ctx.runnable[index].request),
                "input_charge": input_charge,
                "client_queue_empty": client_queue_empty,
                "global_queue_empty": active_clients.is_empty(),
                "queue_version": state.shared["vtc_queue_versions"][&client]
                    .as_u64()
                    .unwrap_or(0)
            }));
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        state.shared["vtc_pending"][ctx.meta.opportunity_id.as_str()] = json!(pending);
        trim_vtc_pending(state);
        Ok(SchedulePlan { selections })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            match &record.subject {
                FeedbackSubject::ScheduleSelection(subject) => {
                    if record.outcome != OutcomeKind::Progress {
                        continue;
                    }
                    let opportunity = subject.opportunity_id.as_str();
                    let selection_index = u64::from(subject.selection_index);
                    let pending = state.shared["vtc_pending"][opportunity]
                        .as_array()
                        .and_then(|entries| {
                            entries
                                .iter()
                                .find(|entry| {
                                    entry["selection_index"].as_u64() == Some(selection_index)
                                })
                                .cloned()
                        });
                    let Some(pending) = pending else {
                        continue;
                    };
                    let scheduled =
                        record.facts["scheduled_tokens"]
                            .as_u64()
                            .unwrap_or_else(|| {
                                u64::from(record.facts["status"].as_str() != Some("not-enacted"))
                            });
                    let client = pending["client_id"].as_str().unwrap_or("default");
                    if scheduled > 0 {
                        let request_key = pending["request_key"].as_str().unwrap_or("");
                        if !state.shared["vtc_input_charged"][request_key]
                            .as_bool()
                            .unwrap_or(false)
                        {
                            state.shared["vtc"][client] = json!(
                                state.shared["vtc"][client]
                                    .as_u64()
                                    .unwrap_or(0)
                                    .saturating_add(pending["input_charge"].as_u64().unwrap_or(0))
                            );
                            state.shared["vtc_input_charged"][request_key] = json!(true);
                        }
                    }
                    if let Some(entries) = state.shared["vtc_pending"][opportunity].as_array_mut() {
                        if let Some(entry) = entries.iter_mut().find(|entry| {
                            entry["selection_index"].as_u64() == Some(selection_index)
                        }) {
                            entry["resolved"] = json!(true);
                            entry["enacted"] = json!(scheduled > 0);
                        }
                    }
                    let resolved = state.shared["vtc_pending"][opportunity]
                        .as_array()
                        .is_some_and(|entries| {
                            !entries.is_empty()
                                && entries
                                    .iter()
                                    .all(|entry| entry["resolved"].as_bool().unwrap_or(false))
                        });
                    if resolved {
                        let entries = state.shared["vtc_pending"][opportunity]
                            .as_array()
                            .cloned()
                            .unwrap_or_default();
                        let mut queued = state.shared["vtc_queued_clients"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|value| value.as_str().map(str::to_owned))
                            .collect::<BTreeSet<_>>();
                        for client in entries
                            .iter()
                            .filter_map(|entry| entry["client_id"].as_str())
                            .collect::<BTreeSet<_>>()
                        {
                            let client_entries = entries
                                .iter()
                                .filter(|entry| entry["client_id"].as_str() == Some(client))
                                .collect::<Vec<_>>();
                            let Some(last) = client_entries.last() else {
                                continue;
                            };
                            let current_version = state.shared["vtc_queue_versions"][client]
                                .as_u64()
                                .unwrap_or(0);
                            if last["client_queue_empty"].as_bool().unwrap_or(false)
                                && last["queue_version"].as_u64() == Some(current_version)
                                && client_entries
                                    .iter()
                                    .all(|entry| entry["enacted"].as_bool().unwrap_or(false))
                            {
                                queued.remove(client);
                            }
                        }
                        if queued.is_empty()
                            && entries.last().is_some_and(|entry| {
                                entry["global_queue_empty"].as_bool().unwrap_or(false)
                            })
                            && entries
                                .iter()
                                .all(|entry| entry["enacted"].as_bool().unwrap_or(false))
                        {
                            state.shared["vtc_last_client"] = json!(
                                entries
                                    .last()
                                    .and_then(|entry| { entry["client_id"].as_str() })
                            );
                        }
                        state.shared["vtc_queued_clients"] = json!(queued);
                        state
                            .shared
                            .get_mut("vtc_pending")
                            .and_then(|pending| pending.as_object_mut())
                            .map(|pending| pending.remove(opportunity));
                    }
                }
                FeedbackSubject::Request(request_id) => {
                    if record.outcome != OutcomeKind::Progress {
                        if let Ok(request) = state.request(request_id.as_str()) {
                            let request_key = vtc_request_key(request.reference());
                            state
                                .shared
                                .get_mut("vtc_input_charged")
                                .and_then(|charged| charged.as_object_mut())
                                .map(|charged| charged.remove(&request_key));
                        }
                        continue;
                    }
                    let client = record.facts["client_id"].as_str().unwrap_or("default");
                    let output_weight = record.facts["output_weight"].as_u64().unwrap_or(1);
                    let fair_weight_ppm = record.facts["fair_weight_ppm"]
                        .as_u64()
                        .unwrap_or(1_000_000)
                        .max(1);
                    let output_price = record.facts["output_price"].as_u64().unwrap_or(2);
                    let charge = vtc_charge(
                        record.facts["output_tokens"].as_u64().unwrap_or(0),
                        output_weight,
                        output_price,
                        fair_weight_ppm,
                    );
                    state.shared["vtc"][client] = json!(
                        state.shared["vtc"][client]
                            .as_u64()
                            .unwrap_or(0)
                            .saturating_add(charge)
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }
}

const VTC_FIXED_POINT: u128 = 1_000_000;
const PARTS_PER_MILLION: u128 = 1_000_000;

fn vtc_charge(tokens: u64, weight: u64, price: u64, fair_weight_ppm: u64) -> u64 {
    let numerator = u128::from(tokens)
        .saturating_mul(u128::from(weight))
        .saturating_mul(u128::from(price))
        .saturating_mul(VTC_FIXED_POINT)
        .saturating_mul(PARTS_PER_MILLION);
    let charge = numerator / u128::from(fair_weight_ppm.max(1));
    charge.min(u128::from(u64::MAX)) as u64
}

fn vtc_request_key(request: &plex::RequestRef) -> String {
    format!("{}#{}", request.request_id.as_str(), request.generation_id)
}

fn trim_vtc_pending(state: &mut State) {
    const MAX_PENDING_OPPORTUNITIES: usize = 64;
    let Some(pending) = state.shared["vtc_pending"].as_object_mut() else {
        return;
    };
    if pending.len() <= MAX_PENDING_OPPORTUNITIES {
        return;
    }
    let mut opportunities = pending.keys().cloned().collect::<Vec<_>>();
    opportunities.sort();
    for opportunity in opportunities
        .into_iter()
        .take(pending.len() - MAX_PENDING_OPPORTUNITIES)
    {
        pending.remove(&opportunity);
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
        let mut order = Vec::new();
        for (index, candidate) in ctx.candidates.iter().enumerate() {
            let user = fairserve_user(&candidate.facts);
            let application = fairserve_application(&candidate.facts);
            let counted = state
                .request(candidate.request.request_id.as_str())?
                .scratch["fairserve_arrival_counted"]
                .as_bool()
                .unwrap_or(false);
            let mut user_count = None;
            let mut application_count = None;
            if !counted {
                if let Some(now_ms) = candidate.facts["now_ms"].as_u64() {
                    let window_ms = candidate.facts["rpm_window_ms"]
                        .as_u64()
                        .unwrap_or(60_000)
                        .max(1);
                    user_count = Some(record_fairserve_arrival(
                        state, "users", user, now_ms, window_ms,
                    ));
                    application_count = Some(record_fairserve_arrival(
                        state,
                        "applications",
                        application,
                        now_ms,
                        window_ms,
                    ));
                }
                state
                    .request_mut(candidate.request.request_id.as_str())?
                    .scratch["fairserve_arrival_counted"] = json!(true);
            } else if let Some(now_ms) = candidate.facts["now_ms"].as_u64() {
                let window_ms = candidate.facts["rpm_window_ms"]
                    .as_u64()
                    .unwrap_or(60_000)
                    .max(1);
                user_count = Some(fairserve_window_count(
                    state, "users", user, now_ms, window_ms,
                ));
                application_count = Some(fairserve_window_count(
                    state,
                    "applications",
                    application,
                    now_ms,
                    window_ms,
                ));
            }

            let interaction_in_progress = candidate.facts["interaction_in_progress"]
                .as_bool()
                .unwrap_or(false);
            let overloaded = candidate.facts["kv_overloaded"].as_bool().unwrap_or(false);
            let user_over_limit = candidate.facts["user_rpm_limit"]
                .as_u64()
                .zip(user_count)
                .is_some_and(|(limit, count)| count > limit as usize)
                || candidate.facts["user_rpm_limit"].as_u64().is_none()
                    && candidate.facts["user_rpm_remaining"].as_u64() == Some(0);
            let application_over_limit = candidate.facts["app_rpm_limit"]
                .as_u64()
                .zip(application_count)
                .is_some_and(|(limit, count)| count > limit as usize)
                || candidate.facts["app_rpm_limit"].as_u64().is_none()
                    && candidate.facts["app_rpm_remaining"].as_u64() == Some(0);
            if !overloaded
                || interaction_in_progress
                || (!user_over_limit && !application_over_limit)
            {
                order.push(index);
            }
        }
        order.sort_by_key(|&index| {
            let candidate = &ctx.candidates[index];
            let user = fairserve_user(&candidate.facts);
            (
                !candidate.facts["interaction_in_progress"]
                    .as_bool()
                    .unwrap_or(false),
                state.shared["fairserve_users"][user].as_u64().unwrap_or(0),
                candidate.facts["arrival_seq"]
                    .as_u64()
                    .unwrap_or(index as u64),
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
        let previous_active = state.shared["fairserve_active_users"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|user| user.as_str().map(str::to_owned))
            .collect::<BTreeSet<_>>();
        let current_active = ctx
            .runnable
            .iter()
            .map(|candidate| fairserve_user(&candidate.facts).to_owned())
            .collect::<BTreeSet<_>>();
        if previous_active.len() == 1 {
            let previous = previous_active.iter().next().expect("one previous user");
            if !current_active.contains(previous) {
                state.shared["fairserve_last_user"] = json!(previous);
            }
        }
        let mut active_for_lift = current_active
            .iter()
            .filter(|user| previous_active.contains(*user))
            .cloned()
            .collect::<BTreeSet<_>>();
        for candidate in &ctx.runnable {
            let user = fairserve_user(&candidate.facts);
            let became_active = candidate.facts["user_became_active"]
                .as_bool()
                .unwrap_or(!previous_active.contains(user));
            let mut counter = state.shared["fairserve_users"][user].as_u64().unwrap_or(0);
            if became_active {
                let floor = active_for_lift
                    .iter()
                    .filter_map(|active| state.shared["fairserve_users"][active].as_u64())
                    .min()
                    .or_else(|| {
                        state.shared["fairserve_last_user"]
                            .as_str()
                            .and_then(|last| state.shared["fairserve_users"][last].as_u64())
                    })
                    .unwrap_or(0);
                counter = counter.max(floor);
            }
            state.shared["fairserve_users"][user] = json!(counter);
            active_for_lift.insert(user.to_owned());
        }
        state.shared["fairserve_active_users"] = json!(current_active);
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let user = fairserve_user(&candidate.facts);
            (
                !candidate.facts["interaction_in_progress"]
                    .as_bool()
                    .unwrap_or(false),
                state.shared["fairserve_users"][user].as_u64().unwrap_or(0),
                candidate.facts["arrival_seq"]
                    .as_u64()
                    .unwrap_or(index as u64),
                index,
            )
        });
        Ok(select_singletons(ctx, order))
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let FeedbackSubject::Request(request_id) = &record.subject else {
                continue;
            };
            let request_id = request_id.as_str();
            if record.outcome == OutcomeKind::Progress {
                let request = state.request_mut(request_id)?;
                accumulate_fairserve_tokens(request, &record.facts);
                continue;
            }
            if record.outcome != OutcomeKind::Completed {
                continue;
            }

            let (request_facts, input_tokens, system_tokens, output_tokens, already_accounted) = {
                let request = state.request_mut(request_id)?;
                accumulate_fairserve_tokens(request, &record.facts);
                let values = (
                    request.facts().clone(),
                    request.scratch["fairserve_input_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                    request.scratch["fairserve_system_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                    request.scratch["fairserve_output_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                    request.scratch["fairserve_accounted"]
                        .as_bool()
                        .unwrap_or(false),
                );
                request.scratch["fairserve_accounted"] = json!(true);
                values
            };
            if already_accounted {
                continue;
            }

            let user = fairserve_fact_str(&record.facts, &request_facts, "user_id")
                .or_else(|| fairserve_fact_str(&record.facts, &request_facts, "client_id"))
                .unwrap_or("default");
            let application = fairserve_fact_str(&record.facts, &request_facts, "application_id")
                .unwrap_or("default");
            let stage =
                fairserve_fact_str(&record.facts, &request_facts, "stage_id").unwrap_or("default");
            let input_weight =
                fairserve_fact_u64(&record.facts, &request_facts, "input_weight").unwrap_or(1);
            let system_weight =
                fairserve_fact_u64(&record.facts, &request_facts, "system_weight").unwrap_or(2);
            let output_weight =
                fairserve_fact_u64(&record.facts, &request_facts, "output_weight").unwrap_or(1);
            let expected_input =
                fairserve_fact_u64(&record.facts, &request_facts, "expected_input_tokens")
                    .unwrap_or(input_tokens);
            let expected_system =
                fairserve_fact_u64(&record.facts, &request_facts, "expected_system_tokens")
                    .unwrap_or(system_tokens);
            let expected_output =
                fairserve_fact_u64(&record.facts, &request_facts, "expected_output_tokens")
                    .unwrap_or(output_tokens);
            let priority_ppm =
                fairserve_fact_u64(&record.facts, &request_facts, "user_priority_ppm")
                    .unwrap_or(1_000_000);
            let actual = u128::from(input_tokens)
                .saturating_mul(u128::from(input_weight))
                .saturating_add(u128::from(system_tokens).saturating_mul(u128::from(system_weight)))
                .saturating_add(
                    u128::from(output_tokens).saturating_mul(u128::from(output_weight)),
                );
            let expected = u128::from(expected_input)
                .saturating_mul(u128::from(input_weight))
                .saturating_add(
                    u128::from(expected_system).saturating_mul(u128::from(system_weight)),
                )
                .saturating_add(
                    u128::from(expected_output).saturating_mul(u128::from(output_weight)),
                )
                .max(1);
            let increment = actual
                .saturating_mul(u128::from(priority_ppm))
                .checked_div(expected)
                .unwrap_or(0)
                .min(u128::from(u64::MAX)) as u64;
            state.shared["fairserve_users"][user] = json!(
                state.shared["fairserve_users"][user]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(increment)
            );
            let stage_key = format!("{application}::{stage}");
            state.shared["fairserve_stage_service"][user][&stage_key] = json!(
                state.shared["fairserve_stage_service"][user][&stage_key]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(increment)
            );
        }
        Ok(())
    }
}

fn fairserve_user(facts: &plex::Document) -> &str {
    facts["user_id"]
        .as_str()
        .or_else(|| facts["client_id"].as_str())
        .unwrap_or("default")
}

fn fairserve_application(facts: &plex::Document) -> &str {
    facts["application_id"].as_str().unwrap_or("default")
}

fn record_fairserve_arrival(
    state: &mut State,
    scope: &str,
    key: &str,
    now_ms: u64,
    window_ms: u64,
) -> usize {
    let mut arrivals = state.shared["fairserve_arrivals"][scope][key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|arrival| arrival.as_u64())
        .filter(|arrival| now_ms.saturating_sub(*arrival) < window_ms)
        .collect::<Vec<_>>();
    arrivals.push(now_ms);
    let count = arrivals.len();
    state.shared["fairserve_arrivals"][scope][key] = json!(arrivals);
    count
}

fn fairserve_window_count(
    state: &mut State,
    scope: &str,
    key: &str,
    now_ms: u64,
    window_ms: u64,
) -> usize {
    let arrivals = state.shared["fairserve_arrivals"][scope][key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|arrival| arrival.as_u64())
        .filter(|arrival| now_ms.saturating_sub(*arrival) < window_ms)
        .collect::<Vec<_>>();
    let count = arrivals.len();
    state.shared["fairserve_arrivals"][scope][key] = json!(arrivals);
    count
}

fn accumulate_fairserve_tokens(request: &mut plex::Request, facts: &plex::Document) {
    for (field, scratch) in [
        ("input_tokens", "fairserve_input_tokens"),
        ("system_tokens", "fairserve_system_tokens"),
        ("output_tokens", "fairserve_output_tokens"),
    ] {
        let delta = facts[field].as_u64().unwrap_or(0);
        request.scratch[scratch] = json!(
            request.scratch[scratch]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(delta)
        );
    }
}

fn fairserve_fact_u64(
    primary: &plex::Document,
    fallback: &plex::Document,
    field: &str,
) -> Option<u64> {
    primary[field].as_u64().or_else(|| fallback[field].as_u64())
}

fn fairserve_fact_str<'a>(
    primary: &'a plex::Document,
    fallback: &'a plex::Document,
    field: &str,
) -> Option<&'a str> {
    primary[field].as_str().or_else(|| fallback[field].as_str())
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
        let mut tentative = BTreeMap::<(String, String), i64>::new();
        let mut pending = Vec::new();
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let client = request.facts["client_id"].as_str().unwrap_or("default");
            remember_dlpm_client(
                state,
                client,
                request.facts["quantum"].as_i64().unwrap_or(1),
            );
            for (target_index, target) in ctx.targets.iter().enumerate() {
                let target_id = target.target_id.as_str();
                let seeded = ctx
                    .feasible_edges
                    .iter()
                    .find(|edge| {
                        edge.request_index as usize == request_index
                            && edge.target_index as usize == target_index
                    })
                    .and_then(|edge| edge.facts["worker_deficit"].as_i64());
                let deficit = state.shared["dlpm_worker_deficit"][client][target_id]
                    .as_i64()
                    .or(seeded)
                    .unwrap_or(0);
                if state.shared["dlpm_worker_deficit"][client][target_id].is_null() {
                    state.shared["dlpm_worker_deficit"][client][target_id] = json!(deficit);
                }
                tentative.insert((client.to_owned(), target_id.to_owned()), deficit);
            }

            let feasible = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .collect::<Vec<_>>();
            if feasible.is_empty() {
                decisions.push(RouteDecision::Defer);
                continue;
            }
            if !feasible.iter().any(|(_, edge)| {
                let target_id = ctx.targets[edge.target_index as usize].target_id.as_str();
                tentative[&(client.to_owned(), target_id.to_owned())] > 0
            }) {
                refill_dlpm_workers(ctx, state, client, &mut tentative);
            }

            let explicit_longest = feasible
                .iter()
                .any(|(_, edge)| edge.facts["longest_prefix_match"].as_bool() == Some(true));
            let longest = feasible
                .iter()
                .map(|(_, edge)| edge.facts["cached_tokens"].as_u64().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let positive = feasible
                .iter()
                .copied()
                .filter(|(_, edge)| {
                    let target_id = ctx.targets[edge.target_index as usize].target_id.as_str();
                    tentative[&(client.to_owned(), target_id.to_owned())] > 0
                })
                .collect::<Vec<_>>();
            let local = positive
                .iter()
                .copied()
                .filter(|(_, edge)| {
                    if explicit_longest {
                        edge.facts["longest_prefix_match"].as_bool() == Some(true)
                    } else {
                        edge.facts["cached_tokens"].as_u64().unwrap_or(0) == longest
                    }
                })
                .collect::<Vec<_>>();
            let pool = if local.is_empty() { &positive } else { &local };
            let selected = pool.iter().copied().min_by_key(|(edge_index, edge)| {
                let target_id = ctx.targets[edge.target_index as usize].target_id.as_str();
                (
                    edge.facts["queue_size"]
                        .as_u64()
                        .or_else(|| edge.facts["load"].as_u64())
                        .or_else(|| state.shared["dlpm_worker_queue"][target_id].as_u64())
                        .unwrap_or(u64::MAX),
                    edge.target_index,
                    *edge_index,
                )
            });
            if let Some((index, edge)) = selected {
                let target_index = edge.target_index as usize;
                let target_id = ctx.targets[target_index].target_id.as_str();
                target_counts[target_index] += 1;
                decisions.push(RouteDecision::Assign(index as u32));
                let charge = dlpm_charge(
                    request.facts["input_tokens"].as_u64().unwrap_or(0),
                    request.facts["input_weight"]
                        .as_u64()
                        .or_else(|| request.facts["extend_weight"].as_u64())
                        .unwrap_or(1),
                );
                tentative
                    .entry((client.to_owned(), target_id.to_owned()))
                    .and_modify(|deficit| *deficit = deficit.saturating_sub(charge));
                pending.push(json!({
                    "request_index": request_index,
                    "client_id": client,
                    "target_id": target_id,
                    "input_charge": charge
                }));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        state.shared["dlpm_route_pending"][ctx.meta.opportunity_id.as_str()] = json!(pending);
        trim_dlpm_pending(state, "dlpm_route_pending");
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
        for (candidate, client) in ctx.runnable.iter().zip(&clients) {
            remember_dlpm_client(
                state,
                client,
                candidate.facts["client_quantum"]
                    .as_i64()
                    .or_else(|| candidate.facts["quantum"].as_i64())
                    .unwrap_or(1),
            );
        }
        let active_clients = clients.iter().cloned().collect::<BTreeSet<_>>();
        if !active_clients
            .iter()
            .any(|client| state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0) > 0)
        {
            refill_dlpm_clients(state);
        }
        let queued = ctx
            .runnable
            .iter()
            .map(|candidate| {
                candidate.facts["queue_member"]
                    .as_bool()
                    .unwrap_or_else(|| {
                        candidate.facts["scheduler_state"].as_str() != Some("running")
                    })
            })
            .collect::<Vec<_>>();
        let mut order = (0..ctx.runnable.len())
            .filter(|index| queued[*index])
            .collect::<Vec<_>>();
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
        let mut pending = Vec::new();
        let mut tentative = active_clients
            .iter()
            .map(|client| {
                (
                    client.clone(),
                    state.shared["dlpm_deficit"][client].as_i64().unwrap_or(0),
                )
            })
            .collect::<BTreeMap<_, _>>();

        for (index, queued) in queued.iter().enumerate() {
            if *queued
                || remaining_selections == 0
                || remaining_requests == 0
                || remaining_tokens == 0
            {
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
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }

        for index in order {
            if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let client = &clients[index];
            if tentative[client] <= 0 && !active_clients.iter().any(|active| tentative[active] > 0)
            {
                refill_dlpm_clients_tentative(state, &mut tentative);
            }
            if tentative[client] <= 0 {
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
                .as_u64()
                .unwrap_or(u64::from(budget));
            let weight = ctx.runnable[index].facts["extend_weight"]
                .as_u64()
                .or_else(|| ctx.runnable[index].facts["input_weight"].as_u64())
                .unwrap_or(1);
            let charge = dlpm_charge(extend, weight);
            tentative
                .entry(client.clone())
                .and_modify(|deficit| *deficit = deficit.saturating_sub(charge));
            pending.push(json!({
                "selection_index": selections.len() - 1,
                "client_id": client,
                "extend_tokens": extend,
                "extend_weight": weight
            }));
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        state.shared["dlpm_schedule_pending"][ctx.meta.opportunity_id.as_str()] = json!(pending);
        trim_dlpm_pending(state, "dlpm_schedule_pending");
        Ok(SchedulePlan { selections })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            match &record.subject {
                FeedbackSubject::ScheduleSelection(subject)
                    if record.outcome == OutcomeKind::Progress =>
                {
                    let opportunity = subject.opportunity_id.as_str();
                    let selection_index = u64::from(subject.selection_index);
                    let pending = state.shared["dlpm_schedule_pending"][opportunity]
                        .as_array()
                        .and_then(|entries| {
                            entries.iter().find(|entry| {
                                entry["selection_index"].as_u64() == Some(selection_index)
                            })
                        })
                        .cloned();
                    let Some(pending) = pending else {
                        continue;
                    };
                    let scheduled = record.facts["scheduled_tokens"].as_u64().unwrap_or(0);
                    if scheduled > 0 {
                        let extend = pending["extend_tokens"]
                            .as_u64()
                            .unwrap_or(0)
                            .min(scheduled);
                        let charge =
                            dlpm_charge(extend, pending["extend_weight"].as_u64().unwrap_or(1));
                        let client = pending["client_id"].as_str().unwrap_or("default");
                        state.shared["dlpm_deficit"][client] = json!(
                            state.shared["dlpm_deficit"][client]
                                .as_i64()
                                .unwrap_or(0)
                                .saturating_sub(charge)
                        );
                    }
                    remove_dlpm_pending(
                        state,
                        "dlpm_schedule_pending",
                        opportunity,
                        "selection_index",
                        selection_index,
                    );
                }
                FeedbackSubject::RouteAssignment(subject)
                    if record.outcome == OutcomeKind::Progress =>
                {
                    let opportunity = subject.opportunity_id.as_str();
                    let request_index = u64::from(subject.request_index);
                    let pending = state.shared["dlpm_route_pending"][opportunity]
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
                        let client = pending["client_id"].as_str().unwrap_or("default");
                        let target = pending["target_id"].as_str().unwrap_or("default");
                        state.shared["dlpm_worker_deficit"][client][target] = json!(
                            state.shared["dlpm_worker_deficit"][client][target]
                                .as_i64()
                                .unwrap_or(0)
                                .saturating_sub(pending["input_charge"].as_i64().unwrap_or(0))
                        );
                        state.shared["dlpm_worker_queue"][target] = json!(
                            state.shared["dlpm_worker_queue"][target]
                                .as_u64()
                                .unwrap_or(0)
                                .saturating_add(1)
                        );
                    }
                    remove_dlpm_pending(
                        state,
                        "dlpm_route_pending",
                        opportunity,
                        "request_index",
                        request_index,
                    );
                }
                FeedbackSubject::Request(_) => {
                    let client = record.facts["client_id"].as_str().unwrap_or("default");
                    let output_tokens = record.facts["output_tokens"].as_u64().unwrap_or(0);
                    let output_weight = record.facts["output_weight"].as_u64().unwrap_or(1);
                    if matches!(
                        record.outcome,
                        OutcomeKind::Progress | OutcomeKind::Completed
                    ) && output_tokens > 0
                    {
                        state.shared["dlpm_deficit"][client] = json!(
                            state.shared["dlpm_deficit"][client]
                                .as_i64()
                                .unwrap_or(0)
                                .saturating_sub(dlpm_charge(output_tokens, output_weight))
                        );
                    }
                    if (record.outcome == OutcomeKind::Completed
                        || record.facts["request_finished"].as_bool().unwrap_or(false))
                        && let Some(target) = record.facts["target_id"].as_str()
                    {
                        state.shared["dlpm_worker_deficit"][client][target] = json!(
                            state.shared["dlpm_worker_deficit"][client][target]
                                .as_i64()
                                .unwrap_or(0)
                                .saturating_sub(dlpm_charge(output_tokens, output_weight))
                        );
                        state.shared["dlpm_worker_queue"][target] = json!(
                            state.shared["dlpm_worker_queue"][target]
                                .as_u64()
                                .unwrap_or(0)
                                .saturating_sub(1)
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn remember_dlpm_client(state: &mut State, client: &str, quantum: i64) {
    let mut clients = state.shared["dlpm_clients"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    clients.insert(client.to_owned());
    state.shared["dlpm_clients"] = json!(clients);
    state.shared["dlpm_client_quantum"][client] = json!(quantum.max(1));
    if state.shared["dlpm_deficit"][client].is_null() {
        state.shared["dlpm_deficit"][client] = json!(0);
    }
}

fn refill_dlpm_clients(state: &mut State) {
    let clients = state.shared["dlpm_clients"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    for client in clients {
        let deficit = state.shared["dlpm_deficit"][&client].as_i64().unwrap_or(0);
        if deficit <= 0 {
            let quantum = state.shared["dlpm_client_quantum"][&client]
                .as_i64()
                .unwrap_or(1)
                .max(1);
            state.shared["dlpm_deficit"][&client] = json!(deficit.saturating_add(quantum));
        }
    }
}

fn refill_dlpm_clients_tentative(state: &mut State, tentative: &mut BTreeMap<String, i64>) {
    let clients = state.shared["dlpm_clients"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    for client in clients {
        let current = tentative
            .get(&client)
            .copied()
            .unwrap_or_else(|| state.shared["dlpm_deficit"][&client].as_i64().unwrap_or(0));
        if current <= 0 {
            let quantum = state.shared["dlpm_client_quantum"][&client]
                .as_i64()
                .unwrap_or(1)
                .max(1);
            let next = current.saturating_add(quantum);
            tentative.insert(client.clone(), next);
            state.shared["dlpm_deficit"][&client] = json!(
                state.shared["dlpm_deficit"][&client]
                    .as_i64()
                    .unwrap_or(0)
                    .saturating_add(quantum)
            );
        }
    }
}

fn refill_dlpm_workers(
    ctx: &RouteContext,
    state: &mut State,
    client: &str,
    tentative: &mut BTreeMap<(String, String), i64>,
) {
    let rounds = ctx
        .targets
        .iter()
        .map(|target| {
            let target_id = target.target_id.as_str();
            let deficit = tentative[&(client.to_owned(), target_id.to_owned())];
            let quantum = target.facts["worker_quantum"]
                .as_i64()
                .or_else(|| target.facts["quantum"].as_i64())
                .unwrap_or(1)
                .max(1);
            if deficit > 0 {
                0
            } else {
                deficit
                    .unsigned_abs()
                    .saturating_div(quantum as u64)
                    .saturating_add(1)
            }
        })
        .min()
        .unwrap_or(0);
    if rounds == 0 {
        return;
    }
    for target in &ctx.targets {
        let target_id = target.target_id.as_str();
        let quantum = target.facts["worker_quantum"]
            .as_i64()
            .or_else(|| target.facts["quantum"].as_i64())
            .unwrap_or(1)
            .max(1);
        let refill = quantum.saturating_mul(rounds.min(i64::MAX as u64) as i64);
        let persistent = state.shared["dlpm_worker_deficit"][client][target_id]
            .as_i64()
            .unwrap_or(0)
            .saturating_add(refill);
        state.shared["dlpm_worker_deficit"][client][target_id] = json!(persistent);
        let current = tentative[&(client.to_owned(), target_id.to_owned())];
        tentative.insert(
            (client.to_owned(), target_id.to_owned()),
            current.saturating_add(refill),
        );
    }
}

fn dlpm_charge(tokens: u64, weight: u64) -> i64 {
    u128::from(tokens)
        .saturating_mul(u128::from(weight))
        .min(i64::MAX as u128) as i64
}

fn trim_dlpm_pending(state: &mut State, field: &str) {
    const MAX_PENDING_OPPORTUNITIES: usize = 64;
    let Some(pending) = state.shared[field].as_object_mut() else {
        return;
    };
    if pending.len() <= MAX_PENDING_OPPORTUNITIES {
        return;
    }
    let mut opportunities = pending.keys().cloned().collect::<Vec<_>>();
    opportunities.sort();
    for opportunity in opportunities
        .into_iter()
        .take(pending.len() - MAX_PENDING_OPPORTUNITIES)
    {
        pending.remove(&opportunity);
    }
}

fn remove_dlpm_pending(
    state: &mut State,
    field: &str,
    opportunity: &str,
    index_field: &str,
    index: u64,
) {
    if let Some(entries) = state.shared[field][opportunity].as_array_mut() {
        entries.retain(|entry| entry[index_field].as_u64() != Some(index));
    }
    if state.shared[field][opportunity]
        .as_array()
        .is_some_and(Vec::is_empty)
        && let Some(pending) = state.shared[field].as_object_mut()
    {
        pending.remove(opportunity);
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
        let now_us = ctx.capacity.facts["now_us"]
            .as_u64()
            .or_else(|| {
                ctx.runnable
                    .iter()
                    .find_map(|candidate| candidate.facts["now_us"].as_u64())
            })
            .unwrap_or_else(|| state.shared["justitia_last_time_us"].as_u64().unwrap_or(0));
        let total_kv_tokens = ctx.capacity.facts["total_kv_tokens"]
            .as_u64()
            .or_else(|| ctx.capacity.facts["total_kv_blocks"].as_u64())
            .or_else(|| {
                ctx.runnable
                    .iter()
                    .find_map(|candidate| candidate.facts["total_kv_tokens"].as_u64())
            })
            .unwrap_or(1)
            .max(1);
        advance_justitia_clock(state, now_us, total_kv_tokens);

        let mut groups = BTreeMap::<String, Vec<usize>>::new();
        for (index, candidate) in ctx.runnable.iter().enumerate() {
            if !candidate.facts["ready"].as_bool().unwrap_or(true) {
                continue;
            }
            let group_id = candidate
                .request
                .group_id
                .as_ref()
                .map(|group| group.as_str())
                .unwrap_or_else(|| candidate.request.request_id.as_str())
                .to_owned();
            groups.entry(group_id).or_default().push(index);
        }
        let mut active_groups = state.shared["justitia_active_groups"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|group| group.as_str().map(str::to_owned))
            .collect::<BTreeSet<_>>();
        for (group_id, indices) in &groups {
            let initialized = state
                .group(group_id)
                .ok()
                .and_then(|group| group.scratch["justitia_finish_tag_fp"].as_u64())
                .is_some();
            if initialized {
                active_groups.insert(group_id.clone());
                continue;
            }
            let predicted_cost = indices
                .iter()
                .find_map(|index| {
                    ctx.runnable[*index].facts["predicted_agent_kv_token_time"].as_u64()
                })
                .or_else(|| {
                    Some(
                        indices
                            .iter()
                            .filter_map(|index| {
                                justitia_inference_cost(&ctx.runnable[*index].facts)
                            })
                            .fold(0u64, u64::saturating_add),
                    )
                })
                .filter(|cost| *cost > 0)
                .unwrap_or(1);
            let finish_tag = state.shared["justitia_virtual_time_fp"]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(predicted_cost.saturating_mul(1_000_000));
            if let Ok(group) = state.group_mut(group_id) {
                group.scratch["justitia_finish_tag_fp"] = json!(finish_tag);
                group.scratch["justitia_predicted_cost"] = json!(predicted_cost);
                group.scratch["justitia_arrival_time_us"] = json!(now_us);
                group.scratch["justitia_active"] = json!(true);
                group.scratch["justitia_consumed_cost"] = json!(0);
            }
            active_groups.insert(group_id.clone());
        }
        state.shared["justitia_active_groups"] = json!(active_groups);

        if ctx.capacity.max_selections == 0
            || ctx.capacity.max_requests == 0
            || ctx.capacity.max_total_tokens == 0
        {
            return Ok(SchedulePlan {
                selections: Vec::new(),
            });
        }
        let selected_group = groups.keys().min_by_key(|group_id| {
            (
                state
                    .group(group_id)
                    .ok()
                    .and_then(|group| group.scratch["justitia_finish_tag_fp"].as_u64())
                    .unwrap_or(u64::MAX),
                *group_id,
            )
        });
        let Some(selected_group) = selected_group else {
            return Ok(SchedulePlan {
                selections: Vec::new(),
            });
        };
        let mut requests = Vec::new();
        let mut token_budgets = Vec::new();
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        for index in &groups[selected_group] {
            if remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let budget =
                u64::from(ctx.runnable[*index].max_token_budget).min(remaining_tokens) as u32;
            if budget == 0 {
                continue;
            }
            requests.push(*index as u32);
            token_budgets.push(budget);
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        Ok(SchedulePlan {
            selections: if requests.is_empty() {
                Vec::new()
            } else {
                vec![ScheduleSelection {
                    requests,
                    token_budgets,
                }]
            },
        })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            if let (Some(now_us), Some(total_kv_tokens)) = (
                record.facts["now_us"].as_u64(),
                record.facts["total_kv_tokens"].as_u64(),
            ) {
                advance_justitia_clock(state, now_us, total_kv_tokens.max(1));
            }
            match &record.subject {
                FeedbackSubject::WorkGroup(group_id) => {
                    if record.outcome == OutcomeKind::Progress {
                        if let Some(delta) = record.facts["kv_token_time_delta"].as_u64() {
                            let group = state.group_mut(group_id.as_str())?;
                            group.scratch["justitia_consumed_cost"] = json!(
                                group.scratch["justitia_consumed_cost"]
                                    .as_u64()
                                    .unwrap_or(0)
                                    .saturating_add(delta)
                            );
                        }
                    } else if record.outcome == OutcomeKind::Completed {
                        complete_justitia_group(state, group_id.as_str())?;
                    }
                }
                FeedbackSubject::Request(request_id) => {
                    let group_id = state
                        .request(request_id.as_str())?
                        .reference()
                        .group_id
                        .clone();
                    if record.outcome == OutcomeKind::Progress {
                        if let (Some(group_id), Some(delta)) = (
                            group_id.as_ref(),
                            record.facts["kv_token_time_delta"].as_u64(),
                        ) {
                            let group = state.group_mut(group_id.as_str())?;
                            group.scratch["justitia_consumed_cost"] = json!(
                                group.scratch["justitia_consumed_cost"]
                                    .as_u64()
                                    .unwrap_or(0)
                                    .saturating_add(delta)
                            );
                        }
                    } else if record.outcome == OutcomeKind::Completed
                        && record.facts["agent_completed"].as_bool().unwrap_or(false)
                        && let Some(group_id) = group_id
                    {
                        complete_justitia_group(state, group_id.as_str())?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn justitia_inference_cost(facts: &plex::Document) -> Option<u64> {
    facts["predicted_kv_token_time"].as_u64().or_else(|| {
        let prefill = facts["predicted_input_tokens"].as_u64()?;
        let decode = facts["predicted_output_tokens"].as_u64()?;
        let cost = u128::from(prefill)
            .saturating_mul(u128::from(decode))
            .saturating_add(
                u128::from(decode).saturating_mul(u128::from(decode).saturating_add(1)) / 2,
            );
        Some(cost.min(u128::from(u64::MAX)) as u64)
    })
}

fn advance_justitia_clock(state: &mut State, now_us: u64, total_kv_tokens: u64) {
    let last = state.shared["justitia_last_time_us"]
        .as_u64()
        .unwrap_or(now_us);
    if now_us <= last {
        state.shared["justitia_last_time_us"] = json!(last);
        return;
    }
    let active = state.shared["justitia_active_groups"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    if active > 0 {
        let delta = u128::from(now_us - last)
            .saturating_mul(u128::from(total_kv_tokens))
            .saturating_mul(1_000_000)
            / active as u128;
        state.shared["justitia_virtual_time_fp"] = json!(
            state.shared["justitia_virtual_time_fp"]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(delta.min(u128::from(u64::MAX)) as u64)
        );
    }
    state.shared["justitia_last_time_us"] = json!(now_us);
}

fn complete_justitia_group(state: &mut State, group_id: &str) -> plex::Result<()> {
    let group = state.group_mut(group_id)?;
    group.scratch["justitia_active"] = json!(false);
    let mut active = state.shared["justitia_active_groups"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|group| group.as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    active.remove(group_id);
    state.shared["justitia_active_groups"] = json!(active);
    Ok(())
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
