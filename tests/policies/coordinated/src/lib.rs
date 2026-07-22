use plex::serde_json::json;
use plex::{Document, Host, Policy, Request, State};

struct Coordinated;

impl Policy for Coordinated {
    fn route(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let request_id = request_id(ctx)?;
        let previous_target = state.request(&request_id)?.facts()["previous_target"]
            .as_str()
            .map(str::to_owned);
        {
            let request = state.request_mut(&request_id)?;
            request.scratch["route_count"] =
                json!(request.scratch["route_count"].as_u64().unwrap_or(0) + 1);
            request.fields["metadata"]["last_hook"] = json!("route");
            append_prompt(request, "|route");
        }
        state.shared["route_calls"] = json!(state.shared["route_calls"].as_u64().unwrap_or(0) + 1);
        state.shared["last_route_request"] = json!(request_id);

        let capacity_bias = if supports(ctx, "queries", "pie.cluster.capacity@1") {
            host.cluster_capacity(ctx["context"]["model"].as_str().unwrap_or(""))?["route_bias"]
                .as_f64()
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let candidates = ctx["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?;
        let scores = candidates
            .iter()
            .map(|candidate| {
                let cached = candidate["facts"]["cached_tokens"].as_f64().unwrap_or(0.0);
                let queue = candidate["facts"]["queue_depth"].as_f64().unwrap_or(0.0);
                let retained = previous_target.as_deref() == candidate["id"].as_str()
                    && candidate["facts"]["has_request_kv"]
                        .as_bool()
                        .unwrap_or(false);
                let locality = if retained { 1.0e12 } else { 0.0 };
                locality + cached - queue + capacity_bias
            })
            .collect::<Vec<_>>();

        if supports(ctx, "actions", "pie.kv.prefetch@1")
            && let Some(target) = candidates
                .first()
                .and_then(|candidate| candidate["id"].as_str())
        {
            host.prefetch_kv(&request_id, target)?;
        }
        Ok(json!({"scores": scores}))
    }

    fn admit(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let request_id = request_id(ctx)?;
        let request = state.request_mut(&request_id)?;
        if request.fields["metadata"]["last_hook"] != "route" {
            return Err("admit did not observe route mutation".into());
        }
        request.scratch["admission_count"] =
            json!(request.scratch["admission_count"].as_u64().unwrap_or(0) + 1);
        request.fields["metadata"]["last_hook"] = json!("admit");
        append_prompt(request, "|admit");

        if supports(ctx, "actions", "pie.retention.set@1") {
            host.set_retention(&request_id, 5000)?;
        }
        let queue = ctx["target"]["facts"]["queue_depth"].as_u64().unwrap_or(0);
        let decision = if queue < 80 {
            "accept"
        } else if queue < 100 {
            "defer"
        } else {
            "reject"
        };
        Ok(json!({"decision": decision}))
    }

    fn schedule(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let request_ids = ctx["runnable"]
            .as_array()
            .ok_or("runnable must be an array")?
            .iter()
            .map(|candidate| {
                candidate["request_id"]
                    .as_str()
                    .ok_or("runnable request_id must be a string")
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for request_id in &request_ids {
            if state.request(request_id)?.fields["metadata"]["last_hook"] != "admit" {
                return Err("schedule did not observe admit mutation".into());
            }
        }

        let mut decisions = Vec::with_capacity(request_ids.len());
        for request_id in &request_ids {
            let request = state.request_mut(request_id)?;
            request.scratch["schedule_calls"] =
                json!(request.scratch["schedule_calls"].as_u64().unwrap_or(0) + 1);
            let enacted = request.facts()["attained_service"].as_u64().unwrap_or(0);
            let feedback = request.scratch["feedback_service"].as_u64().unwrap_or(0);
            decisions.push(json!({"score": -((enacted + feedback) as f64)}));
            request.fields["metadata"]["last_hook"] = json!("schedule");
        }
        Ok(json!({"decisions": decisions}))
    }

    fn evict(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let units = ctx["resident"]
            .as_array()
            .ok_or("resident must be an array")?
            .iter()
            .map(|unit| {
                (
                    unit["request_id"].as_str().map(str::to_owned),
                    unit["facts"]["reload_cost"].as_f64().unwrap_or(0.0),
                )
            })
            .collect::<Vec<_>>();
        let mut scores = Vec::with_capacity(units.len());
        for (request_id, reload_cost) in units {
            if let Some(request_id) = request_id {
                let request = state.request_mut(&request_id)?;
                request.scratch["eviction_checks"] =
                    json!(request.scratch["eviction_checks"].as_u64().unwrap_or(0) + 1);
            }
            scores.push(reload_cost);
        }
        Ok(json!({"scores": scores}))
    }

    fn feedback(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let records = ctx["records"]
            .as_array()
            .ok_or("records must be an array")?
            .iter()
            .map(|record| {
                Ok((
                    record["event"].as_str().unwrap_or("").to_owned(),
                    record["request_id"]
                        .as_str()
                        .ok_or("record request_id must be a string")?
                        .to_owned(),
                    record["facts"].clone(),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if supports(ctx, "actions", "pie.timer.arm@1")
            && let Some(request_id) = records.first().map(|(_, request_id, _)| request_id)
        {
            host.arm_timer(request_id, 1)?;
        }
        for (event, request_id, facts) in records {
            {
                let request = state.request_mut(&request_id)?;
                match event.as_str() {
                    "progress" => {
                        let delta = facts["committed_tokens"].as_u64().unwrap_or(0);
                        request.scratch["feedback_service"] = json!(
                            request.scratch["feedback_service"].as_u64().unwrap_or(0) + delta
                        );
                    }
                    "tool-boundary" => {
                        request.scratch["tool_calls"] =
                            json!(request.scratch["tool_calls"].as_u64().unwrap_or(0) + 1);
                        request.fields["metadata"]["last_hook"] = json!("tool-boundary");
                    }
                    "action-succeeded" => {
                        request.scratch["actions_succeeded"] =
                            json!(request.scratch["actions_succeeded"].as_u64().unwrap_or(0) + 1);
                    }
                    "action-failed" => {
                        request.scratch["actions_failed"] =
                            json!(request.scratch["actions_failed"].as_u64().unwrap_or(0) + 1);
                    }
                    _ => {}
                }
            }
            state.shared["feedback_records"] =
                json!(state.shared["feedback_records"].as_u64().unwrap_or(0) + 1);
        }
        state.shared["last_feedback_delivery"] = ctx["delivery_id"].clone();
        Ok(json!({}))
    }
}

fn request_id(ctx: &Document) -> Result<String, String> {
    ctx["request_id"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "request_id must be a string".into())
}

fn supports(ctx: &Document, kind: &str, method: &str) -> bool {
    ctx["context"]["capabilities"][kind]
        .as_array()
        .is_some_and(|methods| methods.iter().any(|candidate| candidate == method))
}

fn append_prompt(request: &mut Request, suffix: &str) {
    let prompt = request.fields["body"]["prompt"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    request.fields["body"]["prompt"] = json!(format!("{prompt}{suffix}"));
}

plex::export_policy!(Coordinated);
