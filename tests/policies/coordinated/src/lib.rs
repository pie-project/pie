use plex::serde_json::json;
use plex::{Document, Policy};

struct Coordinated;

impl Policy for Coordinated {
    fn route(input: &mut Document) -> Result<Document, String> {
        let request_id = request_id(input)?;
        {
            let request = &mut input["requests"][request_id.as_str()];
            request["scratch"]["route_count"] =
                json!(request["scratch"]["route_count"].as_u64().unwrap_or(0) + 1);
            request["fields"]["metadata"]["last_hook"] = json!("route");
            append_prompt(request, "|route");
        }
        input["global"]["scratch"]["route_calls"] = json!(
            input["global"]["scratch"]["route_calls"]
                .as_u64()
                .unwrap_or(0)
                + 1
        );
        input["global"]["fields"]["last_route_request"] = json!(request_id);

        let previous_target = input["requests"][request_id.as_str()]["facts"]["previous_target"]
            .as_str()
            .map(str::to_owned);
        let candidates = input["candidates"]
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
                locality + cached - queue
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }

    fn admit(input: &mut Document) -> Result<Document, String> {
        let request_id = request_id(input)?;
        let request = &mut input["requests"][request_id.as_str()];
        if request["fields"]["metadata"]["last_hook"] != "route" {
            return Err("admit did not observe route mutation".into());
        }
        request["scratch"]["admission_count"] =
            json!(request["scratch"]["admission_count"].as_u64().unwrap_or(0) + 1);
        request["fields"]["metadata"]["last_hook"] = json!("admit");
        append_prompt(request, "|admit");

        let queue = input["target"]["facts"]["queue_depth"]
            .as_u64()
            .unwrap_or(0);
        let decision = if queue < 80 {
            "accept"
        } else if queue < 100 {
            "defer"
        } else {
            "reject"
        };
        Ok(json!({"decision": decision}))
    }

    fn schedule(input: &mut Document) -> Result<Document, String> {
        let request_ids = input["runnable"]
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
            if input["requests"][request_id.as_str()]["fields"]["metadata"]["last_hook"] != "admit"
            {
                return Err("schedule did not observe admit mutation".into());
            }
        }

        let mut decisions = Vec::with_capacity(request_ids.len());
        for request_id in &request_ids {
            let request = &mut input["requests"][request_id.as_str()];
            request["scratch"]["schedule_calls"] =
                json!(request["scratch"]["schedule_calls"].as_u64().unwrap_or(0) + 1);
            let enacted = request["facts"]["attained_service"].as_u64().unwrap_or(0);
            let feedback = request["scratch"]["feedback_service"].as_u64().unwrap_or(0);
            decisions.push(json!({"score": -((enacted + feedback) as f64)}));
        }
        for request_id in request_ids {
            input["requests"][request_id.as_str()]["fields"]["metadata"]["last_hook"] =
                json!("schedule");
        }
        Ok(json!({"decisions": decisions}))
    }

    fn evict(input: &mut Document) -> Result<Document, String> {
        let units = input["resident"]
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
                let request = &mut input["requests"][request_id.as_str()];
                request["scratch"]["eviction_checks"] =
                    json!(request["scratch"]["eviction_checks"].as_u64().unwrap_or(0) + 1);
            }
            scores.push(reload_cost);
        }
        Ok(json!({"scores": scores}))
    }

    fn feedback(input: &mut Document) -> Result<Document, String> {
        let records = input["records"]
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
        for (event, request_id, facts) in records {
            let request = &mut input["requests"][request_id.as_str()];
            match event.as_str() {
                "progress" => {
                    let delta = facts["committed_tokens"].as_u64().unwrap_or(0);
                    request["scratch"]["feedback_service"] =
                        json!(request["scratch"]["feedback_service"].as_u64().unwrap_or(0) + delta);
                }
                "tool-boundary" => {
                    request["scratch"]["tool_calls"] =
                        json!(request["scratch"]["tool_calls"].as_u64().unwrap_or(0) + 1);
                    request["fields"]["metadata"]["last_hook"] = json!("tool-boundary");
                }
                _ => {}
            }
            input["global"]["scratch"]["feedback_records"] = json!(
                input["global"]["scratch"]["feedback_records"]
                    .as_u64()
                    .unwrap_or(0)
                    + 1
            );
        }
        input["global"]["fields"]["last_feedback_delivery"] = input["delivery_id"].clone();
        Ok(json!({}))
    }
}

fn request_id(input: &Document) -> Result<String, String> {
    input["request_id"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "request_id must be a string".into())
}

fn append_prompt(request: &mut Document, suffix: &str) {
    let prompt = request["fields"]["body"]["prompt"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    request["fields"]["body"]["prompt"] = json!(format!("{prompt}{suffix}"));
}

plex::export_policy!(Coordinated);
