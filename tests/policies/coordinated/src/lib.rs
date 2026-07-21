use plex::serde_json::json;
use plex::{Document, Policy};

struct Coordinated;

impl Policy for Coordinated {
    fn route(input: &mut Document) -> Result<Document, String> {
        let route_count = input["request"]["state"]["route_count"]
            .as_u64()
            .unwrap_or(0)
            + 1;
        input["request"]["state"]["route_count"] = json!(route_count);
        input["request"]["metadata"]["last_hook"] = json!("route");
        append_prompt(&mut input["request"], "|route");

        let candidates = input["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?;
        let scores = candidates
            .iter()
            .map(|candidate| {
                candidate["facts"]["cached_tokens"].as_f64().unwrap_or(0.0)
                    - candidate["facts"]["queue_depth"].as_f64().unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }

    fn admit(input: &mut Document) -> Result<Document, String> {
        if input["request"]["metadata"]["last_hook"] != "route" {
            return Err("admit did not observe route mutation".into());
        }
        let count = input["request"]["state"]["admission_count"]
            .as_u64()
            .unwrap_or(0)
            + 1;
        input["request"]["state"]["admission_count"] = json!(count);
        input["request"]["metadata"]["last_hook"] = json!("admit");
        append_prompt(&mut input["request"], "|admit");

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
        let runnable = input["runnable"]
            .as_array_mut()
            .ok_or("runnable must be an array")?;
        let mut decisions = Vec::with_capacity(runnable.len());
        for candidate in runnable {
            if candidate["request"]["metadata"]["last_hook"] != "admit" {
                return Err("schedule did not observe admit mutation".into());
            }
            candidate["request"]["metadata"]["last_hook"] = json!("schedule");
            let calls = candidate["request"]["state"]["schedule_calls"]
                .as_u64()
                .unwrap_or(0)
                + 1;
            candidate["request"]["state"]["schedule_calls"] = json!(calls);
            let attained = candidate["request"]["state"]["attained_service"]
                .as_u64()
                .unwrap_or(0);
            decisions.push(json!({"score": -(attained as f64)}));
        }
        Ok(json!({"decisions": decisions}))
    }

    fn evict(input: &mut Document) -> Result<Document, String> {
        let resident = input["resident"]
            .as_array_mut()
            .ok_or("resident must be an array")?;
        let scores = resident
            .iter_mut()
            .map(|unit| {
                if unit["request"].is_object() {
                    let checks = unit["request"]["state"]["eviction_checks"]
                        .as_u64()
                        .unwrap_or(0)
                        + 1;
                    unit["request"]["state"]["eviction_checks"] = json!(checks);
                }
                unit["facts"]["reload_cost"].as_f64().unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }

    fn feedback(input: &mut Document) -> Result<Document, String> {
        let records = input["records"]
            .as_array_mut()
            .ok_or("records must be an array")?;
        for record in records {
            match record["event"].as_str().unwrap_or("") {
                "progress" => {
                    let delta = record["facts"]["committed_tokens"].as_u64().unwrap_or(0);
                    let attained = record["request"]["state"]["attained_service"]
                        .as_u64()
                        .unwrap_or(0);
                    record["request"]["state"]["attained_service"] = json!(attained + delta);
                }
                "tool-boundary" => {
                    let calls = record["request"]["state"]["tool_calls"]
                        .as_u64()
                        .unwrap_or(0)
                        + 1;
                    record["request"]["state"]["tool_calls"] = json!(calls);
                    record["request"]["metadata"]["last_hook"] = json!("tool-boundary");
                }
                _ => {}
            }
        }
        Ok(json!({}))
    }
}

fn append_prompt(request: &mut Document, suffix: &str) {
    let prompt = request["body"]["prompt"].as_str().unwrap_or("").to_owned();
    request["body"]["prompt"] = json!(format!("{prompt}{suffix}"));
}

plex::export_policy!(Coordinated);
