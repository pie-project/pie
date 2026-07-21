//! JSON adaptation of Continuum TTL-aware program-FCFS scheduling.
//! https://arxiv.org/abs/2511.02230

use plex::serde_json::json;
use plex::{Document, Policy};

struct Continuum;

impl Policy for Continuum {
    fn schedule(input: &mut Document) -> Result<Document, String> {
        let runnable = input["runnable"]
            .as_array()
            .ok_or("runnable must be an array")?;
        let decisions = runnable
            .iter()
            .map(|candidate| {
                let preempted = candidate["facts"]["preempted"].as_bool().unwrap_or(false);
                let request_id = candidate["request_id"].as_str().unwrap_or("");
                let pinned = input["requests"][request_id]["scratch"]["ttl_active"]
                    .as_bool()
                    .unwrap_or(false);
                let arrival = candidate["facts"]["program_arrival"]
                    .as_u64()
                    .unwrap_or(u64::MAX);
                json!({
                    "score": (u64::from(preempted) as f64) * 1.0e15
                        + (u64::from(pinned) as f64) * 1.0e12
                        - arrival as f64
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({"decisions": decisions}))
    }

    fn evict(input: &mut Document) -> Result<Document, String> {
        let resident = input["resident"]
            .as_array()
            .ok_or("resident must be an array")?;
        let scores = resident
            .iter()
            .map(|unit| {
                let reload = unit["facts"]["reload_cost"].as_f64().unwrap_or(0.0);
                let pinned = unit["request_id"]
                    .as_str()
                    .and_then(|request_id| {
                        input["requests"][request_id]["scratch"]["ttl_active"].as_bool()
                    })
                    .unwrap_or(false);
                reload + if pinned { 1.0e12 } else { 0.0 }
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }

    fn feedback(input: &mut Document) -> Result<Document, String> {
        let records = input["records"]
            .as_array()
            .ok_or("records must be an array")?
            .iter()
            .map(|record| {
                (
                    record["event"].as_str().unwrap_or("").to_owned(),
                    record["request_id"].as_str().unwrap_or("").to_owned(),
                    record["facts"]["ttl_ms"].as_u64().unwrap_or(0),
                )
            })
            .collect::<Vec<_>>();
        for (event, request_id, ttl_ms) in records {
            if event == "tool-boundary" {
                input["requests"][request_id.as_str()]["scratch"]["ttl_active"] =
                    json!(ttl_ms != 0);
            } else if event == "ttl-expired" {
                input["requests"][request_id.as_str()]["scratch"]["ttl_active"] = json!(false);
            }
        }
        Ok(json!({}))
    }
}

plex::export_policy!(Continuum);
