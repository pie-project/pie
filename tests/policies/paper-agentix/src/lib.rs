//! JSON adaptation of Agentix PLAS/ATLAS.
//! https://www.usenix.org/system/files/nsdi26-luo.pdf

use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct Agentix;

impl Policy for Agentix {
    fn schedule(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let runnable = ctx["runnable"]
            .as_array()
            .ok_or("runnable must be an array")?;
        let decisions = runnable
            .iter()
            .map(|candidate| {
                let request_id = candidate["request_id"].as_str().unwrap_or("");
                let service = state.request(request_id)?.facts()["attained_service"]
                    .as_u64()
                    .unwrap_or(0);
                let waiting = candidate["facts"]["waiting_ms"].as_u64().unwrap_or(0);
                let starved = waiting >= service.max(1).saturating_mul(4);
                let bucket = 64 - service.saturating_add(1).leading_zeros() as u64;
                Ok::<_, String>(json!({
                    "score": if starved {
                        1.0e15 - bucket as f64
                    } else {
                        -(bucket as f64)
                    }
                }))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(json!({"decisions": decisions}))
    }

    fn feedback(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let records = ctx["records"]
            .as_array()
            .ok_or("records must be an array")?
            .iter()
            .map(|record| {
                (
                    record["request_id"].as_str().unwrap_or("").to_owned(),
                    record["facts"]["service_us"].as_u64().unwrap_or(0),
                )
            })
            .collect::<Vec<_>>();
        for (request_id, service) in records {
            let request = state.request_mut(&request_id)?;
            let previous = request.scratch["observed_service"].as_u64().unwrap_or(0);
            request.scratch["observed_service"] = json!(previous + service);
        }
        Ok(json!({}))
    }
}

plex::export_policy!(Agentix);
