//! JSON adaptation of Agentix PLAS/ATLAS.
//! https://www.usenix.org/system/files/nsdi26-luo.pdf

use plex::serde_json::json;
use plex::{Document, Policy};

struct Agentix;

impl Policy for Agentix {
    fn schedule(input: &mut Document) -> Result<Document, String> {
        let runnable = input["runnable"]
            .as_array()
            .ok_or("runnable must be an array")?;
        let decisions = runnable
            .iter()
            .map(|candidate| {
                let service = candidate["request"]["state"]["attained_service"]
                    .as_u64()
                    .unwrap_or(0);
                let waiting = candidate["facts"]["waiting_ms"].as_u64().unwrap_or(0);
                let starved = waiting >= service.max(1).saturating_mul(4);
                let bucket = 64 - service.saturating_add(1).leading_zeros() as u64;
                json!({
                    "score": if starved {
                        1.0e15 - bucket as f64
                    } else {
                        -(bucket as f64)
                    }
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({"decisions": decisions}))
    }

    fn feedback(input: &mut Document) -> Result<Document, String> {
        for record in input["records"]
            .as_array_mut()
            .ok_or("records must be an array")?
        {
            let service = record["facts"]["service_us"].as_u64().unwrap_or(0);
            let previous = record["request"]["state"]["attained_service"]
                .as_u64()
                .unwrap_or(0);
            record["request"]["state"]["attained_service"] = json!(previous + service);
        }
        Ok(json!({}))
    }
}

plex::export_policy!(Agentix);
