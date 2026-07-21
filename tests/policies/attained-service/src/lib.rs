use plex::serde_json::json;
use plex::{Document, Policy};

struct AttainedService;

impl Policy for AttainedService {
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
        let mut decisions = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            let request = &mut input["requests"][request_id.as_str()];
            let attained = request["facts"]["attained_service"].as_u64().unwrap_or(0);
            request["scratch"]["schedule_calls"] =
                json!(request["scratch"]["schedule_calls"].as_u64().unwrap_or(0) + 1);
            decisions.push(json!({"score": -(attained as f64)}));
        }
        Ok(json!({"decisions": decisions}))
    }
}

plex::export_policy!(AttainedService);
