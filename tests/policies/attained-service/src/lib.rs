use plex::serde_json::json;
use plex::{Document, Policy};

struct AttainedService;

impl Policy for AttainedService {
    fn schedule(input: &mut Document) -> Result<Document, String> {
        let runnable = input["runnable"]
            .as_array_mut()
            .ok_or("runnable must be an array")?;
        let decisions = runnable
            .iter_mut()
            .map(|candidate| {
                let attained = candidate["request"]["state"]["attained_service"]
                    .as_u64()
                    .unwrap_or(0);
                candidate["request"]["state"]["schedule_calls"] = json!(
                    candidate["request"]["state"]["schedule_calls"]
                        .as_u64()
                        .unwrap_or(0)
                        + 1
                );
                json!({"score": -(attained as f64)})
            })
            .collect::<Vec<_>>();
        Ok(json!({"decisions": decisions}))
    }
}

plex::export_policy!(AttainedService);
