use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct AttainedService;

impl Policy for AttainedService {
    fn schedule(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        state.shared["working_set_size"] = json!(state.request_ids().count());
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
        let mut decisions = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            let request = state.request_mut(&request_id)?;
            let attained = request.facts()["attained_service"].as_u64().unwrap_or(0);
            request.scratch["schedule_calls"] =
                json!(request.scratch["schedule_calls"].as_u64().unwrap_or(0) + 1);
            decisions.push(json!({"score": -(attained as f64)}));
        }
        Ok(json!({"decisions": decisions}))
    }
}

plex::export_policy!(AttainedService);
