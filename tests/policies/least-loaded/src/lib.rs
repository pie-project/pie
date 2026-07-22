use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct LeastLoaded;

impl Policy for LeastLoaded {
    fn route(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        state.shared["route_owner_calls"] =
            json!(state.shared["route_owner_calls"].as_u64().unwrap_or(0) + 1);
        let request_id = ctx["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?;
        let previous_target = state.request(request_id)?.facts()["previous_target"]
            .as_str()
            .map(str::to_owned);
        let candidates = ctx["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?;
        let scores = candidates
            .iter()
            .map(|candidate| {
                let queue = candidate["facts"]["queue_depth"].as_u64().unwrap_or(0) as f64;
                let cached = candidate["facts"]["cached_tokens"].as_u64().unwrap_or(0) as f64;
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
}

plex::export_policy!(LeastLoaded);
