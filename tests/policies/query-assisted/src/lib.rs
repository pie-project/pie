use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct QueryAssisted;

impl Policy for QueryAssisted {
    fn route(ctx: &Document, _state: &mut State, host: &Host) -> Result<Document, String> {
        let observation = host.cluster_capacity(ctx["context"]["model"].as_str().unwrap_or(""))?;
        let bias = observation["route_bias"].as_f64().unwrap_or(0.0);
        let scores = ctx["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?
            .iter()
            .map(|candidate| bias - candidate["facts"]["queue_depth"].as_f64().unwrap_or(0.0))
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(QueryAssisted);
