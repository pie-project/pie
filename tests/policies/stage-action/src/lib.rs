use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct StageAction;

impl Policy for StageAction {
    fn route(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let request_id = ctx["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?;
        let attempts = state.request(request_id)?.scratch["action_attempts"]
            .as_u64()
            .unwrap_or(0);
        state.request_mut(request_id)?.scratch["action_attempts"] = json!(attempts + 2);
        host.prefetch_kv(request_id, "node-a")?;
        host.set_retention(request_id, 5000)?;
        match ctx["cause"].as_str() {
            Some("action-fallback") => return Err("fallback-required".into()),
            Some("action-trap") => panic!("trap after staging actions"),
            _ => {}
        }
        let count = ctx["candidates"].as_array().map_or(0, Vec::len);
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(StageAction);
