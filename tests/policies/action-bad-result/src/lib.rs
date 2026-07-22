use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct ActionBadResult;

impl Policy for ActionBadResult {
    fn route(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let request_id = ctx["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?;
        state.request_mut(request_id)?.scratch["should_not_commit"] = json!(true);
        host.prefetch_kv(request_id, "node-a")?;
        Ok(json!({"scores": []}))
    }
}

plex::export_policy!(ActionBadResult);
