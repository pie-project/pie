use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct MutateFail;

impl Policy for MutateFail {
    fn route(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let request_id = ctx["request_id"].as_str().unwrap_or("");
        state.request_mut(request_id)?.scratch["should_not_commit"] = json!(true);
        state.shared["should_not_commit"] = json!(true);
        Err("fallback-required".into())
    }
}

plex::export_policy!(MutateFail);
