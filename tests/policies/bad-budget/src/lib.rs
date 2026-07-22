use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct BadBudget;

impl Policy for BadBudget {
    fn schedule(ctx: &Document, _state: &mut State, _host: &Host) -> Result<Document, String> {
        let count = ctx["runnable"].as_array().map_or(0, Vec::len);
        Ok(json!({
            "decisions": (0..count)
                .map(|_| json!({"score": 0.0, "token_budget": 999999}))
                .collect::<Vec<_>>()
        }))
    }
}

plex::export_policy!(BadBudget);
