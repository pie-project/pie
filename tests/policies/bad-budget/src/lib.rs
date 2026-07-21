use plex::serde_json::json;
use plex::{Document, Policy};

struct BadBudget;

impl Policy for BadBudget {
    fn schedule(input: &mut Document) -> Result<Document, String> {
        let count = input["runnable"].as_array().map_or(0, Vec::len);
        Ok(json!({
            "decisions": (0..count)
                .map(|_| json!({"score": 0.0, "token_budget": 999999}))
                .collect::<Vec<_>>()
        }))
    }
}

plex::export_policy!(BadBudget);
