use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateFeedbackFacts;

impl Policy for MutateFeedbackFacts {
    fn feedback(input: &mut Document) -> Result<Document, String> {
        if !input["records"].as_array().is_none_or(Vec::is_empty) {
            input["records"][0]["facts"]["committed_tokens"] = json!(0);
        }
        Ok(json!({}))
    }
}

plex::export_policy!(MutateFeedbackFacts);
