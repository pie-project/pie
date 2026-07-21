use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateCandidates;

impl Policy for MutateCandidates {
    fn route(input: &mut Document) -> Result<Document, String> {
        let count = input["candidates"].as_array().map_or(0, Vec::len);
        if count != 0 {
            input["candidates"][0]["id"] = json!("forged-target");
        }
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(MutateCandidates);
