use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateCandidateFacts;

impl Policy for MutateCandidateFacts {
    fn route(input: &mut Document) -> Result<Document, String> {
        let count = input["candidates"].as_array().map_or(0, Vec::len);
        if count != 0 {
            input["candidates"][0]["facts"]["queue_depth"] = json!(0);
        }
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(MutateCandidateFacts);
