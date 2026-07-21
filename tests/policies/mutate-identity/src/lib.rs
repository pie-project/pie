use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateIdentity;

impl Policy for MutateIdentity {
    fn route(input: &mut Document) -> Result<Document, String> {
        input["request"]["identity"]["generation_id"] = json!(999);
        let count = input["candidates"].as_array().map_or(0, Vec::len);
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(MutateIdentity);
