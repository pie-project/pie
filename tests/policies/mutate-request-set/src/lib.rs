use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateRequestSet;

impl Policy for MutateRequestSet {
    fn route(input: &mut Document) -> Result<Document, String> {
        input["requests"]["forged"] = json!({
            "facts": {"logical_request_id": "forged", "generation_id": 0},
            "fields": {},
            "scratch": {}
        });
        let count = input["candidates"].as_array().map_or(0, Vec::len);
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(MutateRequestSet);
