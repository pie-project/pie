use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateGlobalFacts;

impl Policy for MutateGlobalFacts {
    fn route(input: &mut Document) -> Result<Document, String> {
        input["global"]["facts"]["forged"] = json!(true);
        let count = input["candidates"].as_array().map_or(0, Vec::len);
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(MutateGlobalFacts);
