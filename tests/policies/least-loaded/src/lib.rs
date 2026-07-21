use plex::serde_json::json;
use plex::{Document, Policy};

struct LeastLoaded;

impl Policy for LeastLoaded {
    fn route(input: &mut Document) -> Result<Document, String> {
        let candidates = input["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?;
        let scores = candidates
            .iter()
            .map(|candidate| -(candidate["facts"]["queue_depth"].as_u64().unwrap_or(0) as f64))
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(LeastLoaded);
