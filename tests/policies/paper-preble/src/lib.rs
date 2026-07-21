//! JSON adaptation of Preble E2 routing.
//! https://arxiv.org/abs/2407.00023

use plex::serde_json::json;
use plex::{Document, Policy};

struct Preble;

impl Policy for Preble {
    fn route(input: &mut Document) -> Result<Document, String> {
        let candidates = input["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?;
        let remaining = candidates
            .first()
            .and_then(|candidate| candidate["facts"]["uncached_tokens"].as_u64())
            .unwrap_or(0);
        let longest = candidates
            .iter()
            .filter_map(|candidate| candidate["facts"]["cached_tokens"].as_u64())
            .max()
            .unwrap_or(0);
        let exploit = longest > remaining;
        let scores = candidates
            .iter()
            .map(|candidate| {
                let cached = candidate["facts"]["cached_tokens"].as_u64().unwrap_or(0);
                let load = candidate["facts"]["load_cost"].as_u64().unwrap_or(0);
                let eviction = candidate["facts"]["eviction_cost"].as_u64().unwrap_or(0);
                if exploit {
                    cached as f64
                } else {
                    -(load.saturating_add(eviction).saturating_add(remaining) as f64)
                }
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(Preble);
