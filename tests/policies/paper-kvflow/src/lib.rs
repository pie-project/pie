//! JSON adaptation of KVFlow workflow-aware cache policy.
//! https://arxiv.org/abs/2507.07400

use plex::serde_json::json;
use plex::{Document, Policy};

struct KvFlow;

impl Policy for KvFlow {
    fn schedule(input: &mut Document) -> Result<Document, String> {
        let decisions = input["runnable"]
            .as_array()
            .ok_or("runnable must be an array")?
            .iter()
            .map(|candidate| {
                json!({
                    "score": if candidate["facts"]["cache_ready"]
                        .as_bool()
                        .unwrap_or(false)
                    {
                        1.0
                    } else {
                        -1.0
                    }
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({"decisions": decisions}))
    }

    fn evict(input: &mut Document) -> Result<Document, String> {
        let scores = input["resident"]
            .as_array()
            .ok_or("resident must be an array")?
            .iter()
            .map(|unit| {
                if unit["facts"]["fixed_prefix"].as_bool().unwrap_or(false) {
                    -(unit["facts"]["steps_to_execution"]
                        .as_u64()
                        .unwrap_or(u64::MAX) as f64)
                } else {
                    -1.0e15
                }
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(KvFlow);
