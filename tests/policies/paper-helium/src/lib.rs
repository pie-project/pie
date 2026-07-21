//! JSON adaptation of Helium cache-aware critical-path scheduling.
//! https://arxiv.org/abs/2603.16104

use plex::serde_json::json;
use plex::{Document, Policy};

struct Helium;

impl Policy for Helium {
    fn schedule(input: &mut Document) -> Result<Document, String> {
        let runnable = input["runnable"]
            .as_array()
            .ok_or("runnable must be an array")?;
        let any_ready = runnable
            .iter()
            .any(|candidate| candidate["facts"]["ready"].as_bool().unwrap_or(false));
        let forced = runnable
            .iter()
            .enumerate()
            .min_by_key(|(_, candidate)| {
                candidate["facts"]["earliest_start"]
                    .as_u64()
                    .unwrap_or(u64::MAX)
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let decisions = runnable
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let ready = candidate["facts"]["ready"].as_bool().unwrap_or(false);
                let eligible = if any_ready { ready } else { index == forced };
                let score = if eligible {
                    candidate["facts"]["dependency_depth"]
                        .as_f64()
                        .unwrap_or(0.0)
                        * 1.0e12
                        + candidate["facts"]["prefix_reuse_tokens"]
                            .as_f64()
                            .unwrap_or(0.0)
                            * 1.0e6
                        - candidate["facts"]["earliest_start"].as_f64().unwrap_or(0.0) * 1.0e3
                        - candidate["facts"]["profiled_token_cost"]
                            .as_f64()
                            .unwrap_or(0.0)
                } else {
                    -1.0e18
                };
                json!({"score": score})
            })
            .collect::<Vec<_>>();
        Ok(json!({"decisions": decisions}))
    }
}

plex::export_policy!(Helium);
