use plex::types::{PolicyError, ScheduleInput, ScheduleOutput, ServiceDecision};
use plex::{LinkSetExt, U64Map};

struct ExternalWeight;

impl plex::Policy for ExternalWeight {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let map = U64Map::new(input.links.map(0).ok_or(PolicyError::FallbackRequired)?);
        let weight = map
            .get_bytes(&[0])
            .map_err(|_| PolicyError::FallbackRequired)?
            .ok_or(PolicyError::FallbackRequired)?;
        Ok(ScheduleOutput {
            decisions: input
                .runnable
                .iter()
                .enumerate()
                .map(|(index, _)| ServiceDecision {
                    score: weight as f64 - index as f64,
                    token_budget: None,
                })
                .collect(),
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(ExternalWeight);
