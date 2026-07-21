//! Agentix PLAS/ATLAS policy stress case.
//! Primary source: https://www.usenix.org/system/files/nsdi26-luo.pdf

use plex::types::{
    ColumnValues, FeedbackInput, FeedbackOutput, PolicyError, ScheduleInput, ScheduleOutput,
    ServiceDecision,
};
use plex::{LinkSetExt, LogicalRequestIdExt, RecordBatchExt, U64Map};

struct Agentix;

impl plex::Policy for Agentix {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let wait_handle = input.links.fact(0).ok_or(PolicyError::FallbackRequired)?;
        let map = U64Map::new(input.links.map(0).ok_or(PolicyError::FallbackRequired)?);
        let ColumnValues::Unsigned64s(waiting) = input
            .fields
            .fact(wait_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let mut decisions = Vec::with_capacity(input.runnable.len());
        for (index, candidate) in input.runnable.iter().enumerate() {
            let key = candidate.logical_request_id.to_be_bytes();
            let service = map
                .get_bytes(&key)
                .map_err(|_| PolicyError::FallbackRequired)?
                .unwrap_or(0);
            let wait = waiting
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            let starved = wait >= service.max(1).saturating_mul(4);
            let bucket = 64 - service.saturating_add(1).leading_zeros() as u64;
            decisions.push(ServiceDecision {
                score: if starved {
                    1.0e15 - bucket as f64
                } else {
                    -(bucket as f64)
                },
                token_budget: None,
            });
        }
        Ok(ScheduleOutput {
            decisions,
            mutations: Vec::new(),
        })
    }

    fn feedback(input: FeedbackInput) -> Result<FeedbackOutput, PolicyError> {
        let id_handle = input.links.fact(1).ok_or(PolicyError::FallbackRequired)?;
        let service_handle = input.links.fact(2).ok_or(PolicyError::FallbackRequired)?;
        let map = U64Map::new(input.links.map(0).ok_or(PolicyError::FallbackRequired)?);
        let ColumnValues::ByteLists(ids) = input
            .records
            .fact(id_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(service) = input
            .records
            .fact(service_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let mut mutations = Vec::with_capacity(input.events.len());
        for index in 0..input.events.len() {
            let key = ids
                .get(index)
                .cloned()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            let delta = service
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            mutations.push(map.add_bytes(key, delta, None));
        }
        Ok(FeedbackOutput { mutations })
    }
}

plex::export_policy!(Agentix);
