//! KVFlow step-graph eviction and status-aware scheduling stress case.
//! Primary source: https://arxiv.org/abs/2507.07400

use plex::types::{
    ColumnValues, DenseOutput, EvictionInput, PolicyError, ScheduleInput, ScheduleOutput,
    ServiceDecision,
};
use plex::{LinkSetExt, RecordBatchExt};

struct KvFlow;

impl plex::Policy for KvFlow {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let ready_handle = input.links.fact(2).ok_or(PolicyError::FallbackRequired)?;
        let ColumnValues::Booleans(ready) = input
            .fields
            .fact(ready_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let Some(decisions) = ready
            .iter()
            .map(|ready| {
                ready.map(|ready| ServiceDecision {
                    score: if ready { 1.0 } else { -1.0 },
                    token_budget: None,
                })
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Err(PolicyError::FallbackRequired);
        };
        Ok(ScheduleOutput {
            decisions,
            mutations: Vec::new(),
        })
    }

    fn evict(input: EvictionInput) -> Result<DenseOutput, PolicyError> {
        let steps_handle = input.links.fact(0).ok_or(PolicyError::FallbackRequired)?;
        let fixed_handle = input.links.fact(1).ok_or(PolicyError::FallbackRequired)?;
        let ColumnValues::Unsigned64s(steps) = input
            .fields
            .fact(steps_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Booleans(fixed) = input
            .fields
            .fact(fixed_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let mut scores = Vec::with_capacity(input.resident.len());
        for index in 0..input.resident.len() {
            let steps = steps
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            let fixed = fixed
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            scores.push(if fixed { -(steps as f64) } else { -1.0e15 });
        }
        Ok(DenseOutput {
            scores,
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(KvFlow);
