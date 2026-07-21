//! Continuum TTL-aware program-FCFS stress case.
//! Primary source: https://arxiv.org/abs/2511.02230

use plex::types::{
    ColumnValues, DenseOutput, EvictionInput, FeedbackInput, FeedbackOutput, PolicyError,
    ScheduleInput, ScheduleOutput, ServiceDecision,
};
use plex::{LinkSetExt, LogicalRequestIdExt, RecordBatchExt, U64Map};

struct Continuum;

impl plex::Policy for Continuum {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let preempted_handle = input.links.fact(0).ok_or(PolicyError::FallbackRequired)?;
        let arrival_handle = input.links.fact(1).ok_or(PolicyError::FallbackRequired)?;
        let retention = U64Map::new(input.links.map(0).ok_or(PolicyError::FallbackRequired)?);
        let ColumnValues::Booleans(preempted) = input
            .fields
            .fact(preempted_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(arrival) = input
            .fields
            .fact(arrival_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let mut decisions = Vec::with_capacity(input.runnable.len());
        for (index, candidate) in input.runnable.iter().enumerate() {
            let key = candidate.logical_request_id.to_be_bytes();
            let pinned = retention
                .get_bytes(&key)
                .map_err(|_| PolicyError::FallbackRequired)?
                .is_some();
            let preempted = preempted
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            let arrival = arrival
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            decisions.push(ServiceDecision {
                score: (u64::from(preempted) as f64) * 1.0e15 + (u64::from(pinned) as f64) * 1.0e12
                    - arrival as f64,
                token_budget: None,
            });
        }
        Ok(ScheduleOutput {
            decisions,
            mutations: Vec::new(),
        })
    }

    fn evict(input: EvictionInput) -> Result<DenseOutput, PolicyError> {
        let reload_handle = input.links.fact(2).ok_or(PolicyError::FallbackRequired)?;
        let retention = U64Map::new(input.links.map(0).ok_or(PolicyError::FallbackRequired)?);
        let ColumnValues::Float64s(reload_cost) = input
            .fields
            .fact(reload_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let mut scores = Vec::with_capacity(input.resident.len());
        for (index, candidate) in input.resident.iter().enumerate() {
            let pinned = candidate.logical_request_id.is_some_and(|id| {
                retention
                    .get_bytes(&id.to_be_bytes())
                    .ok()
                    .flatten()
                    .is_some()
            });
            let reload = reload_cost
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            scores.push(reload + if pinned { 1.0e12 } else { 0.0 });
        }
        Ok(DenseOutput {
            scores,
            mutations: Vec::new(),
        })
    }

    fn feedback(input: FeedbackInput) -> Result<FeedbackOutput, PolicyError> {
        let id_handle = input.links.fact(3).ok_or(PolicyError::FallbackRequired)?;
        let ttl_handle = input.links.fact(4).ok_or(PolicyError::FallbackRequired)?;
        let retention = U64Map::new(input.links.map(0).ok_or(PolicyError::FallbackRequired)?);
        let ColumnValues::ByteLists(ids) = input
            .records
            .fact(id_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(ttls) = input
            .records
            .fact(ttl_handle)
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
            let ttl = ttls
                .get(index)
                .copied()
                .flatten()
                .ok_or(PolicyError::FallbackRequired)?;
            mutations.push(retention.upsert_bytes(key, 1, Some(ttl)));
        }
        Ok(FeedbackOutput { mutations })
    }
}

plex::export_policy!(Continuum);
