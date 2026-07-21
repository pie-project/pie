//! Helium cache-aware critical-path scheduling stress case.
//! Primary source: https://arxiv.org/abs/2603.16104

use plex::types::{ColumnValues, PolicyError, ScheduleInput, ScheduleOutput, ServiceDecision};
use plex::{LinkSetExt, RecordBatchExt};

struct Helium;

impl plex::Policy for Helium {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let ready = bool_fact(&input, 0)?;
        let depth = u64_fact(&input, 1)?;
        let earliest = u64_fact(&input, 2)?;
        let reuse = u64_fact(&input, 3)?;
        let cost = u64_fact(&input, 4)?;
        let any_ready = ready.iter().any(|value| *value);
        let earliest_forced = earliest
            .iter()
            .enumerate()
            .min_by_key(|(_, value)| *value)
            .map(|(index, _)| index)
            .ok_or(PolicyError::FallbackRequired)?;
        let decisions = (0..input.runnable.len())
            .map(|index| {
                let eligible = if any_ready {
                    ready[index]
                } else {
                    index == earliest_forced
                };
                ServiceDecision {
                    score: if eligible {
                        depth[index] as f64 * 1.0e12 + reuse[index] as f64 * 1.0e6
                            - earliest[index] as f64 * 1.0e3
                            - cost[index] as f64
                    } else {
                        -1.0e18
                    },
                    token_budget: None,
                }
            })
            .collect();
        Ok(ScheduleOutput {
            decisions,
            mutations: Vec::new(),
        })
    }
}

fn bool_fact(input: &ScheduleInput, declaration: usize) -> Result<Vec<bool>, PolicyError> {
    let handle = input
        .links
        .fact(declaration)
        .ok_or(PolicyError::FallbackRequired)?;
    let ColumnValues::Booleans(values) = input
        .fields
        .fact(handle)
        .ok_or(PolicyError::FallbackRequired)?
    else {
        return Err(PolicyError::FallbackRequired);
    };
    values
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .ok_or(PolicyError::FallbackRequired)
}

fn u64_fact(input: &ScheduleInput, declaration: usize) -> Result<Vec<u64>, PolicyError> {
    let handle = input
        .links
        .fact(declaration)
        .ok_or(PolicyError::FallbackRequired)?;
    let ColumnValues::Unsigned64s(values) = input
        .fields
        .fact(handle)
        .ok_or(PolicyError::FallbackRequired)?
    else {
        return Err(PolicyError::FallbackRequired);
    };
    values
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .ok_or(PolicyError::FallbackRequired)
}

plex::export_policy!(Helium);
