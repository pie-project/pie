//! Preble E2 exploit/explore routing stress case.
//! Primary source: https://arxiv.org/abs/2407.00023

use plex::types::{ColumnValues, DenseOutput, PlacementInput, PolicyError};
use plex::{LinkSetExt, RecordBatchExt};

struct Preble;

impl plex::Policy for Preble {
    fn route(input: PlacementInput) -> Result<DenseOutput, PolicyError> {
        let cached_handle = input.links.fact(0).ok_or(PolicyError::FallbackRequired)?;
        let uncached_handle = input.links.fact(1).ok_or(PolicyError::FallbackRequired)?;
        let load_handle = input.links.fact(2).ok_or(PolicyError::FallbackRequired)?;
        let eviction_handle = input.links.fact(3).ok_or(PolicyError::FallbackRequired)?;
        let ColumnValues::Unsigned64s(cached) = input
            .fields
            .fact(cached_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(uncached) = input
            .fields
            .fact(uncached_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(load) = input
            .fields
            .fact(load_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(eviction) = input
            .fields
            .fact(eviction_handle)
            .ok_or(PolicyError::FallbackRequired)?
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let remaining = uncached
            .first()
            .copied()
            .flatten()
            .ok_or(PolicyError::FallbackRequired)?;
        let longest = cached
            .iter()
            .copied()
            .flatten()
            .max()
            .ok_or(PolicyError::FallbackRequired)?;
        let exploit = longest > remaining;
        let mut scores = Vec::with_capacity(cached.len());
        for index in 0..cached.len() {
            let cached = cached[index].ok_or(PolicyError::FallbackRequired)?;
            let load = load[index].ok_or(PolicyError::FallbackRequired)?;
            let eviction = eviction[index].ok_or(PolicyError::FallbackRequired)?;
            scores.push(if exploit {
                cached as f64
            } else {
                -(load.saturating_add(eviction).saturating_add(remaining) as f64)
            });
        }
        Ok(DenseOutput {
            scores,
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(Preble);
