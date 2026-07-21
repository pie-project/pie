use core::sync::atomic::{AtomicU32, Ordering};

use plex::types::{
    MapAddU64, MapKey, MapMutation, PolicyError, ScheduleInput, ScheduleOutput, ServiceDecision,
};

static CALLS_IN_THIS_INSTANCE: AtomicU32 = AtomicU32::new(0);

struct RetryFresh;

impl plex::Policy for RetryFresh {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        if CALLS_IN_THIS_INSTANCE.fetch_add(1, Ordering::SeqCst) != 0 {
            return Err(PolicyError::FallbackRequired);
        }
        let Some(handle) = input.links.maps.first().copied().flatten() else {
            return Err(PolicyError::FallbackRequired);
        };
        Ok(ScheduleOutput {
            decisions: input
                .runnable
                .iter()
                .map(|_| ServiceDecision {
                    score: 0.0,
                    token_budget: None,
                })
                .collect(),
            mutations: vec![MapMutation::AddU64(MapAddU64 {
                handle,
                key: MapKey::Bytes(vec![0]),
                delta: 1,
                ttl_ms: None,
            })],
        })
    }
}

plex::export_policy!(RetryFresh);
