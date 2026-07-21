use plex::pie::plex::maps;
use plex::types::{
    FeedbackInput, FeedbackOutput, MapAddU64, MapKey, MapMutation, MapValue, PolicyError,
    ScheduleInput, ScheduleOutput, ServiceDecision,
};

struct FeedbackAccounting;

impl plex::Policy for FeedbackAccounting {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let Some(handle) = input.links.maps.first().copied().flatten() else {
            return Err(PolicyError::FallbackRequired);
        };
        let debt = match maps::get(handle, &MapKey::Bytes(vec![0])) {
            Ok(Some(MapValue::Unsigned64(value))) => value,
            Ok(None) => 0,
            _ => return Err(PolicyError::FallbackRequired),
        };
        Ok(ScheduleOutput {
            decisions: input
                .runnable
                .iter()
                .enumerate()
                .map(|(index, _)| ServiceDecision {
                    score: -((debt + index as u64) as f64),
                    token_budget: None,
                })
                .collect(),
            mutations: vec![MapMutation::AddU64(MapAddU64 {
                handle,
                key: MapKey::Bytes(vec![0]),
                delta: 10,
                ttl_ms: None,
            })],
        })
    }

    fn feedback(input: FeedbackInput) -> Result<FeedbackOutput, PolicyError> {
        let Some(handle) = input.links.maps.first().copied().flatten() else {
            return Err(PolicyError::FallbackRequired);
        };
        Ok(FeedbackOutput {
            mutations: vec![MapMutation::AddU64(MapAddU64 {
                handle,
                key: MapKey::Bytes(vec![0]),
                delta: input.events.len() as u64,
                ttl_ms: None,
            })],
        })
    }
}

plex::export_policy!(FeedbackAccounting);
