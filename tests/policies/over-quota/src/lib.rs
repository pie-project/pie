use plex::types::{
    MapAddU64, MapKey, MapMutation, PolicyError, ScheduleInput, ScheduleOutput, ServiceDecision,
};

struct OverQuota;

impl plex::Policy for OverQuota {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let Some(handle) = input.links.maps.first().copied().flatten() else {
            return Err(PolicyError::FallbackRequired);
        };
        let mutation = || {
            MapMutation::AddU64(MapAddU64 {
                handle,
                key: MapKey::Bytes(vec![0]),
                delta: 1,
                ttl_ms: None,
            })
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
            mutations: vec![mutation(), mutation()],
        })
    }
}

plex::export_policy!(OverQuota);
