use plex::pie::plex::maps;
use plex::types::{
    AdmissionDecision, AdmissionInput, AdmissionOutput, DenseOutput, EvictionInput, FeedbackInput,
    FeedbackOutput, MapAddU64, MapKey, MapMutation, MapValue, PlacementInput, PolicyError,
    ScheduleInput, ScheduleOutput, ServiceDecision,
};

struct Coordinated;

fn state(
    input_maps: &[Option<plex::types::MapHandle>],
) -> Result<(plex::types::MapHandle, u64), PolicyError> {
    let Some(handle) = input_maps.first().copied().flatten() else {
        return Err(PolicyError::FallbackRequired);
    };
    let value = match maps::get(handle, &MapKey::Bytes(vec![0])) {
        Ok(Some(MapValue::Unsigned64(value))) => value,
        Ok(None) => 0,
        _ => return Err(PolicyError::FallbackRequired),
    };
    Ok((handle, value))
}

fn add(handle: plex::types::MapHandle, delta: u64) -> MapMutation {
    MapMutation::AddU64(MapAddU64 {
        handle,
        key: MapKey::Bytes(vec![0]),
        delta,
        ttl_ms: None,
    })
}

impl plex::Policy for Coordinated {
    fn admit(input: AdmissionInput) -> Result<AdmissionOutput, PolicyError> {
        let (handle, value) = state(&input.links.maps)?;
        Ok(AdmissionOutput {
            decision: if value < 100 {
                AdmissionDecision::Accept
            } else {
                AdmissionDecision::Defer
            },
            mutations: vec![add(handle, 1)],
        })
    }

    fn route(input: PlacementInput) -> Result<DenseOutput, PolicyError> {
        let (handle, value) = state(&input.links.maps)?;
        Ok(DenseOutput {
            scores: (0..input.placement_count)
                .map(|index| -((value + index as u64) as f64))
                .collect(),
            mutations: vec![add(handle, 2)],
        })
    }

    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let (handle, value) = state(&input.links.maps)?;
        Ok(ScheduleOutput {
            decisions: input
                .runnable
                .iter()
                .enumerate()
                .map(|(index, _)| ServiceDecision {
                    score: -((value + index as u64) as f64),
                    token_budget: None,
                })
                .collect(),
            mutations: vec![add(handle, 3)],
        })
    }

    fn evict(input: EvictionInput) -> Result<DenseOutput, PolicyError> {
        let (handle, value) = state(&input.links.maps)?;
        Ok(DenseOutput {
            scores: input
                .resident
                .iter()
                .enumerate()
                .map(|(index, _)| (value + index as u64) as f64)
                .collect(),
            mutations: vec![add(handle, 4)],
        })
    }

    fn feedback(input: FeedbackInput) -> Result<FeedbackOutput, PolicyError> {
        let (handle, _) = state(&input.links.maps)?;
        Ok(FeedbackOutput {
            mutations: vec![add(handle, input.events.len() as u64)],
        })
    }
}

plex::export_policy!(Coordinated);
