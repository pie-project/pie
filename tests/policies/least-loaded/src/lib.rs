use plex::types::{ColumnValues, DenseOutput, PlacementInput, PolicyError};

struct LeastLoaded;

impl plex::Policy for LeastLoaded {
    fn route(input: PlacementInput) -> Result<DenseOutput, PolicyError> {
        let Some(handle) = input.links.facts.first().copied().flatten() else {
            return Err(PolicyError::FallbackRequired);
        };
        let Some(column) = input
            .fields
            .facts
            .iter()
            .find(|column| column.handle.value == handle.value)
        else {
            return Err(PolicyError::FallbackRequired);
        };
        let ColumnValues::Unsigned64s(loads) = &column.values else {
            return Err(PolicyError::FallbackRequired);
        };
        let Some(scores) = loads
            .iter()
            .map(|load| load.map(|load| -(load as f64)))
            .collect::<Option<Vec<_>>>()
        else {
            return Err(PolicyError::FallbackRequired);
        };
        Ok(DenseOutput {
            scores,
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(LeastLoaded);
