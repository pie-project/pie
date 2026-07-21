use plex::types::{ColumnValues, DenseOutput, EvictionInput, PolicyError};

struct RetentionScore;

impl plex::Policy for RetentionScore {
    fn evict(input: EvictionInput) -> Result<DenseOutput, PolicyError> {
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
        let ColumnValues::Float64s(values) = &column.values else {
            return Err(PolicyError::FallbackRequired);
        };
        let Some(scores) = values.iter().copied().collect::<Option<Vec<_>>>() else {
            return Err(PolicyError::FallbackRequired);
        };
        Ok(DenseOutput {
            scores,
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(RetentionScore);
