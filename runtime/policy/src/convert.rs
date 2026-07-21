use pie_plex as contract;

use crate::bindings::pie::plex::types as wit;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ConversionError {
    #[error("policy returned a non-finite map value")]
    NonFiniteMapValue,
}

pub(crate) fn admission_input(input: &contract::AdmissionInput) -> wit::AdmissionInput {
    wit::AdmissionInput {
        links: link_set(&input.links),
        request: request_context(&input.request),
    }
}

pub(crate) fn admission_output(
    output: wit::AdmissionOutput,
) -> Result<contract::AdmissionOutput, ConversionError> {
    Ok(contract::AdmissionOutput {
        decision: match output.decision {
            wit::AdmissionDecision::Accept => contract::AdmissionDecision::Accept,
            wit::AdmissionDecision::Defer => contract::AdmissionDecision::Defer,
            wit::AdmissionDecision::Reject => contract::AdmissionDecision::Reject,
        },
        mutations: map_mutations(output.mutations)?,
    })
}

pub(crate) fn placement_input(input: &contract::PlacementInput) -> wit::PlacementInput {
    wit::PlacementInput {
        links: link_set(&input.links),
        cause: match input.cause {
            contract::PlacementCause::GenerationArrival => wit::PlacementCause::GenerationArrival,
            contract::PlacementCause::StageTransition => wit::PlacementCause::StageTransition,
            contract::PlacementCause::Continuation => wit::PlacementCause::Continuation,
        },
        request: request_context(&input.request),
        placement_count: u32::try_from(input.placements.candidates.len()).unwrap_or(u32::MAX),
        fields: record_batch(&input.placements.fields),
    }
}

pub(crate) fn dense_output(
    output: wit::DenseOutput,
) -> Result<contract::DenseScores, ConversionError> {
    Ok(contract::DenseScores {
        scores: output.scores,
        mutations: map_mutations(output.mutations)?,
    })
}

pub(crate) fn schedule_input(input: &contract::ScheduleInput) -> wit::ScheduleInput {
    wit::ScheduleInput {
        links: link_set(&input.links),
        cause: match input.cause {
            contract::ScheduleCause::Enqueue => wit::ScheduleCause::Enqueue,
            contract::ScheduleCause::ServiceStep => wit::ScheduleCause::ServiceStep,
        },
        runnable: input
            .runnable
            .candidates
            .iter()
            .map(|candidate| wit::ServiceCandidate {
                logical_request_id: logical_request_id(candidate.logical_request_id),
                generation_id: candidate.generation_id.get(),
                max_token_budget: candidate.max_token_budget,
            })
            .collect(),
        fields: record_batch(&input.runnable.fields),
        capacity: wit::ServiceCapacity {
            max_selected: input.capacity.max_selected,
            max_total_tokens: input.capacity.max_total_tokens,
            max_token_budget: input.capacity.max_token_budget,
        },
    }
}

pub(crate) fn eviction_input(input: &contract::EvictionInput) -> wit::EvictionInput {
    wit::EvictionInput {
        links: link_set(&input.links),
        cause: match input.cause {
            contract::EvictionCause::AllocationDeficit => wit::EvictionCause::AllocationDeficit,
            contract::EvictionCause::MemoryWatermark => wit::EvictionCause::MemoryWatermark,
        },
        bytes_needed: input.bytes_needed,
        resident: input
            .resident
            .candidates
            .iter()
            .map(|candidate| wit::ResidentUnit {
                size_bytes: candidate.size_bytes,
                logical_request_id: candidate.logical_request_id.map(logical_request_id),
                generation_id: candidate.generation_id.map(contract::GenerationId::get),
            })
            .collect(),
        fields: record_batch(&input.resident.fields),
    }
}

pub(crate) fn service_plan(
    output: wit::ScheduleOutput,
) -> Result<contract::ServicePlan, ConversionError> {
    Ok(contract::ServicePlan {
        decisions: output
            .decisions
            .into_iter()
            .map(|decision| contract::ServiceDecision {
                score: decision.score,
                token_budget: decision.token_budget,
            })
            .collect(),
        mutations: output
            .mutations
            .into_iter()
            .map(map_mutation)
            .collect::<Result<_, _>>()?,
    })
}

pub(crate) fn feedback_input(input: &contract::FeedbackBatch) -> wit::FeedbackInput {
    wit::FeedbackInput {
        links: link_set(&input.links),
        delivery_id: delivery_id(input.delivery_id),
        events: input
            .events
            .iter()
            .map(|handle| wit::EventHandle {
                value: handle.get(),
            })
            .collect(),
        subjects: input
            .subjects
            .iter()
            .map(|subject| wit::FeedbackSubject {
                logical_request_id: logical_request_id(subject.logical_request_id),
                generation_id: subject.generation_id.map(contract::GenerationId::get),
                terminal_outcome: subject.terminal_outcome.map(|outcome| match outcome {
                    contract::TerminalOutcome::Completed => wit::TerminalOutcome::Completed,
                    contract::TerminalOutcome::Cancelled => wit::TerminalOutcome::Cancelled,
                    contract::TerminalOutcome::Failed => wit::TerminalOutcome::Failed,
                }),
            })
            .collect(),
        records: record_batch(&input.records),
    }
}

pub(crate) fn feedback_output(
    output: wit::FeedbackOutput,
) -> Result<contract::FeedbackOutput, ConversionError> {
    Ok(contract::FeedbackOutput {
        mutations: output
            .mutations
            .into_iter()
            .map(map_mutation)
            .collect::<Result<_, _>>()?,
    })
}

fn link_set(links: &contract::LinkSet) -> wit::LinkSet {
    wit::LinkSet {
        facts: links
            .facts
            .iter()
            .map(|handle| {
                handle.map(|handle| wit::FactHandle {
                    value: handle.get(),
                })
            })
            .collect(),
        metadata: links
            .metadata
            .iter()
            .map(|handle| {
                handle.map(|handle| wit::MetadataHandle {
                    value: handle.get(),
                })
            })
            .collect(),
        maps: links
            .maps
            .iter()
            .map(|handle| {
                handle.map(|handle| wit::MapHandle {
                    value: handle.get(),
                })
            })
            .collect(),
        events: links
            .events
            .iter()
            .map(|handle| {
                handle.map(|handle| wit::EventHandle {
                    value: handle.get(),
                })
            })
            .collect(),
        capabilities: links
            .capabilities
            .iter()
            .map(|handle| {
                handle.map(|handle| wit::CapabilityHandle {
                    value: handle.get(),
                })
            })
            .collect(),
    }
}

fn logical_request_id(id: contract::LogicalRequestId) -> wit::LogicalRequestId {
    let bytes = id.into_bytes();
    wit::LogicalRequestId {
        high: u64::from_be_bytes(bytes[..8].try_into().expect("eight-byte high half")),
        low: u64::from_be_bytes(bytes[8..].try_into().expect("eight-byte low half")),
    }
}

fn request_context(request: &contract::RequestContext) -> wit::RequestContext {
    wit::RequestContext {
        logical_request_id: logical_request_id(request.logical_request_id),
        generation_id: request.generation_id.map(contract::GenerationId::get),
        fields: record_batch(&request.fields),
    }
}

fn delivery_id(id: contract::DeliveryId) -> wit::DeliveryId {
    let bytes = id.into_bytes();
    wit::DeliveryId {
        high: u64::from_be_bytes(bytes[..8].try_into().expect("eight-byte high half")),
        low: u64::from_be_bytes(bytes[8..].try_into().expect("eight-byte low half")),
    }
}

fn record_batch(batch: &contract::RecordBatch) -> wit::RecordBatch {
    wit::RecordBatch {
        rows: batch.rows,
        facts: batch
            .facts
            .iter()
            .map(|column| wit::FactColumn {
                handle: wit::FactHandle {
                    value: column.handle.get(),
                },
                values: column_values(&column.values),
            })
            .collect(),
        metadata: batch
            .metadata
            .iter()
            .map(|column| wit::MetadataColumn {
                handle: wit::MetadataHandle {
                    value: column.handle.get(),
                },
                values: column_values(&column.values),
            })
            .collect(),
    }
}

fn column_values(values: &contract::ColumnValues) -> wit::ColumnValues {
    match values {
        contract::ColumnValues::Bool(values) => wit::ColumnValues::Booleans(values.clone()),
        contract::ColumnValues::I64(values) => wit::ColumnValues::Signed64s(values.clone()),
        contract::ColumnValues::U64(values) => wit::ColumnValues::Unsigned64s(values.clone()),
        contract::ColumnValues::F64(values) => wit::ColumnValues::Float64s(values.clone()),
        contract::ColumnValues::String(values) => wit::ColumnValues::Texts(values.clone()),
        contract::ColumnValues::Bytes(values) => wit::ColumnValues::ByteLists(values.clone()),
    }
}

fn map_mutation(mutation: wit::MapMutation) -> Result<contract::MapMutation, ConversionError> {
    Ok(match mutation {
        wit::MapMutation::Upsert(upsert) => contract::MapMutation::Upsert {
            map: contract::MapHandle::new(upsert.handle.value),
            key: contract_map_key(upsert.key),
            value: map_value(upsert.value)?,
            ttl_ms: upsert.ttl_ms,
        },
        wit::MapMutation::AddI64(add) => contract::MapMutation::AddI64 {
            map: contract::MapHandle::new(add.handle.value),
            key: contract_map_key(add.key),
            delta: add.delta,
            ttl_ms: add.ttl_ms,
        },
        wit::MapMutation::AddU64(add) => contract::MapMutation::AddU64 {
            map: contract::MapHandle::new(add.handle.value),
            key: contract_map_key(add.key),
            delta: add.delta,
            ttl_ms: add.ttl_ms,
        },
        wit::MapMutation::Delete(delete) => contract::MapMutation::Delete {
            map: contract::MapHandle::new(delete.handle.value),
            key: contract_map_key(delete.key),
        },
    })
}

fn map_mutations(
    mutations: Vec<wit::MapMutation>,
) -> Result<Vec<contract::MapMutation>, ConversionError> {
    mutations.into_iter().map(map_mutation).collect()
}

pub(crate) fn contract_map_key(key: wit::MapKey) -> contract::MapKey {
    match key {
        wit::MapKey::Boolean(value) => contract::MapKey::Bool(value),
        wit::MapKey::Signed64(value) => contract::MapKey::I64(value),
        wit::MapKey::Unsigned64(value) => contract::MapKey::U64(value),
        wit::MapKey::Text(value) => contract::MapKey::String(value),
        wit::MapKey::Bytes(value) => contract::MapKey::Bytes(value),
    }
}

pub(crate) fn wit_map_value(value: contract::TypedValue) -> wit::MapValue {
    match value {
        contract::TypedValue::Bool(value) => wit::MapValue::Boolean(value),
        contract::TypedValue::I64(value) => wit::MapValue::Signed64(value),
        contract::TypedValue::U64(value) => wit::MapValue::Unsigned64(value),
        contract::TypedValue::F64(value) => wit::MapValue::Float64(value),
        contract::TypedValue::String(value) => wit::MapValue::Text(value),
        contract::TypedValue::Bytes(value) => wit::MapValue::Bytes(value),
    }
}

fn map_value(value: wit::MapValue) -> Result<contract::TypedValue, ConversionError> {
    let value = match value {
        wit::MapValue::Boolean(value) => contract::TypedValue::Bool(value),
        wit::MapValue::Signed64(value) => contract::TypedValue::I64(value),
        wit::MapValue::Unsigned64(value) => contract::TypedValue::U64(value),
        wit::MapValue::Float64(value) => contract::TypedValue::F64(value),
        wit::MapValue::Text(value) => contract::TypedValue::String(value),
        wit::MapValue::Bytes(value) => contract::TypedValue::Bytes(value),
    };
    value
        .validate()
        .map_err(|_| ConversionError::NonFiniteMapValue)?;
    Ok(value)
}
