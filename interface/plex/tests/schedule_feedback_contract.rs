use std::collections::BTreeSet;

use pie_plex::{
    CandidateSet, ColumnValues, ContractVersion, DeliveryId, DependencyRequirement,
    EventDeclaration, EventHandle, FactColumn, FactDeclaration, FactHandle, FeedbackBatch,
    FeedbackSubject, FieldLocation, FieldUse, GenerationId, InvocationMode, LinkSet,
    LogicalRequestId, Manifest, MapClass, MapDeclaration, MapKeyType, MapPersistence, MapSchema,
    MetadataColumn, MetadataDeclaration, MetadataHandle, MetadataScope, Operation, PolicyLimits,
    RecordBatch, ScheduleCause, ScheduleInput, ServiceCandidate, ServiceCapacity, ServiceDecision,
    ServicePlan, Symbol, ValueType, validate_service_plan,
};

fn limits() -> PolicyLimits {
    PolicyLimits {
        memory_bytes: 1 << 20,
        fuel: 100_000,
        deadline_ms: 10,
        input_bytes: 1 << 16,
        output_bytes: 1 << 16,
        map_calls: 64,
        map_bytes: 1 << 14,
        staged_mutations: 16,
        feedback_records: 64,
        telemetry_records: 0,
        telemetry_bytes: 0,
    }
}

#[test]
fn schedule_and_feedback_share_a_typed_contract() {
    let manifest = Manifest {
        contract: ContractVersion::V0_1,
        package_name: "attained-service".into(),
        package_version: "0.1.0".into(),
        operations: BTreeSet::from([Operation::Schedule, Operation::Feedback]),
        invocation_mode: InvocationMode::SetDependent,
        capabilities: Vec::new(),
        facts: vec![FactDeclaration {
            name: Symbol::new("pie.attained-service@1").unwrap(),
            value_type: ValueType::U64,
            requirement: DependencyRequirement::Required,
            max_value_bytes: 8,
            uses: BTreeSet::from([FieldUse {
                operation: Operation::Schedule,
                location: FieldLocation::Candidate,
            }]),
        }],
        metadata: vec![MetadataDeclaration {
            name: Symbol::new("acme.expected-output-tokens@1").unwrap(),
            value_type: ValueType::U64,
            scope: MetadataScope::Generation,
            requirement: DependencyRequirement::Optional,
            max_value_bytes: 8,
            uses: BTreeSet::from([FieldUse {
                operation: Operation::Schedule,
                location: FieldLocation::Candidate,
            }]),
        }],
        events: vec![EventDeclaration {
            name: Symbol::new("pie.progress@1").unwrap(),
            requirement: DependencyRequirement::Required,
        }],
        maps: vec![MapDeclaration {
            name: Symbol::new("policy.accounting@1").unwrap(),
            class: MapClass::PolicyOwned {
                persistence: MapPersistence::Attachment,
            },
            schema: MapSchema {
                key_type: MapKeyType::Bytes,
                value_type: ValueType::U64,
                max_entries: 1024,
                max_key_bytes: 16,
                max_value_bytes: 8,
                default_ttl_ms: None,
                max_ttl_ms: None,
            },
        }],
        limits: limits(),
    };
    manifest.validate().unwrap();

    let schedule = ScheduleInput {
        links: LinkSet::default(),
        cause: ScheduleCause::ServiceStep,
        runnable: CandidateSet {
            candidates: vec![
                ServiceCandidate {
                    logical_request_id: LogicalRequestId::new([1; 16]),
                    generation_id: GenerationId::new(0),
                    max_token_budget: 8,
                },
                ServiceCandidate {
                    logical_request_id: LogicalRequestId::new([2; 16]),
                    generation_id: GenerationId::new(0),
                    max_token_budget: 8,
                },
            ],
            fields: RecordBatch {
                rows: 2,
                facts: vec![FactColumn {
                    handle: FactHandle::new(0),
                    values: ColumnValues::U64(vec![Some(128), Some(64)]),
                }],
                metadata: vec![MetadataColumn {
                    handle: MetadataHandle::new(0),
                    values: ColumnValues::U64(vec![Some(32), None]),
                }],
            },
        },
        capacity: ServiceCapacity {
            max_selected: 2,
            max_total_tokens: 16,
            max_token_budget: 8,
        },
    };
    let policy_limits = limits();
    schedule.validate(&policy_limits).unwrap();

    let plan = ServicePlan {
        decisions: vec![
            ServiceDecision {
                score: -128.0,
                token_budget: Some(8),
            },
            ServiceDecision {
                score: -64.0,
                token_budget: Some(8),
            },
        ],
        mutations: Vec::new(),
    };
    validate_service_plan(&plan, &schedule, true, &policy_limits).unwrap();

    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([3; 16]),
        events: vec![EventHandle::new(0), EventHandle::new(0)],
        subjects: vec![
            FeedbackSubject {
                logical_request_id: LogicalRequestId::new([1; 16]),
                generation_id: Some(GenerationId::new(0)),
                terminal_outcome: None,
            },
            FeedbackSubject {
                logical_request_id: LogicalRequestId::new([2; 16]),
                generation_id: Some(GenerationId::new(0)),
                terminal_outcome: None,
            },
        ],
        records: RecordBatch {
            rows: 2,
            facts: vec![FactColumn {
                handle: FactHandle::new(0),
                values: ColumnValues::U64(vec![Some(8), Some(8)]),
            }],
            metadata: Vec::new(),
        },
    };
    feedback.validate(&policy_limits).unwrap();
}
