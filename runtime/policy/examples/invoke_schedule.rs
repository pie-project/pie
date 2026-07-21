use std::collections::BTreeSet;

use pie_plex::{
    CandidateSet, ContractVersion, GenerationId, InvocationMode, LinkSet, LogicalRequestId,
    Manifest, Operation, PolicyLimits, RecordBatch, ScheduleCause, ScheduleInput, ServiceCandidate,
    ServiceCapacity,
};
use pie_policy::{AttachedPolicy, CapabilityCatalog, Invocation, PolicyEngine, PolicyEngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let component_path = std::env::args()
        .nth(1)
        .ok_or("usage: invoke_schedule <policy-component.wasm>")?;
    let component = std::fs::read(component_path)?;

    let limits = PolicyLimits {
        memory_bytes: 2 << 20,
        fuel: 1_000_000,
        deadline_ms: 50,
        input_bytes: 1 << 16,
        output_bytes: 1 << 16,
        map_calls: 16,
        map_bytes: 1 << 12,
        staged_mutations: 8,
        feedback_records: 32,
        telemetry_records: 0,
        telemetry_bytes: 0,
    };
    let manifest = Manifest {
        contract: ContractVersion::V0_1,
        package_name: "attained-service".into(),
        package_version: "0.1.0".into(),
        operations: BTreeSet::from([Operation::Schedule, Operation::Feedback]),
        invocation_mode: InvocationMode::SetDependent,
        capabilities: Vec::new(),
        facts: Vec::new(),
        metadata: Vec::new(),
        events: Vec::new(),
        maps: Vec::new(),
        limits: limits.clone(),
    };

    let engine = PolicyEngine::new(PolicyEngineConfig::default())?;
    let policy =
        AttachedPolicy::compile(engine, &component, manifest, &CapabilityCatalog::default())?;
    let input = ScheduleInput {
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
            fields: RecordBatch::empty(2),
        },
        capacity: ServiceCapacity {
            max_selected: 2,
            max_total_tokens: 16,
            max_token_budget: 8,
        },
    };

    let invoke = |input| -> Result<_, Box<dyn std::error::Error>> {
        match policy.schedule(input) {
            Invocation::Success(prepared) => Ok(prepared.commit().0),
            Invocation::Unavailable => Err("schedule operation unavailable".into()),
            Invocation::FallbackRequired(failure) => {
                Err(format!("policy invocation failed: {failure:?}").into())
            }
        }
    };
    let first = invoke(input.clone())?;
    let second = invoke(input)?;
    assert_eq!(first, second);
    assert_eq!(first.decisions.len(), 2);
    println!("{first:?}");
    Ok(())
}
