use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use pie_plex::{
    CandidateSet, ContractVersion, GenerationId, InvocationMode, LinkSet, LogicalRequestId,
    Manifest, MapClass, MapDeclaration, MapHandle, MapKey, MapKeyType, MapMutation, MapPersistence,
    MapSchema, Operation, PolicyLimits, RecordBatch, ScheduleCause, ScheduleInput,
    ServiceCandidate, ServiceCapacity, ServiceDecision, ServicePlan, Symbol, TypedValue, ValueType,
};
use pie_policy::{
    AttachedPolicy, CapabilityCatalog, DedupLimits, Invocation, ManualClock, MapStore,
    PolicyEngine, PolicyEngineConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let component = std::env::args()
        .nth(1)
        .ok_or("usage: benchmark <attained-service.component.wasm>")?;
    let component = std::fs::read(component)?;
    let limits = limits();
    let manifest = Manifest {
        contract: ContractVersion::V0_1,
        package_name: "benchmark".into(),
        package_version: "0.1.0".into(),
        operations: BTreeSet::from([Operation::Schedule]),
        invocation_mode: InvocationMode::SetDependent,
        capabilities: Vec::new(),
        facts: Vec::new(),
        metadata: Vec::new(),
        events: Vec::new(),
        maps: Vec::new(),
        limits: limits.clone(),
    };
    let policy = AttachedPolicy::compile(
        PolicyEngine::new(PolicyEngineConfig::default())?,
        &component,
        manifest,
        &CapabilityCatalog::default(),
    )?;
    let iterations = std::env::var("PLEX_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20)
        .max(1);

    println!(
        "candidates,materialize_ns,input_validate_ns,snapshot_ns,instantiate_ns,execute_ns,output_validate_ns,prepare_ns,total_ns"
    );
    for candidates in [8usize, 64, 256, 1024] {
        let mut materialize = 0u128;
        let mut input_validate = 0u128;
        let mut snapshot = 0u128;
        let mut instantiate = 0u128;
        let mut execute = 0u128;
        let mut output_validate = 0u128;
        let mut prepare = 0u128;
        let mut total = 0u128;
        for _ in 0..iterations {
            let started = Instant::now();
            let input = schedule_input(candidates);
            materialize += started.elapsed().as_nanos();

            let started = Instant::now();
            input.validate(&limits)?;
            input_validate += started.elapsed().as_nanos();

            let output = ServicePlan {
                decisions: (0..candidates)
                    .map(|index| ServiceDecision {
                        score: -(index as f64),
                        token_budget: None,
                    })
                    .collect(),
                mutations: Vec::new(),
            };
            let started = Instant::now();
            pie_plex::validate_service_plan(&output, &input, false, &limits)?;
            output_validate += started.elapsed().as_nanos();

            let started = Instant::now();
            let prepared = match policy.schedule(black_box(input)) {
                Invocation::Success(prepared) => prepared,
                other => return Err(format!("benchmark invocation failed: {other:?}").into()),
            };
            total += started.elapsed().as_nanos();
            let metrics = prepared.metrics();
            snapshot += u128::from(metrics.snapshot_ns);
            instantiate += u128::from(metrics.instantiate_ns);
            execute += u128::from(metrics.execute_ns);
            output_validate += u128::from(metrics.validate_ns);
            prepare += u128::from(metrics.prepare_ns);
            black_box(prepared.abort());
        }
        let divisor = u128::from(iterations);
        println!(
            "{candidates},{},{},{},{},{},{},{},{}",
            materialize / divisor,
            input_validate / divisor,
            snapshot / divisor,
            instantiate / divisor,
            execute / divisor,
            output_validate / divisor,
            prepare / divisor,
            total / divisor,
        );
    }

    println!("map_commit_ns,{}", benchmark_commit(iterations)?);
    Ok(())
}

fn limits() -> PolicyLimits {
    PolicyLimits {
        memory_bytes: 4 << 20,
        fuel: 2_000_000,
        deadline_ms: 100,
        input_bytes: 1 << 20,
        output_bytes: 1 << 20,
        map_calls: 32,
        map_bytes: 1 << 16,
        staged_mutations: 8,
        feedback_records: 32,
        telemetry_records: 0,
        telemetry_bytes: 0,
    }
}

fn schedule_input(count: usize) -> ScheduleInput {
    let rows = u32::try_from(count).expect("benchmark count fits u32");
    ScheduleInput {
        links: LinkSet::default(),
        cause: ScheduleCause::ServiceStep,
        runnable: CandidateSet {
            candidates: (0..count)
                .map(|index| ServiceCandidate {
                    logical_request_id: LogicalRequestId::new((index as u128).to_be_bytes()),
                    generation_id: GenerationId::new(index as u64),
                    max_token_budget: 8,
                })
                .collect(),
            fields: RecordBatch::empty(rows),
        },
        capacity: ServiceCapacity {
            max_selected: rows,
            max_total_tokens: rows.saturating_mul(8),
            max_token_budget: 8,
        },
    }
}

fn benchmark_commit(iterations: u32) -> Result<u128, Box<dyn std::error::Error>> {
    let handle = MapHandle::new(0);
    let store = MapStore::new(
        [(
            handle,
            MapDeclaration {
                name: Symbol::new("benchmark.state@1")?,
                class: MapClass::PolicyOwned {
                    persistence: MapPersistence::Attachment,
                },
                schema: MapSchema {
                    key_type: MapKeyType::Bytes,
                    value_type: ValueType::U64,
                    max_entries: 1,
                    max_key_bytes: 8,
                    max_value_bytes: 8,
                    default_ttl_ms: None,
                    max_ttl_ms: None,
                },
            },
        )],
        Arc::new(ManualClock::default()),
        DedupLimits::default(),
    )?;
    let mut elapsed = 0u128;
    for value in 0..iterations {
        let mut transaction = store.begin()?;
        transaction.stage(MapMutation::Upsert {
            map: handle,
            key: MapKey::Bytes(vec![0]),
            value: TypedValue::U64(u64::from(value)),
            ttl_ms: None,
        })?;
        let prepared = transaction.prepare()?;
        let started = Instant::now();
        black_box(prepared.commit());
        elapsed += started.elapsed().as_nanos();
    }
    Ok(elapsed / u128::from(iterations))
}
