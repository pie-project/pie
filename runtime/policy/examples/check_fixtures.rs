use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

use pie_plex::{
    AdmissionDecision, AdmissionInput, CandidateSet, ColumnValues, ContractVersion, DeliveryId,
    DependencyRequirement, EventDeclaration, EventHandle, EvictionCause, EvictionInput, FactColumn,
    FactDeclaration, FeedbackAcknowledgement, FeedbackBatch, FeedbackSubject, FieldLocation,
    FieldUse, GenerationId, InvocationMode, LinkSet, LogicalRequestId, Manifest, MapClass,
    MapDeclaration, MapHandle, MapKey, MapKeyType, MapMutation, MapPersistence, MapSchema,
    Operation, PlacementCandidate, PlacementCause, PlacementInput, PolicyLimits, RecordBatch,
    RequestContext, ResidentUnit, ScheduleCause, ScheduleInput, ServiceCandidate, ServiceCapacity,
    Symbol, TypedValue, ValueType,
};
use pie_policy::{
    AttachedPolicy, AttachmentRegistry, CapabilityCatalog, DedupLimits, Enactment, Invocation,
    InvocationFailureKind, ManualClock, PolicyEngine, PolicyEngineConfig, PolicyPackage,
    PreparedDecision, RegistryError, ReplayCommand, ReplayRunner, ReplayTrace,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: check_fixtures <component-directory>")?;
    let engine = PolicyEngine::new(PolicyEngineConfig::default())?;

    let attained = attach(&engine, &directory, "plex_attained_service", 1_000_000)?;
    let first = expect_success(attained.schedule(input()))?;
    let second = expect_success(attained.schedule(input()))?;
    assert_eq!(
        first, second,
        "fresh instances must produce identical output"
    );

    check_accept_all(&engine, &directory)?;
    check_least_loaded(&engine, &directory)?;
    check_retention_score(&engine, &directory)?;
    check_coordinated(&engine, &directory)?;
    check_over_quota(&engine, &directory)?;
    check_conflict_retry(&directory)?;
    check_attachment_registry(&engine, &directory)?;
    check_paper_policies(&engine, &directory)?;
    check_telemetry(&engine, &directory)?;

    let malformed = attach(&engine, &directory, "plex_malformed", 1_000_000)?;
    expect_failure(
        malformed.schedule(input()),
        InvocationFailureKind::InvalidOutput,
    )?;

    let spin = attach(&engine, &directory, "plex_spin", 10_000)?;
    expect_failure(spin.schedule(input()), InvocationFailureKind::FuelExhausted)?;

    let trap = attach(&engine, &directory, "plex_trap", 1_000_000)?;
    expect_failure(trap.schedule(input()), InvocationFailureKind::Trap)?;

    check_feedback_accounting(&engine, &directory)?;
    check_replay(&engine, &directory)?;

    println!("PLEX policy fixtures passed");
    Ok(())
}

fn check_accept_all(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let policy = attach_manifest(
        engine,
        directory,
        "plex_accept_all",
        manifest_for("plex_accept_all", 1_000_000, [Operation::Admit]),
        &CapabilityCatalog::default(),
    )?;
    let prepared = match policy.admit(admission_input()) {
        Invocation::Success(prepared) => prepared,
        other => return Err(format!("accept-all admission failed: {other:?}").into()),
    };
    assert_eq!(prepared.decision().decision, AdmissionDecision::Accept);
    prepared.commit();
    Ok(())
}

fn check_least_loaded(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let fact = Symbol::new("pie.placement.load@1")?;
    let mut catalog = CapabilityCatalog::default();
    catalog.add_fact(fact.clone(), ValueType::U64, 8)?;
    let mut manifest = manifest_for("plex_least_loaded", 1_000_000, [Operation::Route]);
    manifest.facts.push(FactDeclaration {
        name: fact,
        value_type: ValueType::U64,
        requirement: DependencyRequirement::Required,
        max_value_bytes: 8,
        uses: BTreeSet::from([FieldUse {
            operation: Operation::Route,
            location: FieldLocation::Candidate,
        }]),
    });
    let policy = attach_manifest(engine, directory, "plex_least_loaded", manifest, &catalog)?;
    let handle = policy.resolution().links().facts[0].expect("load fact linked");
    let prepared = match policy.route(PlacementInput {
        links: LinkSet::default(),
        cause: PlacementCause::GenerationArrival,
        request: RequestContext {
            logical_request_id: LogicalRequestId::new([1; 16]),
            generation_id: Some(GenerationId::new(0)),
            fields: RecordBatch::empty(1),
        },
        placements: CandidateSet {
            candidates: vec![PlacementCandidate, PlacementCandidate, PlacementCandidate],
            fields: RecordBatch {
                rows: 3,
                facts: vec![FactColumn {
                    handle,
                    values: ColumnValues::U64(vec![Some(30), Some(10), Some(20)]),
                }],
                metadata: Vec::new(),
            },
        },
    }) {
        Invocation::Success(prepared) => prepared,
        other => return Err(format!("least-loaded route failed: {other:?}").into()),
    };
    assert_eq!(
        pie_plex::rank_placements(
            &PlacementInput {
                links: LinkSet::default(),
                cause: PlacementCause::GenerationArrival,
                request: RequestContext {
                    logical_request_id: LogicalRequestId::new([1; 16]),
                    generation_id: Some(GenerationId::new(0)),
                    fields: RecordBatch::empty(1),
                },
                placements: CandidateSet {
                    candidates: vec![PlacementCandidate, PlacementCandidate, PlacementCandidate,],
                    fields: RecordBatch::empty(3),
                },
            },
            prepared.decision(),
        )?,
        vec![1, 2, 0]
    );
    prepared.commit();
    Ok(())
}

fn check_retention_score(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let fact = Symbol::new("pie.retention-value@1")?;
    let mut catalog = CapabilityCatalog::default();
    catalog.add_fact(fact.clone(), ValueType::F64, 8)?;
    let mut manifest = manifest_for("plex_retention_score", 1_000_000, [Operation::Evict]);
    manifest.facts.push(FactDeclaration {
        name: fact,
        value_type: ValueType::F64,
        requirement: DependencyRequirement::Required,
        max_value_bytes: 8,
        uses: BTreeSet::from([FieldUse {
            operation: Operation::Evict,
            location: FieldLocation::Candidate,
        }]),
    });
    let policy = attach_manifest(
        engine,
        directory,
        "plex_retention_score",
        manifest,
        &catalog,
    )?;
    let handle = policy.resolution().links().facts[0].expect("retention fact linked");
    let input = EvictionInput {
        links: LinkSet::default(),
        cause: EvictionCause::AllocationDeficit,
        bytes_needed: 6,
        resident: CandidateSet {
            candidates: resident_units(),
            fields: RecordBatch {
                rows: 3,
                facts: vec![FactColumn {
                    handle,
                    values: ColumnValues::F64(vec![Some(2.0), Some(1.0), Some(3.0)]),
                }],
                metadata: Vec::new(),
            },
        },
    };
    let prepared = match policy.evict(input.clone()) {
        Invocation::Success(prepared) => prepared,
        other => return Err(format!("retention eviction failed: {other:?}").into()),
    };
    assert_eq!(
        pie_plex::select_evictions(&input, prepared.decision())?
            .iter()
            .map(|selected| selected.candidate_index)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    prepared.commit();
    Ok(())
}

fn check_coordinated(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let event = Symbol::new("pie.progress@1")?;
    let mut catalog = CapabilityCatalog::default();
    catalog.add_event(event.clone())?;
    let mut manifest = manifest_for(
        "plex_coordinated",
        1_000_000,
        [
            Operation::Admit,
            Operation::Route,
            Operation::Schedule,
            Operation::Evict,
            Operation::Feedback,
        ],
    );
    manifest.events.push(EventDeclaration {
        name: event,
        requirement: DependencyRequirement::Required,
    });
    let mut accounting = accounting_map();
    accounting.schema.max_key_bytes = 16;
    manifest.maps.push(accounting);
    let policy = attach_manifest(engine, directory, "plex_coordinated", manifest, &catalog)?;

    expect_prepared(policy.admit(admission_input()), "coordinated admit")?.commit();
    expect_prepared(policy.route(placement_input(3)), "coordinated route")?.commit();
    expect_success(policy.schedule(input()))?;
    expect_prepared(policy.evict(eviction_input()), "coordinated evict")?.commit();

    let event_handle = policy.resolution().links().events[0].expect("event linked");
    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([6; 16]),
        events: vec![event_handle],
        subjects: feedback_subjects(1),
        records: RecordBatch::empty(1),
    };
    assert!(matches!(
        policy.feedback(feedback),
        Invocation::Success(FeedbackAcknowledgement::Committed(_))
    ));
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        Some(TypedValue::U64(11))
    );
    Ok(())
}

fn check_over_quota(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = manifest_for("plex_over_quota", 1_000_000, [Operation::Schedule]);
    let mut accounting = accounting_map();
    accounting.schema.max_key_bytes = 16;
    manifest.maps.push(accounting);
    manifest.limits.staged_mutations = 1;
    let policy = attach_manifest(
        engine,
        directory,
        "plex_over_quota",
        manifest,
        &CapabilityCatalog::default(),
    )?;
    expect_failure(
        policy.schedule(input()),
        InvocationFailureKind::InvalidOutput,
    )?;
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        None
    );
    Ok(())
}

fn check_conflict_retry(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let config = PolicyEngineConfig {
        max_deadline_ms: 500,
        max_conflict_retries: 10_000,
        ..PolicyEngineConfig::default()
    };
    let engine = PolicyEngine::new(config)?;
    let mut manifest = manifest_for("plex_retry_fresh", 1_000_000, [Operation::Schedule]);
    manifest.limits.deadline_ms = 500;
    manifest.maps.push(accounting_map());
    let policy = attach_manifest(
        &engine,
        directory,
        "plex_retry_fresh",
        manifest,
        &CapabilityCatalog::default(),
    )?;

    let old_snapshot = policy.maps().begin()?;
    let mut seed = policy.maps().begin()?;
    seed.stage(MapMutation::Upsert {
        map: MapHandle::new(0),
        key: MapKey::Bytes(vec![0]),
        value: TypedValue::U64(0),
        ttl_ms: None,
    })?;
    seed.prepare()?.commit();

    let barrier = Arc::new(Barrier::new(2));
    let release = barrier.clone();
    let releaser = std::thread::spawn(move || {
        release.wait();
        std::thread::sleep(std::time::Duration::from_millis(5));
        drop(old_snapshot);
    });
    barrier.wait();
    let prepared = expect_prepared(policy.schedule(input()), "conflict retry")?;
    assert!(
        prepared.attempts() > 1,
        "the retained old snapshot must force at least one retry"
    );
    prepared.commit();
    releaser.join().map_err(|_| "retry releaser panicked")?;
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        Some(TypedValue::U64(1))
    );
    Ok(())
}

fn check_attachment_registry(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = AttachmentRegistry::new(engine.clone(), CapabilityCatalog::default());
    let mut accept_manifest = manifest_for("replaceable", 1_000_000, [Operation::Admit]);
    accept_manifest.package_version = "0.1.0".into();
    let accept = package_bytes(directory, "plex_accept_all", accept_manifest)?;
    assert_eq!(registry.attach(&accept)?, 1);

    let old_snapshot = registry.snapshot()?;
    let old_decision = match old_snapshot.admit(admission_input()) {
        Invocation::Success(decision) => decision,
        other => return Err(format!("old registry admission failed: {other:?}").into()),
    };
    assert_eq!(old_decision.decision().decision, AdmissionDecision::Accept);

    let other = package_bytes(
        directory,
        "plex_defer_all",
        manifest_for("other-owner", 1_000_000, [Operation::Admit]),
    )?;
    assert!(matches!(
        registry.attach(&other),
        Err(RegistryError::OperationAlreadyOwned {
            operation: Operation::Admit,
            ..
        })
    ));

    let mut defer_manifest = manifest_for("replaceable", 1_000_000, [Operation::Admit]);
    defer_manifest.package_version = "0.2.0".into();
    let defer = package_bytes(directory, "plex_defer_all", defer_manifest)?;
    let replacing = registry.clone();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let replacer = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        done_tx.send(replacing.replace(&defer)).unwrap();
    });
    started_rx.recv()?;
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert!(
        done_rx.try_recv().is_err(),
        "replacement must wait for the old snapshot and prepared decision"
    );
    drop(old_snapshot);
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert!(
        done_rx.try_recv().is_err(),
        "prepared decision must retain the invocation snapshot lease"
    );
    old_decision.commit();
    assert_eq!(done_rx.recv_timeout(std::time::Duration::from_secs(1))??, 2);
    replacer.join().map_err(|_| "replacement thread panicked")?;

    let replacement_snapshot = registry.snapshot()?;
    let replacement_decision = match replacement_snapshot.admit(admission_input()) {
        Invocation::Success(decision) => decision,
        other => return Err(format!("replacement admission failed: {other:?}").into()),
    };
    assert_eq!(
        replacement_decision.decision().decision,
        AdmissionDecision::Defer
    );
    replacement_decision.abort();
    let generation = replacement_snapshot.generation();
    drop(replacement_snapshot);

    assert!(matches!(
        registry.replace(&[0]),
        Err(RegistryError::Prepare(_))
    ));
    assert_eq!(registry.snapshot()?.generation(), generation);

    let retained = registry.snapshot()?;
    assert_eq!(registry.detach_operation(Operation::Admit)?, generation + 1);
    assert!(matches!(
        registry.snapshot()?.admit(admission_input()),
        Invocation::Unavailable
    ));
    assert!(matches!(
        retained.admit(admission_input()),
        Invocation::Success(_)
    ));
    drop(retained);

    check_pinned_replacement(engine, directory)
}

fn check_pinned_replacement(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let event = Symbol::new("pie.progress@1")?;
    let mut catalog = CapabilityCatalog::default();
    catalog.add_event(event.clone())?;
    let registry = AttachmentRegistry::new(engine.clone(), catalog);
    let operations = [
        Operation::Admit,
        Operation::Route,
        Operation::Schedule,
        Operation::Evict,
        Operation::Feedback,
    ];
    let mut first = manifest_for("stateful", 1_000_000, operations);
    first.events.push(EventDeclaration {
        name: event.clone(),
        requirement: DependencyRequirement::Required,
    });
    let mut pinned = accounting_map();
    pinned.class = MapClass::PolicyOwned {
        persistence: MapPersistence::Pinned,
    };
    first.maps.push(pinned.clone());
    let first_package = package_bytes(directory, "plex_coordinated", first)?;
    registry.attach(&first_package)?;

    let snapshot = registry.snapshot()?;
    match snapshot.admit(admission_input()) {
        Invocation::Success(decision) => {
            decision.commit();
        }
        other => return Err(format!("stateful admission failed: {other:?}").into()),
    }
    let event_handle = snapshot
        .resolution(Operation::Feedback)
        .expect("feedback owner")
        .links()
        .events[0]
        .expect("event linked");
    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([8; 16]),
        events: vec![event_handle],
        subjects: feedback_subjects(1),
        records: RecordBatch::empty(1),
    };
    let first_ack = match snapshot.feedback(feedback.clone()) {
        Invocation::Success(FeedbackAcknowledgement::Committed(ack)) => ack,
        other => return Err(format!("stateful feedback failed: {other:?}").into()),
    };
    let route = match snapshot.route(placement_input(2)) {
        Invocation::Success(decision) => decision,
        other => return Err(format!("stateful route failed: {other:?}").into()),
    };
    assert_eq!(route.decision().scores[0], -2.0);
    route.commit();
    drop(snapshot);

    let mut second = manifest_for("stateful", 1_000_000, operations);
    second.package_version = "0.2.0".into();
    second.events.push(EventDeclaration {
        name: event,
        requirement: DependencyRequirement::Required,
    });
    second.maps.push(pinned);
    let second_package = package_bytes(directory, "plex_coordinated", second.clone())?;
    registry.replace(&second_package)?;
    let snapshot = registry.snapshot()?;
    let route = match snapshot.route(placement_input(2)) {
        Invocation::Success(decision) => decision,
        other => return Err(format!("transferred route failed: {other:?}").into()),
    };
    assert_eq!(route.decision().scores[0], -4.0);
    route.abort();
    match snapshot.feedback(feedback) {
        Invocation::Success(FeedbackAcknowledgement::Duplicate(ack)) => {
            assert_eq!(ack, first_ack);
        }
        other => return Err(format!("transferred feedback replay failed: {other:?}").into()),
    }
    let next_feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([9; 16]),
        events: vec![event_handle],
        subjects: feedback_subjects(1),
        records: RecordBatch::empty(1),
    };
    match snapshot.feedback(next_feedback) {
        Invocation::Success(FeedbackAcknowledgement::Committed(ack)) => {
            assert!(
                ack.revision > first_ack.revision,
                "replacement revisions must remain monotonic"
            );
        }
        other => return Err(format!("post-replacement feedback failed: {other:?}").into()),
    }
    drop(snapshot);

    second.package_version = "0.3.0".into();
    second.maps[0].schema.value_type = ValueType::I64;
    let incompatible = package_bytes(directory, "plex_coordinated", second)?;
    assert!(matches!(
        registry.replace(&incompatible),
        Err(RegistryError::StateTransfer(_))
    ));
    let snapshot = registry.snapshot()?;
    let route = match snapshot.route(placement_input(1)) {
        Invocation::Success(decision) => decision,
        other => return Err(format!("preserved old route failed: {other:?}").into()),
    };
    assert_eq!(route.decision().scores[0], -5.0);
    route.abort();
    Ok(())
}

fn check_telemetry(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = manifest_for("plex-telemetry-burst", 1_000_000, [Operation::Schedule]);
    manifest.limits.telemetry_records = 2;
    manifest.limits.telemetry_bytes = 128;
    let policy = attach_manifest(
        engine,
        directory,
        "plex_telemetry_burst",
        manifest,
        &CapabilityCatalog::default(),
    )?;
    expect_prepared(policy.schedule(input()), "telemetry burst")?.abort();
    let telemetry = policy.drain_telemetry();
    assert_eq!(telemetry.len(), 2);
    assert_eq!(telemetry[0].value, 1.0);
    assert_eq!(telemetry[1].value, 2.0);
    Ok(())
}

fn check_paper_policies(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read("tests/policies/paper-cases.json")?)?;
    let cases = cases
        .as_array()
        .ok_or("paper-cases.json must be an array")?;
    assert_eq!(cases.len(), 5);
    for case in cases {
        let source = case["source"]
            .as_str()
            .ok_or("paper case source must be a string")?;
        assert!(
            source.starts_with("https://"),
            "paper stress cases must cite a primary HTTPS source"
        );
    }

    check_agentix_policy(engine, directory)?;
    check_continuum_policy(engine, directory)?;
    check_kvflow_policy(engine, directory)?;
    check_preble_policy(engine, directory)?;
    check_helium_policy(engine, directory)
}

fn check_agentix_policy(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let event = Symbol::new("paper.agentix.service@1")?;
    let declarations = [
        (
            Symbol::new("paper.agentix.wait-us@1")?,
            ValueType::U64,
            Operation::Schedule,
            FieldLocation::Candidate,
        ),
        (
            Symbol::new("paper.agentix.program-id@1")?,
            ValueType::Bytes,
            Operation::Feedback,
            FieldLocation::Feedback,
        ),
        (
            Symbol::new("paper.agentix.service-us@1")?,
            ValueType::U64,
            Operation::Feedback,
            FieldLocation::Feedback,
        ),
    ];
    let mut catalog = CapabilityCatalog::default();
    for (name, value_type, _, _) in &declarations {
        catalog.add_fact(name.clone(), *value_type, 16)?;
    }
    catalog.add_event(event.clone())?;
    let mut manifest = manifest_for(
        "paper-agentix",
        2_000_000,
        [Operation::Schedule, Operation::Feedback],
    );
    manifest.facts = declarations
        .into_iter()
        .map(|(name, value_type, operation, location)| FactDeclaration {
            name,
            value_type,
            requirement: DependencyRequirement::Required,
            max_value_bytes: if value_type == ValueType::Bytes {
                16
            } else {
                8
            },
            uses: BTreeSet::from([FieldUse {
                operation,
                location,
            }]),
        })
        .collect();
    manifest.events.push(EventDeclaration {
        name: event,
        requirement: DependencyRequirement::Required,
    });
    let mut accounting = accounting_map();
    accounting.schema.max_key_bytes = 16;
    manifest.maps.push(accounting);
    let policy = attach_manifest(engine, directory, "plex_paper_agentix", manifest, &catalog)?;
    let handles = &policy.resolution().links().facts;
    let wait = handles[0].expect("wait fact linked");
    let program = handles[1].expect("program fact linked");
    let service = handles[2].expect("service fact linked");
    let event = policy.resolution().links().events[0].expect("event linked");

    let before = paper_schedule_input(
        vec![
            LogicalRequestId::new([1; 16]),
            LogicalRequestId::new([2; 16]),
        ],
        vec![FactColumn {
            handle: wait,
            values: ColumnValues::U64(vec![Some(1), Some(1)]),
        }],
    );
    let before = expect_prepared(policy.schedule(before), "Agentix initial schedule")?;
    assert_eq!(before.decision().decisions[0].score, -1.0);
    before.abort();

    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([21; 16]),
        events: vec![event],
        subjects: feedback_subjects(1),
        records: RecordBatch {
            rows: 1,
            facts: vec![
                FactColumn {
                    handle: program,
                    values: ColumnValues::Bytes(vec![Some(vec![1; 16])]),
                },
                FactColumn {
                    handle: service,
                    values: ColumnValues::U64(vec![Some(100)]),
                },
            ],
            metadata: Vec::new(),
        },
    };
    assert!(matches!(
        policy.feedback(feedback),
        Invocation::Success(FeedbackAcknowledgement::Committed(_))
    ));
    let after = paper_schedule_input(
        vec![
            LogicalRequestId::new([1; 16]),
            LogicalRequestId::new([2; 16]),
        ],
        vec![FactColumn {
            handle: wait,
            values: ColumnValues::U64(vec![Some(500), Some(1)]),
        }],
    );
    let after = expect_prepared(policy.schedule(after), "Agentix updated schedule")?;
    assert!(
        after.decision().decisions[0].score > after.decision().decisions[1].score,
        "Agentix anti-starvation must promote the long-waiting program"
    );
    after.abort();
    Ok(())
}

fn check_continuum_policy(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let declarations = [
        (
            Symbol::new("paper.continuum.preempted@1")?,
            ValueType::Bool,
            Operation::Schedule,
            FieldLocation::Candidate,
        ),
        (
            Symbol::new("paper.continuum.program-arrival@1")?,
            ValueType::U64,
            Operation::Schedule,
            FieldLocation::Candidate,
        ),
        (
            Symbol::new("paper.continuum.reload-cost@1")?,
            ValueType::F64,
            Operation::Evict,
            FieldLocation::Candidate,
        ),
        (
            Symbol::new("paper.continuum.logical-id@1")?,
            ValueType::Bytes,
            Operation::Feedback,
            FieldLocation::Feedback,
        ),
        (
            Symbol::new("paper.continuum.ttl-ms@1")?,
            ValueType::U64,
            Operation::Feedback,
            FieldLocation::Feedback,
        ),
    ];
    let event_name = Symbol::new("paper.continuum.tool-boundary@1")?;
    let mut catalog = CapabilityCatalog::default();
    for (name, value_type, _, _) in &declarations {
        catalog.add_fact(name.clone(), *value_type, 16)?;
    }
    catalog.add_event(event_name.clone())?;
    let mut manifest = manifest_for(
        "paper-continuum",
        2_000_000,
        [Operation::Schedule, Operation::Evict, Operation::Feedback],
    );
    manifest.facts = declarations
        .into_iter()
        .map(|(name, value_type, operation, location)| FactDeclaration {
            name,
            value_type,
            requirement: DependencyRequirement::Required,
            max_value_bytes: if value_type == ValueType::Bytes {
                16
            } else {
                8
            },
            uses: BTreeSet::from([FieldUse {
                operation,
                location,
            }]),
        })
        .collect();
    manifest.events.push(EventDeclaration {
        name: event_name,
        requirement: DependencyRequirement::Required,
    });
    let mut retention = accounting_map();
    retention.name = Symbol::new("paper.continuum.retention@1")?;
    retention.schema.max_key_bytes = 16;
    retention.schema.max_ttl_ms = Some(1_000);
    manifest.maps.push(retention);
    let component = std::fs::read(directory.join("plex_paper_continuum.component.wasm"))?;
    let clock = Arc::new(ManualClock::default());
    let policy = AttachedPolicy::compile_with_clock(
        engine.clone(),
        &component,
        manifest,
        &catalog,
        clock.clone(),
        DedupLimits::default(),
    )?;
    let facts = &policy.resolution().links().facts;
    let preempted = facts[0].expect("preempted fact linked");
    let arrival = facts[1].expect("arrival fact linked");
    let reload = facts[2].expect("reload fact linked");
    let id = facts[3].expect("id fact linked");
    let ttl = facts[4].expect("ttl fact linked");
    let event = policy.resolution().links().events[0].expect("event linked");

    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([22; 16]),
        events: vec![event],
        subjects: feedback_subjects(1),
        records: RecordBatch {
            rows: 1,
            facts: vec![
                FactColumn {
                    handle: id,
                    values: ColumnValues::Bytes(vec![Some(vec![1; 16])]),
                },
                FactColumn {
                    handle: ttl,
                    values: ColumnValues::U64(vec![Some(50)]),
                },
            ],
            metadata: Vec::new(),
        },
    };
    assert!(matches!(
        policy.feedback(feedback),
        Invocation::Success(FeedbackAcknowledgement::Committed(_))
    ));

    let schedule = continuum_schedule(preempted, arrival, vec![false, false]);
    let pinned = expect_prepared(policy.schedule(schedule), "Continuum pinned schedule")?;
    assert!(
        pinned.decision().decisions[0].score > pinned.decision().decisions[1].score,
        "TTL-pinned request must outrank plain FCFS"
    );
    pinned.abort();
    let preempted_input = continuum_schedule(preempted, arrival, vec![false, true]);
    let preempted_decision = expect_prepared(
        policy.schedule(preempted_input),
        "Continuum preempted schedule",
    )?;
    assert!(
        preempted_decision.decision().decisions[1].score
            > preempted_decision.decision().decisions[0].score,
        "preempted status must dominate TTL status"
    );
    preempted_decision.abort();

    let eviction = EvictionInput {
        links: LinkSet::default(),
        cause: EvictionCause::AllocationDeficit,
        bytes_needed: 3,
        resident: CandidateSet {
            candidates: vec![
                ResidentUnit {
                    size_bytes: 4,
                    logical_request_id: Some(LogicalRequestId::new([1; 16])),
                    generation_id: None,
                },
                ResidentUnit {
                    size_bytes: 3,
                    logical_request_id: Some(LogicalRequestId::new([2; 16])),
                    generation_id: None,
                },
            ],
            fields: RecordBatch {
                rows: 2,
                facts: vec![FactColumn {
                    handle: reload,
                    values: ColumnValues::F64(vec![Some(1.0), Some(100.0)]),
                }],
                metadata: Vec::new(),
            },
        },
    };
    let evict = expect_prepared(policy.evict(eviction.clone()), "Continuum eviction")?;
    assert_eq!(
        pie_plex::select_evictions(&eviction, evict.decision())?[0].candidate_index,
        1
    );
    evict.abort();

    clock.advance(51);
    let expired = expect_prepared(
        policy.schedule(continuum_schedule(preempted, arrival, vec![false, false])),
        "Continuum expired schedule",
    )?;
    assert!(
        expired.decision().decisions[1].score > expired.decision().decisions[0].score,
        "program FCFS must resume after TTL expiry"
    );
    expired.abort();
    Ok(())
}

fn check_kvflow_policy(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let declarations = [
        (
            Symbol::new("paper.kvflow.steps-to-execution@1")?,
            ValueType::U64,
            Operation::Evict,
        ),
        (
            Symbol::new("paper.kvflow.fixed-prefix@1")?,
            ValueType::Bool,
            Operation::Evict,
        ),
        (
            Symbol::new("paper.kvflow.cache-ready@1")?,
            ValueType::Bool,
            Operation::Schedule,
        ),
    ];
    let mut catalog = CapabilityCatalog::default();
    for (name, value_type, _) in &declarations {
        catalog.add_fact(name.clone(), *value_type, 8)?;
    }
    let mut manifest = manifest_for(
        "paper-kvflow",
        1_000_000,
        [Operation::Schedule, Operation::Evict],
    );
    manifest.facts = declarations
        .into_iter()
        .map(|(name, value_type, operation)| FactDeclaration {
            name,
            value_type,
            requirement: DependencyRequirement::Required,
            max_value_bytes: 8,
            uses: BTreeSet::from([FieldUse {
                operation,
                location: FieldLocation::Candidate,
            }]),
        })
        .collect();
    let policy = attach_manifest(engine, directory, "plex_paper_kvflow", manifest, &catalog)?;
    let facts = &policy.resolution().links().facts;
    let steps = facts[0].expect("steps fact linked");
    let fixed = facts[1].expect("fixed fact linked");
    let ready = facts[2].expect("ready fact linked");
    let schedule = paper_schedule_input(
        vec![
            LogicalRequestId::new([1; 16]),
            LogicalRequestId::new([2; 16]),
        ],
        vec![FactColumn {
            handle: ready,
            values: ColumnValues::Bool(vec![Some(false), Some(true)]),
        }],
    );
    let schedule = expect_prepared(policy.schedule(schedule), "KVFlow schedule")?;
    assert!(schedule.decision().decisions[1].score > schedule.decision().decisions[0].score);
    schedule.abort();
    let eviction = EvictionInput {
        links: LinkSet::default(),
        cause: EvictionCause::AllocationDeficit,
        bytes_needed: 1,
        resident: CandidateSet {
            candidates: resident_units(),
            fields: RecordBatch {
                rows: 3,
                facts: vec![
                    FactColumn {
                        handle: steps,
                        values: ColumnValues::U64(vec![Some(1), Some(5), Some(2)]),
                    },
                    FactColumn {
                        handle: fixed,
                        values: ColumnValues::Bool(vec![Some(true), Some(true), Some(false)]),
                    },
                ],
                metadata: Vec::new(),
            },
        },
    };
    let evict = expect_prepared(policy.evict(eviction.clone()), "KVFlow eviction")?;
    assert_eq!(
        pie_plex::select_evictions(&eviction, evict.decision())?[0].candidate_index,
        2
    );
    evict.abort();
    Ok(())
}

fn check_preble_policy(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let names = [
        "paper.preble.cached-prefix-tokens@1",
        "paper.preble.uncached-tokens@1",
        "paper.preble.load-cost-us@1",
        "paper.preble.eviction-cost-us@1",
    ];
    let mut catalog = CapabilityCatalog::default();
    let mut manifest = manifest_for("paper-preble", 1_000_000, [Operation::Route]);
    for name in names {
        let name = Symbol::new(name)?;
        catalog.add_fact(name.clone(), ValueType::U64, 8)?;
        manifest.facts.push(FactDeclaration {
            name,
            value_type: ValueType::U64,
            requirement: DependencyRequirement::Required,
            max_value_bytes: 8,
            uses: BTreeSet::from([FieldUse {
                operation: Operation::Route,
                location: FieldLocation::Candidate,
            }]),
        });
    }
    let policy = attach_manifest(engine, directory, "plex_paper_preble", manifest, &catalog)?;
    let handles: Vec<_> = policy
        .resolution()
        .links()
        .facts
        .iter()
        .map(|handle| handle.expect("Preble fact linked"))
        .collect();
    let exploit = preble_input(
        &handles,
        vec![100, 80],
        vec![50, 50],
        vec![100, 1],
        vec![0, 0],
    );
    let exploit_result = expect_prepared(policy.route(exploit.clone()), "Preble exploit")?;
    assert_eq!(
        pie_plex::rank_placements(&exploit, exploit_result.decision())?[0],
        0
    );
    exploit_result.abort();
    let explore = preble_input(
        &handles,
        vec![10, 20],
        vec![50, 50],
        vec![100, 10],
        vec![0, 5],
    );
    let explore_result = expect_prepared(policy.route(explore.clone()), "Preble explore")?;
    assert_eq!(
        pie_plex::rank_placements(&explore, explore_result.decision())?[0],
        1
    );
    explore_result.abort();
    Ok(())
}

fn check_helium_policy(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let declarations = [
        ("paper.helium.ready@1", ValueType::Bool),
        ("paper.helium.dependency-depth@1", ValueType::U64),
        ("paper.helium.earliest-start@1", ValueType::U64),
        ("paper.helium.prefix-reuse-tokens@1", ValueType::U64),
        ("paper.helium.profiled-token-cost@1", ValueType::U64),
    ];
    let mut catalog = CapabilityCatalog::default();
    let mut manifest = manifest_for("paper-helium", 1_000_000, [Operation::Schedule]);
    for (name, value_type) in declarations {
        let name = Symbol::new(name)?;
        catalog.add_fact(name.clone(), value_type, 8)?;
        manifest.facts.push(FactDeclaration {
            name,
            value_type,
            requirement: DependencyRequirement::Required,
            max_value_bytes: 8,
            uses: BTreeSet::from([FieldUse {
                operation: Operation::Schedule,
                location: FieldLocation::Candidate,
            }]),
        });
    }
    let policy = attach_manifest(engine, directory, "plex_paper_helium", manifest, &catalog)?;
    let handles: Vec<_> = policy
        .resolution()
        .links()
        .facts
        .iter()
        .map(|handle| handle.expect("Helium fact linked"))
        .collect();
    let ready = helium_input(
        &handles,
        vec![false, true, true],
        vec![9, 2, 4],
        vec![1, 2, 3],
        vec![100, 20, 10],
        vec![10, 10, 10],
    );
    let ready_result = expect_prepared(policy.schedule(ready), "Helium ready schedule")?;
    assert!(
        ready_result.decision().decisions[2].score > ready_result.decision().decisions[1].score
    );
    assert!(ready_result.decision().decisions[0].score < -1.0e17);
    ready_result.abort();
    let forced = helium_input(
        &handles,
        vec![false, false],
        vec![1, 1],
        vec![10, 5],
        vec![0, 0],
        vec![1, 1],
    );
    let forced_result = expect_prepared(policy.schedule(forced), "Helium forced progress")?;
    assert!(
        forced_result.decision().decisions[1].score > forced_result.decision().decisions[0].score
    );
    forced_result.abort();
    Ok(())
}

fn check_feedback_accounting(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let policy =
        attach_feedback_accounting(engine, directory, Arc::new(ManualClock::default()), 1 << 12)?;
    check_feedback_policy(&policy)?;

    let limited =
        attach_feedback_accounting(engine, directory, Arc::new(ManualClock::default()), 1)?;
    let event = limited.resolution().links().events[0].expect("event linked");
    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([4; 16]),
        events: vec![event],
        subjects: feedback_subjects(1),
        records: RecordBatch::empty(1),
    };
    assert!(matches!(
        limited.feedback(feedback),
        Invocation::Success(FeedbackAcknowledgement::Committed(_))
    ));
    expect_failure(
        limited.schedule(input()),
        InvocationFailureKind::MapLimitExceeded,
    )
}

fn attach_feedback_accounting(
    engine: &PolicyEngine,
    directory: &Path,
    clock: Arc<ManualClock>,
    map_bytes: u64,
) -> Result<AttachedPolicy, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(directory.join("plex_feedback_accounting.component.wasm"))?;
    let event = Symbol::new("pie.progress@1")?;
    let mut catalog = CapabilityCatalog::default();
    catalog.add_event(event.clone())?;
    let mut manifest = manifest("plex_feedback_accounting", 1_000_000);
    manifest.events.push(EventDeclaration {
        name: event,
        requirement: DependencyRequirement::Required,
    });
    manifest.maps.push(MapDeclaration {
        name: Symbol::new("policy.accounting@1")?,
        class: MapClass::PolicyOwned {
            persistence: MapPersistence::Attachment,
        },
        schema: MapSchema {
            key_type: MapKeyType::Bytes,
            value_type: ValueType::U64,
            max_entries: 8,
            max_key_bytes: 8,
            max_value_bytes: 8,
            default_ttl_ms: None,
            max_ttl_ms: None,
        },
    });
    manifest.limits.map_bytes = map_bytes;
    Ok(AttachedPolicy::compile_with_clock(
        engine.clone(),
        &bytes,
        manifest,
        &catalog,
        clock,
        DedupLimits::default(),
    )?)
}

fn check_feedback_policy(policy: &AttachedPolicy) -> Result<(), Box<dyn std::error::Error>> {
    let aborted = match policy.schedule(input()) {
        Invocation::Success(prepared) => prepared,
        other => return Err(format!("prepared schedule failed: {other:?}").into()),
    };
    assert_eq!(aborted.decision().decisions[0].score, -0.0);
    aborted.abort();
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        None
    );

    let committed_schedule = match policy.schedule(input()) {
        Invocation::Success(prepared) => prepared,
        other => return Err(format!("prepared schedule failed: {other:?}").into()),
    };
    let (before, _) = committed_schedule.commit();
    assert_eq!(before.decisions[0].score, -0.0);
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        Some(TypedValue::U64(10))
    );

    let event_handle = policy.resolution().links().events[0].expect("event linked");
    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([9; 16]),
        events: vec![event_handle],
        subjects: feedback_subjects(1),
        records: RecordBatch::empty(1),
    };
    let committed = match policy.feedback(feedback.clone()) {
        Invocation::Success(acknowledgement) => acknowledgement,
        other => return Err(format!("feedback failed: {other:?}").into()),
    };
    assert!(matches!(committed, FeedbackAcknowledgement::Committed(_)));
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        Some(TypedValue::U64(11))
    );

    let duplicate = match policy.feedback(feedback) {
        Invocation::Success(acknowledgement) => acknowledgement,
        other => return Err(format!("feedback replay failed: {other:?}").into()),
    };
    assert!(matches!(duplicate, FeedbackAcknowledgement::Duplicate(_)));
    assert_eq!(
        policy
            .maps()
            .read(MapHandle::new(0), &MapKey::Bytes(vec![0]))?,
        Some(TypedValue::U64(11))
    );

    let after = expect_success(policy.schedule(input()))?;
    assert_eq!(after.decisions[0].score, -11.0);
    Ok(())
}

fn check_replay(engine: &PolicyEngine, directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let _ = engine;
    let engine = PolicyEngine::new(PolicyEngineConfig::deterministic_replay())?;
    let event = Symbol::new("pie.progress@1")?;
    let operations = [
        Operation::Admit,
        Operation::Route,
        Operation::Schedule,
        Operation::Evict,
        Operation::Feedback,
    ];
    let mut first_manifest = manifest_for("replay-stateful", 1_000_000, operations);
    first_manifest.events.push(EventDeclaration {
        name: event.clone(),
        requirement: DependencyRequirement::Required,
    });
    let mut pinned = accounting_map();
    pinned.class = MapClass::PolicyOwned {
        persistence: MapPersistence::Pinned,
    };
    first_manifest.maps.push(pinned.clone());
    let first_package = package_bytes(directory, "plex_coordinated", first_manifest)?;
    let mut second_manifest = manifest_for("replay-stateful", 1_000_000, operations);
    second_manifest.package_version = "0.2.0".into();
    second_manifest.events.push(EventDeclaration {
        name: event,
        requirement: DependencyRequirement::Required,
    });
    second_manifest.maps.push(pinned);
    let second_package = package_bytes(directory, "plex_coordinated", second_manifest)?;
    let mut external_manifest = manifest_for("replay-external", 1_000_000, [Operation::Schedule]);
    external_manifest.maps.push(external_weight_map());
    let external_package = package_bytes(directory, "plex_external_weight", external_manifest)?;

    let feedback = FeedbackBatch {
        links: LinkSet::default(),
        delivery_id: DeliveryId::new([5; 16]),
        events: vec![EventHandle::new(0)],
        subjects: feedback_subjects(1),
        records: RecordBatch::empty(1),
    };
    let trace = ReplayTrace {
        commands: vec![
            ReplayCommand::Attach {
                package: "v1".into(),
            },
            ReplayCommand::Admit {
                input: admission_input(),
                enactment: Enactment::Commit,
            },
            ReplayCommand::Route {
                input: placement_input(2),
                enactment: Enactment::Commit,
            },
            ReplayCommand::Schedule {
                input: input(),
                enactment: Enactment::Commit,
            },
            ReplayCommand::Evict {
                input: eviction_input(),
                enactment: Enactment::Commit,
            },
            ReplayCommand::Feedback {
                input: feedback.clone(),
            },
            ReplayCommand::Feedback { input: feedback },
            ReplayCommand::ReadMap {
                package: "replay-stateful".into(),
                handle: MapHandle::new(0),
                key: MapKey::Bytes(vec![0]),
            },
            ReplayCommand::Replace {
                package: "v2".into(),
            },
            ReplayCommand::Route {
                input: placement_input(2),
                enactment: Enactment::Abort,
            },
            ReplayCommand::AdvanceClock { millis: 10 },
            ReplayCommand::DetachOperation {
                operation: Operation::Route,
            },
            ReplayCommand::Route {
                input: placement_input(1),
                enactment: Enactment::Abort,
            },
            ReplayCommand::DetachPackage {
                package: "replay-stateful".into(),
            },
            ReplayCommand::Admit {
                input: admission_input(),
                enactment: Enactment::Abort,
            },
            ReplayCommand::Attach {
                package: "external".into(),
            },
            ReplayCommand::PublishExternal {
                package: "replay-external".into(),
                handle: MapHandle::new(0),
                entries: vec![(MapKey::Bytes(vec![0]), TypedValue::U64(42))],
            },
            ReplayCommand::Schedule {
                input: input(),
                enactment: Enactment::Commit,
            },
            ReplayCommand::ReadMap {
                package: "replay-external".into(),
                handle: MapHandle::new(0),
                key: MapKey::Bytes(vec![0]),
            },
            ReplayCommand::DetachPackage {
                package: "replay-external".into(),
            },
        ],
    };
    let encoded = serde_json::to_vec(&trace)?;
    let decoded: ReplayTrace = serde_json::from_slice(&encoded)?;
    let packages = BTreeMap::from([
        ("v1".into(), first_package),
        ("v2".into(), second_package),
        ("external".into(), external_package),
    ]);
    let first_clock = Arc::new(ManualClock::default());
    let mut first_catalog = CapabilityCatalog::default();
    first_catalog.add_event(Symbol::new("pie.progress@1")?)?;
    first_catalog.add_external_map(
        Symbol::new("operator.weight@1")?,
        external_weight_map().schema,
    )?;
    let first_registry = AttachmentRegistry::new_with_clock(
        engine.clone(),
        first_catalog,
        first_clock.clone(),
        DedupLimits::default(),
    );
    let first = ReplayRunner::new(first_registry, first_clock, packages.clone())?.run(&decoded)?;

    let second_clock = Arc::new(ManualClock::default());
    let mut second_catalog = CapabilityCatalog::default();
    second_catalog.add_event(Symbol::new("pie.progress@1")?)?;
    second_catalog.add_external_map(
        Symbol::new("operator.weight@1")?,
        external_weight_map().schema,
    )?;
    let second_registry = AttachmentRegistry::new_with_clock(
        engine,
        second_catalog,
        second_clock.clone(),
        DedupLimits::default(),
    );
    ReplayRunner::new(second_registry, second_clock, packages)?.verify(&decoded, &first)?;
    Ok(())
}

fn attach(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    fuel: u64,
) -> Result<AttachedPolicy, Box<dyn std::error::Error>> {
    attach_manifest(
        engine,
        directory,
        artifact,
        manifest(artifact, fuel),
        &CapabilityCatalog::default(),
    )
}

fn package_bytes(
    directory: &Path,
    artifact: &str,
    manifest: Manifest,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let component = std::fs::read(directory.join(format!("{artifact}.component.wasm")))?;
    Ok(PolicyPackage::new(manifest, component)?.encode()?)
}

fn attach_manifest(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    manifest: Manifest,
    catalog: &CapabilityCatalog,
) -> Result<AttachedPolicy, Box<dyn std::error::Error>> {
    let package = package_bytes(directory, artifact, manifest)?;
    Ok(AttachedPolicy::compile_package(
        engine.clone(),
        &package,
        catalog,
    )?)
}

fn manifest(name: &str, fuel: u64) -> Manifest {
    manifest_for(name, fuel, [Operation::Schedule, Operation::Feedback])
}

fn manifest_for(
    name: &str,
    fuel: u64,
    operations: impl IntoIterator<Item = Operation>,
) -> Manifest {
    Manifest {
        contract: ContractVersion::V0_1,
        package_name: name.replace('_', "-"),
        package_version: "0.1.0".into(),
        operations: operations.into_iter().collect::<BTreeSet<_>>(),
        invocation_mode: InvocationMode::SetDependent,
        capabilities: Vec::new(),
        facts: Vec::new(),
        metadata: Vec::new(),
        events: Vec::new(),
        maps: Vec::new(),
        limits: PolicyLimits {
            memory_bytes: 2 << 20,
            fuel,
            deadline_ms: 50,
            input_bytes: 1 << 16,
            output_bytes: 1 << 16,
            map_calls: 16,
            map_bytes: 1 << 12,
            staged_mutations: 8,
            feedback_records: 32,
            telemetry_records: 0,
            telemetry_bytes: 0,
        },
    }
}

fn accounting_map() -> MapDeclaration {
    MapDeclaration {
        name: Symbol::new("policy.accounting@1").expect("static symbol"),
        class: MapClass::PolicyOwned {
            persistence: MapPersistence::Attachment,
        },
        schema: MapSchema {
            key_type: MapKeyType::Bytes,
            value_type: ValueType::U64,
            max_entries: 8,
            max_key_bytes: 8,
            max_value_bytes: 8,
            default_ttl_ms: None,
            max_ttl_ms: None,
        },
    }
}

fn external_weight_map() -> MapDeclaration {
    MapDeclaration {
        name: Symbol::new("operator.weight@1").expect("static symbol"),
        class: MapClass::External {
            requirement: DependencyRequirement::Required,
        },
        schema: MapSchema {
            key_type: MapKeyType::Bytes,
            value_type: ValueType::U64,
            max_entries: 8,
            max_key_bytes: 8,
            max_value_bytes: 8,
            default_ttl_ms: None,
            max_ttl_ms: None,
        },
    }
}

fn admission_input() -> AdmissionInput {
    AdmissionInput {
        links: LinkSet::default(),
        request: RequestContext {
            logical_request_id: LogicalRequestId::new([1; 16]),
            generation_id: None,
            fields: RecordBatch::empty(1),
        },
    }
}

fn feedback_subjects(count: usize) -> Vec<FeedbackSubject> {
    (0..count)
        .map(|index| FeedbackSubject {
            logical_request_id: LogicalRequestId::new((index as u128 + 1).to_be_bytes()),
            generation_id: Some(GenerationId::new(index as u64)),
            terminal_outcome: None,
        })
        .collect()
}

fn placement_input(count: usize) -> PlacementInput {
    PlacementInput {
        links: LinkSet::default(),
        cause: PlacementCause::GenerationArrival,
        request: RequestContext {
            logical_request_id: LogicalRequestId::new([1; 16]),
            generation_id: Some(GenerationId::new(0)),
            fields: RecordBatch::empty(1),
        },
        placements: CandidateSet {
            candidates: vec![PlacementCandidate; count],
            fields: RecordBatch::empty(u32::try_from(count).expect("fixture count fits u32")),
        },
    }
}

fn resident_units() -> Vec<ResidentUnit> {
    vec![
        ResidentUnit {
            size_bytes: 4,
            logical_request_id: None,
            generation_id: None,
        },
        ResidentUnit {
            size_bytes: 3,
            logical_request_id: None,
            generation_id: None,
        },
        ResidentUnit {
            size_bytes: 8,
            logical_request_id: None,
            generation_id: None,
        },
    ]
}

fn eviction_input() -> EvictionInput {
    EvictionInput {
        links: LinkSet::default(),
        cause: EvictionCause::AllocationDeficit,
        bytes_needed: 6,
        resident: CandidateSet {
            candidates: resident_units(),
            fields: RecordBatch::empty(3),
        },
    }
}

fn paper_schedule_input(
    logical_ids: Vec<LogicalRequestId>,
    facts: Vec<FactColumn>,
) -> ScheduleInput {
    let rows = u32::try_from(logical_ids.len()).expect("fixture rows fit u32");
    ScheduleInput {
        links: LinkSet::default(),
        cause: ScheduleCause::ServiceStep,
        runnable: CandidateSet {
            candidates: logical_ids
                .into_iter()
                .enumerate()
                .map(|(index, logical_request_id)| ServiceCandidate {
                    logical_request_id,
                    generation_id: GenerationId::new(index as u64),
                    max_token_budget: 8,
                })
                .collect(),
            fields: RecordBatch {
                rows,
                facts,
                metadata: Vec::new(),
            },
        },
        capacity: ServiceCapacity {
            max_selected: rows,
            max_total_tokens: rows.saturating_mul(8),
            max_token_budget: 8,
        },
    }
}

fn continuum_schedule(
    preempted: pie_plex::FactHandle,
    arrival: pie_plex::FactHandle,
    preempted_values: Vec<bool>,
) -> ScheduleInput {
    paper_schedule_input(
        vec![
            LogicalRequestId::new([1; 16]),
            LogicalRequestId::new([2; 16]),
        ],
        vec![
            FactColumn {
                handle: preempted,
                values: ColumnValues::Bool(preempted_values.into_iter().map(Some).collect()),
            },
            FactColumn {
                handle: arrival,
                values: ColumnValues::U64(vec![Some(10), Some(1)]),
            },
        ],
    )
}

fn preble_input(
    handles: &[pie_plex::FactHandle],
    cached: Vec<u64>,
    uncached: Vec<u64>,
    load: Vec<u64>,
    eviction: Vec<u64>,
) -> PlacementInput {
    let rows = u32::try_from(cached.len()).expect("fixture rows fit u32");
    PlacementInput {
        links: LinkSet::default(),
        cause: PlacementCause::GenerationArrival,
        request: RequestContext {
            logical_request_id: LogicalRequestId::new([1; 16]),
            generation_id: Some(GenerationId::new(0)),
            fields: RecordBatch::empty(1),
        },
        placements: CandidateSet {
            candidates: vec![PlacementCandidate; usize::try_from(rows).expect("u32 fits usize")],
            fields: RecordBatch {
                rows,
                facts: vec![
                    FactColumn {
                        handle: handles[0],
                        values: ColumnValues::U64(cached.into_iter().map(Some).collect()),
                    },
                    FactColumn {
                        handle: handles[1],
                        values: ColumnValues::U64(uncached.into_iter().map(Some).collect()),
                    },
                    FactColumn {
                        handle: handles[2],
                        values: ColumnValues::U64(load.into_iter().map(Some).collect()),
                    },
                    FactColumn {
                        handle: handles[3],
                        values: ColumnValues::U64(eviction.into_iter().map(Some).collect()),
                    },
                ],
                metadata: Vec::new(),
            },
        },
    }
}

fn helium_input(
    handles: &[pie_plex::FactHandle],
    ready: Vec<bool>,
    depth: Vec<u64>,
    earliest: Vec<u64>,
    reuse: Vec<u64>,
    cost: Vec<u64>,
) -> ScheduleInput {
    let rows = ready.len();
    paper_schedule_input(
        (0..rows)
            .map(|index| {
                let byte = u8::try_from(index + 1).expect("small fixture index");
                LogicalRequestId::new([byte; 16])
            })
            .collect(),
        vec![
            FactColumn {
                handle: handles[0],
                values: ColumnValues::Bool(ready.into_iter().map(Some).collect()),
            },
            FactColumn {
                handle: handles[1],
                values: ColumnValues::U64(depth.into_iter().map(Some).collect()),
            },
            FactColumn {
                handle: handles[2],
                values: ColumnValues::U64(earliest.into_iter().map(Some).collect()),
            },
            FactColumn {
                handle: handles[3],
                values: ColumnValues::U64(reuse.into_iter().map(Some).collect()),
            },
            FactColumn {
                handle: handles[4],
                values: ColumnValues::U64(cost.into_iter().map(Some).collect()),
            },
        ],
    )
}

fn input() -> ScheduleInput {
    ScheduleInput {
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
    }
}

fn expect_prepared<T: std::fmt::Debug>(
    invocation: Invocation<PreparedDecision<T>>,
    operation: &str,
) -> Result<PreparedDecision<T>, Box<dyn std::error::Error>> {
    match invocation {
        Invocation::Success(prepared) => Ok(prepared),
        Invocation::Unavailable => Err(format!("{operation} unexpectedly unavailable").into()),
        Invocation::FallbackRequired(failure) => {
            Err(format!("{operation} unexpectedly failed: {failure:?}").into())
        }
    }
}

fn expect_success(
    invocation: Invocation<PreparedDecision<pie_plex::ServicePlan>>,
) -> Result<pie_plex::ServicePlan, Box<dyn std::error::Error>> {
    match invocation {
        Invocation::Success(output) => Ok(output.commit().0),
        Invocation::Unavailable => Err("schedule unexpectedly unavailable".into()),
        Invocation::FallbackRequired(failure) => {
            Err(format!("schedule unexpectedly failed: {failure:?}").into())
        }
    }
}

fn expect_failure(
    invocation: Invocation<PreparedDecision<pie_plex::ServicePlan>>,
    expected: InvocationFailureKind,
) -> Result<(), Box<dyn std::error::Error>> {
    match invocation {
        Invocation::FallbackRequired(failure) if failure.kind == expected => Ok(()),
        other => Err(format!("expected {expected:?}, got {other:?}").into()),
    }
}
