use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use pie_plex::{
    ContractVersion, Document, Manifest, Operation, PolicyLimits, rank_route, select_evictions,
    select_schedule,
};
use pie_policy::{
    AttachedPolicy, AttachmentRegistry, CanonicalRequestStore, Invocation, InvocationFailureKind,
    LifecycleHost, PlacementOutcome, PolicyEngine, PolicyEngineConfig, PolicyPackage,
    ReplayCommand, ReplayRunner, ReplayTrace,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: check_fixtures <component-directory>")?;
    let engine = PolicyEngine::new(PolicyEngineConfig::default())?;

    check_required_lifecycle(&engine, &directory)?;
    check_placement_retry_semantics(&engine, &directory)?;
    check_failure_rollback(&engine, &directory)?;
    check_score_and_budget_rules(&engine, &directory)?;
    check_paper_policies(&engine, &directory)?;
    check_replay(&directory)?;

    println!("PLEX JSON policy fixtures passed");
    Ok(())
}

fn check_required_lifecycle(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let package = package_bytes(
        directory,
        "plex_coordinated",
        manifest(
            "coordinated",
            [
                Operation::Route,
                Operation::Admit,
                Operation::Schedule,
                Operation::Evict,
                Operation::Feedback,
            ],
        ),
    )?;
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach(&package)?;
    let store = CanonicalRequestStore::default();
    let lifecycle = LifecycleHost::new(registry, store.clone(), 2);

    let initial = store.create(
        "L",
        json!({"prompt": "hello"}),
        json!({"user": "alice", "replace": "old"}),
    )?;
    assert_eq!(initial["identity"]["generation_id"], 0);

    let placed = expect_success(lifecycle.route_and_admit(
        "L",
        "generation-arrival",
        candidates(),
        context(true),
    ))?;
    let PlacementOutcome::Accepted { target_id, .. } = placed else {
        return Err("initial generation was not accepted".into());
    };
    assert_eq!(target_id, "node-a");
    let after_admit = store.get("L")?;
    assert_eq!(after_admit["metadata"]["user"], "alice");
    assert_eq!(after_admit["metadata"]["last_hook"], "admit");
    assert_eq!(after_admit["state"]["admission_count"], 1);
    assert_eq!(after_admit["body"]["prompt"], "hello|route|admit");

    let schedule = schedule_input(vec![(after_admit, json!({"waiting_ms": 1}))], true);
    let schedule = expect_success(lifecycle.invoke_and_apply(Operation::Schedule, schedule))?;
    assert_eq!(
        schedule.input["runnable"][0]["request"]["metadata"]["last_hook"],
        "schedule"
    );
    assert_eq!(store.get("L")?["state"]["schedule_calls"], 1);

    let evict = eviction_input(store.get("L")?);
    let evict = expect_success(lifecycle.invoke_and_apply(Operation::Evict, evict))?;
    assert_eq!(
        evict.input["resident"][0]["request"]["state"]["eviction_checks"],
        1
    );

    let progress = feedback_input(
        "delivery-progress",
        vec![("progress", store.get("L")?, json!({"committed_tokens": 8}))],
    );
    expect_success(lifecycle.invoke_and_apply(Operation::Feedback, progress.clone()))?;
    assert_eq!(store.get("L")?["state"]["attained_service"], 8);

    let duplicate = expect_success(lifecycle.invoke_and_apply(Operation::Feedback, progress))?;
    assert!(duplicate.duplicate_feedback);
    assert_eq!(store.get("L")?["state"]["attained_service"], 8);

    let boundary = feedback_input(
        "delivery-boundary",
        vec![(
            "tool-boundary",
            store.get("L")?,
            json!({"tool_name": "search"}),
        )],
    );
    expect_success(lifecycle.invoke_and_apply(Operation::Feedback, boundary))?;
    assert_eq!(store.get("L")?["state"]["tool_calls"], 1);

    let continuation = store.continuation(
        "L",
        json!({"messages": [{"role": "user", "content": "tool result"}]}),
        json!({"replace": "new", "continuation": true}),
    )?;
    assert_eq!(continuation["identity"]["generation_id"], 1);
    assert_eq!(continuation["metadata"]["user"], "alice");
    assert_eq!(continuation["metadata"]["replace"], "new");
    assert_eq!(continuation["state"]["attained_service"], 8);
    assert_eq!(continuation["state"]["tool_calls"], 1);

    let placed = expect_success(lifecycle.route_and_admit(
        "L",
        "continuation",
        candidates(),
        context(true),
    ))?;
    assert!(matches!(placed, PlacementOutcome::Accepted { .. }));
    let continued = store.get("L")?;
    assert_eq!(continued["state"]["admission_count"], 2);
    assert_eq!(continued["metadata"]["last_hook"], "admit");

    let schedule = schedule_input(vec![(continued, json!({"waiting_ms": 2}))], true);
    expect_success(lifecycle.invoke_and_apply(Operation::Schedule, schedule))?;
    assert_eq!(store.get("L")?["state"]["schedule_calls"], 2);

    let terminal = feedback_input(
        "delivery-terminal",
        vec![("completed", store.get("L")?, json!({}))],
    );
    expect_success(lifecycle.feedback_and_remove(terminal, &["L".into()]))?;
    assert!(store.is_empty());
    Ok(())
}

fn check_placement_retry_semantics(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach(&package_bytes(
        directory,
        "plex_least_loaded",
        manifest("route", [Operation::Route]),
    )?)?;
    registry.attach(&package_bytes(
        directory,
        "plex_rewrite_admit",
        manifest("admit", [Operation::Admit]),
    )?)?;
    let store = CanonicalRequestStore::default();
    let lifecycle = LifecycleHost::new(registry, store.clone(), 2);
    store.create("R", json!({"prompt": "retry"}), json!({}))?;

    let rejected = expect_success(lifecycle.route_and_admit(
        "R",
        "generation-arrival",
        vec![
            json!({"id": "a", "facts": {"queue_depth": 100}}),
            json!({"id": "b", "facts": {"queue_depth": 101}}),
        ],
        context(false),
    ))?;
    assert_eq!(rejected, PlacementOutcome::Rejected);
    assert_eq!(store.get("R")?["state"]["admission_count"], 2);

    let deferred = expect_success(lifecycle.route_and_admit(
        "R",
        "generation-arrival",
        vec![json!({"id": "c", "facts": {"queue_depth": 90}})],
        context(false),
    ))?;
    assert!(matches!(
        deferred,
        PlacementOutcome::Deferred { attempts: 3, .. }
    ));
    assert_eq!(store.get("R")?["state"]["admission_count"], 5);
    Ok(())
}

fn check_failure_rollback(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for (artifact, expected) in [
        ("plex_malformed", InvocationFailureKind::InvalidOutput),
        ("plex_nonfinite", InvocationFailureKind::InvalidOutput),
        ("plex_mutate_identity", InvocationFailureKind::InvalidOutput),
        (
            "plex_mutate_candidates",
            InvocationFailureKind::InvalidOutput,
        ),
        ("plex_mutate_fail", InvocationFailureKind::PolicyFallback),
        ("plex_trap", InvocationFailureKind::Trap),
    ] {
        let policy = attach_single(engine, directory, artifact, Operation::Route)?;
        let store = CanonicalRequestStore::default();
        store.create("F", json!({"prompt": "stable"}), json!({}))?;
        let lifecycle = LifecycleHost::new(registry_with(engine, policy)?, store.clone(), 1);
        let input = route_input(store.get("F")?, candidates(), context(false));
        expect_failure(
            lifecycle.invoke_and_apply(Operation::Route, input),
            expected,
        )?;
        assert_eq!(store.get("F")?["body"]["prompt"], "stable");
        assert_eq!(store.get("F")?["state"], json!({}));
    }

    let infinity = attach_single(engine, directory, "plex_nonfinite", Operation::Evict)?;
    expect_failure(
        infinity.evict(json!({
            "cause": "allocation-deficit",
            "bytes_needed": 1,
            "resident": [{
                "id": "u",
                "size_bytes": 1,
                "request": request("I", 0),
                "facts": {}
            }],
            "context": {"config": {}}
        })),
        InvocationFailureKind::InvalidOutput,
    )?;

    let fallback = attach_single(engine, directory, "plex_fallback", Operation::Route)?;
    expect_failure(
        fallback.route(route_input(
            request("fallback", 0),
            candidates(),
            context(false),
        )),
        InvocationFailureKind::PolicyFallback,
    )?;
    Ok(())
}

fn check_score_and_budget_rules(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let route = attach_single(engine, directory, "plex_least_loaded", Operation::Route)?;
    let route = expect_success(route.route(route_input(
        request("S", 0),
        vec![
            json!({"id": "a", "facts": {"queue_depth": 10}}),
            json!({"id": "b", "facts": {"queue_depth": 10}}),
        ],
        context(false),
    )))?;
    assert_eq!(rank_route(&route.result, 2)?, vec![0, 1]);

    let evict = attach_single(engine, directory, "plex_retention_score", Operation::Evict)?;
    let evict_input = json!({
        "cause": "allocation-deficit",
        "bytes_needed": 6,
        "resident": [
            {
                "id": "a",
                "size_bytes": 4,
                "request": request("E1", 0),
                "facts": {"reload_cost": 2.0}
            },
            {
                "id": "b",
                "size_bytes": 3,
                "request": request("E2", 0),
                "facts": {"reload_cost": 1.0}
            }
        ],
        "context": {"config": {}}
    });
    let evict = expect_success(evict.evict(evict_input))?;
    let selected = select_evictions(&evict.input, &evict.result)?;
    assert_eq!(selected[0].candidate_index, 1);
    assert_eq!(selected[1].candidate_index, 0);

    let budget = attach_single(engine, directory, "plex_bad_budget", Operation::Schedule)?;
    let schedule = schedule_input(vec![(request("B", 0), json!({}))], true);
    expect_failure(
        budget.schedule(schedule),
        InvocationFailureKind::InvalidOutput,
    )?;

    let attained = attach_single(
        engine,
        directory,
        "plex_attained_service",
        Operation::Schedule,
    )?;
    let schedule = schedule_input(
        vec![
            (
                request_with_state("A", 0, json!({"attained_service": 8})),
                json!({}),
            ),
            (
                request_with_state("B", 0, json!({"attained_service": 1})),
                json!({}),
            ),
        ],
        false,
    );
    let response = expect_success(attained.schedule(schedule))?;
    assert_eq!(
        select_schedule(&response.input, &response.result)?[0].candidate_index,
        1
    );
    Ok(())
}

fn check_paper_policies(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read("tests/policies/paper-cases.json")?)?;
    assert_eq!(cases.as_array().map_or(0, Vec::len), 5);

    let agentix = attach_single(engine, directory, "plex_paper_agentix", Operation::Schedule)?;
    let response = expect_success(agentix.schedule(schedule_input(
        vec![
            (
                request_with_state("A", 0, json!({"attained_service": 100})),
                json!({"waiting_ms": 500}),
            ),
            (
                request_with_state("B", 0, json!({"attained_service": 1})),
                json!({"waiting_ms": 1}),
            ),
        ],
        false,
    )))?;
    assert!(
        response.result["decisions"][0]["score"].as_f64().unwrap()
            > response.result["decisions"][1]["score"].as_f64().unwrap()
    );

    let preble = attach_single(engine, directory, "plex_paper_preble", Operation::Route)?;
    let response = expect_success(preble.route(route_input(
        request("P", 0),
        vec![
            json!({"id": "a", "facts": {
                "cached_tokens": 100, "uncached_tokens": 50,
                "load_cost": 100, "eviction_cost": 0
            }}),
            json!({"id": "b", "facts": {
                "cached_tokens": 80, "uncached_tokens": 50,
                "load_cost": 1, "eviction_cost": 0
            }}),
        ],
        context(false),
    )))?;
    assert_eq!(rank_route(&response.result, 2)?[0], 0);

    for (artifact, operation, input) in [
        (
            "plex_paper_kvflow",
            Operation::Evict,
            json!({
                "cause": "allocation-deficit",
                "bytes_needed": 1,
                "resident": [{
                    "id": "u",
                    "size_bytes": 1,
                    "request": request("K", 0),
                    "facts": {"steps_to_execution": 5, "fixed_prefix": false}
                }],
                "context": {"config": {}}
            }),
        ),
        (
            "plex_paper_helium",
            Operation::Schedule,
            schedule_input(
                vec![(
                    request("H", 0),
                    json!({
                        "ready": true,
                        "dependency_depth": 2,
                        "earliest_start": 0,
                        "prefix_reuse_tokens": 10,
                        "profiled_token_cost": 1
                    }),
                )],
                false,
            ),
        ),
        (
            "plex_paper_continuum",
            Operation::Schedule,
            schedule_input(
                vec![(
                    request_with_state("C", 0, json!({"ttl_active": true})),
                    json!({
                        "preempted": false,
                        "program_arrival": 1
                    }),
                )],
                false,
            ),
        ),
    ] {
        let policy = attach_single(engine, directory, artifact, operation)?;
        let invocation = match operation {
            Operation::Schedule => policy.schedule(input),
            Operation::Evict => policy.evict(input),
            _ => unreachable!(),
        };
        expect_success(invocation)?;
    }
    Ok(())
}

fn check_replay(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let engine = PolicyEngine::new(PolicyEngineConfig::deterministic_replay())?;
    let package = package_bytes(
        directory,
        "plex_coordinated",
        manifest(
            "replay",
            [
                Operation::Route,
                Operation::Admit,
                Operation::Schedule,
                Operation::Feedback,
            ],
        ),
    )?;
    let trace = ReplayTrace {
        commands: vec![
            ReplayCommand::Attach {
                package: "policy".into(),
            },
            ReplayCommand::CreateRequest {
                logical_request_id: "L".into(),
                body: json!({"prompt": "hello"}),
                metadata: json!({"user": "alice"}),
            },
            ReplayCommand::RouteAdmit {
                logical_request_id: "L".into(),
                cause: "generation-arrival".into(),
                candidates: candidates(),
                context: context(true),
            },
            ReplayCommand::Invoke {
                operation: Operation::Schedule,
                input: schedule_input(vec![(request("L", 0), json!({}))], true),
            },
            ReplayCommand::Invoke {
                operation: Operation::Feedback,
                input: feedback_input(
                    "d",
                    vec![("progress", request("L", 0), json!({"committed_tokens": 4}))],
                ),
            },
            ReplayCommand::ContinueRequest {
                logical_request_id: "L".into(),
                body: json!({"prompt": "continue"}),
                metadata: json!({"step": 2}),
            },
            ReplayCommand::ReadRequest {
                logical_request_id: "L".into(),
            },
        ],
    };
    let packages = BTreeMap::from([("policy".into(), package)]);
    let first = ReplayRunner::new(
        AttachmentRegistry::new(engine.clone()),
        CanonicalRequestStore::default(),
        packages.clone(),
        2,
    )?
    .run(&trace)?;
    ReplayRunner::new(
        AttachmentRegistry::new(engine),
        CanonicalRequestStore::default(),
        packages,
        2,
    )?
    .verify(&trace, &first)?;
    Ok(())
}

fn attach_single(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    operation: Operation,
) -> Result<AttachedPolicy, Box<dyn std::error::Error>> {
    let component = std::fs::read(directory.join(format!("{artifact}.component.wasm")))?;
    AttachedPolicy::compile(engine.clone(), &component, manifest(artifact, [operation]))
        .map_err(Into::into)
}

fn registry_with(
    engine: &PolicyEngine,
    policy: AttachedPolicy,
) -> Result<AttachmentRegistry, Box<dyn std::error::Error>> {
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach_prepared(policy)?;
    Ok(registry)
}

fn package_bytes(
    directory: &Path,
    artifact: &str,
    manifest: Manifest,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let component = std::fs::read(directory.join(format!("{artifact}.component.wasm")))?;
    Ok(PolicyPackage::new(manifest, component)?.encode()?)
}

fn manifest(name: &str, operations: impl IntoIterator<Item = Operation>) -> Manifest {
    Manifest {
        contract: ContractVersion::V0_2,
        package_name: name.replace('_', "-"),
        package_version: "0.2.0".into(),
        operations: operations.into_iter().collect::<BTreeSet<_>>(),
        limits: PolicyLimits {
            memory_bytes: 4 << 20,
            fuel: 2_000_000,
            deadline_ms: 100,
            input_bytes: 1 << 20,
            output_bytes: 1 << 20,
        },
    }
}

fn request(id: &str, generation: u64) -> Document {
    request_with_state(id, generation, json!({}))
}

fn request_with_state(id: &str, generation: u64, state: Document) -> Document {
    json!({
        "identity": {
            "logical_request_id": id,
            "generation_id": generation
        },
        "body": {"prompt": "hello"},
        "metadata": {},
        "state": state
    })
}

fn candidates() -> Vec<Document> {
    vec![
        json!({"id": "node-a", "facts": {"queue_depth": 10, "cached_tokens": 100}}),
        json!({"id": "node-b", "facts": {"queue_depth": 20, "cached_tokens": 0}}),
    ]
}

fn context(token_budget: bool) -> Document {
    json!({
        "config": {},
        "capabilities": {"token_budget": token_budget}
    })
}

fn route_input(request: Document, candidates: Vec<Document>, context: Document) -> Document {
    json!({
        "cause": "generation-arrival",
        "request": request,
        "candidates": candidates,
        "context": context
    })
}

fn schedule_input(runnable: Vec<(Document, Document)>, token_budget: bool) -> Document {
    json!({
        "cause": "service-step",
        "runnable": runnable
            .into_iter()
            .map(|(request, facts)| json!({
                "request": request,
                "facts": facts,
                "max_token_budget": 8
            }))
            .collect::<Vec<_>>(),
        "capacity": {
            "max_selected": 8,
            "max_total_tokens": 64,
            "max_token_budget": 8
        },
        "context": context(token_budget)
    })
}

fn eviction_input(request: Document) -> Document {
    json!({
        "cause": "allocation-deficit",
        "bytes_needed": 1,
        "resident": [{
            "id": "unit",
            "size_bytes": 1,
            "request": request,
            "facts": {"reload_cost": 10.0}
        }],
        "context": {"config": {}}
    })
}

fn feedback_input(delivery_id: &str, records: Vec<(&str, Document, Document)>) -> Document {
    json!({
        "delivery_id": delivery_id,
        "records": records
            .into_iter()
            .map(|(event, request, facts)| json!({
                "event": event,
                "request": request,
                "facts": facts
            }))
            .collect::<Vec<_>>(),
        "context": {"config": {}}
    })
}

fn expect_success<T>(invocation: Invocation<T>) -> Result<T, Box<dyn std::error::Error>> {
    match invocation {
        Invocation::Success(value) => Ok(value),
        Invocation::Unavailable => Err("operation unexpectedly unavailable".into()),
        Invocation::FallbackRequired(failure) => {
            Err(format!("operation unexpectedly fell back: {failure:?}").into())
        }
    }
}

fn expect_failure<T: std::fmt::Debug>(
    invocation: Invocation<T>,
    expected: InvocationFailureKind,
) -> Result<(), Box<dyn std::error::Error>> {
    match invocation {
        Invocation::FallbackRequired(failure) if failure.kind == expected => Ok(()),
        other => Err(format!("expected {expected:?}, got {other:?}").into()),
    }
}
