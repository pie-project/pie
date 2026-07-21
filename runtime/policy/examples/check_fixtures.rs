use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use pie_plex::{
    ContractVersion, Document, Manifest, Operation, PolicyLimits, rank_route, select_evictions,
    select_schedule,
};
use pie_policy::{
    AttachmentRegistry, Invocation, InvocationFailureKind, LifecycleHost, PlacementOutcome,
    PolicyEngine, PolicyEngineConfig, PolicyPackage, PolicyStateStore, ReplayCommand, ReplayRunner,
    ReplayTrace,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: check_fixtures <component-directory>")?;
    let engine = PolicyEngine::new(PolicyEngineConfig::default())?;

    check_required_lifecycle(&engine, &directory)?;
    check_placement_retry_and_shared_global(&engine, &directory)?;
    check_global_replacement_and_concurrency(&engine, &directory)?;
    check_ownership_and_failure_rollback(&engine, &directory)?;
    check_batch_identity(&engine, &directory)?;
    check_score_budget_and_research_policies(&engine, &directory)?;
    check_feedback_commit_atomicity(&directory)?;
    check_replay(&directory)?;

    println!("PLEX JSON policy fixtures passed");
    Ok(())
}

fn check_required_lifecycle(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = PolicyStateStore::new(json!({
        "config": {"locality_bonus": 1.0e12},
        "model": "example-model"
    }))?;
    let (lifecycle, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_coordinated",
        "coordinated",
        &[
            Operation::Route,
            Operation::Admit,
            Operation::Schedule,
            Operation::Evict,
            Operation::Feedback,
        ],
        state.clone(),
        2,
    )?;

    let initial = lifecycle.create_request(
        "L",
        json!({"prompt": "hello"}),
        json!({"user": "alice", "replace": "old"}),
    )?;
    assert_eq!(initial["facts"]["generation_id"], 0);

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
    let after_admit = state.read_request("L")?;
    assert_eq!(after_admit["fields"]["metadata"]["user"], "alice");
    assert_eq!(after_admit["fields"]["metadata"]["last_hook"], "admit");
    assert_eq!(after_admit["scratch"]["admission_count"], 1);
    assert_eq!(after_admit["fields"]["body"]["prompt"], "hello|route|admit");
    assert!(after_admit["facts"].get("previous_target").is_none());
    assert_eq!(state.read_global()["scratch"]["route_calls"], 1);
    assert_eq!(state.read_global()["fields"]["last_route_request"], "L");

    let schedule = expect_success(lifecycle.invoke_and_apply(
        Operation::Schedule,
        schedule_input(vec![("L", json!({"waiting_ms": 1}))], true),
    ))?;
    assert_eq!(
        schedule.input["requests"]["L"]["fields"]["metadata"]["last_hook"],
        "schedule"
    );
    assert_eq!(state.read_request("L")?["scratch"]["schedule_calls"], 1);

    let evict = expect_success(lifecycle.invoke_and_apply(
        Operation::Evict,
        eviction_input(vec![("unit", Some("L"), 1, json!({"reload_cost": 10.0}))]),
    ))?;
    assert_eq!(
        evict.input["requests"]["L"]["scratch"]["eviction_checks"],
        1
    );

    let feedback = feedback_input(
        "delivery-batch",
        vec![
            ("progress", "L", json!({"committed_tokens": 8})),
            ("tool-boundary", "L", json!({"tool_name": "search"})),
        ],
    );
    expect_success(lifecycle.invoke_and_apply(Operation::Feedback, feedback.clone()))?;
    let after_feedback = state.read_request("L")?;
    assert_eq!(after_feedback["scratch"]["feedback_service"], 8);
    assert_eq!(after_feedback["scratch"]["tool_calls"], 1);
    assert_eq!(state.read_global()["scratch"]["feedback_records"], 2);

    let duplicate =
        expect_success(lifecycle.invoke_and_apply(Operation::Feedback, feedback.clone()))?;
    assert!(duplicate.duplicate_feedback);
    assert_eq!(state.read_request("L")?["scratch"]["feedback_service"], 8);
    assert_eq!(state.read_global()["scratch"]["feedback_records"], 2);

    assert!(
        state.read_request("L")?["facts"]
            .get("previous_target")
            .is_none()
    );
    lifecycle.record_enacted_placement("L", target_id)?;
    lifecycle.merge_request_facts("L", json!({"current_target": "node-a"}))?;
    assert_eq!(
        state.read_request("L")?["facts"]["previous_target"],
        "node-a"
    );

    let continuation = lifecycle.continue_request(
        "L",
        json!({"messages": [{"role": "user", "content": "tool result"}]}),
        json!({"replace": "new", "continuation": true}),
    )?;
    assert_eq!(continuation["facts"]["generation_id"], 1);
    assert_eq!(continuation["facts"]["previous_target"], "node-a");
    assert!(continuation["facts"].get("current_target").is_none());
    assert_eq!(continuation["fields"]["metadata"]["user"], "alice");
    assert_eq!(continuation["fields"]["metadata"]["replace"], "new");
    assert_eq!(continuation["scratch"]["feedback_service"], 8);
    assert_eq!(continuation["scratch"]["tool_calls"], 1);

    let continuation_candidates = vec![
        json!({"id": "node-a", "facts": {
            "queue_depth": 1, "cached_tokens": 0, "has_request_kv": false
        }}),
        json!({"id": "node-b", "facts": {
            "queue_depth": 20, "cached_tokens": 100, "has_request_kv": true
        }}),
    ];
    let placed = expect_success(lifecycle.route_and_admit(
        "L",
        "continuation",
        continuation_candidates,
        context(true),
    ))?;
    let PlacementOutcome::Accepted { target_id, .. } = placed else {
        return Err("continuation generation was not accepted".into());
    };
    assert_eq!(target_id, "node-b");
    let continued = state.read_request("L")?;
    assert_eq!(continued["scratch"]["admission_count"], 2);
    assert_eq!(continued["scratch"]["route_count"], 2);
    assert_eq!(continued["fields"]["metadata"]["last_hook"], "admit");

    expect_success(lifecycle.invoke_and_apply(
        Operation::Schedule,
        schedule_input(vec![("L", json!({"waiting_ms": 2}))], true),
    ))?;
    assert_eq!(state.read_request("L")?["scratch"]["schedule_calls"], 2);

    let before_failed_terminal_global = state.read_global();
    let before_failed_terminal_request = state.read_request("L")?;
    let failed_terminal = feedback_input(
        "failed-terminal",
        vec![("progress", "L", json!({"committed_tokens": 100}))],
    );
    for _ in 0..2 {
        expect_failure(
            lifecycle.feedback_and_remove(failed_terminal.clone(), &["missing".into()]),
            InvocationFailureKind::InvalidOutput,
        )?;
        assert_eq!(state.read_global(), before_failed_terminal_global);
        assert_eq!(state.read_request("L")?, before_failed_terminal_request);
    }

    let terminal = feedback_input("delivery-terminal", vec![("completed", "L", json!({}))]);
    expect_success(lifecycle.feedback_and_remove(terminal.clone(), &["L".into()]))?;
    assert!(state.is_empty());
    let duplicate = expect_success(lifecycle.feedback_and_remove(terminal, &["L".into()]))?;
    assert!(duplicate.duplicate_feedback);
    assert!(state.is_empty());
    assert_eq!(state.read_global()["scratch"]["route_calls"], 2);
    Ok(())
}

fn check_placement_retry_and_shared_global(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach(&package_bytes(
        directory,
        "plex_least_loaded",
        manifest("route-owner", [Operation::Route]),
    )?)?;
    registry.attach(&package_bytes(
        directory,
        "plex_rewrite_admit",
        manifest("admit-owner", [Operation::Admit]),
    )?)?;
    let state = PolicyStateStore::default();
    let lifecycle = LifecycleHost::new(registry, state.clone(), 2);
    lifecycle.create_request("R", json!({"prompt": "retry"}), json!({}))?;

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
    assert_eq!(state.read_request("R")?["scratch"]["admission_count"], 2);
    assert_eq!(state.read_global()["scratch"]["route_owner_calls"], 1);
    assert_eq!(state.read_global()["fields"]["route_owner_calls_seen"], 1);

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
    assert_eq!(state.read_request("R")?["scratch"]["admission_count"], 5);
    assert_eq!(state.read_global()["scratch"]["route_owner_calls"], 2);
    assert_eq!(state.read_global()["fields"]["route_owner_calls_seen"], 2);
    Ok(())
}

fn check_global_replacement_and_concurrency(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = PolicyStateStore::new(json!({"config": {"mode": "test"}}))?;
    let operations = [
        Operation::Route,
        Operation::Admit,
        Operation::Schedule,
        Operation::Evict,
        Operation::Feedback,
    ];
    let (lifecycle, registry, package) = lifecycle_with_policy(
        engine,
        directory,
        "plex_coordinated",
        "replaceable",
        &operations,
        state.clone(),
        1,
    )?;
    for id in ["A", "B", "C", "D"] {
        lifecycle.create_request(id, json!({"prompt": id}), json!({}))?;
    }

    for id in ["A", "B"] {
        expect_success(lifecycle.route_and_admit(
            id,
            "generation-arrival",
            candidates(),
            context(false),
        ))?;
    }
    assert_eq!(state.read_global()["scratch"]["route_calls"], 2);

    let left = lifecycle.clone();
    let right = lifecycle.clone();
    let left_thread = std::thread::spawn(move || {
        assert!(matches!(
            left.invoke_and_apply(
                Operation::Route,
                route_input("C", candidates(), context(false))
            ),
            Invocation::Success(_)
        ));
    });
    let right_thread = std::thread::spawn(move || {
        assert!(matches!(
            right.invoke_and_apply(
                Operation::Route,
                route_input("D", candidates(), context(false))
            ),
            Invocation::Success(_)
        ));
    });
    left_thread
        .join()
        .map_err(|_| "left route thread panicked")?;
    right_thread
        .join()
        .map_err(|_| "right route thread panicked")?;
    assert_eq!(state.read_global()["scratch"]["route_calls"], 4);

    let before_replace = state.read_global();
    registry.replace(&package)?;
    assert_eq!(state.read_global(), before_replace);

    let delivery = feedback_input(
        "replacement-dedup",
        vec![("progress", "A", json!({"committed_tokens": 1}))],
    );
    expect_success(lifecycle.invoke_and_apply(Operation::Feedback, delivery.clone()))?;
    let committed = state.read_global();
    registry.replace(&package)?;
    let duplicate = expect_success(lifecycle.invoke_and_apply(Operation::Feedback, delivery))?;
    assert!(duplicate.duplicate_feedback);
    assert_eq!(state.read_global(), committed);

    lifecycle.reset_global();
    let reset = state.read_global();
    assert_eq!(reset["facts"]["config"]["mode"], "test");
    assert_eq!(reset["fields"], json!({}));
    assert_eq!(reset["scratch"], json!({}));
    Ok(())
}

fn check_ownership_and_failure_rollback(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for (artifact, expected) in [
        (
            "plex_mutate_global_facts",
            InvocationFailureKind::InvalidOutput,
        ),
        ("plex_mutate_identity", InvocationFailureKind::InvalidOutput),
        (
            "plex_mutate_candidate_facts",
            InvocationFailureKind::InvalidOutput,
        ),
        (
            "plex_mutate_candidates",
            InvocationFailureKind::InvalidOutput,
        ),
        (
            "plex_mutate_request_set",
            InvocationFailureKind::InvalidOutput,
        ),
        ("plex_malformed", InvocationFailureKind::InvalidOutput),
        ("plex_nonfinite", InvocationFailureKind::InvalidOutput),
        ("plex_mutate_fail", InvocationFailureKind::PolicyFallback),
        ("plex_fallback", InvocationFailureKind::PolicyFallback),
        ("plex_trap", InvocationFailureKind::Trap),
    ] {
        let state = PolicyStateStore::new(json!({"stable": true}))?;
        let (lifecycle, _, _) = lifecycle_with_policy(
            engine,
            directory,
            artifact,
            artifact,
            &[Operation::Route],
            state.clone(),
            1,
        )?;
        lifecycle.create_request("F", json!({"prompt": "stable"}), json!({}))?;
        let before_global = state.read_global();
        let before_request = state.read_request("F")?;
        expect_failure(
            lifecycle.invoke_and_apply(
                Operation::Route,
                route_input("F", candidates(), context(false)),
            ),
            expected,
        )?;
        assert_eq!(state.read_global(), before_global);
        assert_eq!(state.read_request("F")?, before_request);
    }

    let state = PolicyStateStore::default();
    let (lifecycle, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_spin",
        "spin",
        &[Operation::Route],
        state.clone(),
        1,
    )?;
    lifecycle.create_request("S", json!({}), json!({}))?;
    expect_failure_one_of(
        lifecycle.invoke_and_apply(
            Operation::Route,
            route_input("S", candidates(), context(false)),
        ),
        &[
            InvocationFailureKind::FuelExhausted,
            InvocationFailureKind::DeadlineExceeded,
        ],
    )?;
    assert_eq!(state.read_request("S")?["scratch"], json!({}));

    let state = PolicyStateStore::default();
    let (lifecycle, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_mutate_feedback_facts",
        "mutate-feedback-facts",
        &[Operation::Feedback],
        state.clone(),
        1,
    )?;
    lifecycle.create_request("E", json!({}), json!({}))?;
    let before_global = state.read_global();
    let before_request = state.read_request("E")?;
    let input = feedback_input(
        "read-only-event",
        vec![("progress", "E", json!({"committed_tokens": 4}))],
    );
    for _ in 0..2 {
        expect_failure(
            lifecycle.invoke_and_apply(Operation::Feedback, input.clone()),
            InvocationFailureKind::InvalidOutput,
        )?;
        assert_eq!(state.read_global(), before_global);
        assert_eq!(state.read_request("E")?, before_request);
    }

    let state = PolicyStateStore::default();
    let (lifecycle, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_feedback_accounting",
        "feedback-facts",
        &[Operation::Feedback],
        state.clone(),
        1,
    )?;
    lifecycle.create_request("E", json!({}), json!({}))?;
    let before = state.read_request("E")?;
    expect_failure(
        lifecycle.invoke_and_apply(
            Operation::Feedback,
            feedback_input("missing", vec![("progress", "missing", json!({}))]),
        ),
        InvocationFailureKind::InvalidInput,
    )?;
    assert_eq!(state.read_request("E")?, before);
    Ok(())
}

fn check_batch_identity(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = PolicyStateStore::default();
    let (lifecycle, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_attained_service",
        "batch",
        &[Operation::Schedule],
        state.clone(),
        1,
    )?;
    for id in ["L", "M", "unreferenced"] {
        lifecycle.create_request(id, json!({}), json!({}))?;
        lifecycle.merge_request_facts(id, json!({"attained_service": 0}))?;
    }
    let response = expect_success(lifecycle.invoke_and_apply(
        Operation::Schedule,
        schedule_input(
            vec![
                ("L", json!({"waiting_ms": 1})),
                ("L", json!({"waiting_ms": 2})),
                ("M", json!({"waiting_ms": 3})),
            ],
            false,
        ),
    ))?;
    assert_eq!(response.input["requests"].as_object().unwrap().len(), 2);
    assert!(response.input["requests"].get("unreferenced").is_none());
    assert_eq!(state.read_request("L")?["scratch"]["schedule_calls"], 2);
    assert_eq!(state.read_request("M")?["scratch"]["schedule_calls"], 1);

    let before = state.read_global();
    expect_failure(
        lifecycle.invoke_and_apply(
            Operation::Schedule,
            schedule_input(vec![("missing", json!({}))], false),
        ),
        InvocationFailureKind::InvalidInput,
    )?;
    assert_eq!(state.read_global(), before);
    Ok(())
}

fn check_score_budget_and_research_policies(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = PolicyStateStore::default();
    let (route, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_least_loaded",
        "least-loaded",
        &[Operation::Route],
        state.clone(),
        1,
    )?;
    route.create_request("S", json!({}), json!({}))?;
    let response = expect_success(route.invoke_and_apply(
        Operation::Route,
        route_input(
            "S",
            vec![
                json!({"id": "a", "facts": {"queue_depth": 10}}),
                json!({"id": "b", "facts": {"queue_depth": 10}}),
            ],
            context(false),
        ),
    ))?;
    assert_eq!(rank_route(&response.result, 2)?, vec![0, 1]);

    let state = PolicyStateStore::default();
    let (evict, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_retention_score",
        "retention",
        &[Operation::Evict],
        state.clone(),
        1,
    )?;
    for id in ["E1", "E2"] {
        evict.create_request(id, json!({}), json!({}))?;
    }
    let response = expect_success(evict.invoke_and_apply(
        Operation::Evict,
        eviction_input(vec![
            ("a", Some("E1"), 4, json!({"reload_cost": 2.0})),
            ("b", Some("E2"), 3, json!({"reload_cost": 1.0})),
        ]),
    ))?;
    let selected = select_evictions(&response.input, &response.result)?;
    assert_eq!(selected[0].candidate_index, 1);
    assert_eq!(selected[1].candidate_index, 0);

    let state = PolicyStateStore::default();
    let (budget, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_bad_budget",
        "bad-budget",
        &[Operation::Schedule],
        state.clone(),
        1,
    )?;
    budget.create_request("B", json!({}), json!({}))?;
    expect_failure(
        budget.invoke_and_apply(
            Operation::Schedule,
            schedule_input(vec![("B", json!({}))], true),
        ),
        InvocationFailureKind::InvalidOutput,
    )?;

    let state = PolicyStateStore::default();
    let (attained, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_attained_service",
        "attained",
        &[Operation::Schedule],
        state.clone(),
        1,
    )?;
    for (id, service) in [("A", 8), ("B", 1)] {
        attained.create_request(id, json!({}), json!({}))?;
        attained.merge_request_facts(id, json!({"attained_service": service}))?;
    }
    let response = expect_success(attained.invoke_and_apply(
        Operation::Schedule,
        schedule_input(vec![("A", json!({})), ("B", json!({}))], false),
    ))?;
    assert_eq!(
        select_schedule(&response.input, &response.result)?[0].candidate_index,
        1
    );

    check_paper_policies(engine, directory)?;
    Ok(())
}

fn check_paper_policies(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read("tests/policies/paper-cases.json")?)?;
    assert_eq!(cases.as_array().map_or(0, Vec::len), 5);

    let state = PolicyStateStore::default();
    let (agentix, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_agentix",
        "agentix",
        &[Operation::Schedule],
        state.clone(),
        1,
    )?;
    for (id, service) in [("A", 100), ("B", 1)] {
        agentix.create_request(id, json!({}), json!({}))?;
        agentix.merge_request_facts(id, json!({"attained_service": service}))?;
    }
    let response = expect_success(agentix.invoke_and_apply(
        Operation::Schedule,
        schedule_input(
            vec![
                ("A", json!({"waiting_ms": 500})),
                ("B", json!({"waiting_ms": 1})),
            ],
            false,
        ),
    ))?;
    assert!(
        response.result["decisions"][0]["score"].as_f64().unwrap()
            > response.result["decisions"][1]["score"].as_f64().unwrap()
    );

    let state = PolicyStateStore::default();
    let (preble, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_preble",
        "preble",
        &[Operation::Route],
        state.clone(),
        1,
    )?;
    preble.create_request("P", json!({}), json!({}))?;
    let response = expect_success(preble.invoke_and_apply(
        Operation::Route,
        route_input(
            "P",
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
        ),
    ))?;
    assert_eq!(rank_route(&response.result, 2)?[0], 0);

    let state = PolicyStateStore::default();
    let (kvflow, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_kvflow",
        "kvflow",
        &[Operation::Evict],
        state.clone(),
        1,
    )?;
    kvflow.create_request("K", json!({}), json!({}))?;
    expect_success(kvflow.invoke_and_apply(
        Operation::Evict,
        eviction_input(vec![(
            "u",
            Some("K"),
            1,
            json!({"steps_to_execution": 5, "fixed_prefix": false}),
        )]),
    ))?;

    let state = PolicyStateStore::default();
    let (helium, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_helium",
        "helium",
        &[Operation::Schedule],
        state.clone(),
        1,
    )?;
    helium.create_request("H", json!({}), json!({}))?;
    expect_success(helium.invoke_and_apply(
        Operation::Schedule,
        schedule_input(
            vec![(
                "H",
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
    ))?;

    let state = PolicyStateStore::default();
    let (continuum, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_continuum",
        "continuum",
        &[Operation::Schedule, Operation::Evict, Operation::Feedback],
        state.clone(),
        1,
    )?;
    continuum.create_request("C", json!({}), json!({}))?;
    expect_success(continuum.invoke_and_apply(
        Operation::Feedback,
        feedback_input("ttl", vec![("tool-boundary", "C", json!({"ttl_ms": 100}))]),
    ))?;
    expect_success(continuum.invoke_and_apply(
        Operation::Schedule,
        schedule_input(
            vec![("C", json!({"preempted": false, "program_arrival": 1}))],
            false,
        ),
    ))?;
    Ok(())
}

fn check_feedback_commit_atomicity(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let engine = PolicyEngine::new(PolicyEngineConfig {
        max_feedback_deliveries: 1,
        ..PolicyEngineConfig::default()
    })?;
    let state = PolicyStateStore::default();
    let (lifecycle, _, _) = lifecycle_with_policy(
        &engine,
        directory,
        "plex_feedback_accounting",
        "bounded-feedback",
        &[Operation::Feedback],
        state.clone(),
        1,
    )?;
    lifecycle.create_request("L", json!({}), json!({}))?;

    expect_success(lifecycle.invoke_and_apply(
        Operation::Feedback,
        feedback_input(
            "accepted",
            vec![("progress", "L", json!({"committed_tokens": 4}))],
        ),
    ))?;
    assert_eq!(state.read_request("L")?["scratch"]["attained_service"], 4);
    let committed = state.read_global();

    let rejected = feedback_input("not-committed", vec![("tool-boundary", "L", json!({}))]);
    for _ in 0..2 {
        expect_failure(
            lifecycle.invoke_and_apply(Operation::Feedback, rejected.clone()),
            InvocationFailureKind::HostSaturated,
        )?;
        assert_eq!(
            state.read_request("L")?["scratch"]["tool_calls"],
            json!(null)
        );
        assert_eq!(state.read_global(), committed);
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
            ReplayCommand::ReplaceGlobalFacts {
                facts: json!({"config": {"mode": "replay"}}),
            },
            ReplayCommand::CreateRequest {
                logical_request_id: "L".into(),
                body: json!({"prompt": "hello"}),
                metadata: json!({"user": "alice"}),
            },
            ReplayCommand::MergeGlobalFacts {
                facts: json!({"replica_count": 2}),
            },
            ReplayCommand::MergeRequestFacts {
                logical_request_id: "L".into(),
                facts: json!({"attained_service": 0}),
            },
            ReplayCommand::RouteAdmit {
                logical_request_id: "L".into(),
                cause: "generation-arrival".into(),
                candidates: candidates(),
                context: context(true),
            },
            ReplayCommand::RecordEnactedPlacement {
                logical_request_id: "L".into(),
                target_id: "node-a".into(),
            },
            ReplayCommand::Invoke {
                operation: Operation::Schedule,
                input: schedule_input(vec![("L", json!({}))], true),
            },
            ReplayCommand::Invoke {
                operation: Operation::Feedback,
                input: feedback_input("d", vec![("progress", "L", json!({"committed_tokens": 4}))]),
            },
            ReplayCommand::ContinueRequest {
                logical_request_id: "L".into(),
                body: json!({"prompt": "continue"}),
                metadata: json!({"step": 2}),
            },
            ReplayCommand::ReadGlobal,
            ReplayCommand::ReadRequest {
                logical_request_id: "L".into(),
            },
        ],
    };
    let packages = BTreeMap::from([("policy".into(), package)]);
    let first = ReplayRunner::new(
        AttachmentRegistry::new(engine.clone()),
        PolicyStateStore::default(),
        packages.clone(),
        2,
    )?
    .run(&trace)?;
    ReplayRunner::new(
        AttachmentRegistry::new(engine),
        PolicyStateStore::default(),
        packages,
        2,
    )?
    .verify(&trace, &first)?;
    Ok(())
}

fn lifecycle_with_policy(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    package_name: &str,
    operations: &[Operation],
    state: PolicyStateStore,
    max_defer_retries: u32,
) -> Result<(LifecycleHost, AttachmentRegistry, Vec<u8>), Box<dyn std::error::Error>> {
    let package = package_bytes(
        directory,
        artifact,
        manifest(package_name, operations.iter().copied()),
    )?;
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach(&package)?;
    let lifecycle = LifecycleHost::new(registry.clone(), state, max_defer_retries);
    Ok((lifecycle, registry, package))
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
        contract: ContractVersion::V0_3,
        package_name: name.replace('_', "-"),
        package_version: "0.3.0".into(),
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

fn candidates() -> Vec<Document> {
    vec![
        json!({"id": "node-a", "facts": {
            "queue_depth": 10, "cached_tokens": 100, "has_request_kv": true
        }}),
        json!({"id": "node-b", "facts": {
            "queue_depth": 20, "cached_tokens": 0, "has_request_kv": false
        }}),
    ]
}

fn context(token_budget: bool) -> Document {
    json!({"capabilities": {"token_budget": token_budget}})
}

fn route_input(request_id: &str, candidates: Vec<Document>, context: Document) -> Document {
    json!({
        "cause": "generation-arrival",
        "request_id": request_id,
        "candidates": candidates,
        "context": context
    })
}

fn schedule_input(runnable: Vec<(&str, Document)>, token_budget: bool) -> Document {
    json!({
        "cause": "service-step",
        "runnable": runnable
            .into_iter()
            .map(|(request_id, facts)| json!({
                "request_id": request_id,
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

fn eviction_input(resident: Vec<(&str, Option<&str>, u64, Document)>) -> Document {
    json!({
        "cause": "allocation-deficit",
        "bytes_needed": 6,
        "resident": resident
            .into_iter()
            .map(|(id, request_id, size_bytes, facts)| json!({
                "id": id,
                "request_id": request_id,
                "size_bytes": size_bytes,
                "facts": facts
            }))
            .collect::<Vec<_>>(),
        "context": context(false)
    })
}

fn feedback_input(delivery_id: &str, records: Vec<(&str, &str, Document)>) -> Document {
    json!({
        "delivery_id": delivery_id,
        "records": records
            .into_iter()
            .map(|(event, request_id, facts)| json!({
                "event": event,
                "request_id": request_id,
                "facts": facts
            }))
            .collect::<Vec<_>>(),
        "context": context(false)
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

fn expect_failure_one_of<T: std::fmt::Debug>(
    invocation: Invocation<T>,
    expected: &[InvocationFailureKind],
) -> Result<(), Box<dyn std::error::Error>> {
    match invocation {
        Invocation::FallbackRequired(failure) if expected.contains(&failure.kind) => Ok(()),
        other => Err(format!("expected one of {expected:?}, got {other:?}").into()),
    }
}
