use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use pie_plex::{
    ContractVersion, Document, Manifest, Operation, PolicyLimits, rank_route, select_evictions,
    select_schedule,
};
use pie_policy::{
    AttachmentRegistry, DictionaryQueryHandler, ENGINE_API_VERSION, FeedbackCommit,
    InMemoryPolicyStateBackend, Invocation, InvocationFailureKind, LifecycleHost, PlacementOutcome,
    PlexError, PlexRuntime, PolicyEngine, PolicyEngineConfig, PolicyPackage, PolicyStateBackend,
    QueryError, QueryHandler, RejectingQueryHandler, ReplayCommand, ReplayRunner, ReplayTrace,
    StateBackendError, StateSnapshot, StateUpdates,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: check_fixtures <component-directory>")?;
    let engine = PolicyEngine::new(PolicyEngineConfig::default())?;

    check_explicit_lifecycle(&engine, &directory)?;
    check_engine_api(&engine, &directory)?;
    check_request_events_and_cleanup(&engine, &directory)?;
    check_working_sets_and_decisions(&engine, &directory)?;
    check_queries_actions_and_conflicts(&engine, &directory)?;
    check_failure_rollback(&engine, &directory)?;
    check_pool_contention(&directory)?;
    check_feedback_dedup(&directory)?;
    check_replay(&directory)?;
    check_adapter_conformance(&engine, &directory)?;
    check_paper_policies(&engine, &directory)?;

    println!("PLEX JSON policy fixtures passed");
    Ok(())
}

fn check_pool_contention(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let engine = PolicyEngine::new(PolicyEngineConfig {
        max_concurrent_invocations: 1,
        epoch_tick: None,
        ..PolicyEngineConfig::default()
    })?;
    let package = package_bytes(
        directory,
        "plex_paper_helium",
        manifest("pool-contention", [Operation::Schedule]),
    )?;
    let registry = AttachmentRegistry::new(engine);
    registry.attach(&package)?;
    let snapshot = Arc::new(registry.snapshot()?);
    let context = schedule_context(
        vec![(
            "P",
            json!({
                "ready": true,
                "dependency_depth": 3,
                "prefix_reuse_tokens": 256,
                "earliest_start": 0,
                "profiled_token_cost": 32
            }),
        )],
        false,
    );
    let request = json!({
        "facts": {
            "logical_request_id": "P",
            "generation_id": 0
        },
        "fields": {},
        "scratch": {}
    });
    let state = StateSnapshot::from_parts(
        json!({}),
        BTreeMap::from([("P".into(), request)]),
        0,
        BTreeMap::from([("P".into(), 0)]),
    )?;
    let barrier = Arc::new(Barrier::new(5));
    let mut workers = Vec::new();
    for _ in 0..4 {
        let snapshot = snapshot.clone();
        let context = context.clone();
        let state = state.clone();
        let barrier = barrier.clone();
        workers.push(thread::spawn(move || -> Result<(u64, u64), String> {
            barrier.wait();
            let mut successes = 0;
            let mut saturated = 0;
            for _ in 0..2_000 {
                match snapshot.invoke(
                    Operation::Schedule,
                    context.clone(),
                    state.clone(),
                    Arc::new(RejectingQueryHandler),
                    Arc::new(BTreeSet::new()),
                ) {
                    Invocation::Success(_) => successes += 1,
                    Invocation::FallbackRequired(failure)
                        if failure.kind == InvocationFailureKind::HostSaturated =>
                    {
                        saturated += 1;
                    }
                    Invocation::FallbackRequired(failure) => {
                        return Err(format!(
                            "unexpected pooled invocation failure {:?}: {}",
                            failure.kind, failure.message
                        ));
                    }
                    Invocation::Unavailable => {
                        return Err("pooled schedule operation became unavailable".into());
                    }
                }
            }
            Ok((successes, saturated))
        }));
    }
    barrier.wait();
    let mut successes = 0;
    let mut saturated = 0;
    for worker in workers {
        let (worker_successes, worker_saturated) = worker
            .join()
            .map_err(|_| "pool contention worker panicked")??;
        successes += worker_successes;
        saturated += worker_saturated;
    }
    assert!(successes > 0);
    assert!(saturated > 0);
    assert_eq!(successes + saturated, 8_000);
    Ok(())
}

fn check_explicit_lifecycle(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(InMemoryPolicyStateBackend::new(json!({
        "tenant_service": {}
    }))?);
    let query = DictionaryQueryHandler::default();
    query.insert("pie.cluster.capacity@1", json!({"route_bias": 0.0}));
    let actions = action_set([
        "pie.kv.prefetch@1",
        "pie.retention.set@1",
        "pie.timer.arm@1",
    ]);
    let (lifecycle, _, _) = lifecycle_with_policy_and_host(
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
        backend.clone(),
        Arc::new(query),
        actions,
        2,
    )?;

    lifecycle.create_request(
        "L",
        json!({"prompt": "hello"}),
        json!({"user": "alice", "replace": "old"}),
    )?;
    let placed = expect_success(lifecycle.route_and_admit(
        "L",
        "generation-arrival",
        candidates(),
        context_with_helpers(
            true,
            &["pie.cluster.capacity@1"],
            &["pie.kv.prefetch@1", "pie.retention.set@1"],
        ),
    ))?;
    let PlacementOutcome::Accepted { target_id, .. } = placed else {
        return Err("initial generation was not accepted".into());
    };
    assert_eq!(target_id, "node-a");
    let after_admit = backend.read_request("L")?;
    assert_eq!(after_admit["fields"]["metadata"]["last_hook"], "admit");
    assert_eq!(after_admit["scratch"]["admission_count"], 1);
    assert_eq!(after_admit["fields"]["body"]["prompt"], "hello|route|admit");
    assert_eq!(backend.read_shared()?["route_calls"], 1);

    let schedule_ctx = schedule_context(vec![("L", json!({"waiting_ms": 1}))], true);
    let schedule =
        expect_success(lifecycle.invoke_and_apply(Operation::Schedule, schedule_ctx.clone()))?;
    assert_eq!(select_schedule(&schedule_ctx, &schedule.result)?.len(), 1);
    assert_eq!(backend.read_request("L")?["scratch"]["schedule_calls"], 1);

    let eviction_ctx = eviction_context(vec![("unit", Some("L"), 8, json!({"reload_cost": 10.0}))]);
    let eviction =
        expect_success(lifecycle.invoke_and_apply(Operation::Evict, eviction_ctx.clone()))?;
    assert_eq!(
        select_evictions(&eviction_ctx, &eviction.result)?[0].candidate_index,
        0
    );

    let feedback = feedback_context(
        "delivery-batch",
        vec![
            ("progress", "L", json!({"committed_tokens": 8})),
            ("tool-boundary", "L", json!({"tool_name": "search"})),
        ],
        context(false),
    );
    expect_success(lifecycle.invoke_and_apply(Operation::Feedback, feedback.clone()))?;
    let after_feedback = backend.read_request("L")?;
    assert_eq!(after_feedback["scratch"]["feedback_service"], 8);
    assert_eq!(after_feedback["scratch"]["tool_calls"], 1);
    let duplicate = expect_success(lifecycle.invoke_and_apply(Operation::Feedback, feedback))?;
    assert!(duplicate.duplicate_feedback);

    lifecycle.record_enacted_placement("L", target_id)?;
    lifecycle.merge_request_facts("L", json!({"current_target": "node-a"}))?;
    let continuation = lifecycle.continue_request(
        "L",
        json!({"messages": [{"role": "user", "content": "tool result"}]}),
        json!({"replace": "new", "continuation": true}),
    )?;
    assert_eq!(continuation["facts"]["generation_id"], 1);
    assert_eq!(continuation["facts"]["previous_target"], "node-a");
    assert!(continuation["facts"].get("current_target").is_none());
    assert_eq!(continuation["scratch"]["feedback_service"], 8);

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
        context(false),
    ))?;
    assert!(matches!(
        placed,
        PlacementOutcome::Accepted { ref target_id, .. } if target_id == "node-b"
    ));
    assert_eq!(backend.read_request("L")?["scratch"]["admission_count"], 2);
    Ok(())
}

fn check_engine_api(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let query = DictionaryQueryHandler::default();
    query.insert(
        "pie.cluster.capacity@1",
        json!({"route_bias": 5.0, "free_slots": 4}),
    );
    let runtime = runtime_with_policy(
        engine,
        directory,
        "plex_coordinated",
        "engine-api",
        &[
            Operation::Route,
            Operation::Admit,
            Operation::Schedule,
            Operation::Feedback,
        ],
        backend.clone(),
        Arc::new(query),
        action_set([
            "pie.kv.prefetch@1",
            "pie.retention.set@1",
            "pie.timer.arm@1",
        ]),
    )?;

    let route = engine_event(
        "route",
        json!({
            "request_id": "L",
            "candidates": candidates(),
            "context": {
                "model": "example-model",
                "capabilities": {
                    "queries": ["pie.cluster.capacity@1"]
                }
            }
        }),
        vec![json!({
            "op": "create",
            "request_id": "L",
            "facts": {
                "logical_request_id": "L",
                "generation_id": 0,
                "attained_service": 0
            },
            "fields": {
                "body": {"prompt": "hello"},
                "metadata": {"user": "alice"}
            }
        })],
    );
    let outcome = runtime.invoke(route)?;
    assert_eq!(outcome["status"], "success");
    assert_eq!(outcome["decision"]["order"], json!([0, 1]));
    assert_eq!(
        outcome["request_fields"]["L"]["body"]["prompt"],
        "hello|route"
    );
    assert_eq!(outcome["actions"][0]["id"], 0);
    assert_eq!(outcome["actions"][0]["method"], "pie.kv.prefetch@1");
    assert!(outcome.get("scratch").is_none());

    let admit = runtime.invoke(engine_event(
        "admit",
        json!({
            "request_id": "L",
            "target": candidates()[0],
            "context": {}
        }),
        vec![],
    ))?;
    assert_eq!(admit["decision"]["decision"], "accept");
    assert_eq!(admit["actions"][0]["id"], 0);
    assert_eq!(admit["actions"][0]["method"], "pie.retention.set@1");
    assert_eq!(
        admit["request_fields"]["L"]["body"]["prompt"],
        "hello|route|admit"
    );

    let schedule = runtime.invoke(engine_event(
        "schedule",
        json!({
            "runnable": [{
                "request_id": "L",
                "facts": {"waiting_ms": 1},
                "max_token_budget": 8
            }],
            "capacity": {
                "max_selected": 1,
                "max_total_tokens": 8,
                "max_token_budget": 8
            },
            "context": {"capabilities": {"token_budget": true}}
        }),
        vec![],
    ))?;
    assert_eq!(schedule["decision"]["selected"][0]["candidate_index"], 0);
    assert_eq!(schedule["decision"]["selected"][0]["token_budget"], 8);

    let feedback = runtime.invoke(engine_event(
        "feedback",
        feedback_body(
            "engine-feedback",
            vec![("progress", "L", json!({"committed_tokens": 4}))],
            context(false),
        ),
        vec![],
    ))?;
    assert_eq!(feedback["status"], "success");
    assert_eq!(feedback["decision"], json!({}));
    assert_eq!(backend.read_request("L")?["scratch"]["feedback_service"], 4);

    let rust_outcome = runtime.invoke(engine_event(
        "evict",
        json!({
            "bytes_needed": 1,
            "resident": [],
            "context": {}
        }),
        vec![],
    ))?;
    let json_outcome: Document = serde_json::from_str(
        &runtime.invoke_json(
            &engine_event(
                "evict",
                json!({
                    "bytes_needed": 1,
                    "resident": [],
                    "context": {}
                }),
                vec![],
            )
            .to_string(),
        )?,
    )?;
    assert_eq!(rust_outcome, json_outcome);
    assert_eq!(rust_outcome["status"], "unavailable");

    assert!(matches!(
        runtime.invoke(json!({
            "api_version": "wrong",
            "hook": "route",
            "context": {},
            "request_events": []
        })),
        Err(PlexError::InvalidEvent(_))
    ));
    assert!(matches!(
        runtime.invoke(json!({
            "api_version": ENGINE_API_VERSION,
            "hook": "route",
            "context": {},
            "request_events": [],
            "extra": true
        })),
        Err(PlexError::InvalidEvent(_))
    ));
    assert!(matches!(
        runtime.invoke(engine_event(
            "route",
            json!({
                "request_id": "missing",
                "candidates": candidates(),
                "context": {}
            }),
            vec![],
        )),
        Err(PlexError::Backend(_))
    ));
    Ok(())
}

fn check_request_events_and_cleanup(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let runtime = runtime_with_policy(
        engine,
        directory,
        "plex_least_loaded",
        "request-events",
        &[Operation::Route],
        backend.clone(),
        Arc::new(RejectingQueryHandler),
        BTreeSet::new(),
    )?;
    runtime.invoke(engine_event(
        "route",
        json!({
            "request_id": "L",
            "candidates": candidates(),
            "context": {}
        }),
        vec![json!({
            "op": "create",
            "request_id": "L",
            "facts": {"generation_id": 0},
            "fields": {
                "body": {"prompt": "first"},
                "metadata": {"keep": 1}
            }
        })],
    ))?;
    runtime.invoke(engine_event(
        "route",
        json!({
            "request_id": "L",
            "candidates": candidates(),
            "context": {}
        }),
        vec![
            json!({
                "op": "continue",
                "request_id": "L",
                "facts": {"generation_id": 1},
                "fields": {
                    "body": {"prompt": "second"},
                    "metadata": {"step": 2}
                }
            }),
            json!({
                "op": "merge-facts",
                "request_id": "L",
                "facts": {"previous_target": "node-a"}
            }),
        ],
    ))?;
    let request = backend.read_request("L")?;
    assert_eq!(request["facts"]["generation_id"], 1);
    assert_eq!(request["facts"]["previous_target"], "node-a");
    assert_eq!(request["fields"]["metadata"], json!({"keep": 1, "step": 2}));

    let duplicate_create = runtime.invoke(engine_event(
        "route",
        json!({
            "request_id": "X",
            "candidates": candidates(),
            "context": {}
        }),
        vec![
            json!({
                "op": "create",
                "request_id": "X",
                "facts": {"generation_id": 0},
                "fields": {"body": {}, "metadata": {}}
            }),
            json!({
                "op": "create",
                "request_id": "X",
                "facts": {"generation_id": 0},
                "fields": {"body": {}, "metadata": {}}
            }),
        ],
    ));
    assert!(matches!(duplicate_create, Err(PlexError::InvalidEvent(_))));
    assert!(matches!(
        backend.read_request("X"),
        Err(StateBackendError::NotFound(_))
    ));

    let unavailable = runtime.invoke(engine_event(
        "feedback",
        feedback_body(
            "cleanup-unavailable",
            vec![("completed", "L", json!({}))],
            context(false),
        ),
        vec![json!({"op": "finish", "request_id": "L"})],
    ))?;
    assert_eq!(unavailable["status"], "unavailable");
    assert!(matches!(
        backend.read_request("L"),
        Err(StateBackendError::NotFound(_))
    ));

    let fallback_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let fallback = runtime_with_policy(
        engine,
        directory,
        "plex_fallback",
        "cleanup-fallback",
        &[Operation::Feedback],
        fallback_backend.clone(),
        Arc::new(RejectingQueryHandler),
        BTreeSet::new(),
    )?;
    fallback
        .backend()
        .create_request("F".into(), json!({}), json!({}))?;
    let outcome = fallback.invoke(engine_event(
        "feedback",
        feedback_body(
            "cleanup-fallback",
            vec![("completed", "F", json!({}))],
            context(false),
        ),
        vec![json!({"op": "finish", "request_id": "F"})],
    ))?;
    assert_eq!(outcome["status"], "fallback");
    assert!(matches!(
        fallback_backend.read_request("F"),
        Err(StateBackendError::NotFound(_))
    ));

    let success_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let success = runtime_with_policy(
        engine,
        directory,
        "plex_feedback_accounting",
        "cleanup-success",
        &[Operation::Feedback],
        success_backend.clone(),
        Arc::new(RejectingQueryHandler),
        BTreeSet::new(),
    )?;
    let outcome = success.invoke(engine_event(
        "feedback",
        feedback_body(
            "cleanup-success",
            vec![("completed", "S", json!({}))],
            context(false),
        ),
        vec![
            json!({
                "op": "create",
                "request_id": "S",
                "facts": {"generation_id": 0},
                "fields": {"body": {}, "metadata": {}}
            }),
            json!({"op": "finish", "request_id": "S"}),
        ],
    ))?;
    assert_eq!(outcome["status"], "success");
    assert!(matches!(
        success_backend.read_request("S"),
        Err(StateBackendError::NotFound(_))
    ));

    success_backend.create_request("T".into(), json!({}), json!({}))?;
    let terminal_retry = engine_event(
        "feedback",
        feedback_body(
            "cleanup-terminal-retry",
            vec![("completed", "T", json!({}))],
            context(false),
        ),
        vec![json!({"op": "finish", "request_id": "T"})],
    );
    let expected = json!({
        "status": "success",
        "decision": {},
        "request_fields": {},
        "actions": []
    });
    assert_eq!(success.invoke(terminal_retry.clone())?, expected);
    assert_eq!(success.invoke(terminal_retry)?, expected);
    assert!(matches!(
        success_backend.read_request("T"),
        Err(StateBackendError::NotFound(_))
    ));
    Ok(())
}

fn check_working_sets_and_decisions(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (schedule, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_attained_service",
        "working-set",
        &[Operation::Schedule],
        backend.clone(),
        1,
    )?;
    for (id, service) in [("L", 8), ("M", 1), ("unreferenced", 0)] {
        schedule.create_request(id, json!({}), json!({}))?;
        schedule.merge_request_facts(id, json!({"attained_service": service}))?;
    }
    let ctx = schedule_context(
        vec![
            ("L", json!({"waiting_ms": 1})),
            ("L", json!({"waiting_ms": 2})),
            ("M", json!({"waiting_ms": 3})),
        ],
        false,
    );
    let response = expect_success(schedule.invoke_and_apply(Operation::Schedule, ctx.clone()))?;
    assert_eq!(backend.read_shared()?["working_set_size"], 2);
    assert_eq!(backend.read_request("L")?["scratch"]["schedule_calls"], 2);
    assert_eq!(backend.read_request("M")?["scratch"]["schedule_calls"], 1);
    assert_eq!(backend.read_request("unreferenced")?["scratch"], json!({}));
    assert_eq!(
        select_schedule(&ctx, &response.result)?[0].candidate_index,
        2
    );

    let engine_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let engine_schedule = runtime_with_policy(
        engine,
        directory,
        "plex_attained_service",
        "scratch-only",
        &[Operation::Schedule],
        engine_backend.clone(),
        Arc::new(RejectingQueryHandler),
        BTreeSet::new(),
    )?;
    let outcome = engine_schedule.invoke(engine_event(
        "schedule",
        json!({
            "runnable": [{
                "request_id": "E",
                "facts": {},
                "max_token_budget": 8
            }],
            "capacity": {
                "max_selected": 1,
                "max_total_tokens": 8,
                "max_token_budget": 8
            },
            "context": {}
        }),
        vec![json!({
            "op": "create",
            "request_id": "E",
            "facts": {"generation_id": 0, "attained_service": 0},
            "fields": {"body": {}, "metadata": {}}
        })],
    ))?;
    assert_eq!(outcome["request_fields"], json!({}));
    assert_eq!(
        engine_backend.read_request("E")?["scratch"]["schedule_calls"],
        1
    );

    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (evict, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_retention_score",
        "retention",
        &[Operation::Evict],
        backend.clone(),
        1,
    )?;
    evict.create_request("E", json!({}), json!({}))?;
    let ctx = eviction_context(vec![
        ("attributed", Some("E"), 4, json!({"reload_cost": 2.0})),
        ("shared", None, 3, json!({"reload_cost": 1.0})),
    ]);
    let response = expect_success(evict.invoke_and_apply(Operation::Evict, ctx.clone()))?;
    assert_eq!(backend.read_shared()?["working_set_size"], 1);
    let selected = select_evictions(&ctx, &response.result)?;
    assert_eq!(selected[0].candidate_index, 1);
    assert_eq!(selected[1].candidate_index, 0);

    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (route, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_least_loaded",
        "stable-route",
        &[Operation::Route],
        backend.clone(),
        1,
    )?;
    route.create_request("S", json!({}), json!({}))?;
    let response = expect_success(route.invoke_and_apply(
        Operation::Route,
        route_context(
            "S",
            "generation-arrival",
            vec![
                json!({"id": "a", "facts": {"queue_depth": 10}}),
                json!({"id": "b", "facts": {"queue_depth": 10}}),
            ],
            context(false),
        ),
    ))?;
    assert_eq!(rank_route(&response.result, 2)?, vec![0, 1]);

    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (budget, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_bad_budget",
        "bad-budget",
        &[Operation::Schedule],
        backend,
        1,
    )?;
    budget.create_request("B", json!({}), json!({}))?;
    expect_failure(
        budget.invoke_and_apply(
            Operation::Schedule,
            schedule_context(vec![("B", json!({}))], true),
        ),
        InvocationFailureKind::InvalidOutput,
    )?;
    Ok(())
}

fn check_queries_actions_and_conflicts(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let query = DictionaryQueryHandler::default();
    query.insert("pie.cluster.capacity@1", json!({"route_bias": 100.0}));
    let backend = Arc::new(InMemoryPolicyStateBackend::new(json!({"stable": true}))?);
    let (query_policy, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_query_assisted",
        "query-assisted",
        &[Operation::Route],
        backend.clone(),
        Arc::new(query),
        BTreeSet::new(),
        1,
    )?;
    query_policy.create_request("Q", json!({}), json!({}))?;
    let before = backend.read_request("Q")?;
    let response = expect_success(query_policy.invoke_and_apply(
        Operation::Route,
        route_context(
            "Q",
            "generation-arrival",
            vec![
                json!({"id": "a", "facts": {"queue_depth": 10}}),
                json!({"id": "b", "facts": {"queue_depth": 2}}),
            ],
            context_with_helpers(false, &["pie.cluster.capacity@1"], &[]),
        ),
    ))?;
    assert_eq!(rank_route(&response.result, 2)?, vec![1, 0]);
    assert_eq!(backend.read_request("Q")?, before);

    let (unsupported, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_query_assisted",
        "unsupported-query",
        &[Operation::Route],
        backend.clone(),
        Arc::new(RejectingQueryHandler),
        BTreeSet::new(),
        1,
    )?;
    expect_failure(
        unsupported.invoke_and_apply(
            Operation::Route,
            route_context(
                "Q",
                "generation-arrival",
                candidates(),
                context_with_helpers(false, &["pie.cluster.capacity@1"], &[]),
            ),
        ),
        InvocationFailureKind::Query,
    )?;

    let action_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (actions, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_stage_action",
        "actions",
        &[Operation::Route],
        action_backend.clone(),
        Arc::new(RejectingQueryHandler),
        action_set(["pie.kv.prefetch@1", "pie.retention.set@1"]),
        1,
    )?;
    actions.create_request("A", json!({}), json!({}))?;
    let response = expect_success(actions.invoke_and_apply(
        Operation::Route,
        route_context("A", "generation-arrival", candidates(), context(false)),
    ))?;
    assert_eq!(response.actions.len(), 2);
    assert_eq!(response.actions[0].id, 0);
    assert_eq!(response.actions[1].id, 1);
    assert_eq!(response.actions[0].method, "pie.kv.prefetch@1");
    assert_eq!(response.actions[1].method, "pie.retention.set@1");

    let snapshot_inner = Arc::new(InMemoryPolicyStateBackend::default());
    let snapshot_loads = Arc::new(AtomicUsize::new(0));
    let snapshot_runtime = runtime_with_policy(
        engine,
        directory,
        "plex_stage_action",
        "snapshot-reporting",
        &[Operation::Route],
        Arc::new(MutatingSecondLoadBackend {
            inner: snapshot_inner,
            load_calls: snapshot_loads.clone(),
        }),
        Arc::new(RejectingQueryHandler),
        action_set(["pie.kv.prefetch@1", "pie.retention.set@1"]),
    )?;
    let outcome = snapshot_runtime.invoke(engine_event(
        "route",
        json!({
            "request_id": "snapshot",
            "candidates": candidates(),
            "context": {}
        }),
        vec![json!({
            "op": "create",
            "request_id": "snapshot",
            "facts": {"generation_id": 0},
            "fields": {"body": {}, "metadata": {}}
        })],
    ))?;
    assert_eq!(outcome["status"], "success");
    assert_eq!(outcome["request_fields"], json!({}));
    assert_eq!(snapshot_loads.load(Ordering::Acquire), 1);

    let unsupported_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (unsupported_action, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_stage_action",
        "unsupported-action",
        &[Operation::Route],
        unsupported_backend.clone(),
        1,
    )?;
    unsupported_action.create_request("U", json!({}), json!({}))?;
    expect_failure(
        unsupported_action.invoke_and_apply(
            Operation::Route,
            route_context("U", "generation-arrival", candidates(), context(false)),
        ),
        InvocationFailureKind::ActionValidation,
    )?;
    assert_eq!(unsupported_backend.read_request("U")?["scratch"], json!({}));

    let malformed_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (malformed, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_action_bad_result",
        "action-bad-result",
        &[Operation::Route],
        malformed_backend.clone(),
        Arc::new(RejectingQueryHandler),
        action_set(["pie.kv.prefetch@1"]),
        1,
    )?;
    malformed.create_request("M", json!({}), json!({}))?;
    expect_failure(
        malformed.invoke_and_apply(
            Operation::Route,
            route_context("M", "generation-arrival", candidates(), context(false)),
        ),
        InvocationFailureKind::InvalidOutput,
    )?;
    assert_eq!(malformed_backend.read_request("M")?["scratch"], json!({}));

    let fallback_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let fallback_runtime = runtime_with_policy(
        engine,
        directory,
        "plex_action_bad_result",
        "engine-action-fallback",
        &[Operation::Route],
        fallback_backend.clone(),
        Arc::new(RejectingQueryHandler),
        action_set(["pie.kv.prefetch@1"]),
    )?;
    let outcome = fallback_runtime.invoke(engine_event(
        "route",
        json!({
            "request_id": "F",
            "candidates": candidates(),
            "context": {}
        }),
        vec![json!({
            "op": "create",
            "request_id": "F",
            "facts": {"generation_id": 0},
            "fields": {"body": {}, "metadata": {}}
        })],
    ))?;
    assert_eq!(outcome["status"], "fallback");
    assert!(outcome.get("actions").is_none());
    assert_eq!(fallback_backend.read_request("F")?["scratch"], json!({}));

    let conflict_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let conflicting_query = MutatingQueryHandler {
        backend: conflict_backend.clone(),
        fired: Arc::new(AtomicBool::new(false)),
    };
    let (conflict, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_coordinated",
        "conflict",
        &[Operation::Route],
        conflict_backend.clone(),
        Arc::new(conflicting_query),
        action_set(["pie.kv.prefetch@1"]),
        1,
    )?;
    conflict.create_request("C", json!({"prompt": "stable"}), json!({}))?;
    expect_failure(
        conflict.invoke_and_apply(
            Operation::Route,
            route_context(
                "C",
                "generation-arrival",
                candidates(),
                context_with_helpers(false, &["pie.cluster.capacity@1"], &["pie.kv.prefetch@1"]),
            ),
        ),
        InvocationFailureKind::StateConflict,
    )?;
    assert_eq!(conflict_backend.read_shared()?, json!({"external": true}));
    assert_eq!(
        conflict_backend.read_request("C")?["fields"]["body"]["prompt"],
        "stable"
    );

    let engine_conflict_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let engine_conflict = runtime_with_policy(
        engine,
        directory,
        "plex_coordinated",
        "engine-conflict",
        &[Operation::Route],
        engine_conflict_backend.clone(),
        Arc::new(MutatingQueryHandler {
            backend: engine_conflict_backend.clone(),
            fired: Arc::new(AtomicBool::new(false)),
        }),
        action_set(["pie.kv.prefetch@1"]),
    )?;
    let outcome = engine_conflict.invoke(engine_event(
        "route",
        json!({
            "request_id": "EC",
            "candidates": candidates(),
            "context": {
                "model": "example-model",
                "capabilities": {"queries": ["pie.cluster.capacity@1"]}
            }
        }),
        vec![json!({
            "op": "create",
            "request_id": "EC",
            "facts": {"generation_id": 0},
            "fields": {"body": {"prompt": "stable"}, "metadata": {}}
        })],
    ))?;
    assert_eq!(outcome["status"], "fallback");
    assert_eq!(outcome["failure"]["kind"], "state-conflict");
    assert!(outcome.get("request_fields").is_none());
    assert!(outcome.get("actions").is_none());

    let helper_query = CapturingQueryHandler::default();
    let helper_calls = helper_query.calls.clone();
    let helper_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (helpers, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_helper_methods",
        "helper-methods",
        &[Operation::Route],
        helper_backend.clone(),
        Arc::new(helper_query),
        action_set([
            "pie.kv.prefetch@1",
            "pie.schedule.preempt@1",
            "pie.route.replicate@1",
            "pie.retention.set@1",
            "pie.timer.arm@1",
        ]),
        1,
    )?;
    helpers.create_request("H", json!({}), json!({}))?;
    let response = expect_success(helpers.invoke_and_apply(
        Operation::Route,
        route_context("H", "generation-arrival", candidates(), context(false)),
    ))?;
    assert_eq!(
        response
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    let calls = helper_calls.lock().unwrap();
    assert_eq!(
        calls[0],
        (
            "pie.kv.lookup@1".into(),
            json!({"request_id": "H", "target": "node-a"})
        )
    );
    assert_eq!(
        calls[1],
        (
            "pie.cluster.capacity@1".into(),
            json!({"model": "example-model"})
        )
    );
    assert_eq!(calls[2], ("pie.model.config@1".into(), json!({})));
    assert_eq!(calls[3], ("pie.clock.now@1".into(), json!({})));
    drop(calls);

    let raw_query = CapturingQueryHandler::default();
    let raw_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (raw, _, _) = lifecycle_with_policy_and_host(
        engine,
        directory,
        "plex_raw_helpers",
        "raw-helpers",
        &[Operation::Route],
        raw_backend.clone(),
        Arc::new(raw_query),
        action_set(["engine.custom-action@1"]),
        1,
    )?;
    raw.create_request("R", json!({}), json!({}))?;
    let raw_response = expect_success(raw.invoke_and_apply(
        Operation::Route,
        route_context("R", "generation-arrival", candidates(), context(false)),
    ))?;
    assert_eq!(raw_response.actions[0].method, "engine.custom-action@1");
    assert_eq!(raw_backend.read_shared()?["raw"]["action_id"], 0);

    let limited_engine = PolicyEngine::new(PolicyEngineConfig {
        max_host_calls: 1,
        ..PolicyEngineConfig::default()
    })?;
    let limited_backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (limited, _, _) = lifecycle_with_policy_and_host(
        &limited_engine,
        directory,
        "plex_stage_action",
        "limited-actions",
        &[Operation::Route],
        limited_backend.clone(),
        Arc::new(RejectingQueryHandler),
        action_set(["pie.kv.prefetch@1", "pie.retention.set@1"]),
        1,
    )?;
    limited.create_request("L", json!({}), json!({}))?;
    expect_failure(
        limited.invoke_and_apply(
            Operation::Route,
            route_context("L", "generation-arrival", candidates(), context(false)),
        ),
        InvocationFailureKind::ActionValidation,
    )?;
    Ok(())
}

fn check_failure_rollback(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for (artifact, expected) in [
        (
            "plex_invalid_shared_update",
            InvocationFailureKind::InvalidOutput,
        ),
        (
            "plex_mutate_request_facts",
            InvocationFailureKind::InvalidOutput,
        ),
        (
            "plex_mutate_unknown_request",
            InvocationFailureKind::InvalidOutput,
        ),
        (
            "plex_malformed_state_update",
            InvocationFailureKind::InvalidOutput,
        ),
        ("plex_malformed", InvocationFailureKind::InvalidOutput),
        ("plex_nonfinite", InvocationFailureKind::InvalidOutput),
        ("plex_mutate_fail", InvocationFailureKind::PolicyFallback),
        ("plex_fallback", InvocationFailureKind::PolicyFallback),
        ("plex_trap", InvocationFailureKind::Trap),
    ] {
        let backend = Arc::new(InMemoryPolicyStateBackend::new(json!({"stable": true}))?);
        let (lifecycle, _, _) = lifecycle_with_policy(
            engine,
            directory,
            artifact,
            artifact,
            &[Operation::Route],
            backend.clone(),
            1,
        )?;
        lifecycle.create_request("F", json!({"prompt": "stable"}), json!({}))?;
        let before_shared = backend.read_shared()?;
        let before_request = backend.read_request("F")?;
        expect_failure(
            lifecycle.invoke_and_apply(
                Operation::Route,
                route_context("F", "generation-arrival", candidates(), context(false)),
            ),
            expected,
        )?;
        assert_eq!(backend.read_shared()?, before_shared);
        assert_eq!(backend.read_request("F")?, before_request);
    }

    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (spin, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_spin",
        "spin",
        &[Operation::Route],
        backend.clone(),
        1,
    )?;
    spin.create_request("S", json!({}), json!({}))?;
    expect_failure_one_of(
        spin.invoke_and_apply(
            Operation::Route,
            route_context("S", "generation-arrival", candidates(), context(false)),
        ),
        &[
            InvocationFailureKind::FuelExhausted,
            InvocationFailureKind::DeadlineExceeded,
        ],
    )?;
    assert_eq!(backend.read_request("S")?["scratch"], json!({}));
    Ok(())
}

fn check_feedback_dedup(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let engine = PolicyEngine::new(PolicyEngineConfig {
        max_feedback_deliveries: 2,
        ..PolicyEngineConfig::default()
    })?;
    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (lifecycle, _, _) = lifecycle_with_policy_and_host(
        &engine,
        directory,
        "plex_coordinated",
        "feedback-dedup",
        &[Operation::Feedback],
        backend.clone(),
        Arc::new(RejectingQueryHandler),
        action_set(["pie.timer.arm@1"]),
        1,
    )?;
    lifecycle.create_request("L", json!({}), json!({}))?;
    let feedback = feedback_context(
        "dedup",
        vec![("action-failed", "L", json!({"method": "prefetch"}))],
        context_with_helpers(false, &[], &["pie.timer.arm@1"]),
    );
    let first = expect_success(lifecycle.invoke_and_apply(Operation::Feedback, feedback.clone()))?;
    assert_eq!(first.actions.len(), 1);
    assert_eq!(first.actions[0].id, 0);
    let duplicate = expect_success(lifecycle.invoke_and_apply(Operation::Feedback, feedback))?;
    assert!(duplicate.duplicate_feedback);
    assert!(duplicate.actions.is_empty());
    assert_eq!(backend.read_request("L")?["scratch"]["actions_failed"], 1);

    let terminal = feedback_context(
        "terminal",
        vec![("completed", "L", json!({}))],
        context(false),
    );
    expect_success(lifecycle.feedback_and_remove(terminal, &["L".into()]))?;
    assert!(matches!(
        backend.read_request("L"),
        Err(StateBackendError::NotFound(_))
    ));
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
            ReplayCommand::ReplaceShared {
                shared: json!({"mode": "replay"}),
            },
            ReplayCommand::CreateRequest {
                logical_request_id: "L".into(),
                body: json!({"prompt": "hello"}),
                metadata: json!({"user": "alice"}),
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
                context: schedule_context(vec![("L", json!({}))], true),
            },
            ReplayCommand::Invoke {
                operation: Operation::Feedback,
                context: feedback_context(
                    "d",
                    vec![("progress", "L", json!({"committed_tokens": 4}))],
                    context(false),
                ),
            },
            ReplayCommand::ReadShared,
            ReplayCommand::ReadRequest {
                logical_request_id: "L".into(),
            },
        ],
    };
    let packages = BTreeMap::from([("policy".into(), package)]);
    let first = ReplayRunner::new(
        AttachmentRegistry::new(engine.clone()),
        Arc::new(InMemoryPolicyStateBackend::default()),
        packages.clone(),
        2,
    )?
    .run(&trace)?;
    ReplayRunner::new(
        AttachmentRegistry::new(engine),
        Arc::new(InMemoryPolicyStateBackend::default()),
        packages,
        2,
    )?
    .verify(&trace, &first)?;
    Ok(())
}

fn check_adapter_conformance(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let cases: Document =
        serde_json::from_slice(&std::fs::read("tests/policies/engine-cases.json")?)?;
    let events = cases["events"]
        .as_array()
        .ok_or("engine cases events must be an array")?;
    let expected = cases["expected_decisions"]
        .as_array()
        .ok_or("engine cases decisions must be an array")?;

    let mut reports = Vec::new();
    for adapter in ["pie", "vllm-mock", "sglang-mock"] {
        let runtime = runtime_with_policy(
            engine,
            directory,
            "plex_coordinated",
            &format!("{adapter}-conformance"),
            &[Operation::Route, Operation::Admit, Operation::Schedule],
            Arc::new(InMemoryPolicyStateBackend::default()),
            Arc::new(RejectingQueryHandler),
            BTreeSet::new(),
        )?;
        reports.push(run_mock_adapter(adapter, &runtime, events)?);
    }
    assert_eq!(reports[0], reports[1]);
    assert_eq!(reports[1], reports[2]);
    assert_eq!(
        reports[0]
            .iter()
            .map(|outcome| outcome["decision"].clone())
            .collect::<Vec<_>>(),
        expected.clone()
    );
    Ok(())
}

fn check_paper_policies(
    engine: &PolicyEngine,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let cases: Document =
        serde_json::from_slice(&std::fs::read("tests/policies/paper-cases.json")?)?;
    assert_eq!(cases.as_array().map_or(0, Vec::len), 5);

    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (agentix, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_agentix",
        "agentix",
        &[Operation::Schedule],
        backend.clone(),
        1,
    )?;
    for (id, service) in [("A", 100), ("B", 1)] {
        agentix.create_request(id, json!({}), json!({}))?;
        agentix.merge_request_facts(id, json!({"attained_service": service}))?;
    }
    let response = expect_success(agentix.invoke_and_apply(
        Operation::Schedule,
        schedule_context(
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

    for (artifact, package, operation, request_id, ctx) in [
        (
            "plex_paper_preble",
            "preble",
            Operation::Route,
            "P",
            route_context(
                "P",
                "generation-arrival",
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
        ),
        (
            "plex_paper_kvflow",
            "kvflow",
            Operation::Evict,
            "K",
            eviction_context(vec![(
                "u",
                Some("K"),
                8,
                json!({"steps_to_execution": 5, "fixed_prefix": false}),
            )]),
        ),
        (
            "plex_paper_helium",
            "helium",
            Operation::Schedule,
            "H",
            schedule_context(
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
        ),
    ] {
        let backend = Arc::new(InMemoryPolicyStateBackend::default());
        let (policy, _, _) = lifecycle_with_policy(
            engine,
            directory,
            artifact,
            package,
            &[operation],
            backend,
            1,
        )?;
        policy.create_request(request_id, json!({}), json!({}))?;
        expect_success(policy.invoke_and_apply(operation, ctx))?;
    }

    let backend = Arc::new(InMemoryPolicyStateBackend::default());
    let (continuum, _, _) = lifecycle_with_policy(
        engine,
        directory,
        "plex_paper_continuum",
        "continuum",
        &[Operation::Schedule, Operation::Evict, Operation::Feedback],
        backend,
        1,
    )?;
    continuum.create_request("C", json!({}), json!({}))?;
    expect_success(continuum.invoke_and_apply(
        Operation::Feedback,
        feedback_context(
            "ttl",
            vec![("tool-boundary", "C", json!({"ttl_ms": 100}))],
            context(false),
        ),
    ))?;
    expect_success(continuum.invoke_and_apply(
        Operation::Schedule,
        schedule_context(
            vec![("C", json!({"preempted": false, "program_arrival": 1}))],
            false,
        ),
    ))?;
    Ok(())
}

#[derive(Clone)]
struct MutatingQueryHandler {
    backend: Arc<InMemoryPolicyStateBackend>,
    fired: Arc<AtomicBool>,
}

struct MutatingSecondLoadBackend {
    inner: Arc<InMemoryPolicyStateBackend>,
    load_calls: Arc<AtomicUsize>,
}

#[derive(Clone, Default)]
struct CapturingQueryHandler {
    calls: Arc<Mutex<Vec<(String, Document)>>>,
}

impl QueryHandler for CapturingQueryHandler {
    fn query(&self, method: &str, args: &Document) -> Result<Document, QueryError> {
        self.calls
            .lock()
            .unwrap()
            .push((method.to_owned(), args.clone()));
        match method {
            "pie.kv.lookup@1" => Ok(json!({"cached": true})),
            "pie.cluster.capacity@1" => Ok(json!({"free_slots": 4})),
            "pie.model.config@1" => Ok(json!({"name": "example-model"})),
            "pie.clock.now@1" => Ok(json!(1234)),
            "engine.custom-query@1" => Ok(json!({"value": 7})),
            _ => Err(QueryError::Unsupported(method.into())),
        }
    }
}

impl QueryHandler for MutatingQueryHandler {
    fn query(&self, method: &str, _args: &Document) -> Result<Document, QueryError> {
        if method != "pie.cluster.capacity@1" {
            return Err(QueryError::Unsupported(method.into()));
        }
        if !self.fired.swap(true, Ordering::AcqRel) {
            self.backend
                .replace_shared(json!({"external": true}))
                .map_err(|error| QueryError::Handler(error.to_string()))?;
        }
        Ok(json!({"route_bias": 0.0}))
    }
}

impl PolicyStateBackend for MutatingSecondLoadBackend {
    fn load(&self, request_ids: &BTreeSet<String>) -> Result<StateSnapshot, StateBackendError> {
        if self.load_calls.fetch_add(1, Ordering::AcqRel) == 1 {
            for request_id in request_ids {
                let mut fields = self.inner.read_request(request_id)?["fields"].clone();
                fields["host_only"] = json!(true);
                self.inner.replace_request_fields(request_id, fields)?;
            }
        }
        self.inner.load(request_ids)
    }

    fn commit(
        &self,
        snapshot: &StateSnapshot,
        updates: &StateUpdates,
        feedback: Option<&FeedbackCommit>,
        terminal_requests: &[String],
    ) -> Result<(), StateBackendError> {
        self.inner
            .commit(snapshot, updates, feedback, terminal_requests)
    }

    fn create_request(
        &self,
        logical_request_id: String,
        body: Document,
        metadata: Document,
    ) -> Result<Document, StateBackendError> {
        self.inner
            .create_request(logical_request_id, body, metadata)
    }

    fn continue_request(
        &self,
        logical_request_id: &str,
        body: Document,
        continuation_metadata: Document,
    ) -> Result<Document, StateBackendError> {
        self.inner
            .continue_request(logical_request_id, body, continuation_metadata)
    }

    fn read_shared(&self) -> Result<Document, StateBackendError> {
        self.inner.read_shared()
    }

    fn replace_shared(&self, shared: Document) -> Result<(), StateBackendError> {
        self.inner.replace_shared(shared)
    }

    fn read_request(&self, logical_request_id: &str) -> Result<Document, StateBackendError> {
        self.inner.read_request(logical_request_id)
    }

    fn remove_request(&self, logical_request_id: &str) -> Result<Document, StateBackendError> {
        self.inner.remove_request(logical_request_id)
    }

    fn request_count(&self) -> Result<usize, StateBackendError> {
        self.inner.request_count()
    }

    fn merge_request_facts(
        &self,
        logical_request_id: &str,
        facts: Document,
    ) -> Result<(), StateBackendError> {
        self.inner.merge_request_facts(logical_request_id, facts)
    }

    fn replace_request_fields(
        &self,
        logical_request_id: &str,
        fields: Document,
    ) -> Result<(), StateBackendError> {
        self.inner
            .replace_request_fields(logical_request_id, fields)
    }

    fn record_enacted_placement(
        &self,
        logical_request_id: &str,
        target_id: String,
    ) -> Result<(), StateBackendError> {
        self.inner
            .record_enacted_placement(logical_request_id, target_id)
    }

    fn feedback_result(&self, delivery_id: &str) -> Result<Option<Document>, StateBackendError> {
        self.inner.feedback_result(delivery_id)
    }
}

fn run_mock_adapter(
    _name: &str,
    runtime: &PlexRuntime,
    events: &[Document],
) -> Result<Vec<Document>, PlexError> {
    events
        .iter()
        .cloned()
        .map(|event| runtime.invoke(event))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn runtime_with_policy(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    package_name: &str,
    operations: &[Operation],
    backend: Arc<dyn PolicyStateBackend>,
    query_handler: Arc<dyn QueryHandler>,
    supported_actions: BTreeSet<String>,
) -> Result<PlexRuntime, Box<dyn std::error::Error>> {
    let package = package_bytes(
        directory,
        artifact,
        manifest(package_name, operations.iter().copied()),
    )?;
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach(&package)?;
    Ok(PlexRuntime::with_parts(
        registry,
        backend,
        query_handler,
        supported_actions,
        2,
    )?)
}

fn lifecycle_with_policy(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    package_name: &str,
    operations: &[Operation],
    backend: Arc<InMemoryPolicyStateBackend>,
    max_defer_retries: u32,
) -> Result<(LifecycleHost, AttachmentRegistry, Vec<u8>), Box<dyn std::error::Error>> {
    lifecycle_with_policy_and_host(
        engine,
        directory,
        artifact,
        package_name,
        operations,
        backend,
        Arc::new(RejectingQueryHandler),
        BTreeSet::new(),
        max_defer_retries,
    )
}

#[allow(clippy::too_many_arguments)]
fn lifecycle_with_policy_and_host(
    engine: &PolicyEngine,
    directory: &Path,
    artifact: &str,
    package_name: &str,
    operations: &[Operation],
    backend: Arc<InMemoryPolicyStateBackend>,
    query_handler: Arc<dyn QueryHandler>,
    supported_actions: BTreeSet<String>,
    max_defer_retries: u32,
) -> Result<(LifecycleHost, AttachmentRegistry, Vec<u8>), Box<dyn std::error::Error>> {
    let package = package_bytes(
        directory,
        artifact,
        manifest(package_name, operations.iter().copied()),
    )?;
    let registry = AttachmentRegistry::new(engine.clone());
    registry.attach(&package)?;
    let lifecycle = LifecycleHost::with_host(
        registry.clone(),
        backend,
        query_handler,
        supported_actions,
        max_defer_retries,
    );
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
        contract: ContractVersion::V0_5,
        package_name: name.replace('_', "-"),
        package_version: "0.5.0".into(),
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

fn action_set<const N: usize>(actions: [&str; N]) -> BTreeSet<String> {
    actions.into_iter().map(str::to_owned).collect()
}

fn engine_event(hook: &str, context: Document, request_events: Vec<Document>) -> Document {
    json!({
        "api_version": ENGINE_API_VERSION,
        "hook": hook,
        "context": context,
        "request_events": request_events,
    })
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
    context_with_helpers(token_budget, &[], &[])
}

fn context_with_helpers(token_budget: bool, queries: &[&str], actions: &[&str]) -> Document {
    json!({
        "model": "example-model",
        "capabilities": {
            "token_budget": token_budget,
            "queries": queries,
            "actions": actions
        }
    })
}

fn route_context(
    request_id: &str,
    cause: &str,
    candidates: Vec<Document>,
    context: Document,
) -> Document {
    json!({
        "cause": cause,
        "request_id": request_id,
        "candidates": candidates,
        "context": context
    })
}

fn schedule_context(runnable: Vec<(&str, Document)>, token_budget: bool) -> Document {
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

fn eviction_context(resident: Vec<(&str, Option<&str>, u64, Document)>) -> Document {
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

fn feedback_body(
    delivery_id: &str,
    records: Vec<(&str, &str, Document)>,
    context: Document,
) -> Document {
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
        "context": context
    })
}

fn feedback_context(
    delivery_id: &str,
    records: Vec<(&str, &str, Document)>,
    context: Document,
) -> Document {
    feedback_body(delivery_id, records, context)
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
