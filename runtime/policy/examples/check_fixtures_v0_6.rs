use std::path::{Path, PathBuf};
use std::sync::Arc;

use pie_plex::v0_6::RequestId;
use pie_policy::{
    DictionaryQueryHandler, HostSupportV0_6, PackageLimits, PlexRuntimeV0_6, PolicyPackageV0_6,
    QueryHandler, ReplayRunnerV0_6,
};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let packages = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: check_fixtures_v0_6 <package-directory>")?,
    );

    check_route_successes(&packages)?;
    check_admit_schedule_cache_feedback(&packages)?;
    check_negative_rollback(&packages)?;
    check_queries_and_actions(&packages)?;
    check_existing_papers(&packages)?;
    check_wave_a(&packages)?;
    check_wave_c(&packages)?;
    check_wave_d(&packages)?;
    check_replication_artifacts(&packages)?;
    check_deterministic_replay(&packages)?;

    println!("PLEX v0.6 policy fixtures passed");
    Ok(())
}

fn check_deterministic_replay(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let runner = ReplayRunnerV0_6::new(
        std::fs::read(packages.join("plex_coordinated.plexpkg"))?,
        HostSupportV0_6::default(),
    );
    let report = runner.verify_deterministic(&[
        admit_event("replay", "replay-group", 0),
        route_event("replay", "replay-group", "replay-route", json!({})),
    ])?;
    assert_eq!(report.outcomes.len(), 2);
    assert_eq!(report.state_metrics.commits, 2);
    Ok(())
}

fn check_wave_d(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for (package, facts) in [
        ("plex_paper_dualmap", json!({"hotspot": true})),
        ("plex_paper_llumnix", json!({"live_reschedule": true})),
        (
            "plex_paper_smetric",
            json!({"generation_id": 1, "tail_outlier": true}),
        ),
        (
            "plex_paper_goodserve",
            json!({"risk_ppm": 900, "migration_threshold_ppm": 800}),
        ),
        ("plex_paper_saga", json!({"steal": true})),
    ] {
        let runtime = runtime(packages, package, &["request.rebalance@1"], None)?;
        let outcome = runtime.invoke(route_event("A", "G", package, facts))?;
        assert_success(&outcome);
        assert_eq!(outcome["actions"][0]["method"], "pie.request.rebalance@1");
    }

    let thunder = runtime(
        packages,
        "plex_paper_thunderagent",
        &["request.cancel@1", "request.rebalance@1"],
        None,
    )?;
    let thundered = thunder.invoke(schedule_event(
        "A",
        "G",
        json!({"tool_ready": true, "migrate_target": "node-a"}),
    ))?;
    assert_success(&thundered);
    assert_eq!(thundered["actions"][0]["method"], "pie.request.rebalance@1");

    let pythia = runtime(packages, "plex_paper_pythia", &["cache.prefetch@1"], None)?;
    let prefetched = pythia.invoke(cache_event(true, false))?;
    assert_success(&prefetched);
    assert_eq!(prefetched["actions"][0]["method"], "pie.cache.prefetch@1");

    let parrot = runtime(packages, "plex_paper_parrot", &[], None)?;
    assert_success(&parrot.invoke(schedule_event("A", "G", json!({"dependency_ready": true})))?);

    let conserve = runtime(packages, "plex_paper_conserve", &[], None)?;
    let placed = conserve.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta("conserve"),
            "cause": "admission",
            "requests": [{"request": request_ref("A", "G"), "facts": {}}],
            "targets": [
                {
                    "target_id": "X",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {}
                },
                {
                    "target_id": "Y",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {}
                }
            ],
            "feasible_edges": [
                {
                    "request_index": 0,
                    "target_index": 0,
                    "demand": [],
                    "facts": {"prefill_capacity": 1}
                },
                {
                    "request_index": 0,
                    "target_index": 1,
                    "demand": [],
                    "facts": {"prefill_capacity": 10}
                }
            ]
        },
        "lifecycle": lifecycle("A", "G", false)
    }))?;
    assert_success(&placed);
    assert_eq!(placed["plan"]["plan"]["assignments"][0]["target_index"], 1);

    let routebalance = runtime(packages, "plex_paper_routebalance", &[], None)?;
    let balanced = routebalance.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta("routebalance"),
            "cause": "admission",
            "requests": [
                {"request": request_ref("A", "GA"), "facts": {}},
                {"request": request_ref("B", "GB"), "facts": {}}
            ],
            "targets": [
                {
                    "target_id": "X",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {}
                },
                {
                    "target_id": "Y",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {}
                }
            ],
            "feasible_edges": [
                {"request_index": 0, "target_index": 0, "demand": [], "facts": {"utility": 10}},
                {"request_index": 0, "target_index": 1, "demand": [], "facts": {"utility": 9}},
                {"request_index": 1, "target_index": 0, "demand": [], "facts": {"utility": 8}},
                {"request_index": 1, "target_index": 1, "demand": [], "facts": {"utility": 0}}
            ]
        },
        "lifecycle": [
            {
                "event": "create-group",
                "group_id": "GA",
                "principal_id": "tenant",
                "limits": {"max_members": 2, "max_scratch_bytes": 4096},
                "facts": {}
            },
            {
                "event": "create-request",
                "request_id": "A",
                "principal_id": "tenant",
                "group_id": "GA",
                "fields": {},
                "facts": {}
            },
            {"event": "admit-request", "request_id": "A"},
            {
                "event": "create-group",
                "group_id": "GB",
                "principal_id": "tenant",
                "limits": {"max_members": 2, "max_scratch_bytes": 4096},
                "facts": {}
            },
            {
                "event": "create-request",
                "request_id": "B",
                "principal_id": "tenant",
                "group_id": "GB",
                "fields": {},
                "facts": {}
            },
            {"event": "admit-request", "request_id": "B"}
        ]
    }))?;
    assert_success(&balanced);
    assert_eq!(
        balanced["plan"]["plan"]["assignments"],
        json!([
            {"request_index": 0, "edge_index": 1, "target_index": 1},
            {"request_index": 1, "edge_index": 2, "target_index": 0}
        ])
    );
    Ok(())
}

fn check_wave_c(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let dlpm = runtime(packages, "plex_paper_dlpm", &[], None)?;
    assert_success(&dlpm.invoke(route_event(
        "A",
        "G",
        "dlpm-route",
        json!({"client_id": "client-a"}),
    ))?);

    let infercept = runtime(packages, "plex_paper_infercept", &["cache.swap@1"], None)?;
    let swapped = infercept.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta("infercept-cache"),
            "cause": "pressure",
            "resident": [{
                "object": {
                    "object_id": "paused-kv",
                    "size_bytes": 1,
                    "beneficiaries": [],
                    "beneficiary_count": 0,
                    "facts": {"expected_reuse_ms": 100, "swap": true}
                },
                "reclaimable": true
            }],
            "prospective": [],
            "capacity": {"max_bytes": 0, "fixed_bytes": 0, "facts": {}},
            "episode": null
        }
    }))?;
    assert_success(&swapped);
    assert_eq!(swapped["actions"][0]["method"], "pie.cache.swap@1");

    let peek = runtime(packages, "plex_paper_peek", &[], None)?;
    assert_success(&peek.invoke(schedule_event(
        "A",
        "G",
        json!({
            "waiting_ms": 100,
            "fairness_threshold_ms": 50,
            "demand_depth": 3
        }),
    ))?);
    assert_success(&peek.invoke(cache_event(false, false))?);

    for (package, facts) in [
        (
            "plex_paper_qlm",
            json!({"estimated_wait_ms": 10, "slo_ms": 100}),
        ),
        (
            "plex_paper_slos_serve",
            json!({"predicted_total_ms": 10, "slo_ms": 100}),
        ),
        (
            "plex_paper_chameleon",
            json!({"weighted_size": 1, "queue_quota": 4}),
        ),
    ] {
        let runtime = runtime(packages, package, &[], None)?;
        let mut event = admit_event("A", "G", 0);
        event["context"]["candidates"][0]["facts"] = facts;
        assert_success(&runtime.invoke(event)?);
    }

    let dynasor = runtime(packages, "plex_paper_dynasor", &["request.cancel@1"], None)?;
    let stopped = dynasor.invoke(schedule_event(
        "A",
        "G",
        json!({
            "confidence_ppm": 900,
            "stop_threshold_ppm": 800,
            "progress_ppm": 900
        }),
    ))?;
    assert_success(&stopped);
    assert_eq!(stopped["actions"][0]["method"], "pie.request.cancel@1");

    let justitia = runtime(packages, "plex_paper_justitia", &[], None)?;
    assert_success(&justitia.invoke(schedule_event("A", "G", json!({})))?);

    let hotprefix = runtime(
        packages,
        "plex_paper_hotprefix",
        &["cache.prefetch@1"],
        None,
    )?;
    assert_success(&hotprefix.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "hotprefix-feedback",
            "records": [{
                "subject": {"kind": "cache-object", "value": "object"},
                "outcome": "progress",
                "facts": {"reuse_count": 4}
            }]
        }
    }))?);
    let hot = hotprefix.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta("hotprefix-cache"),
            "cause": "insertion",
            "resident": [],
            "prospective": [{
                "object_id": "object",
                "size_bytes": 1,
                "beneficiaries": [],
                "beneficiary_count": 0,
                "facts": {}
            }],
            "capacity": {
                "max_bytes": 1,
                "fixed_bytes": 0,
                "facts": {"hot_threshold": 2}
            },
            "episode": null
        }
    }))?;
    assert_success(&hot);
    assert_eq!(hot["actions"][0]["method"], "pie.cache.prefetch@1");

    let pard = runtime(packages, "plex_paper_pard", &["request.cancel@1"], None)?;
    let dropped = pard.invoke(schedule_event(
        "A",
        "G",
        json!({
            "upstream_elapsed_ms": 80,
            "downstream_p95_ms": 40,
            "deadline_ms": 100
        }),
    ))?;
    assert_success(&dropped);
    assert_eq!(dropped["actions"][0]["method"], "pie.request.cancel@1");

    let branches = runtime(
        packages,
        "plex_paper_branch_regulation",
        &["request.cancel@1"],
        None,
    )?;
    let admitted = branches.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta("branch-admit"),
            "cause": "arrival",
            "candidates": [
                {
                    "request": request_ref("A", "G"),
                    "demand": [],
                    "facts": {
                        "branch_limit": 1,
                        "batch_interference": 1,
                        "interference_limit": 2
                    }
                },
                {
                    "request": request_ref("B", "G"),
                    "demand": [],
                    "facts": {
                        "branch_limit": 1,
                        "batch_interference": 1,
                        "interference_limit": 2
                    }
                }
            ],
            "capacity": {"max_accepted": 2, "limits": [], "facts": {}}
        },
        "lifecycle": [
            {
                "event": "create-group",
                "group_id": "G",
                "principal_id": "tenant",
                "limits": {"max_members": 4, "max_scratch_bytes": 4096},
                "facts": {}
            },
            {
                "event": "create-request",
                "request_id": "A",
                "principal_id": "tenant",
                "group_id": "G",
                "fields": {},
                "facts": {}
            },
            {
                "event": "create-request",
                "request_id": "B",
                "principal_id": "tenant",
                "group_id": "G",
                "fields": {},
                "facts": {}
            }
        ]
    }))?;
    assert_success(&admitted);
    assert_eq!(
        admitted["plan"]["plan"]["decisions"],
        json!(["accept", "defer"])
    );
    Ok(())
}

fn check_wave_a(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let vtc = runtime(packages, "plex_paper_vtc", &[], None)?;
    assert_success(&vtc.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "vtc-charge",
            "records": [{
                "subject": {"kind": "request", "value": "A"},
                "outcome": "progress",
                "facts": {"client_id": "client-a", "input_tokens": 100, "output_tokens": 0}
            }]
        },
        "lifecycle": lifecycle("A", "G1", true)
    }))?);
    let vtc_schedule = vtc.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "schedule",
        "context": {
            "meta": meta("vtc-schedule"),
            "cause": "capacity-changed",
            "runnable": [
                {
                    "request": request_ref("A", "G1"),
                    "max_token_budget": 4,
                    "facts": {"client_id": "client-a"}
                },
                {
                    "request": request_ref("B", "G2"),
                    "max_token_budget": 4,
                    "facts": {"client_id": "client-b"}
                }
            ],
            "capacity": {
                "max_selections": 1,
                "max_requests": 1,
                "max_total_tokens": 4,
                "facts": {}
            }
        },
        "lifecycle": lifecycle("B", "G2", true)
    }))?;
    assert_success(&vtc_schedule);
    assert_eq!(
        vtc_schedule["plan"]["plan"]["selections"][0]["requests"],
        json!([1])
    );

    let fairserve = runtime(packages, "plex_paper_fairserve", &[], None)?;
    assert_success(&fairserve.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "fairserve-charge",
            "records": [{
                "subject": {"kind": "request", "value": "A"},
                "outcome": "progress",
                "facts": {"client_id": "client-a", "service_tokens": 100}
            }]
        },
        "lifecycle": lifecycle("A", "G1", false)[..2].to_vec()
    }))?);
    let fair_admit = fairserve.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta("fairserve-admit"),
            "cause": "arrival",
            "candidates": [
                {
                    "request": request_ref("A", "G1"),
                    "demand": [],
                    "facts": {
                        "client_id": "client-a",
                        "weight": 1,
                        "interference_cost": 0
                    }
                },
                {
                    "request": request_ref("B", "G2"),
                    "demand": [],
                    "facts": {
                        "client_id": "client-b",
                        "weight": 4,
                        "interference_cost": 0
                    }
                }
            ],
            "capacity": {"max_accepted": 1, "limits": [], "facts": {}}
        },
        "lifecycle": lifecycle("B", "G2", false)[..2].to_vec()
    }))?;
    assert_success(&fair_admit);
    assert_eq!(
        fair_admit["plan"]["plan"]["decisions"],
        json!(["defer", "accept"])
    );

    let lmetric = runtime(packages, "plex_paper_lmetric", &[], None)?;
    let routed = lmetric.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta("lmetric"),
            "cause": "admission",
            "requests": [{"request": request_ref("A", "G"), "facts": {}}],
            "targets": [
                {
                    "target_id": "X",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {"hotspot": false}
                },
                {
                    "target_id": "Y",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {"hotspot": false}
                }
            ],
            "feasible_edges": [
                {
                    "request_index": 0,
                    "target_index": 0,
                    "demand": [],
                    "facts": {"new_prefill_tokens": 10, "current_batch_size": 4}
                },
                {
                    "request_index": 0,
                    "target_index": 1,
                    "demand": [],
                    "facts": {"new_prefill_tokens": 8, "current_batch_size": 2}
                }
            ]
        },
        "lifecycle": lifecycle("A", "G", false)
    }))?;
    assert_success(&routed);
    assert_eq!(routed["plan"]["plan"]["assignments"][0]["target_index"], 1);

    let marconi = runtime(packages, "plex_paper_marconi", &[], None)?;
    let cached = marconi.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta("marconi"),
            "cause": "insertion",
            "resident": [{
                "object": {
                    "object_id": "resident",
                    "size_bytes": 4,
                    "beneficiaries": [],
                    "beneficiary_count": 0,
                    "facts": {"reuse_probability_ppm": 1, "recompute_flops": 1}
                },
                "reclaimable": true
            }],
            "prospective": [{
                "object_id": "prospective",
                "size_bytes": 4,
                "beneficiaries": [],
                "beneficiary_count": 0,
                "facts": {"reuse_probability_ppm": 1000, "recompute_flops": 1000}
            }],
            "capacity": {"max_bytes": 4, "fixed_bytes": 0, "facts": {}},
            "episode": null
        }
    }))?;
    assert_success(&cached);
    assert_eq!(cached["plan"]["plan"]["admissions"], json!(["cache"]));
    assert_eq!(cached["plan"]["plan"]["reclaim"], json!([0]));

    let ragcache = runtime(packages, "plex_paper_ragcache", &[], None)?;
    let reclaimed = ragcache.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta("ragcache"),
            "cause": "dependency-progress",
            "resident": [
                {
                    "object": {
                        "object_id": "leaf-a",
                        "size_bytes": 1,
                        "beneficiaries": [],
                        "beneficiary_count": 0,
                        "facts": {"leaf": true, "frequency": 1, "recompute_cost": 1, "age": 0}
                    },
                    "reclaimable": true
                },
                {
                    "object": {
                        "object_id": "leaf-b",
                        "size_bytes": 1,
                        "beneficiaries": [],
                        "beneficiary_count": 0,
                        "facts": {"leaf": true, "frequency": 2, "recompute_cost": 10, "age": 0}
                    },
                    "reclaimable": true
                }
            ],
            "prospective": [],
            "capacity": {"max_bytes": 1, "fixed_bytes": 0, "facts": {}},
            "episode": {"episode_id": "rag-episode", "iteration": 0, "max_iterations": 2}
        }
    }))?;
    assert_success(&reclaimed);
    assert_eq!(reclaimed["plan"]["plan"]["reclaim"], json!([0]));
    Ok(())
}

fn check_replication_artifacts(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let policies = packages
        .parent()
        .and_then(Path::parent)
        .ok_or("package directory is not under tests/policies/target")?;
    let replications = policies.join("replications");
    let index: Value = serde_json::from_slice(&std::fs::read(replications.join("index.json"))?)?;
    assert_eq!(index["contract"], json!({"major": 0, "minor": 6}));
    for id in index["replications"]
        .as_array()
        .ok_or("replication index must contain an array")?
    {
        let id = id.as_str().ok_or("replication ID must be a string")?;
        let root = replications.join(id);
        let metadata: Value = serde_json::from_slice(&std::fs::read(root.join("metadata.json"))?)?;
        let _: Value = serde_json::from_slice(&std::fs::read(root.join("cases/basic.json"))?)?;
        let _: Value = serde_json::from_slice(&std::fs::read(root.join("expected/basic.json"))?)?;
        assert_eq!(metadata["id"], id);
        assert_eq!(metadata["validation_status"], "passing");
        assert!(matches!(
            metadata["evidence_level"].as_str(),
            Some("decision-trace-parity-with-deferred-mechanics")
                | Some("policy-kernel-reproduction")
        ));
        let component = metadata["component"]
            .as_str()
            .ok_or("replication component must be a string")?;
        let package = PolicyPackageV0_6::decode(
            &std::fs::read(packages.join(format!("{component}.plexpkg")))?,
            PackageLimits {
                max_package_bytes: 16 * 1024 * 1024,
                max_manifest_bytes: 1024 * 1024,
                max_component_bytes: 15 * 1024 * 1024,
            },
        )?;
        let operations = package
            .manifest()
            .implements
            .iter()
            .map(|operation| serde_json::to_value(operation).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(metadata["implements"], Value::Array(operations));
    }
    Ok(())
}

fn check_route_successes(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for package in ["plex_least_loaded", "plex_paper_preble"] {
        let runtime = runtime(packages, package, &[], None)?;
        assert_success(&runtime.invoke(route_event("A", "G", "route", json!({})))?);
    }
    Ok(())
}

fn check_admit_schedule_cache_feedback(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let rewrite = runtime(packages, "plex_rewrite_admit", &[], None)?;
    let admitted = rewrite.invoke(admit_event("A", "G", 0))?;
    assert_success(&admitted);
    assert_eq!(admitted["plan"]["plan"]["decisions"][0], "accept");
    assert_eq!(
        rewrite
            .backend()
            .read_request(&RequestId::from("A"))?
            .fields["admission_count"],
        1
    );

    let attained = runtime(packages, "plex_attained_service", &[], None)?;
    assert_success(&attained.invoke(schedule_event("A", "G", json!({})))?);

    let retention = runtime(packages, "plex_retention_score", &[], None)?;
    assert_success(&retention.invoke(cache_event(false, false))?);

    let feedback = runtime(packages, "plex_feedback_accounting", &[], None)?;
    let outcome = feedback.invoke(feedback_event("A", "G"))?;
    assert_success(&outcome);
    assert_eq!(
        feedback
            .backend()
            .read_request(&RequestId::from("A"))?
            .scratch["attained_service"],
        4
    );
    Ok(())
}

fn check_negative_rollback(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for (package, expected) in [
        ("plex_bad_budget", "invalid-output"),
        ("plex_malformed", "invalid-output"),
        ("plex_malformed_state_update", "invalid-output"),
        ("plex_invalid_shared_update", "invalid-output"),
        ("plex_mutate_unknown_request", "invalid-output"),
        ("plex_nonfinite", "invalid-output"),
        ("plex_fallback", "policy-fallback"),
        ("plex_mutate_request_facts", "policy-fallback"),
    ] {
        let runtime = runtime(packages, package, &[], None)?;
        let event = if package == "plex_bad_budget" {
            schedule_event("A", "G", json!({}))
        } else {
            route_event("A", "G", package, json!({}))
        };
        let outcome = runtime.invoke(event)?;
        assert_eq!(outcome["status"], "fallback", "{package}: {outcome}");
        assert_eq!(outcome["failure"]["kind"], expected, "{package}: {outcome}");
    }

    for package in ["plex_mutate_fail", "plex_trap"] {
        let runtime = runtime(packages, package, &[], None)?;
        let outcome = runtime.invoke(route_event("A", "G", package, json!({})))?;
        assert_eq!(outcome["status"], "fallback");
        assert!(
            runtime
                .backend()
                .read_request(&RequestId::from("A"))?
                .scratch
                .get("should_not_commit")
                .is_none()
        );
        assert!(
            runtime
                .backend()
                .read_shared()?
                .get("should_not_commit")
                .is_none()
        );
    }

    let spin = runtime(packages, "plex_spin", &[], None)?;
    let outcome = spin.invoke(route_event("A", "G", "spin", json!({})))?;
    assert_eq!(outcome["status"], "fallback");
    assert_eq!(outcome["failure"]["kind"], "fuel-exhausted");

    let action_bad = runtime(
        packages,
        "plex_action_bad_result",
        &["request.rebalance@1"],
        None,
    )?;
    let outcome = action_bad.invoke(route_event("A", "G", "action-bad", json!({})))?;
    assert_eq!(outcome["status"], "fallback");
    assert_eq!(outcome["failure"]["kind"], "invalid-output");
    assert!(outcome.get("actions").is_none());
    assert!(
        action_bad
            .backend()
            .read_request(&RequestId::from("A"))?
            .scratch
            .get("should_not_commit")
            .is_none()
    );

    let staged = runtime(
        packages,
        "plex_stage_action",
        &["request.rebalance@1"],
        None,
    )?;
    for (mode, expected) in [("fallback", "policy-fallback"), ("trap", "trap")] {
        let outcome = staged.invoke(route_event(
            &format!("request-{mode}"),
            &format!("group-{mode}"),
            &format!("stage-{mode}"),
            json!({"mode": mode}),
        ))?;
        assert_eq!(outcome["status"], "fallback");
        assert_eq!(outcome["failure"]["kind"], expected);
        assert!(outcome.get("actions").is_none());
    }
    Ok(())
}

fn check_queries_and_actions(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let queries = DictionaryQueryHandler::default();
    queries.insert(
        "pie.cluster.capacity@1",
        json!({"route_bias": 1, "value": 7}),
    );
    queries.insert("engine.custom-query@1", json!({"value": 7}));
    let queries: Arc<dyn QueryHandler> = Arc::new(queries);

    for package in [
        "plex_query_assisted",
        "plex_helper_methods",
        "plex_raw_helpers",
    ] {
        let mechanics = if package == "plex_query_assisted" {
            &[][..]
        } else {
            &["request.rebalance@1"][..]
        };
        let runtime = runtime(packages, package, mechanics, Some(queries.clone()))?;
        let outcome = runtime.invoke(route_event("A", "G", package, json!({})))?;
        assert_success(&outcome);
        if package != "plex_query_assisted" {
            assert_eq!(outcome["actions"][0]["method"], "pie.request.rebalance@1");
        }
    }
    Ok(())
}

fn check_existing_papers(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for package in [
        "plex_paper_agentix",
        "plex_paper_continuum",
        "plex_paper_helium",
    ] {
        let runtime = runtime(packages, package, &[], None)?;
        let facts = match package {
            "plex_paper_agentix" => json!({"waiting_ms": 100}),
            "plex_paper_continuum" => {
                json!({"preempted": false, "program_arrival": 1})
            }
            _ => json!({
                "ready": true,
                "dependency_depth": 2,
                "prefix_reuse_tokens": 8,
                "earliest_start": 0,
                "profiled_token_cost": 1
            }),
        };
        assert_success(&runtime.invoke(schedule_event("A", "G", facts))?);
    }

    let continuum = runtime(packages, "plex_paper_continuum", &[], None)?;
    assert_success(&continuum.invoke(cache_event(false, false))?);

    let kvflow = runtime(packages, "plex_paper_kvflow", &["cache.prefetch@1"], None)?;
    assert_success(&kvflow.invoke(schedule_event("A", "G", json!({"cache_ready": true})))?);
    let cached = kvflow.invoke(cache_event(true, false))?;
    assert_success(&cached);
    assert_eq!(cached["actions"][0]["method"], "pie.cache.prefetch@1");
    Ok(())
}

fn runtime(
    packages: &Path,
    name: &str,
    mechanics: &[&str],
    query: Option<Arc<dyn QueryHandler>>,
) -> Result<PlexRuntimeV0_6, Box<dyn std::error::Error>> {
    Ok(PlexRuntimeV0_6::from_package_bytes(
        &std::fs::read(packages.join(format!("{name}.plexpkg")))?,
        query,
        HostSupportV0_6::with_standard_ids(mechanics.iter().map(|value| (*value).to_owned()))?,
    )?)
}

fn request_ref(request_id: &str, group_id: &str) -> Value {
    json!({
        "request_id": request_id,
        "generation_id": 0,
        "group_id": group_id,
        "principal_id": "tenant"
    })
}

fn meta(opportunity_id: &str) -> Value {
    json!({
        "opportunity_id": opportunity_id,
        "snapshot": {"id": "host-filled", "revision": 0},
        "attempt": 0,
        "mechanics": []
    })
}

fn lifecycle(request_id: &str, group_id: &str, active: bool) -> Vec<Value> {
    let mut events = vec![
        json!({
            "event": "create-group",
            "group_id": group_id,
            "principal_id": "tenant",
            "limits": {"max_members": 4, "max_scratch_bytes": 4096},
            "facts": {}
        }),
        json!({
            "event": "create-request",
            "request_id": request_id,
            "principal_id": "tenant",
            "group_id": group_id,
            "fields": {},
            "facts": {}
        }),
        json!({"event": "admit-request", "request_id": request_id}),
    ];
    if active {
        events.push(json!({"event": "activate-request", "request_id": request_id}));
    }
    events
}

fn route_event(request_id: &str, group_id: &str, opportunity_id: &str, facts: Value) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "admission",
            "requests": [{"request": request_ref(request_id, group_id), "facts": facts}],
            "targets": [{
                "target_id": "node-a",
                "max_assignments": 1,
                "capacity": [],
                "revision": 1,
                "facts": {}
            }],
            "feasible_edges": [{
                "request_index": 0,
                "target_index": 0,
                "demand": [],
                "facts": {
                    "queue_depth": 0,
                    "cached_tokens": 8,
                    "load_cost": 1,
                    "eviction_cost": 1
                }
            }]
        },
        "lifecycle": lifecycle(request_id, group_id, false)
    })
}

fn admit_event(request_id: &str, group_id: &str, queue_depth: u64) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta(&format!("admit-{request_id}")),
            "cause": "arrival",
            "candidates": [{
                "request": request_ref(request_id, group_id),
                "demand": [],
                "facts": {"queue_depth": queue_depth}
            }],
            "capacity": {"max_accepted": 1, "limits": [], "facts": {}}
        },
        "lifecycle": lifecycle(request_id, group_id, false)[..2].to_vec()
    })
}

fn schedule_event(request_id: &str, group_id: &str, facts: Value) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "schedule",
        "context": {
            "meta": meta(&format!("schedule-{request_id}")),
            "cause": "capacity-changed",
            "runnable": [{
                "request": request_ref(request_id, group_id),
                "max_token_budget": 4,
                "facts": facts
            }],
            "capacity": {
                "max_selections": 1,
                "max_requests": 1,
                "max_total_tokens": 4,
                "facts": {}
            }
        },
        "lifecycle": lifecycle(request_id, group_id, true)
    })
}

fn cache_event(prefetch: bool, swap: bool) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta(&format!("cache-{prefetch}-{swap}")),
            "cause": "insertion",
            "resident": [],
            "prospective": [{
                "object_id": "object",
                "size_bytes": 1,
                "beneficiaries": [],
                "beneficiary_count": 0,
                "facts": {"prefetch": prefetch, "swap": swap}
            }],
            "capacity": {"max_bytes": 1, "fixed_bytes": 0, "facts": {}},
            "episode": null
        }
    })
}

fn feedback_event(request_id: &str, group_id: &str) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": format!("feedback-{request_id}"),
            "records": [{
                "subject": {"kind": "request", "value": request_id},
                "outcome": "progress",
                "facts": {"committed_tokens": 4, "tool_boundary": true}
            }]
        },
        "lifecycle": lifecycle(request_id, group_id, true)
    })
}

fn assert_success(outcome: &Value) {
    assert_eq!(outcome["status"], "success", "{outcome}");
}
