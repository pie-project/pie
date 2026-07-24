use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pie_plex::v0_6::{GroupId, RequestId};
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
    check_unavailable_feedback_cleanup(&packages)?;
    check_queries_and_actions(&packages)?;
    check_existing_papers(&packages)?;
    check_wave_a(&packages)?;
    check_fairness_state_machines(&packages)?;
    check_wave_c(&packages)?;
    check_wave_d(&packages)?;
    check_replication_artifacts(&packages)?;
    check_deterministic_replay(&packages)?;

    println!("PLEX v0.6 policy fixtures passed");
    Ok(())
}

fn check_fairness_state_machines(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    check_vtc_state_machine(packages)?;
    check_agentix_state_machine(packages)?;
    check_fairserve_state_machine(packages)?;
    check_dlpm_state_machine(packages)?;
    check_justitia_state_machine(packages)?;
    Ok(())
}

fn check_vtc_state_machine(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    const SCALE: u64 = 1_000_000;

    let vtc = runtime(packages, "plex_paper_vtc", &[], None)?;
    let lifecycle = active_lifecycle(&[("A1", "GA"), ("A2", "GA"), ("B1", "GB"), ("C1", "GC")]);
    assert_success(&vtc.invoke(feedback_records(
        "vtc-initial-counters",
        vec![
            request_progress(
                "A1",
                json!({
                    "client_id": "client-a",
                    "output_tokens": 5,
                    "output_weight": 1,
                    "output_price": 2,
                    "fair_weight_ppm": 1_000_000
                }),
            ),
            request_progress(
                "C1",
                json!({
                    "client_id": "client-c",
                    "output_tokens": 15,
                    "output_weight": 1,
                    "output_price": 2,
                    "fair_weight_ppm": 1_000_000
                }),
            ),
        ],
        lifecycle,
    ))?);
    let scheduled = vtc.invoke(schedule_many(
        "vtc-lift-fifo",
        vec![
            schedule_candidate(
                "A1",
                "GA",
                1,
                json!({
                    "client_id": "client-a",
                    "client_became_active": false,
                    "queue_member": true,
                    "dispatch_input_tokens": 1
                }),
            ),
            schedule_candidate(
                "A2",
                "GA",
                1,
                json!({
                    "client_id": "client-a",
                    "client_became_active": false,
                    "queue_member": true,
                    "dispatch_input_tokens": 0
                }),
            ),
            schedule_candidate(
                "B1",
                "GB",
                1,
                json!({
                    "client_id": "client-b",
                    "client_became_active": true,
                    "queue_member": true,
                    "dispatch_input_tokens": 3
                }),
            ),
            schedule_candidate(
                "C1",
                "GC",
                1,
                json!({
                    "client_id": "client-c",
                    "client_became_active": false,
                    "queue_member": true,
                    "dispatch_input_tokens": 0
                }),
            ),
        ],
        3,
        3,
        Vec::new(),
    ))?;
    assert_success(&scheduled);
    assert_eq!(
        scheduled["plan"]["plan"]["selections"],
        json!([
            {"requests": [0], "token_budgets": [1]},
            {"requests": [2], "token_budgets": [1]},
            {"requests": [1], "token_budgets": [1]}
        ])
    );
    let shared = vtc.backend().read_shared()?;
    assert_eq!(shared["vtc"]["client-a"], 10 * SCALE);
    assert_eq!(shared["vtc"]["client-b"], 10 * SCALE);
    assert_eq!(shared["vtc"]["client-c"], 30 * SCALE);
    assert_success(
        &vtc.invoke(feedback_records(
            "vtc-lift-fifo-enacted",
            (0..3)
                .map(|selection_index| {
                    schedule_progress("vtc-lift-fifo", selection_index, "enacted", 1)
                })
                .collect(),
            Vec::new(),
        ))?,
    );
    let shared = vtc.backend().read_shared()?;
    assert_eq!(shared["vtc"]["client-a"], 11 * SCALE);
    assert_eq!(shared["vtc"]["client-b"], 13 * SCALE);
    assert_eq!(shared["vtc_queued_clients"], json!(["client-c"]));

    let enactment = runtime(packages, "plex_paper_vtc", &[], None)?;
    let lifecycle = active_lifecycle(&[("R", "GR")]);
    assert_success(&enactment.invoke(schedule_many(
        "vtc-rejected",
        vec![schedule_candidate(
            "R",
            "GR",
            4,
            json!({
                "client_id": "client-r",
                "queue_member": true,
                "dispatch_input_tokens": 5,
                "fair_weight_ppm": 1_000_000
            }),
        )],
        1,
        4,
        lifecycle,
    ))?);
    assert_eq!(enactment.backend().read_shared()?["vtc"]["client-r"], 0);
    assert_success(&enactment.invoke(feedback_records(
        "vtc-rejected-feedback",
        vec![schedule_progress("vtc-rejected", 0, "not-enacted", 0)],
        Vec::new(),
    ))?);
    assert_eq!(enactment.backend().read_shared()?["vtc"]["client-r"], 0);
    assert_success(&enactment.invoke(schedule_many(
        "vtc-partial",
        vec![schedule_candidate(
            "R",
            "GR",
            4,
            json!({
                "client_id": "client-r",
                "queue_member": true,
                "dispatch_input_tokens": 5,
                "fair_weight_ppm": 1_000_000
            }),
        )],
        1,
        4,
        Vec::new(),
    ))?);
    assert_success(&enactment.invoke(feedback_records(
        "vtc-partial-feedback",
        vec![schedule_progress("vtc-partial", 0, "partially-enacted", 1)],
        Vec::new(),
    ))?);
    assert_eq!(
        enactment.backend().read_shared()?["vtc"]["client-r"],
        5 * SCALE
    );
    assert_success(&enactment.invoke(feedback_records(
        "vtc-weighted-output",
        vec![request_progress(
            "R",
            json!({
                "client_id": "client-r",
                "output_tokens": 3,
                "output_weight": 2,
                "output_price": 2,
                "fair_weight_ppm": 2_000_000
            }),
        )],
        Vec::new(),
    ))?);
    assert_eq!(
        enactment.backend().read_shared()?["vtc"]["client-r"],
        11 * SCALE
    );

    let idle = runtime(packages, "plex_paper_vtc", &[], None)?;
    let lifecycle = active_lifecycle(&[("IA", "GIA"), ("IB", "GIB")]);
    assert_success(&idle.invoke(feedback_records(
        "vtc-idle-counter",
        vec![request_progress(
            "IA",
            json!({
                "client_id": "client-a",
                "output_tokens": 5,
                "output_price": 2,
                "fair_weight_ppm": 1_000_000
            }),
        )],
        lifecycle,
    ))?);
    assert_success(&idle.invoke(schedule_many(
        "vtc-drain",
        vec![schedule_candidate(
            "IA",
            "GIA",
            1,
            json!({
                "client_id": "client-a",
                "client_became_active": false,
                "queue_member": true,
                "dispatch_input_tokens": 0
            }),
        )],
        1,
        1,
        Vec::new(),
    ))?);
    assert_success(&idle.invoke(feedback_records(
        "vtc-drain-feedback",
        vec![schedule_progress("vtc-drain", 0, "enacted", 1)],
        Vec::new(),
    ))?);
    assert_eq!(idle.backend().read_shared()?["vtc_last_client"], "client-a");
    assert_success(&idle.invoke(schedule_many(
        "vtc-after-idle",
        vec![schedule_candidate(
            "IB",
            "GIB",
            1,
            json!({
                "client_id": "client-b",
                "client_became_active": true,
                "queue_member": true,
                "dispatch_input_tokens": 0
            }),
        )],
        1,
        1,
        Vec::new(),
    ))?);
    assert_eq!(idle.backend().read_shared()?["vtc"]["client-b"], 10 * SCALE);
    Ok(())
}

fn check_agentix_state_machine(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let queue_facts = json!({
        "agentix_mode": "plas",
        "queue_bounds_us": [1000, 3000],
        "queue_quanta_us": [1000, 2000, 4000],
        "starvation_ratio_ppm": 4_000_000,
        "program_wait_us": 0,
        "call_wait_us": 0
    });

    let agentix = runtime(packages, "plex_paper_agentix", &[], None)?;
    let lifecycle = active_lifecycle(&[("A", "GA"), ("B", "GB")]);
    let mut a_facts = queue_facts.clone();
    a_facts["call_arrival"] = json!(0);
    let mut b_facts = queue_facts.clone();
    b_facts["call_arrival"] = json!(1);
    assert_success(&agentix.invoke(schedule_many(
        "agentix-initialize",
        vec![
            schedule_candidate("A", "GA", 1, a_facts.clone()),
            schedule_candidate("B", "GB", 1, b_facts.clone()),
        ],
        1,
        1,
        lifecycle,
    ))?);
    assert_success(&agentix.invoke(feedback_records(
        "agentix-a-service",
        vec![request_progress(
            "A",
            json!({
                "service_us": 1500,
                "queue_bounds_us": [1000, 3000],
                "queue_quanta_us": [1000, 2000, 4000]
            }),
        )],
        Vec::new(),
    ))?);
    let request = agentix.backend().read_request(&RequestId::from("A"))?;
    assert_eq!(request.scratch["agentix_queue_level"], 1);
    assert_eq!(request.scratch["agentix_remaining_quantum_us"], 1500);
    assert_eq!(request.scratch["agentix_model_time_us"], 1500);
    let scheduled = agentix.invoke(schedule_many(
        "agentix-demotion",
        vec![
            schedule_candidate("A", "GA", 1, a_facts.clone()),
            schedule_candidate("B", "GB", 1, b_facts.clone()),
        ],
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([1])
    );

    a_facts["call_wait_us"] = json!(10_000);
    let scheduled = agentix.invoke(schedule_many(
        "agentix-starvation",
        vec![schedule_candidate("A", "GA", 1, a_facts.clone())],
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([0])
    );
    let request = agentix.backend().read_request(&RequestId::from("A"))?;
    assert_eq!(request.scratch["agentix_queue_level"], 0);
    assert_eq!(request.scratch["agentix_remaining_quantum_us"], 1000);
    assert_eq!(request.scratch["agentix_model_time_us"], 0);
    assert_eq!(request.scratch["agentix_wait_reset_us"], 10_000);
    assert_eq!(request.scratch["agentix_promotions"], 1);

    let plas = runtime(packages, "plex_paper_agentix", &[], None)?;
    let lifecycle = active_lifecycle(&[("S1", "GS"), ("S2", "GS")]);
    assert_success(&plas.invoke(schedule_many(
        "agentix-plas-first",
        vec![schedule_candidate(
            "S1",
            "GS",
            1,
            json!({
                "agentix_mode": "plas",
                "queue_bounds_us": [1000],
                "queue_quanta_us": [1000, 2000],
                "call_arrival": 0
            }),
        )],
        1,
        1,
        lifecycle,
    ))?);
    assert_success(&plas.invoke(feedback_records(
        "agentix-plas-first-progress",
        vec![request_progress(
            "S1",
            json!({
                "service_us": 800,
                "queue_bounds_us": [1000],
                "queue_quanta_us": [1000, 2000]
            }),
        )],
        Vec::new(),
    ))?);
    assert_success(&plas.invoke(feedback_records(
        "agentix-plas-first-complete",
        vec![request_completed("S1", json!({"call_wait_us": 0}))],
        Vec::new(),
    ))?);
    assert_eq!(
        plas.backend().read_group(&GroupId::from("GS"))?.scratch["agentix_service_us"],
        800
    );
    assert_success(&plas.invoke(schedule_many(
        "agentix-plas-second",
        vec![schedule_candidate(
            "S2",
            "GS",
            1,
            json!({
                "agentix_mode": "plas",
                "queue_bounds_us": [1000],
                "queue_quanta_us": [1000, 2000],
                "call_arrival": 1
            }),
        )],
        1,
        1,
        Vec::new(),
    ))?);
    assert_eq!(
        plas.backend().read_request(&RequestId::from("S2"))?.scratch["agentix_inherited_service_us"],
        800
    );
    assert_success(&plas.invoke(feedback_records(
        "agentix-plas-second-progress",
        vec![request_progress(
            "S2",
            json!({
                "service_us": 200,
                "queue_bounds_us": [1000],
                "queue_quanta_us": [1000, 2000]
            }),
        )],
        Vec::new(),
    ))?);
    assert_success(&plas.invoke(feedback_records(
        "agentix-plas-second-complete",
        vec![request_completed("S2", json!({"call_wait_us": 0}))],
        Vec::new(),
    ))?);
    assert_eq!(
        plas.backend().read_group(&GroupId::from("GS"))?.scratch["agentix_service_us"],
        1000
    );

    let atlas = runtime(packages, "plex_paper_agentix", &[], None)?;
    let lifecycle = active_lifecycle(&[
        ("P1", "GP"),
        ("P2", "GP"),
        ("P3", "GP"),
        ("P4", "GP"),
        ("Q1", "GQ"),
        ("Q2", "GQ"),
    ]);
    let atlas_facts = |arrival| {
        json!({
            "agentix_mode": "atlas",
            "queue_bounds_us": [400],
            "queue_quanta_us": [1000, 2000],
            "call_arrival": arrival
        })
    };
    assert_success(&atlas.invoke(schedule_many(
        "agentix-atlas-fork",
        vec![
            schedule_candidate("P1", "GP", 1, atlas_facts(0)),
            schedule_candidate("P2", "GP", 1, atlas_facts(1)),
        ],
        2,
        2,
        lifecycle,
    ))?);
    assert_success(&atlas.invoke(feedback_records(
        "agentix-atlas-fork-progress",
        vec![
            request_progress(
                "P1",
                json!({
                    "service_us": 300,
                    "queue_bounds_us": [400],
                    "queue_quanta_us": [1000, 2000]
                }),
            ),
            request_progress(
                "P2",
                json!({
                    "service_us": 100,
                    "queue_bounds_us": [400],
                    "queue_quanta_us": [1000, 2000]
                }),
            ),
        ],
        Vec::new(),
    ))?);
    assert_success(&atlas.invoke(feedback_records(
        "agentix-atlas-fork-complete",
        vec![
            request_completed("P1", json!({"call_wait_us": 0})),
            request_completed("P2", json!({"call_wait_us": 0})),
        ],
        Vec::new(),
    ))?);
    assert_eq!(
        atlas.backend().read_group(&GroupId::from("GP"))?.scratch["agentix_service_us"],
        300
    );
    assert_success(&atlas.invoke(schedule_many(
        "agentix-atlas-other",
        vec![schedule_candidate("Q1", "GQ", 1, atlas_facts(0))],
        1,
        1,
        Vec::new(),
    ))?);
    assert_success(&atlas.invoke(feedback_records(
        "agentix-atlas-other-progress",
        vec![request_progress(
            "Q1",
            json!({
                "service_us": 500,
                "queue_bounds_us": [400],
                "queue_quanta_us": [1000, 2000]
            }),
        )],
        Vec::new(),
    ))?);
    assert_success(&atlas.invoke(feedback_records(
        "agentix-atlas-other-complete",
        vec![request_completed("Q1", json!({"call_wait_us": 0}))],
        Vec::new(),
    ))?);
    let scheduled = atlas.invoke(schedule_many(
        "agentix-atlas-join",
        vec![
            schedule_candidate("P3", "GP", 1, atlas_facts(2)),
            schedule_candidate("P4", "GP", 1, atlas_facts(3)),
            schedule_candidate("Q2", "GQ", 1, atlas_facts(4)),
        ],
        2,
        2,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"],
        json!([
            {"requests": [0], "token_budgets": [1]},
            {"requests": [1], "token_budgets": [1]}
        ])
    );
    for request_id in ["P3", "P4"] {
        assert_eq!(
            atlas
                .backend()
                .read_request(&RequestId::from(request_id))?
                .scratch["agentix_inherited_service_us"],
            300
        );
    }
    assert_eq!(
        atlas
            .backend()
            .read_request(&RequestId::from("Q2"))?
            .scratch["agentix_inherited_service_us"],
        500
    );
    Ok(())
}

fn check_fairserve_state_machine(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let oit = runtime(packages, "plex_paper_fairserve", &[], None)?;
    let lifecycle =
        pending_lifecycle(&[("U1", "GU1"), ("U2", "GU2"), ("U3", "GU3"), ("U4", "GU4")]);
    let admitted = oit.invoke(admit_many(
        "fairserve-oit",
        vec![
            admit_candidate(
                "U1",
                "GU1",
                json!({
                    "user_id": "user-a",
                    "application_id": "app-a",
                    "now_ms": 0,
                    "rpm_window_ms": 60000,
                    "user_rpm_limit": 2,
                    "app_rpm_limit": 10,
                    "kv_overloaded": true,
                    "interaction_in_progress": false
                }),
            ),
            admit_candidate(
                "U2",
                "GU2",
                json!({
                    "user_id": "user-a",
                    "application_id": "app-a",
                    "now_ms": 1,
                    "rpm_window_ms": 60000,
                    "user_rpm_limit": 2,
                    "app_rpm_limit": 10,
                    "kv_overloaded": true,
                    "interaction_in_progress": false
                }),
            ),
            admit_candidate(
                "U3",
                "GU3",
                json!({
                    "user_id": "user-a",
                    "application_id": "app-a",
                    "now_ms": 2,
                    "rpm_window_ms": 60000,
                    "user_rpm_limit": 2,
                    "app_rpm_limit": 10,
                    "kv_overloaded": true,
                    "interaction_in_progress": false
                }),
            ),
            admit_candidate(
                "U4",
                "GU4",
                json!({
                    "user_id": "user-a",
                    "application_id": "app-a",
                    "now_ms": 3,
                    "rpm_window_ms": 60000,
                    "user_rpm_limit": 2,
                    "app_rpm_limit": 10,
                    "kv_overloaded": true,
                    "interaction_in_progress": true
                }),
            ),
        ],
        4,
        lifecycle,
    ))?;
    assert_success(&admitted);
    assert_eq!(
        admitted["plan"]["plan"]["decisions"],
        json!(["accept", "accept", "defer", "accept"])
    );
    let admitted = oit.invoke(admit_many(
        "fairserve-window-expired",
        vec![admit_candidate(
            "U5",
            "GU5",
            json!({
                "user_id": "user-a",
                "application_id": "app-a",
                "now_ms": 70000,
                "rpm_window_ms": 60000,
                "user_rpm_limit": 2,
                "app_rpm_limit": 10,
                "kv_overloaded": true,
                "interaction_in_progress": false
            }),
        )],
        1,
        pending_lifecycle(&[("U5", "GU5")]),
    ))?;
    assert_eq!(admitted["plan"]["plan"]["decisions"], json!(["accept"]));

    let wsc = runtime(packages, "plex_paper_fairserve", &[], None)?;
    let lifecycle = active_lifecycle(&[("A", "GA"), ("B", "GB")]);
    assert_success(&wsc.invoke(feedback_records(
        "fairserve-user-a-progress",
        vec![request_progress(
            "A",
            json!({
                "input_tokens": 10,
                "system_tokens": 0,
                "output_tokens": 0
            }),
        )],
        lifecycle,
    ))?);
    assert_success(&wsc.invoke(feedback_records(
        "fairserve-user-a-complete",
        vec![request_completed(
            "A",
            json!({
                "user_id": "user-a",
                "application_id": "shared-app",
                "stage_id": "stage-1",
                "expected_input_tokens": 10,
                "expected_system_tokens": 0,
                "expected_output_tokens": 0,
                "user_priority_ppm": 1000000
            }),
        )],
        Vec::new(),
    ))?);
    let shared = wsc.backend().read_shared()?;
    assert_eq!(shared["fairserve_users"]["user-a"], 1_000_000);
    assert_eq!(
        shared["fairserve_stage_service"]["user-a"]["shared-app::stage-1"],
        1_000_000
    );
    let scheduled = wsc.invoke(schedule_many(
        "fairserve-user-isolation",
        vec![
            schedule_candidate(
                "A",
                "GA",
                1,
                json!({
                    "user_id": "user-a",
                    "application_id": "shared-app",
                    "interaction_in_progress": false,
                    "user_became_active": false,
                    "arrival_seq": 0
                }),
            ),
            schedule_candidate(
                "B",
                "GB",
                1,
                json!({
                    "user_id": "user-b",
                    "application_id": "shared-app",
                    "interaction_in_progress": false,
                    "user_became_active": false,
                    "arrival_seq": 1
                }),
            ),
        ],
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([1])
    );
    let scheduled = wsc.invoke(schedule_many(
        "fairserve-interaction-priority",
        vec![
            schedule_candidate(
                "A",
                "GA",
                1,
                json!({
                    "user_id": "user-a",
                    "application_id": "shared-app",
                    "interaction_in_progress": true,
                    "user_became_active": false,
                    "arrival_seq": 0
                }),
            ),
            schedule_candidate(
                "B",
                "GB",
                1,
                json!({
                    "user_id": "user-b",
                    "application_id": "shared-app",
                    "interaction_in_progress": false,
                    "user_became_active": false,
                    "arrival_seq": 1
                }),
            ),
        ],
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([0])
    );

    let normalized = runtime(packages, "plex_paper_fairserve", &[], None)?;
    let lifecycle = active_lifecycle(&[("C", "GC"), ("D", "GD")]);
    assert_success(&normalized.invoke(feedback_records(
        "fairserve-normalized-progress",
        vec![
            request_progress("C", json!({"input_tokens": 10})),
            request_progress("D", json!({"input_tokens": 10})),
        ],
        lifecycle,
    ))?);
    assert_success(&normalized.invoke(feedback_records(
        "fairserve-normalized-complete",
        vec![
            request_completed(
                "C",
                json!({
                    "user_id": "user-c",
                    "application_id": "app-c",
                    "stage_id": "stage-1",
                    "expected_input_tokens": 20
                }),
            ),
            request_completed(
                "D",
                json!({
                    "user_id": "user-d",
                    "application_id": "app-d",
                    "stage_id": "stage-1",
                    "expected_input_tokens": 10
                }),
            ),
        ],
        Vec::new(),
    ))?);
    let shared = normalized.backend().read_shared()?;
    assert_eq!(shared["fairserve_users"]["user-c"], 500_000);
    assert_eq!(shared["fairserve_users"]["user-d"], 1_000_000);
    let scheduled = normalized.invoke(schedule_many(
        "fairserve-stage-normalization",
        vec![
            schedule_candidate(
                "C",
                "GC",
                1,
                json!({
                    "user_id": "user-c",
                    "application_id": "app-c",
                    "user_became_active": false,
                    "arrival_seq": 0
                }),
            ),
            schedule_candidate(
                "D",
                "GD",
                1,
                json!({
                    "user_id": "user-d",
                    "application_id": "app-d",
                    "user_became_active": false,
                    "arrival_seq": 1
                }),
            ),
        ],
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([0])
    );
    Ok(())
}

fn check_dlpm_state_machine(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let local = runtime(packages, "plex_paper_dlpm", &[], None)?;
    let lifecycle = active_lifecycle(&[("A1", "GA"), ("A2", "GA"), ("B1", "GB")]);
    let scheduled = local.invoke(schedule_many(
        "dlpm-local",
        vec![
            schedule_candidate(
                "A1",
                "GA",
                3,
                json!({
                    "client_id": "client-a",
                    "queue_member": true,
                    "cached_tokens": 100,
                    "client_quantum": 4,
                    "extend_tokens": 3,
                    "extend_weight": 1
                }),
            ),
            schedule_candidate(
                "A2",
                "GA",
                3,
                json!({
                    "client_id": "client-a",
                    "queue_member": true,
                    "cached_tokens": 90,
                    "client_quantum": 4,
                    "extend_tokens": 3,
                    "extend_weight": 1
                }),
            ),
            schedule_candidate(
                "B1",
                "GB",
                3,
                json!({
                    "client_id": "client-b",
                    "queue_member": true,
                    "cached_tokens": 80,
                    "client_quantum": 4,
                    "extend_tokens": 3,
                    "extend_weight": 1
                }),
            ),
        ],
        3,
        9,
        lifecycle,
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"],
        json!([
            {"requests": [0], "token_budgets": [3]},
            {"requests": [1], "token_budgets": [3]},
            {"requests": [2], "token_budgets": [3]}
        ])
    );
    let shared = local.backend().read_shared()?;
    assert_eq!(shared["dlpm_deficit"]["client-a"], 4);
    assert_eq!(shared["dlpm_deficit"]["client-b"], 4);
    assert_success(&local.invoke(feedback_records(
        "dlpm-local-enactment",
        vec![
            schedule_progress("dlpm-local", 0, "enacted", 3),
            schedule_progress("dlpm-local", 1, "not-enacted", 0),
            schedule_progress("dlpm-local", 2, "partially-enacted", 2),
        ],
        Vec::new(),
    ))?);
    let shared = local.backend().read_shared()?;
    assert_eq!(shared["dlpm_deficit"]["client-a"], 1);
    assert_eq!(shared["dlpm_deficit"]["client-b"], 2);
    assert_success(&local.invoke(feedback_records(
        "dlpm-local-output",
        vec![request_progress(
            "A1",
            json!({
                "client_id": "client-a",
                "output_tokens": 2,
                "output_weight": 2
            }),
        )],
        Vec::new(),
    ))?);
    assert_eq!(
        local.backend().read_shared()?["dlpm_deficit"]["client-a"],
        -3
    );
    let scheduled = local.invoke(schedule_many(
        "dlpm-positive-only",
        vec![
            schedule_candidate(
                "A1",
                "GA",
                1,
                json!({
                    "client_id": "client-a",
                    "queue_member": true,
                    "cached_tokens": 100,
                    "client_quantum": 4,
                    "extend_tokens": 1
                }),
            ),
            schedule_candidate(
                "B1",
                "GB",
                1,
                json!({
                    "client_id": "client-b",
                    "queue_member": true,
                    "cached_tokens": 10,
                    "client_quantum": 4,
                    "extend_tokens": 1
                }),
            ),
        ],
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([1])
    );

    let distributed = runtime(packages, "plex_paper_dlpm", &[], None)?;
    let lifecycle = active_lifecycle(&[("C1", "GC1"), ("C2", "GC2"), ("C3", "GC3"), ("C4", "GC4")]);
    let route = |opportunity: &str, request_id: &str, input_tokens: u64, lifecycle| {
        route_many(
            opportunity,
            vec![route_candidate(
                request_id,
                &format!("G{request_id}"),
                json!({
                    "client_id": "client-c",
                    "input_tokens": input_tokens,
                    "input_weight": 2
                }),
            )],
            vec![
                route_target("worker-x", 1, json!({"worker_quantum": 10})),
                route_target("worker-y", 1, json!({"worker_quantum": 10})),
                route_target("worker-z", 1, json!({"worker_quantum": 10})),
            ],
            vec![
                route_edge(
                    0,
                    0,
                    json!({
                        "cached_tokens": 100,
                        "longest_prefix_match": true,
                        "queue_size": 5
                    }),
                ),
                route_edge(
                    0,
                    1,
                    json!({
                        "cached_tokens": 100,
                        "longest_prefix_match": true,
                        "queue_size": 1
                    }),
                ),
                route_edge(
                    0,
                    2,
                    json!({
                        "cached_tokens": 10,
                        "longest_prefix_match": false,
                        "queue_size": 0
                    }),
                ),
            ],
            lifecycle,
        )
    };
    let routed = distributed.invoke(route("dlpm-route-1", "C1", 6, lifecycle))?;
    assert_eq!(routed["plan"]["plan"]["assignments"][0]["target_index"], 1);
    assert_success(&distributed.invoke(feedback_records(
        "dlpm-route-1-feedback",
        vec![route_progress("dlpm-route-1", 0, "enacted")],
        Vec::new(),
    ))?);
    assert_eq!(
        distributed.backend().read_shared()?["dlpm_worker_deficit"]["client-c"]["worker-y"],
        -2
    );

    let routed = distributed.invoke(route("dlpm-route-2", "C2", 6, Vec::new()))?;
    assert_eq!(routed["plan"]["plan"]["assignments"][0]["target_index"], 0);
    assert_success(&distributed.invoke(feedback_records(
        "dlpm-route-2-feedback",
        vec![route_progress("dlpm-route-2", 0, "enacted")],
        Vec::new(),
    ))?);
    let routed = distributed.invoke(route("dlpm-route-3", "C3", 10, Vec::new()))?;
    assert_eq!(routed["plan"]["plan"]["assignments"][0]["target_index"], 2);
    assert_success(&distributed.invoke(feedback_records(
        "dlpm-route-3-feedback",
        vec![route_progress("dlpm-route-3", 0, "enacted")],
        Vec::new(),
    ))?);
    let routed = distributed.invoke(route("dlpm-route-4", "C4", 0, Vec::new()))?;
    assert_eq!(routed["plan"]["plan"]["assignments"][0]["target_index"], 1);
    assert_success(&distributed.invoke(feedback_records(
        "dlpm-route-4-feedback",
        vec![route_progress("dlpm-route-4", 0, "enacted")],
        Vec::new(),
    ))?);
    let shared = distributed.backend().read_shared()?;
    assert_eq!(shared["dlpm_worker_deficit"]["client-c"]["worker-x"], 8);
    assert_eq!(shared["dlpm_worker_deficit"]["client-c"]["worker-y"], 8);
    assert_eq!(shared["dlpm_worker_deficit"]["client-c"]["worker-z"], 0);
    assert_success(&distributed.invoke(feedback_records(
        "dlpm-worker-output",
        vec![request_completed(
            "C1",
            json!({
                "client_id": "client-c",
                "target_id": "worker-y",
                "output_tokens": 3,
                "output_weight": 2
            }),
        )],
        Vec::new(),
    ))?);
    assert_eq!(
        distributed.backend().read_shared()?["dlpm_worker_deficit"]["client-c"]["worker-y"],
        2
    );
    Ok(())
}

fn check_justitia_state_machine(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let justitia = runtime(
        packages,
        "plex_paper_justitia",
        &["schedule.atomic-enqueue@1"],
        None,
    )?;
    let lifecycle = active_lifecycle(&[
        ("A1", "GA"),
        ("B1", "GB"),
        ("B2", "GB"),
        ("C1", "GC"),
        ("D1", "GD"),
    ]);
    let scheduled = justitia.invoke(schedule_many_limits(
        "justitia-arrivals",
        vec![
            schedule_candidate(
                "A1",
                "GA",
                1,
                json!({
                    "ready": true,
                    "now_us": 0,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 100
                }),
            ),
            schedule_candidate(
                "B1",
                "GB",
                1,
                json!({
                    "ready": true,
                    "now_us": 0,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 50
                }),
            ),
            schedule_candidate(
                "B2",
                "GB",
                1,
                json!({
                    "ready": true,
                    "now_us": 0,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 50
                }),
            ),
        ],
        1,
        2,
        2,
        lifecycle,
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"],
        json!([{"requests": [1, 2], "token_budgets": [1, 1]}])
    );
    assert_eq!(
        justitia.backend().read_group(&GroupId::from("GA"))?.scratch["justitia_finish_tag_fp"],
        100_000_000
    );
    assert_eq!(
        justitia.backend().read_group(&GroupId::from("GB"))?.scratch["justitia_finish_tag_fp"],
        50_000_000
    );

    assert_success(&justitia.invoke(schedule_many_limits(
        "justitia-immutable-tag",
        vec![
            schedule_candidate(
                "A1",
                "GA",
                1,
                json!({
                    "ready": true,
                    "now_us": 5,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 1000
                }),
            ),
            schedule_candidate(
                "B1",
                "GB",
                1,
                json!({
                    "ready": true,
                    "now_us": 5,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 1000
                }),
            ),
        ],
        1,
        1,
        1,
        Vec::new(),
    ))?);
    assert_eq!(
        justitia.backend().read_group(&GroupId::from("GB"))?.scratch["justitia_finish_tag_fp"],
        50_000_000
    );
    assert_success(&justitia.invoke(feedback_records(
        "justitia-complete-b",
        vec![json!({
            "subject": {"kind": "work-group", "value": "GB"},
            "outcome": "completed",
            "facts": {"now_us": 10, "total_kv_tokens": 100}
        })],
        Vec::new(),
    ))?);
    let shared = justitia.backend().read_shared()?;
    assert_eq!(shared["justitia_virtual_time_fp"], 500_000_000);
    assert_eq!(shared["justitia_active_groups"], json!(["GA"]));

    let scheduled = justitia.invoke(schedule_many_limits(
        "justitia-late-arrival",
        vec![
            schedule_candidate(
                "A1",
                "GA",
                1,
                json!({
                    "ready": true,
                    "now_us": 20,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 100
                }),
            ),
            schedule_candidate(
                "C1",
                "GC",
                1,
                json!({
                    "ready": true,
                    "now_us": 20,
                    "total_kv_tokens": 100,
                    "predicted_agent_kv_token_time": 10
                }),
            ),
            schedule_candidate(
                "D1",
                "GD",
                1,
                json!({
                    "ready": true,
                    "now_us": 20,
                    "total_kv_tokens": 100,
                    "predicted_input_tokens": 10,
                    "predicted_output_tokens": 4
                }),
            ),
        ],
        1,
        1,
        1,
        Vec::new(),
    ))?;
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["requests"],
        json!([0])
    );
    assert_eq!(
        justitia.backend().read_group(&GroupId::from("GC"))?.scratch["justitia_finish_tag_fp"],
        1_510_000_000
    );
    let group_d = justitia.backend().read_group(&GroupId::from("GD"))?;
    assert_eq!(group_d.scratch["justitia_predicted_cost"], 50);
    assert_eq!(group_d.scratch["justitia_finish_tag_fp"], 1_550_000_000);

    assert_success(&justitia.invoke(feedback_records(
        "justitia-progress-and-complete-a",
        vec![
            request_progress(
                "A1",
                json!({
                    "now_us": 25,
                    "total_kv_tokens": 100,
                    "kv_token_time_delta": 20
                }),
            ),
            request_completed(
                "A1",
                json!({
                    "now_us": 25,
                    "total_kv_tokens": 100,
                    "agent_completed": true
                }),
            ),
        ],
        Vec::new(),
    ))?);
    assert_eq!(
        justitia.backend().read_group(&GroupId::from("GA"))?.scratch["justitia_consumed_cost"],
        20
    );
    assert_eq!(
        justitia.backend().read_shared()?["justitia_active_groups"],
        json!(["GC", "GD"])
    );
    Ok(())
}

fn check_unavailable_feedback_cleanup(packages: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = runtime(packages, "plex_paper_helium", &[], None)?;
    assert_success(&runtime.invoke(schedule_event("A", "G", json!({})))?);
    let event = json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "helium-terminal",
            "records": [{
                "subject": {"kind": "request", "value": "A"},
                "outcome": "completed",
                "facts": {"prefiller": false}
            }]
        },
        "cleanup": {
            "requests": [{"request_id": "A", "status": "completed"}],
            "groups": []
        }
    });
    let outcome = runtime.invoke(event.clone())?;
    assert_eq!(outcome["status"], "unavailable");
    assert!(
        runtime
            .backend()
            .read_request(&RequestId::from("A"))
            .is_err()
    );
    assert_eq!(runtime.invoke(event)?, outcome);

    assert_success(&runtime.invoke(schedule_event("B", "H", json!({})))?);
    let cancelled = runtime.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "helium-cancelled",
            "records": [{
                "subject": {"kind": "request", "value": "B"},
                "outcome": "cancelled",
                "facts": {"initiator": "host"}
            }]
        },
        "cleanup": {
            "requests": [{"request_id": "B", "status": "cancelled"}],
            "groups": []
        }
    }))?;
    assert_eq!(cancelled["status"], "unavailable");
    assert!(
        runtime
            .backend()
            .read_request(&RequestId::from("B"))
            .is_err()
    );
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

    let smetric = runtime(packages, "plex_paper_smetric", &[], None)?;
    let outcome = smetric.invoke(route_event(
        "S",
        "GS",
        "smetric-load-route",
        json!({"overload_ppm": 1_500_000, "hit_ratio_ppm": 500_000}),
    ))?;
    assert_success(&outcome);
    assert!(outcome["actions"].as_array().is_some_and(Vec::is_empty));

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
                    "facts": {"prefiller": false}
                },
                {
                    "target_id": "Y",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {"prefiller": true}
                }
            ],
            "feasible_edges": [
                {
                    "request_index": 0,
                    "target_index": 0,
                    "demand": [],
                    "facts": {"active_kv_bytes": 10}
                },
                {
                    "request_index": 0,
                    "target_index": 1,
                    "demand": [],
                    "facts": {"active_kv_bytes": 1}
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
                {
                    "request": request_ref("A", "GA"),
                    "facts": {
                        "predicted_output_tokens": 10,
                        "cost_budget": 100,
                        "quality_weight_ppm": 1000000,
                        "cost_weight_ppm": 0,
                        "latency_weight_ppm": 0
                    }
                },
                {
                    "request": request_ref("B", "GB"),
                    "facts": {
                        "predicted_output_tokens": 100,
                        "cost_budget": 100,
                        "quality_weight_ppm": 1000000,
                        "cost_weight_ppm": 0,
                        "latency_weight_ppm": 0
                    }
                }
            ],
            "targets": [
                {
                    "target_id": "X",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {"queued_tokens": 0}
                },
                {
                    "target_id": "Y",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {"queued_tokens": 0}
                }
            ],
            "feasible_edges": [
                {
                    "request_index": 0,
                    "target_index": 0,
                    "demand": [],
                    "facts": {
                        "quality_ppm": 1000,
                        "cost": 1,
                        "latency_ms": 1,
                        "decode_ms_per_token": 1
                    }
                },
                {
                    "request_index": 0,
                    "target_index": 1,
                    "demand": [],
                    "facts": {
                        "quality_ppm": 900,
                        "cost": 1,
                        "latency_ms": 1,
                        "decode_ms_per_token": 1
                    }
                },
                {
                    "request_index": 1,
                    "target_index": 0,
                    "demand": [],
                    "facts": {
                        "quality_ppm": 1000,
                        "cost": 1,
                        "latency_ms": 1,
                        "decode_ms_per_token": 1
                    }
                },
                {
                    "request_index": 1,
                    "target_index": 1,
                    "demand": [],
                    "facts": {
                        "quality_ppm": 0,
                        "cost": 1,
                        "latency_ms": 1,
                        "decode_ms_per_token": 1
                    }
                }
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
            "current_queue_ms": 10,
            "current_execution_ms": 10,
            "downstream_queue_ms": 10,
            "downstream_execution_ms": 10,
            "downstream_batch_wait_p10_ms": 10,
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
                "facts": {
                    "client_id": "client-a",
                    "input_tokens": 0,
                    "output_tokens": 100,
                    "output_weight": 1
                }
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
        json!([0])
    );

    let fairserve = runtime(packages, "plex_paper_fairserve", &[], None)?;
    assert_success(&fairserve.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "fairserve-charge",
            "records": [
                {
                    "subject": {"kind": "request", "value": "A"},
                    "outcome": "progress",
                    "facts": {"input_tokens": 100}
                },
                {
                    "subject": {"kind": "request", "value": "A"},
                    "outcome": "completed",
                    "facts": {
                        "user_id": "client-a",
                        "application_id": "app",
                        "stage_id": "stage",
                        "expected_input_tokens": 100
                    }
                }
            ]
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
                        "user_id": "client-a",
                        "application_id": "app"
                    }
                },
                {
                    "request": request_ref("B", "G2"),
                    "demand": [],
                    "facts": {
                        "user_id": "client-b",
                        "application_id": "app"
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
                        "size_bytes": 100,
                        "beneficiaries": [],
                        "beneficiary_count": 0,
                        "facts": {"leaf": true, "frequency": 1, "recompute_cost": 0, "age": 100}
                    },
                    "reclaimable": true
                },
                {
                    "object": {
                        "object_id": "leaf-b",
                        "size_bytes": 1,
                        "beneficiaries": [],
                        "beneficiary_count": 0,
                        "facts": {"leaf": true, "frequency": 1, "recompute_cost": 50, "age": 0}
                    },
                    "reclaimable": true
                }
            ],
            "prospective": [],
            "capacity": {"max_bytes": 100, "fixed_bytes": 0, "facts": {}},
            "episode": {"episode_id": "rag-episode", "iteration": 0, "max_iterations": 2}
        }
    }))?;
    assert_success(&reclaimed);
    assert_eq!(reclaimed["plan"]["plan"]["reclaim"], json!([1]));
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
        let cases: Value = serde_json::from_slice(&std::fs::read(root.join("cases/basic.json"))?)?;
        let expected: Value =
            serde_json::from_slice(&std::fs::read(root.join("expected/basic.json"))?)?;
        if cases["schema"] == "plex-paper-oracle@1" {
            assert_eq!(expected["schema"], "plex-paper-oracle@1");
            let case_ids = cases["cases"]
                .as_array()
                .ok_or("paper oracle cases must be an array")?
                .iter()
                .map(|case| {
                    case["id"]
                        .as_str()
                        .ok_or("paper oracle case must have a string id")
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            let expected_ids = expected["expectations"]
                .as_array()
                .ok_or("paper oracle expectations must be an array")?
                .iter()
                .map(|expectation| {
                    expectation["case_id"]
                        .as_str()
                        .ok_or("paper oracle expectation must have a string case_id")
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            assert_eq!(case_ids, expected_ids);
        }
        assert_eq!(metadata["id"], id);
        assert_eq!(metadata["validation_status"], "passing");
        assert!(matches!(
            metadata["evidence_level"].as_str(),
            Some("decision-trace-parity-with-deferred-mechanics")
                | Some("policy-kernel-reproduction")
                | Some("inspired-adaptation")
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
    assert_eq!(outcome["failure"]["kind"], "deadline-exceeded");

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

fn active_lifecycle(requests: &[(&str, &str)]) -> Vec<Value> {
    let groups = requests
        .iter()
        .map(|(_, group_id)| *group_id)
        .collect::<BTreeSet<_>>();
    let mut events = groups
        .into_iter()
        .map(|group_id| {
            json!({
                "event": "create-group",
                "group_id": group_id,
                "principal_id": "tenant",
                "limits": {
                    "max_members": requests.len().max(1),
                    "max_scratch_bytes": 65536
                },
                "facts": {}
            })
        })
        .collect::<Vec<_>>();
    for (request_id, group_id) in requests {
        events.extend([
            json!({
                "event": "create-request",
                "request_id": request_id,
                "principal_id": "tenant",
                "group_id": group_id,
                "fields": {},
                "facts": {}
            }),
            json!({"event": "admit-request", "request_id": request_id}),
            json!({"event": "activate-request", "request_id": request_id}),
        ]);
    }
    events
}

fn pending_lifecycle(requests: &[(&str, &str)]) -> Vec<Value> {
    let groups = requests
        .iter()
        .map(|(_, group_id)| *group_id)
        .collect::<BTreeSet<_>>();
    let mut events = groups
        .into_iter()
        .map(|group_id| {
            json!({
                "event": "create-group",
                "group_id": group_id,
                "principal_id": "tenant",
                "limits": {
                    "max_members": requests.len().max(1),
                    "max_scratch_bytes": 65536
                },
                "facts": {}
            })
        })
        .collect::<Vec<_>>();
    for (request_id, group_id) in requests {
        events.push(json!({
            "event": "create-request",
            "request_id": request_id,
            "principal_id": "tenant",
            "group_id": group_id,
            "fields": {},
            "facts": {}
        }));
    }
    events
}

fn admit_candidate(request_id: &str, group_id: &str, facts: Value) -> Value {
    json!({
        "request": request_ref(request_id, group_id),
        "demand": [],
        "facts": facts
    })
}

fn admit_many(
    opportunity_id: &str,
    candidates: Vec<Value>,
    max_accepted: u32,
    lifecycle: Vec<Value>,
) -> Value {
    let mut event = json!({
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "arrival",
            "candidates": candidates,
            "capacity": {
                "max_accepted": max_accepted,
                "limits": [],
                "facts": {}
            }
        }
    });
    if !lifecycle.is_empty() {
        event["lifecycle"] = json!(lifecycle);
    }
    event
}

fn route_candidate(request_id: &str, group_id: &str, facts: Value) -> Value {
    json!({
        "request": request_ref(request_id, group_id),
        "facts": facts
    })
}

fn route_target(target_id: &str, max_assignments: u32, facts: Value) -> Value {
    json!({
        "target_id": target_id,
        "max_assignments": max_assignments,
        "capacity": [],
        "revision": 1,
        "facts": facts
    })
}

fn route_edge(request_index: u32, target_index: u32, facts: Value) -> Value {
    json!({
        "request_index": request_index,
        "target_index": target_index,
        "demand": [],
        "facts": facts
    })
}

fn route_many(
    opportunity_id: &str,
    requests: Vec<Value>,
    targets: Vec<Value>,
    feasible_edges: Vec<Value>,
    lifecycle: Vec<Value>,
) -> Value {
    let mut event = json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "admission",
            "requests": requests,
            "targets": targets,
            "feasible_edges": feasible_edges
        }
    });
    if !lifecycle.is_empty() {
        event["lifecycle"] = json!(lifecycle);
    }
    event
}

fn schedule_candidate(
    request_id: &str,
    group_id: &str,
    max_token_budget: u32,
    facts: Value,
) -> Value {
    json!({
        "request": request_ref(request_id, group_id),
        "max_token_budget": max_token_budget,
        "facts": facts
    })
}

fn schedule_many(
    opportunity_id: &str,
    runnable: Vec<Value>,
    max_selections: u32,
    max_total_tokens: u64,
    lifecycle: Vec<Value>,
) -> Value {
    schedule_many_limits(
        opportunity_id,
        runnable,
        max_selections,
        max_selections,
        max_total_tokens,
        lifecycle,
    )
}

fn schedule_many_limits(
    opportunity_id: &str,
    runnable: Vec<Value>,
    max_selections: u32,
    max_requests: u32,
    max_total_tokens: u64,
    lifecycle: Vec<Value>,
) -> Value {
    let mut event = json!({
        "api_version": "pie.plex.engine@2",
        "operation": "schedule",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "capacity-changed",
            "runnable": runnable,
            "capacity": {
                "max_selections": max_selections,
                "max_requests": max_requests,
                "max_total_tokens": max_total_tokens,
                "facts": {}
            }
        }
    });
    if !lifecycle.is_empty() {
        event["lifecycle"] = json!(lifecycle);
    }
    event
}

fn feedback_records(delivery_id: &str, records: Vec<Value>, lifecycle: Vec<Value>) -> Value {
    let mut event = json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": delivery_id,
            "records": records
        }
    });
    if !lifecycle.is_empty() {
        event["lifecycle"] = json!(lifecycle);
    }
    event
}

fn request_progress(request_id: &str, facts: Value) -> Value {
    json!({
        "subject": {"kind": "request", "value": request_id},
        "outcome": "progress",
        "facts": facts
    })
}

fn request_completed(request_id: &str, facts: Value) -> Value {
    json!({
        "subject": {"kind": "request", "value": request_id},
        "outcome": "completed",
        "facts": facts
    })
}

fn schedule_progress(
    opportunity_id: &str,
    selection_index: u32,
    status: &str,
    scheduled_tokens: u64,
) -> Value {
    json!({
        "subject": {
            "kind": "schedule-selection",
            "value": {
                "opportunity_id": opportunity_id,
                "selection_index": selection_index
            }
        },
        "outcome": "progress",
        "facts": {
            "status": status,
            "scheduled_tokens": scheduled_tokens
        }
    })
}

fn route_progress(opportunity_id: &str, request_index: u32, status: &str) -> Value {
    json!({
        "subject": {
            "kind": "route-assignment",
            "value": {
                "opportunity_id": opportunity_id,
                "request_index": request_index
            }
        },
        "outcome": "progress",
        "facts": {"status": status}
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
