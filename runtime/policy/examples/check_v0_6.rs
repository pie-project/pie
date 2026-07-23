use std::path::PathBuf;

use pie_policy::{HostSupportV0_6, PlexRuntimeV0_6};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let package = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: check_v0_6 <policy.plexpkg>")?,
    );
    let runtime = PlexRuntimeV0_6::from_package_bytes(
        &std::fs::read(package)?,
        None,
        HostSupportV0_6::default(),
    )?;

    let request = || {
        json!({
            "request_id": "A",
            "generation_id": 0,
            "group_id": "G",
            "principal_id": "tenant"
        })
    };
    let meta = |id: &str| {
        json!({
            "opportunity_id": id,
            "snapshot": {"id": "host-filled", "revision": 0},
            "attempt": 0,
            "mechanics": []
        })
    };

    let admitted = runtime.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta("admit-1"),
            "cause": "arrival",
            "candidates": [{
                "request": request(),
                "demand": [],
                "facts": {"queue_depth": 0}
            }],
            "capacity": {
                "max_accepted": 1,
                "limits": [],
                "facts": {}
            }
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
            }
        ]
    }))?;
    assert_success(&admitted);
    assert_eq!(admitted["plan"]["plan"]["decisions"][0], "accept");

    let routed = runtime.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta("route-1"),
            "cause": "admission",
            "requests": [{"request": request(), "facts": {}}],
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
                "facts": {"queue_depth": 0}
            }]
        },
        "lifecycle": [{"event": "admit-request", "request_id": "A"}]
    }))?;
    assert_success(&routed);
    assert_eq!(routed["plan"]["plan"]["assignments"][0]["target_index"], 0);

    let scheduled = runtime.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "schedule",
        "context": {
            "meta": meta("schedule-1"),
            "cause": "capacity-changed",
            "runnable": [{
                "request": request(),
                "max_token_budget": 4,
                "facts": {}
            }],
            "capacity": {
                "max_selections": 1,
                "max_requests": 1,
                "max_total_tokens": 4,
                "facts": {}
            }
        },
        "lifecycle": [{"event": "activate-request", "request_id": "A"}]
    }))?;
    assert_success(&scheduled);
    assert_eq!(
        scheduled["plan"]["plan"]["selections"][0]["token_budgets"][0],
        4
    );

    let cached = runtime.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta("cache-1"),
            "cause": "insertion",
            "resident": [],
            "prospective": [{
                "object_id": "kv-A",
                "size_bytes": 1,
                "beneficiaries": [{"kind": "request", "id": "A"}],
                "beneficiary_count": 1,
                "facts": {"cache": true}
            }],
            "capacity": {"max_bytes": 1, "fixed_bytes": 0, "facts": {}},
            "episode": null
        }
    }))?;
    assert_success(&cached);
    assert_eq!(cached["plan"]["plan"]["admissions"][0], "cache");

    let feedback_event = json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "delivery-1",
            "records": [
                {
                    "subject": {"kind": "request", "value": "A"},
                    "outcome": "completed",
                    "facts": {}
                },
                {
                    "subject": {"kind": "work-group", "value": "G"},
                    "outcome": "completed",
                    "facts": {}
                }
            ]
        },
        "cleanup": {
            "requests": [{"request_id": "A", "status": "completed"}],
            "groups": [{"group_id": "G", "status": "closed"}]
        }
    });
    let feedback = runtime.invoke(feedback_event.clone())?;
    assert_success(&feedback);
    let duplicate = runtime.invoke(feedback_event)?;
    assert_success(&duplicate);
    assert_eq!(duplicate["duplicate_feedback"], true);
    assert_eq!(duplicate["actions"], json!([]));
    assert_eq!(runtime.metrics().commits, 5);

    println!("PLEX v0.6 typed lifecycle passed");
    Ok(())
}

fn assert_success(outcome: &Value) {
    assert_eq!(outcome["status"], "success", "{outcome}");
}
