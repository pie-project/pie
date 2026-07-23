use std::path::PathBuf;

use pie_policy::{HostSupportV0_6, PlexRuntimeV0_6};
use serde_json::{Value, json};

const MECHANICS: [&str; 6] = [
    "request.cancel@1",
    "group.cancel@1",
    "cache.prefetch@1",
    "cache.swap@1",
    "request.rebalance@1",
    "schedule.atomic-enqueue@1",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let package = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: check_mechanics_v0_6 <policy.plexpkg>")?,
    );
    let package = std::fs::read(package)?;
    let runtime = PlexRuntimeV0_6::from_package_bytes(
        &package,
        None,
        HostSupportV0_6::with_standard_ids(MECHANICS.map(str::to_owned))?,
    )?;

    assert_success(&runtime.invoke(admit_event("A", "G"))?);
    let routed = runtime.invoke(route_event("A", "G", "route-1", true, false, true))?;
    assert_success(&routed);
    assert_eq!(routed["actions"].as_array().unwrap().len(), 1);
    assert_eq!(routed["actions"][0]["method"], "pie.request.rebalance@1");

    let scheduled = runtime.invoke(schedule_event("A", "G", "schedule-1", true, true))?;
    assert_success(&scheduled);
    assert_eq!(scheduled["actions"].as_array().unwrap().len(), 2);
    assert_eq!(scheduled["actions"][0]["method"], "pie.request.cancel@1");
    assert_eq!(scheduled["actions"][1]["method"], "pie.group.cancel@1");

    let cached = runtime.invoke(cache_event("cache-1"))?;
    assert_success(&cached);
    assert_eq!(cached["actions"].as_array().unwrap().len(), 2);
    assert_eq!(cached["actions"][0]["method"], "pie.cache.prefetch@1");
    assert_eq!(cached["actions"][1]["method"], "pie.cache.swap@1");

    let invalid = runtime.invoke(route_event("A", "G", "route-invalid", false, true, false))?;
    assert_eq!(invalid["status"], "fallback");
    assert_eq!(invalid["failure"]["kind"], "action-validation");

    let feedback = feedback_event();
    let outcome = runtime.invoke(feedback.clone())?;
    assert_success(&outcome);
    let duplicate = runtime.invoke(feedback)?;
    assert_success(&duplicate);
    assert_eq!(duplicate["duplicate_feedback"], true);
    assert_eq!(duplicate["actions"], json!([]));
    assert_eq!(runtime.metrics().commits, 5);

    let unknown = runtime.invoke(json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "unknown-action",
            "records": [{
                "subject": {"kind": "action", "value": 9},
                "outcome": "action-succeeded",
                "facts": {
                    "opportunity_id": "missing",
                    "method": "pie.request.cancel@1",
                    "idempotency_key": "missing",
                    "status": "succeeded"
                }
            }]
        }
    }));
    assert!(unknown.is_err());

    let unsupported =
        PlexRuntimeV0_6::from_package_bytes(&package, None, HostSupportV0_6::default())?;
    assert_success(&unsupported.invoke(admit_event("B", "H"))?);
    assert_success(&unsupported.invoke(route_event("B", "H", "route-B", false, false, true))?);
    let unsupported_action =
        unsupported.invoke(schedule_event("B", "H", "schedule-B", true, false))?;
    assert_eq!(unsupported_action["status"], "fallback");
    assert_eq!(
        unsupported_action["failure"]["kind"],
        "unsupported-mechanic"
    );

    println!("PLEX v0.6 optional mechanics passed");
    Ok(())
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

fn admit_event(request_id: &str, group_id: &str) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta(&format!("admit-{request_id}")),
            "cause": "arrival",
            "candidates": [{
                "request": request_ref(request_id, group_id),
                "demand": [],
                "facts": {"queue_depth": 0}
            }],
            "capacity": {"max_accepted": 1, "limits": [], "facts": {}}
        },
        "lifecycle": [
            {
                "event": "create-group",
                "group_id": group_id,
                "principal_id": "tenant",
                "limits": {"max_members": 4, "max_scratch_bytes": 4096},
                "facts": {}
            },
            {
                "event": "create-request",
                "request_id": request_id,
                "principal_id": "tenant",
                "group_id": group_id,
                "fields": {},
                "facts": {}
            }
        ]
    })
}

fn route_event(
    request_id: &str,
    group_id: &str,
    opportunity_id: &str,
    rebalance: bool,
    invalid_cancel: bool,
    admit: bool,
) -> Value {
    let lifecycle = if admit {
        json!([{"event": "admit-request", "request_id": request_id}])
    } else {
        json!([])
    };
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "admission",
            "requests": [{
                "request": request_ref(request_id, group_id),
                "facts": {
                    "rebalance": rebalance,
                    "invalid_cancel": invalid_cancel
                }
            }],
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
        "lifecycle": lifecycle
    })
}

fn schedule_event(
    request_id: &str,
    group_id: &str,
    opportunity_id: &str,
    cancel_request: bool,
    cancel_group: bool,
) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "schedule",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "capacity-changed",
            "runnable": [{
                "request": request_ref(request_id, group_id),
                "max_token_budget": 4,
                "facts": {
                    "cancel_request": cancel_request,
                    "cancel_group": cancel_group
                }
            }],
            "capacity": {
                "max_selections": 1,
                "max_requests": 1,
                "max_total_tokens": 4,
                "facts": {}
            }
        },
        "lifecycle": [{"event": "activate-request", "request_id": request_id}]
    })
}

fn cache_event(opportunity_id: &str) -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "insertion",
            "resident": [],
            "prospective": [{
                "object_id": "kv-A",
                "size_bytes": 1,
                "beneficiaries": [],
                "beneficiary_count": 0,
                "facts": {"prefetch": true, "swap": true}
            }],
            "capacity": {"max_bytes": 1, "fixed_bytes": 0, "facts": {}},
            "episode": null
        }
    })
}

fn feedback_event() -> Value {
    json!({
        "api_version": "pie.plex.engine@2",
        "operation": "feedback",
        "context": {
            "delivery_id": "mechanics-feedback",
            "records": [
                action_record(
                    0,
                    "route-1",
                    "pie.request.rebalance@1",
                    "rebalance-0",
                    "action-succeeded",
                    "succeeded"
                ),
                action_record(
                    0,
                    "schedule-1",
                    "pie.request.cancel@1",
                    "cancel-0",
                    "action-succeeded",
                    "succeeded"
                ),
                action_record(
                    1,
                    "schedule-1",
                    "pie.group.cancel@1",
                    "cancel-group-0",
                    "action-succeeded",
                    "succeeded"
                ),
                action_record(
                    0,
                    "cache-1",
                    "pie.cache.prefetch@1",
                    "prefetch-kv-A",
                    "action-failed",
                    "failed"
                ),
                action_record(
                    1,
                    "cache-1",
                    "pie.cache.swap@1",
                    "swap-kv-A",
                    "action-failed",
                    "unsupported"
                ),
                {
                    "subject": {"kind": "request", "value": "A"},
                    "outcome": "cancelled",
                    "facts": {}
                },
                {
                    "subject": {"kind": "work-group", "value": "G"},
                    "outcome": "cancelled",
                    "facts": {}
                }
            ]
        },
        "cleanup": {
            "requests": [{"request_id": "A", "status": "cancelled"}],
            "groups": [{"group_id": "G", "status": "cancelled"}]
        }
    })
}

fn action_record(
    action_id: u64,
    opportunity_id: &str,
    method: &str,
    idempotency_key: &str,
    outcome: &str,
    status: &str,
) -> Value {
    json!({
        "subject": {"kind": "action", "value": action_id},
        "outcome": outcome,
        "facts": {
            "opportunity_id": opportunity_id,
            "method": method,
            "idempotency_key": idempotency_key,
            "status": status
        }
    })
}

fn assert_success(outcome: &Value) {
    assert_eq!(outcome["status"], "success", "{outcome}");
}
