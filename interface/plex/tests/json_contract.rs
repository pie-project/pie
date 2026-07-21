use pie_plex::{
    AdmissionDecision, rank_route, select_evictions, select_schedule, validate_admit,
    validate_state_envelope,
};
use serde_json::json;

#[test]
fn all_json_operation_results_validate() {
    validate_state_envelope(&json!({
        "global": {"facts": {}, "fields": {}, "scratch": {}},
        "requests": {
            "L": {
                "facts": {"logical_request_id": "L", "generation_id": 0},
                "fields": {"body": {}, "metadata": {}},
                "scratch": {}
            }
        }
    }))
    .unwrap();
    assert_eq!(
        validate_admit(&json!({"decision": "accept"})).unwrap(),
        AdmissionDecision::Accept
    );
    assert_eq!(
        rank_route(&json!({"scores": [2.0, 1.0]}), 2).unwrap(),
        vec![0, 1]
    );
    assert!(
        select_schedule(
            &json!({
                "runnable": [{"max_token_budget": 4}],
                "capacity": {
                    "max_selected": 1,
                    "max_total_tokens": 4,
                    "max_token_budget": 4
                },
                "context": {"capabilities": {"token_budget": true}}
            }),
            &json!({"decisions": [{"score": 1.0, "token_budget": 4}]})
        )
        .is_ok()
    );
    assert!(
        select_evictions(
            &json!({"bytes_needed": 1, "resident": [{"size_bytes": 1}]}),
            &json!({"scores": [0.0]})
        )
        .is_ok()
    );
}
