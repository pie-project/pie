use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use pie_plex::v0_6::{
    MechanicKind, Operation, RouteInvocation, RouteOutput, STANDARD_MECHANICS,
    validate_policy_state, validate_route_context, validate_route_plan, validate_state_update,
};
use serde_json::Value;

const SCHEMAS: &[&str] = &[
    "schema/0.6/document.schema.json",
    "schema/0.6/manifest.schema.json",
    "schema/0.6/policy-state.schema.json",
    "schema/0.6/state-update.schema.json",
    "schema/0.6/replay-fixture.schema.json",
    "schema/0.6/mechanics-registry.schema.json",
    "schema/0.6/actions/action-feedback.schema.json",
    "schema/0.6/actions/request-cancel.schema.json",
    "schema/0.6/actions/group-cancel.schema.json",
    "schema/0.6/actions/cache-prefetch.schema.json",
    "schema/0.6/actions/cache-swap.schema.json",
    "schema/0.6/actions/request-rebalance.schema.json",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_json(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn check_local_refs(value: &Value, base: &Path) {
    match value {
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                if !reference.starts_with('#') && !reference.contains("://") {
                    let path = base.join(reference);
                    assert!(
                        path.is_file(),
                        "missing schema reference {}",
                        path.display()
                    );
                }
            }
            for child in object.values() {
                check_local_refs(child, base);
            }
        }
        Value::Array(values) => {
            for child in values {
                check_local_refs(child, base);
            }
        }
        _ => {}
    }
}

fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::Admit => "admit",
        Operation::Route => "route",
        Operation::Schedule => "schedule",
        Operation::Cache => "cache",
        Operation::Feedback => "feedback",
    }
}

#[test]
fn schemas_and_registry_are_parseable_and_cross_referenced() {
    let root = root();
    for relative in SCHEMAS {
        let path = root.join(relative);
        let schema = read_json(&path);
        assert_eq!(
            schema.get("$schema").and_then(Value::as_str),
            Some("https://json-schema.org/draft/2020-12/schema")
        );
        check_local_refs(&schema, path.parent().unwrap());
    }

    let registry_path = root.join("registry/0.6/standard-mechanics.json");
    let registry = read_json(&registry_path);
    let registry_schema = registry["$schema"].as_str().unwrap();
    assert!(
        registry_path
            .parent()
            .unwrap()
            .join(registry_schema)
            .is_file()
    );
    check_local_refs(&registry, registry_path.parent().unwrap());
    let entries = registry["mechanics"].as_array().unwrap();
    let indexed = entries
        .iter()
        .map(|entry| (entry["id"].as_str().unwrap(), entry))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(indexed.len(), STANDARD_MECHANICS.len());

    for mechanic in STANDARD_MECHANICS {
        let entry = indexed
            .get(mechanic.id)
            .unwrap_or_else(|| panic!("registry is missing {}", mechanic.id));
        let expected_kind = match mechanic.kind {
            MechanicKind::Guarantee => "guarantee",
            MechanicKind::Action => "action",
        };
        assert_eq!(entry["kind"].as_str(), Some(expected_kind));
        assert_eq!(entry["method"].as_str(), mechanic.method);
        assert_eq!(entry["request_schema"].as_str(), mechanic.request_schema);
        assert_eq!(entry["feedback_schema"].as_str(), mechanic.feedback_schema);
        assert_eq!(
            entry["operations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|operation| operation.as_str().unwrap())
                .collect::<Vec<_>>(),
            mechanic
                .operations
                .iter()
                .copied()
                .map(operation_name)
                .collect::<Vec<_>>()
        );
        for field in ["request_schema", "feedback_schema"] {
            if let Some(relative) = entry[field].as_str() {
                let path = registry_path.parent().unwrap().join(relative);
                assert!(path.is_file(), "missing registry schema {}", path.display());
                read_json(&path);
            }
        }
    }
}

#[test]
fn joint_route_replay_example_matches_the_rust_contract() {
    let fixture = read_json(&root().join("schema/0.6/examples/joint-route.json"));
    let invocation: RouteInvocation =
        serde_json::from_value(fixture["invocation"].clone()).unwrap();
    let output: RouteOutput =
        serde_json::from_value(fixture["expected"]["raw_guest_output"].clone()).unwrap();

    validate_policy_state(&invocation.state).unwrap();
    validate_route_context(&invocation.state, &invocation.context).unwrap();
    validate_route_plan(&invocation.context, &output.plan).unwrap();
    validate_state_update(&invocation.state, &output.state_update).unwrap();
}
