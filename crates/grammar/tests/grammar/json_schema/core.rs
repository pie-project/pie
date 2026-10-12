use crate::common::grammar_accepts as is_grammar_accept_string_g;
use ::grammar::json_schema::{JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar};

#[test]
fn core_every_case() {
    test_builtin_json_accepts_basic_types();
    test_object_min_gt_max_properties_error();
    test_object_required_exceeds_max_properties_error();
}

fn test_builtin_json_accepts_basic_types() {
    let g = builtin_json_grammar().unwrap();

    assert!(is_grammar_accept_string_g(&g, "true"));
    assert!(is_grammar_accept_string_g(&g, "false"));
    assert!(is_grammar_accept_string_g(&g, "null"));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(is_grammar_accept_string_g(&g, "-3.14"));
    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
}

fn test_object_min_gt_max_properties_error() {
    let schema = r#"{"type": "object", "minProperties": 5, "maxProperties": 2}"#;
    let opts = JsonSchemaOptions {
        strict_mode: false,
        any_whitespace: false,
    };
    let result = json_schema_to_grammar(schema, &opts);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("minProperties") && msg.contains("maxProperties"),
        "unexpected error: {}",
        msg
    );
}

fn test_object_required_exceeds_max_properties_error() {
    let schema = r#"{
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"}},
        "required": ["a", "b", "c"],
        "maxProperties": 2
    }"#;
    let opts = JsonSchemaOptions {
        any_whitespace: false,
        ..Default::default()
    };
    let result = json_schema_to_grammar(schema, &opts);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("required") && msg.contains("maxProperties"),
        "unexpected error: {}",
        msg
    );
}
