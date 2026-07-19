use crate::common::grammar_accepts;
use pie_grammar::grammar::Grammar;
use pie_grammar::json_schema::{JsonSchemaOptions, json_schema_to_ebnf, json_schema_to_grammar};

#[test]
fn typed_ebnf_rendering_roundtrips() {
    let schema = r#"{
        "type":"object",
        "properties":{
            "name":{"type":"string"},
            "score":{"type":"number","minimum":0,"maximum":10}
        },
        "required":["name"],
        "additionalProperties":false
    }"#;
    let options = JsonSchemaOptions::default();
    let direct = json_schema_to_grammar(schema, &options).unwrap();
    let value = serde_json::from_str(schema).unwrap();
    let rendered = json_schema_to_ebnf(&value, &options).unwrap();
    let reparsed = Grammar::from_ebnf(&rendered, "root").unwrap();

    for accepted in [r#"{"name":"a"}"#, r#"{"name":"a","score":9.5}"#] {
        assert!(grammar_accepts(direct.clone(), accepted));
        assert!(grammar_accepts(reparsed.clone(), accepted));
    }
    for rejected in [r#"{}"#, r#"{"name":"a","score":11}"#] {
        assert!(!grammar_accepts(direct.clone(), rejected));
        assert!(!grammar_accepts(reparsed.clone(), rejected));
    }
}

#[test]
fn root_ref_resolves_local_defs() {
    let schema = r##"{
        "$ref": "#/$defs/name",
        "$defs": {"name": {"type": "string"}}
    }"##;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar, r#""alice""#));
}

#[test]
fn bounded_number_rejects_fraction_above_maximum() {
    let schema = r#"{"type":"number","minimum":1,"maximum":2}"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar.clone(), "1"));
    assert!(grammar_accepts(grammar.clone(), "1.5"));
    assert!(grammar_accepts(grammar.clone(), "2"));
    assert!(grammar_accepts(grammar.clone(), "2.00"));
    assert!(!grammar_accepts(grammar.clone(), "0.9"));
    assert!(!grammar_accepts(grammar.clone(), "2.9"));
    assert!(!grammar_accepts(grammar, "1e1"));

    let negative_schema = r#"{"type":"number","minimum":-2,"maximum":-1}"#;
    let negative = json_schema_to_grammar(negative_schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(negative.clone(), "-2.0"));
    assert!(grammar_accepts(negative.clone(), "-1.5"));
    assert!(!grammar_accepts(negative.clone(), "-2.1"));
    assert!(!grammar_accepts(negative, "-0.9"));
}

#[test]
fn bounded_number_handles_cross_zero_and_one_sided_ranges() {
    let cross_zero = json_schema_to_grammar(
        r#"{"type":"number","minimum":-1,"maximum":2}"#,
        &JsonSchemaOptions::default(),
    )
    .unwrap();
    assert!(grammar_accepts(cross_zero.clone(), "-0.5"));
    assert!(grammar_accepts(cross_zero.clone(), "1.25"));
    assert!(!grammar_accepts(cross_zero.clone(), "-1.1"));
    assert!(!grammar_accepts(cross_zero, "2.1"));

    let lower_bounded = json_schema_to_grammar(
        r#"{"type":"number","minimum":-5}"#,
        &JsonSchemaOptions::default(),
    )
    .unwrap();
    assert!(grammar_accepts(lower_bounded.clone(), "-4.9"));
    assert!(grammar_accepts(lower_bounded.clone(), "100.25"));
    assert!(!grammar_accepts(lower_bounded, "-5.1"));

    assert!(
        json_schema_to_grammar(
            r#"{"type":"number","minimum":1.5}"#,
            &JsonSchemaOptions::default()
        )
        .is_err()
    );
    assert!(
        json_schema_to_grammar(
            r#"{"type":"number","exclusiveMinimum":1}"#,
            &JsonSchemaOptions::default()
        )
        .is_err()
    );
}

#[test]
fn lower_bounded_integer_rejects_smaller_values() {
    let schema = r#"{"type":"integer","minimum":-5}"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar.clone(), "-5"));
    assert!(grammar_accepts(grammar.clone(), "0"));
    assert!(grammar_accepts(grammar.clone(), "123456"));
    assert!(!grammar_accepts(grammar, "-6"));
}

#[test]
fn integer_bounds_combine_without_overflow() {
    let schema = r#"{
        "type":"integer",
        "minimum":0,
        "exclusiveMinimum":10,
        "maximum":100,
        "exclusiveMaximum":90
    }"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar.clone(), "11"));
    assert!(grammar_accepts(grammar.clone(), "89"));
    assert!(!grammar_accepts(grammar.clone(), "10"));
    assert!(!grammar_accepts(grammar, "90"));

    let no_lower_value = r#"{"type":"integer","exclusiveMinimum":9223372036854775807}"#;
    assert!(json_schema_to_grammar(no_lower_value, &JsonSchemaOptions::default()).is_err());
    let no_upper_value = r#"{"type":"integer","exclusiveMaximum":-9223372036854775808}"#;
    assert!(json_schema_to_grammar(no_upper_value, &JsonSchemaOptions::default()).is_err());
}

#[test]
fn optional_properties_can_start_after_an_omission() {
    let schema = r#"{
        "type":"object",
        "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},
        "additionalProperties":false
    }"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar, r#"{"b":1}"#));
}

#[test]
fn min_properties_counts_declared_optional_properties() {
    let schema = r#"{
        "type":"object",
        "properties":{"a":{"type":"integer"}},
        "minProperties":1,
        "additionalProperties":false
    }"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(!grammar_accepts(grammar, "{}"));
}

#[test]
fn pattern_length_combination_is_rejected_explicitly() {
    let schema = r#"{"type":"string","pattern":"a+","minLength":3}"#;
    assert!(json_schema_to_grammar(schema, &JsonSchemaOptions::default()).is_err());
}

#[test]
fn any_schema_rejects_raw_control_characters() {
    let grammar = json_schema_to_grammar("{}", &JsonSchemaOptions::default()).unwrap();
    assert!(!grammar_accepts(grammar, "\"\n\""));
}
