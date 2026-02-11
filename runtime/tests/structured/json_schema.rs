//! Ported from xgrammar: test_json_schema_converter.py
//!
//! Tests JSON Schema → Grammar conversion and acceptance.

use std::sync::Arc;

use pie::structured::grammar::Grammar;
use pie::structured::json_schema::{builtin_json_grammar, json_schema_to_grammar, JsonSchemaOptions};
use pie::structured::matcher::GrammarMatcher;
use pie::tokenizer::Tokenizer;

fn is_grammar_accept_string_g(grammar: &Grammar, input: &str) -> bool {
    let vocab: Vec<String> = vec!["dummy".into()];
    let tok = Arc::new(Tokenizer::from_vocab(&vocab));
    let mut m = GrammarMatcher::new(Arc::new(grammar.clone()), tok, vec![], 10);

    if input.is_empty() {
        return m.can_terminate();
    }

    if !m.accept_string(input) {
        return false;
    }
    m.can_terminate()
}

fn default_opts() -> JsonSchemaOptions {
    JsonSchemaOptions {
        any_whitespace: false,
        ..JsonSchemaOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Builtin JSON grammar
// ---------------------------------------------------------------------------

#[test]
fn test_builtin_json_accepts_basic_types() {
    let g = builtin_json_grammar().unwrap();

    assert!(is_grammar_accept_string_g(&g, "true"));
    assert!(is_grammar_accept_string_g(&g, "false"));
    assert!(is_grammar_accept_string_g(&g, "null"));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(is_grammar_accept_string_g(&g, "-3.14"));
    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
}

#[test]
fn test_builtin_json_accepts_objects() {
    let g = builtin_json_grammar().unwrap();

    assert!(is_grammar_accept_string_g(&g, "{}"));
    assert!(is_grammar_accept_string_g(&g, r#"{"key": "value"}"#));
    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"a": 1, "b": true, "c": null}"#
    ));
}

#[test]
fn test_builtin_json_accepts_arrays() {
    let g = builtin_json_grammar().unwrap();

    assert!(is_grammar_accept_string_g(&g, "[]"));
    assert!(is_grammar_accept_string_g(&g, "[1, 2, 3]"));
    assert!(is_grammar_accept_string_g(
        &g,
        r#"[1, "two", true, null]"#
    ));
}

#[test]
fn test_builtin_json_nested() {
    let g = builtin_json_grammar().unwrap();

    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"a": {"b": [1, 2]}}"#
    ));
}

#[test]
fn test_builtin_json_rejects_invalid() {
    let g = builtin_json_grammar().unwrap();

    assert!(!is_grammar_accept_string_g(&g, ""));
    assert!(!is_grammar_accept_string_g(&g, "hello"));
    assert!(!is_grammar_accept_string_g(&g, "{key: value}"));
    assert!(!is_grammar_accept_string_g(&g, "[1, 2,]"));
}

// ---------------------------------------------------------------------------
// Boolean schema
// ---------------------------------------------------------------------------

#[test]
fn test_boolean_schema() {
    let schema = r#"{"type": "boolean"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "true"));
    assert!(is_grammar_accept_string_g(&g, "false"));
    assert!(!is_grammar_accept_string_g(&g, "null"));
    assert!(!is_grammar_accept_string_g(&g, "1"));
}

// ---------------------------------------------------------------------------
// Null schema
// ---------------------------------------------------------------------------

#[test]
fn test_null_schema() {
    let schema = r#"{"type": "null"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "null"));
    assert!(!is_grammar_accept_string_g(&g, "true"));
    assert!(!is_grammar_accept_string_g(&g, "\"null\""));
}

// ---------------------------------------------------------------------------
// String schema
// ---------------------------------------------------------------------------

#[test]
fn test_string_schema_basic() {
    let schema = r#"{"type": "string"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
    assert!(is_grammar_accept_string_g(&g, r#""""#)); // empty string
    assert!(is_grammar_accept_string_g(&g, r#""with spaces""#));
    assert!(!is_grammar_accept_string_g(&g, "hello")); // unquoted
}

#[test]
fn test_string_schema_with_escapes() {
    let schema = r#"{"type": "string"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello\nworld""#));
    assert!(is_grammar_accept_string_g(&g, r#""tab\there""#));
    assert!(is_grammar_accept_string_g(&g, r#""quote\"inside""#));
    assert!(is_grammar_accept_string_g(&g, r#""backslash\\here""#));
}

#[test]
fn test_string_schema_with_length() {
    let schema = r#"{"type": "string", "minLength": 2, "maxLength": 4}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(!is_grammar_accept_string_g(&g, r#""a""#)); // too short
    assert!(is_grammar_accept_string_g(&g, r#""ab""#));
    assert!(is_grammar_accept_string_g(&g, r#""abcd""#));
    assert!(!is_grammar_accept_string_g(&g, r#""abcde""#)); // too long
}

// ---------------------------------------------------------------------------
// Integer schema
// ---------------------------------------------------------------------------

#[test]
fn test_integer_schema_basic() {
    let schema = r#"{"type": "integer"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "0"));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(is_grammar_accept_string_g(&g, "-7"));
    assert!(is_grammar_accept_string_g(&g, "100"));
    assert!(!is_grammar_accept_string_g(&g, "3.14")); // not an integer
    assert!(!is_grammar_accept_string_g(&g, "\"42\"")); // string, not int
}

#[test]
fn test_integer_schema_with_bounds() {
    let schema = r#"{"type": "integer", "minimum": 0, "maximum": 100}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "0"));
    assert!(is_grammar_accept_string_g(&g, "50"));
    assert!(is_grammar_accept_string_g(&g, "100"));
    assert!(!is_grammar_accept_string_g(&g, "-1"));
    assert!(!is_grammar_accept_string_g(&g, "101"));
}

#[test]
fn test_integer_schema_with_exclusive_bounds() {
    let schema = r#"{"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 10}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "1"));
    assert!(is_grammar_accept_string_g(&g, "5"));
    assert!(is_grammar_accept_string_g(&g, "9"));
    assert!(!is_grammar_accept_string_g(&g, "0")); // exclusive
    assert!(!is_grammar_accept_string_g(&g, "10")); // exclusive
}

#[test]
fn test_integer_negative_range() {
    let schema = r#"{"type": "integer", "minimum": -10, "maximum": -1}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "-1"));
    assert!(is_grammar_accept_string_g(&g, "-5"));
    assert!(is_grammar_accept_string_g(&g, "-10"));
    assert!(!is_grammar_accept_string_g(&g, "0"));
    assert!(!is_grammar_accept_string_g(&g, "-11"));
}

// ---------------------------------------------------------------------------
// Number schema
// ---------------------------------------------------------------------------

#[test]
fn test_number_schema_basic() {
    let schema = r#"{"type": "number"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "0"));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(is_grammar_accept_string_g(&g, "-7"));
    assert!(is_grammar_accept_string_g(&g, "3.14"));
    assert!(is_grammar_accept_string_g(&g, "-0.5"));
}

// ---------------------------------------------------------------------------
// Enum schema
// ---------------------------------------------------------------------------

#[test]
fn test_enum_schema() {
    let schema = r#"{"enum": ["red", "green", "blue"]}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""red""#));
    assert!(is_grammar_accept_string_g(&g, r#""green""#));
    assert!(is_grammar_accept_string_g(&g, r#""blue""#));
    assert!(!is_grammar_accept_string_g(&g, r#""yellow""#));
}

#[test]
fn test_enum_schema_mixed_types() {
    let schema = r#"{"enum": ["hello", 42, true, null]}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(is_grammar_accept_string_g(&g, "true"));
    assert!(is_grammar_accept_string_g(&g, "null"));
    assert!(!is_grammar_accept_string_g(&g, "false"));
}

// ---------------------------------------------------------------------------
// Const schema
// ---------------------------------------------------------------------------

#[test]
fn test_const_schema() {
    let schema = r#"{"const": 42}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(!is_grammar_accept_string_g(&g, "43"));
}

#[test]
fn test_const_string_schema() {
    let schema = r#"{"const": "hello"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
    assert!(!is_grammar_accept_string_g(&g, r#""world""#));
}

// ---------------------------------------------------------------------------
// Array schema
// ---------------------------------------------------------------------------

#[test]
fn test_array_schema_homogeneous() {
    let schema = r#"{"type": "array", "items": {"type": "integer"}}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "[]"));
    assert!(is_grammar_accept_string_g(&g, "[1]"));
    assert!(is_grammar_accept_string_g(&g, "[1,2,3]"));
}

#[test]
fn test_array_schema_rejects_wrong_type() {
    let schema = r#"{"type": "array", "items": {"type": "integer"}}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    // Strings not allowed when items must be integers
    assert!(!is_grammar_accept_string_g(&g, r#"["hello"]"#));
}

// ---------------------------------------------------------------------------
// Object schema
// ---------------------------------------------------------------------------

#[test]
fn test_object_schema_required_only() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"name":"Alice","age":30}"#
    ));
}

#[test]
fn test_object_schema_with_boolean_field() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "active": {"type": "boolean"}
        },
        "required": ["active"]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{"active":true}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"active":false}"#));
}

// ---------------------------------------------------------------------------
// anyOf / oneOf
// ---------------------------------------------------------------------------

#[test]
fn test_any_of_schema() {
    let schema = r#"{"anyOf": [{"type": "string"}, {"type": "integer"}]}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(!is_grammar_accept_string_g(&g, "true"));
}

#[test]
fn test_one_of_schema() {
    let schema = r#"{"oneOf": [{"type": "boolean"}, {"type": "null"}]}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, "true"));
    assert!(is_grammar_accept_string_g(&g, "false"));
    assert!(is_grammar_accept_string_g(&g, "null"));
    assert!(!is_grammar_accept_string_g(&g, "42"));
}

// ---------------------------------------------------------------------------
// Nested object
// ---------------------------------------------------------------------------

#[test]
fn test_nested_object_schema() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"}
                },
                "required": ["street", "city"]
            }
        },
        "required": ["address"]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"address":{"street":"123 Main","city":"NYC"}}"#
    ));
}

// ---------------------------------------------------------------------------
// Type inference
// ---------------------------------------------------------------------------

#[test]
fn test_schema_type_inference_object() {
    // No explicit "type", but has "properties" → should be inferred as object
    let schema = r#"{
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{"name":"test"}"#));
}

// ---------------------------------------------------------------------------
// Array schema with min/max items
// ---------------------------------------------------------------------------

#[test]
fn test_array_min_max_items() {
    let schema = r#"{"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(!is_grammar_accept_string_g(&g, "[]"));
    assert!(!is_grammar_accept_string_g(&g, "[1]"));
    assert!(is_grammar_accept_string_g(&g, "[1,2]"));
    assert!(is_grammar_accept_string_g(&g, "[1,2,3]"));
    assert!(!is_grammar_accept_string_g(&g, "[1,2,3,4]"));
}

#[test]
fn test_array_min_items_only() {
    let schema = r#"{"type": "array", "items": {"type": "integer"}, "minItems": 1}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(!is_grammar_accept_string_g(&g, "[]"));
    assert!(is_grammar_accept_string_g(&g, "[1]"));
    assert!(is_grammar_accept_string_g(&g, "[1,2,3]"));
}

// ---------------------------------------------------------------------------
// $ref / $defs support
// ---------------------------------------------------------------------------

#[test]
fn test_ref_defs_schema() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "value": {"$ref": "#/$defs/nested"}
        },
        "required": ["value"],
        "$defs": {
            "nested": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }"##;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"value":{"name":"John","age":30}}"#
    ));
}

// ---------------------------------------------------------------------------
// Multiple types (type as array)
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_types_schema() {
    let schema = r#"{"type": ["string", "integer"]}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
    assert!(is_grammar_accept_string_g(&g, "42"));
    assert!(!is_grammar_accept_string_g(&g, "true"));
}

// ---------------------------------------------------------------------------
// Complex object with mixed required/optional
// ---------------------------------------------------------------------------

#[test]
fn test_object_mixed_required_optional() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name"]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    // Only required field
    assert!(is_grammar_accept_string_g(&g, r#"{"name":"Alice"}"#));
}

// ---------------------------------------------------------------------------
// Complex nested schema (from test_basic in xgrammar)
// ---------------------------------------------------------------------------

#[test]
fn test_complex_nested_schema() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "integer_field": {"type": "integer"},
            "boolean_field": {"type": "boolean"},
            "array_field": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["integer_field", "boolean_field", "array_field"]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"integer_field":42,"boolean_field":true,"array_field":["foo","bar"]}"#
    ));
}

// ---------------------------------------------------------------------------
// Union schema (from test_union in xgrammar)
// ---------------------------------------------------------------------------

#[test]
fn test_union_schema_objects() {
    let schema = r#"{
        "anyOf": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "color": {"type": "string"}},
                "required": ["name", "color"]
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "breed": {"type": "string"}},
                "required": ["name", "breed"]
            }
        ]
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"name":"kitty","color":"black"}"#
    ));
    assert!(is_grammar_accept_string_g(
        &g,
        r#"{"name":"doggy","breed":"bulldog"}"#
    ));
}

// ---------------------------------------------------------------------------
// String with pattern (from test_json_schema_converter)
// ---------------------------------------------------------------------------

#[test]
fn test_string_with_pattern() {
    let schema = r#"{"type": "string", "pattern": "[a-z]+"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""hello""#));
    assert!(!is_grammar_accept_string_g(&g, r#""HELLO""#));
    assert!(!is_grammar_accept_string_g(&g, r#""123""#));
}

// ---------------------------------------------------------------------------
// String format keyword (Item 5)
// ---------------------------------------------------------------------------

#[test]
fn test_string_format_date() {
    let schema = r#"{"type": "string", "format": "date"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""2024-01-15""#));
    assert!(is_grammar_accept_string_g(&g, r#""0000-01-01""#));
    assert!(is_grammar_accept_string_g(&g, r#""9999-12-31""#));
    assert!(is_grammar_accept_string_g(&g, r#""2024-02-29""#));
    // Invalid dates
    assert!(!is_grammar_accept_string_g(&g, r#""2024-13-01""#)); // month 13
    assert!(!is_grammar_accept_string_g(&g, r#""2024-00-01""#)); // month 00
    assert!(!is_grammar_accept_string_g(&g, r#""2024-01-32""#)); // day 32
    assert!(!is_grammar_accept_string_g(&g, r#""2024-01-00""#)); // day 00
    assert!(!is_grammar_accept_string_g(&g, r#""24-01-01""#));   // 2-digit year
    assert!(!is_grammar_accept_string_g(&g, r#""not-a-date""#));
}

#[test]
fn test_string_format_time() {
    let schema = r#"{"type": "string", "format": "time"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""00:00:00Z""#));
    assert!(is_grammar_accept_string_g(&g, r#""23:59:59Z""#));
    assert!(is_grammar_accept_string_g(&g, r#""14:30:00Z""#));
    assert!(is_grammar_accept_string_g(&g, r#""08:15:30+05:30""#));
    assert!(is_grammar_accept_string_g(&g, r#""22:59:59-07:00""#));
    assert!(is_grammar_accept_string_g(&g, r#""12:34:56.789Z""#));
    assert!(is_grammar_accept_string_g(&g, r#""23:59:60Z""#)); // leap second
    // Invalid times
    assert!(!is_grammar_accept_string_g(&g, r#""24:00:00Z""#)); // hour 24
    assert!(!is_grammar_accept_string_g(&g, r#""00:60:00Z""#)); // minute 60
    assert!(!is_grammar_accept_string_g(&g, r#""12:00:00""#));  // no timezone
}

#[test]
fn test_string_format_datetime() {
    let schema = r#"{"type": "string", "format": "date-time"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""2024-01-15T14:30:00Z""#));
    assert!(is_grammar_accept_string_g(&g, r#""2019-11-30T08:15:27+05:30""#));
    assert!(is_grammar_accept_string_g(&g, r#""2021-07-04T00:00:00.123456Z""#));
    // Invalid
    assert!(!is_grammar_accept_string_g(&g, r#""2024-13-01T00:00:00Z""#)); // month 13
    assert!(!is_grammar_accept_string_g(&g, r#""2024-01-15T24:00:00Z""#)); // hour 24
}

#[test]
fn test_string_format_email() {
    let schema = r#"{"type": "string", "format": "email"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""user@example.com""#));
    assert!(is_grammar_accept_string_g(&g, r#""john.doe@company.org""#));
    assert!(is_grammar_accept_string_g(&g, r#""x@y.z""#));
    assert!(is_grammar_accept_string_g(&g, r#""test+tag@gmail.com""#));
    // Invalid
    assert!(!is_grammar_accept_string_g(&g, r#""invalid.email""#));     // no @
    assert!(!is_grammar_accept_string_g(&g, r#""@example.com""#));      // no local part
}

#[test]
fn test_string_format_uuid() {
    let schema = r#"{"type": "string", "format": "uuid"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""550e8400-e29b-41d4-a716-446655440000""#));
    assert!(is_grammar_accept_string_g(&g, r#""123e4567-e89b-12d3-a456-426614174000""#));
    // Invalid
    assert!(!is_grammar_accept_string_g(&g, r#""not-a-uuid""#));
    assert!(!is_grammar_accept_string_g(&g, r#""550e8400e29b41d4a716446655440000""#)); // no dashes
}

#[test]
fn test_string_format_unknown_ignored() {
    // Unknown format should fall through to default string handling
    let schema = r#"{"type": "string", "format": "custom-format"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    // Should accept any string since unknown format is ignored
    assert!(is_grammar_accept_string_g(&g, r#""anything goes""#));
}

#[test]
fn test_string_format_inferred_type() {
    // Schema without explicit type but with format should be inferred as string
    let schema = r#"{"format": "date"}"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#""2024-01-15""#));
    assert!(!is_grammar_accept_string_g(&g, r#""not-a-date""#));
}

// ---------------------------------------------------------------------------
// prefixItems (Item 7)
// ---------------------------------------------------------------------------

#[test]
fn test_array_prefix_items_tuple_exact() {
    // Pure tuple: exactly 2 elements with specific types
    let schema = r#"{
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "items": false
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"["hello",42]"#));
    assert!(!is_grammar_accept_string_g(&g, r#"["hello"]"#));           // too few
    assert!(!is_grammar_accept_string_g(&g, r#"["hello",42,"extra"]"#)); // too many
}

#[test]
fn test_array_prefix_items_with_additional() {
    // Tuple with additional items of specified type
    let schema = r#"{
        "type": "array",
        "prefixItems": [{"type": "string"}],
        "items": {"type": "integer"}
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"["hello"]"#));
    assert!(is_grammar_accept_string_g(&g, r#"["hello",1]"#));
    assert!(is_grammar_accept_string_g(&g, r#"["hello",1,2,3]"#));
}

#[test]
fn test_array_prefix_items_with_min_max() {
    // Tuple with additional and minItems/maxItems
    let schema = r#"{
        "type": "array",
        "prefixItems": [{"type": "string"}],
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 4
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(!is_grammar_accept_string_g(&g, r#"["hello"]"#));            // below minItems
    assert!(is_grammar_accept_string_g(&g, r#"["hello",1]"#));           // 2 items
    assert!(is_grammar_accept_string_g(&g, r#"["hello",1,2]"#));         // 3 items
    assert!(is_grammar_accept_string_g(&g, r#"["hello",1,2,3]"#));       // 4 items (max)
    assert!(!is_grammar_accept_string_g(&g, r#"["hello",1,2,3,4]"#));    // above maxItems
}

#[test]
fn test_array_prefix_items_no_additional() {
    // prefixItems with items: false — strict tuple
    let schema = r#"{
        "type": "array",
        "prefixItems": [
            {"type": "boolean"},
            {"type": "string"},
            {"type": "integer"}
        ],
        "items": false
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"[true,"hello",42]"#));
    assert!(!is_grammar_accept_string_g(&g, r#"[true,"hello"]"#));       // too few
    assert!(!is_grammar_accept_string_g(&g, r#"[true,"hello",42,1]"#));  // too many
}

#[test]
fn test_array_prefix_items_nested_objects() {
    let schema = r#"{
        "type": "array",
        "prefixItems": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            },
            {"type": "integer"}
        ],
        "items": false
    }"#;
    let g = json_schema_to_grammar(schema, &default_opts()).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"[{"name":"Alice"},42]"#));
    assert!(!is_grammar_accept_string_g(&g, r#"[42,{"name":"Alice"}]"#)); // wrong order
}

// ---------------------------------------------------------------------------
// additionalProperties with schema (Item 6)
// ---------------------------------------------------------------------------

#[test]
fn test_object_additional_properties_string_schema() {
    let schema = r#"{"type": "object", "additionalProperties": {"type": "string"}}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d"}"#));
    // Integer value should be rejected (only strings allowed)
    assert!(!is_grammar_accept_string_g(&g, r#"{"name":42}"#));
}

#[test]
fn test_object_additional_properties_integer_schema() {
    let schema = r#"{"type": "object", "additionalProperties": {"type": "integer"}}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"count":42}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"name":"John"}"#)); // string not allowed
}

#[test]
fn test_object_required_props_with_additional_schema() {
    let schema = r#"{
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": {"type": "integer"}
    }"#;
    let opts = JsonSchemaOptions { any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John","age":30}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John","age":30,"score":100}"#));
}

#[test]
fn test_object_additional_properties_boolean_true() {
    // additionalProperties: true should allow any value
    let schema = r#"{"type": "object", "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"count":42}"#));
}

#[test]
fn test_object_additional_properties_boolean_false() {
    // additionalProperties: false should allow no extra properties
    let schema = r#"{
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": false
    }"#;
    let opts = JsonSchemaOptions { any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John"}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"name":"John","extra":1}"#));
}

// ---------------------------------------------------------------------------
// minProperties / maxProperties (Item 8)
// ---------------------------------------------------------------------------

#[test]
fn test_object_max_properties_zero() {
    // maxProperties: 0 → only empty object allowed
    let schema = r#"{"type": "object", "maxProperties": 0, "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"a":"b"}"#));
}

#[test]
fn test_object_max_properties_limit() {
    // maxProperties: 2 → accept up to 2 properties
    let schema = r#"{"type": "object", "maxProperties": 2, "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d"}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d","e":"f"}"#));
}

#[test]
fn test_object_min_properties() {
    // minProperties: 1 → reject empty objects
    let schema = r#"{"type": "object", "minProperties": 1, "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(!is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d"}"#));
}

#[test]
fn test_object_min_max_properties() {
    // minProperties: 2, maxProperties: 3
    let schema = r#"{"type": "object", "minProperties": 2, "maxProperties": 3, "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(!is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"a":"b"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d"}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d","e":"f"}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d","e":"f","g":"h"}"#));
}

#[test]
fn test_object_required_props_with_max_properties() {
    // 2 required properties + maxProperties: 3 → at most 1 additional
    let schema = r#"{
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": true,
        "maxProperties": 3
    }"#;
    let opts = JsonSchemaOptions { any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John","age":30}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"name":"John","age":30,"extra":"val"}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"name":"John","age":30,"a":"b","c":"d"}"#));
}

#[test]
fn test_object_min_properties_exactly_one() {
    // minProperties: 1, maxProperties: 1 → exactly one property
    let schema = r#"{"type": "object", "minProperties": 1, "maxProperties": 1, "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(!is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b"}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"a":"b","c":"d"}"#));
}

#[test]
fn test_object_min_properties_type_inference() {
    // Schema with minProperties but no explicit type should infer object
    let schema = r#"{"minProperties": 1, "additionalProperties": true}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(!is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":"b"}"#));
}

#[test]
fn test_object_min_gt_max_properties_error() {
    // minProperties > maxProperties → should error
    let schema = r#"{"type": "object", "minProperties": 5, "maxProperties": 2}"#;
    let opts = JsonSchemaOptions { strict_mode: false, any_whitespace: false, ..Default::default() };
    let result = json_schema_to_grammar(schema, &opts);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("minProperties") && msg.contains("maxProperties"), "unexpected error: {}", msg);
}

#[test]
fn test_object_required_exceeds_max_properties_error() {
    // 3 required properties but maxProperties: 2 → should error
    let schema = r#"{
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"}},
        "required": ["a", "b", "c"],
        "maxProperties": 2
    }"#;
    let opts = JsonSchemaOptions { any_whitespace: false, ..Default::default() };
    let result = json_schema_to_grammar(schema, &opts);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("required") && msg.contains("maxProperties"), "unexpected error: {}", msg);
}
