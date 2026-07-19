//! Executable specifications for deliberate limitations and upstream-compatible gaps.
//!
//! Ignored tests must describe desired behavior. Remove `#[ignore]` only with
//! the implementation change that makes the specification pass.

mod common;

use common::{grammar_accepts, regex_accepts};
use pie_grammar::grammar::Grammar;
use pie_grammar::json_schema::{JsonSchemaOptions, json_schema_to_grammar};

#[test]
#[ignore = "JSON Schema pattern currently uses full-match instead of search semantics"]
fn json_schema_pattern_matches_a_substring() {
    let schema = r#"{"type":"string","pattern":"a"}"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar, r#""ba""#));
}

#[test]
fn ebnf_lookahead_is_rejected_explicitly() {
    assert!(Grammar::from_ebnf(r#"root ::= "a" (="b")"#, "root").is_err());
}

#[test]
#[ignore = "prefixItems currently requires every declared prefix item"]
fn json_schema_prefix_items_allows_shorter_arrays() {
    let schema = r#"{
        "type":"array",
        "prefixItems":[{"type":"string"},{"type":"integer"}],
        "items":false
    }"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar_accepts(grammar, r#"["only-first"]"#));
}

#[test]
fn ebnf_rejects_repetition_bounds_above_u32() {
    assert!(Grammar::from_ebnf(r#"root ::= "a"{4294967296}"#, "root").is_err());
}

#[test]
#[ignore = "dot currently includes line terminators instead of JavaScript regex semantics"]
fn regex_dot_rejects_newline() {
    assert!(!regex_accepts(".", "\n"));
}
