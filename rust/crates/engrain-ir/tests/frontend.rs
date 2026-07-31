use engrain_ir::grammar::Grammar;
use engrain_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use engrain_ir::regex::regex_to_grammar;

#[test]
fn ebnf_round_trips() {
    let grammar = Grammar::from_ebnf(r#"root ::= "yes" | "no""#, "root").unwrap();
    assert!(!grammar.rules().is_empty());
}

#[test]
fn json_schema_lowers_to_a_grammar() {
    let schema = r#"{
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": false
    }"#;
    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(!grammar.rules().is_empty());
}

#[test]
fn regex_lowers_to_a_grammar() {
    let grammar = regex_to_grammar(r"[a-z]+@[a-z]+\.[a-z]{2,4}").unwrap();
    assert!(!grammar.rules().is_empty());
}

#[test]
fn lookahead_is_rejected() {
    assert!(Grammar::from_ebnf(r#"root ::= "a" (="b")"#, "root").is_err());
}
