//! EBNF parsing, printing, round-trip, and error behavior.
//!
//! Item 1: Round-trip tests (EBNF → Grammar → Display → verify).
//! Item 2: Error case tests (malformed EBNF → Err with message).

use ::grammar::grammar::Grammar;

fn parse_and_display(input: &str) -> String {
    Grammar::from_ebnf(input, "root").unwrap().to_string()
}

// ---------------------------------------------------------------------------
// Item 1: Output format round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn test_output_simple_literal() {
    let g = parse_and_display(r#"root ::= "abc""#);
    assert!(g.starts_with("root ::= "));
    assert!(g.contains("\"abc\""));
}

#[test]
fn test_output_empty_string() {
    let g = parse_and_display(r#"root ::= """#);
    assert!(g.contains("\"\""));
}

#[test]
fn test_output_character_class() {
    let g = parse_and_display("root ::= [a-z0-9]");
    assert!(g.contains("[a-z0-9]"));
}

#[test]
fn test_output_negated_character_class() {
    let g = parse_and_display("root ::= [^a-z]");
    assert!(g.contains("[^a-z]"));
}

#[test]
fn test_output_string_star() {
    // "a"* DOES need auxiliary rule
    let g = parse_and_display(r#"root ::= "a"*"#);
    assert!(
        g.contains("root_1"),
        "should have aux rule for string star, got: {}",
        g
    );
}

#[test]
fn test_output_alternation() {
    let g = parse_and_display(r#"root ::= "a" | "b" | "c""#);
    assert!(g.contains("\"a\""));
    assert!(g.contains("\"b\""));
    assert!(g.contains("\"c\""));
    assert!(g.contains("|"));
}

#[test]
fn test_output_sequence() {
    let g = parse_and_display(r#"root ::= "a" "b" "c""#);
    assert!(g.contains("\"a\""));
    assert!(g.contains("\"b\""));
    assert!(g.contains("\"c\""));
}

#[test]
fn test_output_repetition_exact() {
    let g = parse_and_display(r#"root ::= "a"{3}"#);
    // {3} becomes Repeat(root_1, 3, 3) where root_1 ::= "a"
    assert!(
        g.contains("root_1{3,3}"),
        "expected root_1{{3,3}}, got: {}",
        g
    );
    assert!(
        g.contains("root_1 ::= \"a\""),
        "expected root_1 rule, got: {}",
        g
    );
}

// ---------------------------------------------------------------------------
// Item 2: Error case tests
// ---------------------------------------------------------------------------

