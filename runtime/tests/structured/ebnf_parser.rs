//! Tests for EBNF parser output format and error cases.
//!
//! Item 1: Round-trip tests (EBNF → Grammar → Display → verify).
//! Item 2: Error case tests (malformed EBNF → Err with message).

use pie::inference::structured::grammar::Grammar;

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
fn test_output_character_class_star() {
    // CharacterClassStar optimization: no aux rule generated
    let g = parse_and_display("root ::= [a-z]*");
    assert!(g.contains("[a-z]*"), "expected CharacterClassStar, got: {}", g);
    // Should NOT have auxiliary rules
    assert!(!g.contains("root_1"), "should not have aux rule for char class star, got: {}", g);
}

#[test]
fn test_output_string_star() {
    // "a"* DOES need auxiliary rule
    let g = parse_and_display(r#"root ::= "a"*"#);
    assert!(g.contains("root_1"), "should have aux rule for string star, got: {}", g);
}

#[test]
fn test_output_string_plus() {
    let g = parse_and_display(r#"root ::= "a"+"#);
    assert!(g.contains("root_1"), "should have aux rule for plus, got: {}", g);
}

#[test]
fn test_output_string_question() {
    let g = parse_and_display(r#"root ::= "a"?"#);
    assert!(g.contains("root_1"), "should have aux rule for ?, got: {}", g);
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
    assert!(g.contains("root_1{3,3}"), "expected root_1{{3,3}}, got: {}", g);
    assert!(g.contains("root_1 ::= \"a\""), "expected root_1 rule, got: {}", g);
}

#[test]
fn test_output_repetition_range() {
    let g = parse_and_display(r#"root ::= "a"{2,4}"#);
    // {2,4} becomes Repeat(root_1, 2, 4)
    assert!(g.contains("root_1{2,4}"), "expected root_1{{2,4}}, got: {}", g);
}

#[test]
fn test_output_repetition_unbounded() {
    let g = parse_and_display(r#"root ::= "a"{2,}"#);
    assert!(g.contains("root_1"), "should have aux rule for {{2,}}, got: {}", g);
}

#[test]
fn test_output_multi_rule() {
    let g = parse_and_display("root ::= digit+\ndigit ::= [0-9]");
    assert!(g.contains("root ::="));
    assert!(g.contains("digit ::="));
    assert!(g.contains("[0-9]"));
}

#[test]
fn test_output_rule_reference() {
    let g = parse_and_display("root ::= item\nitem ::= [a-z]");
    assert!(g.contains("item"), "should reference item rule, got: {}", g);
}

#[test]
fn test_output_unicode_char_class() {
    let g = parse_and_display(r"root ::= [\u0041-\u005A]");
    // A-Z (U+0041 to U+005A) — Display should show them as printable
    assert!(g.contains("A-Z") || g.contains("\\u0041"), "got: {}", g);
}

#[test]
fn test_output_lookahead() {
    let g = parse_and_display(r#"root ::= "a" (="b")"#);
    assert!(g.contains("(="), "should contain lookahead, got: {}", g);
    assert!(g.contains("\"b\""));
}

#[test]
fn test_output_nested_groups() {
    let g = parse_and_display(r#"root ::= ("a" | "b") ("c" | "d")"#);
    assert!(g.contains("\"a\""));
    assert!(g.contains("\"b\""));
    assert!(g.contains("\"c\""));
    assert!(g.contains("\"d\""));
    assert!(g.contains("|"));
}

#[test]
fn test_output_escape_sequences() {
    let g = parse_and_display(r#"root ::= "\n\t\r""#);
    assert!(g.contains("\\n") || g.contains("\\t"), "got: {}", g);
}

#[test]
fn test_output_complex_grammar() {
    let ebnf = r#"
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" pair* "}"
pair ::= string ":" value
array ::= "[" (value ("," value)*)? "]"
string ::= "\"" [^"\\]* "\""
number ::= [0-9]+
"#;
    let g = Grammar::from_ebnf(ebnf, "root").unwrap();
    let output = g.to_string();
    assert!(output.contains("root ::="));
    assert!(output.contains("value ::="));
    assert!(output.contains("object ::="));
    assert!(output.contains("string ::="));
    assert!(output.contains("number ::="));
}

#[test]
fn test_output_comments_ignored() {
    let g = parse_and_display("# this is a comment\nroot ::= \"hello\" # inline comment\n");
    assert!(g.contains("\"hello\""));
    assert!(!g.contains("#"));
}

#[test]
fn test_output_char_class_special_chars() {
    let g = parse_and_display(r"root ::= [a\-z\]\\]");
    // Should preserve special chars in char class
    let output = g;
    assert!(output.contains("["), "got: {}", output);
    assert!(output.contains("]"), "got: {}", output);
}

#[test]
fn test_output_empty_parens() {
    let g = parse_and_display("root ::= ()");
    assert!(g.contains("\"\""), "empty parens should produce empty string, got: {}", g);
}

// ---------------------------------------------------------------------------
// Item 2: Error case tests
// ---------------------------------------------------------------------------

#[test]
fn test_error_unterminated_string() {
    let result = Grammar::from_ebnf(r#"root ::= "abc"#, "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("unterminated") || msg.contains("string"), "unexpected error: {}", msg);
}

#[test]
fn test_error_invalid_escape() {
    let result = Grammar::from_ebnf(r#"root ::= "\q""#, "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("escape") || msg.contains("invalid"), "unexpected error: {}", msg);
}

#[test]
fn test_error_undefined_rule() {
    let result = Grammar::from_ebnf("root ::= other_rule", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not defined"), "unexpected error: {}", msg);
}

#[test]
fn test_error_duplicate_rule() {
    let result = Grammar::from_ebnf("root ::= \"a\"\nroot ::= \"b\"", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("multiple times"), "unexpected error: {}", msg);
}

#[test]
fn test_error_unclosed_char_class() {
    let result = Grammar::from_ebnf("root ::= [a-z", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("unterminated") || msg.contains("character class") || msg.contains("expected ]"),
        "unexpected error: {}", msg
    );
}

#[test]
fn test_error_missing_root_rule() {
    let result = Grammar::from_ebnf("foo ::= \"a\"", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found"), "unexpected error: {}", msg);
}

#[test]
fn test_error_reversed_char_class_range() {
    let result = Grammar::from_ebnf("root ::= [z-a]", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("lower bound") || msg.contains("invalid"),
        "unexpected error: {}", msg
    );
}

#[test]
fn test_error_unclosed_paren() {
    let result = Grammar::from_ebnf(r#"root ::= ("a""#, "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("expected )") || msg.contains("paren"),
        "unexpected error: {}", msg
    );
}

#[test]
fn test_error_unexpected_token() {
    let result = Grammar::from_ebnf("root ::= @", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("unexpected"), "unexpected error: {}", msg);
}

#[test]
fn test_error_empty_grammar() {
    let result = Grammar::from_ebnf("", "root");
    assert!(result.is_err());
}

#[test]
fn test_error_rule_body_empty() {
    // An empty rule body leads to a parse error at the next rule or EOF
    let result = Grammar::from_ebnf("root ::=\nother ::= \"a\"", "root");
    assert!(result.is_err(), "should reject empty rule body");
}

#[test]
fn test_error_repetition_lower_gt_upper() {
    let result = Grammar::from_ebnf(r#"root ::= "a"{5,2}"#, "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("lower bound") || msg.contains("larger"), "unexpected error: {}", msg);
}

#[test]
fn test_error_negative_repetition() {
    let result = Grammar::from_ebnf(r#"root ::= "a"{-1}"#, "root");
    assert!(result.is_err());
}

#[test]
fn test_error_newline_in_string() {
    let result = Grammar::from_ebnf("root ::= \"abc\ndef\"", "root");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("unterminated") || msg.contains("string"),
        "unexpected error: {}", msg
    );
}
