//! Regex-to-EBNF conversion and validation behavior.
//!
//! Tests regex → EBNF conversion correctness.

use ::grammar::regex::regex_to_ebnf;

/// Helper: get the EBNF body (everything after "root ::= ").
fn ebnf_body(pattern: &str) -> String {
    let ebnf = regex_to_ebnf(pattern).unwrap();

    ebnf.strip_prefix("root ::= ")
        .unwrap_or(&ebnf)
        .trim_end()
        .to_string()
}

// ---------------------------------------------------------------------------
// Basic conversion (from test_basic)
// ---------------------------------------------------------------------------

#[test]
fn test_basic_literal() {
    let body = ebnf_body("123");
    assert!(body.contains("\"1\"") || body.contains("\"123\""));
}

// ---------------------------------------------------------------------------
// Anchors (from test_boundary)
// ---------------------------------------------------------------------------

#[test]
fn test_anchors_stripped() {
    let body = ebnf_body("^abc$");
    // Anchors should be stripped, leaving just the literal
    assert!(body.contains("\"a\"") || body.contains("\"abc\""));
    assert!(!body.contains("^"));
    assert!(!body.contains("$"));
}

// ---------------------------------------------------------------------------
// Quantifiers (from test_quantifier)
// ---------------------------------------------------------------------------

#[test]
fn test_star_quantifier_conversion() {
    let body = ebnf_body("a*");
    assert!(body.contains("*"), "body should contain *: {}", body);
}

#[test]
fn test_plus_quantifier_conversion() {
    let body = ebnf_body("a+");
    assert!(body.contains("+"), "body should contain +: {}", body);
}

#[test]
fn test_question_quantifier_conversion() {
    let body = ebnf_body("a?");
    assert!(body.contains("?"), "body should contain ?: {}", body);
}

#[test]
fn test_repetition_exact_conversion() {
    let body = ebnf_body("a{3}");
    assert!(body.contains("{3}"), "body should contain {{3}}: {}", body);
}

#[test]
fn test_repetition_range_conversion() {
    let body = ebnf_body("a{2,5}");
    assert!(
        body.contains("{2,5}"),
        "body should contain {{2,5}}: {}",
        body
    );
}

#[test]
fn test_repetition_unbounded_conversion() {
    let body = ebnf_body("a{1,}");
    assert!(
        body.contains("{1,}") || body.contains('+'),
        "body should contain an unbounded repetition: {}",
        body
    );
}

// ---------------------------------------------------------------------------
// Alternation (from test_disjunction)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Groups (from test_group)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dot (from test_any)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shorthand classes
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Non-greedy quantifiers (from test_non_greedy_quantifier)
// These are ignored for grammar purposes but should not error.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Empty patterns (from test_empty)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Group modifiers (from test_group_modifiers)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Empty alternatives (from test_empty_alternative)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Empty parentheses (from test_empty_parentheses)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Unicode escape (from test_unicode)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Escaped metacharacters
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Complex patterns — end-to-end parse check (no panics)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Error cases (Item 3)
// ---------------------------------------------------------------------------

