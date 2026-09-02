//! Stateful matcher operations (token acceptance, bitmask, rollback,
//! jump-forward, reset, termination), ported from xgrammar.

use crate::common::matcher_from_ebnf as make_matcher;

// ---------------------------------------------------------------------------
// Control char exclusion in JSON-style char classes
// ---------------------------------------------------------------------------

#[test]
fn test_char_class_excludes_control_chars() {
    // Should match any char except quote, backslash, and control chars 0x00-0x1f.
    let ebnf = r#"root ::= [^"\x00-\x1f\\]"#;
    let vocab = &["\n", "\t", "x", " ", "\"", "\\", "a"];
    let mut m = make_matcher(ebnf, "root", vocab);
    // Control chars must be rejected.
    assert!(!m.accept_token(0), "\\n should be rejected");
    let mut m2 = make_matcher(ebnf, "root", vocab);
    assert!(!m2.accept_token(1), "\\t should be rejected");
    // Printable chars and quote/backslash behavior:
    let mut m3 = make_matcher(ebnf, "root", vocab);
    assert!(m3.accept_token(2), "x should be accepted");
    let mut m4 = make_matcher(ebnf, "root", vocab);
    assert!(m4.accept_token(3), "space should be accepted");
    let mut m5 = make_matcher(ebnf, "root", vocab);
    assert!(!m5.accept_token(4), "quote should be rejected");
    let mut m6 = make_matcher(ebnf, "root", vocab);
    assert!(!m6.accept_token(5), "backslash should be rejected");
}

// ---------------------------------------------------------------------------
// Token acceptance
// ---------------------------------------------------------------------------

#[test]
fn test_token_acceptance_sequence() {
    let mut m = make_matcher(
        r#"root ::= "hello" " " "world""#,
        "root",
        &["hello", " ", "world", "x"],
    );

    assert!(m.accept_token(0)); // "hello"
    assert!(m.accept_token(1)); // " "
    assert!(m.accept_token(2)); // "world"
    assert!(m.can_terminate());
}

#[test]
fn test_token_rejection() {
    let mut m = make_matcher(r#"root ::= "hello""#, "root", &["hello", "world"]);

    assert!(!m.accept_token(1)); // "world" should fail
    // Parser unchanged after rejection
    assert!(m.accept_token(0)); // "hello" still works
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// String acceptance
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bitmask generation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rollback
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Termination
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Jump forward
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Mixed operations
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// JSON grammar token operations (from test_token_operations)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Jump forward with specific grammar (from test_get_jump_forward_string)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bitmask with custom vocab size (from test_vocab_size)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fork
// ---------------------------------------------------------------------------

