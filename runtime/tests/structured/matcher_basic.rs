//! Ported from xgrammar: test_grammar_matcher_basic.py
//!
//! Tests basic matcher operations: token acceptance, bitmask, rollback,
//! jump-forward, reset, termination.

use std::sync::Arc;

use pie::inference::structured::bitmask;
use pie::inference::structured::grammar::Grammar;
use pie::inference::structured::matcher::GrammarMatcher;
use pie::model::tokenizer::Tokenizer;

/// Build a matcher with a given vocabulary.
fn make_matcher(ebnf: &str, root: &str, vocab: &[&str]) -> GrammarMatcher {
    let grammar = Arc::new(Grammar::from_ebnf(ebnf, root).unwrap());
    let encoded: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
    let tokenizer = Arc::new(Tokenizer::from_vocab(&encoded));
    GrammarMatcher::new(grammar, tokenizer, vec![], 10)
}

/// Build a matcher with explicit stop tokens.
fn make_matcher_with_stop(
    ebnf: &str,
    root: &str,
    vocab: &[&str],
    stop_ids: Vec<u32>,
) -> GrammarMatcher {
    let grammar = Arc::new(Grammar::from_ebnf(ebnf, root).unwrap());
    let encoded: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
    let tokenizer = Arc::new(Tokenizer::from_vocab(&encoded));
    GrammarMatcher::new(grammar, tokenizer, stop_ids, 10)
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
    let mut m = make_matcher(
        r#"root ::= "hello""#,
        "root",
        &["hello", "world"],
    );

    assert!(!m.accept_token(1)); // "world" should fail
    // Parser unchanged after rejection
    assert!(m.accept_token(0)); // "hello" still works
    assert!(m.can_terminate());
}

#[test]
fn test_token_partial_match() {
    let mut m = make_matcher(
        r#"root ::= "abc""#,
        "root",
        &["a", "ab", "abc", "b", "bc", "c", "x"],
    );

    // Accept "a", then "bc"
    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(4)); // "bc"
    assert!(m.can_terminate());
}

#[test]
fn test_token_multi_token_choices() {
    let mut m = make_matcher(
        r#"root ::= "cat" | "car" | "dog""#,
        "root",
        &["c", "a", "t", "r", "d", "o", "g"],
    );

    // Spell out "car"
    assert!(m.accept_token(0)); // "c"
    assert!(m.accept_token(1)); // "a"
    assert!(m.accept_token(3)); // "r"
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// String acceptance
// ---------------------------------------------------------------------------

#[test]
fn test_accept_string_with_char_class() {
    let mut m = make_matcher(r#"root ::= [a-z]+"#, "root", &["a"]);

    assert!(m.accept_string("hello"));
    assert!(m.can_terminate());
}

#[test]
fn test_accept_string_with_quantifier() {
    let mut m = make_matcher(r#"root ::= "a" [0-9]* "b""#, "root", &["a"]);

    assert!(m.accept_string("a123b"));
    assert!(m.can_terminate());
}

#[test]
fn test_accept_string_rejection() {
    let mut m = make_matcher(r#"root ::= "hello""#, "root", &["a"]);

    assert!(!m.accept_string("world"));
    // Parser should be unchanged after rejection
    assert!(m.accept_string("hello"));
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// Bitmask generation
// ---------------------------------------------------------------------------

#[test]
fn test_bitmask_simple_choices() {
    let mut m = make_matcher(
        r#"root ::= "ab" | "cd""#,
        "root",
        &["ab", "cd", "ef"],
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(3)];
    m.fill_next_token_bitmask(&mut bm);

    assert!(bitmask::get_bit(&bm, 0)); // "ab" allowed
    assert!(bitmask::get_bit(&bm, 1)); // "cd" allowed
    assert!(!bitmask::get_bit(&bm, 2)); // "ef" not allowed
}

#[test]
fn test_bitmask_after_partial_accept() {
    let mut m = make_matcher(
        r#"root ::= "abc""#,
        "root",
        &["a", "ab", "abc", "b", "bc", "c"],
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(6)];

    // Initially, tokens starting with "a" should be valid
    m.fill_next_token_bitmask(&mut bm);
    assert!(bitmask::get_bit(&bm, 0)); // "a"
    assert!(bitmask::get_bit(&bm, 1)); // "ab"
    assert!(bitmask::get_bit(&bm, 2)); // "abc"
    assert!(!bitmask::get_bit(&bm, 3)); // "b"
    assert!(!bitmask::get_bit(&bm, 4)); // "bc"
    assert!(!bitmask::get_bit(&bm, 5)); // "c"

    // After accepting "a", need tokens continuing "bc"
    m.accept_token(0); // "a"
    m.fill_next_token_bitmask(&mut bm);
    assert!(!bitmask::get_bit(&bm, 0)); // "a" — no
    assert!(!bitmask::get_bit(&bm, 1)); // "ab" — no
    assert!(!bitmask::get_bit(&bm, 2)); // "abc" — no
    assert!(bitmask::get_bit(&bm, 3)); // "b" — yes
    assert!(bitmask::get_bit(&bm, 4)); // "bc" — yes
    assert!(!bitmask::get_bit(&bm, 5)); // "c" — no
}

#[test]
fn test_bitmask_stop_tokens() {
    let mut m = make_matcher_with_stop(
        r#"root ::= "a" | "ab""#,
        "root",
        &["a", "ab", "b", "<eos>"],
        vec![3], // <eos> is stop token
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(4)];
    m.fill_next_token_bitmask(&mut bm);

    assert!(bitmask::get_bit(&bm, 0)); // "a"
    assert!(bitmask::get_bit(&bm, 1)); // "ab"
    assert!(!bitmask::get_bit(&bm, 2)); // "b"
    assert!(!bitmask::get_bit(&bm, 3)); // <eos>

    // After "a", grammar can terminate
    m.accept_token(0);
    m.fill_next_token_bitmask(&mut bm);
    assert!(bitmask::get_bit(&bm, 2)); // "b" — completes "ab"
    assert!(bitmask::get_bit(&bm, 3)); // <eos> — grammar can stop
}

#[test]
fn test_bitmask_many_tokens() {
    let mut m = make_matcher(
        r#"root ::= "true" | "false" | "null""#,
        "root",
        &["t", "tr", "true", "f", "fa", "false", "n", "nu", "null", "x"],
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(10)];
    m.fill_next_token_bitmask(&mut bm);

    // Tokens for "true", "false", "null" should be accepted
    assert!(bitmask::get_bit(&bm, 0)); // "t"
    assert!(bitmask::get_bit(&bm, 1)); // "tr"
    assert!(bitmask::get_bit(&bm, 2)); // "true"
    assert!(bitmask::get_bit(&bm, 3)); // "f"
    assert!(bitmask::get_bit(&bm, 4)); // "fa"
    assert!(bitmask::get_bit(&bm, 5)); // "false"
    assert!(bitmask::get_bit(&bm, 6)); // "n"
    assert!(bitmask::get_bit(&bm, 7)); // "nu"
    assert!(bitmask::get_bit(&bm, 8)); // "null"
    assert!(!bitmask::get_bit(&bm, 9)); // "x"
}

// ---------------------------------------------------------------------------
// Rollback
// ---------------------------------------------------------------------------

#[test]
fn test_rollback_single() {
    let mut m = make_matcher(
        r#"root ::= "abc""#,
        "root",
        &["a", "b", "c"],
    );

    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"

    m.rollback(1); // undo "b"

    assert!(m.accept_token(1)); // "b" again
    assert!(m.accept_token(2)); // "c"
    assert!(m.can_terminate());
}

#[test]
fn test_rollback_multiple() {
    let mut m = make_matcher(
        r#"root ::= "abcd""#,
        "root",
        &["a", "b", "c", "d"],
    );

    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"
    assert!(m.accept_token(2)); // "c"

    m.rollback(2); // undo "c" and "b"

    assert!(m.accept_token(1)); // "b"
    assert!(m.accept_token(2)); // "c"
    assert!(m.accept_token(3)); // "d"
    assert!(m.can_terminate());
}

#[test]
fn test_rollback_all() {
    let mut m = make_matcher(
        r#"root ::= "ab""#,
        "root",
        &["a", "b"],
    );

    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"
    assert!(m.can_terminate());

    m.rollback(2); // undo both

    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"
    assert!(m.can_terminate());
}

#[test]
fn test_rollback_with_different_path() {
    let mut m = make_matcher(
        r#"root ::= "abc" | "abd""#,
        "root",
        &["a", "b", "c", "d"],
    );

    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"
    assert!(m.accept_token(2)); // "c" → "abc"
    assert!(m.can_terminate());

    m.rollback(1); // undo "c"

    assert!(m.accept_token(3)); // "d" → "abd"
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

#[test]
fn test_reset() {
    let mut m = make_matcher(
        r#"root ::= "ab""#,
        "root",
        &["a", "b"],
    );

    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"
    assert!(m.can_terminate());

    m.reset();

    assert!(!m.can_terminate());
    assert!(m.accept_token(0)); // "a"
    assert!(m.accept_token(1)); // "b"
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// Termination
// ---------------------------------------------------------------------------

#[test]
fn test_termination_with_stop_token() {
    let mut m = make_matcher_with_stop(
        r#"root ::= "hello""#,
        "root",
        &["hello", "<eos>"],
        vec![1],
    );

    assert!(!m.is_terminated());
    assert!(m.accept_token(0)); // "hello"
    assert!(m.can_terminate());
    assert!(!m.is_terminated());

    assert!(m.accept_token(1)); // <eos>
    assert!(m.is_terminated());

    // After termination, no more tokens accepted
    assert!(!m.accept_token(0));
}

#[test]
fn test_stop_token_rejected_when_incomplete() {
    let mut m = make_matcher_with_stop(
        r#"root ::= "hello""#,
        "root",
        &["hel", "lo", "<eos>"],
        vec![2],
    );

    assert!(m.accept_token(0)); // "hel"
    // Grammar not complete yet, stop token should be rejected
    assert!(!m.accept_token(2)); // <eos>
    // Continue and complete
    assert!(m.accept_token(1)); // "lo"
    assert!(m.accept_token(2)); // <eos> now accepted
    assert!(m.is_terminated());
}

// ---------------------------------------------------------------------------
// Jump forward
// ---------------------------------------------------------------------------

#[test]
fn test_jump_forward_deterministic() {
    let mut m = make_matcher(
        r#"root ::= "hello""#,
        "root",
        &["hello"],
    );

    let jf = m.find_jump_forward_string();
    assert_eq!(jf, "hello");
}

#[test]
fn test_jump_forward_partial() {
    let mut m = make_matcher(
        r#"root ::= "prefix" ("a" | "b")"#,
        "root",
        &["prefix", "a", "b"],
    );

    let jf = m.find_jump_forward_string();
    assert_eq!(jf, "prefix");
}

#[test]
fn test_jump_forward_after_accept() {
    let mut m = make_matcher(
        r#"root ::= "ab" "cd""#,
        "root",
        &["ab", "cd"],
    );

    m.accept_token(0); // "ab"
    let jf = m.find_jump_forward_string();
    assert_eq!(jf, "cd");
}

#[test]
fn test_jump_forward_with_choices() {
    let mut m = make_matcher(
        r#"root ::= "a" | "b""#,
        "root",
        &["a", "b"],
    );

    let jf = m.find_jump_forward_string();
    // Multiple choices → empty jump forward
    assert_eq!(jf, "");
}

#[test]
fn test_jump_forward_does_not_advance_parser() {
    let mut m = make_matcher(
        r#"root ::= "hello""#,
        "root",
        &["hello"],
    );

    let jf1 = m.find_jump_forward_string();
    let jf2 = m.find_jump_forward_string();
    assert_eq!(jf1, jf2); // Should be idempotent

    // Parser should not have advanced
    assert!(m.accept_string("hello"));
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// Mixed operations
// ---------------------------------------------------------------------------

#[test]
fn test_accept_rollback_bitmask() {
    let mut m = make_matcher(
        r#"root ::= "abc""#,
        "root",
        &["a", "b", "c", "x"],
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(4)];

    // Initial bitmask
    m.fill_next_token_bitmask(&mut bm);
    assert!(bitmask::get_bit(&bm, 0)); // "a"
    assert!(!bitmask::get_bit(&bm, 1)); // "b"
    assert!(!bitmask::get_bit(&bm, 3)); // "x"

    // Accept "a"
    m.accept_token(0);
    m.fill_next_token_bitmask(&mut bm);
    assert!(!bitmask::get_bit(&bm, 0)); // "a"
    assert!(bitmask::get_bit(&bm, 1)); // "b"

    // Rollback "a"
    m.rollback(1);
    m.fill_next_token_bitmask(&mut bm);
    assert!(bitmask::get_bit(&bm, 0)); // "a" again valid
    assert!(!bitmask::get_bit(&bm, 1)); // "b" invalid again
}

#[test]
fn test_multitoken_string() {
    let mut m = make_matcher(
        r#"root ::= "the" " " "cat" " " "sat""#,
        "root",
        &["the", " ", "cat", "sat", "dog"],
    );

    assert!(m.accept_token(0)); // "the"
    assert!(m.accept_token(1)); // " "
    assert!(m.accept_token(2)); // "cat"
    assert!(m.accept_token(1)); // " "
    assert!(m.accept_token(3)); // "sat"
    assert!(m.can_terminate());
}

#[test]
fn test_char_class_star_bitmask() {
    let mut m = make_matcher(
        r#"root ::= [a-z]*"#,
        "root",
        &["a", "b", "abc", "1", "A"],
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(5)];
    m.fill_next_token_bitmask(&mut bm);

    assert!(bitmask::get_bit(&bm, 0)); // "a"
    assert!(bitmask::get_bit(&bm, 1)); // "b"
    assert!(bitmask::get_bit(&bm, 2)); // "abc"
    assert!(!bitmask::get_bit(&bm, 3)); // "1"
    assert!(!bitmask::get_bit(&bm, 4)); // "A"
}

#[test]
fn test_grammar_with_many_rules_bitmask() {
    let ebnf = r#"
root ::= greeting " " subject
greeting ::= "hi" | "hello" | "hey"
subject ::= "world" | "there"
"#;
    let mut m = make_matcher(
        ebnf,
        "root",
        &["hi", "hello", "hey", " ", "world", "there", "x"],
    );

    let mut bm = vec![0u32; bitmask::bitmask_size(7)];
    m.fill_next_token_bitmask(&mut bm);

    assert!(bitmask::get_bit(&bm, 0)); // "hi"
    assert!(bitmask::get_bit(&bm, 1)); // "hello"
    assert!(bitmask::get_bit(&bm, 2)); // "hey"
    assert!(!bitmask::get_bit(&bm, 3)); // " "
    assert!(!bitmask::get_bit(&bm, 6)); // "x"

    m.accept_token(0); // "hi"
    m.fill_next_token_bitmask(&mut bm);
    assert!(bitmask::get_bit(&bm, 3)); // " "
    assert!(!bitmask::get_bit(&bm, 0)); // "hi" — no

    m.accept_token(3); // " "
    m.fill_next_token_bitmask(&mut bm);
    assert!(bitmask::get_bit(&bm, 4)); // "world"
    assert!(bitmask::get_bit(&bm, 5)); // "there"
    assert!(!bitmask::get_bit(&bm, 0)); // "hi" — no
}

// ---------------------------------------------------------------------------
// JSON grammar token operations (from test_token_operations)
// ---------------------------------------------------------------------------

const JSON_GRAMMAR_EBNF: &str = r#"
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= ws string ws ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
"#;

#[test]
fn test_json_token_sequence() {
    // Vocab: ["{", "}", "\"", "name", ":", "42", ",", " "]
    let mut m = make_matcher(
        JSON_GRAMMAR_EBNF,
        "root",
        &["{", "}", "\"", "name", ":", "42", ",", " "],
    );

    // Build: {"name": 42}
    assert!(m.accept_token(0)); // "{"
    assert!(m.accept_token(2)); // "\""
    assert!(m.accept_token(3)); // "name"
    assert!(m.accept_token(2)); // "\""
    assert!(m.accept_token(4)); // ":"
    assert!(m.accept_token(7)); // " "
    assert!(m.accept_token(5)); // "42"
    assert!(m.accept_token(1)); // "}"
    assert!(m.can_terminate());
}

#[test]
fn test_json_rollback_and_bitmask_consistency() {
    // Verify that bitmask is identical after rollback + re-accept
    let mut m = make_matcher(
        JSON_GRAMMAR_EBNF,
        "root",
        &["{", "}", "\"", "a", ":", "1", ",", " "],
    );

    let vocab_size = 8;
    let bm_size = bitmask::bitmask_size(vocab_size);

    // Accept "{" then "\""
    assert!(m.accept_token(0)); // "{"
    assert!(m.accept_token(2)); // "\""

    // Record bitmask after two tokens
    let mut bm_after = vec![0u32; bm_size];
    m.fill_next_token_bitmask(&mut bm_after);

    // Rollback both
    m.rollback(2);

    // Re-accept same tokens
    assert!(m.accept_token(0)); // "{"
    assert!(m.accept_token(2)); // "\""

    // Verify bitmask matches
    let mut bm_again = vec![0u32; bm_size];
    m.fill_next_token_bitmask(&mut bm_again);

    assert_eq!(bm_after, bm_again, "bitmask should be identical after rollback + re-accept");
}

#[test]
fn test_graceful_rollback_after_failed_token() {
    // After a token fails, the matcher should gracefully rollback
    let mut m = make_matcher(
        r#"root ::= "ab" | "ac""#,
        "root",
        &["a", "b", "c", "d"],
    );

    assert!(m.accept_token(0)); // "a"

    // Try "d" — should fail
    assert!(!m.accept_token(3));

    // State should be unchanged (after "a")
    assert!(m.accept_token(1)); // "b" → "ab"
    assert!(m.can_terminate());
}

// ---------------------------------------------------------------------------
// Jump forward with specific grammar (from test_get_jump_forward_string)
// ---------------------------------------------------------------------------

#[test]
fn test_jump_forward_xgrammar() {
    let grammar = r#"
root ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
"#;
    let mut m = make_matcher(grammar, "root", &["a", "b", "d"]);

    // Accept "a"
    assert!(m.accept_string("a"));

    // All 3 branches require "bb" next
    let jf = m.find_jump_forward_string();
    assert_eq!(jf, "bb");
}

// ---------------------------------------------------------------------------
// Bitmask with custom vocab size (from test_vocab_size)
// ---------------------------------------------------------------------------

#[test]
fn test_bitmask_larger_vocab_size() {
    // Grammar only accepts "{", but vocabulary has 64 entries
    let mut vocab: Vec<&str> = vec!["<s>"; 64];
    vocab[0] = "{";
    vocab[1] = "}";
    vocab[2] = "\"";
    vocab[3] = "a";

    let mut m = make_matcher(r#"root ::= "{""#, "root", &vocab);

    let mut bm = vec![0u32; bitmask::bitmask_size(64)];
    m.fill_next_token_bitmask(&mut bm);

    // Only token 0 ("{") should be accepted
    assert!(bitmask::get_bit(&bm, 0));
    for i in 1..64 {
        assert!(!bitmask::get_bit(&bm, i), "token {} should be rejected", i);
    }
}
