//! Ported from xgrammar: test_grammar_matcher_regex.py
//!
//! Tests regex-based grammar acceptance/rejection.

use std::sync::Arc;

use pie::inference::structured::matcher::GrammarMatcher;
use pie::inference::structured::regex::regex_to_grammar;
use pie::model::tokenizer::Tokenizer;

fn is_regex_accept_string(pattern: &str, input: &str) -> bool {
    let grammar = match regex_to_grammar(pattern) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("regex_to_grammar failed for {:?}: {}", pattern, e);
            return false;
        }
    };
    let vocab: Vec<String> = vec!["dummy".into()];
    let tok = Arc::new(Tokenizer::from_vocab(&vocab));
    let mut m = GrammarMatcher::new(Arc::new(grammar), tok, vec![], 10);

    if input.is_empty() {
        return m.can_terminate();
    }

    if !m.accept_string(input) {
        return false;
    }
    m.can_terminate()
}

// ---------------------------------------------------------------------------
// Basic patterns (from test_basic_regex)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_basic_literals() {
    assert!(is_regex_accept_string("abc", "abc"));
    assert!(!is_regex_accept_string("abc", "abd"));
    assert!(!is_regex_accept_string("abc", "ab"));
    assert!(!is_regex_accept_string("abc", "abcd"));
}

#[test]
fn test_regex_star() {
    assert!(is_regex_accept_string("a*", ""));
    assert!(is_regex_accept_string("a*", "a"));
    assert!(is_regex_accept_string("a*", "aaaa"));
    assert!(!is_regex_accept_string("a*", "b"));
}

#[test]
fn test_regex_plus() {
    assert!(!is_regex_accept_string("a+", ""));
    assert!(is_regex_accept_string("a+", "a"));
    assert!(is_regex_accept_string("a+", "aaaa"));
    assert!(!is_regex_accept_string("a+", "b"));
}

#[test]
fn test_regex_question() {
    assert!(is_regex_accept_string("a?", ""));
    assert!(is_regex_accept_string("a?", "a"));
    assert!(!is_regex_accept_string("a?", "aa"));
}

#[test]
fn test_regex_char_class() {
    assert!(is_regex_accept_string("[abc]", "a"));
    assert!(is_regex_accept_string("[abc]", "b"));
    assert!(is_regex_accept_string("[abc]", "c"));
    assert!(!is_regex_accept_string("[abc]", "d"));
}

#[test]
fn test_regex_alternation() {
    assert!(is_regex_accept_string("cat|dog", "cat"));
    assert!(is_regex_accept_string("cat|dog", "dog"));
    assert!(!is_regex_accept_string("cat|dog", "bird"));
}

// ---------------------------------------------------------------------------
// Advanced patterns (from test_advanced_regex)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_digits() {
    assert!(is_regex_accept_string(r"\d+", "123"));
    assert!(!is_regex_accept_string(r"\d+", "abc"));
}

#[test]
fn test_regex_word_chars() {
    assert!(is_regex_accept_string(r"\w+", "abc123"));
    assert!(!is_regex_accept_string(r"\w+", "!@#"));
}

#[test]
fn test_regex_capitalized_word() {
    assert!(is_regex_accept_string(r"[A-Z][a-z]*", "Hello"));
    assert!(!is_regex_accept_string(r"[A-Z][a-z]*", "hello"));
}

#[test]
fn test_regex_phone_number() {
    assert!(is_regex_accept_string(
        r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
        "123-456-7890"
    ));
    assert!(!is_regex_accept_string(
        r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
        "12-34-567"
    ));
}

#[test]
fn test_regex_email() {
    assert!(is_regex_accept_string(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}",
        "test@email.com"
    ));
    assert!(!is_regex_accept_string(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}",
        "invalid.email"
    ));
}

// ---------------------------------------------------------------------------
// Repetition ranges (from test_non_greedy_quantifier & related)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_repetition_range() {
    assert!(is_regex_accept_string("a{1,3}", "a"));
    assert!(is_regex_accept_string("a{1,3}", "aa"));
    assert!(is_regex_accept_string("a{1,3}", "aaa"));
    assert!(!is_regex_accept_string("a{1,3}", ""));
    assert!(!is_regex_accept_string("a{1,3}", "aaaa"));
}

#[test]
fn test_regex_exact_repetition() {
    assert!(is_regex_accept_string("a{3}", "aaa"));
    assert!(!is_regex_accept_string("a{3}", "aa"));
    assert!(!is_regex_accept_string("a{3}", "aaaa"));
}

#[test]
fn test_regex_unbounded_repetition() {
    assert!(!is_regex_accept_string("a{2,}", "a"));
    assert!(is_regex_accept_string("a{2,}", "aa"));
    assert!(is_regex_accept_string("a{2,}", "aaa"));
    assert!(is_regex_accept_string("a{2,}", &"a".repeat(100)));
}

// ---------------------------------------------------------------------------
// Groups (from test_group, test_quantifier)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_groups() {
    assert!(is_regex_accept_string("(a|b)(c|d)", "ac"));
    assert!(is_regex_accept_string("(a|b)(c|d)", "bc"));
    assert!(is_regex_accept_string("(a|b)(c|d)", "ad"));
    assert!(is_regex_accept_string("(a|b)(c|d)", "bd"));
    assert!(!is_regex_accept_string("(a|b)(c|d)", "ab"));
}

#[test]
fn test_regex_group_with_quantifier() {
    assert!(is_regex_accept_string("(abc)*", ""));
    assert!(is_regex_accept_string("(abc)*", "abc"));
    assert!(is_regex_accept_string("(abc)*", "abcabc"));
    assert!(!is_regex_accept_string("(abc)*", "abca"));
}

#[test]
fn test_regex_mixed_quantifiers() {
    // (a|b)?[a-z]+(abc)*
    assert!(is_regex_accept_string(
        r"(a|b)?[a-z]+(abc)*",
        "adddabcabc"
    ));
    assert!(is_regex_accept_string(r"(a|b)?[a-z]+(abc)*", "z"));
}

// ---------------------------------------------------------------------------
// Dot pattern (from test_any)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_dot() {
    assert!(is_regex_accept_string(".+a.+", "bbbabb"));
    assert!(!is_regex_accept_string(".+a.+", "bbb"));
}

// ---------------------------------------------------------------------------
// Real-world patterns
// ---------------------------------------------------------------------------

#[test]
fn test_regex_ipv4() {
    let pattern = r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)";
    assert!(is_regex_accept_string(pattern, "123.45.67.89"));
}

#[test]
fn test_regex_date() {
    let pattern = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])$";

    assert!(is_regex_accept_string(pattern, "0024-05-19"));
    assert!(is_regex_accept_string(pattern, "2019-11-30"));
    assert!(is_regex_accept_string(pattern, "2022-12-31"));
    assert!(!is_regex_accept_string(pattern, "2024-13-15"));
    assert!(!is_regex_accept_string(pattern, "2024-12-32"));
}

#[test]
fn test_regex_time() {
    let pattern = r"^([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$";

    assert!(is_regex_accept_string(pattern, "14:23:45Z"));
    assert!(is_regex_accept_string(pattern, "08:15:27+05:30"));
    assert!(is_regex_accept_string(pattern, "22:59:59-07:00"));
    assert!(is_regex_accept_string(pattern, "00:00:00.123456Z"));
    assert!(!is_regex_accept_string(pattern, "24:00:00+05:30"));
}

#[test]
fn test_regex_datetime() {
    let pattern = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])T([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$";

    assert!(is_regex_accept_string(pattern, "2024-05-19T14:23:45Z"));
    assert!(is_regex_accept_string(pattern, "2019-11-30T08:15:27+05:30"));
    assert!(is_regex_accept_string(pattern, "2030-02-01T22:59:59-07:00"));
    assert!(is_regex_accept_string(
        pattern,
        "2021-07-04T00:00:00.123456Z"
    ));
    assert!(!is_regex_accept_string(pattern, "2024-13-15T14:30:00Z"));
    assert!(!is_regex_accept_string(
        pattern,
        "2021-11-05T24:00:00+05:30"
    ));
}

// ---------------------------------------------------------------------------
// Empty/anchor patterns (from test_empty)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_empty() {
    assert!(is_regex_accept_string("", ""));
    assert!(!is_regex_accept_string("", "a"));
}

#[test]
fn test_regex_anchors_only() {
    assert!(is_regex_accept_string("^$", ""));
    assert!(!is_regex_accept_string("^$", "a"));
}

// ---------------------------------------------------------------------------
// Non-capturing group
// ---------------------------------------------------------------------------

#[test]
fn test_regex_non_capturing_group() {
    assert!(is_regex_accept_string("(?:abc)", "abc"));
    assert!(!is_regex_accept_string("(?:abc)", "abd"));
}

// ---------------------------------------------------------------------------
// Empty alternatives (from test_empty_alternative)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_empty_alternative() {
    assert!(is_regex_accept_string("(a|)", "a"));
    assert!(is_regex_accept_string("(a|)", ""));
    assert!(!is_regex_accept_string("(a|)", "b"));
}

#[test]
fn test_regex_nested_empty_alternative() {
    assert!(is_regex_accept_string("ab(c|)", "abc"));
    assert!(is_regex_accept_string("ab(c|)", "ab"));
    assert!(!is_regex_accept_string("ab(c|)", "abd"));
}

// ---------------------------------------------------------------------------
// Escaped metacharacters (from test_escape)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_escaped_dot() {
    assert!(is_regex_accept_string(r"\.", "."));
    assert!(!is_regex_accept_string(r"\.", "a"));
}

#[test]
fn test_regex_escaped_special() {
    assert!(is_regex_accept_string(r"\*", "*"));
    assert!(is_regex_accept_string(r"\+", "+"));
    assert!(is_regex_accept_string(r"\(", "("));
    assert!(is_regex_accept_string(r"\)", ")"));
    assert!(is_regex_accept_string(r"\[", "["));
    assert!(is_regex_accept_string(r"\]", "]"));
}

// ---------------------------------------------------------------------------
// Unsupported features (should error)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_unsupported_lookahead() {
    assert!(regex_to_grammar("(?=abc)").is_err());
    assert!(regex_to_grammar("(?!abc)").is_err());
}

#[test]
fn test_regex_unsupported_lookbehind() {
    assert!(regex_to_grammar("(?<=abc)").is_err());
    assert!(regex_to_grammar("(?<!abc)").is_err());
}

#[test]
fn test_regex_unsupported_backreference() {
    assert!(regex_to_grammar(r"\1").is_err());
}

#[test]
fn test_regex_unsupported_word_boundary() {
    assert!(regex_to_grammar(r"\b").is_err());
}

// ---------------------------------------------------------------------------
// Complex repetition (from test_repetition)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_complex_repetition() {
    // (a|[bc]{4,}){2,3}
    let pattern = r"(a|[bc]{4,}){2,3}";

    assert!(is_regex_accept_string(pattern, "aaa"));
    assert!(is_regex_accept_string(pattern, "abcbc"));
    assert!(is_regex_accept_string(pattern, "bcbcbcbcbc"));
    assert!(is_regex_accept_string(pattern, "bcbcbcbcbcbcbcb"));
    assert!(!is_regex_accept_string(pattern, "d"));
    assert!(!is_regex_accept_string(pattern, "aaaa"));
}

// ---------------------------------------------------------------------------
// Advanced patterns (from test_advanced)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_char_class_plus() {
    assert!(is_regex_accept_string(r"[abc]+", "aabbcc"));
    assert!(!is_regex_accept_string(r"[abc]+", "abcd"));
}

#[test]
fn test_regex_alphanumeric_class() {
    assert!(is_regex_accept_string(r"[a-z0-9]+", "abc123"));
    assert!(!is_regex_accept_string(r"[a-z0-9]+", "ABC"));
}

#[test]
fn test_regex_negated_class() {
    assert!(is_regex_accept_string(r"[^abc]+", "def"));
    assert!(!is_regex_accept_string(r"[^abc]+", "aaa"));
}

#[test]
fn test_regex_star_plus_question_combo() {
    assert!(is_regex_accept_string(r"a*b+c?", "b"));
    assert!(is_regex_accept_string(r"a*b+c?", "aaabbc"));
    assert!(!is_regex_accept_string(r"a*b+c?", "c"));
}

#[test]
fn test_regex_alternation_group_plus() {
    assert!(is_regex_accept_string(r"(abc|def)+", "abcdef"));
    assert!(is_regex_accept_string(r"(abc|def)+", "abcabc"));
    assert!(!is_regex_accept_string(r"(abc|def)+", "ab"));
}

// ---------------------------------------------------------------------------
// Non-greedy quantifiers (should be equivalent to greedy for grammar)
// ---------------------------------------------------------------------------

#[test]
fn test_regex_non_greedy_repetition_range() {
    // Non-greedy a{1,3}? should still accept 1-3 a's
    assert!(is_regex_accept_string(r"a{1,3}?", "a"));
    assert!(is_regex_accept_string(r"a{1,3}?", "aa"));
    assert!(is_regex_accept_string(r"a{1,3}?", "aaa"));
    assert!(!is_regex_accept_string(r"a{1,3}?", "aaaa"));
}

#[test]
fn test_regex_non_greedy_plus() {
    assert!(is_regex_accept_string(r"a+?", "a"));
    assert!(is_regex_accept_string(r"a+?", "aaa"));
    assert!(!is_regex_accept_string(r"a+?", ""));
}

#[test]
fn test_regex_non_greedy_star() {
    assert!(is_regex_accept_string(r"a*?", ""));
    assert!(is_regex_accept_string(r"a*?", "aaa"));
}

#[test]
fn test_regex_non_greedy_question() {
    assert!(is_regex_accept_string(r"a??", ""));
    assert!(is_regex_accept_string(r"a??", "a"));
    assert!(!is_regex_accept_string(r"a??", "aa"));
}

// ---------------------------------------------------------------------------
// Non-greedy with following content
// ---------------------------------------------------------------------------

#[test]
fn test_regex_non_greedy_with_context() {
    // For grammar purposes, [abc]+?abc means [abc]+ followed by "abc"
    assert!(is_regex_accept_string(r"[abc]+?abc", "aabc"));
}
