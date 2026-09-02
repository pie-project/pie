//! Regex matcher acceptance/rejection tests.

use crate::common::regex_accepts as is_regex_accept_string;

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

