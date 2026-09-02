//! Executable specifications for deliberate limitations and upstream-compatible gaps.
//!
//! Ignored tests must describe desired behavior. Remove `#[ignore]` only with
//! the implementation change that makes the specification pass.

mod common;

use ::grammar::grammar::Grammar;

#[test]
fn ebnf_lookahead_is_rejected_explicitly() {
    assert!(Grammar::from_ebnf(r#"root ::= "a" (="b")"#, "root").is_err());
}

#[test]
fn ebnf_rejects_repetition_bounds_above_u32() {
    assert!(Grammar::from_ebnf(r#"root ::= "a"{4294967296}"#, "root").is_err());
}

