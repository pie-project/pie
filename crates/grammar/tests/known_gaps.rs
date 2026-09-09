mod common;

use ::grammar::grammar::Grammar;

#[test]
fn known_gaps_every_case() {
    ebnf_lookahead_is_rejected_explicitly();
    ebnf_rejects_repetition_bounds_above_u32();
}

fn ebnf_lookahead_is_rejected_explicitly() {
    assert!(Grammar::from_ebnf(r#"root ::= "a" (="b")"#, "root").is_err());
}

fn ebnf_rejects_repetition_bounds_above_u32() {
    assert!(Grammar::from_ebnf(r#"root ::= "a"{4294967296}"#, "root").is_err());
}
