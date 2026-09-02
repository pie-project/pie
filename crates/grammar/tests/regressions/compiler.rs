use std::sync::Arc;
use std::time::Duration;

use ::grammar::compiler::{GrammarCompiler, GrammarLimits};
use tokenizer::Tokenizer;

#[test]
fn compiler_enforces_resource_limits() {
    let tokenizer = Arc::new(Tokenizer::from_vocab(&[
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
    ]));

    let source_limited = GrammarCompiler::with_limits(
        tokenizer.clone(),
        GrammarLimits {
            max_source_bytes: 8,
            ..GrammarLimits::default()
        },
    );
    assert!(
        source_limited
            .compile_ebnf(r#"root ::= "too long""#, "root")
            .is_err()
    );

    let repeat_limited = GrammarCompiler::with_limits(
        tokenizer.clone(),
        GrammarLimits {
            max_repetition: 3,
            ..GrammarLimits::default()
        },
    );
    assert!(
        repeat_limited
            .compile_ebnf(r#"root ::= "a"{4}"#, "root")
            .is_err()
    );

    let rule_limited = GrammarCompiler::with_limits(
        tokenizer.clone(),
        GrammarLimits {
            max_rules: 1,
            ..GrammarLimits::default()
        },
    );
    assert!(
        rule_limited
            .compile_ebnf("root ::= other\nother ::= \"a\"", "root")
            .is_err()
    );

    let nfa_limited = GrammarCompiler::with_limits(
        tokenizer.clone(),
        GrammarLimits {
            max_nfa_states_per_rule: 2,
            ..GrammarLimits::default()
        },
    );
    assert!(
        nfa_limited
            .compile_ebnf(r#"root ::= "abc""#, "root")
            .is_err()
    );

    let dfa_limited = GrammarCompiler::with_limits(
        tokenizer.clone(),
        GrammarLimits {
            max_dfa_states_per_rule: 2,
            ..GrammarLimits::default()
        },
    );
    assert!(
        dfa_limited
            .compile_ebnf(r#"root ::= "abc""#, "root")
            .is_err()
    );

    let mask_limited = GrammarCompiler::with_limits(
        tokenizer,
        GrammarLimits {
            max_token_mask_bytes: 1,
            ..GrammarLimits::default()
        },
    );
    assert!(
        mask_limited
            .compile_ebnf(r#"root ::= [a-z]+"#, "root")
            .is_err()
    );

    let deadline_limited = GrammarCompiler::with_limits(
        Arc::new(Tokenizer::from_vocab(&["a".to_string()])),
        GrammarLimits {
            max_compile_duration: Duration::ZERO,
            ..GrammarLimits::default()
        },
    );
    assert!(
        deadline_limited
            .compile_ebnf(r#"root ::= "a""#, "root")
            .is_err()
    );
}
