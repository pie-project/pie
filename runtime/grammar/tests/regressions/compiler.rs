use std::sync::Arc;
use std::time::Duration;

use pie_grammar::compiler::{GrammarCompiler, GrammarLimits};
use pie_grammar::json_schema::JsonSchemaOptions;
use pie_grammar::matcher::GrammarMatcher;
use pie_tokenizer::Tokenizer;

#[test]
fn cache_separates_frontend_namespaces() {
    let source = r#"root ::= "a""#;
    let tokenizer = Arc::new(Tokenizer::from_vocab(&["dummy".to_string()]));
    let compiler = GrammarCompiler::new(tokenizer.clone());

    let ebnf = compiler.compile_ebnf(source, "root").unwrap();
    let regex = compiler.compile_regex(source).unwrap();
    assert!(!Arc::ptr_eq(&ebnf, &regex));
    assert!(Arc::ptr_eq(
        &ebnf,
        &compiler.compile_ebnf(source, "root").unwrap()
    ));

    let mut ebnf_matcher = GrammarMatcher::with_compiled(ebnf, Vec::new(), 0);
    assert!(ebnf_matcher.accept_string("a"));
    assert!(ebnf_matcher.can_terminate());

    let mut regex_matcher = GrammarMatcher::with_compiled(regex, Vec::new(), 0);
    assert!(regex_matcher.accept_string(source));
    assert!(regex_matcher.can_terminate());
}

#[test]
fn cache_separates_json_schema_options() {
    let schema = r#"{"type":"array","items":{"type":"integer"}}"#;
    let tokenizer = Arc::new(Tokenizer::from_vocab(&["dummy".to_string()]));
    let compiler = GrammarCompiler::new(tokenizer);
    let any_whitespace = JsonSchemaOptions::default();
    let no_whitespace = JsonSchemaOptions {
        any_whitespace: false,
        ..JsonSchemaOptions::default()
    };

    let first = compiler
        .compile_json_schema(schema, &any_whitespace)
        .unwrap();
    let second = compiler
        .compile_json_schema(schema, &no_whitespace)
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
}

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
