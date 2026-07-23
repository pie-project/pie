use std::sync::Arc;

use pie_grammar::bitmask::{bitmask_size, get_bit};
use pie_grammar::compiled_grammar::CompiledGrammar;
use pie_grammar::grammar::Grammar;
use pie_grammar::matcher::GrammarMatcher;
use pie_tokenizer::Tokenizer;

#[test]
fn more_than_eight_nullable_rules_complete_at_one_position() {
    let grammar = Grammar::from_ebnf(
        r#"
root ::= a a b b c c d d e e f f g g h h i i "x"
a ::= "" | "a" a
b ::= "" | "b" b
c ::= "" | "c" c
d ::= "" | "d" d
e ::= "" | "e" e
f ::= "" | "f" f
g ::= "" | "g" g
h ::= "" | "h" h
i ::= "" | "i" i
"#,
        "root",
    )
    .unwrap();
    let tokenizer = Arc::new(Tokenizer::from_vocab(&["dummy".to_string()]));
    let mut matcher = GrammarMatcher::new(Arc::new(grammar), tokenizer, Vec::new(), 0);
    assert!(matcher.accept_string("x"));
    assert!(matcher.can_terminate());
}

#[test]
fn shared_compiled_grammar_keeps_stop_tokens_matcher_local() {
    let grammar = Arc::new(Grammar::from_ebnf(r#"root ::= """#, "root").unwrap());
    let tokenizer = Arc::new(Tokenizer::from_vocab(&[
        "x".to_string(),
        "<stop-a>".to_string(),
        "<stop-b>".to_string(),
    ]));
    let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer));

    let mut first = GrammarMatcher::with_compiled(compiled.clone(), vec![1], 0);
    let mut first_mask = vec![0; bitmask_size(tokenizer.vocab_size())];
    first.fill_next_token_bitmask(&mut first_mask);

    let mut second = GrammarMatcher::with_compiled(compiled, vec![2], 0);
    let mut second_mask = vec![0; bitmask_size(tokenizer.vocab_size())];
    second.fill_next_token_bitmask(&mut second_mask);

    assert!(!get_bit(&second_mask, 1));
    assert!(get_bit(&second_mask, 2));
}

#[test]
fn stop_token_rollback_is_one_history_step() {
    let grammar = Arc::new(Grammar::from_ebnf(r#"root ::= "a""#, "root").unwrap());
    let tokenizer = Arc::new(Tokenizer::from_vocab(&[
        "a".to_string(),
        "<eos>".to_string(),
    ]));
    let mut matcher = GrammarMatcher::new(grammar, tokenizer, vec![1], 2);

    assert!(matcher.accept_token(0));
    assert!(matcher.accept_token(1));
    matcher.rollback(1);

    assert!(!matcher.is_terminated());
    assert!(matcher.can_terminate());

    assert!(matcher.accept_token(1));
    matcher.rollback(0);
    assert!(matcher.is_terminated());
}
