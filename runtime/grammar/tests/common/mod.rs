#![allow(dead_code)]

use std::borrow::Borrow;
use std::sync::Arc;

use pie_grammar::grammar::Grammar;
use pie_grammar::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use pie_grammar::matcher::GrammarMatcher;
use pie_grammar::regex::regex_to_grammar;
use pie_tokenizer::Tokenizer;

pub fn grammar_accepts(grammar: impl Borrow<Grammar>, input: &str) -> bool {
    let tokenizer = Arc::new(Tokenizer::from_vocab(&["dummy".to_string()]));
    let mut matcher = GrammarMatcher::new(
        Arc::new(grammar.borrow().clone()),
        tokenizer,
        Vec::new(),
        10,
    );
    (input.is_empty() || matcher.accept_string(input)) && matcher.can_terminate()
}

pub fn ebnf_accepts(source: &str, input: &str) -> bool {
    Grammar::from_ebnf(source, "root")
        .map(|grammar| grammar_accepts(grammar, input))
        .unwrap_or(false)
}

pub fn regex_accepts(pattern: &str, input: &str) -> bool {
    regex_to_grammar(pattern)
        .map(|grammar| grammar_accepts(grammar, input))
        .unwrap_or(false)
}

pub fn schema_accepts(schema: &str, input: &str) -> bool {
    let grammar = json_schema_to_grammar(schema, &no_whitespace_options()).unwrap();
    grammar_accepts(grammar, input)
}

pub fn no_whitespace_options() -> JsonSchemaOptions {
    JsonSchemaOptions {
        any_whitespace: false,
        ..JsonSchemaOptions::default()
    }
}

pub fn matcher_from_ebnf(ebnf: &str, root: &str, vocab: &[&str]) -> GrammarMatcher {
    matcher_with_stop(ebnf, root, vocab, Vec::new())
}

pub fn matcher_with_stop(
    ebnf: &str,
    root: &str,
    vocab: &[&str],
    stop_ids: Vec<u32>,
) -> GrammarMatcher {
    let grammar = Arc::new(Grammar::from_ebnf(ebnf, root).unwrap());
    let vocab: Vec<String> = vocab.iter().map(|token| (*token).to_string()).collect();
    let tokenizer = Arc::new(Tokenizer::from_vocab(&vocab));
    GrammarMatcher::new(grammar, tokenizer, stop_ids, 10)
}
