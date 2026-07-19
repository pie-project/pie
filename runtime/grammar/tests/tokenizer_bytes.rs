#[path = "../../tokenizer/tests/common/mod.rs"]
mod tokenizer_fixtures;

use std::sync::Arc;

use pie_grammar::bitmask;
use pie_grammar::grammar::Grammar;
use pie_grammar::matcher::GrammarMatcher;
use pie_tokenizer::Tokenizer;
use tokenizer_fixtures::{MergeFormat, byte_level_json};

#[test]
fn matcher_consumes_raw_utf8_across_token_boundaries() {
    let json = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    let tokenizer = Arc::new(json.to_string().parse::<Tokenizer>().unwrap());
    let grammar = Arc::new(Grammar::from_ebnf(r#"root ::= "é""#, "root").unwrap());
    let mut matcher = GrammarMatcher::new(grammar, tokenizer, vec![], 10);

    assert!(matcher.accept_token(0xC3));
    assert!(!matcher.can_terminate());
    assert!(matcher.accept_token(0xA9));
    assert!(matcher.can_terminate());
}

#[test]
fn grammar_mask_excludes_special_token_ids() {
    let json = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    let tokenizer = Arc::new(json.to_string().parse::<Tokenizer>().unwrap());
    let grammar = Arc::new(Grammar::from_ebnf(r#"root ::= "<|special|>""#, "root").unwrap());
    let mut matcher = GrammarMatcher::new(grammar, tokenizer, vec![], 10);
    let mask = matcher.fill_next_token_mask();

    assert!(!bitmask::get_bit(&mask, 260));
}
