use gpugrammar_ir::grammar::Grammar;
use gpugrammar_lex::lexicon::{extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer, group_vocabulary};
use gpugrammar_lr::cfg::flatten;
use gpugrammar_lr::tables::build;
use gpugrammar_run::Matcher;
use std::sync::Arc;

use gpugrammar_tables::{Artifact, emit};

const JSON: &str = r#"
value ::= object | array | string | number
object ::= "{" "}" | "{" members "}"
members ::= pair | pair "," members
pair ::= string ":" value
array ::= "[" "]" | "[" items "]"
items ::= value | value "," items
string ::= "\"" chars
chars ::= "\"" | [^"\\] chars
number ::= "-"? [0-9]+
"#;

/// A byte-per-token vocabulary plus a few multi-byte pieces.
fn vocabulary() -> Vec<Vec<u8>> {
    let mut vocabulary: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for piece in ["\":", "\",", "{\"", "\"}", "123"] {
        vocabulary.push(piece.as_bytes().to_vec());
    }
    vocabulary
}

fn compiled() -> (Artifact, Vec<Vec<u8>>) {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));
    let vocab = vocabulary();
    let groups = group_vocabulary(&lexer, &vocab);
    let cfg = flatten(&lexicon);
    let tables = build(&cfg).unwrap();
    (
        emit(&lexicon, &lexer, &groups, &cfg, &tables, vocab.len()).unwrap(),
        vocab,
    )
}

fn token_of(vocab: &[Vec<u8>], text: &str) -> u32 {
    vocab
        .iter()
        .position(|token| token.as_slice() == text.as_bytes())
        .unwrap_or_else(|| panic!("no token {text:?}")) as u32
}

fn feed(matcher: &mut Matcher, vocab: &[Vec<u8>], text: &str) -> bool {
    for character in text.chars() {
        let token = token_of(vocab, &character.to_string());
        if matcher.accept_token(token).is_err() {
            return false;
        }
    }
    true
}

#[test]
fn a_well_formed_document_is_accepted() {
    let (artifact, vocab) = compiled();
    let artifact = Arc::new(artifact);
    let mut matcher = Matcher::new(artifact.clone(), 8);
    assert!(feed(&mut matcher, &vocab, "{\"a\":1}"));
    assert!(matcher.can_terminate());
}

#[test]
fn a_malformed_document_is_refused_at_the_offending_token() {
    let (artifact, vocab) = compiled();
    let artifact = Arc::new(artifact);
    let mut matcher = Matcher::new(artifact.clone(), 8);
    assert!(feed(&mut matcher, &vocab, "{"));
    // A comma cannot open a member.
    assert!(matcher.accept_token(token_of(&vocab, ",")).is_err());
}

#[test]
fn the_mask_admits_exactly_the_tokens_the_matcher_accepts() {
    let (artifact, vocab) = compiled();
    let artifact = Arc::new(artifact);
    let words = artifact.bitset_words as usize;
    let mut mask = vec![0u32; words];

    for prefix in ["", "{", "{\"", "{\"a", "{\"a\"", "{\"a\":", "{\"a\":1"] {
        let mut matcher = Matcher::new(artifact.clone(), 8);
        assert!(feed(&mut matcher, &vocab, prefix), "prefix {prefix:?}");
        matcher.fill_bitmask(&mut mask);

        for (token, _) in vocab.iter().enumerate() {
            let allowed = mask[token / 32] >> (token % 32) & 1 == 1;
            let mut probe = matcher.clone();
            let accepted = probe.accept_token(token as u32).is_ok();
            assert_eq!(
                allowed,
                accepted,
                "prefix {prefix:?} token {token} ({:?})",
                String::from_utf8_lossy(&vocab[token])
            );
        }
    }
}

#[test]
fn rollback_restores_the_earlier_state() {
    let (artifact, vocab) = compiled();
    let artifact = Arc::new(artifact);
    let mut matcher = Matcher::new(artifact.clone(), 8);
    assert!(feed(&mut matcher, &vocab, "{\"a\""));
    let before = (matcher.lexer_state(), matcher.parser_state());

    assert!(feed(&mut matcher, &vocab, ":1"));
    assert_ne!((matcher.lexer_state(), matcher.parser_state()), before);

    matcher.rollback(2);
    assert_eq!((matcher.lexer_state(), matcher.parser_state()), before);
    assert!(feed(&mut matcher, &vocab, ":2}"));
    assert!(matcher.can_terminate());
}

#[test]
fn termination_is_only_offered_at_a_complete_document() {
    let (artifact, vocab) = compiled();
    let artifact = Arc::new(artifact);
    let mut matcher = Matcher::new(artifact.clone(), 8);
    assert!(!matcher.can_terminate());
    assert!(feed(&mut matcher, &vocab, "{"));
    assert!(!matcher.can_terminate());
    assert!(feed(&mut matcher, &vocab, "}"));
    assert!(matcher.can_terminate());
}

#[test]
fn a_multi_byte_token_advances_several_terminals_at_once() {
    let (artifact, vocab) = compiled();
    let artifact = Arc::new(artifact);
    let mut matcher = Matcher::new(artifact.clone(), 8);
    assert!(feed(&mut matcher, &vocab, "{\"a"));
    // A token that closes the key and supplies the colon at once.
    assert!(matcher.accept_token(token_of(&vocab, "\":")).is_ok());
    assert!(feed(&mut matcher, &vocab, "1}"));
    assert!(matcher.can_terminate());
}
