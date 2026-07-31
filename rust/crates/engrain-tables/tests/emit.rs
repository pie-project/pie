use engrain_ir::grammar::Grammar;
use engrain_lex::lexicon::{extract, terminal_automata};
use engrain_lex::regular::analyze;
use engrain_lex::{build_lexer, group_vocabulary};
use engrain_lr::cfg::flatten;
use engrain_lr::tables::build;
use engrain_tables::{Artifact, GroupEntry, SetKind, emit};

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

fn vocabulary() -> Vec<Vec<u8>> {
    let mut vocabulary: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for piece in ["\": ", "\", ", "{\"", "\"}", "], ", "1234", "hello world"] {
        vocabulary.push(piece.as_bytes().to_vec());
    }
    vocabulary
}

fn artifact() -> (Artifact, Vec<Vec<u8>>) {
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

/// Is `token` in this group, whichever form the set is stored in?
fn holds(artifact: &engrain_tables::Artifact, group: &GroupEntry, token: usize) -> bool {
    let set = group.set;
    let from = set.offset as usize;
    let body = &artifact.set_payload[from..from + set.length as usize];
    match set.kind {
        SetKind::Sparse => body.binary_search(&(token as u32)).is_ok(),
        SetKind::Complement => body.binary_search(&(token as u32)).is_err(),
        SetKind::Dense => body[token / 32] >> (token % 32) & 1 == 1,
    }
}

#[test]
fn every_group_holds_exactly_its_tokens() {
    // The set is stored sparsely, as a complement, or densely, whichever is
    // smallest; all three have to answer the same question.
    let (artifact, vocab) = artifact();
    let mut counted = 0usize;
    for group in &artifact.groups {
        let members = (0..vocab.len())
            .filter(|token| holds(&artifact, group, *token))
            .count();
        assert_eq!(members, group.token_count as usize);
        counted += members;
    }
    assert!(counted > 0);
}

#[test]
fn a_small_set_is_not_stored_as_a_whole_vocabulary() {
    let (artifact, _) = artifact();
    let words = artifact.bitset_words as usize;
    let dense = artifact.groups.len() * words;
    assert!(
        artifact.set_payload.len() < dense,
        "storage {} is no smaller than {dense} dense rows",
        artifact.set_payload.len()
    );
}

#[test]
fn a_token_belongs_to_one_group_per_lexer_state() {
    let (artifact, vocab) = artifact();
    for state in 0..artifact.num_lexer_states as usize {
        let from = artifact.group_offsets[state] as usize;
        let to = artifact.group_offsets[state + 1] as usize;
        for token in 0..vocab.len() {
            let hits = artifact.groups[from..to]
                .iter()
                .filter(|group| holds(&artifact, group, token))
                .count();
            assert!(
                hits <= 1,
                "token {token} is in {hits} groups of state {state}"
            );
        }
    }
}

#[test]
fn action_rows_are_sorted_so_the_device_can_search_them() {
    let (artifact, _) = artifact();
    for state in 0..artifact.num_parser_states as usize {
        let from = artifact.action_offsets[state] as usize;
        let to = artifact.action_offsets[state + 1] as usize;
        assert!(
            artifact.action_terminals[from..to]
                .windows(2)
                .all(|w| w[0] < w[1]),
            "row {state} is not sorted"
        );
    }
}

#[test]
fn the_group_table_does_not_scale_with_the_vocabulary() {
    let (small, _) = artifact();

    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));
    // Ten times the tokens, built from the same byte alphabet.
    let mut big = vocabulary();
    for repeat in 0..10 {
        for piece in ["alpha", "beta", "gamma", "delta"] {
            big.push(format!("{piece}{repeat}").into_bytes());
        }
    }
    let groups = group_vocabulary(&lexer, &big);
    let cfg = flatten(&lexicon);
    let tables = build(&cfg).unwrap();
    let large = emit(&lexicon, &lexer, &groups, &cfg, &tables, big.len()).unwrap();

    assert!(big.len() > small.vocab_size as usize);
    assert!(
        large.groups.len() <= small.groups.len() + 4,
        "groups grew from {} to {} for {} extra tokens",
        small.groups.len(),
        large.groups.len(),
        big.len() - small.vocab_size as usize
    );
}

#[test]
fn resident_tables_are_far_smaller_than_per_state_rows() {
    let (artifact, _) = artifact();
    assert!(
        artifact.resident_bytes() < artifact.rows_equivalent_bytes(),
        "resident {} vs rows {}",
        artifact.resident_bytes(),
        artifact.rows_equivalent_bytes()
    );
}
