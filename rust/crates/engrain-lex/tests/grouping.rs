use engrain_ir::fsm::build_rule_fsms;
use engrain_ir::grammar::Grammar;
use engrain_lex::{LexState, START, Terminal, build_lexer, group_vocabulary};

/// Build a lexer whose terminals are the named rules of a regular grammar.
fn lexer_from_rules(source: &str, order: &[&str]) -> engrain_lex::Lexer {
    let grammar = Grammar::from_ebnf(source, order[0]).unwrap();
    let automata = build_rule_fsms(&grammar);
    let mut terminals = Vec::new();
    for name in order {
        let index = grammar
            .rules()
            .iter()
            .position(|rule| rule.name == *name)
            .unwrap();
        terminals.push(Terminal {
            name: (*name).to_string(),
            automaton: automata[index].clone(),
        });
    }
    build_lexer(terminals)
}

const JSON_LEXER: &str = r#"
lbrace ::= "{"
rbrace ::= "}"
lbracket ::= "["
rbracket ::= "]"
comma ::= ","
colon ::= ":"
string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
literal ::= "true" | "false" | "null"
ws ::= [ \t\n\r]+
"#;

const ORDER: &[&str] = &[
    "lbrace", "rbrace", "lbracket", "rbracket", "comma", "colon", "string", "number", "literal",
    "ws",
];

/// Names of the terminals each reading emits.
fn readings<'a>(
    lexer: &'a engrain_lex::Lexer,
    scan: &engrain_lex::Scan,
) -> Vec<Vec<&'a str>> {
    scan.options
        .iter()
        .map(|option| {
            option
                .terminals
                .iter()
                .map(|terminal| lexer.terminal_name(*terminal))
                .collect()
        })
        .collect()
}

#[test]
fn a_structural_token_can_be_read_as_one_terminal() {
    // A scan offers every reading, because where a lexeme ends is a choice the
    // parser has to make: `{` is a complete terminal, and it is also a prefix
    // of longer ones. Carrying it comes first, since longest match wins
    // whenever the parser can follow it, and settling it is offered too.
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    let scan = lexer.scan(b"{", START).unwrap();
    let all = readings(&lexer, &scan);
    assert!(
        all.contains(&vec!["lbrace"]),
        "no reading emits lbrace: {all:?}"
    );
    let settled = scan
        .options
        .iter()
        .find(|option| option.terminals.len() == 1)
        .expect("a settled reading");
    assert_eq!(settled.next_state, START);
}

#[test]
fn a_token_spanning_several_terminals_emits_all_of_them() {
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    let scan = lexer.scan(b"\"a\":", START).unwrap();
    let all = readings(&lexer, &scan);
    assert!(
        all.contains(&vec!["string", "colon"]),
        "no reading emits both: {all:?}"
    );
}

#[test]
fn a_token_ending_mid_lexeme_carries_the_state() {
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    let opening = lexer.scan(b"\"ab", START).unwrap();
    assert!(
        opening
            .options
            .iter()
            .all(|option| option.terminals.is_empty())
    );
    assert_ne!(opening.options[0].next_state, START);

    let closing = lexer.scan(b"cd\"", opening.options[0].next_state).unwrap();
    let all = readings(&lexer, &closing);
    assert!(
        all.contains(&vec!["string"]),
        "the string was never completed: {all:?}"
    );
}

#[test]
fn a_lexically_impossible_token_is_rejected() {
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    assert!(lexer.scan(b"@", START).is_none());
}

#[test]
fn the_vocabulary_collapses_into_few_groups() {
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    // A synthetic vocabulary shaped like a byte-level BPE: every single byte
    // plus some common multi-byte pieces.
    let mut vocabulary: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for piece in [
        "\": ",
        "\", ",
        "{\"",
        "\"}",
        "], ",
        "true",
        "false",
        "null",
        "abc",
        "hello world",
        "1234",
        "-1.5",
    ] {
        vocabulary.push(piece.as_bytes().to_vec());
    }
    let groups = group_vocabulary(&lexer, &vocabulary);

    let at_start = &groups.per_state[START.0 as usize];
    assert!(
        at_start.len() < vocabulary.len() / 4,
        "expected a large collapse, got {} groups for {} tokens",
        at_start.len(),
        vocabulary.len()
    );

    let placed: usize = at_start.iter().map(|g| g.tokens.len()).sum();
    assert_eq!(
        placed + groups.rejected[START.0 as usize] as usize,
        vocabulary.len()
    );

    // Inside a string most of the legal vocabulary behaves identically, which
    // is the wide-row case the GPU side sees. Note the denominator: character
    // classes are Unicode codepoint ranges compiled to UTF-8, so a lone
    // continuation byte is not legal anywhere. Byte-level BPE vocabularies do
    // contain such fragments, and handling them is a known gap.
    let inside = lexer.scan(b"\"", START).unwrap().options[0].next_state;
    let in_string = &groups.per_state[inside.0 as usize];
    let accepted: usize = in_string.iter().map(|g| g.tokens.len()).sum();
    assert!(
        in_string[0].tokens.len() * 2 > accepted,
        "largest in-string group {} of {} accepted tokens in {} groups",
        in_string[0].tokens.len(),
        accepted,
        in_string.len()
    );
    assert!(
        in_string.len() * 8 < vocabulary.len(),
        "expected at least an 8x collapse, got {} groups for {} tokens",
        in_string.len(),
        vocabulary.len()
    );
}

#[test]
fn scanning_is_total_over_every_state() {
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    for state in 0..lexer.num_states() {
        let _ = lexer.scan(b"x", LexState(state as u32));
    }
}

#[test]
fn a_codepoint_split_across_tokens_is_scannable() {
    // Byte-level BPE vocabularies contain pieces that are only part of a UTF-8
    // sequence. Character classes compile to UTF-8 byte paths, so the halves
    // must meet in an intermediate lexer state rather than being rejected.
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    let inside = lexer.scan(b"\"", START).unwrap().options[0].next_state;

    let snowman = "☃".as_bytes();
    assert_eq!(snowman.len(), 3);

    // The whole codepoint in one token stays inside the string.
    let whole = lexer.scan(snowman, inside).expect("whole codepoint");
    assert!(
        whole
            .options
            .iter()
            .all(|option| option.terminals.is_empty())
    );

    // Split after the lead byte: the first half must land in a state that the
    // second half can continue from.
    let first = lexer.scan(&snowman[..1], inside).expect("lead byte");
    assert_ne!(first.options[0].next_state, inside);
    let second = lexer
        .scan(&snowman[1..], first.options[0].next_state)
        .expect("continuation bytes");
    assert_eq!(second.options[0].next_state, whole.options[0].next_state);
}

#[test]
fn a_bare_continuation_byte_is_only_legal_mid_codepoint() {
    let lexer = lexer_from_rules(JSON_LEXER, ORDER);
    let inside = lexer.scan(b"\"", START).unwrap().options[0].next_state;
    let snowman = "☃".as_bytes();

    assert!(lexer.scan(&snowman[1..2], inside).is_none());
    let first = lexer.scan(&snowman[..1], inside).unwrap();
    assert!(
        lexer
            .scan(&snowman[1..2], first.options[0].next_state)
            .is_some()
    );
}
