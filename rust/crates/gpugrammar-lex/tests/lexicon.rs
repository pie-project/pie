use gpugrammar_ir::grammar::Grammar;
use gpugrammar_lex::lexicon::{SkeletonExpr, extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{START, build_lexer, group_vocabulary};

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

#[test]
fn structural_literals_become_terminals() {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let names: Vec<&str> = lexicon
        .terminals
        .iter()
        .map(|terminal| terminal.name.as_str())
        .collect();
    for literal in ["'{'", "'}'", "'['", "']'", "','", "':'"] {
        assert!(names.contains(&literal), "missing {literal} in {names:?}");
    }
}

#[test]
fn a_repeated_literal_interns_to_one_terminal() {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let commas = lexicon
        .terminals
        .iter()
        .filter(|terminal| terminal.name == "','")
        .count();
    assert_eq!(
        commas, 1,
        "the comma appears in two rules but is one terminal"
    );
}

#[test]
fn the_skeleton_keeps_only_structural_rules() {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let names: Vec<&str> = lexicon
        .skeleton
        .iter()
        .map(|rule| rule.name.as_str())
        .collect();
    for structural in ["value", "object", "members", "pair", "array", "items"] {
        assert!(names.contains(&structural), "missing {structural}");
    }
    for lexeme in ["string", "chars", "number"] {
        assert!(!names.contains(&lexeme), "{lexeme} should have been lexed");
    }
}

#[test]
fn adjacent_literals_stay_separate_terminals() {
    // Merging them would make a lone "{" ambiguous: the scanner could not
    // commit without seeing whether a "}" follows.
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    assert!(
        !lexicon
            .terminals
            .iter()
            .any(|terminal| terminal.name.contains("{}")),
        "found a merged terminal in {:?}",
        lexicon
            .terminals
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_skeleton_body_mixes_terminals_and_nonterminals() {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let pair = lexicon
        .skeleton
        .iter()
        .find(|rule| rule.name == "pair")
        .unwrap();
    // Rule bodies are always a choice of alternatives, even when there is one.
    let SkeletonExpr::Choice(alternatives) = &pair.body else {
        panic!("pair should be a choice, got {:?}", pair.body);
    };
    assert_eq!(alternatives.len(), 1);
    let SkeletonExpr::Sequence(parts) = &alternatives[0] else {
        panic!(
            "the alternative should be a sequence, got {:?}",
            alternatives[0]
        );
    };
    assert!(matches!(parts[0], SkeletonExpr::Terminal(_)));
    assert!(matches!(parts[1], SkeletonExpr::Terminal(_)));
    assert!(matches!(parts[2], SkeletonExpr::Nonterminal(_)));
}

#[test]
fn the_extracted_lexicon_drives_a_working_lexer() {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));

    let scan = lexer.scan(b"{", START).unwrap();
    assert_eq!(scan.terminals.len(), 1);
    assert_eq!(lexer.terminal_name(scan.terminals[0]), "'{'");

    let opening = lexer.scan(b"\"ab", START).unwrap();
    assert!(opening.terminals.is_empty());
    assert_ne!(opening.next_state, START);

    assert!(lexer.scan(b"@", START).is_none());
}

#[test]
fn grouping_over_the_extracted_lexer_collapses_the_vocabulary() {
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));

    let mut vocabulary: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for piece in ["\": ", "\", ", "{\"", "\"}", "], ", "1234", "hello world"] {
        vocabulary.push(piece.as_bytes().to_vec());
    }
    let groups = group_vocabulary(&lexer, &vocabulary);
    let at_start = &groups.per_state[START.0 as usize];
    assert!(
        at_start.len() * 8 < vocabulary.len(),
        "expected at least an 8x collapse, got {} groups",
        at_start.len()
    );
}
