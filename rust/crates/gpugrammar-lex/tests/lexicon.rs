use gpugrammar_ir::grammar::Grammar;
use gpugrammar_lex::lexicon::{SkeletonExpr, extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{START, build_lexer};

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
fn a_whole_regular_rule_is_one_terminal() {
    // `string` is regular, so it becomes a single terminal rather than a
    // quote, a body and a quote. Splitting it would make the body a terminal
    // of its own, and a body class such as `[^"\\]` overlaps every
    // punctuation terminal: after a colon the scanner would keep munching as a
    // string body and never commit the colon the parser wanted.
    let grammar = Grammar::from_ebnf(JSON, "value").unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let names: Vec<&str> = lexicon
        .terminals
        .iter()
        .map(|terminal| terminal.name.as_str())
        .collect();
    assert!(
        names.contains(&"string"),
        "the string never became one terminal: {names:?}"
    );
    // Adjacent literals in a sequence stay separate, because a non-regular
    // neighbour breaks the run, so the skeleton still has its punctuation.
    for literal in ["'{'", "'}'", "','", "':'"] {
        assert!(names.contains(&literal), "missing {literal} in {names:?}");
    }
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

    // `object ::= "{" "}"` is regular, so `{}` is a terminal too, and a lone
    // `{` is both a complete terminal and a prefix of it. The scan offers both
    // readings and leaves the choice to the parser; carrying the lexeme comes
    // first, because longest match wins whenever the parser can follow it.
    let opening = lexer.scan(b"{", START).unwrap();
    assert!(
        opening.options[0].terminals.is_empty(),
        "the longest reading should carry the lexeme"
    );
    assert!(
        lexer
            .reachable_terminals(opening.options[0].next_state)
            .iter()
            .any(|terminal| lexer.terminal_name(*terminal) == "'{'")
    );
    assert!(
        opening.options.iter().any(|option| option
            .terminals
            .iter()
            .any(|terminal| lexer.terminal_name(*terminal) == "'{'")),
        "no reading settles the brace"
    );

    // Once the next byte cannot extend it, the longest match is committed.
    let member = lexer.scan(b"{\"", START).unwrap();
    assert_eq!(member.options.len(), 1);
    assert_eq!(
        member.options[0]
            .terminals
            .iter()
            .map(|terminal| lexer.terminal_name(*terminal))
            .collect::<Vec<_>>(),
        vec!["'{'"]
    );

    let string = lexer.scan(b"\"ab", START).unwrap();
    assert!(
        string
            .options
            .iter()
            .all(|option| option.terminals.is_empty())
    );
    assert_ne!(string.options[0].next_state, START);

    assert!(lexer.scan(b"@", START).is_none());
}
