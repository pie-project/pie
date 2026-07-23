use crate::common::grammar_accepts;
use pie_grammar::grammar::Grammar;
use pie_grammar::regex::regex_to_grammar;

#[test]
fn unicode_literals_quantifiers_and_classes() {
    let grammar = regex_to_grammar("é").unwrap();
    assert!(grammar_accepts(grammar.clone(), "é"));
    assert!(!grammar_accepts(grammar, "Ã©"));

    let quantified = regex_to_grammar("é+").unwrap();
    assert!(grammar_accepts(quantified.clone(), "éé"));
    assert!(!grammar_accepts(quantified, "ee"));

    let mixed = regex_to_grammar("aé+").unwrap();
    assert!(grammar_accepts(mixed.clone(), "aéé"));
    assert!(!grammar_accepts(mixed, "aa"));

    let character_class = regex_to_grammar("[é-ê]+").unwrap();
    assert!(grammar_accepts(character_class.clone(), "éêé"));
    assert!(!grammar_accepts(character_class, "e"));
}

#[test]
fn exact_zero_repetition_matches_empty_string() {
    let grammar = Grammar::from_ebnf(r#"root ::= "a"{0}"#, "root").unwrap();
    assert!(grammar_accepts(grammar, ""));
    assert!(!grammar_accepts(
        Grammar::from_ebnf(r#"root ::= "a"{0}"#, "root").unwrap(),
        "a"
    ));

    let concatenated = Grammar::from_ebnf(r#"root ::= "b" "a"{0,0} "c""#, "root").unwrap();
    assert!(grammar_accepts(concatenated.clone(), "bc"));
    assert!(!grammar_accepts(concatenated, "bac"));
}
