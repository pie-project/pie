use engrain_ir::grammar::{Grammar, RuleId};
use engrain_lex::regular::analyze;

fn rule_named(grammar: &Grammar, name: &str) -> RuleId {
    RuleId(
        grammar
            .rules()
            .iter()
            .position(|rule| rule.name == name)
            .unwrap() as u32,
    )
}

#[test]
fn a_rule_without_references_is_regular() {
    let grammar = Grammar::from_ebnf(r#"root ::= [a-z]+ "!""#, "root").unwrap();
    let regularity = analyze(&grammar);
    assert!(regularity.is_regular(rule_named(&grammar, "root")));
}

#[test]
fn right_recursion_is_regular() {
    // This shape is exactly how these front ends emit a JSON string body, so
    // rejecting it would push strings into the parser.
    let grammar = Grammar::from_ebnf(
        r#"
root ::= "\"" body
body ::= "\"" | [^"\\] body
"#,
        "root",
    )
    .unwrap();
    let regularity = analyze(&grammar);
    assert!(regularity.is_regular(rule_named(&grammar, "body")));
    assert!(regularity.is_regular(rule_named(&grammar, "root")));
}

#[test]
fn balanced_recursion_is_not_regular() {
    let grammar = Grammar::from_ebnf(
        r#"
root ::= "(" root ")" | ""
"#,
        "root",
    )
    .unwrap();
    let regularity = analyze(&grammar);
    assert!(!regularity.is_regular(rule_named(&grammar, "root")));
}

#[test]
fn a_rule_referencing_a_non_regular_rule_is_not_regular() {
    let grammar = Grammar::from_ebnf(
        r#"
root ::= "<" nested ">"
nested ::= "(" nested ")" | ""
"#,
        "root",
    )
    .unwrap();
    let regularity = analyze(&grammar);
    assert!(!regularity.is_regular(rule_named(&grammar, "nested")));
    assert!(!regularity.is_regular(rule_named(&grammar, "root")));
}

#[test]
fn mutual_right_recursion_is_regular() {
    let grammar = Grammar::from_ebnf(
        r#"
root ::= "a" other | "x"
other ::= "b" root | "y"
"#,
        "root",
    )
    .unwrap();
    let regularity = analyze(&grammar);
    assert!(regularity.is_regular(rule_named(&grammar, "root")));
    assert!(regularity.is_regular(rule_named(&grammar, "other")));
}

#[test]
fn a_json_skeleton_splits_into_lexemes_and_structure() {
    let grammar = Grammar::from_ebnf(
        r#"
value ::= object | array | string | number
object ::= "{" "}" | "{" members "}"
members ::= pair | pair "," members
pair ::= string ":" value
array ::= "[" "]" | "[" items "]"
items ::= value | value "," items
string ::= "\"" chars
chars ::= "\"" | [^"\\] chars
number ::= "-"? [0-9]+
"#,
        "value",
    )
    .unwrap();
    let regularity = analyze(&grammar);

    for lexeme in ["string", "chars", "number"] {
        assert!(
            regularity.is_regular(rule_named(&grammar, lexeme)),
            "{lexeme} should be a lexeme"
        );
    }
    for structural in ["value", "object", "members", "pair", "array", "items"] {
        assert!(
            !regularity.is_regular(rule_named(&grammar, structural)),
            "{structural} should stay in the parser"
        );
    }
}
