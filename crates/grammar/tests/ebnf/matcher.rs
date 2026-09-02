//! EBNF matcher acceptance behavior, ported from xgrammar.
//!
//! Tests EBNF grammar acceptance/rejection via the GrammarMatcher.

use crate::common::{
    ebnf_accepts as is_grammar_accept_string, grammar_accepts as is_grammar_accept_string_g,
};
use ::grammar::grammar::Grammar;

// ---------------------------------------------------------------------------
// JSON acceptance (from test_json_pressure / test_json_grammar)
// ---------------------------------------------------------------------------

const JSON_GRAMMAR: &str = r#"
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= ws string ws ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
"#;

#[test]
fn test_json_complex() {
    let complex_json = r#"{
    "web-app": {
    "servlet": [
        {
        "servlet-name": "cofaxCDS",
        "servlet-class": "org.cofax.cds.CDSServlet",
        "init-param": {
            "configGlossary:installationAt": "Philadelphia, PA",
            "configGlossary:adminEmail": "ksm@pobox.com",
            "useJSP": false,
            "cachePackageTagsTrack": 200,
            "useDataStore": true
        }
        }
    ],
    "servlet-mapping": {
        "cofaxCDS": "/",
        "cofaxEmail": "/cofaxutil/aemail/*"
    }
    }
}"#;
    assert!(is_grammar_accept_string(JSON_GRAMMAR, complex_json));
}

// ---------------------------------------------------------------------------
// Nullable grammar (from test_nullable_grammar)
// ---------------------------------------------------------------------------

#[test]
fn test_nullable_grammar() {
    let grammar = r#"
    root ::= rule1 | (rule1 rule1 rule1 rule3)+
    rule1 ::= rule2
    rule2 ::= [0-9]*
    rule3 ::= [a-z]
"#;
    // Empty string accepted (rule2 is [0-9]* which matches empty)
    assert!(is_grammar_accept_string(grammar, ""));
    // Mixed string accepted
    assert!(is_grammar_accept_string(grammar, "abc12312398014a"));
}

// ---------------------------------------------------------------------------
// Predict/Complete (from test_predict_complete)
// ---------------------------------------------------------------------------

#[test]
fn test_predict_complete_complex() {
    let grammar = r#"root ::= rule1 [0-9]?
    rule1 ::= rule2 [0-9]? | rule4 [0-9]?
    rule2 ::= rule3 [0-9]? | rule2 [0-9]? | rule1 [0-9]?
    rule3 ::= rule4 [0-9]? | rule5 [0-9]?
    rule4 ::= rule5 [0-9]? | rule6 [0-9]?
    rule5 ::= rule6 [0-9]? | rule7 [0-9]? | rule8 [0-9]?
    rule6 ::= rule7 [0-9]? | rule1 [0-9]?
    rule7 ::= rule8 [0-9]? | rule9 [0-9]?
    rule8 ::= rule9 [0-9]? | rule7 [0-9]?
    rule9 ::= [0-9]?
    "#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    // Empty string through strings of increasing length
    let mut input = String::new();
    for _ in 0..=10 {
        assert!(
            is_grammar_accept_string_g(&g, &input),
            "should accept {:?}",
            input
        );
        input.push('0');
    }
    assert!(is_grammar_accept_string_g(&g, &input));
}

// ---------------------------------------------------------------------------
// Advance (from test_advance)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// UTF-8 tests (from test_character_class_star_utf8, test_positive_utf8_*)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// UTF-8 with quantifiers (from test_positive_utf8_character_class_with_quantifier)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// NFA test (from test_nfa)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Non-neighbor character class (from test_not_neighbour_character_class)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Repetition tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Complex rule interactions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Simple rule interaction (from test_simple)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Complex repetition (from test_repetition)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// JSON acceptance: more complex inputs (from test_json_accept)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// JSON rejection (from test_json_refuse)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// JSON pressure test (from test_json_pressure)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// UTF-8 comma character class (from test_utf8)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Custom root rule (from test_custom_root_rule)
// ---------------------------------------------------------------------------

