use crate::common::{
    ebnf_accepts as is_grammar_accept_string, grammar_accepts as is_grammar_accept_string_g,
    matcher_from_ebnf as make_matcher,
};
use ::grammar::bitmask;
use ::grammar::grammar::Grammar;
use ::grammar::json_schema::{JsonSchemaOptions, json_schema_to_grammar};

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
fn real_world_every_case() {
    test_json_all_escape_types();
    test_json_negative_cases();
    test_bitmask_mid_json_object();
    test_json_schema_optional_fields();
    test_json_number_edge_cases();
}

fn test_json_all_escape_types() {
    let input = r#""quote\" backslash\\ slash\/ backspace\b formfeed\f newline\n return\r tab\t unicode\u0041""#;
    assert!(is_grammar_accept_string(JSON_GRAMMAR, input));
}

fn test_json_negative_cases() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    let cases = [
        (r#"{"key": "val"#, "truncated object"),
        (r#"[1, 2, 3"#, "truncated array"),
        (r#"{"key" "value"}"#, "missing colon"),
        (r#"[1,,2]"#, "double comma"),
        ("042", "leading zero"),
        ("{'key': 'val'}", "single quotes"),
        (r#"{"a":1} extra"#, "trailing content"),
        ("{\"a\":}", "missing value"),
        ("[,1]", "leading comma"),
        ("\"unclosed", "unclosed string"),
        ("{", "single open brace"),
        ("[", "single open bracket"),
        ("{]", "mismatched brackets"),
        ("", "empty input"),
        (" ", "whitespace only"),
    ];

    for (input, desc) in &cases {
        assert!(
            !is_grammar_accept_string_g(&g, input),
            "should reject {}: {:?}",
            desc,
            input
        );
    }
}

fn test_bitmask_mid_json_object() {
    let mut m = make_matcher(
        JSON_GRAMMAR,
        "root",
        &["{", "}", "\"", "name", ":", " ", "42", "true", "[", "]"],
    );

    assert!(m.accept_string(r#"{"name":"#));

    let bm = m.fill_next_token_mask();

    assert!(
        bitmask::get_bit(&bm, 2),
        "quote (string start) should be valid"
    );
    assert!(bitmask::get_bit(&bm, 6), "42 (number) should be valid");
    assert!(bitmask::get_bit(&bm, 7), "true should be valid");
    assert!(
        bitmask::get_bit(&bm, 0),
        "{{ (nested object) should be valid"
    );
    assert!(bitmask::get_bit(&bm, 8), "[ (array) should be valid");
    assert!(
        !bitmask::get_bit(&bm, 1),
        "}} should not be valid after colon"
    );
    assert!(
        !bitmask::get_bit(&bm, 9),
        "] should not be valid after colon"
    );
}

fn test_json_schema_optional_fields() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "string"}
        },
        "additionalProperties": false
    }"#;

    let opts = JsonSchemaOptions {
        any_whitespace: false,
        strict_mode: false,
    };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":42}"#));
    assert!(is_grammar_accept_string_g(&g, r#"{"a":42,"b":"hello"}"#));
    assert!(!is_grammar_accept_string_g(&g, r#"{"b":"hello","a":42}"#));
}

fn test_json_number_edge_cases() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "0"));
    assert!(is_grammar_accept_string_g(&g, "-0"));
    assert!(is_grammar_accept_string_g(&g, "0.0"));
    assert!(is_grammar_accept_string_g(&g, "1e0"));
    assert!(is_grammar_accept_string_g(&g, "1E+0"));
    assert!(is_grammar_accept_string_g(&g, "1e-0"));
    assert!(is_grammar_accept_string_g(
        &g,
        "123456789012345678901234567890"
    ));

    assert!(!is_grammar_accept_string_g(&g, "+1"));
    assert!(!is_grammar_accept_string_g(&g, ".5"));
    assert!(!is_grammar_accept_string_g(&g, "1."));
    assert!(!is_grammar_accept_string_g(&g, "01"));
    assert!(!is_grammar_accept_string_g(&g, "1e"));
}
