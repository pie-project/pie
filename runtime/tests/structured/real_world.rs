//! Real-world production readiness tests.
//!
//! Covers deep nesting, all escape types, large collections, function-calling schemas,
//! comprehensive negative cases, bitmask verification, and mixed-operation edge cases.

use std::sync::Arc;

use pie::structured::bitmask;
use pie::structured::grammar::Grammar;
use pie::structured::json_schema::{json_schema_to_grammar, JsonSchemaOptions};
use pie::structured::matcher::GrammarMatcher;
use pie::structured::tokenizer::{TokenizerInfo, VocabType};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_grammar_accept_string(grammar_ebnf: &str, input: &str) -> bool {
    let grammar = match Grammar::from_ebnf(grammar_ebnf, "root") {
        Ok(g) => g,
        Err(_) => return false,
    };
    is_grammar_accept_string_g(&grammar, input)
}

fn is_grammar_accept_string_g(grammar: &Grammar, input: &str) -> bool {
    let vocab: Vec<String> = vec!["dummy".into()];
    let tok = TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap();
    let mut m = GrammarMatcher::new(Arc::new(grammar.clone()), Arc::new(tok), vec![], 10);

    if input.is_empty() {
        return m.can_terminate();
    }

    if !m.accept_string(input) {
        return false;
    }
    m.can_terminate()
}

fn schema_accepts(schema: &str, input: &str) -> bool {
    let opts = JsonSchemaOptions {
        any_whitespace: false,
        ..JsonSchemaOptions::default()
    };
    let g = json_schema_to_grammar(schema, &opts).unwrap();
    is_grammar_accept_string_g(&g, input)
}

fn make_matcher(ebnf: &str, root: &str, vocab: &[&str]) -> GrammarMatcher {
    let grammar = Arc::new(Grammar::from_ebnf(ebnf, root).unwrap());
    let encoded: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
    let tokenizer = Arc::new(
        TokenizerInfo::new(&encoded, VocabType::Raw, None).unwrap(),
    );
    GrammarMatcher::new(grammar, tokenizer, vec![], 10)
}

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

// ===========================================================================
// 3a. Deep JSON Nesting
// ===========================================================================

#[test]
fn test_json_deeply_nested_10_levels() {
    // {"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":{"j":"leaf"}}}}}}}}}}
    let input = r#"{"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":{"j":"leaf"}}}}}}}}}}"#;
    assert!(is_grammar_accept_string(JSON_GRAMMAR, input));
}

#[test]
fn test_json_deeply_nested_arrays() {
    // [[[[[[[[[[42]]]]]]]]]]
    let mut input = String::new();
    for _ in 0..10 {
        input.push('[');
    }
    input.push_str("42");
    for _ in 0..10 {
        input.push(']');
    }
    assert!(is_grammar_accept_string(JSON_GRAMMAR, &input));
}

// ===========================================================================
// 3b. All JSON Escape Types
// ===========================================================================

#[test]
fn test_json_all_escape_types() {
    // A single string containing every JSON escape sequence
    let input = r#""quote\" backslash\\ slash\/ backspace\b formfeed\f newline\n return\r tab\t unicode\u0041""#;
    assert!(is_grammar_accept_string(JSON_GRAMMAR, input));
}

#[test]
fn test_json_escape_sequences_in_object() {
    let input = r#"{"key\twith\ttabs": "val\nwith\nnewlines", "unicode\u00e9": "backslash\\"}"#;
    assert!(is_grammar_accept_string(JSON_GRAMMAR, input));
}

// ===========================================================================
// 3c. Large Collections
// ===========================================================================

#[test]
fn test_json_large_array_100_elements() {
    let elements: Vec<String> = (0..100).map(|i| i.to_string()).collect();
    let input = format!("[{}]", elements.join(","));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, &input));
}

#[test]
fn test_json_large_object_50_keys() {
    let pairs: Vec<String> = (0..50)
        .map(|i| format!(r#""key{}":"value{}""#, i, i))
        .collect();
    let input = format!("{{{}}}", pairs.join(","));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, &input));
}

// ===========================================================================
// 3d. Very Long Strings
// ===========================================================================

#[test]
fn test_json_long_string_value() {
    let long = "a".repeat(2000);
    let input = format!(r#""{}""#, long);
    assert!(is_grammar_accept_string(JSON_GRAMMAR, &input));
}

#[test]
fn test_json_long_key_and_value() {
    let long_key = "k".repeat(500);
    let long_val = "v".repeat(500);
    let input = format!(r#"{{"{}":"{}"}}"#, long_key, long_val);
    assert!(is_grammar_accept_string(JSON_GRAMMAR, &input));
}

// ===========================================================================
// 3e. Comprehensive Negative Cases
// ===========================================================================

#[test]
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

// ===========================================================================
// 3f. Function Calling / Tool Use Schemas
// ===========================================================================

#[test]
fn test_schema_function_calling_weather() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string", "const": "get_weather"},
            "arguments": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"],
                "additionalProperties": false
            }
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    }"#;

    assert!(schema_accepts(
        schema,
        r#"{"name":"get_weather","arguments":{"location":"San Francisco","unit":"celsius"}}"#,
    ));
    assert!(schema_accepts(
        schema,
        r#"{"name":"get_weather","arguments":{"location":"Tokyo","unit":"fahrenheit"}}"#,
    ));
    // Wrong function name
    assert!(!schema_accepts(
        schema,
        r#"{"name":"get_temperature","arguments":{"location":"NYC","unit":"celsius"}}"#,
    ));
    // Invalid unit
    assert!(!schema_accepts(
        schema,
        r#"{"name":"get_weather","arguments":{"location":"NYC","unit":"kelvin"}}"#,
    ));
}

#[test]
fn test_schema_function_calling_search() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["news", "images", "videos"]},
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string", "format": "date"},
                            "end": {"type": "string", "format": "date"}
                        },
                        "required": ["start", "end"],
                        "additionalProperties": false
                    }
                },
                "required": ["category"],
                "additionalProperties": false
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
        },
        "required": ["query", "filters", "limit"],
        "additionalProperties": false
    }"#;

    assert!(schema_accepts(
        schema,
        r#"{"query":"rust programming","filters":{"category":"news","date_range":{"start":"2024-01-01","end":"2024-12-31"}},"limit":10}"#,
    ));
    assert!(schema_accepts(
        schema,
        r#"{"query":"cats","filters":{"category":"images"},"limit":50}"#,
    ));
}

#[test]
fn test_schema_multi_tool_response() {
    let schema = r#"{
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "tool": {"const": "calculator"},
                    "expression": {"type": "string"}
                },
                "required": ["tool", "expression"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "tool": {"const": "translator"},
                    "text": {"type": "string"},
                    "target_lang": {"type": "string"}
                },
                "required": ["tool", "text", "target_lang"],
                "additionalProperties": false
            }
        ]
    }"#;

    assert!(schema_accepts(
        schema,
        r#"{"tool":"calculator","expression":"2+2"}"#,
    ));
    assert!(schema_accepts(
        schema,
        r#"{"tool":"translator","text":"hello","target_lang":"es"}"#,
    ));
}

// ===========================================================================
// 3g. Complex Real-World Schemas
// ===========================================================================

#[test]
fn test_schema_deeply_nested_3_levels() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "company": {
                "type": "object",
                "properties": {
                    "departments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "employees": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "role": {"type": "string"}
                                        },
                                        "required": ["name", "role"],
                                        "additionalProperties": false
                                    }
                                }
                            },
                            "required": ["name", "employees"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["departments"],
                "additionalProperties": false
            }
        },
        "required": ["company"],
        "additionalProperties": false
    }"#;

    assert!(schema_accepts(
        schema,
        r#"{"company":{"departments":[{"name":"Engineering","employees":[{"name":"Alice","role":"Lead"},{"name":"Bob","role":"Dev"}]},{"name":"Sales","employees":[{"name":"Carol","role":"Manager"}]}]}}"#,
    ));
}

#[test]
fn test_schema_ref_defs_nested() {
    // Uses $defs with $ref (the supported reference pattern)
    let schema = r##"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "children": {
                "type": "array",
                "items": {"$ref": "#/$defs/child"}
            }
        },
        "required": ["name", "children"],
        "additionalProperties": false,
        "$defs": {
            "child": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": false
            }
        }
    }"##;

    // Leaf (empty children)
    assert!(schema_accepts(
        schema,
        r#"{"name":"root","children":[]}"#,
    ));
    // With children
    assert!(schema_accepts(
        schema,
        r#"{"name":"root","children":[{"name":"Alice","age":10},{"name":"Bob","age":8}]}"#,
    ));
}

#[test]
fn test_schema_all_types_combined() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "str_field": {"type": "string"},
            "int_field": {"type": "integer"},
            "num_field": {"type": "number"},
            "bool_field": {"type": "boolean"},
            "null_field": {"type": "null"},
            "arr_field": {"type": "array", "items": {"type": "integer"}},
            "obj_field": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": ["inner"],
                "additionalProperties": false
            },
            "enum_field": {"enum": ["a", "b", 42]}
        },
        "required": ["str_field", "int_field", "num_field", "bool_field", "null_field", "arr_field", "obj_field", "enum_field"],
        "additionalProperties": false
    }"#;

    assert!(schema_accepts(
        schema,
        r#"{"str_field":"hello","int_field":42,"num_field":3.14,"bool_field":true,"null_field":null,"arr_field":[1,2,3],"obj_field":{"inner":"nested"},"enum_field":"a"}"#,
    ));
    assert!(schema_accepts(
        schema,
        r#"{"str_field":"","int_field":-1,"num_field":0,"bool_field":false,"null_field":null,"arr_field":[],"obj_field":{"inner":""},"enum_field":42}"#,
    ));
}

// ===========================================================================
// 3h. Bitmask at Intermediate Positions
// ===========================================================================

#[test]
fn test_bitmask_mid_json_object() {
    // Vocabulary that distinguishes value types after a colon
    let mut m = make_matcher(
        JSON_GRAMMAR,
        "root",
        &["{", "}", "\"", "name", ":", " ", "42", "true", "[", "]"],
    );

    let vocab_size = 10;
    let mut bm = vec![0u32; bitmask::bitmask_size(vocab_size)];

    // Accept {"name":
    assert!(m.accept_string(r#"{"name":"#));

    m.fill_next_token_bitmask(&mut bm);

    // After colon, expect value tokens: "\"", "42", "true", "{", "["
    assert!(bitmask::get_bit(&bm, 2), "quote (string start) should be valid");
    assert!(bitmask::get_bit(&bm, 6), "42 (number) should be valid");
    assert!(bitmask::get_bit(&bm, 7), "true should be valid");
    assert!(bitmask::get_bit(&bm, 0), "{{ (nested object) should be valid");
    assert!(bitmask::get_bit(&bm, 8), "[ (array) should be valid");
    // Should not accept "}" or "]" at this position
    assert!(!bitmask::get_bit(&bm, 1), "}} should not be valid after colon");
    assert!(!bitmask::get_bit(&bm, 9), "] should not be valid after colon");
}

#[test]
fn test_bitmask_after_comma_in_array() {
    let mut m = make_matcher(
        JSON_GRAMMAR,
        "root",
        &["[", "]", ",", "1", "2", "3", " ", "\"", "true"],
    );

    let vocab_size = 9;
    let mut bm = vec![0u32; bitmask::bitmask_size(vocab_size)];

    // Accept [1,
    assert!(m.accept_string("[1,"));

    m.fill_next_token_bitmask(&mut bm);

    // After comma in array, expect value tokens
    assert!(bitmask::get_bit(&bm, 3), "1 should be valid");
    assert!(bitmask::get_bit(&bm, 4), "2 should be valid");
    assert!(bitmask::get_bit(&bm, 5), "3 should be valid");
    assert!(bitmask::get_bit(&bm, 7), "quote (string start) should be valid");
    assert!(bitmask::get_bit(&bm, 8), "true should be valid");
    // Should NOT accept ] immediately after comma (no trailing comma in JSON)
    assert!(!bitmask::get_bit(&bm, 1), "] should not be valid after comma");
}

// ===========================================================================
// 3i. Mixed Operations & Edge Cases
// ===========================================================================

#[test]
fn test_interleave_accept_string_and_token() {
    let mut m = make_matcher(
        JSON_GRAMMAR,
        "root",
        &["{", "}", "\"", "a", ":", "1", ","],
    );

    // Build {"a":1} by interleaving accept_string and accept_token
    assert!(m.accept_string("{\"a\""));  // accept_string for {\"a\"
    assert!(m.accept_token(4));          // accept_token for :
    assert!(m.accept_string("1"));       // accept_string for 1
    assert!(m.accept_token(1));          // accept_token for }
    assert!(m.can_terminate());
}

#[test]
fn test_rollback_across_utf8_boundary() {
    // Grammar that accepts multi-byte UTF-8 characters
    let grammar = "root ::= [\u{4E00}-\u{9FA5}]+";
    let mut m = make_matcher(grammar, "root", &["dummy"]);

    // Accept 你好 (each char is 3 bytes)
    assert!(m.accept_string("\u{4F60}\u{597D}"));
    assert!(m.can_terminate());

    // Rollback one character (3 bytes = 1 token worth, but accept_string is 6 bytes total)
    // We need to rollback 6 bytes to undo the accept_string
    m.rollback(1);  // rollback one "token" (the whole accept_string)

    // Should be able to accept again
    assert!(m.accept_string("\u{4F60}\u{597D}"));
    assert!(m.can_terminate());
}

#[test]
fn test_rollback_deep_then_diverge() {
    let mut m = make_matcher(
        r#"root ::= [a-z]+ ("!" | "?")"#,
        "root",
        &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
          "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "!","?"],
    );

    // Accept 20 letter tokens: a, b, c, ..., t
    for i in 0..20 {
        assert!(m.accept_token(i), "should accept token {}", i);
    }

    // Rollback 15
    m.rollback(15);

    // Take a different path: accept 5 new letters then "!"
    for i in 10..15 {
        assert!(m.accept_token(i), "should accept token {} after rollback", i);
    }
    assert!(m.accept_token(20)); // "!"
    assert!(m.can_terminate());
}

#[test]
fn test_reset_mid_parse_json() {
    let mut m = make_matcher(
        JSON_GRAMMAR,
        "root",
        &["{", "}", "\"", "x", ":", "1"],
    );

    // Start parsing an object
    assert!(m.accept_string(r#"{"x":"#));
    assert!(!m.can_terminate());

    // Reset mid-parse
    m.reset();

    // Should be able to parse a completely different value
    assert!(m.accept_string("42"));
    assert!(m.can_terminate());
}

#[test]
fn test_json_schema_optional_fields() {
    // Schema with 2 optional fields — grammar enforces property ordering (a before b)
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
        ..JsonSchemaOptions::default()
    };
    let g = json_schema_to_grammar(schema, &opts).unwrap();

    // No fields
    assert!(is_grammar_accept_string_g(&g, r#"{}"#));
    // Only a
    assert!(is_grammar_accept_string_g(&g, r#"{"a":42}"#));
    // Both fields (a before b — enforced ordering)
    assert!(is_grammar_accept_string_g(&g, r#"{"a":42,"b":"hello"}"#));
    // Reverse order not accepted (grammar enforces ordering)
    assert!(!is_grammar_accept_string_g(&g, r#"{"b":"hello","a":42}"#));
}

#[test]
fn test_json_number_edge_cases() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    // Valid edge cases
    assert!(is_grammar_accept_string_g(&g, "0"));
    assert!(is_grammar_accept_string_g(&g, "-0"));
    assert!(is_grammar_accept_string_g(&g, "0.0"));
    assert!(is_grammar_accept_string_g(&g, "1e0"));
    assert!(is_grammar_accept_string_g(&g, "1E+0"));
    assert!(is_grammar_accept_string_g(&g, "1e-0"));
    assert!(is_grammar_accept_string_g(&g, "123456789012345678901234567890"));

    // Invalid
    assert!(!is_grammar_accept_string_g(&g, "+1"));
    assert!(!is_grammar_accept_string_g(&g, ".5"));
    assert!(!is_grammar_accept_string_g(&g, "1."));
    assert!(!is_grammar_accept_string_g(&g, "01"));
    assert!(!is_grammar_accept_string_g(&g, "1e"));
}

#[test]
fn test_json_whitespace_variations() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    // Whitespace is allowed: after { and [, before } and ], after comma, around colon in pair
    // Note: the EBNF grammar does NOT allow whitespace before commas (only after)
    assert!(is_grammar_accept_string_g(
        &g,
        r#"{ "a" : 1, "b" : 2 }"#,
    ));
    assert!(is_grammar_accept_string_g(
        &g,
        "{\n\t\"a\"\n:\n1,\n\"b\"\n:\n2\n}",
    ));
    assert!(is_grammar_accept_string_g(
        &g,
        "[\r\n  1,\r\n  2,\r\n  3\r\n]",
    ));
}
