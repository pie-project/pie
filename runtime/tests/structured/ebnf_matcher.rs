//! Ported from xgrammar: test_grammar_matcher_ebnf.py
//!
//! Tests EBNF grammar acceptance/rejection via the GrammarMatcher.

use std::sync::Arc;

use pie::structured::grammar::Grammar;
use pie::structured::matcher::GrammarMatcher;
use pie::structured::tokenizer::{TokenizerInfo, VocabType};

/// Helper: does the grammar accept the given string?
/// Mirrors xgrammar's `_is_grammar_accept_string`.
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
fn test_json_simple_object() {
    assert!(is_grammar_accept_string(
        JSON_GRAMMAR,
        r#"{"name": "John", "age": 30}"#
    ));
}

#[test]
fn test_json_nested() {
    assert!(is_grammar_accept_string(
        JSON_GRAMMAR,
        r#"{"a": {"b": [1, 2, 3]}}"#
    ));
}

#[test]
fn test_json_array() {
    assert!(is_grammar_accept_string(JSON_GRAMMAR, r#"[1, 2, 3]"#));
}

#[test]
fn test_json_boolean_null() {
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "true"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "false"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "null"));
}

#[test]
fn test_json_string_escapes() {
    assert!(is_grammar_accept_string(
        JSON_GRAMMAR,
        r#""hello\nworld""#
    ));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, r#""tab\there""#));
    assert!(is_grammar_accept_string(
        JSON_GRAMMAR,
        r#""unicode\u0041""#
    ));
}

#[test]
fn test_json_numbers() {
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "0"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "42"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "-17"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "3.14"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "1e10"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "2.5E-3"));
}

#[test]
fn test_json_empty_collections() {
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "{}"));
    assert!(is_grammar_accept_string(JSON_GRAMMAR, "[]"));
}

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

#[test]
fn test_right_recursion() {
    let grammar = "root ::= [a-z] root | [a-z]";
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    let accept = ["a", "ab", "abc", "abcd", "abcde"];
    let reject = ["", "1", "a1", "ab1", "abc1"];

    for s in &accept {
        assert!(
            is_grammar_accept_string_g(&g, s),
            "should accept {:?}",
            s
        );
    }
    for s in &reject {
        assert!(
            !is_grammar_accept_string_g(&g, s),
            "should reject {:?}",
            s
        );
    }
}

#[test]
fn test_balanced_braces() {
    let grammar = r#"root ::= rule1
    rule1 ::= "{" rule2 | ""
    rule2 ::= root "}"
    "#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    let accept = ["", "{}", "{{}}", "{{{}}}", "{{{{}}}}"];
    let reject = ["{", "{}{}", "{{{{}", "{{}}}", "{{{{{}}}}}}"];

    for s in &accept {
        assert!(
            is_grammar_accept_string_g(&g, s),
            "should accept {:?}",
            s
        );
    }
    for s in &reject {
        assert!(
            !is_grammar_accept_string_g(&g, s),
            "should reject {:?}",
            s
        );
    }
}

// ---------------------------------------------------------------------------
// Advance (from test_advance)
// ---------------------------------------------------------------------------

#[test]
fn test_advance_repeated_a() {
    let grammar = r#"root ::= rule1
    rule1 ::= [a] | [a-b] | [a-c]* | "a" | "aaaaaaaaaaaaaaaaaaa"
    "#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    for i in 0..10 {
        let input = "a".repeat(i);
        assert!(
            is_grammar_accept_string_g(&g, &input),
            "should accept {} 'a's",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// UTF-8 tests (from test_character_class_star_utf8, test_positive_utf8_*)
// ---------------------------------------------------------------------------

#[test]
fn test_character_class_star_utf8() {
    let grammar = r#"root ::= [^0-9]*"#;
    // "worldせかい世界" — all non-digit characters
    assert!(is_grammar_accept_string(
        grammar,
        "world\u{305b}\u{304b}\u{3044}\u{4e16}\u{754c}"
    ));
}

#[test]
fn test_cyrillic_char_class() {
    // Cyrillic lowercase а-я (U+0430 to U+044F)
    let grammar = "root ::= [\u{0430}-\u{044F}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    // Single Cyrillic characters
    assert!(is_grammar_accept_string_g(&g, "\u{0430}")); // а
    assert!(is_grammar_accept_string_g(&g, "\u{044F}")); // я
    assert!(is_grammar_accept_string_g(&g, "\u{043F}")); // п

    // Multiple Cyrillic characters: "привет"
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}"
    ));
    // "абвгд"
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{0430}\u{0431}\u{0432}\u{0433}\u{0434}"
    ));

    // Reject ASCII
    assert!(!is_grammar_accept_string_g(&g, "hello"));
    assert!(!is_grammar_accept_string_g(&g, "123"));
    assert!(!is_grammar_accept_string_g(&g, ""));
}

#[test]
fn test_cyrillic_uppercase_char_class() {
    // Uppercase Cyrillic А-Я (U+0410 to U+042F)
    let grammar = "root ::= [\u{0410}-\u{042F}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "\u{0410}")); // А
    assert!(is_grammar_accept_string_g(&g, "\u{042F}")); // Я
    // "ПРИВЕТ"
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{041F}\u{0420}\u{0418}\u{0412}\u{0415}\u{0422}"
    ));
    // Reject lowercase
    assert!(!is_grammar_accept_string_g(
        &g,
        "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}"
    ));
}

#[test]
fn test_cjk_char_class() {
    // CJK Unified Ideographs subset: 一-龥 (U+4E00 to U+9FA5)
    let grammar = "root ::= [\u{4E00}-\u{9FA5}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "\u{4E00}")); // 一
    assert!(is_grammar_accept_string_g(&g, "\u{4E2D}")); // 中
    assert!(is_grammar_accept_string_g(&g, "\u{9FA5}")); // 龥

    // "你好"
    assert!(is_grammar_accept_string_g(&g, "\u{4F60}\u{597D}"));
    // "世界"
    assert!(is_grammar_accept_string_g(&g, "\u{4E16}\u{754C}"));
    // "中文测试"
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{4E2D}\u{6587}\u{6D4B}\u{8BD5}"
    ));

    // Reject ASCII and Cyrillic
    assert!(!is_grammar_accept_string_g(&g, "hello"));
    assert!(!is_grammar_accept_string_g(
        &g,
        "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}"
    ));
    assert!(!is_grammar_accept_string_g(&g, ""));
}

#[test]
fn test_hiragana_char_class() {
    // Hiragana: あ-ん (U+3041 to U+3093)
    let grammar = "root ::= [\u{3041}-\u{3093}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "\u{3041}")); // あ
    assert!(is_grammar_accept_string_g(&g, "\u{3093}")); // ん
    // "こんにちは"
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}"
    ));
    // Reject Kanji
    assert!(!is_grammar_accept_string_g(&g, "\u{6F22}\u{5B57}"));
}

#[test]
fn test_emoji_char_class() {
    // Emoji range: 😀-😿 (U+1F600 to U+1F63F)
    let grammar = "root ::= [\u{1F600}-\u{1F63F}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "\u{1F600}")); // 😀
    assert!(is_grammar_accept_string_g(&g, "\u{1F603}")); // 😃
    assert!(is_grammar_accept_string_g(&g, "\u{1F63F}")); // 😿

    // Multiple emojis
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{1F600}\u{1F603}\u{1F604}"
    ));

    // Reject other characters
    assert!(!is_grammar_accept_string_g(&g, "hello"));
    assert!(!is_grammar_accept_string_g(&g, "\u{1F30D}")); // 🌍 different range
    assert!(!is_grammar_accept_string_g(&g, ""));
}

#[test]
fn test_mixed_utf8_ranges() {
    // ASCII + Cyrillic + CJK
    let grammar = "root ::= [a-z\u{0430}-\u{044F}\u{4E00}-\u{9FA5}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "hello"));
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}"
    )); // привет
    assert!(is_grammar_accept_string_g(&g, "\u{4F60}\u{597D}")); // 你好

    // Mixed content
    assert!(is_grammar_accept_string_g(
        &g,
        "hello\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}\u{4F60}\u{597D}"
    ));

    // Reject uppercase ASCII and digits
    assert!(!is_grammar_accept_string_g(&g, "HELLO"));
    assert!(!is_grammar_accept_string_g(&g, "123"));
}

#[test]
fn test_single_utf8_char_class() {
    // Single Cyrillic character (not a range)
    let grammar = "root ::= [\u{0430}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "\u{0430}")); // а
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{0430}\u{0430}\u{0430}"
    )); // ааа
    assert!(!is_grammar_accept_string_g(&g, "\u{0431}")); // б
    assert!(!is_grammar_accept_string_g(&g, "a")); // ASCII 'a' != Cyrillic 'а'

    // Single CJK character
    let grammar_cjk = "root ::= [\u{4E2D}]+";
    let g_cjk = Grammar::from_ebnf(&grammar_cjk, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g_cjk, "\u{4E2D}")); // 中
    assert!(is_grammar_accept_string_g(
        &g_cjk,
        "\u{4E2D}\u{4E2D}\u{4E2D}"
    ));
    assert!(!is_grammar_accept_string_g(&g_cjk, "\u{56FD}")); // 国
}

// ---------------------------------------------------------------------------
// UTF-8 with quantifiers (from test_positive_utf8_character_class_with_quantifier)
// ---------------------------------------------------------------------------

#[test]
fn test_utf8_char_class_with_quantifier() {
    let grammar = "root ::= [a-z\u{0430}-\u{044F}\u{4E00}-\u{9FA5}]{0,2048}";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    // Empty string (min is 0)
    assert!(is_grammar_accept_string_g(&g, ""));

    // Individual types
    assert!(is_grammar_accept_string_g(&g, "hello"));
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}"
    ));
    assert!(is_grammar_accept_string_g(
        &g,
        "\u{4F60}\u{597D}\u{4E16}\u{754C}"
    ));

    // Mixed
    assert!(is_grammar_accept_string_g(
        &g,
        "hello\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}\u{4F60}\u{597D}"
    ));

    // Long strings
    let long_a = "a".repeat(100);
    assert!(is_grammar_accept_string_g(&g, &long_a));

    let long_cyrillic = "\u{044F}".repeat(100);
    assert!(is_grammar_accept_string_g(&g, &long_cyrillic));

    let long_cjk = "\u{4E2D}".repeat(100);
    assert!(is_grammar_accept_string_g(&g, &long_cjk));

    // Reject
    assert!(!is_grammar_accept_string_g(&g, "HELLO"));
    assert!(!is_grammar_accept_string_g(&g, "123"));
    assert!(!is_grammar_accept_string_g(&g, "hello!"));
}

// ---------------------------------------------------------------------------
// NFA test (from test_nfa)
// ---------------------------------------------------------------------------

#[test]
fn test_nfa_grammar() {
    let grammar = r#"
root ::= rule1 | rule2 | rule3
rule1 ::= "abc" | ""
rule2 ::= "abd" | ""
rule3 ::= [a-n] [b-c] "x" | ""
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "abc"));
    assert!(is_grammar_accept_string_g(&g, "abx"));
    assert!(is_grammar_accept_string_g(&g, "ccx"));
    assert!(!is_grammar_accept_string_g(&g, "abb"));
    assert!(!is_grammar_accept_string_g(&g, "ad"));
}

// ---------------------------------------------------------------------------
// Non-neighbor character class (from test_not_neighbour_character_class)
// ---------------------------------------------------------------------------

#[test]
fn test_non_neighbor_char_class() {
    let grammar = "root ::= [a-cx-z]*";
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    // Accepts chars in [a-c] and [x-z]
    assert!(is_grammar_accept_string_g(&g, ""));
    assert!(is_grammar_accept_string_g(&g, "a"));
    assert!(is_grammar_accept_string_g(&g, "abc"));
    assert!(is_grammar_accept_string_g(&g, "xyz"));
    assert!(is_grammar_accept_string_g(&g, "axbycz"));

    // Rejects chars not in range
    assert!(!is_grammar_accept_string_g(&g, "d"));
    assert!(!is_grammar_accept_string_g(&g, "m"));
    assert!(!is_grammar_accept_string_g(&g, "w"));
}

// ---------------------------------------------------------------------------
// Repetition tests
// ---------------------------------------------------------------------------

#[test]
fn test_repetition_exact() {
    let grammar = r#"root ::= "a"{3}"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(!is_grammar_accept_string_g(&g, "aa"));
    assert!(is_grammar_accept_string_g(&g, "aaa"));
    assert!(!is_grammar_accept_string_g(&g, "aaaa"));
}

#[test]
fn test_repetition_range() {
    let grammar = r#"root ::= "a"{2,4}"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(!is_grammar_accept_string_g(&g, "a"));
    assert!(is_grammar_accept_string_g(&g, "aa"));
    assert!(is_grammar_accept_string_g(&g, "aaa"));
    assert!(is_grammar_accept_string_g(&g, "aaaa"));
    assert!(!is_grammar_accept_string_g(&g, "aaaaa"));
}

#[test]
fn test_repetition_unbounded() {
    let grammar = r#"root ::= "a"{2,}"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(!is_grammar_accept_string_g(&g, ""));
    assert!(!is_grammar_accept_string_g(&g, "a"));
    assert!(is_grammar_accept_string_g(&g, "aa"));
    assert!(is_grammar_accept_string_g(&g, "aaa"));
    assert!(is_grammar_accept_string_g(&g, "a".repeat(100).as_str()));
}

// ---------------------------------------------------------------------------
// Complex rule interactions
// ---------------------------------------------------------------------------

#[test]
fn test_optional_multiple_branches() {
    let grammar = r#"
root ::= "a" rest
rest ::= "b" rest | "c" | ""
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "a"));
    assert!(is_grammar_accept_string_g(&g, "ac"));
    assert!(is_grammar_accept_string_g(&g, "abc"));
    assert!(is_grammar_accept_string_g(&g, "abbc"));
    assert!(is_grammar_accept_string_g(&g, "ab"));
    assert!(!is_grammar_accept_string_g(&g, ""));
    assert!(!is_grammar_accept_string_g(&g, "b"));
}

#[test]
fn test_deeply_nested_rules() {
    let grammar = r#"
root ::= a
a ::= b | "x"
b ::= c | "y"
c ::= d | "z"
d ::= "w"
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "w"));
    assert!(is_grammar_accept_string_g(&g, "x"));
    assert!(is_grammar_accept_string_g(&g, "y"));
    assert!(is_grammar_accept_string_g(&g, "z"));
    assert!(!is_grammar_accept_string_g(&g, "a"));
    assert!(!is_grammar_accept_string_g(&g, ""));
}

#[test]
fn test_left_recursion_via_rules() {
    // Indirect left recursion through rule chain, with termination
    let grammar = r#"
root ::= item+
item ::= [a-z] | [0-9]
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "a"));
    assert!(is_grammar_accept_string_g(&g, "abc123"));
    assert!(is_grammar_accept_string_g(&g, "9"));
    assert!(!is_grammar_accept_string_g(&g, ""));
    assert!(!is_grammar_accept_string_g(&g, "ABC"));
}

#[test]
fn test_negated_char_class_acceptance() {
    let grammar = r#"root ::= [^abc]+"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "xyz"));
    assert!(is_grammar_accept_string_g(&g, "123"));
    assert!(is_grammar_accept_string_g(&g, "def"));
    assert!(!is_grammar_accept_string_g(&g, "a"));
    assert!(!is_grammar_accept_string_g(&g, "abc"));
    assert!(!is_grammar_accept_string_g(&g, ""));
}

// ---------------------------------------------------------------------------
// Simple rule interaction (from test_simple)
// ---------------------------------------------------------------------------

#[test]
fn test_simple_rule_interaction() {
    let grammar = r#"
root ::= rule1 rule2
rule1 ::= (rule2 | rule3) "a"
rule2 ::= "b"
rule3 ::= "c"
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "bab"));
    assert!(is_grammar_accept_string_g(&g, "cab"));
    assert!(!is_grammar_accept_string_g(&g, "abb"));
}

// ---------------------------------------------------------------------------
// Complex repetition (from test_repetition)
// ---------------------------------------------------------------------------

#[test]
fn test_repetition_nested_complex() {
    // (a|[bc]{4,}){2,3}
    let grammar = r#"
root ::= rule {2, 3}
rule ::= ("a" | [bc] {4,})
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "aaa"));
    assert!(is_grammar_accept_string_g(&g, "abcbc"));
    assert!(is_grammar_accept_string_g(&g, "bcbcbcbcbc"));
    assert!(is_grammar_accept_string_g(&g, "bcbcbcbcbcbcbcb"));
    assert!(!is_grammar_accept_string_g(&g, "d"));
    assert!(!is_grammar_accept_string_g(&g, "aaaa"));
}

#[test]
fn test_repetition_with_empty() {
    let grammar = r#"
root ::= rule {2, 3} "d"?
rule ::= ("a" | [bc] {4,}) | ""
"#;
    let g = Grammar::from_ebnf(grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "aaa"));
    assert!(is_grammar_accept_string_g(&g, "abcbc"));
    assert!(is_grammar_accept_string_g(&g, "bcbcbcbcbc"));
    assert!(is_grammar_accept_string_g(&g, "bcbcbcbcbcbcbcb"));
    assert!(!is_grammar_accept_string_g(&g, "aaaa")); // 4 non-empty > max 3
    assert!(is_grammar_accept_string_g(&g, ""));       // all empty
    assert!(is_grammar_accept_string_g(&g, "a"));      // 1 non-empty + empties
    assert!(is_grammar_accept_string_g(&g, "d"));       // all empty + "d"?
}

// ---------------------------------------------------------------------------
// JSON acceptance: more complex inputs (from test_json_accept)
// ---------------------------------------------------------------------------

#[test]
fn test_json_accept_complex_objects() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    let cases = [
        r#"{"name": "Alice", "age": 30, "city": "New York"}"#,
        r#"{"name": "Mike", "hobbies": ["reading", "cycling", "hiking"]}"#,
        r#"{"name": "Emma", "address": {"street": "Maple Street", "city": "Boston"}}"#,
        r#"[{"name": "David"}, {"name": "Sophia"}]"#,
        r#"{"name": "William", "age": null, "married": true, "children": ["Liam", "Olivia"], "hasPets": false}"#,
        r#"{"name": "Olivia", "contact": {"email": "olivia@example.com", "address": {"city": "Chicago", "zipcode": "60601"}}}"#,
        r#"{"name": "Liam", "skills": ["Java", "Python"], "experience": [{"company": "CompanyA", "years": 5}, {"company": "CompanyB", "years": 3}]}"#,
    ];

    for case in &cases {
        assert!(
            is_grammar_accept_string_g(&g, case),
            "should accept: {}",
            case
        );
    }
}

// ---------------------------------------------------------------------------
// JSON rejection (from test_json_refuse)
// ---------------------------------------------------------------------------

#[test]
fn test_json_refuse_invalid() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    let cases = [
        r#"{ name: "John" }"#,                    // unquoted key
        r#"{ "name": "John", "age": 30, }"#,      // trailing comma
        r#"{ "name": "John", "age": 30.5.7 }"#,   // invalid number
        r#"{ "name": "John, "age": 30 }"#,         // unclosed string
    ];

    for case in &cases {
        assert!(
            !is_grammar_accept_string_g(&g, case),
            "should reject: {}",
            case
        );
    }
}

// ---------------------------------------------------------------------------
// JSON pressure test (from test_json_pressure)
// ---------------------------------------------------------------------------

#[test]
fn test_json_pressure_deep_nesting() {
    let g = Grammar::from_ebnf(JSON_GRAMMAR, "root").unwrap();

    // Deeply nested structure
    let deep = r#"{"person": {"name": "Ethan", "age": 40}, "education": {"degree": "Masters", "university": "XYZ University"}, "work": [{"company": "ABC Corp", "position": "Manager"}, {"company": "DEF Corp", "position": "Senior Manager"}]}"#;
    assert!(is_grammar_accept_string_g(&g, deep));

    // Even deeper nesting
    let deeper = r#"{"name": "Charlotte", "details": {"personal": {"age": 35, "hobbies": ["gardening", "painting"]}, "professional": {"occupation": "Engineer", "skills": ["CAD", "Project Management"], "projects": [{"name": "Project A", "status": "Completed"}, {"name": "Project B", "status": "In Progress"}]}}}"#;
    assert!(is_grammar_accept_string_g(&g, deeper));
}

// ---------------------------------------------------------------------------
// UTF-8 comma character class (from test_utf8)
// ---------------------------------------------------------------------------

#[test]
fn test_utf8_comma_char_class() {
    // Chinese comma: ，(U+FF0C)
    let grammar = "root ::= [\u{FF0C}]+";
    let g = Grammar::from_ebnf(&grammar, "root").unwrap();

    assert!(is_grammar_accept_string_g(&g, "\u{FF0C}"));
    assert!(is_grammar_accept_string_g(&g, "\u{FF0C}\u{FF0C}\u{FF0C}"));
    // Many commas
    let many = "\u{FF0C}".repeat(22);
    assert!(is_grammar_accept_string_g(&g, &many));
}

// ---------------------------------------------------------------------------
// Custom root rule (from test_custom_root_rule)
// ---------------------------------------------------------------------------

#[test]
fn test_custom_root_rule() {
    let grammar = r#"
root ::= basic_object
basic_any ::= basic_string | basic_object
basic_string ::= "\"" basic_string_1 "\""
basic_string_1 ::= "" | [^"\\] basic_string_1 | "\\" ["\\/bfnrt] basic_string_1
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
"#;

    // Parse with basic_string as root
    let g = Grammar::from_ebnf(grammar, "basic_string").unwrap();
    assert!(is_grammar_accept_string_g(&g, r#""abc\r\n""#));
    // An object should not match basic_string
    assert!(!is_grammar_accept_string_g(
        &g,
        r#"{"name": "John" }"#
    ));
}
