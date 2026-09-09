use ::grammar::regex::regex_to_ebnf;

fn ebnf_body(pattern: &str) -> String {
    let ebnf = regex_to_ebnf(pattern).unwrap();

    ebnf.strip_prefix("root ::= ")
        .unwrap_or(&ebnf)
        .trim_end()
        .to_string()
}

#[test]
fn converter_every_case() {
    test_basic_literal();
    test_anchors_stripped();
    test_star_quantifier_conversion();
    test_plus_quantifier_conversion();
    test_question_quantifier_conversion();
    test_repetition_exact_conversion();
    test_repetition_range_conversion();
    test_repetition_unbounded_conversion();
}

fn test_basic_literal() {
    let body = ebnf_body("123");
    assert!(body.contains("\"1\"") || body.contains("\"123\""));
}

fn test_anchors_stripped() {
    let body = ebnf_body("^abc$");
    assert!(body.contains("\"a\"") || body.contains("\"abc\""));
    assert!(!body.contains("^"));
    assert!(!body.contains("$"));
}

fn test_star_quantifier_conversion() {
    let body = ebnf_body("a*");
    assert!(body.contains("*"), "body should contain *: {}", body);
}

fn test_plus_quantifier_conversion() {
    let body = ebnf_body("a+");
    assert!(body.contains("+"), "body should contain +: {}", body);
}

fn test_question_quantifier_conversion() {
    let body = ebnf_body("a?");
    assert!(body.contains("?"), "body should contain ?: {}", body);
}

fn test_repetition_exact_conversion() {
    let body = ebnf_body("a{3}");
    assert!(body.contains("{3}"), "body should contain {{3}}: {}", body);
}

fn test_repetition_range_conversion() {
    let body = ebnf_body("a{2,5}");
    assert!(
        body.contains("{2,5}"),
        "body should contain {{2,5}}: {}",
        body
    );
}

fn test_repetition_unbounded_conversion() {
    let body = ebnf_body("a{1,}");
    assert!(
        body.contains("{1,}") || body.contains('+'),
        "body should contain an unbounded repetition: {}",
        body
    );
}
