use ::grammar::grammar::Grammar;

fn parse_and_display(input: &str) -> String {
    Grammar::from_ebnf(input, "root").unwrap().to_string()
}

#[test]
fn parser_every_case() {
    test_output_simple_literal();
    test_output_empty_string();
    test_output_character_class();
    test_output_negated_character_class();
    test_output_string_star();
    test_output_alternation();
    test_output_sequence();
    test_output_repetition_exact();
}

fn test_output_simple_literal() {
    let g = parse_and_display(r#"root ::= "abc""#);
    assert!(g.starts_with("root ::= "));
    assert!(g.contains("\"abc\""));
}

fn test_output_empty_string() {
    let g = parse_and_display(r#"root ::= """#);
    assert!(g.contains("\"\""));
}

fn test_output_character_class() {
    let g = parse_and_display("root ::= [a-z0-9]");
    assert!(g.contains("[a-z0-9]"));
}

fn test_output_negated_character_class() {
    let g = parse_and_display("root ::= [^a-z]");
    assert!(g.contains("[^a-z]"));
}

fn test_output_string_star() {
    let g = parse_and_display(r#"root ::= "a"*"#);
    assert!(
        g.contains("root_1"),
        "should have aux rule for string star, got: {}",
        g
    );
}

fn test_output_alternation() {
    let g = parse_and_display(r#"root ::= "a" | "b" | "c""#);
    assert!(g.contains("\"a\""));
    assert!(g.contains("\"b\""));
    assert!(g.contains("\"c\""));
    assert!(g.contains("|"));
}

fn test_output_sequence() {
    let g = parse_and_display(r#"root ::= "a" "b" "c""#);
    assert!(g.contains("\"a\""));
    assert!(g.contains("\"b\""));
    assert!(g.contains("\"c\""));
}

fn test_output_repetition_exact() {
    let g = parse_and_display(r#"root ::= "a"{3}"#);
    assert!(
        g.contains("root_1{3,3}"),
        "expected root_1{{3,3}}, got: {}",
        g
    );
    assert!(
        g.contains("root_1 ::= \"a\""),
        "expected root_1 rule, got: {}",
        g
    );
}
