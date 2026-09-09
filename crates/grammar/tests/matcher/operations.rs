use crate::common::matcher_from_ebnf as make_matcher;

#[test]
fn operations_every_case() {
    test_char_class_excludes_control_chars();
    test_token_acceptance_sequence();
    test_token_rejection();
}

fn test_char_class_excludes_control_chars() {
    let ebnf = r#"root ::= [^"\x00-\x1f\\]"#;
    let vocab = &["\n", "\t", "x", " ", "\"", "\\", "a"];
    let mut m = make_matcher(ebnf, "root", vocab);
    assert!(!m.accept_token(0), "\\n should be rejected");
    let mut m2 = make_matcher(ebnf, "root", vocab);
    assert!(!m2.accept_token(1), "\\t should be rejected");
    let mut m3 = make_matcher(ebnf, "root", vocab);
    assert!(m3.accept_token(2), "x should be accepted");
    let mut m4 = make_matcher(ebnf, "root", vocab);
    assert!(m4.accept_token(3), "space should be accepted");
    let mut m5 = make_matcher(ebnf, "root", vocab);
    assert!(!m5.accept_token(4), "quote should be rejected");
    let mut m6 = make_matcher(ebnf, "root", vocab);
    assert!(!m6.accept_token(5), "backslash should be rejected");
}

fn test_token_acceptance_sequence() {
    let mut m = make_matcher(
        r#"root ::= "hello" " " "world""#,
        "root",
        &["hello", " ", "world", "x"],
    );

    assert!(m.accept_token(0)); // "hello"
    assert!(m.accept_token(1)); // " "
    assert!(m.accept_token(2)); // "world"
    assert!(m.can_terminate());
}

fn test_token_rejection() {
    let mut m = make_matcher(r#"root ::= "hello""#, "root", &["hello", "world"]);

    assert!(!m.accept_token(1)); // "world" should fail
    assert!(m.accept_token(0)); // "hello" still works
    assert!(m.can_terminate());
}
