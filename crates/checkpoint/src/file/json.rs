//! The JSON a checkpoint ships beside its weights, as the tools that wrote it
//! spell it. Python's `json` writes `Infinity`, `-Infinity` and `NaN`, which
//! JSON has no words for (a `time_step_limit` of `[0.0, Infinity]` is common);
//! [`parse`] reads them as the largest finite double either way, and null.

/// `text` parsed as JSON, Python's non-finite words read as [`finite`] spells them.
pub fn parse(text: &str) -> serde_json::Result<serde_json::Value> {
    serde_json::from_str(&finite(text))
}

/// `text` with every `Infinity`, `-Infinity` and `NaN` outside a string
/// spelled as JSON can read it.
#[must_use]
pub fn finite(text: &str) -> String {
    const SPELLED: [(&str, &str); 3] = [
        ("-Infinity", "-1.7976931348623157e308"),
        ("Infinity", "1.7976931348623157e308"),
        ("NaN", "null"),
    ];
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_string {
            in_string = !(c == '"' && !escaped);
            escaped = c == '\\' && !escaped;
        } else if c == '"' {
            in_string = true;
        } else if let Some((word, json)) = SPELLED.iter().find(|(word, _)| rest.starts_with(word)) {
            out.push_str(json);
            rest = &rest[word.len()..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_configuration_python_wrote_reads() {
        let read =
            parse(r#"{"a": [0.0, Infinity, -Infinity, NaN], "b": "NaN \"Infinity"}"#).unwrap();
        assert_eq!(read["a"][1], serde_json::json!(f64::MAX));
        assert_eq!(read["a"][2], serde_json::json!(-f64::MAX));
        assert!(read["a"][3].is_null());
        assert_eq!(read["b"], "NaN \"Infinity");
    }
}
