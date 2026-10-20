//! How an HTTP request reaches the program.
//!
//! The gateway launches a built-in inferlet with this JSON as its input:
//!
//! ```json
//! {"path": "/v1/chat/completions", "query": {"alt": "sse"}, "body": {…}, "time": 1758400000}
//! ```
//!
//! `path` and `query` are there for the APIs that put meaning in them (Gemini
//! chooses streaming by path); `time` is the request's wall-clock arrival in
//! unix seconds, because the guest has no wall clock of its own. An input
//! without a `body` key is taken to be the body itself, so `pie run` with a
//! hand-written request works too.

use std::collections::BTreeMap;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    pub path: String,
    pub query: BTreeMap<String, String>,
    pub body: Value,
    /// Unix seconds at the gateway, or 0 when the program was not launched
    /// through it.
    pub time: u64,
}

impl Envelope {
    pub fn parse(input: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(input).map_err(|e| format!("input is not JSON: {e}"))?;
        let Value::Object(mut object) = value else {
            return Err("input is not a JSON object".into());
        };
        let Some(body) = object.remove("body") else {
            return Ok(Self {
                path: String::new(),
                query: BTreeMap::new(),
                body: Value::Object(object),
                time: 0,
            });
        };
        let path = object
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let query = object
            .get("query")
            .and_then(Value::as_object)
            .map(|q| {
                q.iter()
                    .map(|(k, v)| {
                        let v = match v {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (k.clone(), v)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let time = object.get("time").and_then(Value::as_u64).unwrap_or(0);
        Ok(Self {
            path,
            query,
            body,
            time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_every_case() {
        let e = Envelope::parse(
            r#"{"path":"/v1/x","query":{"alt":"sse","n":2},"body":{"a":1},"time":7}"#,
        )
        .unwrap();
        assert_eq!(e.path, "/v1/x");
        assert_eq!(e.query.get("alt").map(String::as_str), Some("sse"));
        assert_eq!(e.query.get("n").map(String::as_str), Some("2"));
        assert_eq!(e.body, serde_json::json!({"a": 1}));
        assert_eq!(e.time, 7);

        let bare = Envelope::parse(r#"{"messages":[]}"#).unwrap();
        assert_eq!(bare.path, "");
        assert_eq!(bare.body, serde_json::json!({"messages": []}));
        assert_eq!(bare.time, 0);

        assert!(Envelope::parse("[]").is_err());
        assert!(Envelope::parse("nope").is_err());
    }
}
