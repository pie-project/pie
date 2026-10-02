//! An API error the gateway turns into an HTTP status and body.
//!
//! The program fails with a string. When that string parses as
//! `{"status": <u16>, "body": <json>}` the gateway answers with exactly that
//! status and body; anything else becomes a 500 with the string as the
//! message. Each adapter shapes `body` the way its API's clients expect.

use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub body: Value,
}

impl ApiError {
    pub fn new(status: u16, body: Value) -> Self {
        Self { status, body }
    }
}

impl From<ApiError> for String {
    fn from(error: ApiError) -> String {
        json!({"status": error.status, "body": error.body}).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_api_error_is_its_status_and_body() {
        let e = ApiError::new(400, json!({"error": {"message": "bad"}}));
        let s: String = e.into();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["status"], 400);
        assert_eq!(v["body"]["error"]["message"], "bad");
    }
}
