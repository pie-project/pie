//! JSON guest SDK for the breaking PLEX v0.3 proof of concept.

#![forbid(unsafe_code)]

pub use serde_json;
pub use wit_bindgen;

pub type Document = serde_json::Value;

wit_bindgen::generate!({
    path: "wit",
    world: "plex-policy",
    pub_export_macro: true,
    generate_all,
});

pub trait Policy {
    fn route(input: &mut Document) -> Result<Document, String> {
        let _ = input;
        Err("fallback-required".into())
    }

    fn admit(input: &mut Document) -> Result<Document, String> {
        let _ = input;
        Err("fallback-required".into())
    }

    fn schedule(input: &mut Document) -> Result<Document, String> {
        let _ = input;
        Err("fallback-required".into())
    }

    fn evict(input: &mut Document) -> Result<Document, String> {
        let _ = input;
        Err("fallback-required".into())
    }

    fn feedback(input: &mut Document) -> Result<Document, String> {
        let _ = input;
        Err("fallback-required".into())
    }
}

pub fn invoke(
    input_json: String,
    operation: impl FnOnce(&mut Document) -> Result<Document, String>,
) -> Result<String, String> {
    let mut input: Document = serde_json::from_str(&input_json)
        .map_err(|error| format!("invalid JSON input: {error}"))?;
    if !input.is_object() {
        return Err("input must be a top-level JSON object".into());
    }
    let result = operation(&mut input).map_err(|error| format!("policy error: {error}"))?;
    let response = serde_json::json!({
        "input": input,
        "result": result,
    });
    serde_json::to_string(&response)
        .map_err(|error| format!("failed to serialize policy response: {error}"))
}

#[macro_export]
macro_rules! export_policy {
    ($policy:ty) => {
        struct __PlexGuest;

        impl $crate::exports::pie::plex::policy::Guest for __PlexGuest {
            fn route(input_json: String) -> Result<String, String> {
                $crate::invoke(input_json, <$policy as $crate::Policy>::route)
            }

            fn admit(input_json: String) -> Result<String, String> {
                $crate::invoke(input_json, <$policy as $crate::Policy>::admit)
            }

            fn schedule(input_json: String) -> Result<String, String> {
                $crate::invoke(input_json, <$policy as $crate::Policy>::schedule)
            }

            fn evict(input_json: String) -> Result<String, String> {
                $crate::invoke(input_json, <$policy as $crate::Policy>::evict)
            }

            fn feedback(input_json: String) -> Result<String, String> {
                $crate::invoke(input_json, <$policy as $crate::Policy>::feedback)
            }
        }

        $crate::export!(__PlexGuest with_types_in $crate);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_returns_mutated_input_and_result() {
        let response = invoke(
            r#"{"global":{"facts":{},"fields":{},"scratch":{}},"requests":{}}"#.into(),
            |input| {
                input["global"]["scratch"]["calls"] = serde_json::json!(1);
                Ok(serde_json::json!({"decision": "accept"}))
            },
        )
        .unwrap();
        let response: Document = serde_json::from_str(&response).unwrap();
        assert_eq!(response["input"]["global"]["scratch"]["calls"], 1);
        assert_eq!(response["result"]["decision"], "accept");
    }

    #[test]
    fn wrapper_rejects_non_object_and_clear_policy_errors() {
        assert_eq!(
            invoke("[]".into(), |_| Ok(serde_json::json!({}))).unwrap_err(),
            "input must be a top-level JSON object"
        );
        assert_eq!(
            invoke("{}".into(), |_| Err("fallback-required".into())).unwrap_err(),
            "policy error: fallback-required"
        );
    }
}
