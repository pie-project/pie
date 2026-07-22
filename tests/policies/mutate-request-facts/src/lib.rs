use plex::exports::pie::plex::policy::{Guest, Invocation, PolicyOutput};
use plex::serde_json::{Value, json};

macro_rules! fallback_hooks {
    () => {
        fn admit(_: Invocation) -> Result<PolicyOutput, String> {
            Err("fallback-required".into())
        }
        fn schedule(_: Invocation) -> Result<PolicyOutput, String> {
            Err("fallback-required".into())
        }
        fn evict(_: Invocation) -> Result<PolicyOutput, String> {
            Err("fallback-required".into())
        }
        fn feedback(_: Invocation) -> Result<PolicyOutput, String> {
            Err("fallback-required".into())
        }
    };
}

struct MutateRequestFacts;

impl Guest for MutateRequestFacts {
    fn route(input: Invocation) -> Result<PolicyOutput, String> {
        plex::link_host_interface();
        let context: Value = plex::serde_json::from_str(&input.context_json).unwrap();
        let request_id = context["request_id"].as_str().unwrap_or("");
        Ok(PolicyOutput {
            result_json: r#"{"scores":[0.0]}"#.into(),
            state_update_json: json!({
                "requests": {
                    request_id: {
                        "facts": {"generation_id": 999},
                        "fields": {},
                        "scratch": {}
                    }
                }
            })
            .to_string(),
        })
    }

    fallback_hooks!();
}

plex::export!(MutateRequestFacts with_types_in plex);
