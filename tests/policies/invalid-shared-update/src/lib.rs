use plex::exports::pie::plex::policy::{Guest, Invocation, PolicyOutput};

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

struct InvalidSharedUpdate;

impl Guest for InvalidSharedUpdate {
    fn route(_: Invocation) -> Result<PolicyOutput, String> {
        plex::link_host_interface();
        Ok(PolicyOutput {
            result_json: r#"{"scores":[0.0]}"#.into(),
            state_update_json: r#"{"shared":[]}"#.into(),
        })
    }

    fallback_hooks!();
}

plex::export!(InvalidSharedUpdate with_types_in plex);
