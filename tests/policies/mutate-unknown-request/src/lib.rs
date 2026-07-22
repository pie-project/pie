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

struct MutateUnknownRequest;

impl Guest for MutateUnknownRequest {
    fn route(_: Invocation) -> Result<PolicyOutput, String> {
        plex::link_host_interface();
        Ok(PolicyOutput {
            result_json: r#"{"scores":[0.0]}"#.into(),
            state_update_json: r#"{"requests":{"forged":{"fields":{},"scratch":{}}}}"#.into(),
        })
    }

    fallback_hooks!();
}

plex::export!(MutateUnknownRequest with_types_in plex);
