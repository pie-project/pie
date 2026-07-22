use plex::exports::pie::plex::policy::{Guest, Invocation, PolicyOutput};

struct NonFinite;

impl Guest for NonFinite {
    fn route(_: Invocation) -> Result<PolicyOutput, String> {
        output(r#"{"scores":[NaN]}"#)
    }

    fn admit(_: Invocation) -> Result<PolicyOutput, String> {
        Err("fallback-required".into())
    }

    fn schedule(_: Invocation) -> Result<PolicyOutput, String> {
        Err("fallback-required".into())
    }

    fn evict(_: Invocation) -> Result<PolicyOutput, String> {
        output(r#"{"scores":[Infinity]}"#)
    }

    fn feedback(_: Invocation) -> Result<PolicyOutput, String> {
        Err("fallback-required".into())
    }
}

fn output(result_json: &str) -> Result<PolicyOutput, String> {
    plex::link_host_interface();
    Ok(PolicyOutput {
        result_json: result_json.into(),
        state_update_json: "{}".into(),
    })
}

plex::export!(NonFinite with_types_in plex);
