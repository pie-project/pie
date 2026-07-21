use plex::exports::pie::plex::policy::Guest;

struct NonFinite;

impl Guest for NonFinite {
    fn route(input_json: String) -> Result<String, String> {
        Ok(format!(
            r#"{{"input":{input_json},"result":{{"scores":[NaN]}}}}"#
        ))
    }

    fn admit(_input_json: String) -> Result<String, String> {
        Err("fallback-required".into())
    }

    fn schedule(_input_json: String) -> Result<String, String> {
        Err("fallback-required".into())
    }

    fn evict(input_json: String) -> Result<String, String> {
        Ok(format!(
            r#"{{"input":{input_json},"result":{{"scores":[Infinity]}}}}"#
        ))
    }

    fn feedback(_input_json: String) -> Result<String, String> {
        Err("fallback-required".into())
    }
}

plex::export!(NonFinite with_types_in plex);
