use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateFail;

impl Policy for MutateFail {
    fn route(input: &mut Document) -> Result<Document, String> {
        input["request"]["state"]["should_not_commit"] = json!(true);
        Err("fallback-required".into())
    }
}

plex::export_policy!(MutateFail);
