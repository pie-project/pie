use plex::serde_json::json;
use plex::{Document, Policy};

struct MutateFail;

impl Policy for MutateFail {
    fn route(input: &mut Document) -> Result<Document, String> {
        let request_id = input["request_id"].as_str().unwrap_or("").to_owned();
        input["requests"][request_id.as_str()]["scratch"]["should_not_commit"] = json!(true);
        input["global"]["scratch"]["should_not_commit"] = json!(true);
        Err("fallback-required".into())
    }
}

plex::export_policy!(MutateFail);
