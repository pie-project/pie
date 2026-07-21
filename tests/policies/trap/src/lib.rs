use plex::serde_json::json;
use plex::{Document, Policy};

struct Trap;

impl Policy for Trap {
    fn route(input: &mut Document) -> Result<Document, String> {
        let request_id = input["request_id"].as_str().unwrap_or("").to_owned();
        input["requests"][request_id.as_str()]["scratch"]["should_not_commit"] = json!(true);
        input["global"]["scratch"]["should_not_commit"] = json!(true);
        panic!("injected JSON PLEX trap")
    }
}

plex::export_policy!(Trap);
