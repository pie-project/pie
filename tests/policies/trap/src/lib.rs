use plex::serde_json::json;
use plex::{Document, Policy};

struct Trap;

impl Policy for Trap {
    fn route(input: &mut Document) -> Result<Document, String> {
        input["request"]["state"]["should_not_commit"] = json!(true);
        panic!("injected JSON PLEX trap")
    }
}

plex::export_policy!(Trap);
