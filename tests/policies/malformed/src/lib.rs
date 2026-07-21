use plex::serde_json::json;
use plex::{Document, Policy};

struct Malformed;

impl Policy for Malformed {
    fn route(_input: &mut Document) -> Result<Document, String> {
        Ok(json!({"scores": []}))
    }
}

plex::export_policy!(Malformed);
