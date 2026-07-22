use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct Malformed;

impl Policy for Malformed {
    fn route(_ctx: &Document, _state: &mut State, _host: &Host) -> Result<Document, String> {
        Ok(json!({"scores": []}))
    }
}

plex::export_policy!(Malformed);
