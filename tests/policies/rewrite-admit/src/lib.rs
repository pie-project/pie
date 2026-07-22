use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct RewriteAdmit;

impl Policy for RewriteAdmit {
    fn admit(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let route_calls = state.shared["route_owner_calls"].clone();
        state.shared["route_owner_calls_seen"] = route_calls;
        let request_id = ctx["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?
            .to_owned();
        let request = state.request_mut(&request_id)?;
        let count = request.scratch["admission_count"].as_u64().unwrap_or(0) + 1;
        request.scratch["admission_count"] = json!(count);
        let prompt = request.fields["body"]["prompt"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        request.fields["body"]["prompt"] = json!(format!("{prompt}|admit-{count}"));
        let queue = ctx["target"]["facts"]["queue_depth"].as_u64().unwrap_or(0);
        Ok(json!({
            "decision": if queue < 80 {
                "accept"
            } else if queue < 100 {
                "defer"
            } else {
                "reject"
            }
        }))
    }
}

plex::export_policy!(RewriteAdmit);
