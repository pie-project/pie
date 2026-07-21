use plex::serde_json::json;
use plex::{Document, Policy};

struct RewriteAdmit;

impl Policy for RewriteAdmit {
    fn admit(input: &mut Document) -> Result<Document, String> {
        input["global"]["fields"]["route_owner_calls_seen"] =
            input["global"]["scratch"]["route_owner_calls"].clone();
        let request_id = input["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?
            .to_owned();
        let request = &mut input["requests"][request_id.as_str()];
        let count = request["scratch"]["admission_count"].as_u64().unwrap_or(0) + 1;
        request["scratch"]["admission_count"] = json!(count);
        let prompt = request["fields"]["body"]["prompt"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        request["fields"]["body"]["prompt"] = json!(format!("{prompt}|admit-{count}"));
        let queue = input["target"]["facts"]["queue_depth"]
            .as_u64()
            .unwrap_or(0);
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
