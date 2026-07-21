use plex::serde_json::json;
use plex::{Document, Policy};

struct RewriteAdmit;

impl Policy for RewriteAdmit {
    fn admit(input: &mut Document) -> Result<Document, String> {
        let count = input["request"]["state"]["admission_count"]
            .as_u64()
            .unwrap_or(0)
            + 1;
        input["request"]["state"]["admission_count"] = json!(count);
        let prompt = input["request"]["body"]["prompt"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        input["request"]["body"]["prompt"] = json!(format!("{prompt}|admit-{count}"));
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
