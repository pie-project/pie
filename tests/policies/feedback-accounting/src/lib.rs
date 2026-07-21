use plex::serde_json::json;
use plex::{Document, Policy};

struct FeedbackAccounting;

impl Policy for FeedbackAccounting {
    fn feedback(input: &mut Document) -> Result<Document, String> {
        let records = input["records"]
            .as_array_mut()
            .ok_or("records must be an array")?;
        for record in records {
            match record["event"].as_str().unwrap_or("") {
                "progress" => {
                    let committed = record["facts"]["committed_tokens"].as_u64().unwrap_or(0);
                    let previous = record["request"]["state"]["attained_service"]
                        .as_u64()
                        .unwrap_or(0);
                    record["request"]["state"]["attained_service"] = json!(previous + committed);
                }
                "tool-boundary" => {
                    let previous = record["request"]["state"]["tool_calls"]
                        .as_u64()
                        .unwrap_or(0);
                    record["request"]["state"]["tool_calls"] = json!(previous + 1);
                }
                _ => {}
            }
        }
        Ok(json!({}))
    }
}

plex::export_policy!(FeedbackAccounting);
