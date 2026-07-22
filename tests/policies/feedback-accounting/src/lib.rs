use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct FeedbackAccounting;

impl Policy for FeedbackAccounting {
    fn feedback(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        let records = ctx["records"]
            .as_array()
            .ok_or("records must be an array")?
            .iter()
            .map(|record| {
                (
                    record["event"].as_str().unwrap_or("").to_owned(),
                    record["request_id"]
                        .as_str()
                        .ok_or("record request_id must be a string")
                        .map(str::to_owned),
                    record["facts"].clone(),
                )
            })
            .map(|(event, request_id, facts)| request_id.map(|id| (event, id, facts)))
            .collect::<Result<Vec<_>, _>>()?;
        for (event, request_id, facts) in records {
            let request = state.request_mut(&request_id)?;
            match event.as_str() {
                "progress" => {
                    let committed = facts["committed_tokens"].as_u64().unwrap_or(0);
                    let previous = request.scratch["attained_service"].as_u64().unwrap_or(0);
                    request.scratch["attained_service"] = json!(previous + committed);
                }
                "tool-boundary" => {
                    let previous = request.scratch["tool_calls"].as_u64().unwrap_or(0);
                    request.scratch["tool_calls"] = json!(previous + 1);
                }
                _ => {}
            }
            state.shared["feedback_records"] =
                json!(state.shared["feedback_records"].as_u64().unwrap_or(0) + 1);
        }
        Ok(json!({}))
    }
}

plex::export_policy!(FeedbackAccounting);
