use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct RawHelpers;

impl Policy for RawHelpers {
    fn route(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let query = host.query_raw("engine.custom-query@1", &json!({"value": 7}))?;
        let action_id = host.action_raw(
            "engine.custom-action@1",
            &json!({"request_id": ctx["request_id"]}),
        )?;
        state.shared["raw"] = json!({"query": query, "action_id": action_id});
        let count = ctx["candidates"].as_array().map_or(0, Vec::len);
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(RawHelpers);
