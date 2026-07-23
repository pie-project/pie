use plex::serde_json::json;
use plex::{FeedbackContext, FeedbackSubject, Host, OutcomeKind, Policy, State};

struct FeedbackAccounting;

impl Policy for FeedbackAccounting {
    fn feedback(
        ctx: &FeedbackContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<()> {
        for record in &ctx.records {
            let FeedbackSubject::Request(request_id) = &record.subject else {
                continue;
            };
            if matches!(
                record.outcome,
                OutcomeKind::Completed
                    | OutcomeKind::Failed
                    | OutcomeKind::Cancelled
                    | OutcomeKind::Expired
            ) {
                continue;
            }
            let request = state.request_mut(request_id.as_str())?;
            let committed = record.facts["committed_tokens"].as_u64().unwrap_or(0);
            request.scratch["attained_service"] =
                json!(request.scratch["attained_service"].as_u64().unwrap_or(0) + committed);
            if record.facts["tool_boundary"].as_bool() == Some(true) {
                request.scratch["tool_calls"] =
                    json!(request.scratch["tool_calls"].as_u64().unwrap_or(0) + 1);
            }
            state.shared["feedback_records"] =
                json!(state.shared["feedback_records"].as_u64().unwrap_or(0) + 1);
        }
        Ok(())
    }
}

plex::export_policy!(FeedbackAccounting);
