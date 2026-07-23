use plex::serde_json::json;
use plex::{AdmitContext, AdmitPlan, AdmissionDecision, Host, Policy, State};

struct RewriteAdmit;

impl Policy for RewriteAdmit {
    fn admit(ctx: &AdmitContext, state: &mut State, _host: &Host) -> plex::Result<AdmitPlan> {
        let decisions = ctx
            .candidates
            .iter()
            .map(|candidate| {
                let request = state.request_mut(candidate.request.request_id.as_str())?;
                let count = request.scratch["admission_count"].as_u64().unwrap_or(0) + 1;
                request.scratch["admission_count"] = json!(count);
                request.fields["admission_count"] = json!(count);
                let queue = candidate.facts["queue_depth"].as_u64().unwrap_or(0);
                Ok(if queue < 80 {
                    AdmissionDecision::Accept
                } else if queue < 100 {
                    AdmissionDecision::Defer
                } else {
                    AdmissionDecision::Reject
                })
            })
            .collect::<plex::Result<Vec<_>>>()?;
        Ok(AdmitPlan { decisions })
    }
}

plex::export_policy!(RewriteAdmit);
