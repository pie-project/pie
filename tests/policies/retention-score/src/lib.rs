use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct RetentionScore;

impl Policy for RetentionScore {
    fn evict(ctx: &Document, state: &mut State, _host: &Host) -> Result<Document, String> {
        state.shared["working_set_size"] = json!(state.request_ids().count());
        let units = ctx["resident"]
            .as_array()
            .ok_or("resident must be an array")?
            .iter()
            .map(|unit| {
                (
                    unit["request_id"].as_str().map(str::to_owned),
                    unit["facts"]["reload_cost"].as_f64().unwrap_or(0.0),
                )
            })
            .collect::<Vec<_>>();
        let mut scores = Vec::with_capacity(units.len());
        for (request_id, reload) in units {
            let retention = if let Some(request_id) = request_id {
                let request = state.request_mut(&request_id)?;
                let retention = request.scratch["retention_bonus"].as_f64().unwrap_or(0.0);
                request.scratch["eviction_checks"] =
                    json!(request.scratch["eviction_checks"].as_u64().unwrap_or(0) + 1);
                retention
            } else {
                0.0
            };
            scores.push(reload + retention);
        }
        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(RetentionScore);
