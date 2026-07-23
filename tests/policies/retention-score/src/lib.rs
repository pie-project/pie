use plex::serde_json::json;
use plex::{CacheAdmission, CacheContext, CachePlan, Host, Policy, State};

struct RetentionScore;

impl Policy for RetentionScore {
    fn cache(ctx: &CacheContext, state: &mut State, _host: &Host) -> plex::Result<CachePlan> {
        state.shared["working_set_size"] = json!(state.request_ids().count());
        let mut reclaim = ctx
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.reclaimable)
            .map(|(index, resident)| {
                let mut score = resident.object.facts["reload_cost"].as_i64().unwrap_or(0);
                for beneficiary in &resident.object.beneficiaries {
                    if let plex::Beneficiary::Request(request_id) = beneficiary
                        && let Ok(request) = state.request_mut(request_id.as_str())
                    {
                        score += request.scratch["retention_bonus"].as_i64().unwrap_or(0);
                        request.scratch["cache_checks"] =
                            json!(request.scratch["cache_checks"].as_u64().unwrap_or(0) + 1);
                    }
                }
                (score, index as u32)
            })
            .collect::<Vec<_>>();
        reclaim.sort_by_key(|(score, index)| (*score, *index));
        Ok(CachePlan {
            admissions: vec![CacheAdmission::Bypass; ctx.prospective.len()],
            reclaim: reclaim.into_iter().map(|(_, index)| index).collect(),
        })
    }
}

plex::export_policy!(RetentionScore);
