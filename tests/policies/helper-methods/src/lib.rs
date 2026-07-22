use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct HelperMethods;

impl Policy for HelperMethods {
    fn route(ctx: &Document, state: &mut State, host: &Host) -> Result<Document, String> {
        let request_id = ctx["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?;
        let target = ctx["candidates"][0]["id"]
            .as_str()
            .ok_or("candidate ID must be a string")?;
        let lookup = host.kv_lookup(request_id, target)?;
        let capacity = host.cluster_capacity("example-model")?;
        let config = host.model_config()?;
        let now_ms = host.now_ms()?;
        let action_ids = [
            host.prefetch_kv(request_id, target)?,
            host.preempt(request_id)?,
            host.replicate(request_id, &[target, "node-b"])?,
            host.set_retention(request_id, 5000)?,
            host.arm_timer(request_id, 10)?,
        ];
        state.shared["helpers"] = json!({
            "lookup": lookup,
            "capacity": capacity,
            "config": config,
            "now_ms": now_ms,
            "action_ids": action_ids,
        });
        let count = ctx["candidates"].as_array().map_or(0, Vec::len);
        Ok(json!({"scores": vec![0.0; count]}))
    }
}

plex::export_policy!(HelperMethods);
