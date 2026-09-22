use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::{Json, extract::State};
use serde_json::{Value, json};
use std::sync::OnceLock;

use crate::GatewayState;

static START: OnceLock<Instant> = OnceLock::new();

pub fn started() {
    START.get_or_init(Instant::now);
}

pub async fn status(State(state): State<GatewayState>) -> Json<Value> {
    let table = state.routing.table();
    let mut workers: Vec<Value> = Vec::with_capacity(table.workers.len());
    for w in &table.workers {
        let memory = match state.workers.client(w.id) {
            Some(client) => client
                .memory(tarpc::context::current())
                .await
                .ok()
                .flatten()
                .map(|m| {
                    json!({
                        "working_set": m.working_set,
                        "ceiling": m.ceiling,
                        "weights": m.weights,
                        "scratch": m.scratch,
                        "floor": m.floor,
                        "pool": m.pool,
                        "pool_needed": m.minimum,
                        "free": m.pool.saturating_sub(m.minimum),
                    })
                }),
            None => None,
        };
        workers.push(json!({
            "id": w.id.to_string(),
            "model": w.model,
            "role": format!("{:?}", w.role).to_lowercase(),
            "health": format!("{:?}", w.health).to_lowercase(),
            "inflight": w.coarse_load.inflight,
            "kv_pressure_bucket": w.coarse_load.kv_pressure_bucket,
            "memory": memory,
        }));
    }
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": START.get().map_or(0, |t| t.elapsed().as_secs()),
        "time": SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        "sessions": state.sessions.live(),
        "routing_epoch": table.epoch,
        "workers": workers,
    }))
}
