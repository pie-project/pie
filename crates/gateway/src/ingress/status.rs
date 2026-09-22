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
    let workers: Vec<Value> = table
        .workers
        .iter()
        .map(|w| {
            json!({
                "id": w.id.to_string(),
                "model": w.model,
                "role": format!("{:?}", w.role).to_lowercase(),
                "health": format!("{:?}", w.health).to_lowercase(),
                "inflight": w.coarse_load.inflight,
                "kv_pressure_bucket": w.coarse_load.kv_pressure_bucket,
            })
        })
        .collect();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": START.get().map_or(0, |t| t.elapsed().as_secs()),
        "time": SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        "sessions": state.sessions.live(),
        "routing_epoch": table.epoch,
        "workers": workers,
    }))
}
