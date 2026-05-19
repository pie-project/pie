//! Minimal HTTP control plane.
//!
//! Routes:
//!   * `GET /healthz`    — liveness probe; returns engine status + first
//!     model name + uptime in seconds.
//!   * `GET /v1/models`  — OpenAI-shape model list (single entry, the
//!     primary model registered with the engine).
//!
//! Bound separately from the WebSocket inference server (`crate::server`).
//! See `bootstrap::bootstrap_inner` for wiring + stdout handshake.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

#[derive(Clone)]
pub struct State_ {
    pub model_name: String,
    pub created_unix: u64,
    pub started: Instant,
}

#[derive(Debug, Clone)]
pub struct HttpHandle {
    pub addr: SocketAddr,
    pub task: JoinHandleArc,
}

pub type JoinHandleArc = Arc<tokio::sync::Mutex<Option<JoinHandle<()>>>>;

/// Bind to `host:port` and return the bound `TcpListener`. Port `0`
/// asks the OS for an ephemeral port (mirrors how `pie serve --port 0`
/// works for the websocket server).
pub async fn bind(host: &str, port: u16) -> Result<TcpListener> {
    let addr = format!("{host}:{port}");
    TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind http listener on {addr}"))
}

/// Spawn the HTTP server on a pre-bound listener. Returns the bound
/// socket address + a join handle so the engine can await shutdown.
pub fn spawn_listener(listener: TcpListener, state: State_) -> Result<HttpHandle> {
    let addr = listener
        .local_addr()
        .context("read http listener local_addr")?;
    let app = router(state);
    let task = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("http listener exited with error: {e}");
        }
    });
    Ok(HttpHandle {
        addr,
        task: Arc::new(tokio::sync::Mutex::new(Some(task))),
    })
}

/// Bind + spawn in one shot.
pub async fn spawn(host: &str, port: u16, state: State_) -> Result<HttpHandle> {
    let listener = bind(host, port).await?;
    spawn_listener(listener, state)
}

pub fn router(state: State_) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .with_state(state)
}

#[derive(Serialize)]
struct Healthz {
    status: &'static str,
    model: String,
    uptime_secs: u64,
}

async fn healthz(State(s): State<State_>) -> Json<Healthz> {
    Json(Healthz {
        status: "ok",
        model: s.model_name.clone(),
        uptime_secs: s.started.elapsed().as_secs(),
    })
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

async fn list_models(State(s): State<State_>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: s.model_name.clone(),
            object: "model",
            created: s.created_unix,
            owned_by: "pie",
        }],
    })
}

/// Helper for builders that want the current epoch second without
/// pulling chrono everywhere.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn fixture_state(model: &str) -> State_ {
        State_ {
            model_name: model.to_string(),
            created_unix: 1_700_000_000,
            started: Instant::now(),
        }
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn healthz_reports_ok_and_model() {
        let app = router(fixture_state("llama-3"));
        let resp = app
            .oneshot(Request::builder().uri("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["status"], "ok");
        assert_eq!(body["model"], "llama-3");
        assert!(body["uptime_secs"].is_u64());
    }

    #[tokio::test]
    async fn list_models_openai_shape_single_entry() {
        let app = router(fixture_state("qwen"));
        let resp = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().expect("data array");
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "qwen");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "pie");
        assert_eq!(data[0]["created"], 1_700_000_000);
    }

    #[tokio::test]
    async fn end_to_end_spawn_bind_zero() {
        let handle = spawn("127.0.0.1", 0, fixture_state("m")).await.unwrap();
        let url = format!("http://{}/healthz", handle.addr);
        let body = reqwest::get(&url).await.unwrap().text().await.unwrap();
        assert!(body.contains("\"status\":\"ok\""));
        // Shutdown by aborting the task.
        if let Some(t) = handle.task.lock().await.take() {
            t.abort();
        }
    }
}
