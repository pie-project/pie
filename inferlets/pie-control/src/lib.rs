//! pie-control: engine control-plane inferlet.
//!
//! Exports `wasi:http/incoming-handler@0.2.4`. A launcher process
//! installs this component, then calls `launch_daemon(inferlet_id, port)`
//! over WS; the pie host binds the listener and routes each request here.
//!
//! Routes:
//!   * `GET  /healthz`    -> `{"status":"ok"}`
//!   * `GET  /v1/models`  -> OpenAI-shape model list (from `inferlet::runtime::models()`)
//!
//! Model name is sourced from the host via the `pie:core/runtime.models`
//! WIT import (no per-request header injection / no daemon-side context).

use serde::Serialize;
use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};

#[wstd::http_server]
async fn main(req: Request<IncomingBody>, res: Responder) -> Finished {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    match (method, path.as_str()) {
        (Method::GET, "/healthz") => healthz(res).await,
        (Method::GET, "/v1/models") => list_models(res).await,
        _ => not_found(res).await,
    }
}

#[derive(Serialize)]
struct Healthz<'a> {
    status: &'a str,
}

async fn healthz(res: Responder) -> Finished {
    let body = serde_json::to_string(&Healthz { status: "ok" })
        .unwrap_or_else(|_| "{\"status\":\"ok\"}".into());
    let response = Response::builder()
        .header("Content-Type", "application/json")
        .body(body.into_body())
        .unwrap();
    res.respond(response).await
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

async fn list_models(res: Responder) -> Finished {
    let data: Vec<ModelObject> = inferlet::runtime::models()
        .into_iter()
        .map(|id| ModelObject {
            id,
            object: "model",
            owned_by: "pie",
        })
        .collect();
    let list = ModelList {
        object: "list",
        data,
    };
    let body = serde_json::to_string(&list).unwrap_or_else(|_| "{}".into());
    let response = Response::builder()
        .header("Content-Type", "application/json")
        .body(body.into_body())
        .unwrap();
    res.respond(response).await
}

async fn not_found(res: Responder) -> Finished {
    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body("404 Not Found\n".into_body())
        .unwrap();
    res.respond(response).await
}
