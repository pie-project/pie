//! OpenResponses-compliant HTTP server inferlet for Pie.
//!
//! This server implements the OpenResponses specification, providing a standard
//! API for LLM inference that is compatible with multiple providers.
//!
//! ## Endpoints
//!
//! - `POST /responses` - Create a response (main endpoint)
//!
//! ## Usage
//!
//! ```bash
//! # Build
//! cargo build -p openresponses-server --target wasm32-wasip2 --release
//!
//! # Run
//! pie http --path ./openresponses_server.wasm --port 8080
//!
//! # Test (non-streaming)
//! curl -X POST http://localhost:8080/responses \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"auto","input":[{"type":"message","role":"user","content":"Hello!"}]}'
//!
//! # Test (streaming)
//! curl -X POST http://localhost:8080/responses \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"auto","input":[{"type":"message","role":"user","content":"Hello!"}],"stream":true}'
//! ```
//!
//! ## Specification
//!
//! See: https://www.openresponses.org/specification

mod handler;
mod streaming;
mod types;

use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};
use wstd::io::AsyncRead;

#[wstd::http_server]
async fn main(mut req: Request<IncomingBody>, res: Responder) -> Finished {
    let path = req.uri().path();
    let method = req.method().clone();

    match (method, path) {
        (Method::POST, "/responses") => {
            // Read the request body
            let mut body_bytes = Vec::new();
            if let Err(_) = read_body(req.body_mut(), &mut body_bytes).await {
                return error_response(res, 400, "Failed to read request body").await;
            }

            handler::handle_responses::<IncomingBody>(body_bytes, res).await
        }

        (Method::GET, "/") => {
            let info = r#"{
  "name": "Pie OpenResponses Server",
  "version": "0.1.0",
  "spec": "https://www.openresponses.org/specification",
  "endpoints": {
    "POST /responses": "Create a response"
  }
}
"#;
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(info.into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::OPTIONS, _) => {
            // CORS preflight
            let response = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                .header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                .body("".into_body())
                .unwrap();
            res.respond(response).await
        }

        _ => {
            not_found(res).await
        }
    }
}

/// Read the entire request body into a Vec<u8>
async fn read_body(body: &mut IncomingBody, buf: &mut Vec<u8>) -> Result<(), ()> {
    let mut chunk = [0u8; 4096];
    loop {
        match body.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => return Err(()),
        }
    }
    Ok(())
}

/// Return an error response
async fn error_response(res: Responder, status: u16, message: &str) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": "invalid_request",
            "message": message,
        }
    });

    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    res.respond(response).await
}

/// Return 404 Not Found
async fn not_found(res: Responder) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": "not_found",
            "message": "Endpoint not found",
        }
    });

    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    res.respond(response).await
}
