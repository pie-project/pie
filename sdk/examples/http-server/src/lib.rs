//! A simple HTTP server inferlet demonstrating the wasi:http/incoming-handler interface.
//!
//! This example shows how to create an HTTP server inferlet that handles incoming
//! requests and sends responses. Unlike regular inferlets that use `inferlet:core/run`,
//! server inferlets implement the `wasi:http/incoming-handler` interface.
//!
//! ## Endpoints
//!
//! - `/` - Home page, returns a greeting
//! - `/wait` - Demonstrates async sleep and timing
//! - `/echo` - Echoes back the request body
//! - `/echo-headers` - Echoes back the request headers
//! - `/info` - Returns server and request information as JSON
//!
//! ## Running
//!
//! To run this server inferlet:
//!
//! ```bash
//! pie http --path ./http-server.wasm --port 8080
//! ```
//!
//! Then send requests:
//!
//! ```bash
//! curl http://localhost:8080/
//! curl http://localhost:8080/wait
//! curl -X POST -d "Hello" http://localhost:8080/echo
//! ```

use wstd::http::body::{BodyForthcoming, IncomingBody};
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Request, Response, StatusCode};
use wstd::io::{AsyncWrite, copy, empty};
use wstd::time::{Duration, Instant};

#[wstd::http_server]
async fn main(req: Request<IncomingBody>, res: Responder) -> Finished {
    // Route requests based on path
    let path = req.uri().path();

    match path {
        "/" => home(req, res).await,
        "/wait" => wait(req, res).await,
        "/echo" => echo(req, res).await,
        "/echo-headers" => echo_headers(req, res).await,
        "/info" => info(req, res).await,
        "/sse" => sse_test(req, res).await,
        _ => not_found(req, res).await,
    }
}

/// Home page handler - returns a simple greeting
async fn home(_req: Request<IncomingBody>, res: Responder) -> Finished {
    let body = "Hello from the Pie HTTP Server Inferlet!\n\
                \n\
                Available endpoints:\n\
                  /       - This page\n\
                  /wait   - Async sleep demo (sleeps for 1 second)\n\
                  /echo   - Echo back the request body\n\
                  /echo-headers - Echo back request headers\n\
                  /info   - Server and request information\n"
        .into_body();

    res.respond(Response::new(body)).await
}

/// Demonstrates async sleep and timing functionality
async fn wait(_req: Request<IncomingBody>, res: Responder) -> Finished {
    // Get the time now
    let now = Instant::now();

    // Sleep for one second
    wstd::task::sleep(Duration::from_secs(1)).await;

    // Compute how long we actually slept
    let elapsed = Instant::now().duration_since(now).as_millis();

    // Stream the response body (demonstrates BodyForthcoming pattern)
    let mut body = res.start_response(Response::new(BodyForthcoming));
    let result = body
        .write_all(format!("Slept for {} milliseconds\n", elapsed).as_bytes())
        .await;

    Finished::finish(body, result, None)
}

/// Echoes back the request body
async fn echo(mut req: Request<IncomingBody>, res: Responder) -> Finished {
    // Stream data from the request body to the response body
    let mut body = res.start_response(Response::new(BodyForthcoming));
    let result = copy(req.body_mut(), &mut body).await;

    Finished::finish(body, result, None)
}

/// Echoes back the request headers as response headers
async fn echo_headers(req: Request<IncomingBody>, responder: Responder) -> Finished {
    let mut res = Response::builder();

    // Copy request headers to response headers
    *res.headers_mut().unwrap() = req.into_parts().0.headers;

    let response = res.body(empty()).unwrap();
    responder.respond(response).await
}

/// Returns server and request information as JSON
async fn info(req: Request<IncomingBody>, res: Responder) -> Finished {
    let method = format!("{:?}", req.method());
    let path = req.uri().path().to_string();
    let query = req.uri().query().unwrap_or("").to_string();

    // Build a simple JSON response manually
    let json = format!(
        r#"{{
  "method": "{}",
  "path": "{}",
  "query": "{}",
  "message": "Server inferlet running successfully!"
}}
"#,
        method, path, query
    );

    let response = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();

    res.respond(response).await
}

/// 404 Not Found handler
async fn not_found(_req: Request<IncomingBody>, responder: Responder) -> Finished {
    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body("404 Not Found\n".into_body())
        .unwrap();

    responder.respond(response).await
}

/// SSE streaming test - sends events with flush after each one
async fn sse_test(_req: Request<IncomingBody>, res: Responder) -> Finished {
    use wstd::time::Duration;

    // Start SSE response
    let sse_response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(BodyForthcoming)
        .unwrap();

    let mut body = res.start_response(sse_response);

    // Send 5 SSE events with delays
    for i in 1..=5 {
        let event = format!("event: message\ndata: {{\"count\": {}}}\n\n", i);

        // Write the event
        if let Err(e) = body.write_all(event.as_bytes()).await {
            return Finished::finish(body, Err(e), None);
        }

        // CRITICAL: Flush to push data to client immediately
        if let Err(e) = body.flush().await {
            return Finished::finish(body, Err(e), None);
        }

        // Small delay between events
        wstd::task::sleep(Duration::from_millis(100)).await;
    }

    // Send done event
    let done = "data: [DONE]\n\n";
    let result = body.write_all(done.as_bytes()).await;

    Finished::finish(body, result, None)
}
