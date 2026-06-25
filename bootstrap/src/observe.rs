//! Observability — tracing init and a minimal Prometheus-text `/metrics`
//! endpoint.
//!
//! Per ruling R2 (minimal-start, YAGNI): structured `tracing` logs plus a
//! lightweight `/metrics` endpoint on every daemon. The full OTel collector
//! pipeline is deferred behind this seam. The endpoint is a tiny hand-rolled
//! HTTP/1.1 responder (no axum/hyper/metrics-ecosystem dep) serving a base set
//! (`pie_build_info`, `pie_uptime_seconds`); richer metrics can graduate to a
//! `metrics` facade later without changing the bin seam.

use std::net::SocketAddr;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing_subscriber::EnvFilter;

/// Initialise the global tracing subscriber: logs to stderr (stdout stays clean
/// for piping), level from `RUST_LOG` if set, else `log_level`.
pub(crate) fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    // `try_init` so a second init (e.g. in tests) is a no-op rather than a panic.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Bind the `/metrics` listener (synchronously, so a bad/in-use address fails
/// fast in `init`) and spawn its accept loop onto the ambient runtime. `start`
/// anchors the uptime gauge. Must be called from within a tokio runtime context.
pub(crate) fn spawn_metrics(
    addr: SocketAddr,
    start: Instant,
    component: &'static str,
    version: &'static str,
) -> Result<()> {
    // Sync bind → propagate the error out of `init`; then hand the socket to
    // tokio for the async accept loop.
    let std_listener =
        std::net::TcpListener::bind(addr).with_context(|| format!("bind /metrics on {addr}"))?;
    std_listener
        .set_nonblocking(true)
        .context("set /metrics listener non-blocking")?;
    let listener = TcpListener::from_std(std_listener).context("adopt /metrics listener")?;
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((sock, _)) => {
                    tokio::spawn(handle_scrape(sock, start, component, version));
                }
                Err(e) => tracing::warn!("/metrics accept error: {e}"),
            }
        }
    });
    Ok(())
}

/// Answer one scrape: `GET /metrics` → 200 Prometheus text, anything else → 404.
async fn handle_scrape(mut sock: TcpStream, start: Instant, component: &str, version: &str) {
    // The request line is first, so a single read is enough to route.
    let mut buf = [0u8; 1024];
    let n = sock.read(&mut buf).await.unwrap_or(0);
    let req = String::from_utf8_lossy(&buf[..n]);
    let resp = if req.starts_with("GET /metrics") {
        let body = render(start, component, version);
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/plain; version=0.0.4\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    } else {
        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
    };
    let _ = sock.write_all(resp.as_bytes()).await;
}

/// The base Prometheus-text body.
fn render(start: Instant, component: &str, version: &str) -> String {
    let uptime = start.elapsed().as_secs_f64();
    format!(
        "# HELP pie_build_info Build/identity info (always 1).\n\
         # TYPE pie_build_info gauge\n\
         pie_build_info{{component=\"{component}\",version=\"{version}\"}} 1\n\
         # HELP pie_uptime_seconds Seconds since process start.\n\
         # TYPE pie_uptime_seconds gauge\n\
         pie_uptime_seconds {uptime}\n"
    )
}
