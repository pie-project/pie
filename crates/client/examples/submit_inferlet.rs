//! Submit a single inferlet to a running Pie engine and print its return
//! value. Boots no engine of its own: point it at an already-running
//! engine's client WebSocket, and it adds the inferlet program, launches
//! it, and prints the `Return` value (the inferlet's `Result<String>`).
//!
//! Usage:
//! ```text
//! cargo run -p pie-client --example submit_inferlet -- \
//!     <ws_host> <inferlet_name@version> <wasm_path> <manifest_path> [input_json]
//! ```
//! e.g.
//! ```text
//! cargo run -p pie-client --example submit_inferlet -- \
//!     ws://127.0.0.1:9123 mirostat@0.1.0 \
//!     runtime/tests/inferlets/target/wasm32-wasip2/debug/mirostat.wasm \
//!     runtime/tests/inferlets/mirostat/Pie.toml '{}'
//! ```
//! Exits non-zero if the inferlet returns an error or never produces a result.

use std::path::Path;

use ::client::client::Client;
use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <ws_host> <inferlet_name@version> <wasm_path> <manifest_path> [input_json]",
            args[0]
        );
        std::process::exit(2);
    }
    let ws_host = &args[1];
    let inferlet = &args[2];
    let wasm_path = Path::new(&args[3]);
    let manifest_path = Path::new(&args[4]);
    let input = args.get(5).cloned().unwrap_or_else(|| "{}".to_string());

    let result = submit_inferlet(ws_host, inferlet, wasm_path, manifest_path, &input).await?;
    println!("{result}");
    Ok(())
}

/// Connect to an already-running engine, register the inferlet program,
/// launch it with `input`, and return the inferlet's `Return` value.
/// Mirrors the canonical `pie` CLI submit flow (connect, authenticate,
/// add, launch, recv).
pub async fn submit_inferlet(
    ws_host: &str,
    inferlet: &str,
    wasm_path: &Path,
    manifest_path: &Path,
    input: &str,
) -> Result<String> {
    // the gateway's `/v1/ws` upgrade rejects a missing `x-pie-identity`, so a
    // standalone engine needs the header supplied here.
    let identity = std::env::var("PIE_IDENTITY").unwrap_or_else(|_| "test-user".to_string());
    let client = Client::connect_with_identity(ws_host, &identity)
        .await
        .with_context(|| format!("connect to engine at {ws_host}"))?;

    // no-auth path: the bench/test engine disables public-key auth.
    client
        .authenticate("test-user", &None)
        .await
        .context("authenticate")?;

    client
        .add_program(wasm_path, manifest_path, true)
        .await
        .with_context(|| format!("add_program {inferlet}"))?;

    let mut proc = client
        .launch_process(inferlet.to_string(), input.to_string(), true)
        .await
        .with_context(|| format!("launch_process {inferlet}"))?;

    // forwards inferlet stdout/stderr for live debugging.
    proc.wait_for_return().await
}
