use std::path::Path;

use ::client::client::Client;
use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <ws_host> <program_path> [input_json]", args[0]);
        std::process::exit(2);
    }
    let ws_host = &args[1];
    let path = Path::new(&args[2]);
    let input = args.get(3).cloned().unwrap_or_else(|| "{}".to_string());

    let result = submit_inferlet(ws_host, path, &input).await?;
    println!("{result}");
    Ok(())
}

pub async fn submit_inferlet(ws_host: &str, path: &Path, input: &str) -> Result<String> {
    let identity = std::env::var("PIE_IDENTITY").unwrap_or_else(|_| "test-user".to_string());
    let client = Client::connect_with_identity(ws_host, &identity)
        .await
        .with_context(|| format!("connect to engine at {ws_host}"))?;

    client
        .authenticate("test-user", &None)
        .await
        .context("authenticate")?;

    let inferlet = client
        .add_program(path, None, true)
        .await
        .with_context(|| format!("add_program {}", path.display()))?;

    let mut proc = client
        .launch_process(inferlet.to_string(), input.to_string(), true)
        .await
        .with_context(|| format!("launch_process {inferlet}"))?;

    proc.wait_for_return().await
}
