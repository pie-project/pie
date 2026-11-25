use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use anyhow::{Context, Result, anyhow};
use clap::Args;

#[derive(Args, Debug)]
pub struct ReloadArgs {
    /// A Hugging Face directory or repo id, or a `.zt` artifact, of the model the server serves.
    pub source: String,

    /// The weights version the server reports from here on (default: one past the current).
    #[arg(long)]
    pub version: Option<u64>,

    /// The running server (`host:port`; default: the config's `[server] listen`).
    #[arg(long, value_name = "ADDR")]
    pub server: Option<String>,
}

pub fn run(args: ReloadArgs, global: &crate::args::GlobalArgs) -> Result<crate::ui::Answer> {
    let source = super::import::resolve_source(&args.source, None, false)?;
    let path = std::fs::canonicalize(&source.path)
        .with_context(|| format!("resolve {}", source.path.display()))?;
    let addr = match args.server {
        Some(addr) => addr,
        None => listen_of(global)?,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let answer = runtime.block_on(async {
        let client = client::client::Client::connect_with_identity(
            &format!("ws://{addr}/v1/ws"),
            "pie-model-reload",
        )
        .await
        .with_context(|| format!("connect to the server at {addr}"))?;
        client.authenticate("pie-model-reload", None).await?;
        let answer = client
            .reload_weights(&path.to_string_lossy(), args.version)
            .await;
        let _ = client.close().await;
        answer.map_err(|why| anyhow!("{why}"))
    })?;
    let reloaded: serde_json::Value = serde_json::from_str(&answer)?;
    let whole = |key: &str| reloaded.get(key).and_then(serde_json::Value::as_u64);
    let seconds = reloaded.get("seconds").and_then(serde_json::Value::as_f64);
    Ok(crate::ui::Answer::did(format!(
        "{addr} serves {} as weights version {} — {} written in {}",
        crate::ui::short_path(&path),
        whole("version").unwrap_or(0),
        crate::ui::bytes(whole("bytes").unwrap_or(0)),
        crate::ui::duration(std::time::Duration::from_secs_f64(seconds.unwrap_or(0.0))),
    )))
}

fn listen_of(global: &crate::args::GlobalArgs) -> Result<String> {
    let (path, _) = crate::args::config_path(global);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {} (or pass `--server host:port`)", path.display()))?;
    let table: toml::Table = toml::from_str(&text)?;
    let listen = table
        .get("server")
        .and_then(|server| server.get("listen"))
        .and_then(toml::Value::as_str)
        .unwrap_or("127.0.0.1:8080");
    let listen: SocketAddr = listen.parse()?;
    let host = match listen.ip() {
        ip if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        ip => ip,
    };
    Ok(SocketAddr::new(host, listen.port()).to_string())
}
