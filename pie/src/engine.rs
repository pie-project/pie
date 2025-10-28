use anyhow::{Context, Result};
use pie_client::auth;
use rand::TryRngCore;
use rand::rngs::OsRng;
use std::fs;
use std::path::PathBuf;
use tokio::sync::oneshot;

// Re-export core components from internal modules
use crate::kvs::KeyValueStore;
use crate::messaging::{PubSub, PushPull};
use crate::runtime::Runtime;
use crate::server::Server;
use crate::service::install_service;

/// Configuration for the PIE engine.
#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub auth_secret: String,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log: Option<PathBuf>,
}

/// Runs the PIE server logic within an existing Tokio runtime.
///
/// This async function sets up all the engine's services and listens for a shutdown
/// signal to terminate gracefully.
pub async fn run_server(
    config: Config,
    ready_tx: oneshot::Sender<String>,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<()> {
    // Ensure the cache directory exists
    fs::create_dir_all(&config.cache_dir).with_context(|| {
        let err_msg = format!(
            "Setup failure: could not create cache dir at {:?}",
            &config.cache_dir
        );
        tracing::error!(error = %err_msg);
        err_msg
    })?;

    if config.enable_auth {
        tracing::info!("Authentication is enabled.");
        auth::init_secret(&config.auth_secret);
        let token = auth::create_jwt("default", auth::Role::User)?;
        tracing::info!("Use this token to authenticate: {}", token);
    } else {
        tracing::info!("Authentication is disabled.");
    }

    // Set up core services
    let runtime = Runtime::new(&config.cache_dir);
    runtime.load_existing_programs()?;

    let server_url = format!("{}:{}", config.host, config.port);

    // Generate a random 64-character string for internal client connection authentication.
    // Use OsRng for cryptographic randomness.
    const CHARSET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let internal_auth_token: String = (0..64)
        .map(|_| {
            OsRng
                .try_next_u32()
                .map(|n| CHARSET[n as usize % CHARSET.len()] as char)
        })
        .collect::<Result<String, _>>()?;

    let server = Server::new(&server_url, config.enable_auth, internal_auth_token.clone());
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();
    let kv_store = KeyValueStore::new();

    install_service("runtime", runtime);
    install_service("server", server);
    install_service("kvs", kv_store);
    install_service("messaging-inst2inst", messaging_inst2inst);
    install_service("messaging-user2inst", messaging_user2inst);

    tracing::info!("✅ PIE runtime started successfully on {}", server_url);
    ready_tx.send(internal_auth_token).unwrap();

    shutdown_rx.await?;
    tracing::info!("Shutdown signal received, shutting down.");

    Ok(())
}
