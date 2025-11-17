use anyhow::{Context, Result, anyhow};
use ring::rand::{SecureRandom, SystemRandom};
use std::fs;
use std::path::PathBuf;
use tokio::sync::oneshot;

use crate::auth::AuthorizedUsers;
use crate::kvs::KeyValueStore;
use crate::messaging::{PubSub, PushPull};
use crate::runtime::Runtime;
use crate::server::Server;
use crate::service::install_legacy_service;

/// Configuration for the PIE engine.
#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
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
    authorized_users: AuthorizedUsers,
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
    } else {
        tracing::info!("Authentication is disabled.");
    }

    // Set up core services
    let runtime = Runtime::new(&config.cache_dir);
    runtime.load_existing_programs()?;

    let server_url = format!("{}:{}", config.host, config.port);

    // Generate a random 64-character string for internal client connection authentication.
    // Use `ring::rand::SystemRandom` for cryptographic randomness with rejection sampling
    // to avoid modulo bias.
    const CHARSET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let rng = SystemRandom::new();
    let mut internal_auth_token = String::with_capacity(64);

    // Rejection sampling threshold: 256 - (256 % 62) = 248
    // Accept only bytes < 248 to ensure uniform distribution
    let threshold = 256 - (256 % CHARSET.len());

    while internal_auth_token.len() < 64 {
        let mut random_bytes = [0u8; 128];
        rng.fill(&mut random_bytes).map_err(|e| {
            anyhow!(
                "Failed to generate random bytes for internal auth token: {}",
                e
            )
        })?;

        for &byte in &random_bytes {
            if internal_auth_token.len() >= 64 {
                break;
            }

            // Reject bytes >= threshold to avoid modulo bias
            if (byte as usize) < threshold {
                let idx = (byte as usize) % CHARSET.len();
                internal_auth_token.push(CHARSET[idx] as char);
            }
        }
    }

    let server = Server::new(
        &server_url,
        config.enable_auth,
        authorized_users,
        internal_auth_token.clone(),
    );
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();
    let kv_store = KeyValueStore::new();

    install_legacy_service("runtime", runtime);
    install_legacy_service("server", server);
    install_legacy_service("kvs", kv_store);
    install_legacy_service("messaging-inst2inst", messaging_inst2inst);
    install_legacy_service("messaging-user2inst", messaging_user2inst);

    tracing::info!("✅ PIE runtime started successfully on {}", server_url);
    ready_tx.send(internal_auth_token).unwrap();

    shutdown_rx.await?;
    tracing::info!("Shutdown signal received, shutting down.");

    Ok(())
}
