//! Engine client utilities for the Pie CLI.

use crate::path;
use anyhow::{Context, Result};
use pie_client::client::{self, Client, Instance, InstanceEvent};
use pie_client::crypto::ParsedPrivateKey;
use pie_client::message::EventCode;
use std::fs;
use std::path::PathBuf;

// Helper struct for what client commands need to know
pub struct ClientConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub private_key: ParsedPrivateKey,
}

impl ClientConfig {
    pub fn new(
        config_path: Option<PathBuf>,
        host: Option<String>,
        port: Option<u16>,
        username: Option<String>,
        private_key_path: Option<PathBuf>,
    ) -> Result<Self> {
        // Read config file only if when any parameter is missing
        let config_file = if host.is_none()
            || port.is_none()
            || username.is_none()
            || private_key_path.is_none()
        {
            let config_str = match config_path {
                Some(path) => fs::read_to_string(&path)
                    .context(format!("Failed to read config file at {:?}", path))?,
                None => fs::read_to_string(&crate::path::get_default_config_path()?).context(
                    "Failed to read default config file. Try running `pie-cli config init` first.",
                )?,
            };
            Some(toml::from_str::<crate::config::ConfigFile>(&config_str)?)
        } else {
            None
        };

        // Prefer command-line arguments and use config file values if not provided
        let host = host
            .or_else(|| config_file.as_ref().and_then(|cfg| cfg.host.clone()))
            .unwrap_or("127.0.0.1".to_string());

        let port = port
            .or_else(|| config_file.as_ref().and_then(|cfg| cfg.port))
            .unwrap_or(8080);

        let username = username
            .or(config_file.as_ref().and_then(|cfg| cfg.username.clone()))
            .unwrap_or(whoami::username());

        // Get the private key path from either command-line or config file
        let private_key_path = private_key_path
            .or(config_file
                .as_ref()
                .and_then(|cfg| cfg.private_key_path.clone()))
            .map(|p| {
                p.to_str()
                    .map(|s| s.to_owned())
                    .context("Private key path is not a valid UTF-8 string")
            })
            .transpose()?
            .map(|p| path::expand_tilde(&p))
            .transpose()?
            .context("Private key is required for authentication")?;

        // Read and parse the private key from the file if a path is provided
        let key_content = fs::read_to_string(&private_key_path).context(format!(
            "Failed to read private key file at {:?}",
            private_key_path
        ))?;
        let private_key = ParsedPrivateKey::parse(&key_content).context(format!(
            "Failed to parse private key from {:?}",
            private_key_path
        ))?;

        Ok(Self {
            host,
            port,
            username,
            private_key,
        })
    }
}

/// Submits an inferlet to the engine and waits for it to finish.
pub async fn submit_inferlet_and_wait(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<()> {
    let instance = submit_inferlet(client_config, inferlet_path, arguments).await?;
    stream_inferlet_output(instance).await
}

/// Submits an inferlet to the engine and returns the instance.
async fn submit_inferlet(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<Instance> {
    let client = connect_and_authenticate(client_config).await?;

    let inferlet_blob = fs::read(&inferlet_path)
        .with_context(|| format!("Failed to read Wasm file at {:?}", inferlet_path))?;
    let hash = client::hash_blob(&inferlet_blob);
    println!("Inferlet hash: {}", hash);

    if !client.program_exists(&hash).await? {
        client.upload_program(&inferlet_blob).await?;
        println!("✅ Inferlet upload successful.");
    }

    let instance = client.launch_instance(&hash, arguments).await?;
    println!("✅ Inferlet launched with ID: {}", instance.id());
    Ok(instance)
}

/// Streams the output of an inferlet.
async fn stream_inferlet_output(mut instance: Instance) -> Result<()> {
    let instance_id = instance.id().to_string();
    loop {
        let event = match instance.recv().await {
            Ok(ev) => ev,
            Err(e) => {
                println!("[Inferlet {}] ReceiveError: {}", instance_id, e);
                return Err(e);
            }
        };
        match event {
            // Handle events that have a specific code and a text message.
            InstanceEvent::Event { code, message } => {
                // Format the output string.
                // Using the Debug representation of `code` is a clean way to get its name (e.g., "Completed").
                println!("[Inferlet {}] {:?}: {}", instance_id, code, message);

                match code {
                    EventCode::Completed => return Ok(()),
                    EventCode::Message => continue,
                    EventCode::Aborted
                    | EventCode::Exception
                    | EventCode::ServerError
                    | EventCode::OutOfResources => {
                        anyhow::bail!("inferlet terminated with status {:?}", code)
                    }
                }
            }
            // If we receive a raw data blob, we'll ignore it and wait for the next event.
            InstanceEvent::Blob(_) => continue,
        }
    }
}

/// Connects to the engine and authenticates the client.
pub async fn connect_and_authenticate(client_config: &ClientConfig) -> Result<Client> {
    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    let client = match Client::connect(&url).await {
        Ok(c) => c,
        Err(_) => {
            anyhow::bail!("Could not connect to engine at {}. Is it running?", url);
        }
    };

    client
        .authenticate(&client_config.username, &client_config.private_key)
        .await?;
    Ok(client)
}
