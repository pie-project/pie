//! Engine client utilities for the Pie CLI.

use anyhow::{Context, Result};
use libpie::client::{Instance, InstanceEvent};
use libpie::server::EventCode;
use libpie::{
    auth::create_jwt,
    client::{self, Client},
};
use rand::{Rng, distr::Alphanumeric};
use std::fs;
use std::path::PathBuf;

// Helper struct for what client commands need to know
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub host: String,
    pub port: u16,
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
async fn connect_and_authenticate(client_config: &ClientConfig) -> Result<Client> {
    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    let client = match Client::connect(&url).await {
        Ok(c) => c,
        Err(_) => {
            anyhow::bail!("Could not connect to engine at {}. Is it running?", url);
        }
    };

    let token = create_jwt("default", libpie::auth::Role::User)?;
    client.authenticate(&token).await?;
    Ok(client)
}

/// Generates a random authentication secret.
pub fn generate_random_auth_secret() -> String {
    rand::rng()
        .sample_iter(&Alphanumeric)
        .take(64)
        .map(char::from)
        .collect()
}
