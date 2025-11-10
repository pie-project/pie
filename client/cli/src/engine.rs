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
    pub private_key: Option<ParsedPrivateKey>,
    pub enable_auth: bool,
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
        let config_file =
            if host.is_none() || port.is_none() || username.is_none() || private_key_path.is_none()
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

        // Get enable_auth setting (default to true if not specified)
        let enable_auth = config_file
            .as_ref()
            .and_then(|cfg| cfg.enable_auth)
            .unwrap_or(true);

        let mut private_key = None;

        // Load the private key only if authentication is enabled
        if enable_auth {
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
                .context("Private key is required when authentication is enabled")?;

            // Check private key file permissions (Unix only)
            #[cfg(unix)]
            path::check_private_key_permissions(&private_key_path)?;

            // Read and parse the private key from the file if a path is provided
            let key_content = fs::read_to_string(&private_key_path).context(format!(
                "Failed to read private key file at {:?}",
                private_key_path
            ))?;
            private_key = Some(ParsedPrivateKey::parse(&key_content).context(format!(
                "Failed to parse private key from {:?}",
                private_key_path
            ))?);
        }

        Ok(Self {
            host,
            port,
            username,
            private_key,
            enable_auth,
        })
    }
}

/// Submits an inferlet to the engine and returns the instance.
pub async fn submit_inferlet(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
    detached: bool,
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

    let cmd_name = inferlet_path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let instance = client
        .launch_instance(hash, cmd_name, arguments, detached)
        .await?;
    println!("✅ Inferlet launched with ID: {}", instance.id());
    Ok(instance)
}

/// Streams the output of an inferlet.
pub async fn stream_inferlet_output(mut instance: Instance) -> Result<()> {
    let instance_id = instance.id().to_string();
    let short_id = instance_id[..instance_id.len().min(8)].to_string();
    let mut at_line_start_stdout = true;
    let mut at_line_start_stderr = true;

    loop {
        let event = match instance.recv().await {
            Ok(ev) => ev,
            Err(e) => {
                println!("[Instance {}] ReceiveError: {}", short_id, e);
                return Err(e);
            }
        };
        match event {
            // Handle events that have a specific code and a text message.
            InstanceEvent::Event { code, message } => {
                // Format the output string.
                // Using the Debug representation of `code` is a clean way to get its name (e.g., "Completed").
                println!("[Instance {}] {:?}: {}", short_id, code, message);

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
            // Handle streaming stdout
            InstanceEvent::Stdout(content) => {
                handle_streaming_output(false, content, &short_id, &mut at_line_start_stdout).await;
            }
            // Handle streaming stderr
            InstanceEvent::Stderr(content) => {
                handle_streaming_output(true, content, &short_id, &mut at_line_start_stderr).await;
            }
            // If we receive a raw data blob, we'll ignore it and wait for the next event.
            InstanceEvent::Blob(_) => continue,
        }
    }
}

/// Helper to handle streaming output (stdout or stderr).
async fn handle_streaming_output(
    is_stderr: bool,
    content: String,
    short_id: &str,
    at_line_start: &mut bool,
) {
    let at_start = *at_line_start;
    let short_id = short_id.to_string();
    *at_line_start = tokio::task::spawn_blocking(move || {
        let mut at_start_local = at_start;
        if is_stderr {
            write_with_prefix(
                std::io::stderr().lock(),
                &content,
                &short_id,
                &mut at_start_local,
            );
        } else {
            write_with_prefix(
                std::io::stdout().lock(),
                &content,
                &short_id,
                &mut at_start_local,
            );
        }
        at_start_local
    })
    .await
    .unwrap_or(at_start);
}

/// Helper function to print output with instance ID prefix only at the start of new lines.
fn write_with_prefix(
    mut writer: impl std::io::Write,
    content: &str,
    short_id: &str,
    at_line_start: &mut bool,
) {
    if content.is_empty() {
        return;
    }

    let lines = content.split('\n');
    let mut first = true;

    for line in lines {
        if !first {
            // We encountered a '\n' separator, print it
            let _ = writeln!(writer);
            *at_line_start = true;
        }
        first = false;

        // Add prefix only if we're at line start and the line is non-empty
        if !line.is_empty() {
            if *at_line_start {
                let _ = write!(writer, "[Instance {}] ", short_id);
                *at_line_start = false;
            }
            let _ = write!(writer, "{}", line);
        }
    }
}

/// Connects to the engine and authenticates the client if authentication is enabled.
pub async fn connect_and_authenticate(client_config: &ClientConfig) -> Result<Client> {
    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    let client = match Client::connect(&url).await {
        Ok(c) => c,
        Err(_) => {
            anyhow::bail!("Could not connect to engine at {}. Is it running?", url);
        }
    };

    let result = client
        .authenticate(&client_config.username, &client_config.private_key)
        .await;

    if client_config.enable_auth {
        result.context("Failed to authenticate with engine using the specified private key")?;
    } else {
        result.context(
            "Failed to authenticate with engine (client public key authentication disabled)",
        )?;
    }

    Ok(client)
}
