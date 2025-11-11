//! Engine client utilities for the Pie CLI.

use crate::path;
use anyhow::{Context, Result};
use pie_client::client::{Client, Instance, InstanceEvent};
use pie_client::crypto::ParsedPrivateKey;
use pie_client::message::EventCode;
use std::fs;
use std::path::PathBuf;
use tokio::signal;

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

/// Streams the output of an inferlet.
///
/// Behavior:
/// - Ctrl-C (SIGINT): Sends a terminate request to the server to kill the inferlet
/// - Ctrl-D (EOF): Detaches from the inferlet (continues running on server)
pub async fn stream_inferlet_output(mut instance: Instance, client: Client) -> Result<()> {
    let instance_id = instance.id().to_string();
    let short_id = instance_id[..instance_id.len().min(8)].to_string();
    let mut at_line_start_stdout = true;
    let mut at_line_start_stderr = true;

    // Set up Ctrl-C signal handler
    let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
        .context("Failed to set up SIGINT handler")?;

    // We use a separate OS thread with blocking I/O to monitor stdin for EOF, rather than
    // using `tokio::io::stdin()`. This is because async stdin puts the terminal in non-blocking
    // mode, which can leave the terminal in an inconsistent state if we exit through other
    // paths (e.g., when the inferlet completes naturally). The blocking thread approach keeps
    // the terminal in normal mode and ensures clean process termination.
    let (eof_tx, mut eof_rx) = tokio::sync::mpsc::channel::<()>(1);
    let _stdin_monitor = std::thread::spawn(move || {
        use std::io::{self, Read};
        let mut stdin = io::stdin();
        let mut buf = [0u8; 1];

        loop {
            match stdin.read(&mut buf) {
                Ok(0) | Err(_) => {
                    // EOF detected (Ctrl-D pressed) or stdin closed
                    let _ = eof_tx.blocking_send(());
                    break;
                }
                Ok(_) => {
                    // Ignore any input data and continue monitoring
                    continue;
                }
            }
        }
    });

    // Main event loop: stream output and handle signals
    loop {
        tokio::select! {
            // Ctrl-C: Send termination request to server
            _ = sigint.recv() => {
                println!("\n[Instance {}] Received Ctrl-C, terminating instance ...", short_id);
                if let Err(e) = client.terminate_instance(&instance_id).await {
                    eprintln!("[Instance {}] Failed to send terminate request: {}", short_id, e);
                }
                return Ok(());
            }

            // Ctrl-D: Detach without terminating
            _ = eof_rx.recv() => {
                println!("\n[Instance {}] Detached from instance ...", short_id);
                return Ok(());
            }

            // Process instance events (output, completion, errors)
            event_result = instance.recv() => {
                let event = match event_result {
                    Ok(ev) => ev,
                    Err(e) => {
                        println!("[Instance {}] ReceiveError: {}", short_id, e);
                        return Err(e);
                    }
                };

                match event {
                    InstanceEvent::Event { code, message } => {
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

                    InstanceEvent::Stdout(content) => {
                        handle_streaming_output(false, content, &short_id, &mut at_line_start_stdout).await;
                    }

                    InstanceEvent::Stderr(content) => {
                        handle_streaming_output(true, content, &short_id, &mut at_line_start_stderr).await;
                    }

                    InstanceEvent::Blob(_) => {
                        // Ignore binary blobs
                        continue;
                    }
                }
            }
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
