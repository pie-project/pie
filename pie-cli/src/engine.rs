//! Engine and backend management for the Pie CLI.

use anyhow::{Context, Result};
use pie::client::InstanceEvent;
use pie::server::EventCode;
use pie::{
    Config as EngineConfig,
    auth::{create_jwt, init_secret},
    client::{self, Client},
};
use rand::{Rng, distr::Alphanumeric};
use std::path::Path;
use std::sync::Arc;
use std::{fs, path::PathBuf, process::Stdio};
use tokio::io::BufReader;
use tokio::process::{Child, Command as TokioCommand};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::oneshot::{self, Sender};
use tokio::task::JoinHandle;

use crate::config::ConfigFile;
use crate::output::SharedPrinter;
use crate::path;

// Helper struct for what client commands need to know
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub host: String,
    pub port: u16,
    pub auth_secret: String,
}

/// Parses the engine and backend configuration from files and command line arguments.
pub fn parse_engine_and_backend_config(
    config_path: Option<PathBuf>,
    no_auth: bool,
    host: Option<String>,
    port: Option<u16>,
    verbose: bool,
    log: Option<PathBuf>,
) -> Result<(EngineConfig, Vec<toml::Value>)> {
    let config_str = match config_path {
        Some(path) => fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file at {:?}", path))?,
        None => fs::read_to_string(&path::get_default_config_path()?)
            .context("Failed to read default config file. Try running `pie config init` first.")?,
    };
    let cfg_file: ConfigFile = toml::from_str(&config_str)?;

    let enable_auth = if no_auth {
        false
    } else {
        cfg_file.enable_auth.unwrap_or(true)
    };
    let auth_secret = cfg_file.auth_secret.unwrap_or_else(|| {
        rand::rng()
            .sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    });

    let engine_config = EngineConfig {
        host: host
            .clone()
            .or(cfg_file.host)
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        port: port.or(cfg_file.port).unwrap_or(8080),
        enable_auth,
        auth_secret,
        cache_dir: cfg_file
            .cache_dir
            .unwrap_or_else(|| path::get_pie_home().unwrap().join("programs")),
        verbose: verbose || cfg_file.verbose.unwrap_or(false),
        log: log.clone().or(cfg_file.log),
    };

    if cfg_file.backend.is_empty() {
        anyhow::bail!("No backend configurations found in the configuration file.");
    }

    // Return both the engine config and the backend configs
    Ok((engine_config, cfg_file.backend))
}

/// Starts the Pie engine and all configured backend services.
pub async fn start_engine_and_backend(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
    printer: SharedPrinter,
) -> Result<(Sender<()>, JoinHandle<()>, Vec<Child>, ClientConfig)> {
    // Initialize engine and client configurations
    let client_config = ClientConfig {
        host: engine_config.host.clone(),
        port: engine_config.port,
        auth_secret: engine_config.auth_secret.clone(),
    };
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let (ready_tx, ready_rx) = oneshot::channel();

    // Start the main Pie engine server
    println!("🚀 Starting Pie engine...");
    let server_handle = tokio::spawn(async move {
        if let Err(e) = pie::run_server(engine_config, ready_tx, shutdown_rx).await {
            eprintln!("\n[Engine Error] Engine failed: {}", e);
        }
    });
    ready_rx.await.unwrap();
    println!("✅ Engine started.");

    // Launch all configured backend services
    let mut backend_processes = Vec::new();
    if !backend_configs.is_empty() {
        println!("🚀 Launching backend services...");
        init_secret(&client_config.auth_secret);
        let auth_token = create_jwt("backend-service", pie::auth::Role::User)?;

        for backend_config in &backend_configs {
            let backend_table = backend_config
                .as_table()
                .context("Each [[backend]] entry in config.toml must be a table.")?;
            let backend_type = backend_table
                .get("backend_type")
                .and_then(|v| v.as_str())
                .context("`backend_type` is missing or not a string.")?;
            let exec_path = backend_table
                .get("exec_path")
                .and_then(|v| v.as_str())
                .context("`exec_path` is missing or not a string.")?;
            let exec_parent_path = Path::new(exec_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .context("`exec_path` has no parent directory.")?;

            let mut cmd = if backend_type == "python" {
                let mut cmd = TokioCommand::new("uv");
                cmd.arg("--project");
                cmd.arg(exec_parent_path);
                cmd.arg("run");
                cmd.arg("python");
                cmd.arg("-u");
                cmd.arg(exec_path);
                cmd
            } else {
                TokioCommand::new(exec_path)
            };

            let random_port: u16 = rand::rng().random_range(49152..=65535);
            cmd.arg("--host")
                .arg("localhost")
                .arg("--port")
                .arg(random_port.to_string())
                .arg("--controller_host")
                .arg(&client_config.host)
                .arg("--controller_port")
                .arg(client_config.port.to_string())
                .arg("--auth_token")
                .arg(&auth_token);

            for (key, value) in backend_table {
                if key == "backend_type" || key == "exec_path" {
                    continue;
                }
                cmd.arg(format!("--{}", key))
                    .arg(value.to_string().trim_matches('"').to_string());
            }

            // On Linux, ask the kernel to send SIGKILL to this process when
            // the parent (the Rust program) dies. This handles accidental termination.
            #[cfg(target_os = "linux")]
            unsafe {
                cmd.pre_exec(|| {
                    {
                        // libc::PR_SET_PDEATHSIG is the raw constant for this operation.
                        // SIGKILL is a non-catchable, non-ignorable signal.
                        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) < 0 {
                            // If prctl fails, return an error from the closure.
                            return Err(std::io::Error::last_os_error());
                        }
                    }
                    Ok(())
                });
            }

            println!("- Spawning backend: {}", exec_path);
            let mut child = cmd
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .with_context(|| format!("Failed to spawn backend process: '{}'", exec_path))?;

            // Stream backend output using the external printer to avoid corrupting the prompt
            let stdout = child
                .stdout
                .take()
                .context("Could not capture stdout from backend process.")?;
            let stderr = child
                .stderr
                .take()
                .context("Could not capture stderr from backend process.")?;

            let printer_clone = Arc::clone(&printer);
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = BufReader::new(stdout);
                let mut buffer = [0; 1024]; // Read in 1KB chunks
                loop {
                    match reader.read(&mut buffer).await {
                        Ok(0) => break, // EOF, the child process has closed its stdout
                        Ok(n) => {
                            // We've received `n` bytes. Convert to a string (lossily) and print.
                            let output = String::from_utf8_lossy(&buffer[..n]);
                            // Use `print!` to avoid adding an extra newline
                            printer_clone
                                .lock()
                                .await
                                .print(format!("[Backend] {}", output))
                                .unwrap();
                        }
                        Err(e) => {
                            // Handle read error, e.g., print it and break
                            printer_clone
                                .lock()
                                .await
                                .print(format!("[Backend Read Error] {}", e))
                                .unwrap();
                            break;
                        }
                    }
                }
            });

            let printer_clone = Arc::clone(&printer);
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = BufReader::new(stderr);
                let mut buffer = [0; 1024];
                loop {
                    match reader.read(&mut buffer).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let output = String::from_utf8_lossy(&buffer[..n]);
                            printer_clone
                                .lock()
                                .await
                                .print(format!("[Backend] {}", output))
                                .unwrap();
                        }
                        Err(e) => {
                            printer_clone
                                .lock()
                                .await
                                .print(format!("[Backend Read Error] {}", e))
                                .unwrap();
                            break;
                        }
                    }
                }
            });

            backend_processes.push(child);
        }
    }

    wait_for_backend_ready(&client_config, backend_processes.len()).await?;

    Ok((shutdown_tx, server_handle, backend_processes, client_config))
}

/// Stops the backend heartbeat.
pub async fn stop_backend_heartbeat(client_config: &ClientConfig) -> Result<()> {
    let client = connect_and_authenticate(client_config).await?;
    println!("🔄 Stopping backend heartbeat...");
    client.stop_backend_heartbeat().await?;
    println!("✅ Backend heartbeat stopped.");
    Ok(())
}

/// Terminates the engine and backend processes.
pub async fn terminate_engine_and_backend(
    client_config: &ClientConfig,
    backend_processes: Vec<Child>,
    shutdown_tx: oneshot::Sender<()>,
    server_handle: tokio::task::JoinHandle<()>,
) -> Result<()> {
    println!();

    // Stop the backend heartbeat before sending the signals to the backend processes.
    // This is to avoid broken pipe errors due to sending signals to the backend processes
    // after they have exited.
    stop_backend_heartbeat(client_config).await?;
    println!("🔄 Terminating backend processes...");

    // Iterate through the child processes, signal them, and wait for them to exit.
    for mut child in backend_processes {
        if let Some(pid) = child.id() {
            let pid = nix::unistd::Pid::from_raw(pid as i32);
            println!("🔄 Terminating backend uv process with PID: {}", pid);

            // Send SIGTERM to the `uv` process. It will forward the signal to the backend process.
            if let Err(e) = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM) {
                eprintln!("  Failed to send SIGTERM to uv process {}: {}", pid, e);
            }

            // Wait for the `uv` process to exit. By the time it exits, the backend process will
            // have been terminated.
            let exit_status = child.wait().await;

            if let Err(e) = exit_status {
                eprintln!("  Error while waiting for uv process to exit: {}", e);
            }
        }
    }

    let _ = shutdown_tx.send(());
    server_handle.await?;
    println!("✅ Shutdown complete.");

    Ok(())
}

/// Runs an inferlet on the engine.
pub async fn run_inferlet(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
    detach: bool,
    printer: &SharedPrinter,
) -> Result<()> {
    let client = connect_and_authenticate(client_config).await?;

    let inferlet_blob = fs::read(&inferlet_path)
        .with_context(|| format!("Failed to read Wasm file at {:?}", inferlet_path))?;
    let hash = client::hash_blob(&inferlet_blob);
    println!("Inferlet hash: {}", hash);

    if !client.program_exists(&hash).await? {
        client.upload_program(&inferlet_blob).await?;
        println!("✅ Inferlet upload successful.");
    }

    let mut instance = client.launch_instance(&hash, arguments).await?;
    println!("✅ Inferlet launched with ID: {}", instance.id());

    if !detach {
        let instance_id = instance.id().to_string();

        let printer_clone = Arc::clone(printer);
        tokio::spawn(async move {
            while let Ok(event) = instance.recv().await {
                match event {
                    // Handle events that have a specific code and a text message.
                    InstanceEvent::Event { code, message } => {
                        // Determine if this event signals the end of the instance's execution.
                        // Any event other than a simple 'Message' is considered a final state.
                        let is_terminated = !matches!(code, EventCode::Message);

                        // Format the output string.
                        // Using the Debug representation of `code` is a clean way to get its name (e.g., "Completed").
                        let output = format!("[Inferlet {}] {:?}: {}", instance_id, code, message);

                        // Lock the printer, print the message, and then immediately release the lock.
                        printer_clone.lock().await.print(output).unwrap();

                        // If the instance's execution is finished, break out of the loop.
                        if is_terminated {
                            break;
                        }
                    }
                    // If we receive a raw data blob, we'll ignore it and wait for the next event.
                    InstanceEvent::Blob(_) => continue,
                }
            }
            // No more "Press Enter" message needed!
        });
    }

    Ok(())
}

/// Waits for the instance to finish.
pub async fn wait_for_instance_finish(client_config: &ClientConfig) -> Result<()> {
    let client = connect_and_authenticate(client_config).await?;

    // Query the number of attached, detached, and rejected instances.
    let (mut num_attached, mut num_detached, mut num_rejected) =
        client.wait_instance_change(None, None, None).await?;

    // If no instances are attached, detached, or rejected, wait for a change.
    while num_attached == 0 && num_detached == 0 && num_rejected == 0 {
        (num_attached, num_detached, num_rejected) = client
            .wait_instance_change(Some(0), Some(0), Some(0))
            .await?;
    }

    // We expect either the inferlet was launched successfully (num_attached == 1)
    // or the inferlet was already terminated (num_attached == 0 && num_detached == 1).
    if !((num_attached == 1 && num_detached == 0 && num_rejected == 0)
        || (num_attached == 1 && num_detached == 1 && num_rejected == 0))
    {
        anyhow::bail!(
            "Unexpected instance state: {} instance(s) attached, {} instance(s) detached, {} instance(s) rejected",
            num_attached,
            num_detached,
            num_rejected
        );
    }

    // If the inferlet was just started, wait for it to finish.
    while num_attached == 1 && num_detached == 0 && num_rejected == 0 {
        (num_attached, num_detached, num_rejected) = client
            .wait_instance_change(Some(1), Some(0), Some(0))
            .await?;
    }

    // Check that the inferlet was terminated.
    if !(num_attached == 1 && num_detached == 1 && num_rejected == 0) {
        anyhow::bail!(
            "Unexpected instance state: {} instance(s) attached, {} instance(s) detached, {} instance(s) rejected",
            num_attached,
            num_detached,
            num_rejected
        );
    }

    Ok(())
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

    let token = create_jwt("default", pie::auth::Role::User)?;
    client.authenticate(&token).await?;
    Ok(client)
}

/// Waits for all backend processes to be attached. If any backend process terminates prematurely
/// (before registering), this function will fail immediately.
pub async fn wait_for_backend_ready(
    client_config: &ClientConfig,
    num_backends: usize,
) -> Result<()> {
    let backend_query_client = connect_and_authenticate(&client_config).await?;

    // Set up SIGCHLD signal handler to detect when child processes terminate prematurely
    let mut sigchld =
        signal(SignalKind::child()).context("Failed to set up SIGCHLD signal handler")?;

    // Get initial backend state
    let (mut num_attached, mut num_rejected) = tokio::select! {
        result = backend_query_client.wait_backend_change(None, None) => {
            result?
        }
        _ = sigchld.recv() => {
            anyhow::bail!(
                "Backend process(es) terminated prematurely before registering. \
                 Check backend logs for errors."
            );
        }
    };

    // Wait for all backends to be attached
    while (num_attached as usize) < num_backends && num_rejected == 0 {
        (num_attached, num_rejected) = tokio::select! {
            result = backend_query_client
                .wait_backend_change(
                    Some(num_attached),
                    Some(num_rejected)
                ) => {
                result?
            }
            _ = sigchld.recv() => {
                anyhow::bail!(
                    "Backend process(es) terminated prematurely before registering. \\
                     Check backend logs for errors."
                );
            }
        };
    }

    // We expect no backends to be rejected and the number of attached backends
    // to match the number of backend processes.
    if (num_attached as usize) != num_backends || num_rejected != 0 {
        anyhow::bail!(
            "Unexpected backend state: {} backend(s) attached, {} backend(s) rejected",
            num_attached,
            num_rejected
        );
    }

    Ok(())
}

/// Prints backend statistics using the shared printer.
pub async fn print_backend_stats(
    client_config: &ClientConfig,
    printer: &SharedPrinter,
) -> Result<()> {
    let client = connect_and_authenticate(client_config).await?;
    let stats = client.query_backend_stats().await?;
    let _ =
        crate::output::print_with_printer(printer, format!("Backend runtime stats:\n{}\n", stats))
            .await;
    Ok(())
}
