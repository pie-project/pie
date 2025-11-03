//! Engine and backend management for the Pie CLI.

use crate::auth::AuthorizedUsers;
use crate::config::ConfigFile;
use crate::dummy::{self, DummyBackendConfig};
use crate::engine;
use crate::output::SharedPrinter;
use crate::path;
use crate::server::InternalEvent;

use anyhow::{Context, Result};
use engine::Config as EngineConfig;
use pie_client::client::{self, Client};
use pie_client::client::{Instance, InstanceEvent};
use pie_client::message::{EventCode, QUERY_BACKEND_STATS};
use rand::Rng;
use std::path::Path;
use std::{fs, path::PathBuf, process::Stdio};
use tokio::io::BufReader;
use tokio::process::{Child, Command as TokioCommand};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::oneshot::{self, Sender};
use tokio::task::JoinHandle;

// Helper struct for what client commands need to know
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub host: String,
    pub port: u16,
    pub internal_auth_token: Option<String>,
}

/// Helper function to add a TOML value as a command-line argument.
///
/// For boolean values: only adds the flag (--key) if the value is true.
/// For other values: adds both the flag (--key) and the value.
fn add_toml_value_as_arg(cmd: &mut TokioCommand, key: &str, value: &toml::Value) {
    match value {
        toml::Value::Boolean(true) => {
            // For boolean flags that are true, only add the flag without a value
            cmd.arg(format!("--{}", key));
        }
        toml::Value::Boolean(false) => {
            // For boolean flags that are false, don't add anything
        }
        _ => {
            // For all other types, add both the flag and the value
            cmd.arg(format!("--{}", key))
                .arg(value.to_string().trim_matches('"').to_string());
        }
    }
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

    let engine_config = EngineConfig {
        host: host
            .clone()
            .or(cfg_file.host)
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        port: port.or(cfg_file.port).unwrap_or(8080),
        enable_auth,
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
    printer: Option<SharedPrinter>,
) -> Result<(Sender<()>, JoinHandle<()>, Vec<Child>, ClientConfig)> {
    // Initialize engine and client configurations
    let mut client_config = ClientConfig {
        host: engine_config.host.clone(),
        port: engine_config.port,
        internal_auth_token: None,
    };

    let authorized_users_path = path::get_authorized_users_path()?;
    let authorized_users = if authorized_users_path.exists() {
        AuthorizedUsers::load(&authorized_users_path)?
    } else {
        AuthorizedUsers::default()
    };

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let (ready_tx, ready_rx) = oneshot::channel();

    // Start the main Pie engine server
    println!("🚀 Starting Pie engine...");
    let server_handle = tokio::spawn(async move {
        if let Err(e) =
            engine::run_server(engine_config, authorized_users, ready_tx, shutdown_rx).await
        {
            eprintln!("\n[Engine Error] Engine failed: {}", e);
        }
    });
    let internal_auth_token = ready_rx.await.unwrap();
    client_config.internal_auth_token = Some(internal_auth_token);
    println!("✅ Engine started.");

    // Launch all configured backend services
    let mut backend_processes = Vec::new();
    let mut num_dummy_backends = 0;

    if !backend_configs.is_empty() {
        println!("🚀 Launching backend services...");

        for backend_config in &backend_configs {
            let backend_table = backend_config
                .as_table()
                .context("Each [[backend]] entry in config.toml must be a table.")?;
            let backend_type = backend_table
                .get("backend_type")
                .and_then(|v| v.as_str())
                .context("`backend_type` is missing or not a string.")?;

            // Handle dummy backend specially (no exec_path required)
            if backend_type == "dummy" {
                println!("- Starting dummy backend");

                let dummy_config = DummyBackendConfig {
                    controller_host: client_config.host.clone(),
                    controller_port: client_config.port,
                    internal_auth_token: client_config.internal_auth_token.clone().unwrap(),
                };

                // Spawn as detached task - no need to track the handle
                tokio::spawn(async move {
                    if let Err(e) = dummy::start_dummy_backend(dummy_config).await {
                        eprintln!("[Dummy Backend] Error: {}", e);
                    }
                });

                num_dummy_backends += 1;
                continue;
            }

            // For non-dummy backends, exec_path is required
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
                .arg("--internal_auth_token")
                .arg(&client_config.internal_auth_token.as_ref().unwrap());

            for (key, value) in backend_table {
                if key == "backend_type" || key == "exec_path" {
                    continue;
                }
                add_toml_value_as_arg(&mut cmd, key, value);
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

            let printer_clone = printer.clone();
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
                            if let Some(printer) = &printer_clone {
                                printer
                                    .lock()
                                    .await
                                    .print(format!("[Backend] {}", output))
                                    .unwrap();
                            } else {
                                print!("[Backend] {}", output);
                            }
                        }
                        Err(e) => {
                            // Handle read error, e.g., print it and break
                            if let Some(printer) = &printer_clone {
                                printer
                                    .lock()
                                    .await
                                    .print(format!("[Backend Read Error] {}", e))
                                    .unwrap();
                            } else {
                                eprint!("[Backend Read Error] {}", e);
                            }
                            break;
                        }
                    }
                }
            });

            let printer_clone = printer.clone();
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = BufReader::new(stderr);
                let mut buffer = [0; 1024];
                loop {
                    match reader.read(&mut buffer).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let output = String::from_utf8_lossy(&buffer[..n]);
                            if let Some(printer) = &printer_clone {
                                printer
                                    .lock()
                                    .await
                                    .print(format!("[Backend] {}", output))
                                    .unwrap();
                            } else {
                                eprint!("[Backend] {}", output);
                            }
                        }
                        Err(e) => {
                            if let Some(printer) = &printer_clone {
                                printer
                                    .lock()
                                    .await
                                    .print(format!("[Backend Read Error] {}", e))
                                    .unwrap();
                            } else {
                                eprint!("[Backend Read Error] {}", e);
                            }
                            break;
                        }
                    }
                }
            });

            backend_processes.push(child);
        }
    }

    wait_for_backend_ready(backend_processes.len() + num_dummy_backends).await?;

    Ok((shutdown_tx, server_handle, backend_processes, client_config))
}

/// Stops the backend heartbeat.
pub async fn stop_backend_heartbeat() -> Result<()> {
    println!("🔄 Stopping backend heartbeat...");
    let (tx, rx) = oneshot::channel();
    InternalEvent::StopBackendHeartbeat { tx }.dispatch()?;
    rx.await?;
    println!("✅ Backend heartbeat stopped.");
    Ok(())
}

/// Terminates the engine and backend processes.
pub async fn terminate_engine_and_backend(
    backend_processes: Vec<Child>,
    shutdown_tx: oneshot::Sender<()>,
    server_handle: tokio::task::JoinHandle<()>,
) -> Result<()> {
    println!();

    // Stop the backend heartbeat before sending the signals to the backend processes.
    // This is to avoid broken pipe errors due to sending signals to the backend processes
    // after they have exited.
    stop_backend_heartbeat().await?;

    // Note: Dummy backends are spawned as detached tokio tasks and will be automatically
    // terminated when the engine shuts down, since they lose their connection.
    // They are not included in the backend_processes vector.

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

/// Submits an inferlet to the engine but does not wait for it to finish.
pub async fn submit_detached_inferlet(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
    stream_output: bool,
    printer: SharedPrinter,
) -> Result<()> {
    let instance = submit_inferlet(client_config, inferlet_path, arguments).await?;

    if stream_output {
        tokio::spawn(stream_inferlet_output(instance, Some(printer)));
    }

    Ok(())
}

/// Submits an inferlet to the engine and waits for it to finish.
pub async fn submit_inferlet_and_wait(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
    printer: Option<SharedPrinter>,
) -> Result<()> {
    let instance = submit_inferlet(client_config, inferlet_path, arguments).await?;
    stream_inferlet_output(instance, printer).await
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

/// Streams the output of an inferlet to the printer.
async fn stream_inferlet_output(
    mut instance: Instance,
    printer: Option<SharedPrinter>,
) -> Result<()> {
    let instance_id = instance.id().to_string();
    loop {
        let event = match instance.recv().await {
            Ok(ev) => ev,
            Err(e) => {
                // The print operation should not fail.
                if let Some(printer) = &printer {
                    printer
                        .lock()
                        .await
                        .print(format!("[Inferlet {}] ReceiveError: {}", instance_id, e))
                        .unwrap();
                } else {
                    eprint!("[Inferlet {}] ReceiveError: {}", instance_id, e);
                }
                return Err(e);
            }
        };
        match event {
            // Handle events that have a specific code and a text message.
            InstanceEvent::Event { code, message } => {
                // Format the output string.
                // Using the Debug representation of `code` is a clean way to get its name (e.g., "Completed").
                let output = format!("[Inferlet {}] {:?}: {}", instance_id, code, message);

                // The print operation should not fail.
                if let Some(printer) = &printer {
                    printer.lock().await.print(output).unwrap();
                } else {
                    print!("{}", output);
                }

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
    let client = Client::connect(&url)
        .await
        .with_context(|| format!("Could not connect to engine at {}. Is it running?", url))?;

    let token = client_config.internal_auth_token.as_ref().unwrap();
    client.internal_authenticate(token).await?;
    Ok(client)
}

/// Waits for all backend processes to be attached. If any backend process terminates prematurely
/// (before registering), this function will fail immediately.
pub async fn wait_for_backend_ready(num_backends: usize) -> Result<()> {
    // Set up SIGCHLD signal handler to detect when child processes terminate prematurely
    let mut sigchld =
        signal(SignalKind::child()).context("Failed to set up SIGCHLD signal handler")?;

    let (tx, rx) = oneshot::channel();
    InternalEvent::WaitBackendChange {
        cur_num_attached_backends: None,
        cur_num_rejected_backends: None,
        tx,
    }
    .dispatch()?;

    // Get initial backend state
    let (mut num_attached, mut num_rejected) = tokio::select! {
        result = rx => {
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
        let (tx, rx) = oneshot::channel();
        InternalEvent::WaitBackendChange {
            cur_num_attached_backends: Some(num_attached),
            cur_num_rejected_backends: Some(num_rejected),
            tx,
        }
        .dispatch()?;
        (num_attached, num_rejected) = tokio::select! {
            result = rx => {
                result?
            }
            _ = sigchld.recv() => {
                anyhow::bail!(
                    "Backend process(es) terminated prematurely before registering. \
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
    let stats = client.query(QUERY_BACKEND_STATS, "".to_string()).await?;
    let _ =
        crate::output::print_with_printer(printer, format!("Backend runtime stats:\n{}\n", stats))
            .await;
    Ok(())
}
