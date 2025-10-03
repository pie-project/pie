use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use pie::client::InstanceEvent;
use pie::server::EventCode;
use pie::{
    Config as EngineConfig,
    auth::{create_jwt, init_secret},
    client::{self, Client},
};
use rand::{Rng, distr::Alphanumeric};
use reqwest::Client as HttpClient;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, Helper}; // The Helper trait is still needed
use serde::Deserialize;
use std::sync::Arc;
use std::{
    env,
    fs::{self},
    io::{self, Write},
    path::PathBuf,
    process::Stdio, // MODIFIED: Added for process spawning
};
use tokio::io::BufReader;
use tokio::process::Command as TokioCommand;
use tokio::sync::Mutex;
use tokio::sync::oneshot;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, FmtSubscriber, Layer};
//================================================================================//
// SECTION: CLI Command & Config Structs
//================================================================================//

#[derive(Parser, Debug)]
#[command(author, version, about = "PIE Command Line Interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the PIE engine and enter an interactive session.
    Serve(ServeArgs),
    #[command(subcommand)]
    /// Manage local models.
    Model(ModelCommands),
}

#[derive(Args, Debug)]
/// Arguments for starting the PIE engine.
pub struct ServeArgs {
    /// Path to a custom TOML configuration file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// The network host to bind to.
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to use.
    #[arg(long)]
    pub port: Option<u16>,
    /// Disable authentication.
    #[arg(long)]
    pub no_auth: bool,
    /// A log file to write to.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Enable verbose console logging.
    #[arg(long, short)]
    pub verbose: bool,
}

/// Helper for clap to expand `~` in path arguments.
fn expand_tilde(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(shellexpand::tilde(s).as_ref()))
}

#[derive(Parser, Debug)]
/// Arguments to submit an inferlet (Wasm program) to the engine in the shell.
pub struct ShellRunArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = expand_tilde)]
    pub wasm_path: PathBuf,

    /// Run the inferlet in the background and print its instance ID.
    #[arg(long, short)]
    pub detach: bool,

    /// Arguments to pass to the Wasm program.
    pub arguments: Vec<String>,
}

// Other command structs (ModelCommands, etc.) remain the same
#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    List,
    Add(AddModelArgs),
    Remove(RemoveModelArgs),
}
#[derive(Args, Debug)]
pub struct AddModelArgs {
    pub model_name: String,
}
#[derive(Args, Debug)]
pub struct RemoveModelArgs {
    pub model_name: String,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Debug, Default)]
struct ConfigFile {
    host: Option<String>,
    port: Option<u16>,
    enable_auth: Option<bool>,
    auth_secret: Option<String>,
    cache_dir: Option<PathBuf>,
    verbose: Option<bool>,
    log: Option<PathBuf>,
    #[serde(default)]
    backend: Vec<toml::Value>,
}

// Helper struct for what client commands need to know
struct ClientConfig {
    host: String,
    port: u16,
    auth_secret: String,
}

//================================================================================//
// SECTION: Main Entrypoint
//================================================================================//

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => {
            // MODIFIED: Build both engine and backend configs.
            let (engine_config, backend_configs) = build_configs(&args)?;

            // 2. Initialize logging based on the config and get the file-writer guard
            let _guard = init_logging(&engine_config)?;

            // 3. Start the interactive session, passing both configs
            start_interactive_session(engine_config, backend_configs).await?;
        }
        Commands::Model(cmd) => {
            // Model commands don't start the engine, so they can use a simple logger
            let subscriber = FmtSubscriber::builder()
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
            handle_model_command(cmd).await?;
        }
    }
    Ok(())
}
//================================================================================//
// SECTION: Command Handlers
//================================================================================//

// The incorrect `#[derive]` attribute has been removed.
struct MyHelper;

// To satisfy the `Helper` trait bounds, we must implement all its component traits.
// For now, we'll provide empty implementations for the ones we don't need.

impl Completer for MyHelper {
    type Candidate = String;
}

impl Hinter for MyHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        None // No hints for now
    }
}

// Your existing Highlighter implementation is correct.
impl Highlighter for MyHelper {}

impl Validator for MyHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None)) // No validation
    }
}

// Finally, we implement the `Helper` marker trait itself.
impl Helper for MyHelper {}

/// Starts the engine and drops into a client command-prompt session.
async fn start_interactive_session(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
) -> Result<()> {
    // 1. Initialize engine and client configurations
    let client_config = ClientConfig {
        host: engine_config.host.clone(),
        port: engine_config.port,
        auth_secret: engine_config.auth_secret.clone(),
    };
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    // 2. Start the main PIE engine server in a background task
    println!("🚀 Starting PIE engine in background...");
    let server_handle = tokio::spawn(async move {
        if let Err(e) = pie::run_server(engine_config, shutdown_rx).await {
            eprintln!("\n[Engine Error] Engine failed: {}", e);
        }
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    println!("✅ Engine started.");

    // 3. Initialize the interactive prompt with highlighting and an external printer
    println!("Entering interactive session. Type 'help' for commands or use ↑/↓ for history.");
    let mut rl = Editor::new()?;
    rl.set_helper(Some(MyHelper)); // Enable our custom highlighter
    let printer: Arc<Mutex<dyn rustyline::ExternalPrinter + Send>> =
        Arc::new(Mutex::new(rl.create_external_printer()?));
    let history_path = get_pie_home()?.join(".pie_history");
    let _ = rl.load_history(&history_path);

    // 4. Launch all configured backend services
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

            let mut cmd = if backend_type == "python" {
                let mut cmd = TokioCommand::new("uv");
                cmd.arg("--project");
                cmd.arg("../backend/backend-python");
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

            // Make sure the backend process is a process group leader.
            unsafe {
                cmd.pre_exec(|| {
                    nix::unistd::setsid()?;

                    // On Linux, ask the kernel to send SIGKILL to this process when
                    // the parent (the Rust program) dies. This handles accidental termination.
                    #[cfg(target_os = "linux")]
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

            // Clone the Arc for the new task.
            let printer_stdout = Arc::clone(&printer);
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
                            let mut p = printer_stdout.lock().await;
                            // Use `print!` to avoid adding an extra newline
                            p.print(format!("[Backend] {}", output)).unwrap();
                        }
                        Err(e) => {
                            // Handle read error, e.g., print it and break
                            let mut p = printer_stdout.lock().await;
                            p.print(format!("[Backend Read Error] {}", e)).unwrap();
                            break;
                        }
                    }
                }
            });

            // Clone the Arc again for the stderr task.
            let printer_stderr = Arc::clone(&printer);
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = BufReader::new(stderr);
                let mut buffer = [0; 1024];
                loop {
                    match reader.read(&mut buffer).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let output = String::from_utf8_lossy(&buffer[..n]);
                            let mut p = printer_stderr.lock().await;
                            p.print(format!("[Backend] {}", output)).unwrap();
                        }
                        Err(e) => {
                            let mut p = printer_stderr.lock().await;
                            p.print(format!("[Backend Read Error] {}", e)).unwrap();
                            break;
                        }
                    }
                }
            });

            backend_processes.push(child);
        }
    }

    // 5. Start the main interactive loop
    loop {
        match rl.readline("pie> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let parts: Vec<String> = match shlex::split(&line) {
                    Some(parts) => parts,
                    None => {
                        eprintln!("Error: Mismatched quotes in command.");
                        continue;
                    }
                };
                if parts.is_empty() {
                    continue;
                }

                // FIX: Pass the printer to the command handler.
                match handle_interactive_command(
                    &parts[0],
                    &parts[1..].iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                    &client_config,
                    &printer,
                )
                .await
                {
                    Ok(should_exit) if should_exit => break,
                    Ok(_) => (),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => println!("(To exit, type 'exit' or press Ctrl-D)"),
            Err(ReadlineError::Eof) => {
                println!("Exiting...");
                break;
            }
            Err(err) => {
                eprintln!("Error reading line: {}", err);
                break;
            }
        }
    }

    // 6. Begin graceful shutdown
    println!("Shutting down services...");
    if let Err(err) = rl.save_history(&history_path) {
        eprintln!("Warning: Failed to save command history: {}", err);
    }

    // Iterate through the child processes, signal them, and wait for them to exit.
    for mut child in backend_processes {
        // <-- Make `child` mutable to call .wait()
        if let Some(pid) = child.id() {
            let pgid = nix::unistd::Pid::from_raw(pid as i32);
            println!("- Terminating backend process group with PID: {}", pid);

            // Send SIGTERM to the entire process group.
            if let Err(e) = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM) {
                eprintln!("  Failed to send SIGTERM to process group {}: {}", pid, e);
            }
        }

        // This prevents the main program from exiting before cleanup is complete.
        if let Err(e) = child.wait().await {
            eprintln!("  Error while waiting for backend process to exit: {}", e);
        }
    }

    let _ = shutdown_tx.send(());
    server_handle.await?;
    println!("✅ Shutdown complete.");

    Ok(())
}

/// Parses and executes commands entered in the interactive session.
async fn handle_interactive_command(
    command: &str,
    args: &[&str],
    client_config: &ClientConfig,
    printer: &Arc<Mutex<dyn rustyline::ExternalPrinter + Send>>,
) -> Result<bool> {
    match command {
        "run" => {
            // Prepend a dummy command name so clap can parse the args slice.
            let clap_args = std::iter::once("run").chain(args.iter().copied());

            match ShellRunArgs::try_parse_from(clap_args) {
                Ok(run_args) => {
                    if let Err(e) = handle_run(run_args, client_config, printer).await {
                        // Use the printer to avoid corrupting the prompt.
                        let mut p = printer.lock().await;
                        p.print(format!("Error running inferlet: {e}")).unwrap();
                    }
                }
                Err(e) => {
                    // Clap's error messages are user-friendly and include usage.
                    let mut p = printer.lock().await;
                    p.print(e.to_string()).unwrap();
                }
            }
        }
        "query" => {
            println!("(Query functionality not yet implemented)");
        }
        "help" => {
            println!("Available commands:");
            println!(
                "  run [--detach] <path> [ARGS]... - Run a .wasm inferlet with optional arguments"
            );
            println!("  query                  - (Placeholder) Query the engine state");
            println!("  exit                   - Exit the PIE session");
            println!("  help                   - Show this help message");
        }
        "exit" => {
            println!("Exiting...");
            return Ok(true);
        }
        _ => {
            println!(
                "Unknown command: '{}'. Type 'help' for a list of commands.",
                command
            );
        }
    }
    Ok(false)
}

/// Connects to the engine and executes a Wasm inferlet.
async fn handle_run(
    args: ShellRunArgs,
    client_config: &ClientConfig,
    printer: &Arc<Mutex<dyn rustyline::ExternalPrinter + Send>>,
) -> Result<()> {
    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    let client = match Client::connect(&url).await {
        Ok(c) => c,
        Err(_) => {
            return Err(anyhow!(
                "Could not connect to engine at {}. Is it running?",
                url
            ));
        }
    };

    let token = create_jwt("default", pie::auth::Role::User)?;
    client.authenticate(&token).await?;

    let wasm_blob = fs::read(&args.wasm_path)
        .with_context(|| format!("Failed to read Wasm file at {:?}", args.wasm_path))?;
    let hash = client::hash_blob(&wasm_blob);
    println!("Inferlet hash: {}", hash);

    if !client.program_exists(&hash).await? {
        client.upload_program(&wasm_blob).await?;
        println!("✅ Inferlet upload successful.");
    }

    let arguments = args.arguments.clone();
    let mut instance = client.launch_instance(&hash, arguments).await?;
    println!("✅ Inferlet launched with ID: {}", instance.id());

    if !args.detach {
        let instance_id = instance.id().to_string();

        // FIX: Clone the Arc<Mutex<...>> for the new task.
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

async fn handle_model_command(command: ModelCommands) -> Result<()> {
    match command {
        ModelCommands::List => {
            println!("📚 Available local models:");
            let models_dir = get_pie_home()?.join("models");
            if !models_dir.exists() {
                println!("  No models found.");
                return Ok(());
            }
            for entry in fs::read_dir(models_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    println!("  - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        ModelCommands::Add(args) => {
            println!("➕ Adding model: {}", args.model_name);

            let models_root = get_pie_home()?.join("models");
            let model_files_dir = models_root.join(&args.model_name);
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));

            if metadata_path.exists() || model_files_dir.exists() {
                print!(
                    "⚠️ Model '{}' already exists. Overwrite? [y/N] ",
                    args.model_name
                );
                io::stdout().flush().context("Failed to flush stdout")?;

                let mut confirmation = String::new();
                io::stdin()
                    .read_line(&mut confirmation)
                    .context("Failed to read user input")?;

                if confirmation.trim().to_lowercase() != "y" {
                    println!("Aborted by user.");
                    return Ok(());
                }
            }
            fs::create_dir_all(&model_files_dir)?;
            println!("Parameters will be stored at {:?}", model_files_dir);

            let model_index_base =
                "https://raw.githubusercontent.com/pie-project/model-index/refs/heads/main";
            let metadata_url = format!("{}/{}.toml", model_index_base, args.model_name);

            let metadata_raw =
                match download_file_with_progress(&metadata_url, "Downloading metadata...").await {
                    Ok(data) => data,
                    Err(e) => {
                        if let Some(req_err) = e.downcast_ref::<reqwest::Error>() {
                            if req_err.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                                anyhow::bail!(
                                    "Model '{}' not found in the official index.",
                                    args.model_name
                                );
                            }
                        }
                        return Err(e.context("Failed to download model metadata"));
                    }
                };
            let metadata_str = String::from_utf8(metadata_raw)
                .context("Failed to parse model metadata as UTF-8")?;
            let metadata: toml::Value = toml::from_str(&metadata_str)?;

            fs::write(&metadata_path, &metadata_str)?;

            if let Some(source) = metadata.get("source").and_then(|s| s.as_table()) {
                for (name, url_val) in source {
                    if let Some(url) = url_val.as_str() {
                        let file_data =
                            download_file_with_progress(url, &format!("Downloading {}...", name))
                                .await?;
                        fs::write(model_files_dir.join(name), file_data)?;
                    }
                }
            }
            println!("✅ Model '{}' added successfully!", args.model_name);
        }
        ModelCommands::Remove(args) => {
            println!("🗑️ Removing model: {}", args.model_name);
            let models_root = get_pie_home()?.join("models");
            let model_files_dir = models_root.join(&args.model_name);
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));
            let mut was_removed = false;
            if model_files_dir.exists() {
                fs::remove_dir_all(&model_files_dir)?;
                was_removed = true;
            }
            if metadata_path.exists() {
                fs::remove_file(&metadata_path)?;
                was_removed = true;
            }
            if was_removed {
                println!("✅ Model '{}' removed.", args.model_name);
            } else {
                anyhow::bail!("Model '{}' not found locally.", args.model_name);
            }
        }
    }
    Ok(())
}

//================================================================================//
// SECTION: Helpers
//================================================================================//

/// Merges config from file and CLI args to create the final EngineConfig.
// MODIFIED: Renamed to build_configs and now returns backend configs as well.
fn build_configs(args: &ServeArgs) -> Result<(EngineConfig, Vec<toml::Value>)> {
    let file_config: ConfigFile = if let Some(path) = &args.config {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file at {:?}", path))?;
        toml::from_str(&content)?
    } else {
        ConfigFile::default()
    };

    let enable_auth = if args.no_auth {
        false
    } else {
        file_config.enable_auth.unwrap_or(true)
    };
    let auth_secret = file_config.auth_secret.unwrap_or_else(|| {
        rand::rng()
            .sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    });

    let engine_config = EngineConfig {
        host: args
            .host
            .clone()
            .or(file_config.host)
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        port: args.port.or(file_config.port).unwrap_or(8080),
        enable_auth,
        auth_secret,
        cache_dir: file_config
            .cache_dir
            .unwrap_or_else(|| get_pie_home().unwrap().join("programs")),
        verbose: args.verbose || file_config.verbose.unwrap_or(false),
        log: args.log.clone().or(file_config.log),
    };

    // Return both the engine config and the backend configs
    Ok((engine_config, file_config.backend))
}

fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        dirs::cache_dir()
            .map(|p| p.join("pie"))
            .ok_or_else(|| anyhow!("Failed to find home dir"))
    }
}

async fn download_file_with_progress(url: &str, message: &str) -> Result<Vec<u8>> {
    let client = HttpClient::new();
    let res = client.get(url).send().await?.error_for_status()?;
    let total_size = res.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")?
            .progress_chars("##-"),
    );
    pb.set_message(message.to_string());

    let mut downloaded = 0;
    let mut stream = res.bytes_stream();
    let mut content = Vec::with_capacity(total_size as usize);

    while let Some(item) = stream.next().await {
        let chunk = item?;
        downloaded += chunk.len();
        content.extend_from_slice(&chunk);
        pb.set_position(downloaded as u64);
    }
    let file_name = url
        .rsplit('/')
        .next()
        .unwrap_or("file")
        .split('?')
        .next()
        .unwrap_or("file");

    pb.finish_with_message(format!("✅ Downloaded {}", file_name));
    Ok(content)
}

fn init_logging(config: &EngineConfig) -> Result<Option<WorkerGuard>> {
    let mut guard = None;

    // Console logger setup
    let console_filter = if config.verbose {
        // If -v is passed, show info for everything
        EnvFilter::new("info")
    } else {
        // Otherwise, use RUST_LOG or default to "warn"
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stdout)
        .with_filter(console_filter);

    // File logger setup
    let file_layer = if let Some(log_path) = &config.log {
        let parent = log_path
            .parent()
            .context("Log path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create log directory at {:?}", parent))?;

        let file_appender = tracing_appender::rolling::never(parent, log_path.file_name().unwrap());
        let (non_blocking_writer, worker_guard) = tracing_appender::non_blocking(file_appender);

        // Save the guard to be returned
        guard = Some(worker_guard);

        let layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking_writer)
            .with_ansi(false) // No colors in files
            .with_filter(EnvFilter::new("info")); // Log `INFO` and above to the file

        Some(layer)
    } else {
        None
    };

    // Register the layers
    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    Ok(guard)
}
