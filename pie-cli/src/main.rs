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
use rustyline::history::FileHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, Helper}; // The Helper trait is still needed
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::{
    env,
    fs::{self},
    io::{self, Write},
    path::PathBuf,
    process::Stdio, // MODIFIED: Added for process spawning
};
use tokio::io::BufReader;
use tokio::process::{Child, Command as TokioCommand};
use tokio::sync::Mutex;
use tokio::sync::oneshot::{self, Sender};
use tokio::task::JoinHandle;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, FmtSubscriber, Layer};
//================================================================================//
// SECTION: CLI Command & Config Structs
//================================================================================//

#[derive(Parser, Debug)]
#[command(author, version, about = "Pie Command Line Interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the Pie engine and enter an interactive session.
    Serve(ServeArgs),
    /// Run an inferlet with a one-shot Pie engine.
    Run(RunArgs),
    #[command(subcommand)]
    /// Manage local models.
    Model(ModelCommands),
    #[command(subcommand)]
    /// Manage configuration.
    Config(ConfigCommands),
}

#[derive(Args, Debug)]
/// Arguments for starting the Pie engine.
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

#[derive(Args, Debug)]
/// Arguments to submit an inferlet (Wasm program) to the engine in the shell.
pub struct RunArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = expand_tilde)]
    pub inferlet: PathBuf,
    /// Path to a custom TOML configuration file.
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    /// A log file to write to.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Arguments to pass to the inferlet after `--`.
    #[arg(last = true)]
    pub arguments: Vec<String>,
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
    pub inferlet_path: PathBuf,

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

#[derive(Subcommand, Debug)]
pub enum ConfigCommands {
    /// Create a default config file.
    Init(ConfigInitArgs),
    /// Update the entries of the default config file.
    Update(ConfigUpdateArgs),
    /// Show the content of the default config file.
    Show,
}

#[derive(Args, Debug)]
pub struct ConfigInitArgs {
    /// Backend type (e.g., "python", "metal")
    pub backend_type: String,
    /// Path to the backend executable
    pub exec_path: String,
}

#[derive(Args, Debug)]
pub struct ConfigUpdateArgs {
    // Engine configuration options
    /// The network host to bind to
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to use
    #[arg(long)]
    pub port: Option<u16>,
    /// Enable or disable authentication
    #[arg(long)]
    pub enable_auth: Option<bool>,
    /// Authentication secret
    #[arg(long)]
    pub auth_secret: Option<String>,
    /// Cache directory path
    #[arg(long)]
    pub cache_dir: Option<String>,
    /// Enable verbose logging
    #[arg(long)]
    pub verbose: Option<bool>,
    /// Log file path
    #[arg(long)]
    pub log: Option<String>,

    // Backend configuration options (prefixed with backend-)
    /// Backend type
    #[arg(long)]
    pub backend_type: Option<String>,
    /// Backend executable path
    #[arg(long)]
    pub backend_exec_path: Option<String>,
    /// Model name
    #[arg(long)]
    pub backend_model: Option<String>,
    /// Device (e.g., "cuda:0", "mps")
    #[arg(long)]
    pub backend_device: Option<String>,
    /// Data type (e.g., "bfloat16", "float16")
    #[arg(long)]
    pub backend_dtype: Option<String>,
    /// KV page size
    #[arg(long)]
    pub backend_kv_page_size: Option<i64>,
    /// Maximum batch tokens
    #[arg(long)]
    pub backend_max_batch_tokens: Option<i64>,
    /// Maximum distribution size
    #[arg(long)]
    pub backend_max_dist_size: Option<i64>,
    /// Maximum number of KV pages
    #[arg(long)]
    pub backend_max_num_kv_pages: Option<i64>,
    /// Maximum number of embeddings
    #[arg(long)]
    pub backend_max_num_embeds: Option<i64>,
    /// Maximum number of adapters
    #[arg(long)]
    pub backend_max_num_adapters: Option<i64>,
    /// Maximum adapter rank
    #[arg(long)]
    pub backend_max_adapter_rank: Option<i64>,
    /// GPU memory headroom
    #[arg(long)]
    pub backend_gpu_mem_headroom: Option<f64>,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Serialize, Debug)]
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
            let (engine_config, backend_configs) = build_configs(
                args.config,
                args.no_auth,
                args.host,
                args.port,
                args.verbose,
                args.log,
            )?;

            // Initialize logging based on the config and get the file-writer guard
            let _guard = init_logging(&engine_config)?;

            handle_serve_command(engine_config, backend_configs).await?;
        }
        Commands::Run(args) => {
            // Build both engine and backend configs.
            let (engine_config, backend_configs) =
                build_configs(args.config, false, None, None, false, args.log)?;

            // Initialize logging based on the config and get the file-writer guard
            let _guard = init_logging(&engine_config)?;

            handle_run_command(
                engine_config,
                backend_configs,
                args.inferlet,
                args.arguments,
            )
            .await?;
        }
        Commands::Model(cmd) => {
            // Model commands don't start the engine, so they can use a simple logger
            let subscriber = FmtSubscriber::builder()
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
            handle_model_command(cmd).await?;
        }
        Commands::Config(cmd) => {
            // Config commands don't start the engine, so they can use a simple logger
            let subscriber = FmtSubscriber::builder()
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
            handle_config_command(cmd).await?;
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

/// Handles the `pie serve` command.
async fn handle_serve_command(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
) -> Result<()> {
    let (rl, printer) = create_editor_and_printer().await?;

    // Start the engine and backend services
    let (shutdown_tx, server_handle, backend_processes, client_config) =
        start_engine_and_backend(engine_config, backend_configs, printer.clone()).await?;

    // Start the interactive session, passing both configs
    run_shell(client_config, rl, printer).await?;

    // Terminate the engine and backend services
    terminate_engine_and_backend(backend_processes, shutdown_tx, server_handle).await?;

    Ok(())
}

/// Handles the `pie run` command.
async fn handle_run_command(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<()> {
    let (_rl, printer) = create_editor_and_printer().await?;

    // Start the engine and backend services
    let (shutdown_tx, server_handle, backend_processes, client_config) =
        start_engine_and_backend(engine_config, backend_configs, printer.clone()).await?;

    // Run the inferlet
    run_inferlet(&client_config, inferlet_path, arguments, false, &printer).await?;
    wait_for_instance_finish(&client_config).await?;

    // Terminate the engine and backend services
    terminate_engine_and_backend(backend_processes, shutdown_tx, server_handle).await?;
    Ok(())
}

/// Parses and executes commands in the shell.
async fn handle_shell_command(
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
                    if let Err(e) = run_inferlet(
                        client_config,
                        run_args.inferlet_path,
                        run_args.arguments,
                        run_args.detach,
                        printer,
                    )
                    .await
                    {
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
            println!("  exit                   - Exit the Pie session");
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

/// Handles the `pie model` command.
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

/// Handles the `pie config` command.
async fn handle_config_command(command: ConfigCommands) -> Result<()> {
    match command {
        ConfigCommands::Init(args) => {
            init_default_config_file(&args.exec_path, &args.backend_type)?;
        }
        ConfigCommands::Update(args) => {
            update_default_config_file(args)?;
        }
        ConfigCommands::Show => {
            show_default_config_file()?;
        }
    }
    Ok(())
}

//================================================================================//
// SECTION: Helpers
//================================================================================//

fn build_configs(
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
        None => fs::read_to_string(&get_default_config_path()?)
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
            .unwrap_or_else(|| get_pie_home().unwrap().join("programs")),
        verbose: verbose || cfg_file.verbose.unwrap_or(false),
        log: log.clone().or(cfg_file.log),
    };

    if cfg_file.backend.is_empty() {
        anyhow::bail!("No backend configurations found in the configuration file.");
    }

    // Return both the engine config and the backend configs
    Ok((engine_config, cfg_file.backend))
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

fn get_shell_history_path() -> Result<PathBuf> {
    let pie_home = get_pie_home()?;
    let history_path = pie_home.join(".pie_history");
    Ok(history_path)
}

fn get_default_config_path() -> Result<PathBuf> {
    let pie_home = get_pie_home()?;
    let config_path = pie_home.join("config.toml");
    Ok(config_path)
}

fn create_default_config_content(exec_path: &str, backend_type: &str) -> Result<String> {
    // Create the backend configuration as a TOML table
    let backend_table = [
        (
            "backend_type",
            toml::Value::String(backend_type.to_string()),
        ),
        ("exec_path", toml::Value::String(exec_path.to_string())),
        ("model", toml::Value::String("qwen-3-0.6b".into())),
        ("device", toml::Value::String("cuda:0".into())),
        ("dtype", toml::Value::String("bfloat16".into())),
        ("kv_page_size", toml::Value::Integer(16)),
        ("max_batch_tokens", toml::Value::Integer(10240)),
        ("max_dist_size", toml::Value::Integer(32)),
        ("max_num_kv_pages", toml::Value::Integer(10240)),
        ("max_num_embeds", toml::Value::Integer(128)),
        ("max_num_adapters", toml::Value::Integer(32)),
        ("max_adapter_rank", toml::Value::Integer(8)),
        ("gpu_mem_headroom", toml::Value::Float(10.0)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect::<toml::Table>();

    // Create the ConfigFile object
    let config_file = ConfigFile {
        host: Some("127.0.0.1".to_string()),
        port: Some(8080),
        enable_auth: None,
        auth_secret: None,
        cache_dir: None,
        verbose: None,
        log: None,
        backend: vec![toml::Value::Table(backend_table)],
    };

    // Serialize to TOML string
    let config_content =
        toml::to_string_pretty(&config_file).context("Failed to serialize config to TOML")?;

    Ok(config_content)
}

fn init_default_config_file(exec_path: &str, backend_type: &str) -> Result<()> {
    println!("⚙️ Initializing Pie configuration...");

    let config_path = get_default_config_path()?;

    // Check if config file already exists
    if config_path.exists() {
        print!(
            "⚠️ Configuration file already exists at {:?}. Overwrite? [y/N] ",
            config_path
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

    // Create the directory if it doesn't exist
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config directory at {:?}", parent))?;
    }

    // Create the default config file
    let config_content = create_default_config_content(&exec_path, &backend_type)?;
    fs::write(&config_path, &config_content)
        .with_context(|| format!("Failed to write config file at {:?}", config_path))?;

    println!("✅ Configuration file created at {:?}", config_path);
    println!("Config file content:");
    println!("{}", config_content);

    Ok(())
}

fn update_default_config_file(args: ConfigUpdateArgs) -> Result<()> {
    // Check if any update options were provided
    let has_engine_updates = args.host.is_some()
        || args.port.is_some()
        || args.enable_auth.is_some()
        || args.auth_secret.is_some()
        || args.cache_dir.is_some()
        || args.verbose.is_some()
        || args.log.is_some();

    let has_backend_updates = args.backend_type.is_some()
        || args.backend_exec_path.is_some()
        || args.backend_model.is_some()
        || args.backend_device.is_some()
        || args.backend_dtype.is_some()
        || args.backend_kv_page_size.is_some()
        || args.backend_max_batch_tokens.is_some()
        || args.backend_max_dist_size.is_some()
        || args.backend_max_num_kv_pages.is_some()
        || args.backend_max_num_embeds.is_some()
        || args.backend_max_num_adapters.is_some()
        || args.backend_max_adapter_rank.is_some()
        || args.backend_gpu_mem_headroom.is_some();

    if !has_engine_updates && !has_backend_updates {
        println!("⚠️ No configuration options provided to update.");
        println!("Use `pie config update --help` to see available options.");
        return Ok(());
    }

    println!("⚙️ Updating Pie configuration...");

    let config_path = get_default_config_path()?;

    // Check if config file exists
    if !config_path.exists() {
        anyhow::bail!(
            "Configuration file not found at {:?}. Run `pie config init` first.",
            config_path
        );
    }

    // Read and parse the existing config file
    let config_str = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file at {:?}", config_path))?;
    let mut config_file: ConfigFile = toml::from_str(&config_str)
        .with_context(|| format!("Failed to parse config file at {:?}", config_path))?;

    // Update engine configuration fields
    if let Some(host) = args.host {
        config_file.host = Some(host);
        println!("✅ Updated host");
    }
    if let Some(port) = args.port {
        config_file.port = Some(port);
        println!("✅ Updated port");
    }
    if let Some(enable_auth) = args.enable_auth {
        config_file.enable_auth = Some(enable_auth);
        println!("✅ Updated enable_auth");
    }
    if let Some(auth_secret) = args.auth_secret {
        config_file.auth_secret = Some(auth_secret);
        println!("✅ Updated auth_secret");
    }
    if let Some(cache_dir) = args.cache_dir {
        config_file.cache_dir = Some(PathBuf::from(cache_dir));
        println!("✅ Updated cache_dir");
    }
    if let Some(verbose) = args.verbose {
        config_file.verbose = Some(verbose);
        println!("✅ Updated verbose");
    }
    if let Some(log) = args.log {
        config_file.log = Some(PathBuf::from(log));
        println!("✅ Updated log");
    }

    // Update backend configuration fields
    if has_backend_updates {
        if config_file.backend.is_empty() {
            anyhow::bail!(
                "No backend configuration found in config file. Cannot update backend settings."
            );
        }

        // Update the first backend entry (assuming single backend for now)
        if let Some(toml::Value::Table(backend_table)) = config_file.backend.get_mut(0) {
            if let Some(backend_type) = args.backend_type {
                backend_table.insert(
                    "backend_type".to_string(),
                    toml::Value::String(backend_type),
                );
                println!("✅ Updated backend_type");
            }
            if let Some(exec_path) = args.backend_exec_path {
                backend_table.insert("exec_path".to_string(), toml::Value::String(exec_path));
                println!("✅ Updated backend exec_path");
            }
            if let Some(model) = args.backend_model {
                backend_table.insert("model".to_string(), toml::Value::String(model));
                println!("✅ Updated backend model");
            }
            if let Some(device) = args.backend_device {
                backend_table.insert("device".to_string(), toml::Value::String(device));
                println!("✅ Updated backend device");
            }
            if let Some(dtype) = args.backend_dtype {
                backend_table.insert("dtype".to_string(), toml::Value::String(dtype));
                println!("✅ Updated backend dtype");
            }
            if let Some(kv_page_size) = args.backend_kv_page_size {
                backend_table.insert(
                    "kv_page_size".to_string(),
                    toml::Value::Integer(kv_page_size),
                );
                println!("✅ Updated backend kv_page_size");
            }
            if let Some(max_batch_tokens) = args.backend_max_batch_tokens {
                backend_table.insert(
                    "max_batch_tokens".to_string(),
                    toml::Value::Integer(max_batch_tokens),
                );
                println!("✅ Updated backend max_batch_tokens");
            }
            if let Some(max_dist_size) = args.backend_max_dist_size {
                backend_table.insert(
                    "max_dist_size".to_string(),
                    toml::Value::Integer(max_dist_size),
                );
                println!("✅ Updated backend max_dist_size");
            }
            if let Some(max_num_kv_pages) = args.backend_max_num_kv_pages {
                backend_table.insert(
                    "max_num_kv_pages".to_string(),
                    toml::Value::Integer(max_num_kv_pages),
                );
                println!("✅ Updated backend max_num_kv_pages");
            }
            if let Some(max_num_embeds) = args.backend_max_num_embeds {
                backend_table.insert(
                    "max_num_embeds".to_string(),
                    toml::Value::Integer(max_num_embeds),
                );
                println!("✅ Updated backend max_num_embeds");
            }
            if let Some(max_num_adapters) = args.backend_max_num_adapters {
                backend_table.insert(
                    "max_num_adapters".to_string(),
                    toml::Value::Integer(max_num_adapters),
                );
                println!("✅ Updated backend max_num_adapters");
            }
            if let Some(max_adapter_rank) = args.backend_max_adapter_rank {
                backend_table.insert(
                    "max_adapter_rank".to_string(),
                    toml::Value::Integer(max_adapter_rank),
                );
                println!("✅ Updated backend max_adapter_rank");
            }
            if let Some(gpu_mem_headroom) = args.backend_gpu_mem_headroom {
                backend_table.insert(
                    "gpu_mem_headroom".to_string(),
                    toml::Value::Float(gpu_mem_headroom),
                );
                println!("✅ Updated backend gpu_mem_headroom");
            }
        } else {
            anyhow::bail!("Invalid backend configuration format in config file.");
        }
    }

    // Serialize and write the updated config back to file
    let updated_config_content = toml::to_string_pretty(&config_file)
        .context("Failed to serialize updated config to TOML")?;

    fs::write(&config_path, updated_config_content)
        .with_context(|| format!("Failed to write updated config file at {:?}", config_path))?;

    println!("✅ Configuration file updated at {:?}", config_path);
    Ok(())
}

fn show_default_config_file() -> Result<()> {
    let config_path = get_default_config_path()?;

    // Check if config file exists
    if !config_path.exists() {
        anyhow::bail!(
            "Configuration file not found at {:?}. Run `pie config init` first.",
            config_path
        );
    }

    // Read and display the config file content
    let config_content = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file at {:?}", config_path))?;

    println!("📄 Configuration file at {:?}:", config_path);
    println!();
    println!("{}", config_content);

    Ok(())
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

async fn start_engine_and_backend(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
    printer: Arc<Mutex<dyn rustyline::ExternalPrinter + Send>>,
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

async fn create_editor_and_printer() -> Result<(
    Editor<MyHelper, FileHistory>,
    Arc<Mutex<dyn rustyline::ExternalPrinter + Send>>,
)> {
    let mut rl = Editor::new()?;
    rl.set_helper(Some(MyHelper)); // Enable our custom highlighter
    let printer: Arc<Mutex<dyn rustyline::ExternalPrinter + Send>> =
        Arc::new(Mutex::new(rl.create_external_printer()?));
    let history_path = get_shell_history_path()?;
    let _ = rl.load_history(&history_path);

    Ok((rl, printer))
}

/// Runs the interactive shell.
async fn run_shell(
    client_config: ClientConfig,
    mut rl: Editor<MyHelper, FileHistory>,
    printer: Arc<Mutex<dyn rustyline::ExternalPrinter + Send>>,
) -> Result<()> {
    println!("Entering interactive session. Type 'help' for commands or use ↑/↓ for history.");

    // The main interactive loop
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

                match handle_shell_command(
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

    println!("Shutting down services...");
    if let Err(err) = rl.save_history(&get_shell_history_path()?) {
        eprintln!("Warning: Failed to save command history: {}", err);
    }

    Ok(())
}

/// Runs the inferlet.
async fn run_inferlet(
    client_config: &ClientConfig,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
    detach: bool,
    printer: &Arc<Mutex<dyn rustyline::ExternalPrinter + Send>>,
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

/// Terminates the engine and backend processes.
async fn terminate_engine_and_backend(
    backend_processes: Vec<Child>,
    shutdown_tx: oneshot::Sender<()>,
    server_handle: tokio::task::JoinHandle<()>,
) -> Result<()> {
    println!();
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

/// Connects to the engine and authenticates the client.
async fn connect_and_authenticate(client_config: &ClientConfig) -> Result<Client> {
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

/// Waits for all backend processes to be attached.
async fn wait_for_backend_ready(client_config: &ClientConfig, num_backends: usize) -> Result<()> {
    let backend_query_client = connect_and_authenticate(&client_config).await?;

    // Query the number of attached and rejected backends.
    let (mut num_attached, mut num_rejected) =
        backend_query_client.wait_backend_change(None, None).await?;

    // If backends have not all been attached, wait for them to be attached.
    while (num_attached as usize) < num_backends && num_rejected == 0 {
        (num_attached, num_rejected) = backend_query_client
            .wait_backend_change(Some(num_attached), Some(num_rejected))
            .await?;
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

/// Waits for the instance to finish.
async fn wait_for_instance_finish(client_config: &ClientConfig) -> Result<()> {
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
