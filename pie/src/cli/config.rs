//! Configuration management commands for the Pie CLI.
//!
//! This module implements the `pie config` subcommands for managing Pie configuration files,
//! including initializing, updating, and displaying configuration settings.

use super::{path, service};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

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
    /// Enable profiling
    #[arg(long)]
    pub backend_enable_profiling: Option<bool>,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Serialize, Debug)]
pub struct ConfigFile {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub enable_auth: Option<bool>,
    pub auth_secret: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub verbose: Option<bool>,
    pub log: Option<PathBuf>,
    #[serde(default)]
    pub backend: Vec<toml::Value>,
}

/// Handles the `pie config` command.
pub async fn handle_config_command(command: ConfigCommands) -> Result<()> {
    match command {
        ConfigCommands::Init(args) => handle_config_init_subcommand(args).await,
        ConfigCommands::Update(args) => handle_config_update_subcommand(args).await,
        ConfigCommands::Show => handle_config_show_subcommand().await,
    }
}

/// Handles the `pie config init` subcommand.
async fn handle_config_init_subcommand(args: ConfigInitArgs) -> Result<()> {
    init_default_config_file(&args.exec_path, &args.backend_type)
}

/// Handles the `pie config update` subcommand.
async fn handle_config_update_subcommand(args: ConfigUpdateArgs) -> Result<()> {
    update_default_config_file(args)
}

/// Handles the `pie config show` subcommand.
async fn handle_config_show_subcommand() -> Result<()> {
    show_default_config_file()
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
        ("enable_profiling", toml::Value::Boolean(false)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect::<toml::Table>();

    // Create the ConfigFile object
    let config_file = ConfigFile {
        host: Some("127.0.0.1".to_string()),
        port: Some(8080),
        enable_auth: Some(true),
        auth_secret: Some(service::generate_random_auth_secret()),
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

    let config_path = path::get_default_config_path()?;

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

// Macro to check if any of the provided Option fields are Some
macro_rules! any_some {
    ($($field:expr),+ $(,)?) => {
        $($field.is_some())||+
    };
}

// Macro to update multiple config fields declaratively (handles mixed types)
macro_rules! update_fields {
    ($config:expr, { $( $item:tt ),* $(,)? }) => {
        $(
            update_fields!(@item $config, $item);
        )*
    };
    (@item $config:expr, ($field_name:ident, $var:ident)) => {
        if let Some(val) = $var {
            $config.$field_name = Some(val);
            println!("✅ Updated {}", stringify!($field_name));
        }
    };
    (@item $config:expr, ($field_name:ident, $var:ident, PathBuf)) => {
        if let Some(val) = $var {
            $config.$field_name = Some(PathBuf::from(val));
            println!("✅ Updated {}", stringify!($field_name));
        }
    };
}

// Macro to update multiple backend TOML table fields declaratively
macro_rules! update_backend_fields {
    ($table:expr, { $( ($key:literal, $var:ident) ),* $(,)? }) => {
        $(
            if let Some(val) = $var {
                $table.insert($key.to_string(), toml::Value::from(val));
                println!("✅ Updated backend {}", $key);
            }
        )*
    };
}

fn update_default_config_file(args: ConfigUpdateArgs) -> Result<()> {
    // Destructure the entire struct to ensure we handle all fields
    let ConfigUpdateArgs {
        // Engine fields
        host,
        port,
        enable_auth,
        auth_secret,
        cache_dir,
        verbose,
        log,
        // Backend fields
        backend_type,
        backend_exec_path,
        backend_model,
        backend_device,
        backend_dtype,
        backend_kv_page_size,
        backend_max_batch_tokens,
        backend_max_dist_size,
        backend_max_num_kv_pages,
        backend_max_num_embeds,
        backend_max_num_adapters,
        backend_max_adapter_rank,
        backend_gpu_mem_headroom,
        backend_enable_profiling,
    } = args;

    // Check if any update options were provided
    let has_engine_updates = any_some![
        host,
        port,
        enable_auth,
        auth_secret,
        cache_dir,
        verbose,
        log,
    ];

    let has_backend_updates = any_some![
        backend_type,
        backend_exec_path,
        backend_model,
        backend_device,
        backend_dtype,
        backend_kv_page_size,
        backend_max_batch_tokens,
        backend_max_dist_size,
        backend_max_num_kv_pages,
        backend_max_num_embeds,
        backend_max_num_adapters,
        backend_max_adapter_rank,
        backend_gpu_mem_headroom,
        backend_enable_profiling,
    ];

    if !has_engine_updates && !has_backend_updates {
        println!("⚠️ No configuration options provided to update.");
        println!("Use `pie config update --help` to see available options.");
        return Ok(());
    }

    println!("⚙️ Updating Pie configuration...");

    let config_path = path::get_default_config_path()?;

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
    update_fields!(config_file, {
        (host, host),
        (port, port),
        (enable_auth, enable_auth),
        (auth_secret, auth_secret),
        (verbose, verbose),
        (cache_dir, cache_dir, PathBuf),
        (log, log, PathBuf),
    });

    // Update backend configuration fields
    if has_backend_updates {
        if config_file.backend.is_empty() {
            anyhow::bail!(
                "No backend configuration found in config file. Cannot update backend settings."
            );
        }

        // Update the first backend entry (assuming single backend for now)
        if let Some(toml::Value::Table(backend_table)) = config_file.backend.get_mut(0) {
            update_backend_fields!(backend_table, {
                ("backend_type", backend_type),
                ("exec_path", backend_exec_path),
                ("model", backend_model),
                ("device", backend_device),
                ("dtype", backend_dtype),
                ("kv_page_size", backend_kv_page_size),
                ("max_batch_tokens", backend_max_batch_tokens),
                ("max_dist_size", backend_max_dist_size),
                ("max_num_kv_pages", backend_max_num_kv_pages),
                ("max_num_embeds", backend_max_num_embeds),
                ("max_num_adapters", backend_max_num_adapters),
                ("max_adapter_rank", backend_max_adapter_rank),
                ("gpu_mem_headroom", backend_gpu_mem_headroom),
                ("enable_profiling", backend_enable_profiling),
            });
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
    let config_path = path::get_default_config_path()?;

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
