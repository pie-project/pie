//! Configuration management commands for the Pie CLI.
//!
//! This module implements the `pie config` subcommands for managing Pie configuration files,
//! including initializing, updating, and displaying configuration settings.

use crate::path;
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
        || args.backend_gpu_mem_headroom.is_some()
        || args.backend_enable_profiling.is_some();

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
            if let Some(enable_profiling) = args.backend_enable_profiling {
                backend_table.insert(
                    "enable_profiling".to_string(),
                    toml::Value::Boolean(enable_profiling),
                );
                println!("✅ Updated backend enable_profiling");
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
