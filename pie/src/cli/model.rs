//! Model management commands for the Pie CLI.
//!
//! This module implements the `pie model` subcommands for managing local AI models
//! and listing available models from the model registry.

use crate::path;
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use reqwest::Client as HttpClient;
use serde::Deserialize;
use serde_json;
use std::{
    fs,
    io::{self, Write},
};

#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    /// List downloaded models.
    List,
    /// Download a model from the model registry.
    Add(AddModelArgs),
    /// Delete a downloaded model.
    Remove(RemoveModelArgs),
    /// Search for models in the model registry.
    Search(SearchModelArgs),
    /// Show information about a model from the model registry.
    Info(InfoModelArgs),
}

#[derive(Args, Debug)]
pub struct AddModelArgs {
    pub model_name: String,
}

#[derive(Args, Debug)]
pub struct RemoveModelArgs {
    pub model_name: String,
}

#[derive(Args, Debug)]
pub struct SearchModelArgs {
    /// Optional regular expression to filter model names.
    pub pattern: Option<String>,
}

#[derive(Args, Debug)]
pub struct InfoModelArgs {
    /// Name of the model to get information about.
    pub model_name: String,
}

/// Handles the `pie model` command.
pub async fn handle_model_command(command: ModelCommands) -> Result<()> {
    match command {
        ModelCommands::List => handle_model_list_subcommand().await,
        ModelCommands::Add(args) => handle_model_add_subcommand(args).await,
        ModelCommands::Remove(args) => handle_model_remove_subcommand(args).await,
        ModelCommands::Search(args) => handle_model_search_subcommand(args).await,
        ModelCommands::Info(args) => handle_model_info_subcommand(args).await,
    }
}

/// Handles the `pie model list` subcommand.
async fn handle_model_list_subcommand() -> Result<()> {
    println!("📚 Available local models:");
    let models_dir = path::get_pie_cache_home()?.join("models");
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
    Ok(())
}

/// Handles the `pie model add` subcommand.
async fn handle_model_add_subcommand(args: AddModelArgs) -> Result<()> {
    println!("➕ Adding model: {}", args.model_name);

    let models_root = path::get_pie_cache_home()?.join("models");
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
    let metadata_str =
        String::from_utf8(metadata_raw).context("Failed to parse model metadata as UTF-8")?;
    let metadata: toml::Value = toml::from_str(&metadata_str)?;

    fs::write(&metadata_path, &metadata_str)?;

    if let Some(source) = metadata.get("source").and_then(|s| s.as_table()) {
        for (name, url_val) in source {
            if let Some(url) = url_val.as_str() {
                let file_data =
                    download_file_with_progress(url, &format!("Downloading {}...", name)).await?;
                fs::write(model_files_dir.join(name), file_data)?;
            }
        }
    }
    println!("✅ Model '{}' added successfully!", args.model_name);
    Ok(())
}

/// Handles the `pie model remove` subcommand.
async fn handle_model_remove_subcommand(args: RemoveModelArgs) -> Result<()> {
    println!("🗑️ Removing model: {}", args.model_name);
    let models_root = path::get_pie_cache_home()?.join("models");
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
    Ok(())
}

/// Handles the `pie model search` subcommand.
async fn handle_model_search_subcommand(args: SearchModelArgs) -> Result<()> {
    println!("🔍 Searching for models...");

    #[derive(Deserialize)]
    struct Item {
        name: String,
        #[serde(rename = "type")]
        item_type: String,
    }

    // Check available files under pie-project/model-index repository
    let url = "https://api.github.com/repos/pie-project/model-index/contents";
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .query(&[("ref", "main")])
        .header("User-Agent", "pie-index-list/1.0")
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    // Try to parse as an array first, if that fails, check if it's an error response
    let items: Vec<Item> = match serde_json::from_str(&response_text) {
        Ok(items) => items,
        // Try to instead parse as a GitHub API error response if the initial parsing fails
        Err(_) => {
            #[derive(Deserialize)]
            struct GitHubError {
                message: String,
            }

            if let Ok(error) = serde_json::from_str::<GitHubError>(&response_text) {
                anyhow::bail!("{}", error.message);
            } else {
                anyhow::bail!("GitHub API request failed with status: {}", status);
            }
        }
    };

    // Compile regex if pattern is provided
    let regex = if let Some(pattern) = &args.pattern {
        Some(Regex::new(pattern).context("Invalid regular expression")?)
    } else {
        None
    };

    // Print matching model names
    for name in items
        .into_iter()
        // Keep only files that are .toml and not traits.toml
        .filter(|i| i.item_type == "file" && i.name.ends_with(".toml") && i.name != "traits.toml")
        // Remove the .toml suffix from the model name
        .map(|i| i.name.strip_suffix(".toml").unwrap_or(&i.name).to_string())
        // Filter by regex if provided
        .filter(|name| regex.as_ref().map_or(true, |r| r.is_match(name)))
    {
        println!("{name}");
    }
    Ok(())
}

/// Handles the `pie model info` subcommand.
async fn handle_model_info_subcommand(args: InfoModelArgs) -> Result<()> {
    println!("📋 Getting model information for '{}'...", args.model_name);

    // Fetch the TOML file content
    let url = format!(
        "https://raw.githubusercontent.com/pie-project/model-index/main/{}.toml",
        args.model_name
    );
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "pie-index-info/1.0")
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Model '{}' not found in the registry", args.model_name);
    }

    let toml_content = response.text().await?;

    // Parse the TOML content
    let parsed_toml: toml::Value =
        toml::from_str(&toml_content).context("Failed to parse TOML file")?;

    // Extract and display the architecture section
    if let Some(architecture) = parsed_toml.get("architecture") {
        println!("\n🏗️  Architecture:");
        match architecture {
            toml::Value::Table(table) => {
                print_toml_table(table, 1);
            }
            _ => {
                println!("  {}", format_toml_value(architecture));
            }
        }
    } else {
        println!("❌ No architecture section found in the model configuration");
    }

    // Check if the model is downloaded locally
    let models_root = path::get_pie_cache_home()?.join("models");
    let model_files_dir = models_root.join(&args.model_name);
    let metadata_path = models_root.join(format!("{}.toml", args.model_name));

    let is_downloaded = model_files_dir.exists() && metadata_path.exists();
    println!("\n📦 Download Status:");
    if is_downloaded {
        println!("  ✅ Downloaded locally");
        println!("  📁 Location: {}", model_files_dir.display());
    } else {
        println!("  ❌ Not downloaded");
        println!("  💡 Use `pie model add {}` to download", args.model_name);
    }
    Ok(())
}

/// Helper function to download a file with progress bar
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

/// Helper function to print TOML tables with proper indentation
fn print_toml_table(table: &toml::value::Table, indent_level: usize) {
    let indent = "  ".repeat(indent_level);

    for (key, value) in table {
        match value {
            toml::Value::Table(nested_table) => {
                println!("{}{}:", indent, key);
                print_toml_table(nested_table, indent_level + 1);
            }
            _ => {
                println!("{}{}: {}", indent, key, format_toml_value(value));
            }
        }
    }
}

/// Helper function to format TOML values for display
fn format_toml_value(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => f.to_string(),
        toml::Value::Boolean(b) => b.to_string(),
        toml::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_toml_value).collect();
            format!("[{}]", items.join(", "))
        }
        toml::Value::Table(table) => {
            // For inline tables, keep the compact format
            let items: Vec<String> = table
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_toml_value(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        toml::Value::Datetime(dt) => dt.to_string(),
    }
}
