//! `pie model { list | download | remove }` — manage HF cache.
//!
//! Mirrors `pie/src/pie_cli/commands/model.py` against the same
//! `~/.cache/huggingface/hub/` layout (resolved via `hf-hub`'s
//! `resolve_cache_dir`). Models pulled via either the Python CLI or
//! the standalone are interchangeable.

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum ModelCmd {
    /// List repo IDs already in the local HF cache.
    List,
    /// Download a model snapshot by HuggingFace repo ID.
    Download { repo_id: String },
    /// Remove a cached model by HuggingFace repo ID.
    Remove { repo_id: String },
}

pub fn run(cmd: ModelCmd) -> Result<()> {
    match cmd {
        ModelCmd::List => list(),
        ModelCmd::Download { repo_id } => download(repo_id),
        ModelCmd::Remove { repo_id } => remove(repo_id),
    }
}

/// HF cache root for model snapshots: `<HF_HOME or ~/.cache/huggingface>/hub/`.
/// `hf_hub::resolve_cache_dir` already appends `/hub` (see
/// `hf-hub::constants::resolve_cache_dir`), so don't double it.
fn hub_dir() -> std::path::PathBuf {
    hf_hub::resolve_cache_dir()
}

/// Convert `models--org--name` ↔ `org/name`. Mirrors
/// `pie_driver.hf_utils.parse_repo_id_from_dirname`.
fn dirname_to_repo_id(dir: &str) -> Option<String> {
    let stripped = dir.strip_prefix("models--")?;
    let parts: Vec<&str> = stripped.split("--").collect();
    match parts.len() {
        1 => Some(parts[0].to_string()),
        2 => Some(format!("{}/{}", parts[0], parts[1])),
        _ => None,
    }
}

fn repo_id_to_dirname(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn list() -> Result<()> {
    let hub = hub_dir();
    if !hub.exists() {
        println!("(no HuggingFace cache at {})", hub.display());
        return Ok(());
    }

    let mut models: Vec<String> = std::fs::read_dir(&hub)
        .map_err(|e| anyhow!("read {hub:?}: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| dirname_to_repo_id(&e.file_name().to_string_lossy()))
        .collect();
    models.sort();

    if models.is_empty() {
        println!("(no models in cache)");
    } else {
        for m in &models {
            println!("  {m}");
        }
    }
    println!("\n{}", hub.display());
    Ok(())
}

fn download(repo_id: String) -> Result<()> {
    println!("Downloading {repo_id}…");
    // Reuse the same path serve.rs uses, so cache + download semantics
    // stay identical between `pie serve` and `pie model download`.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let path = runtime
        .block_on(crate::hf::resolve_or_download(&repo_id))
        .map_err(|e| anyhow!("download {repo_id}: {e}"))?;
    println!("✓ Downloaded to {}", path.display());
    Ok(())
}

fn remove(repo_id: String) -> Result<()> {
    let hub = hub_dir();
    let model_dir = hub.join(repo_id_to_dirname(&repo_id));
    if !model_dir.exists() {
        bail!("model {repo_id:?} not found in cache ({})", model_dir.display());
    }

    let size = dir_size(&model_dir).unwrap_or(0);
    let mb = size as f64 / (1024.0 * 1024.0);
    println!("Removing {repo_id} ({mb:.1} MiB)…");

    std::fs::remove_dir_all(&model_dir)
        .map_err(|e| anyhow!("remove {model_dir:?}: {e}"))?;
    println!("✓ Removed");
    Ok(())
}

fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        } else if metadata.is_file() {
            total += metadata.len();
        }
        // Symlinks (HF cache uses them for snapshot/blob deduplication)
        // are intentionally skipped — counting through them would
        // double-count the blobs they point at.
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirname_round_trips() {
        assert_eq!(
            dirname_to_repo_id("models--Qwen--Qwen3-0.6B").as_deref(),
            Some("Qwen/Qwen3-0.6B"),
        );
        assert_eq!(
            dirname_to_repo_id("models--bert-base-uncased").as_deref(),
            Some("bert-base-uncased"),
        );
        assert_eq!(dirname_to_repo_id("not-a-model"), None);
        assert_eq!(dirname_to_repo_id("models--a--b--c"), None);

        assert_eq!(repo_id_to_dirname("Qwen/Qwen3-0.6B"), "models--Qwen--Qwen3-0.6B");
        assert_eq!(repo_id_to_dirname("bert-base-uncased"), "models--bert-base-uncased");
    }
}
