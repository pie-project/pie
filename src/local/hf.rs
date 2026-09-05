//! HuggingFace download helpers for `pie model pull`. The weight-download IO
//! lives only in `pie`, never the worker daemon.
//!
//! The worker lib resolves already-present snapshots (`weights::resolve`); this
//! crate owns the *fetch*. Downloads are runtime-artifact selective: safetensors
//! weights plus config/tokenizer files, excluding alternate checkpoint formats
//! the engines cannot load (`.pt`/`.bin`/`.gguf`/`consolidated.safetensors`).

pub mod download;

pub use download::{Progress, snapshot_download};

/// Where the HuggingFace CLI and libraries keep their blob cache.
///
/// Same precedence the `huggingface_hub` python package uses, so a snapshot
/// pulled by `huggingface-cli` and one pulled by pie land in one place:
/// `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`,
/// else `~/.cache/huggingface/hub`.
pub fn resolve_cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("huggingface").join("hub")
}

/// Files required by Pie's runtime loaders. Broad for small metadata + tokenizer
/// artifacts, but narrow for weights: the CUDA/Metal loaders consume
/// `model.safetensors` / `model-*.safetensors` shards, not duplicate `.pt`,
/// `.bin`, `.gguf`, or `consolidated.safetensors` artifacts. Used by
/// `pie model pull` to restrict the HF snapshot download.
pub fn runtime_snapshot_allow_patterns() -> Vec<String> {
    [
        "*.json",
        "*.model",
        "*.txt",
        "*.tiktoken",
        "*.jinja",
        "model*.safetensors",
        "**/*.json",
        "**/*.model",
        "**/*.txt",
        "**/*.tiktoken",
        "**/*.jinja",
        "**/model*.safetensors",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}
