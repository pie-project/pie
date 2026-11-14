pub mod download;

pub use download::{Progress, snapshot_download};

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
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("huggingface").join("hub")
}

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
        "**/diffusion_pytorch_model*.safetensors",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}
