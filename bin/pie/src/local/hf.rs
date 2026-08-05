//! HuggingFace download helpers for `pie model pull` (R3 — the weight-download
//! IO lives only in `bin/pie`, never the worker daemon).
//!
//! The worker lib resolves already-present snapshots (`weights::resolve`); this
//! crate owns the *fetch*. Downloads are runtime-artifact selective: safetensors
//! weights plus config/tokenizer files, excluding alternate checkpoint formats
//! the drivers cannot load (`.pt`/`.bin`/`.gguf`/`consolidated.safetensors`).

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_allowlist_keeps_weights_safetensors_specific() {
        let patterns = runtime_snapshot_allow_patterns();
        assert!(patterns.iter().any(|p| p == "model*.safetensors"));
        assert!(patterns.iter().any(|p| p == "**/model*.safetensors"));
        assert!(patterns.iter().any(|p| p == "*.json"));
        assert!(!patterns.iter().any(|p| p == "*.safetensors"));
        assert!(!patterns.iter().any(|p| p == "**/*.safetensors"));
        assert!(!patterns.iter().any(|p| p.ends_with(".pt")));
        assert!(!patterns.iter().any(|p| p.ends_with(".bin")));
    }
}
