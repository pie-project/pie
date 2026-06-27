//! Weight snapshot resolution — **load-at-boot only**.
//!
//! R3: the worker daemon never downloads. `[model].hf_repo` must point at a
//! LOCAL snapshot the worker can load at boot — an HF safetensors directory, or
//! a single `.gguf` file (the metal driver loads either; see
//! `driver/metal/src/model.cpp::Model`). A bare `owner/name` repo ID is NOT
//! resolved here (that would mean downloading / cache provisioning); resolve it
//! ahead of time with `pie model pull` (the `bin/pie` op that owns `hf-hub`)
//! and configure the resulting local path.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// Resolve `hf_repo` to a local snapshot path for load-at-boot. Accepts a local
/// directory (HF safetensors snapshot) or a single `.gguf` file; a bare repo ID
/// errors with a `pie model pull` hint — the worker never provisions (R3).
pub fn resolve(hf_repo: &str) -> Result<PathBuf> {
    let p = Path::new(hf_repo);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if p.is_file() && p.extension().is_some_and(|e| e == "gguf") {
        return Ok(p.to_path_buf());
    }
    if looks_like_repo_id(hf_repo) {
        bail!(
            "hf_repo {hf_repo:?} is a repo ID, but the worker never downloads \
             (R3): resolve it with `pie model pull {hf_repo}` and set hf_repo to \
             the resulting local snapshot path"
        );
    }
    bail!("hf_repo {hf_repo:?} is not a local snapshot directory or .gguf file");
}

/// A bare `owner/name` with no path-like prefix — an HF repo ID, not a path.
fn looks_like_repo_id(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('/')
        && !s.starts_with("./")
        && !s.starts_with("../")
        && !s.contains('\\')
        && s.matches('/').count() == 1
        && s.split('/').all(|seg| !seg.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_directory_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(resolve(tmp.path().to_str().unwrap()).unwrap(), tmp.path());
    }

    #[test]
    fn local_gguf_file_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("model.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(resolve(gguf.to_str().unwrap()).unwrap(), gguf);
    }

    #[test]
    fn repo_id_errors_with_pull_hint() {
        let err = resolve("meta-llama/Llama-3").unwrap_err().to_string();
        assert!(err.contains("pie model pull"), "got: {err}");
    }

    #[test]
    fn missing_path_errors() {
        assert!(resolve("/nonexistent/path/xyz-should-not-exist").is_err());
    }
}
