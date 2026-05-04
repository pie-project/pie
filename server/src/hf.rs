//! HuggingFace snapshot resolver.
//!
//! `[[model]].hf_repo` accepts either a local snapshot directory or a
//! HuggingFace repo ID (`owner/name`). [`resolve_or_download`] returns
//! a usable on-disk path either way:
//!
//!   * Local path → returned as-is.
//!   * Repo ID → resolved against the HF cache (`~/.cache/huggingface/hub/`,
//!     overridable via `$HF_HOME`); downloaded if missing.
//!
//! Mirrors the Python pie behavior in
//! `pie_driver_cuda_native/worker.py` (which calls
//! `pie_driver_dev.hf_utils.get_hf_snapshot_dir`). The cache layout is
//! identical to the Python `huggingface_hub` package's, so a model
//! downloaded via `huggingface-cli` or `pie model download` from the
//! Python side is reused here without re-downloading.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

/// Resolve `repo_id_or_path` to a snapshot directory on disk. Local
/// directories win; otherwise the input is parsed as an `owner/name`
/// HF repo ID and downloaded via the HF Hub.
///
/// Network access is required only on cache miss for repo IDs.
pub async fn resolve_or_download(repo_id_or_path: &str) -> Result<PathBuf> {
    let p = Path::new(repo_id_or_path);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }

    // Looks-like-a-path heuristic: if the user typed an absolute path,
    // a `./`/`../` prefix, or anything with backslashes, they meant a
    // local directory and got the path wrong — surface that instead of
    // sending it through the HF parser, which would fail with a less
    // useful error.
    if p.is_absolute()
        || repo_id_or_path.starts_with("./")
        || repo_id_or_path.starts_with("../")
        || repo_id_or_path.contains('\\')
    {
        bail!(
            "hf_repo {repo_id_or_path:?} looks like a local path but does not \
             exist or is not a directory"
        );
    }

    let (owner, name) = parse_repo_id(repo_id_or_path)?;
    download_snapshot(&owner, &name).await
}

/// Parse `owner/name` into its two components. Rejects nested paths
/// (`owner/sub/name`) and empty halves; HF repos are always exactly
/// two segments.
fn parse_repo_id(s: &str) -> Result<(String, String)> {
    let mut parts = s.splitn(2, '/');
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        bail!(
            "hf_repo {s:?} is neither a local directory nor a valid HF repo \
             ID (expected `owner/name`)"
        );
    }
    Ok((owner.to_string(), name.to_string()))
}

async fn download_snapshot(owner: &str, name: &str) -> Result<PathBuf> {
    let client = hf_hub::HFClient::new()
        .map_err(|e| anyhow!("init HF client: {e}"))?;
    let repo = client.model(owner.to_string(), name.to_string());

    // Touch the cache once locally first to print a helpful "downloading"
    // line only when we'll actually fetch anything. `local_files_only`
    // returns the snapshot dir on cache hit and an error on miss.
    let cache_hit = repo
        .snapshot_download()
        .local_files_only(true)
        .send()
        .await
        .ok();
    if cache_hit.is_none() {
        eprintln!(
            "hf: {owner}/{name} not in local cache; downloading \
             (this may take a while)…"
        );
    }

    repo.snapshot_download()
        .send()
        .await
        .map_err(|e| anyhow!("download {owner}/{name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_owner_name_repo_id() {
        let (o, n) = parse_repo_id("Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(o, "Qwen");
        assert_eq!(n, "Qwen3-0.6B");
    }

    #[test]
    fn rejects_repo_id_without_slash() {
        let err = parse_repo_id("Qwen3-0.6B").unwrap_err().to_string();
        assert!(err.contains("owner/name"), "got: {err}");
    }

    #[test]
    fn rejects_nested_repo_path() {
        let err = parse_repo_id("a/b/c").unwrap_err().to_string();
        assert!(err.contains("owner/name"), "got: {err}");
    }

    #[test]
    fn rejects_empty_components() {
        assert!(parse_repo_id("/foo").is_err());
        assert!(parse_repo_id("foo/").is_err());
        assert!(parse_repo_id("/").is_err());
    }

    #[tokio::test]
    async fn local_directory_short_circuits() {
        let tmp = tempfile::tempdir().unwrap();
        let resolved = resolve_or_download(tmp.path().to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(resolved, tmp.path());
    }

    #[tokio::test]
    async fn rejects_typo_path_clearly() {
        let err = resolve_or_download("/nonexistent/path/that/should/not/exist")
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("looks like a local path"), "got: {err}");
    }
}
