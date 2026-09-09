use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

const PARALLEL_FILES: usize = 8;

const ATTEMPTS: u32 = 4;

pub trait Progress: Send + Sync {
    fn start(&self, files: u64, bytes: u64);
    fn advance(&self, bytes: u64);
}

#[derive(Debug, Clone)]
struct Entry {
    path: String,
    size: u64,
    etag: String,
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
        .trim_end_matches('/')
        .to_string()
}

fn token() -> Option<String> {
    for var in [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ] {
        if let Ok(value) = std::env::var(var) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    let home = std::env::var_os("HF_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache").join("huggingface"))
        })?;
    let raw = std::fs::read_to_string(home.join("token")).ok()?;
    let raw = raw.trim().to_string();
    (!raw.is_empty()).then_some(raw)
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(concat!("pie/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .context("building HTTP client")
}

fn authorized(request: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    match token {
        Some(t) => request.bearer_auth(t),
        None => request,
    }
}

fn hub_error(repo_id: &str, status: reqwest::StatusCode) -> anyhow::Error {
    match status {
        reqwest::StatusCode::NOT_FOUND
        | reqwest::StatusCode::UNAUTHORIZED
        | reqwest::StatusCode::FORBIDDEN => anyhow!(
            "cannot read {repo_id} on {endpoint} ({status}): check the name, or -- if it is \
             private or gated -- accept its terms at {endpoint}/{repo_id} and give this machine \
             a token with `hf auth login` or HF_TOKEN",
            endpoint = endpoint()
        ),
        other => anyhow!("{repo_id}: hub returned {other}"),
    }
}

async fn revision_sha(
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<String> {
    let url = format!("{}/api/models/{repo_id}/revision/{revision}", endpoint());
    let response = authorized(client.get(&url), token)
        .send()
        .await
        .with_context(|| format!("asking {} about {repo_id}", endpoint()))?;
    if !response.status().is_success() {
        return Err(hub_error(repo_id, response.status()));
    }
    let body = response.text().await.context("reading revision response")?;
    let json: serde_json::Value =
        serde_json::from_str(&body).context("parsing revision response")?;
    json.get("sha")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("{repo_id}: revision {revision} has no commit sha"))
}

async fn list_files(
    client: &reqwest::Client,
    repo_id: &str,
    sha: &str,
    token: Option<&str>,
) -> Result<Vec<Entry>> {
    let mut url = format!(
        "{}/api/models/{repo_id}/tree/{sha}?recursive=1&expand=1",
        endpoint()
    );
    let mut entries = Vec::new();

    loop {
        let response = authorized(client.get(&url), token)
            .send()
            .await
            .with_context(|| format!("listing {repo_id}@{sha}"))?;
        if !response.status().is_success() {
            return Err(hub_error(repo_id, response.status()));
        }
        let next = next_page(response.headers());
        let body = response.text().await.context("reading tree response")?;
        let page: Vec<serde_json::Value> =
            serde_json::from_str(&body).context("parsing tree response")?;

        for item in page {
            if item.get("type").and_then(|v| v.as_str()) != Some("file") {
                continue;
            }
            let Some(path) = item.get("path").and_then(|v| v.as_str()) else {
                continue;
            };
            let lfs = item.get("lfs");
            let etag = lfs
                .and_then(|l| l.get("oid"))
                .or_else(|| item.get("oid"))
                .and_then(|v| v.as_str());
            let size = lfs
                .and_then(|l| l.get("size"))
                .or_else(|| item.get("size"))
                .and_then(serde_json::Value::as_u64);
            let (Some(etag), Some(size)) = (etag, size) else {
                continue;
            };
            entries.push(Entry {
                path: path.to_string(),
                size,
                etag: etag.to_string(),
            });
        }

        match next {
            Some(link) => url = link,
            None => break,
        }
    }

    Ok(entries)
}

fn next_page(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let link = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    link.split(',').find_map(|part| {
        if !part.contains("rel=\"next\"") {
            return None;
        }
        let start = part.find('<')? + 1;
        let end = part[start..].find('>')? + start;
        Some(part[start..end].to_string())
    })
}

fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    segments_match(&pattern, &path)
}

fn segments_match(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.first() {
        None => path.is_empty(),
        Some(&"**") => (0..=path.len()).any(|skip| segments_match(&pattern[1..], &path[skip..])),
        Some(head) => match path.first() {
            Some(segment) if segment_match(head, segment) => {
                segments_match(&pattern[1..], &path[1..])
            }
            _ => false,
        },
    }
}

fn segment_match(pattern: &str, segment: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let segment: Vec<char> = segment.chars().collect();
    let (mut p, mut s) = (0usize, 0usize);
    let (mut star, mut resume) = (None, 0usize);
    while s < segment.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == segment[s]) {
            p += 1;
            s += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some(p);
            resume = s;
            p += 1;
        } else if let Some(star) = star {
            p = star + 1;
            resume += 1;
            s = resume;
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|c| *c == '*')
}

pub async fn snapshot_download(
    repo_id: &str,
    allow_patterns: &[String],
    progress: Arc<dyn Progress>,
) -> Result<PathBuf> {
    let client = client()?;
    let token = token();
    let sha = revision_sha(&client, repo_id, "main", token.as_deref()).await?;
    let files = list_files(&client, repo_id, &sha, token.as_deref()).await?;

    let wanted: Vec<Entry> = files
        .into_iter()
        .filter(|entry| {
            allow_patterns.is_empty()
                || allow_patterns
                    .iter()
                    .any(|pattern| glob_match(pattern, &entry.path))
        })
        .collect();
    if wanted.is_empty() {
        bail!("{repo_id}@{sha} has no files pie can use");
    }

    let repo_dir =
        super::resolve_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshot_dir = repo_dir.join("snapshots").join(&sha);
    let blobs_dir = repo_dir.join("blobs");
    std::fs::create_dir_all(&blobs_dir)
        .with_context(|| format!("creating {}", blobs_dir.display()))?;
    std::fs::create_dir_all(&snapshot_dir)
        .with_context(|| format!("creating {}", snapshot_dir.display()))?;

    let mut pending = Vec::new();
    let mut pending_bytes = 0u64;
    for entry in wanted {
        if is_complete(&snapshot_dir, &blobs_dir, &entry) {
            continue;
        }
        let partial = std::fs::metadata(incomplete_path(&blobs_dir, &entry.etag))
            .map(|m| m.len())
            .unwrap_or(0);
        pending_bytes += entry.size.saturating_sub(partial);
        pending.push(entry);
    }
    progress.start(pending.len() as u64, pending_bytes);

    if !pending.is_empty() {
        let permits = Arc::new(tokio::sync::Semaphore::new(PARALLEL_FILES));
        let mut tasks = tokio::task::JoinSet::new();
        for entry in pending {
            let (client, token) = (client.clone(), token.clone());
            let (repo_id, sha) = (repo_id.to_string(), sha.clone());
            let (snapshot_dir, blobs_dir) = (snapshot_dir.clone(), blobs_dir.clone());
            let (progress, permits) = (progress.clone(), permits.clone());
            tasks.spawn(async move {
                let _permit = permits.acquire_owned().await;
                fetch_file(
                    &client,
                    &repo_id,
                    &sha,
                    &entry,
                    &snapshot_dir,
                    &blobs_dir,
                    token.as_deref(),
                    progress.as_ref(),
                )
                .await
                .with_context(|| format!("downloading {}", entry.path))
            });
        }
        while let Some(joined) = tasks.join_next().await {
            if let Err(error) = joined.context("download task panicked")? {
                tasks.abort_all();
                return Err(error);
            }
        }
    }

    let refs_dir = repo_dir.join("refs");
    if std::fs::create_dir_all(&refs_dir).is_ok() {
        let _ = std::fs::write(refs_dir.join("main"), &sha);
    }

    Ok(snapshot_dir)
}

fn incomplete_path(blobs_dir: &Path, etag: &str) -> PathBuf {
    blobs_dir.join(format!("{etag}.incomplete"))
}

fn is_complete(snapshot_dir: &Path, blobs_dir: &Path, entry: &Entry) -> bool {
    let linked = snapshot_dir.join(&entry.path);
    if std::fs::symlink_metadata(&linked).is_err() {
        return false;
    }
    match std::fs::metadata(blobs_dir.join(&entry.etag)) {
        Ok(meta) => meta.len() == entry.size,
        Err(_) => false,
    }
}

#[allow(clippy::too_many_arguments)]
async fn fetch_file(
    client: &reqwest::Client,
    repo_id: &str,
    sha: &str,
    entry: &Entry,
    snapshot_dir: &Path,
    blobs_dir: &Path,
    token: Option<&str>,
    progress: &dyn Progress,
) -> Result<()> {
    let blob = blobs_dir.join(&entry.etag);
    if std::fs::metadata(&blob).map(|m| m.len()).ok() != Some(entry.size) {
        download_blob(
            client, repo_id, sha, entry, &blob, blobs_dir, token, progress,
        )
        .await?;
    }
    link_into_snapshot(snapshot_dir, blobs_dir, entry)
}

#[allow(clippy::too_many_arguments)]
async fn download_blob(
    client: &reqwest::Client,
    repo_id: &str,
    sha: &str,
    entry: &Entry,
    blob: &Path,
    blobs_dir: &Path,
    token: Option<&str>,
    progress: &dyn Progress,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    let url = format!("{}/{repo_id}/resolve/{sha}/{}", endpoint(), entry.path);
    let temp = incomplete_path(blobs_dir, &entry.etag);
    let mut last_error = None;

    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(attempt as u64)).await;
        }

        let have = std::fs::metadata(&temp).map(|m| m.len()).unwrap_or(0);
        let mut request = authorized(client.get(&url), token);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(anyhow!(error));
                continue;
            }
        };

        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED
            || status == reqwest::StatusCode::FORBIDDEN
            || status == reqwest::StatusCode::NOT_FOUND
        {
            return Err(hub_error(repo_id, status));
        }
        if !status.is_success() {
            last_error = Some(anyhow!("hub returned {status}"));
            continue;
        }
        let resuming = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
        if resuming {
            progress.advance(0);
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resuming)
            .truncate(!resuming)
            .open(&temp)
            .await
            .with_context(|| format!("opening {}", temp.display()))?;

        match stream_to_file(response, &mut file, progress).await {
            Ok(()) => {
                file.flush().await.ok();
                drop(file);
                let written = std::fs::metadata(&temp).map(|m| m.len()).unwrap_or(0);
                if written != entry.size {
                    last_error = Some(anyhow!("short read: got {written} of {} bytes", entry.size));
                    continue;
                }
                std::fs::rename(&temp, blob)
                    .with_context(|| format!("moving {} into place", temp.display()))?;
                return Ok(());
            }
            Err(error) => {
                last_error = Some(error);
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| anyhow!("gave up"))
        .context(format!("after {ATTEMPTS} attempts")))
}

async fn stream_to_file(
    mut response: reqwest::Response,
    file: &mut tokio::fs::File,
    progress: &dyn Progress,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = response.chunk().await.context("reading response body")? {
        file.write_all(&chunk).await.context("writing to cache")?;
        progress.advance(chunk.len() as u64);
    }
    Ok(())
}

fn link_into_snapshot(snapshot_dir: &Path, blobs_dir: &Path, entry: &Entry) -> Result<()> {
    let linked = snapshot_dir.join(&entry.path);
    if let Some(parent) = linked.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let _ = std::fs::remove_file(&linked);

    let depth = entry.path.matches('/').count() + 2;
    let mut target = PathBuf::new();
    for _ in 0..depth {
        target.push("..");
    }
    let target = target.join("blobs").join(&entry.etag);

    #[cfg(unix)]
    let linked_ok = std::os::unix::fs::symlink(&target, &linked).is_ok();
    #[cfg(windows)]
    let linked_ok = std::os::windows::fs::symlink_file(&target, &linked).is_ok();

    if linked_ok {
        return Ok(());
    }
    std::fs::copy(blobs_dir.join(&entry.etag), &linked)
        .with_context(|| format!("materializing {}", linked.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_every_case() {
        weight_shards_match_and_alternates_do_not();
        a_pipelines_components_are_fetched_and_its_bundle_is_not();
    }

    fn weight_shards_match_and_alternates_do_not() {
        let allow = super::super::runtime_snapshot_allow_patterns();
        let matches = |path: &str| allow.iter().any(|p| glob_match(p, path));

        assert!(matches("model.safetensors"));
        assert!(matches("model-00001-of-00004.safetensors"));
        assert!(matches("model.safetensors.index.json"));
        assert!(matches("tokenizer.json"));
        assert!(matches("chat_template.jinja"));
        assert!(!matches("consolidated.safetensors"));
        assert!(!matches("pytorch_model.bin"));
        assert!(!matches("model.gguf"));
    }

    fn a_pipelines_components_are_fetched_and_its_bundle_is_not() {
        let allow = super::super::runtime_snapshot_allow_patterns();
        let matches = |path: &str| allow.iter().any(|p| glob_match(p, path));

        assert!(matches("model_index.json"));
        assert!(matches("transformer/config.json"));
        assert!(matches("transformer/diffusion_pytorch_model.safetensors"));
        assert!(matches(
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
        ));
        assert!(matches(
            "transformer/diffusion_pytorch_model.safetensors.index.json"
        ));
        assert!(matches("text_encoder/model-00001-of-00003.safetensors"));
        assert!(matches("text_encoder/model.safetensors.index.json"));
        assert!(matches("vae/diffusion_pytorch_model.safetensors"));
        assert!(matches("tokenizer/tokenizer.json"));
        assert!(matches("tokenizer/merges.txt"));
        assert!(matches("tokenizer/spiece.model"));
        assert!(matches("scheduler/scheduler_config.json"));

        assert!(!matches("flux-2-klein-4b.safetensors"));
        assert!(!matches("README.md"));
        assert!(!matches("editing.jpg"));
        assert!(!matches("assets/teaser.png"));
        assert!(!matches("transformer/diffusion_pytorch_model.bin"));
    }
}
