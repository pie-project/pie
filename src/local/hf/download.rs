//! Snapshot downloads from the HuggingFace hub.
//!
//! This is deliberately a few hundred lines rather than a dependency. The
//! `hf-hub` crate hard-depends on `hf-xet`, whose client pulls `reqwest` with
//! its default TLS on -- and reqwest 0.13's `default-tls` *is* rustls with the
//! aws-lc backend, so no feature flag on our side could keep `aws-lc-sys` out
//! of the build. That one C/assembly build script was the single most
//! expensive unit in a cold `pie` build, and the xet stack behind it was six
//! more crates. What we actually need from a model hub is "list a revision,
//! fetch these files into the cache the rest of the world reads", which is two
//! JSON endpoints and a GET.
//!
//! The on-disk layout is `huggingface_hub`'s, not ours, and that is the point:
//! blobs are content-addressed under `blobs/<etag>`, revisions are trees of
//! symlinks under `snapshots/<sha>/`. A snapshot `huggingface-cli` already
//! fetched is one this code finds complete and skips, and vice versa.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

/// Files fetched at once. The hub rate-limits per connection rather than per
/// account, and past ~8 the wall-clock stops improving while the progress
/// line turns into noise.
const PARALLEL_FILES: usize = 8;

/// Attempts per file before the download gives up. Hub CDN redirects expire
/// and connections drop mid-transfer on long fetches; both are retryable and
/// the retry resumes rather than restarts.
const ATTEMPTS: u32 = 4;

/// What a caller wants to render while bytes move.
///
/// A trait rather than a channel because the only implementation is a terminal
/// bar that owns its own redraw throttle -- handing it events costs less than
/// handing it a runtime.
pub trait Progress: Send + Sync {
    /// Called once, after the plan is known: what is left to move.
    fn start(&self, files: u64, bytes: u64);
    /// Called per chunk written to disk.
    fn advance(&self, bytes: u64);
}

/// One file in a revision, as the tree endpoint describes it.
#[derive(Debug, Clone)]
struct Entry {
    /// Repo-relative path; also the path under `snapshots/<sha>/`.
    path: String,
    /// Size in bytes of the real content (not the LFS pointer).
    size: u64,
    /// The blob's name in the cache. For LFS files this is the sha256 the hub
    /// serves as `x-linked-etag`; for plain files it is the git blob sha1.
    /// Either way it is what `huggingface_hub` names the blob, which is what
    /// makes the two caches one cache.
    etag: String,
}

/// `https://huggingface.co`, or whatever `HF_ENDPOINT` points at (mirrors and
/// enterprise hubs are the reason the variable exists).
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// The hub token, if this machine has one.
///
/// Same sources `huggingface_hub` reads, in the same order, so `hf auth login`
/// is enough to make gated repos work here too.
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
        // No total-request timeout: a shard is gigabytes and a slow link is
        // slow, not broken. The connect timeout is what catches a dead host.
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

/// Turn a hub HTTP status into the sentence a person can act on.
///
/// 401 does not mean "gated" on its own: the hub answers unknown, private and
/// gated repos alike with it, precisely so that a stranger cannot use the
/// status code to learn which private repos exist. So the message names all
/// three rather than guessing one and sending the reader after the wrong fix.
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

/// Resolve a revision name (`main`, a tag, a branch) to the commit it names.
///
/// The sha, not the name, is what the snapshot directory is keyed by -- so a
/// repo that moved on gets a new directory instead of a half-updated one.
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

/// Every file in one revision, following the endpoint's pagination.
///
/// `recursive=1` flattens subdirectories (multimodal repos keep processor and
/// tokenizer files a level down), and the cursor loop is what makes repos with
/// more than a page of shards work.
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
        // Read before the body: `Response::text` consumes the response.
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
            // The LFS block is the truth for both fields when it is there: a
            // pointer file's `oid`/`size` describe the 130-byte pointer, not
            // the weights behind it.
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

/// The `rel="next"` target of a `Link` header, if the page has one.
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

/// Does `path` match a shell-style `pattern`?
///
/// Segment-aware, like the globset the allow-list was written against: `*` and
/// `?` stop at `/`, and only `**` crosses a directory boundary. `*.json` and
/// `**/*.json` therefore mean different things, which is why
/// [`super::runtime_snapshot_allow_patterns`] lists both.
fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    segments_match(&pattern, &path)
}

fn segments_match(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.first() {
        None => path.is_empty(),
        Some(&"**") => {
            // Zero or more segments: try every split point.
            (0..=path.len()).any(|skip| segments_match(&pattern[1..], &path[skip..]))
        }
        Some(head) => match path.first() {
            Some(segment) if segment_match(head, segment) => {
                segments_match(&pattern[1..], &path[1..])
            }
            _ => false,
        },
    }
}

/// `*`/`?` matching within one path segment.
fn segment_match(pattern: &str, segment: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let segment: Vec<char> = segment.chars().collect();
    // Iterative backtracking rather than recursion: patterns are short but
    // `*`-heavy, and this keeps the worst case linear in practice.
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

/// Fetch one revision of a repo into the HuggingFace cache, and return the
/// snapshot directory holding it.
///
/// Idempotent: files already in the cache (by blob name and size) are counted
/// as done rather than refetched, and a transfer interrupted halfway resumes
/// from the `.incomplete` blob it left behind.
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

    // Plan before moving anything, so the bar's total is the work that is
    // actually left rather than the size of the repo.
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
            // Abort the rest on the first failure: the alternative is watching
            // seven more files finish before being told the fetch failed.
            if let Err(error) = joined.context("download task panicked")? {
                tasks.abort_all();
                return Err(error);
            }
        }
    }

    // `refs/main` is how `huggingface_hub` answers "which snapshot is main"
    // without a network call. Written last: it should name a complete tree.
    let refs_dir = repo_dir.join("refs");
    if std::fs::create_dir_all(&refs_dir).is_ok() {
        let _ = std::fs::write(refs_dir.join("main"), &sha);
    }

    Ok(snapshot_dir)
}

fn incomplete_path(blobs_dir: &Path, etag: &str) -> PathBuf {
    blobs_dir.join(format!("{etag}.incomplete"))
}

/// Is this file already in the cache, whole?
///
/// Size against the blob rather than a hash: rehashing a 100 GB checkpoint on
/// every `import` would cost more than the download it is trying to skip, and
/// the blob name is already a content hash the hub vouched for.
fn is_complete(snapshot_dir: &Path, blobs_dir: &Path, entry: &Entry) -> bool {
    let linked = snapshot_dir.join(&entry.path);
    if std::fs::symlink_metadata(&linked).is_err() {
        return false;
    }
    // A dangling link (blob cleared from under the snapshot) reads as
    // incomplete, which is what it is.
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
            // Linear backoff: the failures worth retrying here are expired CDN
            // signatures and dropped connections, neither of which needs more
            // than a breath.
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
            // Not retryable: no number of attempts produces a token.
            return Err(hub_error(repo_id, status));
        }
        if !status.is_success() {
            last_error = Some(anyhow!("hub returned {status}"));
            continue;
        }
        // A range request the server ignored restarts the file rather than
        // appending a second copy of it onto the first.
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
                // The bytes already on disk stay: the next attempt resumes
                // from them rather than starting the shard again.
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

    // `chunk()` rather than a `Stream` adapter: it is the one streaming API
    // reqwest exposes without pulling `futures` in beside it.
    while let Some(chunk) = response.chunk().await.context("reading response body")? {
        file.write_all(&chunk).await.context("writing to cache")?;
        progress.advance(chunk.len() as u64);
    }
    Ok(())
}

/// Point `snapshots/<sha>/<path>` at the blob holding its content.
///
/// A relative symlink, like `huggingface_hub` writes: the cache stays valid
/// when the whole `hub/` directory is moved or bind-mounted somewhere else.
fn link_into_snapshot(snapshot_dir: &Path, blobs_dir: &Path, entry: &Entry) -> Result<()> {
    let linked = snapshot_dir.join(&entry.path);
    if let Some(parent) = linked.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    // Replace whatever is there: a dangling link from a cleared blob, or a
    // stale entry from an interrupted run.
    let _ = std::fs::remove_file(&linked);

    // `snapshots/<sha>/a/b.json` is three levels below the repo root, so it
    // reaches `blobs/` with one `..` per path segment plus two for
    // `snapshots/<sha>`.
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
    // Windows without developer mode, and filesystems that refuse links, get a
    // copy: twice the disk, but a snapshot the loaders can read.
    std::fs::copy(blobs_dir.join(&entry.etag), &linked)
        .with_context(|| format!("materializing {}", linked.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_stops_at_a_directory_boundary() {
        assert!(glob_match("*.json", "config.json"));
        assert!(!glob_match("*.json", "nested/config.json"));
        assert!(glob_match("**/*.json", "nested/config.json"));
        assert!(glob_match("**/*.json", "config.json"));
    }

    #[test]
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

    #[test]
    fn question_mark_is_one_character() {
        assert!(glob_match("model-?.safetensors", "model-1.safetensors"));
        assert!(!glob_match("model-?.safetensors", "model-12.safetensors"));
    }

    #[test]
    fn nested_paths_link_back_to_the_blob() {
        // The `..` count is the one piece of this that a reader cannot check by
        // eye, and a wrong one produces a dangling link rather than an error:
        // the snapshot looks fetched and the loader fails later, elsewhere.
        let root = std::env::temp_dir().join(format!("pie-hf-link-{}", std::process::id()));
        let blobs = root.join("blobs");
        let snapshot = root.join("snapshots").join("deadbeef");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();

        for path in ["config.json", "nested/config.json", "a/b/c/config.json"] {
            let entry = Entry {
                path: path.to_string(),
                size: 5,
                etag: format!("etag-{}", path.replace('/', "-")),
            };
            std::fs::write(blobs.join(&entry.etag), b"bytes").unwrap();
            link_into_snapshot(&snapshot, &blobs, &entry).unwrap();

            let linked = snapshot.join(path);
            assert!(
                std::fs::symlink_metadata(&linked).unwrap().is_symlink(),
                "{path} should be a symlink"
            );
            assert_eq!(
                std::fs::read(&linked).unwrap(),
                b"bytes",
                "{path} should resolve to its blob"
            );
        }

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn next_page_reads_the_link_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::LINK,
            "<https://hf.co/api/models/x/tree/main?cursor=abc>; rel=\"next\""
                .parse()
                .unwrap(),
        );
        assert_eq!(
            next_page(&headers).as_deref(),
            Some("https://hf.co/api/models/x/tree/main?cursor=abc")
        );
        assert_eq!(next_page(&reqwest::header::HeaderMap::new()), None);
    }
}
