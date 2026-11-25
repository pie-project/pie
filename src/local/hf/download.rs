use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

const PARALLEL_FILES: usize = 8;

const ATTEMPTS: u32 = 4;

pub trait Progress: Send + Sync {
    /// `bytes` are still to fetch, in `files` files; `held` of the wanted
    /// bytes are on disk already.
    fn start(&self, files: u64, bytes: u64, held: u64);
    fn advance(&self, bytes: u64);
}

#[derive(Debug, Clone)]
struct Entry {
    path: String,
    size: u64,
    etag: String,
    /// An LFS file, named by its SHA-256; any other is named by its git blob SHA-1.
    lfs: bool,
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
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface")))?;
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
                lfs: lfs.and_then(|l| l.get("oid")).is_some(),
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

/// Fetches `repo_id` at `revision` (a full commit sha), or at `main`'s head
/// when none is named, into the Hugging Face cache, and returns its snapshot.
pub async fn snapshot_download(
    repo_id: &str,
    revision: Option<&str>,
    allow_patterns: &[String],
    progress: Arc<dyn Progress>,
) -> Result<PathBuf> {
    let client = client()?;
    let token = token();
    let sha = revision_sha(
        &client,
        repo_id,
        revision.unwrap_or("main"),
        token.as_deref(),
    )
    .await?;
    if let Some(revision) = revision
        && sha != revision
    {
        bail!(
            "{repo_id}: the hub resolves revision {revision} to commit {sha}; refusing to \
             import a commit other than the one asked for"
        );
    }
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
    if let Some(entry) = wanted.iter().find(|entry| !is_safe(entry)) {
        bail!(
            "{repo_id}@{sha} lists {:?} named {:?}, which is not a relative path of plain \
             segments named by a lowercase hex object id; refusing the listing",
            entry.path,
            entry.etag
        );
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
    let mut held = 0u64;
    for entry in wanted {
        if is_complete(&snapshot_dir, &blobs_dir, &entry) {
            held += entry.size;
            continue;
        }
        let partial = std::fs::metadata(incomplete_path(&blobs_dir, &entry.etag))
            .map(|m| m.len())
            .unwrap_or(0);
        held += partial.min(entry.size);
        pending_bytes += entry.size.saturating_sub(partial);
        pending.push(entry);
    }
    progress.start(pending.len() as u64, pending_bytes, held);

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

    // A pinned commit is no branch's head, so it names no ref.
    let refs_dir = repo_dir.join("refs");
    if revision.is_none() && std::fs::create_dir_all(&refs_dir).is_ok() {
        let _ = std::fs::write(refs_dir.join("main"), &sha);
    }

    Ok(snapshot_dir)
}

/// The files a check reads whole: what a checkpoint or an adapter states of itself.
const STATED: [&str; 6] = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "adapter_config.json",
    "stanza.json",
    "model.safetensors.index.json",
];

/// The files an adapter is made of, which `pie model fetch` fetches.
pub const ADAPTER_FILES: [&str; 2] = ["*.json", "adapter*.safetensors"];

/// The most bytes one of [`STATED`] may take.
const STATED_CAP: u64 = 16 << 20;

/// The most bytes a safetensors header may take.
const HEADER_CAP: u64 = 64 << 20;

/// A repository at one commit, as a check read it.
pub struct Stated {
    /// The commit.
    pub sha: String,
    /// The bytes an import fetches at that commit.
    pub download_bytes: u64,
    /// The bytes a fetch of it as an adapter takes ([`ADAPTER_FILES`]).
    pub adapter_bytes: u64,
    /// The bytes of its `model*.safetensors` files.
    pub weights_bytes: u64,
}

/// The hub refused a file of a repository whose terms this machine has not accepted.
#[derive(Debug)]
pub struct Gated(pub String);

impl std::fmt::Display for Gated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is gated: its files need an account that accepted its terms",
            self.0
        )
    }
}

impl std::error::Error for Gated {}

/// Fetches what `repo_id` at `revision` (or `main`'s head) states of itself
/// into `dir`, and never a weight: each of [`STATED`] it lists, hash-checked,
/// and each top-level `model*.safetensors` as a sparse file of its listed
/// size that holds only its header, read through two range requests.
pub async fn stated_download(repo_id: &str, revision: Option<&str>, dir: &Path) -> Result<Stated> {
    let client = client()?;
    let token = token();
    let sha = revision_sha(
        &client,
        repo_id,
        revision.unwrap_or("main"),
        token.as_deref(),
    )
    .await?;
    if let Some(revision) = revision
        && sha != revision
    {
        bail!("{repo_id}: the hub resolves revision {revision} to commit {sha}");
    }
    let files = list_files(&client, repo_id, &sha, token.as_deref()).await?;
    // The hub masks the object ids of a gated repository's weights from an
    // account that has not accepted its terms.
    if files
        .iter()
        .any(|entry| !entry.etag.is_empty() && entry.etag.bytes().all(|b| b == b'*'))
    {
        return Err(Gated(repo_id.to_string()).into());
    }
    if let Some(entry) = files.iter().find(|entry| !is_safe(entry)) {
        bail!(
            "{repo_id}@{sha} lists {:?} named {:?}, which is not a relative path of plain \
             segments named by a lowercase hex object id; refusing the listing",
            entry.path,
            entry.etag
        );
    }
    let allow = super::runtime_snapshot_allow_patterns();
    let wanted = |entry: &&Entry| allow.iter().any(|pattern| glob_match(pattern, &entry.path));
    let weights = |entry: &&Entry| glob_match("model*.safetensors", &entry.path);
    let adapter = |entry: &&Entry| {
        ADAPTER_FILES
            .iter()
            .any(|pattern| glob_match(pattern, &entry.path))
    };
    let stated = Stated {
        sha: sha.clone(),
        download_bytes: files.iter().filter(wanted).map(|entry| entry.size).sum(),
        adapter_bytes: files.iter().filter(adapter).map(|entry| entry.size).sum(),
        weights_bytes: files.iter().filter(weights).map(|entry| entry.size).sum(),
    };
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    for entry in &files {
        let url = format!("{}/{repo_id}/resolve/{sha}/{}", endpoint(), entry.path);
        let at = dir.join(&entry.path);
        if STATED.contains(&entry.path.as_str()) {
            if entry.size > STATED_CAP {
                bail!(
                    "{repo_id}: {} is {} bytes, over the {STATED_CAP} a check reads",
                    entry.path,
                    entry.size
                );
            }
            let bytes = ranged(&client, repo_id, &url, None, token.as_deref()).await?;
            if bytes.len() as u64 != entry.size {
                bail!("{repo_id}: {} came short", entry.path);
            }
            let mut hash = Hash::of(entry);
            hash.update(&bytes);
            std::fs::write(&at, &bytes).with_context(|| format!("writing {}", at.display()))?;
            hash.check(entry, &at)?;
        } else if weights(&entry) {
            let length = ranged(&client, repo_id, &url, Some((0, 8)), token.as_deref()).await?;
            let length = u64::from_le_bytes(
                length
                    .as_slice()
                    .try_into()
                    .map_err(|_| anyhow!("{repo_id}: {} has no header length", entry.path))?,
            );
            if length > HEADER_CAP || 8 + length > entry.size {
                bail!(
                    "{repo_id}: {} states a {length}-byte header, which is not a safetensors \
                     header a check reads",
                    entry.path
                );
            }
            let header =
                ranged(&client, repo_id, &url, Some((8, length)), token.as_deref()).await?;
            if header.len() as u64 != length
                || serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&header)
                    .is_err()
            {
                bail!("{repo_id}: {} has no safetensors header", entry.path);
            }
            let file =
                std::fs::File::create(&at).with_context(|| format!("creating {}", at.display()))?;
            use std::io::Write;
            (&file)
                .write_all(&length.to_le_bytes())
                .and_then(|()| (&file).write_all(&header))
                .and_then(|()| file.set_len(entry.size))
                .with_context(|| format!("writing {}", at.display()))?;
        }
    }
    Ok(stated)
}

/// The bytes of `url`, or the `range` (start, length) of them, refusing a
/// reply longer than asked or, with no range, than [`STATED_CAP`].
async fn ranged(
    client: &reqwest::Client,
    repo_id: &str,
    url: &str,
    range: Option<(u64, u64)>,
    token: Option<&str>,
) -> Result<Vec<u8>> {
    let mut request = authorized(client.get(url), token);
    if let Some((start, length)) = range {
        request = request.header(
            reqwest::header::RANGE,
            format!("bytes={start}-{}", start + length - 1),
        );
    }
    let mut response = request
        .send()
        .await
        .with_context(|| format!("reading {url}"))?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(Gated(repo_id.to_string()).into());
    }
    if !status.is_success() || (range.is_some() && status != reqwest::StatusCode::PARTIAL_CONTENT) {
        bail!("{repo_id}: the hub answered {status} for {url}");
    }
    let most = range.map_or(STATED_CAP, |(_, length)| length);
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.context("reading response body")? {
        bytes.extend_from_slice(&chunk);
        if bytes.len() as u64 > most {
            bail!("{repo_id}: the hub sent more than asked for {url}");
        }
    }
    Ok(bytes)
}

/// A listed file pie may join onto its folders: a relative path of plain
/// segments, named by an object id of its hash's length in lowercase hex.
fn is_safe(entry: &Entry) -> bool {
    let plain = !entry.path.contains('\\')
        && entry
            .path
            .split('/')
            .all(|segment| !matches!(segment, "" | "." | ".."))
        && Path::new(&entry.path)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)));
    let hex = entry.etag.len() == if entry.lfs { 64 } else { 40 }
        && entry
            .etag
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    plain && hex
}

/// The hash a listing names a file by: an LFS file's SHA-256 of its bytes,
/// or any other file's git blob SHA-1, which hashes a `blob <size>\0` header
/// before them.
enum Hash {
    Lfs(sha2::Sha256),
    Git(sha1::Sha1),
}

impl Hash {
    fn of(entry: &Entry) -> Self {
        use sha2::Digest;
        if entry.lfs {
            return Self::Lfs(sha2::Sha256::new());
        }
        let mut git = sha1::Sha1::new();
        git.update(format!("blob {}\0", entry.size));
        Self::Git(git)
    }

    fn update(&mut self, bytes: &[u8]) {
        use sha2::Digest;
        match self {
            Self::Lfs(hash) => hash.update(bytes),
            Self::Git(hash) => hash.update(bytes),
        }
    }

    async fn read(&mut self, path: &Path) -> Result<()> {
        use tokio::io::AsyncReadExt;

        let mut file = tokio::fs::File::open(path)
            .await
            .with_context(|| format!("opening {}", path.display()))?;
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = file
                .read(&mut buffer)
                .await
                .with_context(|| format!("reading {}", path.display()))?;
            if read == 0 {
                return Ok(());
            }
            self.update(&buffer[..read]);
        }
    }

    /// Deletes `file` and fails, naming the entry, unless the bytes hashed
    /// are the ones the listing names.
    fn check(self, entry: &Entry, file: &Path) -> Result<()> {
        use sha2::Digest;
        let got: String = match self {
            Self::Lfs(hash) => hash.finalize().iter().map(|b| format!("{b:02x}")).collect(),
            Self::Git(hash) => hash.finalize().iter().map(|b| format!("{b:02x}")).collect(),
        };
        if got == entry.etag {
            return Ok(());
        }
        let _ = std::fs::remove_file(file);
        bail!(
            "{}: its bytes hash to {got}, not to the {} the listing names; the file is deleted",
            entry.path,
            entry.etag
        )
    }
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
    if std::fs::metadata(&blob).map(|m| m.len()).ok() == Some(entry.size) {
        let mut hash = Hash::of(entry);
        hash.read(&blob).await?;
        hash.check(entry, &blob)?;
    } else {
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
        let mut hash = Hash::of(entry);
        if resuming {
            progress.advance(0);
            hash.read(&temp).await?;
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resuming)
            .truncate(!resuming)
            .open(&temp)
            .await
            .with_context(|| format!("opening {}", temp.display()))?;

        match stream_to_file(response, &mut file, progress, &mut hash).await {
            Ok(()) => {
                file.flush().await.ok();
                drop(file);
                let written = std::fs::metadata(&temp).map(|m| m.len()).unwrap_or(0);
                if written != entry.size {
                    last_error = Some(anyhow!("short read: got {written} of {} bytes", entry.size));
                    continue;
                }
                hash.check(entry, &temp)?;
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
    hash: &mut Hash,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = response.chunk().await.context("reading response body")? {
        file.write_all(&chunk).await.context("writing to cache")?;
        hash.update(&chunk);
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
