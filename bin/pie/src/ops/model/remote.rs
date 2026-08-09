//! Remote, metadata-first checkpoint source for `pie model import`.
//!
//! This module is the feature boundary around network provisioning. The
//! importer sees repository metadata, auxiliary bytes, and exact byte ranges;
//! no HTTP client, retry policy, signed URL, or connection-pool type crosses
//! the seam. Disabling `remote-import` removes this file from the build without
//! changing local checkpoint handling.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "remote-import-xet")]
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_RANGE, HeaderMap, HeaderValue, RANGE, RETRY_AFTER};
use reqwest::{StatusCode, Url};
use serde::Deserialize;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};

const MAX_ATTEMPTS: usize = 4;
const DEFAULT_CONCURRENCY: usize = 16;

#[derive(Clone, Debug)]
pub(crate) struct RangeRequest {
    pub(crate) ordinal: usize,
    pub(crate) file_id: u32,
    pub(crate) range: Range<u64>,
}

#[derive(Debug)]
pub(crate) struct RangeResult {
    pub(crate) ordinal: usize,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct RemoteFileInfo {
    pub(crate) path: String,
    pub(crate) size_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Telemetry {
    pub(crate) resolver_calls: u64,
    pub(crate) url_refreshes: u64,
    pub(crate) range_attempts: u64,
    pub(crate) ranges_ok: u64,
    pub(crate) retries_429: u64,
    pub(crate) retries_5xx: u64,
    pub(crate) retries_auth: u64,
    pub(crate) retries_transport: u64,
    pub(crate) retries_malformed: u64,
    pub(crate) requested_bytes: u64,
    pub(crate) received_bytes: u64,
    pub(crate) peak_in_flight: usize,
    pub(crate) peak_buffer_bytes: u64,
    pub(crate) peak_buffer_tensors: usize,
    pub(crate) elapsed: Duration,
    pub(crate) latency_micros: Vec<u64>,
}

impl Telemetry {
    pub(crate) fn render(&self, repo: &str, revision: &str, shards: usize) -> String {
        let mut latency = self.latency_micros.clone();
        latency.sort_unstable();
        let percentile = |p: usize| -> u64 {
            if latency.is_empty() {
                0
            } else {
                latency[(latency.len() - 1) * p / 100]
            }
        };
        let seconds = self.elapsed.as_secs_f64().max(f64::EPSILON);
        format!(
            "remote-import repo={repo} commit={revision} shards={shards} resolver_calls={} \
             url_refreshes={} range_attempts={} ranges_206={} retries_429={} retries_5xx={} \
             retries_auth={} retries_transport={} retries_malformed={} requested_bytes={} \
             received_bytes={} requests_per_second={:.2} bytes_per_second={:.2} \
             latency_us_p50={} latency_us_p95={} latency_us_p99={} latency_us_max={} \
             peak_in_flight={} reorder_peak_tensors={} reorder_peak_bytes={}",
            self.resolver_calls,
            self.url_refreshes,
            self.range_attempts,
            self.ranges_ok,
            self.retries_429,
            self.retries_5xx,
            self.retries_auth,
            self.retries_transport,
            self.retries_malformed,
            self.requested_bytes,
            self.received_bytes,
            self.ranges_ok as f64 / seconds,
            self.received_bytes as f64 / seconds,
            percentile(50),
            percentile(95),
            percentile(99),
            latency.last().copied().unwrap_or(0),
            self.peak_in_flight,
            self.peak_buffer_tensors,
            self.peak_buffer_bytes,
        )
    }
}

#[derive(Deserialize)]
struct RepoInfo {
    sha: Option<String>,
    siblings: Option<Vec<Sibling>>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
}

struct RemoteFile {
    path: String,
    size_bytes: u64,
    signed_url: AsyncMutex<Url>,
    refresh: AsyncMutex<()>,
}

struct Counters {
    started: Instant,
    resolver_calls: AtomicU64,
    url_refreshes: AtomicU64,
    range_attempts: AtomicU64,
    ranges_ok: AtomicU64,
    retries_429: AtomicU64,
    retries_5xx: AtomicU64,
    retries_auth: AtomicU64,
    retries_transport: AtomicU64,
    retries_malformed: AtomicU64,
    requested_bytes: AtomicU64,
    received_bytes: AtomicU64,
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
    peak_buffer_bytes: AtomicU64,
    peak_buffer_tensors: AtomicUsize,
    latency_micros: Mutex<Vec<u64>>,
}

impl Counters {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            resolver_calls: AtomicU64::new(0),
            url_refreshes: AtomicU64::new(0),
            range_attempts: AtomicU64::new(0),
            ranges_ok: AtomicU64::new(0),
            retries_429: AtomicU64::new(0),
            retries_5xx: AtomicU64::new(0),
            retries_auth: AtomicU64::new(0),
            retries_transport: AtomicU64::new(0),
            retries_malformed: AtomicU64::new(0),
            requested_bytes: AtomicU64::new(0),
            received_bytes: AtomicU64::new(0),
            in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
            peak_buffer_bytes: AtomicU64::new(0),
            peak_buffer_tensors: AtomicUsize::new(0),
            latency_micros: Mutex::new(Vec::new()),
        }
    }

    fn enter(&self) -> InFlight<'_> {
        let now = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_in_flight.fetch_max(now, Ordering::Relaxed);
        InFlight(self)
    }
}

struct InFlight<'a>(&'a Counters);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Inner {
    repo: String,
    revision: String,
    endpoint: Url,
    client: reqwest::Client,
    resolver: reqwest::Client,
    #[cfg(feature = "remote-import-xet")]
    xet_repo: Option<hf_hub::HFRepository<hf_hub::RepoTypeModel>>,
    files: Vec<Arc<RemoteFile>>,
    counters: Counters,
}

pub(crate) struct RemoteSnapshot {
    runtime: tokio::runtime::Runtime,
    inner: Arc<Inner>,
    aux: BTreeMap<String, Vec<u8>>,
}

impl RemoteSnapshot {
    pub(crate) fn open(spec: &str) -> Result<Self> {
        let (repo, requested_revision) = split_spec(spec)?;
        #[cfg(feature = "remote-import-xet")]
        let xet_repo = match std::env::var("PIE_REMOTE_IMPORT_TRANSPORT") {
            Ok(value) if value == "xet" => {
                let client =
                    hf_hub::HFClient::new().context("initialize Hugging Face Xet client")?;
                let (owner, name) = hf_hub::split_id(&repo);
                Some(client.model(owner, name))
            }
            Ok(value) if value == "http" => None,
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) => {
                bail!("PIE_REMOTE_IMPORT_TRANSPORT must be 'http' or 'xet', got {value:?}")
            }
            Err(error) => bail!("cannot read PIE_REMOTE_IMPORT_TRANSPORT: {error}"),
        };
        let endpoint = Url::parse(
            &std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()),
        )?;
        let headers = auth_headers()?;
        let client = reqwest::Client::builder()
            .default_headers(headers.clone())
            .user_agent(format!("pie/{}", env!("CARGO_PKG_VERSION")))
            .build()?;
        let resolver = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(format!("pie/{}", env!("CARGO_PKG_VERSION")))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (revision, file_infos, aux, files) = runtime.block_on(async {
            let info_url = api_model_url(&endpoint, &repo, &requested_revision)?;
            let response = client
                .get(info_url)
                .query(&[("blobs", "true")])
                .send()
                .await?
                .error_for_status()?;
            let info: RepoInfo = response.json().await?;
            let revision = info.sha.ok_or_else(|| {
                anyhow!("{repo}@{requested_revision}: Hub response has no commit SHA")
            })?;
            let siblings = info.siblings.unwrap_or_default();
            let weight_paths = if siblings
                .iter()
                .any(|entry| entry.rfilename == "model.safetensors.index.json")
            {
                let url = resolve_url(&endpoint, &repo, &revision, "model.safetensors.index.json")?;
                let index: serde_json::Value = client
                    .get(url)
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                index
                    .get("weight_map")
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| anyhow!("model.safetensors.index.json has no weight_map"))?
                    .values()
                    .map(|value| {
                        value.as_str().map(str::to_string).ok_or_else(|| {
                            anyhow!("model.safetensors.index.json contains a non-string shard")
                        })
                    })
                    .collect::<Result<BTreeSet<_>>>()?
            } else if siblings
                .iter()
                .any(|entry| entry.rfilename == "model.safetensors")
            {
                BTreeSet::from(["model.safetensors".to_string()])
            } else {
                siblings
                    .iter()
                    .filter(|entry| entry.rfilename.ends_with(".safetensors"))
                    .map(|entry| entry.rfilename.clone())
                    .collect()
            };
            let mut weights: Vec<RemoteFileInfo> = weight_paths
                .into_iter()
                .map(|path| {
                    let sibling = siblings
                        .iter()
                        .find(|entry| entry.rfilename == path)
                        .ok_or_else(|| {
                            anyhow!("model.safetensors.index.json names absent shard {path:?}")
                        })?;
                    Ok(RemoteFileInfo {
                        path,
                        size_bytes: sibling.size.unwrap_or(0),
                    })
                })
                .collect::<Result<_>>()?;
            weights.sort_unstable_by(|a, b| a.path.cmp(&b.path));
            if weights.is_empty() {
                bail!("{repo}@{revision} has no model safetensors files");
            }

            let mut aux = BTreeMap::new();
            for name in [
                "config.json",
                "tokenizer.json",
                "tiktoken.model",
                "tokenizer_config.json",
            ] {
                if siblings.iter().any(|entry| entry.rfilename == name) {
                    let url = resolve_url(&endpoint, &repo, &revision, name)?;
                    let bytes = client
                        .get(url)
                        .send()
                        .await?
                        .error_for_status()?
                        .bytes()
                        .await?;
                    aux.insert(name.to_string(), bytes.to_vec());
                }
            }

            let mut remote_files = Vec::with_capacity(weights.len());
            for file in &mut weights {
                let (signed_url, resolved_size) =
                    resolve_signed_url(&resolver, &endpoint, &repo, &revision, &file.path).await?;
                if file.size_bytes == 0 {
                    file.size_bytes = resolved_size;
                }
                if file.size_bytes == 0 {
                    bail!("{}: Hub did not report a file size", file.path);
                }
                remote_files.push(Arc::new(RemoteFile {
                    path: file.path.clone(),
                    size_bytes: file.size_bytes,
                    signed_url: AsyncMutex::new(signed_url),
                    refresh: AsyncMutex::new(()),
                }));
            }
            Ok::<_, anyhow::Error>((revision, weights, aux, remote_files))
        })?;
        let inner = Arc::new(Inner {
            repo,
            revision,
            endpoint,
            client,
            resolver,
            #[cfg(feature = "remote-import-xet")]
            xet_repo,
            files,
            counters: Counters::new(),
        });
        let snapshot = Self {
            runtime,
            inner,
            aux,
        };
        // Count the initial URL resolution once the counters exist.
        snapshot
            .inner
            .counters
            .resolver_calls
            .store(file_infos.len() as u64, Ordering::Relaxed);
        Ok(snapshot)
    }

    pub(crate) fn repo(&self) -> &str {
        &self.inner.repo
    }

    pub(crate) fn revision(&self) -> &str {
        &self.inner.revision
    }

    pub(crate) fn files(&self) -> Vec<RemoteFileInfo> {
        self.inner
            .files
            .iter()
            .map(|file| RemoteFileInfo {
                path: file.path.clone(),
                size_bytes: file.size_bytes,
            })
            .collect()
    }

    pub(crate) fn aux(&self, name: &str) -> Option<&[u8]> {
        self.aux.get(name).map(Vec::as_slice)
    }

    pub(crate) fn read_exact(&self, file_id: u32, range: Range<u64>) -> Result<Vec<u8>> {
        let mut results = self.read_window(
            &[RangeRequest {
                ordinal: 0,
                file_id,
                range,
            }],
            u64::MAX,
        )?;
        Ok(results.remove(0).bytes)
    }

    pub(crate) fn read_window(
        &self,
        requests: &[RangeRequest],
        byte_ceiling: u64,
    ) -> Result<Vec<RangeResult>> {
        let requested_bytes = requests.iter().try_fold(0u64, |total, request| {
            let len = request
                .range
                .end
                .checked_sub(request.range.start)
                .ok_or_else(|| anyhow!("range goes backwards: {:?}", request.range))?;
            if len > byte_ceiling {
                bail!("tensor range is {len} bytes, above the {byte_ceiling}-byte reorder ceiling");
            }
            total
                .checked_add(len)
                .ok_or_else(|| anyhow!("reorder byte count overflows"))
        })?;
        if requested_bytes > byte_ceiling {
            bail!(
                "reorder window requires {requested_bytes} bytes, above its {byte_ceiling}-byte ceiling"
            );
        }
        self.inner
            .counters
            .peak_buffer_bytes
            .fetch_max(requested_bytes, Ordering::Relaxed);
        self.inner
            .counters
            .peak_buffer_tensors
            .fetch_max(requests.len(), Ordering::Relaxed);
        let semaphore = Arc::new(Semaphore::new(DEFAULT_CONCURRENCY));
        let inner = Arc::clone(&self.inner);
        let mut owned = requests.to_vec();
        owned.sort_unstable_by_key(|request| (request.file_id, request.range.start));
        self.runtime.block_on(async move {
            let mut tasks = tokio::task::JoinSet::new();
            for request in owned {
                let inner = Arc::clone(&inner);
                let semaphore = Arc::clone(&semaphore);
                tasks.spawn(async move {
                    let _permit = semaphore.acquire_owned().await?;
                    fetch_range(inner, request).await
                });
            }
            let mut results = Vec::new();
            while let Some(result) = tasks.join_next().await {
                results.push(result.context("remote range task panicked")??);
            }
            results.sort_unstable_by_key(|result| result.ordinal);
            Ok(results)
        })
    }

    pub(crate) fn telemetry(&self) -> Telemetry {
        let c = &self.inner.counters;
        Telemetry {
            resolver_calls: c.resolver_calls.load(Ordering::Relaxed),
            url_refreshes: c.url_refreshes.load(Ordering::Relaxed),
            range_attempts: c.range_attempts.load(Ordering::Relaxed),
            ranges_ok: c.ranges_ok.load(Ordering::Relaxed),
            retries_429: c.retries_429.load(Ordering::Relaxed),
            retries_5xx: c.retries_5xx.load(Ordering::Relaxed),
            retries_auth: c.retries_auth.load(Ordering::Relaxed),
            retries_transport: c.retries_transport.load(Ordering::Relaxed),
            retries_malformed: c.retries_malformed.load(Ordering::Relaxed),
            requested_bytes: c.requested_bytes.load(Ordering::Relaxed),
            received_bytes: c.received_bytes.load(Ordering::Relaxed),
            peak_in_flight: c.peak_in_flight.load(Ordering::Relaxed),
            peak_buffer_bytes: c.peak_buffer_bytes.load(Ordering::Relaxed),
            peak_buffer_tensors: c.peak_buffer_tensors.load(Ordering::Relaxed),
            elapsed: c.started.elapsed(),
            latency_micros: c.latency_micros.lock().unwrap().clone(),
        }
    }
}

async fn fetch_range(inner: Arc<Inner>, request: RangeRequest) -> Result<RangeResult> {
    let file = inner
        .files
        .get(request.file_id as usize)
        .ok_or_else(|| anyhow!("remote checkpoint has no file id {}", request.file_id))?
        .clone();
    if request.range.end > file.size_bytes {
        bail!(
            "{}: range {:?} is outside the {}-byte file",
            file.path,
            request.range,
            file.size_bytes
        );
    }
    let expected_len = request.range.end - request.range.start;
    if expected_len == 0 {
        return Ok(RangeResult {
            ordinal: request.ordinal,
            bytes: Vec::new(),
        });
    }
    inner
        .counters
        .requested_bytes
        .fetch_add(expected_len, Ordering::Relaxed);
    let _in_flight = inner.counters.enter();

    #[cfg(feature = "remote-import-xet")]
    if inner.xet_repo.is_some() {
        return fetch_xet_range(Arc::clone(&inner), file, request, expected_len).await;
    }

    for attempt in 0..MAX_ATTEMPTS {
        inner
            .counters
            .range_attempts
            .fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let url = file.signed_url.lock().await.clone();
        let response = inner
            .client
            .get(url.clone())
            .header(
                RANGE,
                format!("bytes={}-{}", request.range.start, request.range.end - 1),
            )
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) if attempt + 1 < MAX_ATTEMPTS => {
                inner
                    .counters
                    .retries_transport
                    .fetch_add(1, Ordering::Relaxed);
                backoff(attempt, None).await;
                tracing::warn!(file = %file.path, ?error, attempt, "remote range transport retry");
                continue;
            }
            Err(error) => return Err(error).context(format!("range GET {}", file.path)),
        };
        let status = response.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            if attempt + 1 == MAX_ATTEMPTS {
                bail!("{}: signed URL remained {status} after refresh", file.path);
            }
            inner.counters.retries_auth.fetch_add(1, Ordering::Relaxed);
            refresh_url(&inner, &file, &url).await?;
            continue;
        }
        if status == StatusCode::TOO_MANY_REQUESTS {
            if attempt + 1 == MAX_ATTEMPTS {
                bail!("{}: range GET remained 429 after retries", file.path);
            }
            inner.counters.retries_429.fetch_add(1, Ordering::Relaxed);
            let retry_after = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs);
            backoff(attempt, retry_after).await;
            continue;
        }
        if status.is_server_error() {
            if attempt + 1 == MAX_ATTEMPTS {
                bail!("{}: range GET remained {status} after retries", file.path);
            }
            inner.counters.retries_5xx.fetch_add(1, Ordering::Relaxed);
            backoff(attempt, None).await;
            continue;
        }
        if status != StatusCode::PARTIAL_CONTENT {
            bail!(
                "{}: range {:?} returned {status}, expected 206 Partial Content",
                file.path,
                request.range
            );
        }
        let expected_content_range = format!(
            "bytes {}-{}/{}",
            request.range.start,
            request.range.end - 1,
            file.size_bytes
        );
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if content_range != expected_content_range {
            inner
                .counters
                .retries_malformed
                .fetch_add(1, Ordering::Relaxed);
            bail!(
                "{}: range {:?} returned Content-Range {content_range:?}, expected {expected_content_range:?}",
                file.path,
                request.range
            );
        }
        let body = match response.bytes().await {
            Ok(body) if body.len() as u64 == expected_len => body,
            Ok(body) if attempt + 1 < MAX_ATTEMPTS => {
                inner
                    .counters
                    .retries_malformed
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    file = %file.path,
                    got = body.len(),
                    expected = expected_len,
                    attempt,
                    "short remote range body; retrying"
                );
                backoff(attempt, None).await;
                continue;
            }
            Ok(body) => bail!(
                "{}: range body is {} bytes, expected {expected_len}",
                file.path,
                body.len()
            ),
            Err(error) if attempt + 1 < MAX_ATTEMPTS => {
                inner
                    .counters
                    .retries_transport
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(file = %file.path, ?error, attempt, "remote range body retry");
                backoff(attempt, None).await;
                continue;
            }
            Err(error) => return Err(error).context(format!("read range body from {}", file.path)),
        };
        inner.counters.ranges_ok.fetch_add(1, Ordering::Relaxed);
        inner
            .counters
            .received_bytes
            .fetch_add(body.len() as u64, Ordering::Relaxed);
        inner
            .counters
            .latency_micros
            .lock()
            .unwrap()
            .push(started.elapsed().as_micros() as u64);
        return Ok(RangeResult {
            ordinal: request.ordinal,
            bytes: body.to_vec(),
        });
    }
    unreachable!("bounded retry loop returns or errors")
}

#[cfg(feature = "remote-import-xet")]
async fn fetch_xet_range(
    inner: Arc<Inner>,
    file: Arc<RemoteFile>,
    request: RangeRequest,
    expected_len: u64,
) -> Result<RangeResult> {
    let repo = inner
        .xet_repo
        .as_ref()
        .expect("Xet range fetch requires an Xet repository");
    for attempt in 0..MAX_ATTEMPTS {
        inner
            .counters
            .range_attempts
            .fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let result = async {
            let (_, mut stream) = repo
                .download_file_stream()
                .filename(file.path.clone())
                .revision(inner.revision.clone())
                .range(request.range.clone())
                .send()
                .await?;
            let capacity = usize::try_from(expected_len)
                .map_err(|_| anyhow!("range length {expected_len} does not fit this machine"))?;
            let mut bytes = Vec::with_capacity(capacity);
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                let next_len = bytes
                    .len()
                    .checked_add(chunk.len())
                    .ok_or_else(|| anyhow!("Xet range byte count overflows"))?;
                if next_len > capacity {
                    bail!(
                        "{}: Xet range {:?} exceeded expected length {expected_len}",
                        file.path,
                        request.range
                    );
                }
                bytes.extend_from_slice(&chunk);
            }
            if bytes.len() as u64 != expected_len {
                bail!(
                    "{}: Xet range {:?} returned {} bytes, expected {expected_len}",
                    file.path,
                    request.range,
                    bytes.len()
                );
            }
            Ok::<_, anyhow::Error>(bytes)
        }
        .await;
        match result {
            Ok(bytes) => {
                inner.counters.ranges_ok.fetch_add(1, Ordering::Relaxed);
                inner
                    .counters
                    .received_bytes
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                inner
                    .counters
                    .latency_micros
                    .lock()
                    .unwrap()
                    .push(started.elapsed().as_micros() as u64);
                return Ok(RangeResult {
                    ordinal: request.ordinal,
                    bytes,
                });
            }
            Err(error) if attempt + 1 < MAX_ATTEMPTS => {
                inner
                    .counters
                    .retries_transport
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(file = %file.path, ?error, attempt, "Xet range retry");
                backoff(attempt, None).await;
            }
            Err(error) => {
                return Err(error).context(format!("Xet range {}", file.path));
            }
        }
    }
    unreachable!("bounded retry loop returns or errors")
}

async fn refresh_url(inner: &Inner, file: &RemoteFile, rejected_url: &Url) -> Result<()> {
    let _refresh = file.refresh.lock().await;
    // Several in-flight ranges can discover the same expired URL together.
    // The first waiter refreshes it; later waiters reuse that result instead
    // of serially resolving the same shard once per failed request.
    if *file.signed_url.lock().await != *rejected_url {
        return Ok(());
    }
    let (url, size) = resolve_signed_url(
        &inner.resolver,
        &inner.endpoint,
        &inner.repo,
        &inner.revision,
        &file.path,
    )
    .await?;
    if size != 0 && size != file.size_bytes {
        bail!(
            "{} changed size at pinned commit {}: {} -> {size}",
            file.path,
            inner.revision,
            file.size_bytes
        );
    }
    *file.signed_url.lock().await = url;
    inner
        .counters
        .resolver_calls
        .fetch_add(1, Ordering::Relaxed);
    inner.counters.url_refreshes.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

async fn backoff(attempt: usize, retry_after: Option<Duration>) {
    let exponential = Duration::from_millis(100u64.saturating_mul(1 << attempt));
    let jitter = Duration::from_millis(((attempt * 37 + 17) % 53) as u64);
    tokio::time::sleep(retry_after.unwrap_or(exponential + jitter)).await;
}

async fn resolve_signed_url(
    resolver: &reqwest::Client,
    endpoint: &Url,
    repo: &str,
    revision: &str,
    path: &str,
) -> Result<(Url, u64)> {
    let url = resolve_url(endpoint, repo, revision, path)?;
    let response = resolver.head(url.clone()).send().await?;
    if !response.status().is_redirection() {
        bail!(
            "{repo}@{revision}/{path}: resolver returned {}, expected a redirect",
            response.status()
        );
    }
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| anyhow!("{repo}@{revision}/{path}: resolver redirect has no Location"))?;
    let signed = url.join(location)?;
    let size = response
        .headers()
        .get("x-linked-size")
        .or_else(|| response.headers().get(reqwest::header::CONTENT_LENGTH))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if let Some(commit) = response
        .headers()
        .get("x-repo-commit")
        .and_then(|value| value.to_str().ok())
        && commit != revision
    {
        bail!("{path}: resolver returned commit {commit}, expected {revision}");
    }
    Ok((signed, size))
}

fn split_spec(spec: &str) -> Result<(String, String)> {
    let (repo, revision) = spec.rsplit_once('@').unwrap_or((spec, "main"));
    let mut parts = repo.split('/');
    if parts.next().is_none() || parts.next().is_none() || parts.next().is_some() {
        bail!("expected owner/name or owner/name@revision, got {spec:?}");
    }
    Ok((repo.to_string(), revision.to_string()))
}

fn api_model_url(endpoint: &Url, repo: &str, revision: &str) -> Result<Url> {
    url_with_segments(endpoint, ["api", "models"])
        .and_then(|url| url_with_segments(&url, repo.split('/')))
        .and_then(|url| url_with_segments(&url, ["revision", revision]))
}

fn resolve_url(endpoint: &Url, repo: &str, revision: &str, path: &str) -> Result<Url> {
    url_with_segments(endpoint, repo.split('/'))
        .and_then(|url| url_with_segments(&url, ["resolve", revision]))
        .and_then(|url| url_with_segments(&url, path.split('/')))
}

fn url_with_segments<'a>(base: &Url, segments: impl IntoIterator<Item = &'a str>) -> Result<Url> {
    let mut url = base.clone();
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| anyhow!("HF endpoint cannot carry path segments"))?;
        path.pop_if_empty();
        path.extend(segments);
    }
    Ok(url)
}

fn auth_headers() -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let implicit_disabled =
        std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|value| !value.is_empty());
    let token = if implicit_disabled {
        None
    } else if let Ok(token) = std::env::var("HF_TOKEN") {
        (!token.is_empty()).then_some(token)
    } else if let Ok(path) = std::env::var("HF_TOKEN_PATH") {
        std::fs::read_to_string(path)
            .ok()
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty())
    } else {
        let hf_home = std::env::var_os("HF_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .map(std::path::PathBuf::from)
                    .map(|path| path.join("huggingface"))
            })
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(std::path::PathBuf::from)
                    .map(|path| path.join(".cache/huggingface"))
            });
        hf_home
            .and_then(|home| std::fs::read_to_string(home.join("token")).ok())
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty())
    };
    if let Some(token) = token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}"))
                .context("HF token is not a valid HTTP header value")?,
        );
    }
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use super::*;

    struct TestServer {
        url: Url,
        requests: Arc<AtomicUsize>,
    }

    impl TestServer {
        fn start(responses: Vec<String>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
            let requests = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&requests);
            thread::spawn(move || {
                for response in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    let mut request = [0u8; 4096];
                    let _ = stream.read(&mut request).unwrap();
                    observed.fetch_add(1, Ordering::SeqCst);
                    stream.write_all(response.as_bytes()).unwrap();
                }
            });
            Self { url, requests }
        }
    }

    fn response(status: &str, headers: &[(&str, &str)], body: &str) -> String {
        let mut response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str("\r\n");
        response.push_str(body);
        response
    }

    fn inner(signed_url: Url, endpoint: Url, size_bytes: u64) -> Arc<Inner> {
        Arc::new(Inner {
            repo: "owner/model".to_string(),
            revision: "commit".to_string(),
            endpoint,
            client: reqwest::Client::new(),
            resolver: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            files: vec![Arc::new(RemoteFile {
                path: "model.safetensors".to_string(),
                size_bytes,
                signed_url: AsyncMutex::new(signed_url),
                refresh: AsyncMutex::new(()),
            })],
            counters: Counters::new(),
        })
    }

    fn request(range: Range<u64>) -> RangeRequest {
        RangeRequest {
            ordinal: 7,
            file_id: 0,
            range,
        }
    }

    #[tokio::test]
    async fn accepts_only_the_exact_requested_range() {
        let server = TestServer::start(vec![response(
            "206 Partial Content",
            &[("Content-Range", "bytes 1-3/5")],
            "bcd",
        )]);
        let result = fetch_range(
            inner(server.url.clone(), server.url.clone(), 5),
            request(1..4),
        )
        .await
        .unwrap();
        assert_eq!(result.ordinal, 7);
        assert_eq!(result.bytes, b"bcd");
    }

    #[tokio::test]
    async fn rejects_a_mismatched_content_range() {
        let server = TestServer::start(vec![response(
            "206 Partial Content",
            &[("Content-Range", "bytes 0-2/5")],
            "abc",
        )]);
        let error = fetch_range(
            inner(server.url.clone(), server.url.clone(), 5),
            request(1..4),
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("returned Content-Range"));
    }

    #[tokio::test]
    async fn retries_429_within_the_fixed_attempt_budget() {
        let server = TestServer::start(vec![
            response("429 Too Many Requests", &[("Retry-After", "0")], ""),
            response(
                "206 Partial Content",
                &[("Content-Range", "bytes 1-3/5")],
                "bcd",
            ),
        ]);
        let source = inner(server.url.clone(), server.url.clone(), 5);
        let result = fetch_range(Arc::clone(&source), request(1..4))
            .await
            .unwrap();
        assert_eq!(result.bytes, b"bcd");
        assert_eq!(source.counters.retries_429.load(Ordering::Relaxed), 1);
        assert_eq!(server.requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn resolver_rejects_a_different_commit() {
        let server = TestServer::start(vec![response(
            "302 Found",
            &[
                ("Location", "/signed"),
                ("X-Repo-Commit", "other-commit"),
                ("X-Linked-Size", "5"),
            ],
            "",
        )]);
        let resolver = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let error = resolve_signed_url(
            &resolver,
            &server.url,
            "owner/model",
            "commit",
            "model.safetensors",
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("expected commit"));
    }

    #[tokio::test]
    async fn concurrent_expiry_refreshes_resolve_the_shard_once() {
        let server = TestServer::start(vec![response(
            "302 Found",
            &[
                ("Location", "/fresh-signed-url"),
                ("X-Repo-Commit", "commit"),
                ("X-Linked-Size", "5"),
            ],
            "",
        )]);
        let rejected = Url::parse("http://127.0.0.1:1/expired").unwrap();
        let source = inner(rejected.clone(), server.url.clone(), 5);
        let file = Arc::clone(&source.files[0]);
        let (first, second) = tokio::join!(
            refresh_url(&source, &file, &rejected),
            refresh_url(&source, &file, &rejected)
        );
        first.unwrap();
        second.unwrap();
        assert_eq!(server.requests.load(Ordering::SeqCst), 1);
        assert_eq!(source.counters.url_refreshes.load(Ordering::Relaxed), 1);
        assert!(
            file.signed_url
                .lock()
                .await
                .path()
                .ends_with("fresh-signed-url")
        );
    }
}
