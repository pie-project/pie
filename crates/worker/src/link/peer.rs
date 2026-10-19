//! The worker-to-worker link: this worker serves [`Peer`] on
//! `offload.listen`, and dials the workers named in `offload.peers` to
//! prefill passes on them.
//!
//! A peer is taken only when it serves the same model, in the same class and
//! page layout: v1 moves a page's bytes as they are. The bytes ride the bulk
//! link (`transport::tcp`), one port above the peer link.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use futures::future::BoxFuture;
use ids::WorkerId;
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};
use worker_api::dispatch_codec;
use worker_api::peer::{Hello, PEER_MAX_FRAME_BYTES, Pages, Peer, PeerClient, Prefill};

/// A prefill may run long; the link waits this long for it.
/// How long a pass waits for the directory to say who holds its pages.
const LOCATE_DEADLINE: Duration = Duration::from_millis(500);

const PREFILL_DEADLINE: Duration = Duration::from_secs(300);

/// What this worker says of itself. Its class is the engine and the
/// deployment, which names the served variant; the page size rides in the
/// format.
pub fn hello(worker: WorkerId, engine: &str) -> Hello {
    let identity = runtime::model::identity().cloned();
    let page_tokens = runtime::offload::prefill::page_tokens();
    Hello {
        worker,
        class: identity.as_ref().map(|_| state::NumericsClass {
            engine: engine.to_string(),
            weights: untensored(runtime::model::model().name()),
            kv: String::new(),
        }),
        format: page_tokens.map(|page_tokens| state::StateFormat {
            layers: 0,
            planes: state::Planes::KeyValue,
            kv_heads: 0,
            head_dim: 0,
            page_tokens,
            dtype: String::new(),
            dtype_bits: 0,
            head_major: false,
            rank: 0,
            ranks: 1,
        }),
        identity,
    }
}

/// A deployment's name without its tensor-parallel suffix: how a page is
/// split across ranks is a layout, which the landing side converts, not a
/// class.
fn untensored(deployment: &str) -> String {
    match deployment.rsplit_once("-tp") {
        Some((base, ranks)) if !ranks.is_empty() && ranks.bytes().all(|b| b.is_ascii_digit()) => {
            base.to_string()
        }
        _ => deployment.to_string(),
    }
}

#[derive(Clone)]
struct PeerServer {
    hello: Arc<Hello>,
}

/// This worker's way to other workers' pages: a bulk fetch from their
/// bulk-bytes address.
#[derive(Default)]
pub struct Clients;

impl runtime::offload::directory::Remote for Clients {
    fn fetch(
        &self,
        addr: String,
        keys: Vec<[u8; 32]>,
        from: u32,
    ) -> BoxFuture<'static, Result<runtime::offload::prefill::PageStream>> {
        Box::pin(async move {
            let stream = transport::tcp::fetch_stream(&addr, &keys, from)
                .await
                .map_err(|why| anyhow::anyhow!("fetching pages: {why}"))?;
            Ok(runtime::offload::prefill::PageStream {
                count: stream.count,
                page_bytes: stream.page_bytes,
                layout: serde_json::from_slice(&stream.meta).ok(),
                chunks: stream.chunks,
            })
        })
    }
}

/// Makes the bulk link the runtime's way to other workers' pages.
pub fn clients() {
    runtime::offload::directory::set_remote(Some(Arc::new(Clients)));
}

/// Serves this worker's sealed pages on the bulk link.
struct Source;

impl transport::tcp::Source for Source {
    fn read(&self, keys: Vec<[u8; 32]>, from: u32) -> transport::tcp::Answer {
        Box::pin(async move {
            let pages = runtime::offload::prefill::fetch(keys, from)
                .await
                .map_err(|e| format!("{e:#}"))?;
            Ok(transport::tcp::Stream {
                count: pages.count,
                page_bytes: pages.page_bytes,
                meta: pages
                    .layout
                    .as_ref()
                    .and_then(|layout| serde_json::to_vec(layout).ok())
                    .unwrap_or_default(),
                chunks: pages.chunks,
            })
        })
    }
}

/// The bulk link listens one port above the peer link.
pub fn bulk_addr(listen: &str) -> Result<String> {
    let addr: std::net::SocketAddr = listen
        .strip_prefix("tcp://")
        .unwrap_or(listen)
        .parse()
        .with_context(|| format!("`{listen}` is not an address and port"))?;
    Ok(std::net::SocketAddr::new(addr.ip(), addr.port() + 1).to_string())
}

static LISTEN: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Where this worker serves its pages' bytes, as it tells the directory and
/// the peers it sends passes to.
pub fn set_listen(addr: String) {
    let _ = LISTEN.set(addr);
}

pub fn listen() -> Option<String> {
    LISTEN.get().cloned()
}

async fn connect(addr: &str) -> Result<PeerClient> {
    let addr = addr.strip_prefix("tcp://").unwrap_or(addr);
    let mut conn = tcp::connect(addr, dispatch_codec);
    conn.config_mut().max_frame_length(PEER_MAX_FRAME_BYTES);
    let transport = conn
        .await
        .with_context(|| format!("dialing the peer at {addr}"))?;
    let _ = transport.get_ref().set_nodelay(true);
    Ok(PeerClient::new(tarpc::client::Config::default(), transport).spawn())
}

impl Peer for PeerServer {
    async fn hello(self, _: tarpc::context::Context, _: Hello) -> Result<Hello, String> {
        Ok((*self.hello).clone())
    }

    async fn prefill(self, _: tarpc::context::Context, request: Prefill) -> Result<Pages, String> {
        let Some(class) = self.hello.class.clone() else {
            return Err("this worker serves no model".into());
        };
        let format = self
            .hello
            .format
            .clone()
            .ok_or("this worker has no kv store")?;
        let started = Instant::now();
        let bulk = listen().ok_or("this worker serves no bulk link")?;
        let pages =
            runtime::offload::prefill::serve(request.tokens, request.from_page, request.home)
                .await
                .map_err(|e| format!("{e:#}"))?;
        tracing::info!(
            target: "pie::offload",
            pages = pages.count,
            ms = started.elapsed().as_millis() as u64,
            "prefilled a peer's pass"
        );
        Ok(Pages {
            start: pages.start,
            count: pages.count as u32,
            class,
            format,
            bulk,
        })
    }
}

/// Serves [`Peer`] on `listen` for as long as the process lives.
pub async fn serve(listen: &str, hello: Hello) -> Result<std::net::SocketAddr> {
    let mut incoming = tcp::listen(listen, dispatch_codec)
        .await
        .with_context(|| format!("bind the peer listener at {listen}"))?;
    incoming.config_mut().max_frame_length(PEER_MAX_FRAME_BYTES);
    let bound = incoming.local_addr();
    let server = PeerServer {
        hello: Arc::new(hello),
    };
    let bulk = bulk_addr(listen)?;
    let bulk_listener = tokio::net::TcpListener::bind(&bulk)
        .await
        .with_context(|| format!("bind the bulk listener at {bulk}"))?;
    tokio::spawn(transport::tcp::serve(bulk_listener, Arc::new(Source)));
    tokio::spawn(async move {
        while let Some(conn) = incoming.next().await {
            let transport = match conn {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "peer accept error");
                    continue;
                }
            };
            let _ = transport.get_ref().set_nodelay(true);
            let server = server.clone();
            tokio::spawn(
                BaseChannel::with_defaults(transport)
                    .execute(server.serve())
                    .for_each_concurrent(None, |req| async move {
                        tokio::spawn(req);
                    }),
            );
        }
    });
    tracing::info!(%bound, %bulk, "serving peers");
    Ok(bound)
}

struct Dialed {
    client: PeerClient,
    /// This worker's bulk-bytes address, for the peer to fetch prefixes from.
    home: Option<String>,
}

impl runtime::offload::prefill::PrefillPeer for Dialed {
    fn prefill(
        &self,
        tokens: Vec<u32>,
        from_page: u32,
    ) -> BoxFuture<'static, Result<runtime::offload::prefill::RemotePages>> {
        let client = self.client.clone();
        let home = self.home.clone();
        Box::pin(async move {
            let mut ctx = tarpc::context::current();
            ctx.deadline = Instant::now() + PREFILL_DEADLINE;
            let t0 = Instant::now();
            let pages = client
                .prefill(
                    ctx,
                    Prefill {
                        tokens,
                        from_page,
                        home,
                    },
                )
                .await
                .context("the peer link")?
                .map_err(|why| anyhow::anyhow!("the peer refused: {why}"))?;
            tracing::debug!(target: "pie::offload", ms = t0.elapsed().as_millis() as u64, "the peer answered");
            Ok(runtime::offload::prefill::RemotePages {
                start: pages.start,
                page_tokens: pages.format.page_tokens,
                count: pages.count as usize,
                addr: pages.bulk,
            })
        })
    }
}

/// Dials `addr` and, if it can take this worker's passes, makes it the
/// runtime's prefill peer.
pub async fn dial(addr: &str, ours: &Hello, listen: Option<String>) -> Result<()> {
    let (client, theirs) = greet(addr, ours).await?;
    runtime::offload::prefill::set_peer(Some(Arc::new(Dialed {
        client,
        home: listen,
    })));
    tracing::info!(peer = %addr, worker = %theirs.worker, "prefilling passes on a peer");
    Ok(())
}

/// The peer link to `addr`, once its worker said it can take this worker's
/// passes: one model, one class, one page format.
async fn greet(addr: &str, ours: &Hello) -> Result<(PeerClient, Hello)> {
    let client = connect(addr).await?;
    let theirs = client
        .hello(tarpc::context::current(), ours.clone())
        .await
        .context("greeting the peer")?
        .map_err(|why| anyhow::anyhow!("the peer refused the greeting: {why}"))?;
    if theirs.identity.is_none() || theirs.identity != ours.identity {
        bail!("the peer at {addr} serves another model");
    }
    if theirs.class != ours.class {
        bail!(
            "the peer at {addr} computes in another class ({:?}); this build moves pages only between workers of one class",
            theirs.class
        );
    }
    let page = |hello: &Hello| hello.format.as_ref().map(|f| f.page_tokens);
    if page(&theirs) != page(ours) {
        bail!("the peer at {addr} pages its kv differently");
    }
    Ok((client, theirs))
}

/// The peer link listens one port below the bulk link.
fn peer_of_bulk(bulk: &str) -> Result<String> {
    let addr: std::net::SocketAddr = bulk
        .parse()
        .with_context(|| format!("`{bulk}` is not an address and port"))?;
    let port = addr
        .port()
        .checked_sub(1)
        .with_context(|| format!("`{bulk}` has no port below it"))?;
    Ok(std::net::SocketAddr::new(addr.ip(), port).to_string())
}

/// Dials the executors a policy names, by their bulk addresses.
pub struct Dialer {
    pub hello: Hello,
}

impl runtime::offload::prefill::PrefillDialer for Dialer {
    fn dial(
        &self,
        bulk: String,
    ) -> BoxFuture<'static, Result<Arc<dyn runtime::offload::prefill::PrefillPeer>>> {
        let hello = self.hello.clone();
        Box::pin(async move {
            let addr = peer_of_bulk(&bulk)?;
            let (client, theirs) = greet(&addr, &hello).await?;
            tracing::info!(executor = %addr, worker = %theirs.worker, "an executor the policy named takes passes");
            let peer: Arc<dyn runtime::offload::prefill::PrefillPeer> = Arc::new(Dialed {
                client,
                home: listen(),
            });
            Ok(peer)
        })
    }
}

/// The gateway's directory, asked by this worker.
struct GatewayDirectory {
    gateway: worker_api::GatewayInboundClient,
    worker: WorkerId,
}

impl runtime::offload::directory::Directory for GatewayDirectory {
    fn locate(
        &self,
        keys: Vec<[u8; 32]>,
    ) -> BoxFuture<'static, Result<Vec<runtime::offload::directory::Holder>>> {
        let (gateway, worker) = (self.gateway.clone(), self.worker);
        Box::pin(async move {
            // An optimization: past a moment the pass is computed here.
            let mut ctx = tarpc::context::current();
            ctx.deadline = Instant::now() + LOCATE_DEADLINE;
            let found = gateway
                .locate(ctx, worker, keys)
                .await
                .context("asking the gateway's directory")?;
            Ok(found
                .into_iter()
                .map(|(_, addr, pages)| runtime::offload::directory::Holder { addr, pages })
                .collect())
        })
    }
}

/// How often this worker tells the gateway what it holds.
const REPORT_EVERY: Duration = Duration::from_secs(1);

/// Tells `gateway` what sealed pages this worker holds, all of it first and
/// then what changed, for as long as the link lives; and makes it the
/// runtime's directory.
pub async fn report_state(gateway: worker_api::GatewayInboundClient, worker: WorkerId) {
    clients();
    runtime::offload::directory::set_directory(Some(Arc::new(GatewayDirectory {
        gateway: gateway.clone(),
        worker,
    })));
    let peer = listen();
    let mut said: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut said_runs: std::collections::HashMap<[u8; 32], Vec<[u8; 32]>> =
        std::collections::HashMap::new();
    let mut full = true;
    let mut said_published = Vec::new();
    loop {
        let now = runtime::offload::directory::sealed_keys();
        let published: Vec<(String, String, String)> = runtime::policy::published()
            .into_iter()
            .map(|(of, name, value)| (of, name, value.to_string()))
            .collect();
        let runs: std::collections::HashMap<[u8; 32], Vec<[u8; 32]>> =
            runtime::offload::directory::cached_runs()
                .into_iter()
                .filter_map(|run| Some((*run.last()?, run)))
                .collect();
        let report = worker_api::StateReport {
            peer: peer.clone(),
            added: if full {
                now.iter().copied().collect()
            } else {
                now.difference(&said).copied().collect()
            },
            removed: if full {
                Vec::new()
            } else {
                said.difference(&now).copied().collect()
            },
            full,
            runs_added: runs
                .iter()
                .filter(|(last, _)| full || !said_runs.contains_key(*last))
                .map(|(_, run)| run.clone())
                .collect(),
            runs_removed: if full {
                Vec::new()
            } else {
                said_runs
                    .keys()
                    .filter(|last| !runs.contains_key(*last))
                    .copied()
                    .collect()
            },
            pool: false,
            published: (full || published != said_published).then(|| published.clone()),
        };
        // Sent even when nothing changed: it is also how a link finds its
        // gateway gone, and ends so the link is dialed again.
        match gateway
            .state(tarpc::context::current(), worker, report)
            .await
        {
            Ok(()) => {
                said = now;
                said_runs = runs;
                said_published = published;
                full = false;
            }
            Err(tarpc::client::RpcError::Shutdown) => {
                tracing::info!("the gateway link is down; ending it");
                return;
            }
            Err(error) => {
                tracing::debug!(%error, "state report not delivered; resending whole");
                full = true;
            }
        }
        tokio::time::sleep(REPORT_EVERY).await;
    }
}

/// Sends every run this worker keeps to `pools`, each from the first page a
/// pool has not been sent, for as long as the process lives: a cache that
/// outlives this worker's, for any worker to fetch from.
pub async fn push_to_pools(pools: Vec<String>) {
    let mut sent: std::collections::HashMap<String, std::collections::HashSet<[u8; 32]>> =
        std::collections::HashMap::new();
    loop {
        tokio::time::sleep(REPORT_EVERY).await;
        let runs = runtime::offload::directory::cached_runs();
        for pool in &pools {
            let sent = sent.entry(pool.clone()).or_default();
            for run in &runs {
                let Some(from) = run.iter().position(|key| !sent.contains(key)) else {
                    continue;
                };
                match push(pool, run, from).await {
                    Ok(pages) => {
                        sent.extend(run[from..].iter().copied());
                        tracing::debug!(target: "pie::offload", %pool, pages, "sent a run to a pool");
                    }
                    Err(error) => {
                        tracing::debug!(target: "pie::offload", %pool, %error, "a run not sent to a pool");
                    }
                }
            }
        }
    }
}

async fn push(pool: &str, run: &[[u8; 32]], from: usize) -> Result<usize> {
    let stream = runtime::offload::prefill::fetch(run.to_vec(), from as u32).await?;
    let meta = stream
        .layout
        .as_ref()
        .and_then(|layout| serde_json::to_vec(layout).ok())
        .unwrap_or_default();
    let pages = transport::tcp::Stream {
        count: stream.count,
        page_bytes: stream.page_bytes,
        meta,
        chunks: stream.chunks,
    }
    .collect()
    .await
    .map_err(|why| anyhow::anyhow!(why))?;
    let count = pages.count;
    transport::tcp::put(pool, &run[from..], &pages)
        .await
        .map_err(|why| anyhow::anyhow!(why))?;
    Ok(count)
}
