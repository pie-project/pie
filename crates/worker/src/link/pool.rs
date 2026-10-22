//! A pool worker: it loads no model and computes nothing. It keeps the
//! sealed pages workers send it (`engine_cpu::Pool`), serves them back by key
//! on the bulk link, and tells the gateway's directory what it holds, the
//! runs it was asked to take among it, so a worker that lacks a run finds it
//! here as it would at any worker, and a cluster policy sees its groups.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use ids::{ReqId, SessionId, WorkerId};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};
use worker_api::{
    Accepted, Priority, Request, WorkerControl, connect_gateway_link, dispatch_codec,
};

/// How often the pool tells the gateway what it holds.
const REPORT_EVERY: Duration = Duration::from_secs(1);

#[derive(Clone)]
struct NoTurns {
    pool: engine_cpu::Pool,
}

/// The gateway dispatches no turn to a pool: no controller lists it. It
/// asks a pool to take runs, as the cluster policy replicates them.
impl WorkerControl for NoTurns {
    async fn replicate(
        self,
        _: tarpc::context::Context,
        keys: Vec<[u8; 32]>,
        from: String,
        retention: i64,
    ) -> Result<u32, String> {
        let pages = transport::tcp::fetch(&from, &keys, 0).await?;
        let count = pages.count as u32;
        let pool = self.pool.clone();
        tokio::task::spawn_blocking(move || pool.put_run(&keys, 0, pages, retention))
            .await
            .map_err(|e| e.to_string())??;
        tracing::info!(pages = count, %from, retention, "took a run");
        Ok(count)
    }

    async fn dispatch(self, _: tarpc::context::Context, _: Request) -> Accepted {
        Accepted::Reject
    }
    async fn cancel(self, _: tarpc::context::Context, _: ReqId) {}
    async fn set_priority(self, _: tarpc::context::Context, _: ReqId, _: Priority) {}
    async fn end_session(self, _: tarpc::context::Context, _: SessionId) {}
    async fn drain(self, _: tarpc::context::Context) {}
    async fn plex_session(self, _: tarpc::context::Context, _: String, _: Option<String>) {}
}

/// Where a pool keeps every page beyond its memory.
pub enum PoolStore {
    /// None: what memory cannot hold is refused.
    Memory,
    /// Files in a directory of its own.
    Dir(std::path::PathBuf),
    /// Objects under `prefix` in a store a NIXL storage plugin fronts
    /// (`OBJ` for S3), set up by the plugin's own params.
    Objects {
        plugin: String,
        params: Vec<(String, String)>,
        prefix: String,
    },
}

impl PoolStore {
    fn open(self, capacity: usize, agent: &str) -> Result<engine_cpu::Pool> {
        let store: Arc<dyn engine_cpu::Store> = match self {
            PoolStore::Memory => return Ok(engine_cpu::Pool::new(capacity)),
            PoolStore::Dir(dir) => Arc::new(
                engine_cpu::Dir::open(dir.clone())
                    .with_context(|| format!("open the pool's directory {}", dir.display()))?,
            ),
            PoolStore::Objects {
                plugin,
                params,
                prefix,
            } => objects(agent, &plugin, &params, prefix)?,
        };
        Ok(engine_cpu::Pool::with_store(capacity, store))
    }
}

#[cfg(feature = "nixl")]
fn objects(
    agent: &str,
    plugin: &str,
    params: &[(String, String)],
    prefix: String,
) -> Result<Arc<dyn engine_cpu::Store>> {
    let objects = transport::Objects::open(agent, plugin, params)
        .with_context(|| format!("open the object store through NIXL's `{plugin}`"))?;
    let store = engine_cpu::store::Objects::open(objects, prefix)
        .map_err(|e| anyhow::anyhow!("open the pool's objects: {e}"))?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "nixl"))]
fn objects(
    _: &str,
    _: &str,
    _: &[(String, String)],
    _: String,
) -> Result<Arc<dyn engine_cpu::Store>> {
    anyhow::bail!("this build keeps no object store: build with the `nixl` feature")
}

/// Serves a pool of `capacity` bytes of memory, beyond which it keeps pages
/// in `store`, on `listen` (the bulk link), and reports it to the gateway's
/// worker port `gateway` as worker `id`, dialing it again whenever the link
/// goes: what the pool holds outlives a gateway restart.
pub async fn run(
    listen: &str,
    gateway: &str,
    capacity: usize,
    store: PoolStore,
    id: WorkerId,
) -> Result<()> {
    let agent = format!("pie-pool-{id}");
    let pool = tokio::task::spawn_blocking(move || store.open(capacity, &agent)).await??;
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind the pool's bulk listener at {listen}"))?;
    tokio::spawn(transport::tcp::serve(listener, Arc::new(pool.clone())));
    tracing::info!(%id, %listen, capacity, "pool serving");
    loop {
        if let Err(error) = link(&pool, listen, gateway, id).await {
            tracing::warn!(error = %format!("{error:#}"), "the pool's gateway link is down; dialing again");
        }
        tokio::time::sleep(REDIAL_AFTER).await;
    }
}

/// A link that goes is dialed again after this.
const REDIAL_AFTER: Duration = Duration::from_secs(1);

/// One link to the gateway: registers, then reports what the pool holds,
/// its pages and its runs (the whole of it first), each tick until the link
/// goes.
async fn link(pool: &engine_cpu::Pool, listen: &str, gateway: &str, id: WorkerId) -> Result<()> {
    let addr = gateway.strip_prefix("tcp://").unwrap_or(gateway);
    let mut conn = tcp::connect(addr, dispatch_codec);
    conn.config_mut().max_frame_length(64 << 20);
    let transport = conn
        .await
        .with_context(|| format!("dialing the gateway at {gateway}"))?;
    let _ = transport.get_ref().set_nodelay(true);
    let (server_half, gateway) = connect_gateway_link(transport);
    let serving = tokio::spawn(
        BaseChannel::with_defaults(server_half)
            .execute(NoTurns { pool: pool.clone() }.serve())
            .for_each_concurrent(None, |req| async move {
                tokio::spawn(req);
            }),
    );
    let _serving = AbortOnDrop(serving);
    gateway
        .register(tarpc::context::current(), id)
        .await
        .context("registering the pool with the gateway")?;
    tracing::info!(%id, "pool registered with the gateway");

    let mut said: HashSet<[u8; 32]> = HashSet::new();
    let mut said_runs: HashMap<[u8; 32], Vec<[u8; 32]>> = HashMap::new();
    let mut full = true;
    loop {
        let now: HashSet<[u8; 32]> = pool.keys().into_iter().collect();
        let runs: HashMap<[u8; 32], Vec<[u8; 32]>> = pool
            .runs()
            .into_iter()
            .filter_map(|run| Some((*run.last()?, run)))
            .collect();
        // Sent each tick, changed or not: it is how the link is seen gone.
        let report = worker_api::StateReport {
            peer: Some(listen.to_string()),
            added: now.difference(&said).copied().collect(),
            removed: said.difference(&now).copied().collect(),
            full,
            runs_added: runs
                .iter()
                .filter(|(last, _)| !said_runs.contains_key(*last))
                .map(|(_, run)| run.clone())
                .collect(),
            runs_removed: said_runs
                .keys()
                .filter(|last| !runs.contains_key(*last))
                .copied()
                .collect(),
            pool: true,
            ..Default::default()
        };
        gateway
            .state(tarpc::context::current(), id, report)
            .await
            .context("the link to the gateway")?;
        tracing::debug!(
            pages = now.len(),
            runs = runs.len(),
            mb = pool.bytes() >> 20,
            "pool reported"
        );
        said = now;
        said_runs = runs;
        full = false;
        tokio::time::sleep(REPORT_EVERY).await;
    }
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}
