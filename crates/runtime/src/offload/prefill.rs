//! A pass prefilled on another worker.
//!
//! Home: before a pass fires, if its uncached rows are many and a prefill
//! peer is set, the home sends the peer its tokens. The peer prefills them,
//! and its sealed pages come back into the home's host tier as a cache
//! entry. The pass then runs as it always does: `supply` finds those pages,
//! promotes them, and maps them in. The page holding the pass's last row is
//! never one of them, so the home computes it, epilogue and all.
//!
//! Executor: [`serve`] prefills with the built-in `prefill` program, which
//! leaves its pages sealed in this worker's cache; the home then fetches
//! them by key over the bulk link ([`fetch`] answers it).
//!
//! Both workers hash under one model identity, so a page's key names the
//! same prefix on both. Their engines must lay a page out alike: v1 moves a
//! page's bytes as they are.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use futures::future::BoxFuture;

use crate::store::kv::hash::{self, Hash256};
use crate::store::registry::{Stores, with_kv_lock};

/// Pages' bytes as they come, in chunks of whole pages laid out as an
/// engine lays out a host slot (each chunk a read of its own); `layout` says
/// how, where the engine can.
pub struct PageStream {
    pub count: usize,
    pub page_bytes: usize,
    pub layout: Option<state::PageLayout>,
    /// The pages' tokens' hashes, page after page; empty when the source
    /// has none.
    pub slots: Vec<Hash256>,
    /// The groups (spans of a page) the chunks are parts of, each chunk one
    /// `transport::tcp::part`; empty: whole pages.
    pub groups: Vec<Vec<(u64, u64)>>,
    pub chunks: tokio::sync::mpsc::Receiver<std::result::Result<Vec<u8>, String>>,
}

/// Pages read out and sent at a time: the next is read while the link
/// carries this one.
const CHUNK_PAGES: usize = 32;

/// This worker's own host-tier layout.
pub async fn host_layout() -> Option<state::PageLayout> {
    crate::scheduler::host_kv_layout(0).await
}

/// The pages a peer prefilled, left sealed in its cache: `count` from token
/// `start` (a page boundary), to fetch from `addr`.
#[derive(Debug, Clone)]
pub struct RemotePages {
    pub start: u64,
    pub page_tokens: u32,
    pub count: usize,
    pub addr: String,
    /// The pages held on the executor's device for the asker to read in
    /// place, when it asked to and they were there to hold.
    pub held: Option<Held>,
}

/// An executor's pages held for one read: their device page ids, in run
/// order, until the asker releases the lease.
#[derive(Debug, Clone)]
pub struct Held {
    pub lease: u64,
    pub pages: Vec<u32>,
}

/// The worker a pass is offloaded to: the link layer's.
pub trait PrefillPeer: Send + Sync {
    /// Prefills on the peer; with `in_place` the pages are held there for
    /// [`PrefillPeer::read`].
    fn prefill(
        &self,
        tokens: Vec<u32>,
        from_page: u32,
        in_place: bool,
    ) -> BoxFuture<'static, Result<RemotePages>>;

    /// Whether this worker can read the peer's kv in place.
    fn reads_in_place(&self) -> bool {
        false
    }

    /// Reads the peer's device pages `remote` into this worker's `local`.
    fn read(&self, remote: Vec<u32>, local: Vec<u32>) -> BoxFuture<'static, Result<()>> {
        let _ = (remote, local);
        Box::pin(async { Err(anyhow!("this peer's kv is not read in place")) })
    }

    /// Lets the peer give the pages of a lease back.
    fn release(&self, lease: u64) -> BoxFuture<'static, Result<()>> {
        let _ = lease;
        Box::pin(async { Ok(()) })
    }
}

static PEER: RwLock<Option<Arc<dyn PrefillPeer>>> = RwLock::new(None);

pub fn set_peer(peer: Option<Arc<dyn PrefillPeer>>) {
    *PEER.write().unwrap() = peer;
}

fn peer() -> Option<Arc<dyn PrefillPeer>> {
    PEER.read().unwrap().clone()
}

pub fn has_peer() -> bool {
    PEER.read().unwrap().is_some()
}

/// Tokens per kv page on this worker's engine, once one serves a model.
pub fn page_tokens() -> Option<u32> {
    crate::store::registry::try_get(0, 0).map(|stores| stores.kv_page_size)
}

/// A pass's pages `[first, limit)` (the page holding its last row is the
/// home's), of which those before `have` this worker holds; and the tokens
/// up to `limit`.
struct Pass {
    first: u64,
    limit: u64,
    have: u64,
    tokens: Vec<u32>,
    keys: Vec<Hash256>,
    token_hashes: Vec<Vec<Option<Hash256>>>,
}

impl Pass {
    fn at(&self, page: u64) -> usize {
        (page - self.first) as usize
    }
}

/// Before a keyed, single-lane pass, for what its rows reach that this
/// worker holds nowhere: first the pages another worker holds, fetched by
/// key where the directory names one; then, when enough is left, a peer's
/// prefill of the rest. Any failure leaves the pass to run here.
pub(crate) async fn before_pass(
    pid: crate::inferlet::process::ProcessId,
    stores: &Stores,
    engine: usize,
    ws: crate::store::kv::page_table::WorkingSetId,
    held: u64,
    rows: &[u32],
    slots: &[Hash256],
) {
    let Some(mut pass) = plan(stores, ws, held, rows, slots) else {
        return;
    };
    // A run's keys name it from token 0: a pass that starts there can be
    // found whole in another worker's cache.
    if held == 0
        && let Some(directory) = super::directory::directory()
        && let Some(remote) = super::directory::remote()
    {
        let started = std::time::Instant::now();
        match fetch_held(stores, engine, &mut pass, &*directory, &*remote).await {
            Ok(0) => {}
            Ok(pages) => tracing::info!(
                target: "pie::offload",
                pages,
                ms = started.elapsed().as_millis() as u64,
                "a pass's pages came from another worker's cache"
            ),
            Err(error) => tracing::warn!(
                target: "pie::offload",
                %error,
                "another worker's cache could not give a pass's pages"
            ),
        }
    }
    let Some(min_tokens) = min_tokens(pid) else {
        return;
    };
    let page = stores.kv_page_size;
    if pass.have >= pass.limit || (pass.limit - pass.have) * u64::from(page) < min_tokens {
        return;
    }
    let Some(peer) = peer() else {
        return;
    };
    let started = std::time::Instant::now();
    match take(pid, stores, engine, peer, pass).await {
        Ok(pages) => {
            tracing::info!(
                target: "pie::offload",
                pages,
                ms = started.elapsed().as_millis() as u64,
                "a pass's pages were prefilled on the peer"
            );
            crate::policy::report(crate::policy::Event::Offloaded(
                pid,
                "peer".into(),
                u32::try_from(pages).unwrap_or(u32::MAX) * page,
            ));
        }
        Err(error) => {
            tracing::warn!(
                target: "pie::offload",
                %error,
                "the peer could not prefill a pass; it runs here"
            );
            crate::policy::report(crate::policy::Event::OffloadFailed(
                pid,
                format!("{error:#}"),
            ));
        }
    }
}

/// The fewest uncached tokens worth a peer's prefill: the worker policy's
/// rule, else the configuration's. Unstated, 2048: the pages come back
/// inline. `None`: this process's passes stay home.
fn min_tokens(pid: crate::inferlet::process::ProcessId) -> Option<u64> {
    use crate::policy::offload::Mode;
    match crate::policy::offload::rule(pid) {
        Some(None) => None,
        Some(Some(rule)) => match rule.mode {
            Mode::Never => None,
            Mode::Always => Some(u64::from(rule.min_tokens).max(1)),
            Mode::Cost => Some(u64::from(rule.min_tokens)),
        },
        None => Some(match super::prefill_min_suffix_tokens() {
            0 => 2048,
            n => n as u64,
        }),
    }
}

fn plan(
    stores: &Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    held: u64,
    rows: &[u32],
    slots: &[Hash256],
) -> Option<Pass> {
    let page = u64::from(stores.kv_page_size.max(1));
    if !held.is_multiple_of(page) || rows.is_empty() || slots.len() != rows.len() {
        return None;
    }
    let limit = (held + rows.len() as u64 - 1) / page;
    let first = held / page;
    if limit <= first {
        return None;
    }
    with_kv_lock(&stores.kv, "offload-plan", |kv| {
        let keys = kv.page_keys(ws, held, slots, stores.kv_page_size).ok()??;
        // The run is counted from token 0, as the keys are.
        let have = (kv.cached_run_any_tier(&keys).len() as u64).max(first);
        if have >= limit {
            return None;
        }
        let logged = kv.logged(ws).ok()?;
        if (logged.len() as u64) < held {
            return None;
        }
        let mut tokens = logged[..held as usize].to_vec();
        tokens.extend_from_slice(&rows[..(limit * page - held) as usize]);
        let local = |p: u64| ((p - first) * page) as usize;
        Some(Pass {
            first,
            limit,
            have,
            tokens,
            keys: keys[first as usize..limit as usize].to_vec(),
            token_hashes: (first..limit)
                .map(|p| {
                    slots[local(p)..local(p + 1)]
                        .iter()
                        .map(|h| Some(*h))
                        .collect()
                })
                .collect(),
        })
    })
}

/// Fetches, from the worker the directory names, the pass's pages past what
/// is held here; the pass then holds them too.
async fn fetch_held(
    stores: &Stores,
    engine: usize,
    pass: &mut Pass,
    directory: &dyn super::directory::Directory,
    remote: &dyn super::directory::Remote,
) -> Result<usize> {
    let holders = directory.locate(pass.keys.clone()).await?;
    let mut last = None;
    // Longest first; a holder that cannot give its pages (gone, going, or
    // its copy dropped since it reported) gives way to the next.
    for holder in holders {
        let to = (pass.first + u64::from(holder.pages)).min(pass.limit);
        if to <= pass.have {
            break;
        }
        let (have, end) = (pass.at(pass.have), pass.at(to));
        let fetched = async {
            let bytes = remote
                .fetch(holder.addr.clone(), pass.keys[..end].to_vec(), have as u32)
                .await?;
            if bytes.count != end - have {
                bail!(
                    "{} answered {} pages of {}",
                    holder.addr,
                    bytes.count,
                    end - have
                );
            }
            land(
                stores,
                engine,
                &pass.keys[..have],
                pass.token_hashes[have..end].to_vec(),
                pass.keys[have..end].to_vec(),
                bytes,
            )
            .await
        }
        .await;
        match fetched {
            Ok(()) => {
                pass.have = to;
                return Ok(end - have);
            }
            Err(error) => {
                tracing::debug!(target: "pie::offload", addr = %holder.addr, %error, "a holder gave nothing");
                last = Some(error);
            }
        }
    }
    match last {
        Some(error) => Err(error),
        None => Ok(0),
    }
}

async fn take(
    pid: crate::inferlet::process::ProcessId,
    stores: &Stores,
    engine: usize,
    peer: Arc<dyn PrefillPeer>,
    pass: Pass,
) -> Result<usize> {
    let t0 = std::time::Instant::now();
    let pages = peer
        .prefill(
            pass.tokens.clone(),
            u32::try_from(pass.have)?,
            peer.reads_in_place(),
        )
        .await?;
    let t_rpc = t0.elapsed();
    let page = u64::from(stores.kv_page_size);
    let want = (pass.limit - pass.have) as usize;
    if pages.page_tokens != stores.kv_page_size || pages.start != pass.have * page {
        bail!(
            "the peer answered pages of {} tokens from {}; this worker asked for {} from {}",
            pages.page_tokens,
            pages.start,
            stores.kv_page_size,
            pass.have * page
        );
    }
    if pages.count != want {
        bail!("the peer prefilled {} pages of {want}", pages.count);
    }
    let have = pass.at(pass.have);
    if let Some(held) = pages.held {
        let lease = held.lease;
        let read = read_in_place(pid, stores, &*peer, &pass, have, held).await;
        if let Err(error) = peer.release(lease).await {
            tracing::warn!(target: "pie::offload", %error, lease, "a peer's lease was not released");
        }
        read?;
        tracing::debug!(
            target: "pie::offload",
            rpc_ms = t_rpc.as_millis() as u64,
            read_ms = (t0.elapsed() - t_rpc).as_millis() as u64,
            "read a peer's pages in place"
        );
        return Ok(want);
    }
    let remote = super::directory::remote().ok_or_else(|| anyhow!("no bulk link here"))?;
    let bytes = remote
        .fetch(pages.addr.clone(), pass.keys.clone(), have as u32)
        .await?;
    if bytes.count != want {
        bail!("{} answered {} pages of {want}", pages.addr, bytes.count);
    }
    let n = bytes.count;
    land(
        stores,
        engine,
        &pass.keys[..have],
        pass.token_hashes[have..].to_vec(),
        pass.keys[have..].to_vec(),
        bytes,
    )
    .await?;
    tracing::debug!(
        target: "pie::offload",
        rpc_ms = t_rpc.as_millis() as u64,
        write_ms = (t0.elapsed() - t_rpc).as_millis() as u64,
        "took a peer's pages"
    );
    Ok(n)
}

/// Reads a peer's held pages straight into device pages this worker
/// reserves for `pid` through the planner, and files them as one cache
/// entry, device-resident, as [`land`] files host slots.
async fn read_in_place(
    pid: crate::inferlet::process::ProcessId,
    stores: &Stores,
    peer: &dyn PrefillPeer,
    pass: &Pass,
    have: usize,
    held: Held,
) -> Result<()> {
    let n = held.pages.len();
    let planner = crate::planner::planner().ok_or_else(|| anyhow!("no kv planner here"))?;
    let demand = crate::planner::Demand {
        kv_pages: u32::try_from(n)?,
        ..crate::planner::Demand::default()
    };
    let crate::planner::Acquired::Granted(mut grant) = planner
        .acquire(pid, pid, demand)
        .await
        .map_err(|error| anyhow!("reserving pages to read a peer's into: {error}"))?
    else {
        bail!("the planner asked the pass to yield before its pages were read");
    };
    let local: Vec<_> = grant.lend_kv().drain(..n).collect();
    drop(grant);
    if let Err(error) = peer
        .read(held.pages, local.iter().map(|page| page.0).collect())
        .await
    {
        with_kv_lock(&stores.kv, "offload-read", |kv| {
            kv.release_device_reservation(local)
        });
        return Err(error);
    }
    let filed = with_kv_lock(&stores.kv, "offload-read", |kv| {
        kv.import_cached_at(
            &pass.keys[..have],
            pass.token_hashes[have..].to_vec(),
            pass.keys[have..].to_vec(),
            local
                .into_iter()
                .map(crate::store::kv::page_table::KvPageBacking::Resident)
                .collect(),
        )
    });
    if !filed {
        bail!("the prefix the pages extend went while they were read");
    }
    Ok(())
}

/// Keeps the run `keys` (named from token 0) that the worker at bulk
/// address `from` holds, in this worker's host tier: what of it is not here
/// already, as one cache entry, the way the cluster policy replicates state.
/// The source must send the pages' tokens' hashes: a run kept without them
/// names its pages, but nothing after it could seal on it.
pub async fn replicate(keys: Vec<Hash256>, from: String) -> Result<usize> {
    let stores = crate::store::registry::try_get(0, 0)
        .ok_or_else(|| anyhow!("no engine serves a model here"))?;
    let have = with_kv_lock(&stores.kv, "offload-replicate", |kv| {
        kv.cached_run_any_tier(&keys).len()
    });
    let want = keys.len().saturating_sub(have);
    if want == 0 {
        return Ok(0);
    }
    let remote = super::directory::remote().ok_or_else(|| anyhow!("no bulk link here"))?;
    let mut pages = remote
        .fetch(from.clone(), keys.clone(), have as u32)
        .await?;
    if pages.count != want {
        bail!("{from} answered {} pages of {want}", pages.count);
    }
    let page = stores.kv_page_size as usize;
    let slots = std::mem::take(&mut pages.slots);
    if slots.len() != want * page {
        bail!("{from} sent the pages without their tokens' hashes");
    }
    let token_hashes = slots
        .chunks(page)
        .map(|hashes| hashes.iter().copied().map(Some).collect())
        .collect();
    land(
        &stores,
        0,
        &keys[..have],
        token_hashes,
        keys[have..].to_vec(),
        pages,
    )
    .await?;
    Ok(want)
}

/// Lands pages another worker sealed in this worker's host tier, as one
/// cache entry, writing each chunk as it comes.
async fn land(
    stores: &Stores,
    engine: usize,
    prefix: &[Hash256],
    token_hashes: Vec<Vec<Option<Hash256>>>,
    keys: Vec<Hash256>,
    mut stream: PageStream,
) -> Result<()> {
    let n = stream.count;
    if n == 0 {
        return Ok(());
    }
    if !stream.groups.is_empty() {
        bail!("pages sent a part at a time are laid back out before they land");
    }
    // Pages another layout holds (another split across ranks) are laid out
    // as this engine holds them, a chunk at a time.
    let to = crate::scheduler::host_kv_layout(engine).await;
    let convert = match (&stream.layout, &to) {
        (Some(from), Some(to)) if from != to => Some((from.clone(), to.clone())),
        _ => None,
    };
    let t0 = std::time::Instant::now();
    let host = with_kv_lock(&stores.kv, "offload-land", |kv| kv.alloc_host_slots(n))
        .ok_or_else(|| anyhow!("no room in the host tier for {n} pages"))?;
    let ids: Vec<u32> = host.iter().map(|slot| slot.0).collect();
    let written = async {
        let mut at = 0usize;
        while let Some(chunk) = stream.chunks.recv().await {
            let chunk = chunk.map_err(|why| anyhow!(why))?;
            let k = chunk.len().checked_div(stream.page_bytes).unwrap_or(0);
            if k == 0 || at + k > n || k * stream.page_bytes != chunk.len() {
                bail!("a chunk of {} bytes is not whole pages", chunk.len());
            }
            let data = match &convert {
                Some((from, to)) => {
                    state::reshard(&chunk, k, from, to).map_err(|why| anyhow!(why))?
                }
                None => chunk,
            };
            crate::scheduler::write_host_kv(engine, ids[at..at + k].to_vec(), data).await?;
            at += k;
        }
        if at != n {
            bail!("the pages stopped at {at} of {n}");
        }
        Ok(())
    }
    .await;
    if let Err(error) = written {
        with_kv_lock(&stores.kv, "offload-land", |kv| kv.free_host_slots(host));
        return Err(error);
    }
    let elapsed = t0.elapsed();
    let landed = with_kv_lock(&stores.kv, "offload-land", |kv| {
        kv.import_cached(prefix, token_hashes, keys, host)
    });
    tracing::debug!(
        target: "pie::offload",
        pages = n,
        ms = elapsed.as_millis() as u64,
        "landed pages"
    );
    if !landed {
        bail!("the prefix the pages extend went while they came");
    }
    Ok(())
}

/// Each whole page of `tokens` from token 0: its slots' hashes and its key.
fn keyed_pages(stores: &Stores, tokens: &[u32]) -> (Vec<Vec<Option<Hash256>>>, Vec<Hash256>) {
    let page = stores.kv_page_size.max(1) as usize;
    let whole = tokens.len() / page * page;
    let domain = with_kv_lock(&stores.kv, "offload-keys", |kv| kv.domain());
    let slots = hash::chain_slots(&domain, None, 0, &tokens[..whole]);
    let token_hashes: Vec<Vec<Option<Hash256>>> = slots
        .chunks_exact(page)
        .map(|page| page.iter().map(|h| Some(*h)).collect())
        .collect();
    let keys = token_hashes
        .iter()
        .map(|page| hash::page_hash(page))
        .collect();
    (token_hashes, keys)
}

/// Prefills `tokens` here and reads out the pages from `from_page` that the
/// run fills completely. Pages before `from_page` this worker holds nowhere
/// are first fetched from `home`, which holds them, so only the rest is
/// computed.
/// How long an executor holds pages for an asker that never releases them
/// (it died, or its link did).
const LEASE_TTL: Duration = Duration::from_secs(60);

struct Lease {
    kv: Arc<parking_lot::Mutex<crate::store::kv::KvStore>>,
    pin: crate::store::kv::ReaderPin,
    since: std::time::Instant,
}

/// Pages this executor holds for askers to read in place, by lease.
static LEASES: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<u64, Lease>>> =
    std::sync::LazyLock::new(Default::default);

static NEXT_LEASE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Gives back the pages lease `lease` held; a lease already gone is no fault.
pub fn release(lease: u64) {
    let held = LEASES.lock().unwrap().remove(&lease);
    if let Some(held) = held {
        held.kv.lock().release_reader(held.pin);
    }
}

/// Gives back what askers held past [`LEASE_TTL`].
fn expire_leases() {
    let stale: Vec<Lease> = {
        let mut leases = LEASES.lock().unwrap();
        let old: Vec<u64> = leases
            .iter()
            .filter(|(_, lease)| lease.since.elapsed() >= LEASE_TTL)
            .map(|(&id, _)| id)
            .collect();
        old.into_iter()
            .filter_map(|id| leases.remove(&id))
            .collect()
    };
    for lease in stale {
        tracing::warn!(target: "pie::offload", "an asker never released its pages; given back");
        lease.kv.lock().release_reader(lease.pin);
    }
}

/// Holds `keys[from..]`, device-resident, for an asker to read in place;
/// `None` when some of them are not on the device.
fn hold(stores: &Stores, keys: &[Hash256], from: usize) -> Option<Held> {
    let (pin, pages) = with_kv_lock(&stores.kv, "offload-hold", |kv| {
        let run = kv.cached_run_any_tier(keys);
        (run.len() == keys.len() && from < run.len())
            .then(|| kv.pin_resident(&run[from..]))
            .flatten()
    })?;
    let lease = NEXT_LEASE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    LEASES.lock().unwrap().insert(
        lease,
        Lease {
            kv: stores.kv.clone(),
            pin,
            since: std::time::Instant::now(),
        },
    );
    Some(Held {
        lease,
        pages: pages.into_iter().map(|page| page.0).collect(),
    })
}

pub async fn serve(
    tokens: Vec<u32>,
    from_page: u32,
    home: Option<String>,
    in_place: bool,
) -> Result<RemotePages> {
    expire_leases();
    let stores = crate::store::registry::try_get(0, 0)
        .ok_or_else(|| anyhow!("no engine serves a model here"))?;
    let engine = 0usize;
    let page = u64::from(stores.kv_page_size.max(1));
    let from = u64::from(from_page);
    let (token_hashes, keys) = keyed_pages(&stores, &tokens);
    let pages = keys.len() as u64;
    if from >= pages {
        return Ok(RemotePages {
            start: from * page,
            page_tokens: stores.kv_page_size,
            count: 0,
            addr: String::new(),
            held: None,
        });
    }
    let t0 = std::time::Instant::now();
    let have = with_kv_lock(&stores.kv, "offload-serve", |kv| {
        kv.cached_run_any_tier(&keys[..from as usize]).len()
    });
    let mut fetched = 0;
    if have < from as usize
        && let Some(home) = home
        && let Some(remote) = super::directory::remote()
    {
        let range = have..from as usize;
        match remote
            .fetch(home, keys[..from as usize].to_vec(), have as u32)
            .await
        {
            Ok(bytes) if bytes.count == range.len() => {
                fetched = bytes.count;
                land(
                    &stores,
                    engine,
                    &keys[..have],
                    token_hashes[range.clone()].to_vec(),
                    keys[range].to_vec(),
                    bytes,
                )
                .await?;
            }
            Ok(bytes) => tracing::warn!(
                target: "pie::offload",
                got = bytes.count,
                "the home answered a short prefix; the rest is computed here"
            ),
            Err(error) => tracing::warn!(
                target: "pie::offload",
                %error,
                "the home could not give its prefix; it is computed here"
            ),
        }
    }
    let t_fetch = t0.elapsed();
    run_prefill(&tokens[..(pages * page) as usize]).await?;
    let t_run = t0.elapsed();
    // The program has exited; its working set's pages are kept as it is
    // released, which may land just after.
    let mut kept = false;
    for _ in 0..50 {
        kept = with_kv_lock(&stores.kv, "offload-serve", |kv| {
            kv.cached_run_any_tier(&keys).len() == keys.len()
        });
        if kept {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    if !kept {
        bail!("the prefill's pages are not all cached here");
    }
    tracing::debug!(
        target: "pie::offload",
        have,
        fetched,
        fetch_ms = t_fetch.as_millis() as u64,
        run_ms = (t_run - t_fetch).as_millis() as u64,
        "served a prefill"
    );
    Ok(RemotePages {
        start: from * page,
        page_tokens: stores.kv_page_size,
        count: (pages - from) as usize,
        addr: String::new(),
        held: in_place
            .then(|| hold(&stores, &keys, from as usize))
            .flatten(),
    })
}

/// The bytes of this worker's sealed pages `keys[from..]`, for a peer that
/// asks by key. `keys` run from token 0: a run is found from its first page.
pub async fn fetch(keys: Vec<Hash256>, from: u32) -> Result<PageStream> {
    let stores = crate::store::registry::try_get(0, 0)
        .ok_or_else(|| anyhow!("no engine serves a model here"))?;
    export(&stores, 0, &keys, from as usize, false).await
}

/// Reads out sealed pages `keys[from..]` a chunk at a time; all of `keys`
/// must be found here as one run, and what is read must be on the device.
/// They are held until the last chunk is read. With `wait`, gives a
/// just-ended program's release a moment to keep them.
async fn export(
    stores: &Stores,
    engine: usize,
    keys: &[Hash256],
    from: usize,
    wait: bool,
) -> Result<PageStream> {
    let tries = if wait { 50 } else { 1 };
    let mut held = None;
    for _ in 0..tries {
        held = with_kv_lock(&stores.kv, "offload-export", |kv| {
            let run = kv.cached_run_any_tier(keys);
            if run.len() != keys.len() || from >= run.len() {
                return None;
            }
            let hashes = kv.run_slots(&run[from..]).unwrap_or_default();
            kv.pin_resident(&run[from..])
                .map(|(pin, device)| (pin, device, hashes))
        });
        if held.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let (pin, device, hashes) =
        held.ok_or_else(|| anyhow!("those pages are not all cached on the device here"))?;
    let release = |stores: &Stores, pin| {
        with_kv_lock(&stores.kv, "offload-export", |kv| kv.release_reader(pin))
    };
    let n = device.len();
    let staging = match with_kv_lock(&stores.kv, "offload-export", |kv| kv.alloc_host_slots(n)) {
        Some(staging) => staging,
        None => {
            release(stores, pin);
            bail!("no room in the host tier to stage {n} pages");
        }
    };
    let slots: Vec<u32> = staging.iter().map(|slot| slot.0).collect();
    let layout = crate::scheduler::host_kv_layout(engine).await;
    // One read: a round trip through the engine's lane per chunk costs more
    // than the chunks save. The link and the asker's landing go by chunks.
    let t0 = std::time::Instant::now();
    let read = read_chunk(engine, &device, &slots).await;
    with_kv_lock(&stores.kv, "offload-export", |kv| {
        kv.free_host_slots(staging)
    });
    release(stores, pin);
    let bytes = read?;
    tracing::debug!(
        target: "pie::offload",
        pages = n,
        ms = t0.elapsed().as_millis() as u64,
        "read pages out"
    );
    let page_bytes = bytes.len() / n;
    let (tx, chunks) = tokio::sync::mpsc::channel(2);
    tokio::spawn(async move {
        for chunk in bytes.chunks(CHUNK_PAGES * page_bytes) {
            if tx.send(Ok(chunk.to_vec())).await.is_err() {
                return;
            }
        }
    });
    Ok(PageStream {
        count: n,
        page_bytes,
        layout,
        slots: hashes,
        groups: Vec::new(),
        chunks,
    })
}

/// Device pages `device`, copied to host `slots` and read.
async fn read_chunk(
    engine: usize,
    device: &[crate::store::kv::page_table::PhysicalKvPageId],
    slots: &[u32],
) -> Result<Vec<u8>> {
    let t0 = std::time::Instant::now();
    let bytes = crate::scheduler::read_device_kv(
        engine,
        device.iter().map(|id| id.0).collect(),
        slots[..device.len()].to_vec(),
    )
    .await?;
    tracing::debug!(
        target: "pie::offload",
        ms = t0.elapsed().as_millis() as u64,
        "read a chunk"
    );
    if bytes.is_empty() || bytes.len() % device.len() != 0 {
        bail!(
            "{} bytes do not split into {} pages",
            bytes.len(),
            device.len()
        );
    }
    Ok(bytes)
}

/// The built-in program [`serve`] runs; no HTTP route serves it.
pub const PROGRAM: &str = "prefill";

async fn run_prefill(tokens: &[u32]) -> Result<()> {
    use crate::inferlet::program;
    let name = program::newest(PROGRAM)
        .await
        .ok_or_else(|| anyhow!("the built-in `prefill` program is not installed"))?;
    program::install(&name).await?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    let input = serde_json::json!({ "tokens": tokens }).to_string();
    crate::inferlet::process::spawn(
        "pie".into(),
        name,
        input,
        None,
        false,
        Some(tx),
        crate::policy::Arrival::default(),
    )?;
    rx.await
        .map_err(|_| anyhow!("the prefill program went away"))?
        .map_err(|why| anyhow!("the prefill program failed: {why}"))?;
    Ok(())
}
