//! Reading another worker's kv in place. A worker whose engine exported its
//! kv (`engine::Engine::export_kv_handle`) opens one transfer agent, its kv
//! registered with it, and says so in its peer [`Hello`]; a worker that
//! greets it and holds kv of the same layout learns its agent, and reads its
//! pages straight into its own.
//!
//! [`Hello`]: worker_api::peer::Hello

use anyhow::Result;
use futures::future::BoxFuture;
use worker_api::peer::KvAccess;

/// This worker's open kv agent: peers read and write the kv it registered
/// until it drops, and not after.
#[must_use = "the agent closes when this drops"]
pub struct KvAgent(());

impl Drop for KvAgent {
    fn drop(&mut self) {
        imp::close();
    }
}

/// Opens this worker's agent over its exported kv; its name is the worker's
/// bulk address, which is its own on a host. A build without a transfer
/// agent opens none, and the bulk link carries every page.
///
/// # Safety
///
/// As for [`transport::Backend::register`]: every region of `handle` must
/// stay allocated, and unraced by this worker while peers may read or write
/// it, until the returned [`KvAgent`] drops.
#[allow(unsafe_code)]
pub unsafe fn open(name: &str, handle: engine::KvHandle) -> Result<KvAgent> {
    // SAFETY: the caller's promise, passed on.
    unsafe { imp::open(name, handle) }?;
    Ok(KvAgent(()))
}

/// What this worker's peer [`worker_api::peer::Hello`] says of its kv.
pub fn access() -> Option<KvAccess> {
    imp::access()
}

/// Learns a peer's agent, if its kv has this worker's layout; the key to
/// read it by.
pub fn connect(access: &KvAccess) -> Result<Option<u64>> {
    imp::connect(access)
}

/// Reads the peer's pages `remote` (its device page ids) into this worker's
/// pages `local`, page for page, every plane of each.
pub fn read(peer: u64, remote: Vec<u32>, local: Vec<u32>) -> BoxFuture<'static, Result<()>> {
    imp::read(peer, remote, local)
}

#[cfg(feature = "nixl")]
mod imp {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use anyhow::{Context, Result, anyhow, bail};
    use futures::future::BoxFuture;
    use transport::{Backend, Completion, PageSet, PeerConn, RegisteredHandle, WorkerId};
    use worker_api::peer::KvAccess;

    /// How long a read is polled between looks.
    const POLL: Duration = Duration::from_micros(200);

    /// How long closing waits for the reads in flight to let the agent go.
    const DRAIN: Duration = Duration::from_secs(10);

    struct Agent {
        backend: transport::NixlBackend,
        mine: RegisteredHandle,
        access: KvAccess,
        /// Peers learned, by their agent's name: the key each is read by.
        peers: Mutex<HashMap<Vec<u8>, u64>>,
    }

    /// The open agent, if any. A read holds its own reference for as long as
    /// its transfer may be in flight, so closing waits on it.
    static AGENT: Mutex<Option<Arc<Agent>>> = Mutex::new(None);

    fn agent() -> Option<Arc<Agent>> {
        AGENT.lock().expect("kv agent lock poisoned").clone()
    }

    /// The key this worker's own kv is registered under.
    const OURS: WorkerId = WorkerId(0);

    #[allow(unsafe_code)]
    pub unsafe fn open(name: &str, handle: engine::KvHandle) -> Result<()> {
        let backend = transport::NixlBackend::new(name).context("opening the kv transfer agent")?;
        // SAFETY: `open`'s caller keeps `handle`'s regions allocated and
        // unraced until its `KvAgent` drops, and the agent cannot outlive
        // that: it lands in `AGENT`, which `close` empties and drains, or is
        // dropped right here.
        let mine = unsafe { backend.register(OURS, handle.clone()) }
            .context("registering this worker's kv with its transfer agent")?;
        let access = KvAccess {
            handle: serde_json::to_vec(&handle)?,
            agent: backend.local_metadata()?,
        };
        let mut open = AGENT.lock().expect("kv agent lock poisoned");
        if open.is_some() {
            bail!("this worker's kv transfer agent is open already");
        }
        *open = Some(Arc::new(Agent {
            backend,
            mine,
            access,
            peers: Mutex::new(HashMap::new()),
        }));
        tracing::info!(agent = name, "kv readable in place by peers");
        Ok(())
    }

    /// Takes the agent down: no read starts after, and the backend (whose
    /// drop releases its transfers and destroys the agent with every
    /// registration) drops once the reads in flight let it go.
    pub fn close() {
        let Some(mut agent) = AGENT.lock().expect("kv agent lock poisoned").take() else {
            return;
        };
        let deadline = Instant::now() + DRAIN;
        loop {
            match Arc::try_unwrap(agent) {
                Ok(agent) => {
                    drop(agent);
                    return;
                }
                Err(held) if Instant::now() < deadline => {
                    agent = held;
                    std::thread::sleep(POLL);
                }
                Err(_) => {
                    tracing::warn!(
                        "a kv read held the transfer agent past closing; it drops with the read"
                    );
                    return;
                }
            }
        }
    }

    pub fn access() -> Option<KvAccess> {
        Some(agent()?.access.clone())
    }

    pub fn connect(access: &KvAccess) -> Result<Option<u64>> {
        let Some(agent) = agent() else {
            return Ok(None);
        };
        let handle: engine::KvHandle =
            serde_json::from_slice(&access.handle).context("reading a peer's kv handle")?;
        if !agent.mine.handle().layout.compatible_with(&handle.layout) {
            return Ok(None);
        }
        let mut peers = agent.peers.lock().unwrap();
        if let Some(&key) = peers.get(&access.agent) {
            return Ok(Some(key));
        }
        let key = peers.len() as u64 + 1;
        agent.backend.connect(&PeerConn {
            worker: WorkerId(key),
            handle,
            metadata: access.agent.clone(),
        })?;
        peers.insert(access.agent.clone(), key);
        Ok(Some(key))
    }

    pub fn read(peer: u64, remote: Vec<u32>, local: Vec<u32>) -> BoxFuture<'static, Result<()>> {
        Box::pin(async move {
            let agent = agent().ok_or_else(|| anyhow!("this worker has no kv transfer agent"))?;
            let id = agent.backend.recv_mapped(
                &agent.mine,
                &PageSet::new(local),
                &PageSet::new(remote),
                WorkerId(peer),
            )?;
            loop {
                match agent.backend.poll(id)? {
                    Completion::Done => return Ok(()),
                    Completion::Pending => tokio::time::sleep(POLL).await,
                    Completion::Failed(why) => bail!("reading a peer's pages: {why}"),
                }
            }
        })
    }
}

#[cfg(not(feature = "nixl"))]
mod imp {
    use anyhow::{Result, bail};
    use futures::future::BoxFuture;
    use worker_api::peer::KvAccess;

    #[allow(unsafe_code)]
    pub unsafe fn open(_name: &str, _handle: engine::KvHandle) -> Result<()> {
        Ok(())
    }

    pub fn close() {}

    pub fn access() -> Option<KvAccess> {
        None
    }

    pub fn connect(_access: &KvAccess) -> Result<Option<u64>> {
        Ok(None)
    }

    pub fn read(_peer: u64, _remote: Vec<u32>, _local: Vec<u32>) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { bail!("this build reads no peer's kv in place") })
    }
}
