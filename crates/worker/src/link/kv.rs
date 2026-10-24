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

/// Opens this worker's agent over its exported kv; its name is the worker's
/// bulk address, which is its own on a host. A build without a transfer
/// agent opens none, and the bulk link carries every page.
///
/// # Safety
///
/// As for [`transport::Backend::register`]: every region of `handle` must
/// stay allocated, and unraced by this worker while peers may read or write
/// it, for the rest of the process, since the agent is never closed.
#[allow(unsafe_code)]
pub unsafe fn open(name: &str, handle: engine::KvHandle) -> Result<()> {
    // SAFETY: the caller's promise, passed on.
    unsafe { imp::open(name, handle) }
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
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;

    use anyhow::{Context, Result, anyhow, bail};
    use futures::future::BoxFuture;
    use transport::{Backend, Completion, PageSet, PeerConn, RegisteredHandle, WorkerId};
    use worker_api::peer::KvAccess;

    /// How long a read is polled between looks.
    const POLL: Duration = Duration::from_micros(200);

    struct Agent {
        backend: Arc<transport::NixlBackend>,
        mine: RegisteredHandle,
        access: KvAccess,
        /// Peers learned, by their agent's name: the key each is read by.
        peers: Mutex<HashMap<Vec<u8>, u64>>,
    }

    static AGENT: OnceLock<Agent> = OnceLock::new();

    /// The key this worker's own kv is registered under.
    const OURS: WorkerId = WorkerId(0);

    #[allow(unsafe_code)]
    pub unsafe fn open(name: &str, handle: engine::KvHandle) -> Result<()> {
        let backend =
            Arc::new(transport::NixlBackend::new(name).context("opening the kv transfer agent")?);
        // SAFETY: `open`'s caller keeps `handle`'s regions allocated and
        // unraced for the rest of the process, which the agent cannot
        // outlive: it lands in `AGENT` or is dropped right here.
        let mine = unsafe { backend.register(OURS, handle.clone()) }
            .context("registering this worker's kv with its transfer agent")?;
        let access = KvAccess {
            handle: serde_json::to_vec(&handle)?,
            agent: backend.local_metadata()?,
        };
        AGENT
            .set(Agent {
                backend,
                mine,
                access,
                peers: Mutex::new(HashMap::new()),
            })
            .map_err(|_| anyhow!("this worker's kv transfer agent is open already"))?;
        tracing::info!(agent = name, "kv readable in place by peers");
        Ok(())
    }

    pub fn access() -> Option<KvAccess> {
        Some(AGENT.get()?.access.clone())
    }

    pub fn connect(access: &KvAccess) -> Result<Option<u64>> {
        let Some(agent) = AGENT.get() else {
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
            let agent = AGENT
                .get()
                .ok_or_else(|| anyhow!("this worker has no kv transfer agent"))?;
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
