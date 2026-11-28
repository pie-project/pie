//! Reading another worker's kv in place. A worker whose engine exported its
//! kv (`engine::Peers::export`) opens one transfer agent, its kv
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

/// The transfers one agent has posted, counted from posting until the agent
/// reports each final, and the agent's closing, which waits out exactly
/// those. Kept apart from NIXL so its rules hold for any agent that reports
/// a transfer's status when asked.
#[cfg(any(feature = "nixl", test))]
mod transfers {
    use std::collections::{HashMap, HashSet};
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    use anyhow::{Result, anyhow, bail};

    /// How long a transfer is left between looks at its status. NIXL tells
    /// the side that posts a transfer of its end only when asked
    /// (`nixl_capi_get_xfer_status`): its notifications (`gen_notif`,
    /// `get_notifs`, a transfer's `has_notif`) are delivered to the remote
    /// agent, here the peer whose pages are read, and are themselves polled;
    /// its progress thread (`enable_prog_thread`) advances the backend but
    /// signals nothing. No fd, event or callback wakes the poster, so a look
    /// is the only way to learn a transfer ended.
    pub(super) const POLL: Duration = Duration::from_micros(200);

    /// Where one transfer stands, as its agent reports it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) enum Status {
        Pending,
        Done,
        Failed(String),
    }

    /// An agent's word on the transfers it posted, by id. Once it reports
    /// one final (done, failed, or an error) the transfer is out of flight
    /// and the agent forgets it.
    pub(super) trait Agent {
        fn status(&self, id: u64) -> Result<Status>;
    }

    pub(super) struct Transfers<A> {
        state: Mutex<State<A>>,
        /// Signalled when closing has dropped the agent, and when a read's
        /// wait ends: closing sleeps on it while reads still drive the looks.
        changed: Condvar,
    }

    struct State<A> {
        /// The agent, until closing has waited out every transfer it posted.
        agent: Option<A>,
        /// No transfer posts once set.
        closing: bool,
        /// Posted and not yet reported final.
        inflight: HashSet<u64>,
        /// Final statuses a look collected for a transfer other than its
        /// own, kept for the read that posted it.
        settled: HashMap<u64, Status>,
        /// Reads waiting on a transfer now: each looks once a [`POLL`].
        waiting: usize,
    }

    impl<A: Agent> State<A> {
        /// Asks the agent about `id`, moving it out of flight once final.
        fn ask(&mut self, id: u64) {
            let Some(agent) = self.agent.as_ref() else {
                return;
            };
            let status = match agent.status(id) {
                Ok(Status::Pending) => return,
                Ok(status) => status,
                Err(error) => Status::Failed(format!("{error:#}")),
            };
            self.inflight.remove(&id);
            self.settled.insert(id, status);
        }

        /// One look on the transfer progress path: `id`'s, and while closing
        /// every transfer in flight (a read whose future was dropped has no
        /// one else to look at it). The agent drops the moment closing finds
        /// nothing in flight.
        fn progress(&mut self, id: Option<u64>) {
            if let Some(id) = id.filter(|id| self.inflight.contains(id)) {
                self.ask(id);
            }
            if self.closing {
                let ids: Vec<u64> = self.inflight.iter().copied().collect();
                for id in ids {
                    self.ask(id);
                }
                if self.inflight.is_empty() {
                    drop(self.agent.take());
                }
            }
        }
    }

    impl<A: Agent> Transfers<A> {
        pub fn new(agent: A) -> Self {
            Self {
                state: Mutex::new(State {
                    agent: Some(agent),
                    closing: false,
                    inflight: HashSet::new(),
                    settled: HashMap::new(),
                    waiting: 0,
                }),
                changed: Condvar::new(),
            }
        }

        fn state(&self) -> std::sync::MutexGuard<'_, State<A>> {
            self.state.lock().expect("kv transfers lock poisoned")
        }

        /// Runs `f` on the agent, unless it is closing.
        pub fn with<R>(&self, f: impl FnOnce(&A) -> Result<R>) -> Result<R> {
            let state = self.state();
            match &state.agent {
                Some(agent) if !state.closing => f(agent),
                _ => bail!("this worker's kv transfer agent is closing"),
            }
        }

        /// Posts a transfer with `post`, which returns its id; it counts as
        /// in flight until reported final.
        pub fn post(&self, post: impl FnOnce(&A) -> Result<u64>) -> Result<u64> {
            let mut state = self.state();
            let id = match &state.agent {
                Some(agent) if !state.closing => post(agent)?,
                _ => bail!("this worker's kv transfer agent is closing"),
            };
            state.inflight.insert(id);
            Ok(id)
        }

        /// One look for transfer `id`'s read: its final status, taken once,
        /// or `Pending`.
        fn look(&self, id: u64) -> Result<Status> {
            let mut state = self.state();
            state.progress(Some(id));
            if let Some(status) = state.settled.remove(&id) {
                return Ok(status);
            }
            if !state.inflight.contains(&id) {
                bail!("kv transfer {id} is not in flight");
            }
            let dropped = state.agent.is_none();
            drop(state);
            if dropped {
                self.changed.notify_all();
            }
            Ok(Status::Pending)
        }

        /// Waits transfer `id`, a read of `peer`'s pages, to its end; a
        /// failure names the peer. Its looks are the progress path closing
        /// rides on.
        pub async fn wait(&self, peer: u64, id: u64) -> Result<()> {
            let _waiting = Waiting::new(self);
            loop {
                match self.look(id) {
                    Ok(Status::Done) => return Ok(()),
                    Ok(Status::Pending) => tokio::time::sleep(POLL).await,
                    Ok(Status::Failed(why)) => bail!("reading peer {peer}'s pages: {why}"),
                    Err(error) => return Err(error.context(format!("reading peer {peer}'s pages"))),
                }
            }
        }

        /// Closes: no transfer posts after, every one in flight is waited
        /// out to the status its agent reports final, and then the agent
        /// drops. A read whose future was dropped still counts: its transfer
        /// is in flight until the agent says otherwise. While reads are
        /// waiting, their looks progress every transfer and closing sleeps
        /// until they end or the agent drops; with none, closing looks
        /// itself, once a [`POLL`], since nothing else will.
        pub fn close(&self) {
            let mut state = self.state();
            state.closing = true;
            loop {
                state.progress(None);
                if state.agent.is_none() {
                    return;
                }
                if state.waiting > 0 {
                    state = self
                        .changed
                        .wait_while(state, |state| state.agent.is_some() && state.waiting > 0)
                        .expect("kv transfers lock poisoned");
                } else {
                    drop(state);
                    std::thread::sleep(POLL);
                    state = self.state();
                }
            }
        }
    }

    /// A read counted as waiting while its wait runs, its future dropped
    /// or not.
    struct Waiting<'a, A>(&'a Transfers<A>);

    impl<'a, A> Waiting<'a, A> {
        fn new(transfers: &'a Transfers<A>) -> Self {
            transfers
                .state
                .lock()
                .expect("kv transfers lock poisoned")
                .waiting += 1;
            Self(transfers)
        }
    }

    impl<A> Drop for Waiting<'_, A> {
        fn drop(&mut self) {
            if let Ok(mut state) = self.0.state.lock() {
                state.waiting -= 1;
            }
            self.0.changed.notify_all();
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};

        use super::*;

        /// An agent whose transfers end when the test says.
        #[derive(Default)]
        struct Fake {
            transfers: States,
            dropped: Arc<AtomicBool>,
        }

        impl Fake {
            fn start(&self) -> u64 {
                let mut transfers = self.transfers.lock().unwrap();
                let id = transfers.len() as u64 + 100;
                transfers.insert(id, Status::Pending);
                id
            }
        }

        impl Agent for Fake {
            fn status(&self, id: u64) -> Result<Status> {
                let mut transfers = self.transfers.lock().unwrap();
                let status = transfers
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| anyhow!("no transfer {id}"))?;
                if status != Status::Pending {
                    transfers.remove(&id);
                }
                Ok(status)
            }
        }

        impl Drop for Fake {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::SeqCst);
            }
        }

        type States = Arc<Mutex<HashMap<u64, Status>>>;

        fn agent() -> (Arc<Transfers<Fake>>, States, Arc<AtomicBool>) {
            let fake = Fake::default();
            let (states, dropped) = (fake.transfers.clone(), fake.dropped.clone());
            (Arc::new(Transfers::new(fake)), states, dropped)
        }

        fn end(states: &Mutex<HashMap<u64, Status>>, id: u64, status: Status) {
            states.lock().unwrap().insert(id, status);
        }

        #[tokio::test]
        async fn closing_waits_out_exactly_the_transfers_in_flight() {
            let (transfers, states, dropped) = agent();
            let a = transfers.post(|fake| Ok(fake.start())).unwrap();
            // Its read was dropped: nobody waits on it, yet it is in flight.
            let b = transfers.post(|fake| Ok(fake.start())).unwrap();

            let closing = std::thread::spawn({
                let transfers = transfers.clone();
                move || transfers.close()
            });
            while transfers.with(|_| Ok(())).is_ok() {
                std::thread::yield_now();
            }
            assert!(transfers.post(|fake| Ok(fake.start())).is_err());
            assert!(!dropped.load(Ordering::SeqCst));

            end(&states, a, Status::Done);
            transfers.wait(1, a).await.unwrap();
            assert!(!dropped.load(Ordering::SeqCst), "b is still in flight");

            end(&states, b, Status::Done);
            closing.join().unwrap();
            assert!(dropped.load(Ordering::SeqCst));
            // Closing collected b's end; its read, had it stayed, finds it.
            transfers.wait(1, b).await.unwrap();
        }
    }
}

#[cfg(feature = "nixl")]
mod imp {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use anyhow::{Context, Result, anyhow, bail};
    use futures::future::BoxFuture;
    use transport::{
        Backend, Completion, PageSet, PeerConn, RegisteredHandle, TransferId, WorkerId,
    };
    use worker_api::peer::KvAccess;

    use super::transfers::{Status, Transfers};

    /// The NIXL agent, with this worker's kv registered on it.
    struct Nixl {
        backend: transport::NixlBackend,
        mine: RegisteredHandle,
    }

    impl super::transfers::Agent for Nixl {
        fn status(&self, id: u64) -> Result<Status> {
            Ok(match self.backend.poll(TransferId(id))? {
                Completion::Pending => Status::Pending,
                Completion::Done => Status::Done,
                Completion::Failed(why) => Status::Failed(why),
            })
        }
    }

    struct Agent {
        transfers: Transfers<Nixl>,
        access: KvAccess,
        /// Peers learned, by their agent's name: the key each is read by.
        peers: Mutex<HashMap<Vec<u8>, u64>>,
    }

    /// The open agent, if any. A read holds a reference to wait its transfer
    /// out, but not the NIXL agent: closing drops that once the transfers in
    /// flight end, whoever still holds this.
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
        // that: it lands in `AGENT`, whose `close` drops it once no transfer
        // of it is in flight, or is dropped right here.
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
            transfers: Transfers::new(Nixl { backend, mine }),
            access,
            peers: Mutex::new(HashMap::new()),
        }));
        tracing::info!(agent = name, "kv readable in place by peers");
        Ok(())
    }

    /// Takes the agent down: no read starts after, and the NIXL agent (whose
    /// drop destroys it with every registration) drops once each transfer it
    /// posted has ended, as NIXL reports it.
    pub fn close() {
        let Some(agent) = AGENT.lock().expect("kv agent lock poisoned").take() else {
            return;
        };
        agent.transfers.close();
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
        let mut peers = agent.peers.lock().unwrap();
        if let Some(&key) = peers.get(&access.agent) {
            return Ok(Some(key));
        }
        let key = peers.len() as u64 + 1;
        let connected = agent.transfers.with(|nixl| {
            if !nixl.mine.handle().layout.compatible_with(&handle.layout) {
                return Ok(false);
            }
            nixl.backend.connect(&PeerConn {
                worker: WorkerId(key),
                handle,
                metadata: access.agent.clone(),
            })?;
            Ok(true)
        })?;
        if !connected {
            return Ok(None);
        }
        peers.insert(access.agent.clone(), key);
        Ok(Some(key))
    }

    pub fn read(peer: u64, remote: Vec<u32>, local: Vec<u32>) -> BoxFuture<'static, Result<()>> {
        Box::pin(async move {
            let agent = agent().ok_or_else(|| anyhow!("this worker has no kv transfer agent"))?;
            let id = agent.transfers.post(|nixl| {
                Ok(nixl
                    .backend
                    .recv_mapped(
                        &nixl.mine,
                        &PageSet::new(local),
                        &PageSet::new(remote),
                        WorkerId(peer),
                    )?
                    .0)
            })?;
            agent.transfers.wait(peer, id).await
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
