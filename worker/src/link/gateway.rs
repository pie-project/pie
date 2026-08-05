//! Worker data-plane: **dial INTO the gateway** (post-inversion, design §8/M3).
//!
//! Pre-0.5.0 the gateway dialed the worker's edge-rpc listener and pulled the
//! session stream (`WorkerSessionApi::recv` long-poll). Post-inversion the
//! topology flips: the **gateway is the listening server** (1:N fan-in) and the
//! worker **dials in**. One worker-initiated connection carries both data-plane
//! services, split with [`pie_worker_rpc::connect_gateway_link`]:
//!
//! - the worker **serves** [`pie_worker_rpc::WorkerControl`] (the gateway calls
//!   `dispatch`/`cancel`/`set_priority`/`drain`), and
//! - the worker **holds** a [`GatewayInboundClient`] to push the token stream
//!   back (`push_tokens`), announce itself (`register`), and bounce turns
//!   (`redirect`).
//!
//! The token stream rides the plain client→server direction (worker→gateway
//! `push_tokens`); latency-sensitive commands go reverse. `register(worker_id)`
//! is the FIRST call on a fresh connection so the gateway can key this worker's
//! reverse `WorkerControlClient` into its registry before any `dispatch`.
//!
//! ## Runtime bridge
//! Each gateway logical [`SessionId`] maps to one runtime session
//! ([`pie_engine::server::open_session`]) — warm KV across a multi-turn session. A
//! per-session driver task feeds each turn's [`Request::message`] into the
//! runtime ([`pie_engine::server::send_client_message`]) and pumps the resulting
//! `ServerMessage`s back out as [`Tokens::Chunk`], terminated by one
//! [`Tokens::Eos`] when the turn completes. Backpressure is inherent: the
//! runtime outbox is bounded, so a slow `push_tokens` (slow gateway/user) stalls
//! the pump and backpressures generation (design §6).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use pie_client_api::{ClientMessage, ServerMessage};
use pie_controller_rpc::GatewayEndpoint;
use pie_engine::server::ClientId;
use pie_ids::{ReqId, SessionId, WorkerId};
use pie_worker_rpc::{
    Accepted, Control, GatewayInboundClient, Priority, Request, Tokens, WorkerControl,
    connect_gateway_link, dispatch_codec,
};
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel};
use tokio::sync::{Mutex, Notify, mpsc};

/// Max frame on the gateway link's read side. A `dispatch` carries one
/// `Request` whose `ClientMessage` can hold a large prompt / upload chunk, so we
/// keep the generous cap the old edge path used. Token chunks (the reverse,
/// gateway-decoded direction) are small now that blobs ride HTTP, so the
/// gateway sets its own (smaller) cap independently.
const LINK_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// `push_tokens` client deadline. Generous so a backpressured (slow-consumer)
/// push blocks rather than spuriously erroring — the gateway replies `Control`
/// once its bounded pipe has room (design §6). A true transport error surfaces
/// immediately regardless of this bound.
const PUSH_DEADLINE: Duration = Duration::from_secs(300);

/// Per-session driver mailbox depth: queued turns awaiting the in-flight one.
const TURN_QUEUE_DEPTH: usize = 64;

/// A live dial-in connection to one gateway: the task serving `WorkerControl`
/// over the split's server-half. The connection's mux pump tasks are spawned
/// inside [`connect_gateway_link`] and die with the transport; aborting the
/// serve task tears the link down on shutdown.
pub struct GatewayLink {
    serve_task: tokio::task::JoinHandle<()>,
}

/// Tear the dial-in serve loop down when the link is dropped, so reconciliation
/// (dropping a link that left the roster) and shutdown (dropping the manager)
/// both stop the connection.
impl Drop for GatewayLink {
    fn drop(&mut self) {
        self.serve_task.abort();
    }
}

/// Dial `addr` (`tcp://host:port`, a bare `host:port`, or `unix:/path`), split
/// the connection into the two data-plane services, `register(worker_id)` first,
/// then serve `WorkerControl` for the gateway to dispatch onto this worker.
pub async fn connect_gateway(addr: &str, worker_id: WorkerId) -> Result<GatewayLink> {
    let (server_half, gateway) = if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        let mut conn = unix::connect(path, dispatch_codec);
        conn.config_mut().max_frame_length(LINK_MAX_FRAME_BYTES);
        let transport = conn
            .await
            .with_context(|| format!("dialing gateway at {addr}"))?;
        connect_gateway_link(transport)
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let mut conn = tcp::connect(tcp_addr, dispatch_codec);
        conn.config_mut().max_frame_length(LINK_MAX_FRAME_BYTES);
        let transport = conn
            .await
            .with_context(|| format!("dialing gateway at {addr}"))?;
        connect_gateway_link(transport)
    };

    // Register FIRST: keys this worker's reverse WorkerControlClient into the
    // gateway's registry before any dispatch can target it.
    gateway
        .register(tarpc::context::current(), worker_id)
        .await
        .with_context(|| format!("registering worker with gateway at {addr}"))?;
    tracing::info!(%worker_id, gateway = %addr, "worker registered with gateway (dial-in)");

    let server = WorkerControlServer {
        worker_id,
        gateway,
        sessions: Arc::new(SessionRegistry::default()),
    };
    let serve_task = tokio::spawn(
        BaseChannel::with_defaults(server_half)
            .execute(server.serve())
            .for_each_concurrent(None, |req| async move {
                tokio::spawn(req);
            }),
    );

    Ok(GatewayLink { serve_task })
}

/// The worker's live dial-in links, reconciled against the controller-pushed
/// gateway roster (design `gateway.md`). Owns one [`GatewayLink`] per dialed
/// gateway, keyed by dial address.
///
/// `pinned` addresses (the static `--gateway` override) are always kept dialed
/// regardless of the roster — the override for fixed/local topologies and the
/// single-node in-proc gateway. Roster-derived links are added and dropped as
/// the gateway fleet scales up/down within one watch round-trip.
pub struct GatewayLinkManager {
    worker_id: WorkerId,
    pinned: HashSet<String>,
    links: HashMap<String, GatewayLink>,
}

impl GatewayLinkManager {
    /// A manager for `worker_id`, pinning `pinned` addresses (never dropped by
    /// reconciliation). Nothing is dialed yet — call [`dial_pinned`] for boot
    /// readiness and [`reconcile`] on each roster update. Addresses are
    /// canonicalized (see [`canonical_addr`]) so a pinned `tcp://h:p` and a
    /// roster `h:p` for the same gateway resolve to a single link.
    ///
    /// [`dial_pinned`]: Self::dial_pinned
    /// [`reconcile`]: Self::reconcile
    pub fn new(worker_id: WorkerId, pinned: Vec<String>) -> Self {
        Self {
            worker_id,
            pinned: pinned.iter().map(|a| canonical_addr(a)).collect(),
            links: HashMap::new(),
        }
    }

    /// Dial every pinned address now, failing if any pinned dial fails — the
    /// static override is a hard boot requirement (matches the pre-dynamic
    /// contract). Roster dials, by contrast, are best-effort and never fail boot.
    pub async fn dial_pinned(&mut self) -> Result<()> {
        let mut pinned: Vec<String> = self.pinned.iter().cloned().collect();
        pinned.sort(); // deterministic dial order / logs
        for addr in pinned {
            if self.links.contains_key(&addr) {
                continue;
            }
            let link = connect_gateway(&addr, self.worker_id)
                .await
                .with_context(|| format!("dialing pinned gateway at {addr}"))?;
            self.links.insert(addr, link);
        }
        Ok(())
    }

    /// Reconcile dial-in links against a fresh gateway roster: the desired set is
    /// `pinned ∪ roster` (canonicalized). Drop (and abort, via [`GatewayLink`]'s
    /// `Drop`) every link no longer desired, then dial every desired address not
    /// yet linked. A roster dial that fails is logged and retried on the next
    /// roster update.
    pub async fn reconcile(&mut self, roster: &[GatewayEndpoint]) {
        let desired: HashSet<String> = self
            .pinned
            .iter()
            .cloned()
            .chain(roster.iter().map(|g| canonical_addr(&g.addr)))
            .collect();

        // Drop links that left the roster (and are not pinned).
        let stale: Vec<String> = self
            .links
            .keys()
            .filter(|addr| !desired.contains(*addr))
            .cloned()
            .collect();
        for addr in stale {
            self.links.remove(&addr); // Drop aborts the serve task
            tracing::info!(
                worker = %self.worker_id,
                gateway = %addr,
                "dropped gateway link (left roster)"
            );
        }

        // Dial newly-added desired addresses.
        let to_dial: Vec<String> = desired
            .into_iter()
            .filter(|addr| !self.links.contains_key(addr))
            .collect();
        for addr in to_dial {
            match connect_gateway(&addr, self.worker_id).await {
                Ok(link) => {
                    self.links.insert(addr, link);
                }
                Err(e) => tracing::warn!(
                    worker = %self.worker_id,
                    gateway = %addr,
                    error = %e,
                    "gateway dial failed; will retry on next roster update"
                ),
            }
        }
    }

    /// The addresses currently linked — the advertised URL / boot banner.
    pub fn addrs(&self) -> Vec<String> {
        let mut addrs: Vec<String> = self.links.keys().cloned().collect();
        addrs.sort();
        addrs
    }
}

/// Canonicalize a dial address so the same gateway reached via a pinned
/// `tcp://host:port` and a roster `host:port` maps to one link key. Strips the
/// `tcp://` scheme (which [`connect_gateway`] also treats as optional) while
/// leaving `unix:` addresses intact.
fn canonical_addr(addr: &str) -> String {
    if addr.starts_with("unix:") {
        addr.to_string()
    } else {
        addr.strip_prefix("tcp://").unwrap_or(addr).to_string()
    }
}

/// The worker's `WorkerControl` server for one gateway connection. Cloned per
/// request by tarpc, so all fields are cheap to clone (the registry is shared).
#[derive(Clone)]
struct WorkerControlServer {
    worker_id: WorkerId,
    /// Push side back to THIS gateway; cloned into each session driver so a
    /// turn's tokens return to the gateway that dispatched it.
    gateway: GatewayInboundClient,
    sessions: Arc<SessionRegistry>,
}

/// Per-connection session state: a driver per logical session plus a
/// `ReqId → SessionId` index so `cancel(req_id)` can reach the right driver.
#[derive(Default)]
struct SessionRegistry {
    sessions: Mutex<HashMap<SessionId, SessionHandle>>,
    active: Mutex<HashMap<ReqId, SessionId>>,
}

/// Handle to one logical session's driver task.
struct SessionHandle {
    /// Hands turns to the driver (it feeds the runtime + pumps tokens back).
    turns: mpsc::Sender<Request>,
    /// One abort signal PER live turn. A session runs its turns concurrently,
    /// so a session-wide `Notify` could only cancel an arbitrary one of them.
    cancels: Arc<Mutex<HashMap<ReqId, Arc<Notify>>>>,
}

impl WorkerControl for WorkerControlServer {
    async fn dispatch(self, _: tarpc::context::Context, req: Request) -> Accepted {
        match self.admit(req).await {
            Ok(()) => Accepted::Ok {
                worker: self.worker_id,
            },
            Err(e) => {
                tracing::warn!(error = %e, "dispatch rejected (setup failed)");
                Accepted::Reject
            }
        }
    }

    async fn cancel(self, _: tarpc::context::Context, req_id: ReqId) {
        let session = self.sessions.active.lock().await.get(&req_id).copied();
        if let Some(session) = session
            && let Some(handle) = self.sessions.sessions.lock().await.get(&session)
        {
            let notify = handle.cancels.lock().await.get(&req_id).cloned();
            if let Some(notify) = notify {
                // `notify_one`, not `notify_waiters`: only the former stores a
                // permit when nobody is parked yet. `run_turn` has a waiter
                // registered only while it sits in its `select!` -- not while
                // it is between `cancels.insert` and its first poll, and not
                // during `push_tokens`, which is the backpressure path and may
                // block for the whole 300 s `PUSH_DEADLINE`. A cancel arriving
                // in either window would simply vanish.
                notify.notify_one();
                tracing::debug!(%req_id, %session, "reverse cancel signalled");
            }
        }
    }

    async fn set_priority(self, _: tarpc::context::Context, req_id: ReqId, p: Priority) {
        // Spec-locked surface; the runtime has no priority hook yet (M5).
        tracing::debug!(%req_id, ?p, "set_priority: no runtime hook (no-op)");
    }

    async fn drain(self, _: tarpc::context::Context) {
        // Spec-locked surface; the runtime has no drain hook yet (M5).
        tracing::info!("drain: no runtime hook (best-effort no-op)");
    }
}

impl WorkerControlServer {
    /// Worker-final-admission + turn hand-off. Fetches/verifies any blobs,
    /// ensures the logical session's runtime broker + driver exist, then queues
    /// the turn. Errors map to `Accepted::Reject` (the gateway re-routes).
    async fn admit(&self, req: Request) -> Result<()> {
        // Blob bytes ride out-of-band over HTTP (design §9); fetch + verify here.
        // Feeding them into the runtime needs a runtime image API — a tracked
        // follow-on, so for now we verify integrity and log.
        for blob in &req.blobs {
            let bytes = super::blob::fetch(blob).await?;
            tracing::debug!(
                hash = %blob.hash,
                bytes = bytes.len(),
                "blob fetched + verified (runtime-consume pending)"
            );
        }

        let turns = self.session_turns(req.session).await?;
        turns
            .send(req)
            .await
            .map_err(|_| anyhow!("session driver gone"))?;
        Ok(())
    }

    /// The turn sender for `session`, opening the runtime broker session +
    /// spawning its driver on first use (warm KV anchor across the session's
    /// turns).
    async fn session_turns(&self, session: SessionId) -> Result<mpsc::Sender<Request>> {
        let mut map = self.sessions.sessions.lock().await;
        if let Some(handle) = map.get(&session)
            && !handle.turns.is_closed()
        {
            return Ok(handle.turns.clone());
        }
        let client_id =
            pie_engine::server::open_session().map_err(|e| anyhow!("open session: {e}"))?;
        let (turns_tx, turns_rx) = mpsc::channel::<Request>(TURN_QUEUE_DEPTH);
        let cancels: Arc<Mutex<HashMap<ReqId, Arc<Notify>>>> = Arc::default();
        tokio::spawn(session_driver(
            session,
            client_id,
            self.gateway.clone(),
            turns_rx,
            cancels.clone(),
            Arc::downgrade(&self.sessions),
        ));
        map.insert(
            session,
            SessionHandle {
                turns: turns_tx.clone(),
                cancels,
            },
        );
        Ok(turns_tx)
    }
}

/// Outcome of running one turn, deciding whether the session continues.
enum TurnEnd {
    /// Clean `Eos` sent (or the turn produced nothing to stream).
    Done,
    /// Aborted (reverse `cancel` or piggybacked `Control::Abort`); session alive.
    Aborted,
    /// The gateway link died (push transport error); tear the session down.
    LinkGone,
}

/// Routing table for one session's live turns. The runtime hands every message
/// for a session to ONE mailbox (`recv_messages(client_id, ..)`), so with
/// concurrent turns something has to say which turn each message belongs to.
///
/// Draining that mailbox from the turns themselves cannot work: whichever turn
/// polled first would steal the others' messages and push them to the gateway
/// under its own `req_id`. A single router owns the drain instead and fans out
/// by identity — `corr_id` for a reply, `process_id` for a process event.
#[derive(Default)]
struct TurnRoutes {
    /// Turn awaiting the reply to this `corr_id`.
    by_corr: HashMap<u32, ReqId>,
    /// Turn owning this process's event stream.
    by_pid: HashMap<String, ReqId>,
    /// Where a routed message is delivered.
    inboxes: HashMap<ReqId, mpsc::Sender<ServerMessage>>,
    /// Launch turns that have not learned their `process_id` yet, by `corr_id`.
    awaiting_pid: HashSet<u32>,
}

/// How many routed messages may queue for one turn before the router stops
/// draining the runtime.
///
/// The mailbox the router drains is bounded, and design §6 leans on that: a
/// slow `push_tokens` stalls the pump and backpressures generation. Unbounded
/// per-turn queues would have deleted that invariant, letting the worker buffer
/// a whole generation in memory against a stalled consumer. Deep enough that a
/// turn briefly behind its own pushes does not stall its siblings, shallow
/// enough that a genuinely stuck one still pushes back.
///
/// A full inbox stalls the router and so the whole session, siblings included.
/// That is deliberate, not a limitation to route around: a session's lifetime
/// is ONE websocket connection (`gateway/src/ingress/ws.rs`), so every turn on
/// it egresses through the same socket. For this inbox to fill while a sibling
/// still wants to drain, that shared socket must already be stalled -- which
/// stalls the sibling anyway. The runtime's outbox is bounded per session
/// (`server.rs`, `open_session`) for the same reason: the bound is scoped to
/// the real shared resource. Splitting either per turn would buy no isolation
/// and would oblige the runtime to keep its own `corr_id`/`process_id` routing
/// table -- the exact machinery that made attached processes unroutable here --
/// one layer below where "turn" is even a concept.
const TURN_INBOX_DEPTH: usize = 256;

impl TurnRoutes {
    fn open(
        &mut self,
        req_id: ReqId,
        corr: Option<u32>,
        binding: &ProcBinding,
    ) -> mpsc::Receiver<ServerMessage> {
        let (tx, rx) = mpsc::channel(TURN_INBOX_DEPTH);
        self.inboxes.insert(req_id, tx);
        if let Some(corr) = corr {
            self.by_corr.insert(corr, req_id);
            if matches!(binding, ProcBinding::FromReply) {
                self.awaiting_pid.insert(corr);
            }
        }
        // An attach names its process in the REQUEST. Its reply's `result` is
        // the literal string "Process attached" (`handler.rs`), so learning the
        // id from the reply would key `by_pid` on that text and silently drop
        // every event the attached process emits.
        if let ProcBinding::Known(pid) = binding {
            self.by_pid.insert(pid.clone(), req_id);
        }
        rx
    }

    fn close(&mut self, req_id: ReqId) {
        self.inboxes.remove(&req_id);
        // `awaiting_pid` is retired alongside `by_corr`, not left behind. A
        // launch turn that ends before its `Response` arrives (reverse cancel,
        // `Control::Abort`, `LinkGone`) never reaches the `remove` in `target`
        // -- that path returns early once `by_corr` no longer has the entry --
        // so the set would grow for the session's life, and a later non-launch
        // turn reusing the `corr_id` would take the launch branch and register
        // its arbitrary `result` string as a process id.
        let stale: Vec<u32> = self
            .by_corr
            .iter()
            .filter(|(_, id)| **id == req_id)
            .map(|(corr, _)| *corr)
            .collect();
        for corr in stale {
            self.by_corr.remove(&corr);
            self.awaiting_pid.remove(&corr);
        }
        self.by_pid.retain(|_, id| *id != req_id);
    }

    /// The turn `msg` belongs to, learning a launch's `process_id` on the way.
    ///
    /// A process's events can only follow the launch reply that named it, and
    /// the mailbox is FIFO, so recording the id here — before the next message
    /// is looked at — is enough for every later event to find its turn.
    fn target(&mut self, msg: &ServerMessage) -> Option<ReqId> {
        match msg {
            ServerMessage::Response { corr_id, result, .. } => {
                let req_id = self.by_corr.get(corr_id).copied()?;
                if self.awaiting_pid.remove(corr_id) {
                    self.by_pid.insert(result.clone(), req_id);
                }
                Some(req_id)
            }
            ServerMessage::ProcessEvent { process_id, .. }
            | ServerMessage::File { process_id, .. } => self.by_pid.get(process_id).copied(),
        }
    }
}

/// One logical session's driver: owns the runtime session and runs its turns
/// CONCURRENTLY. Each turn feeds the runtime and streams its own messages back
/// to the gateway under its own `req_id`; a session-wide router decides which
/// turn each runtime message belongs to.
///
/// Turns used to run strictly one at a time, and a process-launching turn does
/// not end until its process does — so a client that launched two processes on
/// one connection could never have them resident together, and the second
/// launch sat in this queue behind the first. Concurrency here is what lets one
/// session put more than one lane in front of the scheduler.
///
/// Exits (closing the runtime session) when the connection's server drops the
/// turn sender (link gone) or a push fails.
///
/// Holds the registry by [`Weak`] so an idle driver never keeps the registry
/// (and thus its own turn-sender) alive: when the connection's server drops,
/// the registry — and the sender in it — drop, unblocking `turns.recv()`.
async fn session_driver(
    session: SessionId,
    client_id: ClientId,
    gateway: GatewayInboundClient,
    mut turns: mpsc::Receiver<Request>,
    cancels: Arc<Mutex<HashMap<ReqId, Arc<Notify>>>>,
    registry: Weak<SessionRegistry>,
) {
    let routes: Arc<Mutex<TurnRoutes>> = Arc::default();
    let (link_gone_tx, mut link_gone) = mpsc::unbounded_channel::<()>();
    let router = tokio::spawn(message_router(client_id, routes.clone()));
    let mut running = tokio::task::JoinSet::new();

    loop {
        let req = tokio::select! {
            req = turns.recv() => match req {
                Some(req) => req,
                None => break,
            },
            _ = link_gone.recv() => {
                tracing::debug!(%session, "gateway link gone; ending session");
                break;
            }
        };
        let req_id = req.req_id;
        let corr = corr_id_of(&req.message);
        let binding = ProcBinding::of(&req.message);
        let inbox = routes.lock().await.open(req_id, corr, &binding);

        let cancel = Arc::new(Notify::new());
        cancels.lock().await.insert(req_id, cancel.clone());
        if let Some(reg) = registry.upgrade() {
            reg.active.lock().await.insert(req_id, session);
        }

        // The runtime is fed HERE, in the driver loop, and not from the spawned
        // task. Chunked `AddProgram` uploads arrive as one turn per chunk, and
        // `runtime/engine/src/server/data_transfer.rs` hard-rejects a chunk
        // that is not the one it expects next, tearing the upload down. Feeding
        // from the tasks would have made chunk order depend on `JoinSet::spawn`
        // first-poll order across worker threads, which tokio's LIFO slot and
        // work stealing actively reorder. Dequeue order is the only order there
        // is, so the feed has to happen in the dequeue.
        let fed = feed_turn(client_id, req_id, req.message);

        let gateway = gateway.clone();
        let routes = routes.clone();
        let cancels = cancels.clone();
        let registry = registry.clone();
        let link_gone_tx = link_gone_tx.clone();
        running.spawn(async move {
            let outcome =
                run_turn(client_id, &gateway, &cancel, req_id, fed, corr, binding, inbox).await;
            routes.lock().await.close(req_id);
            cancels.lock().await.remove(&req_id);
            if let Some(reg) = registry.upgrade() {
                reg.active.lock().await.remove(&req_id);
            }
            if let TurnEnd::LinkGone = outcome {
                let _ = link_gone_tx.send(());
            }
        });

        // Reap finished turns so the set does not grow for the session's life.
        while running.try_join_next().is_some() {}
    }

    router.abort();
    running.shutdown().await;
    pie_engine::server::close_session(client_id);
    // Best-effort removal; if the registry is already gone (the connection's
    // server dropped) the stale entry died with it.
    if let Some(reg) = registry.upgrade() {
        reg.sessions.lock().await.remove(&session);
    }
    tracing::debug!(%session, "session driver exited");
}

/// Drain the session's single runtime mailbox and fan each message out to the
/// turn that owns it. Runs until aborted by `session_driver`.
async fn message_router(client_id: ClientId, routes: Arc<Mutex<TurnRoutes>>) {
    loop {
        let msgs = match pie_engine::server::recv_messages(client_id, 200, 64).await {
            Ok(msgs) => msgs,
            Err(e) => {
                tracing::warn!(error = %e, "runtime recv failed; router stopping");
                return;
            }
        };
        for msg in msgs {
            // The sender is cloned out and the lock released BEFORE awaiting
            // the send: the inbox is bounded now, so this await can block, and
            // blocking under the lock would stop the turn tasks that need it to
            // retire their routes.
            let inbox = {
                let mut routes = routes.lock().await;
                match routes.target(&msg) {
                    Some(req_id) => routes.inboxes.get(&req_id).cloned(),
                    // A message whose turn already ended (a late event on a
                    // cancelled turn, say) has nowhere to go. Dropping it is the
                    // only option, but it is worth seeing.
                    None => {
                        tracing::debug!(?msg, "runtime message matched no live turn");
                        None
                    }
                }
            };
            if let Some(inbox) = inbox {
                // Awaited, not `try_send`: a full inbox must stall this drain,
                // which stalls the bounded runtime outbox, which backpressures
                // generation (design §6). That chain is the whole mechanism.
                let _ = inbox.send(msg).await;
            }
        }
    }
}

/// Push one turn's message into the runtime, in the caller's order.
///
/// Split out of `run_turn` because ordering across turns is only guaranteed
/// where the turns are dequeued; see the call site.
enum Fed {
    /// Accepted, and the turn now streams the runtime's replies.
    Streaming,
    /// Nothing will come back for this turn: a non-final upload chunk (accepted
    /// as `InProgress` with no reply), a `corr_id`-less signal the runtime never
    /// answers, or a feed that failed. Close the stream immediately -- waiting
    /// for a terminal message that can never arrive would leave the turn open
    /// forever.
    Silent,
}

fn feed_turn(client_id: ClientId, req_id: ReqId, message: ClientMessage) -> Fed {
    let non_final_chunk = matches!(upload_chunk_info(&message), Some((idx, total)) if idx + 1 < total);
    let expects_reply = corr_id_of(&message).is_some();
    if let Err(e) = pie_engine::server::send_client_message(client_id, message) {
        tracing::warn!(%req_id, error = %e, "feeding turn into runtime failed");
        return Fed::Silent;
    }
    if non_final_chunk || !expects_reply {
        Fed::Silent
    } else {
        Fed::Streaming
    }
}

/// Stream one already-fed turn's messages back as `Tokens`, terminated by
/// exactly one `Eos`. Selects against `cancel` so a reverse `cancel` aborts
/// mid-turn even while the worker is between pushes.
async fn run_turn(
    client_id: ClientId,
    gateway: &GatewayInboundClient,
    cancel: &Notify,
    req_id: ReqId,
    fed: Fed,
    corr: Option<u32>,
    binding: ProcBinding,
    mut inbox: mpsc::Receiver<ServerMessage>,
) -> TurnEnd {
    let proc_launch = binding.is_process();
    // An attach already knows its process; a launch learns it from the reply.
    let mut process_id: Option<String> = match binding {
        ProcBinding::Known(pid) => Some(pid),
        _ => None,
    };

    if let Fed::Silent = fed {
        return push_eos(gateway, req_id).await;
    }

    loop {
        tokio::select! {
            _ = cancel.notified() => {
                tracing::debug!(%req_id, "turn cancelled");
                if let Some(pid) = &process_id {
                    let _ = pie_engine::server::send_client_message(client_id, terminate(pid));
                }
                // Abort = bare channel-close on the gateway side (no Eos), per
                // the Tokens contract; the gateway's TokenRx observes the close.
                return TurnEnd::Aborted;
            }
            msg = inbox.recv() => {
                let Some(msg) = msg else {
                    // The router closed this turn's inbox: the session is
                    // going away.
                    return TurnEnd::Aborted;
                };
                let terminal = turn_terminal(&msg, corr, proc_launch, &mut process_id);
                match gateway.push_tokens(push_ctx(), req_id, Tokens::Chunk(msg)).await {
                    Ok(Control::Continue) => {}
                    Ok(Control::Abort) => {
                        tracing::debug!(%req_id, "gateway piggybacked abort");
                        if let Some(pid) = &process_id {
                            let _ = pie_engine::server::send_client_message(client_id, terminate(pid));
                        }
                        return TurnEnd::Aborted;
                    }
                    Err(e) => {
                        tracing::warn!(%req_id, error = %e, "push_tokens transport error");
                        return TurnEnd::LinkGone;
                    }
                }
                if terminal {
                    return push_eos(gateway, req_id).await;
                }
            }
        }
    }
}

/// Send the clean end-of-turn marker. A transport error here is the link dying.
async fn push_eos(gateway: &GatewayInboundClient, req_id: ReqId) -> TurnEnd {
    match gateway.push_tokens(push_ctx(), req_id, Tokens::Eos).await {
        Ok(_) => TurnEnd::Done,
        Err(e) => {
            tracing::warn!(%req_id, error = %e, "push Eos transport error");
            TurnEnd::LinkGone
        }
    }
}

/// A `push_tokens` context with the generous backpressure deadline.
fn push_ctx() -> tarpc::context::Context {
    let mut ctx = tarpc::context::current();
    ctx.deadline = std::time::Instant::now() + PUSH_DEADLINE;
    ctx
}

/// Whether `msg` ends the current turn, learning the launched `process_id` from
/// the launch ack along the way.
///
/// - A process-launching turn's first matching `Response{corr}` carries the
///   `process_id` as its `result`; the turn then runs until that process emits a
///   terminal `ProcessEvent` (`event == "return" | "error"`).
/// - A non-process command's single matching `Response{corr}` is itself terminal.
fn turn_terminal(
    msg: &ServerMessage,
    corr: Option<u32>,
    proc_launch: bool,
    process_id: &mut Option<String>,
) -> bool {
    match msg {
        ServerMessage::Response {
            corr_id, result, ..
        } if Some(*corr_id) == corr => {
            if proc_launch {
                if process_id.is_none() {
                    *process_id = Some(result.clone());
                }
                false
            } else {
                true
            }
        }
        ServerMessage::ProcessEvent {
            process_id: pid,
            event,
            ..
        } => process_id.as_deref() == Some(pid.as_str()) && (event == "return" || event == "error"),
        _ => false,
    }
}

/// A `TerminateProcess` message to stop a running process (reverse-cancel path).
fn terminate(process_id: &str) -> ClientMessage {
    ClientMessage::TerminateProcess {
        corr_id: 0,
        process_id: process_id.to_string(),
    }
}

/// The correlation id a client message carries, if any (process/file signals
/// have none).
fn corr_id_of(m: &ClientMessage) -> Option<u32> {
    use ClientMessage::*;
    match m {
        AuthIdentify { corr_id, .. }
        | AuthProve { corr_id, .. }
        | CheckProgram { corr_id, .. }
        | Query { corr_id, .. }
        | AddProgram { corr_id, .. }
        | LaunchProcess { corr_id, .. }
        | AttachProcess { corr_id, .. }
        | TerminateProcess { corr_id, .. }
        | ListProcesses { corr_id }
        | Ping { corr_id } => Some(*corr_id),
        SignalProcess { .. } | TransferFile { .. } => None,
    }
}

/// Whether a turn launches/attaches a process (so its output streams as process
/// events terminated by `return`/`error`, not a single `Response`).
/// How a turn comes to own a process's event stream.
///
/// The two process-bearing messages name their process at opposite ends:
/// `LaunchProcess` learns the id from its reply, `AttachProcess` states it in
/// the request and gets back a fixed "Process attached" string. Treating them
/// alike keyed the routing table on that string.
enum ProcBinding {
    /// Not a process turn.
    None,
    /// `LaunchProcess`: the id arrives as the reply's `result`.
    FromReply,
    /// `AttachProcess`: the id is already known.
    Known(String),
}

impl ProcBinding {
    fn of(m: &ClientMessage) -> Self {
        match m {
            ClientMessage::LaunchProcess { .. } => Self::FromReply,
            ClientMessage::AttachProcess { process_id, .. } => Self::Known(process_id.clone()),
            _ => Self::None,
        }
    }

    /// Whether the turn streams a process's events (rather than ending at its
    /// own reply).
    fn is_process(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// `(chunk_index, total_chunks)` for a chunked upload message, else `None`.
/// `AddProgram` is delivered as `total_chunks` messages sharing one `corr_id`;
/// only the final chunk produces a `Response`, so the bridge must not await a
/// reply for the earlier ones (see `run_turn`).
fn upload_chunk_info(m: &ClientMessage) -> Option<(usize, usize)> {
    match m {
        ClientMessage::AddProgram {
            chunk_index,
            total_chunks,
            ..
        } => Some((*chunk_index, *total_chunks)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(corr_id: u32, result: &str) -> ServerMessage {
        ServerMessage::Response {
            corr_id,
            ok: true,
            result: result.to_string(),
        }
    }

    fn event(process_id: &str, event: &str) -> ServerMessage {
        ServerMessage::ProcessEvent {
            process_id: process_id.to_string(),
            event: event.to_string(),
            value: String::new(),
        }
    }

    /// The regression this whole seam exists for: two processes launched on ONE
    /// session must each get their OWN events. Every message used to be pumped
    /// by whichever turn happened to poll the shared mailbox first, so the older
    /// launch lost its `return` and its client waited forever.
    #[test]
    fn two_concurrent_launches_each_receive_their_own_events() {
        let (a, b) = (ReqId(1), ReqId(2));
        let mut routes = TurnRoutes::default();
        let _a_inbox = routes.open(a, Some(10), &ProcBinding::FromReply);
        let _b_inbox = routes.open(b, Some(11), &ProcBinding::FromReply);

        // Each launch reply names its process; the router learns the mapping.
        assert_eq!(routes.target(&response(10, "pid-a")), Some(a));
        assert_eq!(routes.target(&response(11, "pid-b")), Some(b));

        // Events then route by process, in any interleaving.
        assert_eq!(routes.target(&event("pid-b", "token")), Some(b));
        assert_eq!(routes.target(&event("pid-a", "token")), Some(a));
        assert_eq!(routes.target(&event("pid-a", "return")), Some(a));
        assert_eq!(routes.target(&event("pid-b", "return")), Some(b));
    }

    /// A finished turn releases both of its routing keys, so a later message
    /// naming it is reported as unroutable instead of being handed to whatever
    /// turn reused the id.
    #[test]
    fn closing_a_turn_retires_its_routes() {
        let a = ReqId(1);
        let mut routes = TurnRoutes::default();
        let _inbox = routes.open(a, Some(10), &ProcBinding::FromReply);
        assert_eq!(routes.target(&response(10, "pid-a")), Some(a));

        routes.close(a);
        assert_eq!(routes.target(&event("pid-a", "return")), None);
        assert_eq!(routes.target(&response(10, "pid-a")), None);
    }

    /// A non-launch turn owns only its reply — it must never capture the
    /// `result` string as a process id and start stealing another turn's
    /// events.
    #[test]
    fn a_plain_command_turn_claims_no_process() {
        let (query, launch) = (ReqId(1), ReqId(2));
        let mut routes = TurnRoutes::default();
        let _q = routes.open(query, Some(10), &ProcBinding::None);
        let _l = routes.open(launch, Some(11), &ProcBinding::FromReply);

        assert_eq!(routes.target(&response(10, "pid-a")), Some(query));
        assert_eq!(routes.target(&response(11, "pid-a")), Some(launch));
        assert_eq!(routes.target(&event("pid-a", "return")), Some(launch));
    }

    /// An attach names its process in the REQUEST and gets back the literal
    /// string "Process attached" as its `result`. Learning the id from that
    /// reply would key the table on the message text and silently drop every
    /// event the attached process emits — the turn would then block forever.
    #[test]
    fn an_attached_process_routes_by_its_requested_id() {
        let a = ReqId(1);
        let mut routes = TurnRoutes::default();
        let _inbox = routes.open(a, Some(10), &ProcBinding::Known("pid-a".into()));

        // Events route before the reply is even seen.
        assert_eq!(routes.target(&event("pid-a", "stdout")), Some(a));
        assert_eq!(routes.target(&response(10, "Process attached")), Some(a));
        assert_eq!(routes.target(&event("pid-a", "return")), Some(a));
        // The reply text never becomes a process id.
        assert_eq!(routes.target(&event("Process attached", "return")), None);
    }

    /// A launch that ends before its `Response` (reverse cancel, abort, link
    /// gone) must not leave its `corr_id` in `awaiting_pid`: a later turn
    /// reusing that id would take the launch branch and register whatever its
    /// reply happened to say as a process id.
    #[test]
    fn closing_a_turn_retires_a_pending_launch() {
        let (launch, query) = (ReqId(1), ReqId(2));
        let mut routes = TurnRoutes::default();
        let _l = routes.open(launch, Some(10), &ProcBinding::FromReply);
        routes.close(launch);
        assert!(routes.awaiting_pid.is_empty());

        // Same `corr_id`, reused by a plain command turn.
        let _q = routes.open(query, Some(10), &ProcBinding::None);
        assert_eq!(routes.target(&response(10, "some-result")), Some(query));
        assert_eq!(routes.target(&event("some-result", "return")), None);
    }
}
