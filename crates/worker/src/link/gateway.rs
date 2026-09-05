//! Worker data-plane: the worker dials into the gateway (1:N fan-in),
//! serving `WorkerControl` while pushing tokens back over the same
//! connection.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use client_api::{ClientMessage, ServerMessage};
use controller_api::GatewayEndpoint;
use futures::StreamExt;
use ids::{ReqId, SessionId, WorkerId};
use runtime::server::ClientId;
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel};
use tokio::sync::{Mutex, Notify, mpsc};
use worker_api::{
    Accepted, Control, GatewayInboundClient, Priority, Request, Tokens, WorkerControl,
    connect_gateway_link, dispatch_codec,
};

/// Max read-side frame size; generous since a `dispatch` can carry a large
/// upload chunk. The gateway sets its own (smaller) cap for the reverse
/// direction independently.
const LINK_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// `push_tokens` deadline, generous so a backpressured push blocks rather
/// than errors; a real transport error still surfaces immediately.
const PUSH_DEADLINE: Duration = Duration::from_secs(300);

/// Per-session driver mailbox depth: queued turns awaiting the in-flight one.
const TURN_QUEUE_DEPTH: usize = 64;

/// A live dial-in connection to one gateway: the task serving `WorkerControl`.
pub struct GatewayLink {
    serve_task: tokio::task::JoinHandle<()>,
}

/// Aborting the serve task tears the connection down when the link is
/// dropped.
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

    // Register before serving, so the gateway's registry has this worker
    // before any dispatch.
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

/// The worker's live dial-in links, reconciled against the gateway roster.
/// Owns one [`GatewayLink`] per dialed gateway, keyed by dial address.
///
/// `pinned` addresses (the static `--gateway` override) are always kept
/// dialed regardless of the roster.
pub struct GatewayLinkManager {
    worker_id: WorkerId,
    pinned: HashSet<String>,
    links: HashMap<String, GatewayLink>,
}

impl GatewayLinkManager {
    /// A manager for `worker_id`, pinning `pinned` addresses. Nothing is
    /// dialed yet. Addresses are canonicalized so a pinned `tcp://h:p` and a
    /// roster `h:p` for the same gateway resolve to a single link.
    pub fn new(worker_id: WorkerId, pinned: Vec<String>) -> Self {
        Self {
            worker_id,
            pinned: pinned.iter().map(|a| canonical_addr(a)).collect(),
            links: HashMap::new(),
        }
    }

    /// Dial every pinned address now, failing if any pinned dial fails (a
    /// hard boot requirement). Roster dials are best-effort and never fail
    /// boot.
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

    /// Reconcile dial-in links against a fresh roster: desired = `pinned ∪
    /// roster`. Drops undesired links, dials newly-desired ones. A failed
    /// roster dial is logged and retried on the next update.
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

    /// Drop links whose serve task has ended, so the next [`reconcile`] dials
    /// them again. Returns the addresses reaped.
    ///
    /// A link dies when its transport does — a codec error on an oversized
    /// frame, a gateway restart, a dropped connection. Nothing used to remove
    /// it: the entry stayed in `links`, `reconcile`'s dial step skips every
    /// address already there, and the worker sat holding a dead handle while
    /// every dispatch timed out. In a standalone deployment, where the roster
    /// never changes and `reconcile` is never called again, that was permanent.
    ///
    /// [`reconcile`]: Self::reconcile
    pub fn reap_dead(&mut self) -> Vec<String> {
        let dead: Vec<String> = self
            .links
            .iter()
            .filter(|(_, link)| link.serve_task.is_finished())
            .map(|(addr, _)| addr.clone())
            .collect();
        for addr in &dead {
            self.links.remove(addr);
            tracing::warn!(
                worker = %self.worker_id,
                gateway = %addr,
                "gateway link died; dropping it so it can be re-dialed"
            );
        }
        dead
    }

    /// The addresses currently linked — the advertised URL / boot banner.
    pub fn addrs(&self) -> Vec<String> {
        let mut addrs: Vec<String> = self.links.keys().cloned().collect();
        addrs.sort();
        addrs
    }
}

/// Canonicalizes a dial address so pinned and roster forms of the same
/// gateway map to one link key. Strips the `tcp://` scheme, leaves `unix:`
/// addresses intact.
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
    /// Push side back to this gateway, cloned into each session driver.
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
    /// One cancel signal per live turn (turns run concurrently).
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
                // `notify_one`, not `notify_waiters`: stores a permit so a
                // cancel racing before `run_turn` reaches its `select!` isn't
                // lost.
                notify.notify_one();
                tracing::debug!(%req_id, %session, "reverse cancel signalled");
            }
        }
    }

    async fn set_priority(self, _: tarpc::context::Context, req_id: ReqId, p: Priority) {
        // Spec-locked surface; the runtime has no priority hook.
        tracing::debug!(%req_id, ?p, "set_priority: no runtime hook (no-op)");
    }

    async fn drain(self, _: tarpc::context::Context) {
        // Spec-locked surface; the runtime has no drain hook.
        tracing::info!("drain: no runtime hook (best-effort no-op)");
    }
}

impl WorkerControlServer {
    /// Worker-final-admission + turn hand-off: verifies blobs, ensures the
    /// session driver exists, then queues the turn. Errors map to
    /// `Accepted::Reject`.
    async fn admit(&self, req: Request) -> Result<()> {
        // Blobs ride out-of-band over HTTP; fetched + verified here
        // (runtime-consume pending).
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

    /// Turn sender for `session`; opens the runtime session and spawns its
    /// driver on first use.
    async fn session_turns(&self, session: SessionId) -> Result<mpsc::Sender<Request>> {
        let mut map = self.sessions.sessions.lock().await;
        if let Some(handle) = map.get(&session)
            && !handle.turns.is_closed()
        {
            return Ok(handle.turns.clone());
        }
        let client_id =
            runtime::server::open_session().map_err(|e| anyhow!("open session: {e}"))?;
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

/// Routing table for one session's live turns. The runtime delivers all
/// messages to one mailbox, so this fans them out by identity (`corr_id` for
/// a reply, `process_id` for a process event).
///
/// A single router owns the drain rather than the turns themselves, since
/// concurrent polling would let one turn steal another's messages.
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

/// Queue depth for one turn's routed messages before the router stalls.
///
/// A full inbox stalls the whole session (all turns share one websocket
/// egress), which is deliberate: it's the backpressure mechanism limiting
/// in-memory generation buffering.
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
        // An attach names its process in the request; its reply is a fixed
        // "Process attached" string, so only launches learn the id from the
        // reply.
        if let ProcBinding::Known(pid) = binding {
            self.by_pid.insert(pid.clone(), req_id);
        }
        rx
    }

    fn close(&mut self, req_id: ReqId) {
        self.inboxes.remove(&req_id);
        // Retire `awaiting_pid` alongside `by_corr`: an ended launch must not
        // leave a stale corr_id for a later turn to inherit as its process id.
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

    /// The turn `msg` belongs to; learns a launch's `process_id` from its
    /// reply along the way.
    fn target(&mut self, msg: &ServerMessage) -> Option<ReqId> {
        match msg {
            ServerMessage::Response {
                corr_id, result, ..
            } => {
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

/// One logical session's driver: owns the runtime session and runs turns
/// concurrently, each streaming its own messages back under its own
/// `req_id`.
///
/// Exits when the turn sender is dropped (link gone) or a push fails. Holds
/// the registry by [`Weak`] so an idle driver doesn't keep it (and its own
/// sender) alive.
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

        // Fed here in the driver loop, not the spawned task: chunk order must
        // match dequeue order, since the runtime rejects out-of-order
        // `AddProgram` chunks and task spawn/poll order isn't guaranteed.
        let fed = feed_turn(client_id, req_id, req.message);

        let gateway = gateway.clone();
        let routes = routes.clone();
        let cancels = cancels.clone();
        let registry = registry.clone();
        let link_gone_tx = link_gone_tx.clone();
        running.spawn(async move {
            let outcome = run_turn(
                client_id, &gateway, &cancel, req_id, fed, corr, binding, inbox,
            )
            .await;
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
    runtime::server::close_session(client_id);
    // Best-effort: if the registry is already gone, the entry died with it.
    if let Some(reg) = registry.upgrade() {
        reg.sessions.lock().await.remove(&session);
    }
    tracing::debug!(%session, "session driver exited");
}

/// Drain the session's single runtime mailbox and fan each message out to the
/// turn that owns it. Runs until aborted by `session_driver`.
async fn message_router(client_id: ClientId, routes: Arc<Mutex<TurnRoutes>>) {
    loop {
        let msgs = match runtime::server::recv_messages(client_id, 200, 64).await {
            Ok(msgs) => msgs,
            Err(e) => {
                tracing::warn!(error = %e, "runtime recv failed; router stopping");
                return;
            }
        };
        for msg in msgs {
            // Clone the sender and release the lock before awaiting the send:
            // the inbox is bounded, and blocking under the lock would starve
            // turns retiring their routes.
            let inbox = {
                let mut routes = routes.lock().await;
                match routes.target(&msg) {
                    Some(req_id) => routes.inboxes.get(&req_id).cloned(),
                    // No live turn for this message; drop it, but log.
                    None => {
                        tracing::debug!(?msg, "runtime message matched no live turn");
                        None
                    }
                }
            };
            if let Some(inbox) = inbox {
                // Awaited, not `try_send`: a full inbox stalls this drain,
                // backpressuring generation via the bounded runtime outbox.
                let _ = inbox.send(msg).await;
            }
        }
    }
}

/// Whether feeding a turn into the runtime produced a reply.
enum Fed {
    /// Accepted, and the turn now streams the runtime's replies.
    Streaming,
    /// Nothing will come back: a non-final upload chunk, a corr_id-less
    /// signal, or a failed feed. Close the stream immediately.
    Silent,
}

fn feed_turn(client_id: ClientId, req_id: ReqId, message: ClientMessage) -> Fed {
    let non_final_chunk =
        matches!(upload_chunk_info(&message), Some((idx, total)) if idx + 1 < total);
    let expects_reply = corr_id_of(&message).is_some();
    if let Err(e) = runtime::server::send_client_message(client_id, message) {
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
#[allow(
    clippy::too_many_arguments,
    reason = "one turn's whole context: identity, transport, cancellation and \
              inbox are each owned by a different layer above"
)]
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
                    let _ = runtime::server::send_client_message(client_id, terminate(pid));
                }
                // Abort = channel close with no Eos, per the Tokens contract.
                return TurnEnd::Aborted;
            }
            msg = inbox.recv() => {
                let Some(msg) = msg else {
                    // Inbox closed: session going away.
                    return TurnEnd::Aborted;
                };
                let terminal = turn_terminal(&msg, corr, proc_launch, &mut process_id);
                match gateway.push_tokens(push_ctx(), req_id, Tokens::Chunk(msg)).await {
                    Ok(Control::Continue) => {}
                    Ok(Control::Abort) => {
                        tracing::debug!(%req_id, "gateway piggybacked abort");
                        if let Some(pid) = &process_id {
                            let _ = runtime::server::send_client_message(client_id, terminate(pid));
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

/// Whether `msg` ends the turn, learning a launched `process_id` from the ack
/// along the way.
///
/// A launch's first matching reply carries `process_id` as `result`; the turn
/// then runs until that process emits a terminal event. A non-process reply
/// is itself terminal.
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

/// How a turn comes to own a process's event stream: `LaunchProcess` learns
/// its id from the reply, `AttachProcess` states it in the request.
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

/// `(chunk_index, total_chunks)` for a chunked `AddProgram` upload; only the
/// final chunk gets a `Response`.
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
