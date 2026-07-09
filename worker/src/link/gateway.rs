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
use pie_engine::server::ClientId;
use pie_client_api::{ClientMessage, ServerMessage};
use pie_controller_rpc::GatewayEndpoint;
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
    /// Notified to abort the in-flight turn (reverse `cancel`).
    cancel: Arc<Notify>,
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
            handle.cancel.notify_one();
            tracing::debug!(%req_id, %session, "reverse cancel signalled");
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
        if std::env::var_os("PIE_BRIDGE_TRACE").is_some() {
            eprintln!(
                "[bridge] admit req_id={} session={} blobs={}",
                req.req_id,
                req.session,
                req.blobs.len()
            );
        }
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
        let client_id = pie_engine::server::open_session().map_err(|e| anyhow!("open session: {e}"))?;
        let (turns_tx, turns_rx) = mpsc::channel::<Request>(TURN_QUEUE_DEPTH);
        let cancel = Arc::new(Notify::new());
        tokio::spawn(session_driver(
            session,
            client_id,
            self.gateway.clone(),
            turns_rx,
            cancel.clone(),
            Arc::downgrade(&self.sessions),
        ));
        map.insert(
            session,
            SessionHandle {
                turns: turns_tx.clone(),
                cancel,
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

/// One logical session's driver: owns the runtime session and serializes its
/// turns. Each turn feeds the runtime then pumps `ServerMessage`s back to the
/// gateway until the turn terminates. Exits (closing the runtime session) when
/// the connection's server drops the turn sender (link gone) or a push fails.
///
/// Holds the registry by [`Weak`] so an idle driver never keeps the registry
/// (and thus its own turn-sender) alive: when the connection's server drops,
/// the registry — and the sender in it — drop, unblocking `turns.recv()`.
async fn session_driver(
    session: SessionId,
    client_id: ClientId,
    gateway: GatewayInboundClient,
    mut turns: mpsc::Receiver<Request>,
    cancel: Arc<Notify>,
    registry: Weak<SessionRegistry>,
) {
    while let Some(req) = turns.recv().await {
        let req_id = req.req_id;
        if let Some(reg) = registry.upgrade() {
            reg.active.lock().await.insert(req_id, session);
        }
        let outcome = run_turn(client_id, &gateway, &cancel, req).await;
        if let Some(reg) = registry.upgrade() {
            reg.active.lock().await.remove(&req_id);
        }
        if let TurnEnd::LinkGone = outcome {
            tracing::debug!(%session, "gateway link gone; ending session");
            break;
        }
    }
    pie_engine::server::close_session(client_id);
    // Best-effort removal; if the registry is already gone (the connection's
    // server dropped) the stale entry died with it.
    if let Some(reg) = registry.upgrade() {
        reg.sessions.lock().await.remove(&session);
    }
    tracing::debug!(%session, "session driver exited");
}

/// Feed one turn into the runtime and stream its output back as `Tokens`,
/// terminated by exactly one `Eos`. Selects against `cancel` so a reverse
/// `cancel` aborts mid-turn even while the worker is between pushes.
async fn run_turn(
    client_id: ClientId,
    gateway: &GatewayInboundClient,
    cancel: &Notify,
    req: Request,
) -> TurnEnd {
    let req_id = req.req_id;
    let corr = corr_id_of(&req.message);
    let proc_launch = is_process_launch(&req.message);
    let mut process_id: Option<String> = None;

    let bridge_trace = std::env::var_os("PIE_BRIDGE_TRACE").is_some();
    if bridge_trace {
        eprintln!(
            "[bridge] run_turn entry req_id={req_id} corr={corr:?} proc_launch={proc_launch} chunk={:?}",
            upload_chunk_info(&req.message)
        );
    }

    // Chunked uploads (`AddProgram`) span many messages under one `corr_id`,
    // but only the FINAL chunk yields a `Response` — every intermediate chunk
    // is accepted as `InProgress` with no reply. Since `session_driver` runs
    // turns strictly sequentially, awaiting a response for a non-final chunk
    // would block forever (the next chunk never dequeues, so the upload never
    // completes and the client's `add_program` hangs). Feed a non-final chunk
    // into the runtime and close its stream immediately; the final chunk takes
    // the normal request/response turn that surfaces the install result.
    if let Some((idx, total)) = upload_chunk_info(&req.message) {
        if idx + 1 < total {
            if let Err(e) = pie_engine::server::send_client_message(client_id, req.message) {
                tracing::warn!(%req_id, error = %e, "feeding upload chunk into runtime failed");
            }
            return push_eos(gateway, req_id).await;
        }
    }

    if let Err(e) = pie_engine::server::send_client_message(client_id, req.message) {
        tracing::warn!(%req_id, error = %e, "feeding turn into runtime failed");
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
            recv = pie_engine::server::recv_messages(client_id, 200, 64) => {
                let msgs = match recv {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(%req_id, error = %e, "runtime recv failed; aborting turn");
                        return TurnEnd::Aborted;
                    }
                };
                if bridge_trace && !msgs.is_empty() {
                    eprintln!("[bridge] req_id={req_id} drained {} runtime msg(s)", msgs.len());
                }
                for msg in msgs {
                    let terminal = turn_terminal(&msg, corr, proc_launch, &mut process_id);
                    if bridge_trace {
                        eprintln!("[bridge] req_id={req_id} push msg terminal={terminal}");
                    }
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
        | AuthByToken { corr_id, .. }
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
fn is_process_launch(m: &ClientMessage) -> bool {
    matches!(
        m,
        ClientMessage::LaunchProcess { .. } | ClientMessage::AttachProcess { .. }
    )
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
