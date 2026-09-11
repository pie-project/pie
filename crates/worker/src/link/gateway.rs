use std::collections::{HashMap, HashSet};
#[cfg(feature = "session-lifetime-diagnostic")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use client_api::{ClientMessage, ServerMessage};
use controller_api::GatewayEndpoint;
use futures::StreamExt;
use ids::{ReqId, SessionId, WorkerId};
use runtime::server::ClientId;
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::server::{BaseChannel, Channel};
use tokio::sync::{Mutex, Notify, mpsc};
use worker_api::{
    Accepted, Control, GatewayInboundClient, Priority, Request, Tokens, WorkerControl,
    connect_gateway_link, dispatch_codec,
};

const LINK_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

const PUSH_DEADLINE: Duration = Duration::from_secs(300);

const TURN_QUEUE_DEPTH: usize = 64;

const TURN_DRAIN_LIMIT: Duration = Duration::from_secs(5);

pub struct GatewayLink {
    serve_task: tokio::task::JoinHandle<()>,
    #[cfg(feature = "session-lifetime-diagnostic")]
    sessions: Weak<SessionRegistry>,
}

#[cfg(feature = "session-lifetime-diagnostic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionOwnerKeys {
    pub sessions: Vec<(SessionId, ClientId)>,
    pub active: Vec<(ReqId, SessionId)>,
    pub runtime: runtime::server::SessionOwnerKeys,
}

#[cfg(feature = "session-lifetime-diagnostic")]
#[derive(Clone)]
pub struct SessionObserver {
    sessions: Weak<SessionRegistry>,
}

#[cfg(feature = "session-lifetime-diagnostic")]
impl SessionObserver {
    /// Reads exact real registry keys sequentially at a quiescent checkpoint.
    pub async fn session_owner_keys(&self) -> Option<SessionOwnerKeys> {
        let registry = self.sessions.upgrade()?;
        let mut sessions: Vec<_> = registry
            .sessions
            .lock()
            .await
            .iter()
            .map(|(session, handle)| (*session, handle.client_id))
            .collect();
        let mut active: Vec<_> = registry
            .active
            .lock()
            .await
            .iter()
            .map(|(request, (session, _))| (*request, *session))
            .collect();
        sessions.sort_unstable();
        active.sort_unstable();
        Some(SessionOwnerKeys {
            sessions,
            active,
            runtime: runtime::server::session_owner_keys(),
        })
    }

    /// Hold the next terminal reply after its response push completes but before
    /// Eos, leaving a real active turn for frontend-close cancellation.
    pub fn hold_next_terminal(&self) -> bool {
        self.sessions
            .upgrade()
            .is_some_and(|registry| !registry.diagnostic_hold.armed.swap(true, Ordering::AcqRel))
    }

    pub async fn close_session(&self, session: SessionId) {
        if let Some(registry) = self.sessions.upgrade() {
            registry.close_session(session).await;
        }
    }

    pub async fn held_terminal(&self) -> Option<ReqId> {
        let registry = self.sessions.upgrade()?;
        loop {
            if let Some(req_id) = *registry.diagnostic_hold.held.lock().await {
                return Some(req_id);
            }
            registry.diagnostic_hold.reached.notified().await;
        }
    }
}

#[cfg(feature = "session-lifetime-diagnostic")]
impl GatewayLink {
    pub fn session_observer(&self) -> SessionObserver {
        SessionObserver {
            sessions: self.sessions.clone(),
        }
    }
}

impl Drop for GatewayLink {
    fn drop(&mut self) {
        self.serve_task.abort();
    }
}

pub async fn connect_gateway(addr: &str, worker_id: WorkerId) -> Result<GatewayLink> {
    let (server_half, gateway) = if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        #[cfg(unix)]
        {
            let mut conn = unix::connect(path, dispatch_codec);
            conn.config_mut().max_frame_length(LINK_MAX_FRAME_BYTES);
            let transport = conn
                .await
                .with_context(|| format!("dialing gateway at {addr}"))?;
            connect_gateway_link(transport)
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            anyhow::bail!(
                "{addr}: a `unix://` gateway address is distributed serving, which needs a unix-domain socket; this build is single-node and speaks `tcp://`"
            )
        }
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let mut conn = tcp::connect(tcp_addr, dispatch_codec);
        conn.config_mut().max_frame_length(LINK_MAX_FRAME_BYTES);
        let transport = conn
            .await
            .with_context(|| format!("dialing gateway at {addr}"))?;
        let _ = transport.get_ref().set_nodelay(true);
        connect_gateway_link(transport)
    };

    gateway
        .register(tarpc::context::current(), worker_id)
        .await
        .with_context(|| format!("registering worker with gateway at {addr}"))?;
    tracing::info!(%worker_id, gateway = %addr, "worker registered with gateway (dial-in)");

    let sessions = Arc::new(SessionRegistry::default());
    let server = WorkerControlServer {
        worker_id,
        gateway,
        sessions: sessions.clone(),
    };
    let serve_task = tokio::spawn(
        BaseChannel::with_defaults(server_half)
            .execute(server.serve())
            .for_each_concurrent(None, |req| async move {
                tokio::spawn(req);
            }),
    );

    Ok(GatewayLink {
        serve_task,
        #[cfg(feature = "session-lifetime-diagnostic")]
        sessions: Arc::downgrade(&sessions),
    })
}

pub struct GatewayLinkManager {
    worker_id: WorkerId,
    pinned: HashSet<String>,
    links: HashMap<String, GatewayLink>,
}

impl GatewayLinkManager {
    pub fn new(worker_id: WorkerId, pinned: Vec<String>) -> Self {
        Self {
            worker_id,
            pinned: pinned.iter().map(|a| canonical_addr(a)).collect(),
            links: HashMap::new(),
        }
    }

    pub async fn dial_pinned(&mut self) -> Result<()> {
        let mut pinned: Vec<String> = self.pinned.iter().cloned().collect();
        pinned.sort();
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

    pub async fn reconcile(&mut self, roster: &[GatewayEndpoint]) {
        let desired: HashSet<String> = self
            .pinned
            .iter()
            .cloned()
            .chain(roster.iter().map(|g| canonical_addr(&g.addr)))
            .collect();

        let stale: Vec<String> = self
            .links
            .keys()
            .filter(|addr| !desired.contains(*addr))
            .cloned()
            .collect();
        for addr in stale {
            self.links.remove(&addr);
            tracing::info!(
                worker = %self.worker_id,
                gateway = %addr,
                "dropped gateway link (left roster)"
            );
        }

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

    pub fn addrs(&self) -> Vec<String> {
        let mut addrs: Vec<String> = self.links.keys().cloned().collect();
        addrs.sort();
        addrs
    }
}

fn canonical_addr(addr: &str) -> String {
    if addr.starts_with("unix:") {
        addr.to_string()
    } else {
        addr.strip_prefix("tcp://").unwrap_or(addr).to_string()
    }
}

#[derive(Clone)]
struct WorkerControlServer {
    worker_id: WorkerId,
    gateway: GatewayInboundClient,
    sessions: Arc<SessionRegistry>,
}

#[derive(Default)]
struct SessionRegistry {
    sessions: Mutex<HashMap<SessionId, SessionHandle>>,
    active: Mutex<HashMap<ReqId, (SessionId, ClientId)>>,
    #[cfg(feature = "session-lifetime-diagnostic")]
    diagnostic_hold: DiagnosticHold,
}

#[cfg(feature = "session-lifetime-diagnostic")]
#[derive(Default)]
struct DiagnosticHold {
    armed: AtomicBool,
    held: Mutex<Option<ReqId>>,
    reached: Notify,
}

impl SessionRegistry {
    async fn close_session(&self, session: SessionId) {
        if self.sessions.lock().await.remove(&session).is_some() {
            tracing::debug!(%session, "frontend session ended; releasing worker session");
        }
    }
}

struct SessionHandle {
    #[cfg(feature = "session-lifetime-diagnostic")]
    client_id: ClientId,
    turns: mpsc::Sender<Request>,
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
        let owner = self.sessions.active.lock().await.get(&req_id).copied();
        if let Some((session, _)) = owner
            && let Some(handle) = self.sessions.sessions.lock().await.get(&session)
        {
            let notify = handle.cancels.lock().await.get(&req_id).cloned();
            if let Some(notify) = notify {
                notify.notify_one();
                tracing::debug!(%req_id, %session, "reverse cancel signalled");
            }
        }
    }

    async fn set_priority(self, _: tarpc::context::Context, req_id: ReqId, p: Priority) {
        tracing::debug!(%req_id, ?p, "set_priority: no runtime hook (no-op)");
    }

    async fn drain(self, _: tarpc::context::Context) {
        tracing::info!("drain: no runtime hook (best-effort no-op)");
    }

    async fn close_session(self, _: tarpc::context::Context, session: SessionId) {
        self.sessions.close_session(session).await;
    }
}

impl WorkerControlServer {
    async fn admit(&self, req: Request) -> Result<()> {
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
                #[cfg(feature = "session-lifetime-diagnostic")]
                client_id,
                turns: turns_tx.clone(),
                cancels,
            },
        );
        Ok(turns_tx)
    }
}

enum TurnEnd {
    Done,
    Aborted,
    LinkGone,
}

#[derive(Default)]
struct TurnRoutes {
    by_corr: HashMap<u32, ReqId>,
    by_pid: HashMap<String, ReqId>,
    inboxes: HashMap<ReqId, mpsc::Sender<ServerMessage>>,
    awaiting_pid: HashSet<u32>,
}

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
        if let ProcBinding::Known(pid) = binding {
            self.by_pid.insert(pid.clone(), req_id);
        }
        rx
    }

    fn close(&mut self, req_id: ReqId) {
        self.inboxes.remove(&req_id);
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
            reg.active.lock().await.insert(req_id, (session, client_id));
        }

        let fed = feed_turn(client_id, req_id, req.message);

        let gateway = gateway.clone();
        let routes = routes.clone();
        let cancels = cancels.clone();
        let registry = registry.clone();
        let link_gone_tx = link_gone_tx.clone();
        running.spawn(async move {
            let outcome = run_turn(
                client_id,
                &gateway,
                &cancel,
                req_id,
                fed,
                corr,
                binding,
                inbox,
                #[cfg(feature = "session-lifetime-diagnostic")]
                registry.clone(),
            )
            .await;
            routes.lock().await.close(req_id);
            cancels.lock().await.remove(&req_id);
            if let Some(reg) = registry.upgrade() {
                let mut active = reg.active.lock().await;
                if active
                    .get(&req_id)
                    .is_some_and(|(_, owner)| *owner == client_id)
                {
                    active.remove(&req_id);
                }
                #[cfg(feature = "session-lifetime-diagnostic")]
                {
                    let mut held = reg.diagnostic_hold.held.lock().await;
                    if *held == Some(req_id) {
                        *held = None;
                    }
                }
            }
            if let TurnEnd::LinkGone = outcome {
                let _ = link_gone_tx.send(());
            }
        });

        while running.try_join_next().is_some() {}
    }

    // Stop taking turns before the teardown below, so a dispatch racing a
    // close gets a fresh session instead of queueing onto a driver that has
    // already stopped reading.
    drop(turns);

    // The prompt path, not the guarantee. A cancelled turn terminates the
    // process it owns on its way out, which frees it sooner than session
    // cleanup would; but the worker cannot promise that for every turn — a
    // cancel arriving before a launch reply has no pid to terminate, and a
    // turn blocked in `push_tokens` never reaches its `select!`. What holds
    // regardless is `runtime::server::close_session` below: the session's
    // `Drop` terminates the launches it owns. So this wait expiring is a
    // slower release, not a leaked process.
    for notify in cancels.lock().await.values() {
        notify.notify_one();
    }
    let drained = tokio::time::timeout(TURN_DRAIN_LIMIT, async {
        while running.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        tracing::warn!(%session, "turns outlived the drain limit; aborting them");
    }
    running.shutdown().await;
    router.abort();
    runtime::server::close_session(client_id);
    if let Some(reg) = registry.upgrade() {
        reg.active
            .lock()
            .await
            .retain(|_, (_, owner)| *owner != client_id);
        // A dispatch racing the close may already have put a fresh driver's
        // handle under this `SessionId`; only this driver's own entry, whose
        // sender the `drop` above closed, is ours to retire.
        let mut sessions = reg.sessions.lock().await;
        if sessions.get(&session).is_some_and(|h| h.turns.is_closed()) {
            sessions.remove(&session);
        }
    }
    tracing::debug!(%session, "session driver exited");
}

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
            let inbox = {
                let mut routes = routes.lock().await;
                match routes.target(&msg) {
                    Some(req_id) => routes.inboxes.get(&req_id).cloned(),
                    None => {
                        tracing::debug!(?msg, "runtime message matched no live turn");
                        None
                    }
                }
            };
            if let Some(inbox) = inbox {
                let _ = inbox.send(msg).await;
            }
        }
    }
}

enum Fed {
    Streaming,
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
    #[cfg(feature = "session-lifetime-diagnostic")] registry: Weak<SessionRegistry>,
) -> TurnEnd {
    let proc_launch = binding.is_process();
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
                return TurnEnd::Aborted;
            }
            msg = inbox.recv() => {
                let Some(msg) = msg else {
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
                    #[cfg(feature = "session-lifetime-diagnostic")]
                    if let Some(registry) = registry.upgrade()
                        && registry.diagnostic_hold.armed.swap(false, Ordering::AcqRel)
                    {
                        *registry.diagnostic_hold.held.lock().await = Some(req_id);
                        registry.diagnostic_hold.reached.notify_one();
                        continue;
                    }
                    return push_eos(gateway, req_id).await;
                }
            }
        }
    }
}

async fn push_eos(gateway: &GatewayInboundClient, req_id: ReqId) -> TurnEnd {
    match gateway.push_tokens(push_ctx(), req_id, Tokens::Eos).await {
        Ok(_) => TurnEnd::Done,
        Err(e) => {
            tracing::warn!(%req_id, error = %e, "push Eos transport error");
            TurnEnd::LinkGone
        }
    }
}

fn push_ctx() -> tarpc::context::Context {
    let mut ctx = tarpc::context::current();
    ctx.deadline = std::time::Instant::now() + PUSH_DEADLINE;
    ctx
}

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

fn terminate(process_id: &str) -> ClientMessage {
    ClientMessage::TerminateProcess {
        corr_id: 0,
        process_id: process_id.to_string(),
    }
}

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

enum ProcBinding {
    None,
    FromReply,
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

    fn is_process(&self) -> bool {
        !matches!(self, Self::None)
    }
}

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
