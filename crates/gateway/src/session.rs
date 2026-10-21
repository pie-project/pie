use std::collections::HashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use client_api::{ClientMessage, ServerMessage};
use ids::{ReqId, SessionId, TenantId, WorkerId};
use tokio::sync::{mpsc, watch};
use worker_api::{BlobRef, Priority, Request, Tokens};

use crate::policy::wit::{By, Outcome};

const DEFAULT_PIPE_CAP: usize = 256;

const DRAIN_POLL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone)]
pub struct Identity {
    pub tenant: TenantId,
    pub user: String,
    pub client_ip: Option<IpAddr>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TurnInput {
    pub message: ClientMessage,
    pub blobs: Vec<BlobRef>,
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Affinity {
    Ephemeral,
    Sticky,
}

#[derive(Debug, Clone)]
pub enum SessionError {
    Admission(String),
    NoWorker,
    Draining,
    Launch(String),
    Refused(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Admission(r) => write!(f, "admission rejected: {r}"),
            SessionError::NoWorker => f.write_str("no worker available"),
            SessionError::Draining => f.write_str("gateway draining"),
            SessionError::Launch(r) => write!(f, "launch refused: {r}"),
            SessionError::Refused(r) => write!(f, "refused by the cluster policy: {r}"),
        }
    }
}

impl std::error::Error for SessionError {}

pub struct TokenRx {
    rx: mpsc::Receiver<Tokens>,
}

impl TokenRx {
    pub async fn recv(&mut self) -> Option<Tokens> {
        self.rx.recv().await
    }
}

#[derive(Debug, Clone)]
pub enum AdmitReject {
    Full(String),
    Refused(String),
}

#[derive(Debug, Clone)]
pub struct DispatchFail;

#[async_trait::async_trait]
pub trait TurnRouter: Send + Sync + 'static {
    async fn admit(&self, req: &Request) -> Result<(), AdmitReject>;

    async fn dispatch(
        &self,
        req: &Request,
        affinity: Option<u64>,
    ) -> Result<WorkerId, DispatchFail>;

    async fn cancel(&self, worker: WorkerId, req: ReqId);

    async fn end_session(&self, session: SessionId);

    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>>;

    fn launched(&self, _req: ReqId, _process: &str, _worker: WorkerId) {}

    fn counted(&self, _req: &Request, _prompt: u64, _generated: u64) {}

    fn ended(&self, _req: &Request, _outcome: Outcome, _by: By) {}

    /// A running turn moved to `worker`, or (`None`) could not.
    fn moved(&self, _req: &Request, _worker: Option<WorkerId>, _why: &str) {}
}

struct TurnState {
    worker: Option<WorkerId>,
    request: Request,
    affinity: Option<u64>,
    emitted: bool,
    sink: mpsc::Sender<Tokens>,
    open: Arc<AtomicBool>,
    outcome: Option<Outcome>,
    /// The program's resume log as of what the client has been sent: what
    /// a move relaunches it with. None: it keeps none, and a turn that has
    /// sent anything cannot be moved.
    resume: Option<String>,
    /// Moved: the worker it left, whose late frames are dropped, and the
    /// new launch's answer, which the client does not see again.
    left: Option<WorkerId>,
    relaunched: bool,
}

struct Inner {
    turns: Mutex<HashMap<ReqId, TurnState>>,
    router: Arc<dyn TurnRouter>,
    next_req: AtomicU64,
    next_session: AtomicU64,
    draining: AtomicBool,
    live: AtomicUsize,
    pipe_cap: usize,
}

/// A program's resume note ([`client_api::resume`]): kept on the turn, and
/// what the client sees of it, the message it carries, put in its place.
/// `false`: the note carries no message, so the client sees nothing.
fn take_note(turn: &mut TurnState, message: &mut ServerMessage) -> bool {
    let ServerMessage::ProcessEvent { event, value, .. } = message else {
        return true;
    };
    if event != client_api::resume::EVENT {
        return true;
    }
    let Ok(note) = serde_json::from_str::<client_api::resume::Note>(value) else {
        tracing::warn!("a resume note that does not parse is dropped");
        return false;
    };
    turn.resume = note.apply(turn.resume.take());
    match note.message {
        Some(sent) => {
            *event = "message".to_string();
            *value = sent;
            true
        }
        None => false,
    }
}

/// Readies a turn to be launched again away from `old`: its launch carries
/// what the client was sent, which the program continues after, and `old`'s
/// late frames and the new launch's answer are kept from the client.
fn relaunch(turn: &mut TurnState, old: WorkerId) {
    if let ClientMessage::LaunchProcess { resume, .. } = &mut turn.request.message {
        *resume = turn.resume.clone();
    }
    turn.worker = None;
    turn.left = Some(old);
    turn.relaunched = true;
}

fn is_launch_answer(turn: &TurnState, message: &ServerMessage) -> bool {
    matches!(
        (&turn.request.message, message),
        (ClientMessage::LaunchProcess { corr_id, .. }, ServerMessage::Response { corr_id: answered, .. })
            if corr_id == answered
    )
}

impl Inner {
    async fn run_turn(
        self: &Arc<Self>,
        session: SessionId,
        tenant: &TenantId,
        mut input: TurnInput,
        affinity: Option<u64>,
        open: &Arc<AtomicBool>,
    ) -> Result<(ReqId, TokenRx), SessionError> {
        if let ClientMessage::LaunchProcess { props, session, .. } = &mut input.message {
            client_api::props::from_client(props).map_err(SessionError::Launch)?;
            client_api::props::check_session(session.as_deref()).map_err(SessionError::Launch)?;
        }
        let req_id = ReqId(self.next_req.fetch_add(1, Ordering::Relaxed));
        let request = Request {
            req_id,
            session,
            tenant: tenant.clone(),
            priority: input.priority,
            blobs: input.blobs,
            message: input.message,
            cluster_props: Default::default(),
            executors: Vec::new(),
        };

        if let Err(refusal) = self.router.admit(&request).await {
            let r = match refusal {
                AdmitReject::Refused(why) => return Err(SessionError::Refused(why)),
                AdmitReject::Full(why) => why,
            };
            // The client hears each refusal under its own id; the server log
            // hears the first and then every doubling, so a pool that admits
            // nothing for minutes leaves a trace without a line per launch.
            static REFUSED: AtomicU64 = AtomicU64::new(0);
            let refused = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
            if refused.is_power_of_two() {
                tracing::warn!(
                    tenant = %tenant,
                    refused,
                    "turn refused at admission: {}",
                    r
                );
            }
            return Err(SessionError::Admission(r));
        }

        let (tx, rx) = mpsc::channel(self.pipe_cap);
        {
            let mut turns = self.turns.lock().unwrap();
            turns.insert(
                req_id,
                TurnState {
                    worker: None,
                    request: request.clone(),
                    affinity,
                    emitted: false,
                    sink: tx,
                    open: open.clone(),
                    outcome: None,
                    resume: None,
                    left: None,
                    relaunched: false,
                },
            );
        }

        // On a task of its own so a caller dropped mid-dispatch still learns
        // where the turn landed, and `dispatch` can cancel it there.
        let inner = self.clone();
        let open = open.clone();
        match tokio::spawn(async move { inner.dispatch(req_id, &request, affinity, &open).await })
            .await
        {
            Ok(Ok(_)) => Ok((req_id, TokenRx { rx })),
            _ => Err(SessionError::NoWorker),
        }
    }

    /// Dispatches a turn and binds it to the worker that took it. A turn
    /// aborted meanwhile, or whose receiver is gone, is cancelled there, and
    /// its session ended there too once closed, since the dispatch reopened it.
    async fn dispatch(
        &self,
        req_id: ReqId,
        request: &Request,
        affinity: Option<u64>,
        open: &AtomicBool,
    ) -> Result<WorkerId, DispatchFail> {
        let worker = match self.router.dispatch(request, affinity).await {
            Ok(worker) => worker,
            Err(DispatchFail) => {
                self.end(req_id, Outcome::Failed, By::Host);
                return Err(DispatchFail);
            }
        };
        let bound = {
            let mut turns = self.turns.lock().unwrap();
            match turns.get_mut(&req_id) {
                Some(turn) if !turn.sink.is_closed() => {
                    turn.worker = Some(worker);
                    true
                }
                Some(_) => {
                    let turn = turns.remove(&req_id).expect("just found");
                    self.router
                        .ended(&turn.request, Outcome::Cancelled, By::Client);
                    false
                }
                None => false,
            }
        };
        if !bound {
            self.router.cancel(worker, req_id).await;
            if !open.load(Ordering::Acquire) {
                self.router.end_session(request.session).await;
            }
        }
        Ok(worker)
    }

    fn abort_turn(&self, req_id: ReqId) -> Option<WorkerId> {
        self.end(req_id, Outcome::Cancelled, By::Client)
    }

    /// A stream that already said how it ends is believed: a client that leaves
    /// with its answer cancelled nothing.
    fn end(&self, req_id: ReqId, outcome: Outcome, by: By) -> Option<WorkerId> {
        let turn = self.turns.lock().unwrap().remove(&req_id)?;
        let (outcome, by) = turn.outcome.map_or((outcome, by), |said| (said, By::Host));
        self.router.ended(&turn.request, outcome, by);
        turn.worker
    }
}

#[derive(Clone)]
pub struct Sessions {
    inner: Arc<Inner>,
}

impl Sessions {
    pub fn new(router: Arc<dyn TurnRouter>) -> Self {
        Self::with_pipe_cap(router, DEFAULT_PIPE_CAP)
    }

    pub fn with_pipe_cap(router: Arc<dyn TurnRouter>, pipe_cap: usize) -> Self {
        let inner = Arc::new(Inner {
            turns: Mutex::new(HashMap::new()),
            router,
            next_req: AtomicU64::new(0),
            next_session: AtomicU64::new(0),
            draining: AtomicBool::new(false),
            live: AtomicUsize::new(0),
            pipe_cap,
        });
        spawn_drop_watcher(inner.clone());
        Self { inner }
    }

    pub async fn create(
        &self,
        ident: Identity,
        first: TurnInput,
        affinity: Affinity,
    ) -> Result<(SessionHandle, TokenRx), SessionError> {
        if self.inner.draining.load(Ordering::Acquire) {
            return Err(SessionError::Draining);
        }
        let session = SessionId(self.inner.next_session.fetch_add(1, Ordering::Relaxed));
        let affinity_key = match affinity {
            Affinity::Ephemeral => None,
            Affinity::Sticky => Some(session.0),
        };
        self.inner.live.fetch_add(1, Ordering::Relaxed);
        // Made before the first turn so a create dropped mid-dispatch still
        // releases its count and ends the session.
        let handle = SessionHandle {
            inner: self.inner.clone(),
            session,
            tenant: ident.tenant,
            affinity_key,
            current: Mutex::new(Vec::new()),
            open: Arc::new(AtomicBool::new(true)),
        };
        match handle.turn(first).await {
            Ok(rx) => Ok((handle, rx)),
            Err(e) => {
                // Nothing reached a worker, so there is no session to end.
                handle.open.store(false, Ordering::Release);
                Err(e)
            }
        }
    }

    pub async fn feed(&self, req_id: ReqId, chunk: Tokens) -> worker_api::Control {
        self.feed_from(req_id, chunk, None).await
    }

    /// A chunk from worker `from`: one from a worker the turn has left is
    /// dropped, and a program's resume note is kept on the turn for a move,
    /// the client seeing only the message it carries.
    pub async fn feed_from(
        &self,
        req_id: ReqId,
        mut chunk: Tokens,
        from: Option<WorkerId>,
    ) -> worker_api::Control {
        use worker_api::Control;

        let sink = {
            let mut turns = self.inner.turns.lock().unwrap();
            let Some(turn) = turns.get_mut(&req_id) else {
                return Control::Abort;
            };
            if from.is_some() && turn.left.is_some() && from == turn.left {
                return Control::Abort;
            }
            if let Tokens::Chunk(message) = &mut chunk {
                if !take_note(turn, message) {
                    return Control::Continue;
                }
                let relaunch = turn.relaunched && is_launch_answer(turn, message);
                self.note(turn, req_id, message);
                if relaunch {
                    turn.relaunched = false;
                    return Control::Continue;
                }
            }
            turn.sink.clone()
        };

        let is_eos = matches!(chunk, Tokens::Eos);
        match sink.send(chunk).await {
            Ok(()) => {
                if is_eos {
                    self.inner.end(req_id, Outcome::Completed, By::Host);
                } else if let Some(turn) = self.inner.turns.lock().unwrap().get_mut(&req_id) {
                    turn.emitted = true;
                }
                Control::Continue
            }
            Err(_) => {
                self.inner.end(req_id, Outcome::Cancelled, By::Client);
                Control::Abort
            }
        }
    }

    pub fn counted(&self, req_id: ReqId, prompt: u64, generated: u64) {
        if let Some(turn) = self.inner.turns.lock().unwrap().get(&req_id) {
            self.inner.router.counted(&turn.request, prompt, generated);
        }
    }

    fn note(&self, turn: &mut TurnState, req_id: ReqId, message: &ServerMessage) {
        let ClientMessage::LaunchProcess { corr_id, .. } = &turn.request.message else {
            return;
        };
        match message {
            ServerMessage::Response {
                corr_id: answered,
                ok,
                result,
            } if answered == corr_id => {
                if !ok {
                    turn.outcome = Some(Outcome::Failed);
                } else if let Some(worker) = turn.worker
                    && !result.is_empty()
                {
                    self.inner.router.launched(req_id, result, worker);
                }
            }
            ServerMessage::ProcessEvent { event, .. } if event == "return" => {
                turn.outcome = Some(Outcome::Completed);
            }
            ServerMessage::ProcessEvent { event, .. } if event == "error" => {
                turn.outcome = Some(Outcome::Failed);
            }
            _ => {}
        }
    }

    /// Moves a running turn to `target`: ends it where it runs, and launches
    /// it again with what the client was sent, which the program continues
    /// after under the same response id. The client's stream does not break.
    /// The placement table already names `target`, so the dispatch goes
    /// there.
    pub async fn migrate(&self, req_id: ReqId, target: WorkerId) -> bool {
        let entry = {
            let mut turns = self.inner.turns.lock().unwrap();
            let Some(turn) = turns.get_mut(&req_id) else {
                return false;
            };
            let Some(old) = turn.worker else {
                return false;
            };
            if old == target || turn.outcome.is_some() {
                return false;
            }
            // Its program keeps no resume log: run again, it would send
            // what the client already has.
            if turn.emitted && turn.resume.is_none() {
                return false;
            }
            relaunch(turn, old);
            (old, turn.request.clone(), turn.affinity, turn.open.clone())
        };
        let (old, request, affinity, open) = entry;
        tracing::info!(req = %req_id, from = %old, to = %target, "moving a turn");
        self.inner.router.cancel(old, req_id).await;
        match self.inner.dispatch(req_id, &request, affinity, &open).await {
            Ok(worker) => {
                self.inner.router.moved(&request, Some(worker), "");
                true
            }
            Err(_) => {
                self.inner
                    .router
                    .moved(&request, None, "no worker took the moved turn");
                false
            }
        }
    }

    pub async fn terminate(&self, req_id: ReqId) {
        if let Some(worker) = self.inner.end(req_id, Outcome::Cancelled, By::Policy) {
            self.inner.router.cancel(worker, req_id).await;
        }
    }

    pub fn redirect(&self, req_id: ReqId) {
        let inner = self.inner.clone();
        let entry = {
            let mut turns = inner.turns.lock().unwrap();
            match turns.get_mut(&req_id) {
                Some(turn) => {
                    turn.worker = None;
                    Some((turn.request.clone(), turn.affinity, turn.open.clone()))
                }
                None => None,
            }
        };
        let Some((request, affinity, open)) = entry else {
            return;
        };
        tokio::spawn(async move {
            let _ = inner.dispatch(req_id, &request, affinity, &open).await;
        });
    }

    pub async fn drain(&self, max: Duration) {
        self.inner.draining.store(true, Ordering::Release);
        let deadline = Instant::now() + max;
        while self.inner.live.load(Ordering::Acquire) > 0 {
            if Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(DRAIN_POLL).await;
        }
        self.inner.turns.lock().unwrap().clear();
    }

    pub fn live(&self) -> usize {
        self.inner.live.load(Ordering::Acquire)
    }
}

pub struct SessionHandle {
    inner: Arc<Inner>,
    session: SessionId,
    tenant: TenantId,
    affinity_key: Option<u64>,
    current: Mutex<Vec<ReqId>>,
    open: Arc<AtomicBool>,
}

impl SessionHandle {
    pub async fn turn(&self, next: TurnInput) -> Result<TokenRx, SessionError> {
        let (req_id, rx) = self
            .inner
            .run_turn(
                self.session,
                &self.tenant,
                next,
                self.affinity_key,
                &self.open,
            )
            .await?;
        self.current.lock().unwrap().push(req_id);
        Ok(rx)
    }

    pub async fn cancel(&self) {
        let req_ids = std::mem::take(&mut *self.current.lock().unwrap());
        for req_id in req_ids {
            if let Some(worker) = self.inner.abort_turn(req_id) {
                self.inner.router.cancel(worker, req_id).await;
            }
        }
    }

    pub async fn close(&self) {
        self.cancel().await;
        if self.open.swap(false, Ordering::AcqRel) {
            self.inner.router.end_session(self.session).await;
        }
    }

    pub fn id(&self) -> SessionId {
        self.session
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.inner.live.fetch_sub(1, Ordering::Relaxed);
        let req_ids = std::mem::take(&mut *self.current.lock().unwrap());
        for req_id in req_ids {
            if let Some(worker) = self.inner.abort_turn(req_id) {
                let router = self.inner.router.clone();
                tokio::spawn(async move { router.cancel(worker, req_id).await });
            }
        }
        if self.open.swap(false, Ordering::AcqRel) {
            let router = self.inner.router.clone();
            let session = self.session;
            tokio::spawn(async move { router.end_session(session).await });
        }
    }
}

fn spawn_drop_watcher(inner: Arc<Inner>) {
    let mut connected = inner.router.connected();
    tokio::spawn(async move {
        loop {
            if connected.changed().await.is_err() {
                break;
            }
            let live: Arc<HashSet<WorkerId>> = connected.borrow_and_update().clone();

            let affected: Vec<_> = {
                let turns = inner.turns.lock().unwrap();
                turns
                    .iter()
                    .filter(|(_, t)| t.worker.is_some_and(|w| !live.contains(&w)))
                    .map(|(id, t)| {
                        (
                            *id,
                            t.emitted,
                            t.request.clone(),
                            t.affinity,
                            t.open.clone(),
                        )
                    })
                    .collect()
            };

            // A turn that has said nothing is launched again as it came; one
            // whose frames named what they covered goes on after it, as a
            // moved turn does; any other has said what cannot be unsaid.
            for (req_id, _, _, affinity, open) in affected {
                let (request, resumed) = {
                    let mut turns = inner.turns.lock().unwrap();
                    let Some(turn) = turns.get_mut(&req_id) else {
                        continue;
                    };
                    let Some(old) = turn.worker else {
                        continue;
                    };
                    let resumed = turn.emitted;
                    if !turn.emitted {
                        turn.worker = None;
                    } else if turn.resume.is_some() && turn.outcome.is_none() {
                        relaunch(turn, old);
                        tracing::info!(req = %req_id, from = %old, "resuming a turn its worker dropped");
                    } else {
                        drop(turns);
                        inner.end(req_id, Outcome::Failed, By::Host);
                        continue;
                    }
                    (turn.request.clone(), resumed)
                };
                match inner.dispatch(req_id, &request, affinity, &open).await {
                    Ok(worker) if resumed => inner.router.moved(&request, Some(worker), ""),
                    Err(_) if resumed => {
                        inner
                            .router
                            .moved(&request, None, "no worker took the dropped turn")
                    }
                    _ => {}
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_api::ServerMessage;
    use worker_api::Control;

    struct MockRouter {
        admit_ok: AtomicBool,
        dispatch_worker: Mutex<Option<WorkerId>>,
        dispatched: Mutex<Vec<ReqId>>,
        affinities: Mutex<Vec<Option<u64>>>,
        cancels: Mutex<Vec<(WorkerId, ReqId)>>,
        ended: Mutex<Vec<SessionId>>,
        held: AtomicBool,
        launches: Mutex<Vec<(ReqId, String, WorkerId)>>,
        endings: Mutex<Vec<(ReqId, Outcome, By)>>,
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
        messages: Mutex<Vec<ClientMessage>>,
        moves: Mutex<Vec<(ReqId, Option<WorkerId>)>>,
        _connected_tx: watch::Sender<Arc<HashSet<WorkerId>>>,
        connected_rx: watch::Receiver<Arc<HashSet<WorkerId>>>,
    }

    impl MockRouter {
        fn new(worker: Option<WorkerId>) -> Arc<Self> {
            let init: Arc<HashSet<WorkerId>> = Arc::new(worker.into_iter().collect::<HashSet<_>>());
            let (tx, rx) = watch::channel(init);
            Arc::new(Self {
                admit_ok: AtomicBool::new(true),
                dispatch_worker: Mutex::new(worker),
                dispatched: Mutex::new(Vec::new()),
                affinities: Mutex::new(Vec::new()),
                cancels: Mutex::new(Vec::new()),
                ended: Mutex::new(Vec::new()),
                held: AtomicBool::new(false),
                launches: Mutex::new(Vec::new()),
                endings: Mutex::new(Vec::new()),
                entered: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
                messages: Mutex::new(Vec::new()),
                moves: Mutex::new(Vec::new()),
                _connected_tx: tx,
                connected_rx: rx,
            })
        }
    }

    #[async_trait::async_trait]
    impl TurnRouter for MockRouter {
        async fn admit(&self, _req: &Request) -> Result<(), AdmitReject> {
            if self.admit_ok.load(Ordering::Acquire) {
                Ok(())
            } else {
                Err(AdmitReject::Full("full".into()))
            }
        }
        async fn dispatch(
            &self,
            req: &Request,
            affinity: Option<u64>,
        ) -> Result<WorkerId, DispatchFail> {
            self.dispatched.lock().unwrap().push(req.req_id);
            self.messages.lock().unwrap().push(req.message.clone());
            self.affinities.lock().unwrap().push(affinity);
            if self.held.load(Ordering::Acquire) {
                self.entered.notify_one();
                self.release.notified().await;
            }
            match *self.dispatch_worker.lock().unwrap() {
                Some(w) => Ok(w),
                None => Err(DispatchFail),
            }
        }
        async fn cancel(&self, worker: WorkerId, req: ReqId) {
            self.cancels.lock().unwrap().push((worker, req));
        }
        async fn end_session(&self, session: SessionId) {
            self.ended.lock().unwrap().push(session);
        }
        fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
            self.connected_rx.clone()
        }
        fn launched(&self, req: ReqId, process: &str, worker: WorkerId) {
            self.launches
                .lock()
                .unwrap()
                .push((req, process.to_string(), worker));
        }
        fn ended(&self, req: &Request, outcome: Outcome, by: By) {
            self.endings.lock().unwrap().push((req.req_id, outcome, by));
        }
        fn moved(&self, req: &Request, worker: Option<WorkerId>, _why: &str) {
            self.moves.lock().unwrap().push((req.req_id, worker));
        }
    }

    fn launch() -> TurnInput {
        TurnInput {
            message: ClientMessage::LaunchProcess {
                corr_id: 4,
                inferlet: "chat".into(),
                input: String::new(),
                capture_outputs: true,
                props: Default::default(),
                session: None,
                resume: None,
                cluster_props: Default::default(),
                executors: Vec::new(),
            },
            blobs: Vec::new(),
            priority: Priority::Normal,
        }
    }

    fn event(event: &str) -> Tokens {
        Tokens::Chunk(ServerMessage::ProcessEvent {
            process_id: "p".into(),
            event: event.into(),
            value: String::new(),
        })
    }

    #[tokio::test]
    async fn the_router_hears_where_a_launch_started_and_how_each_turn_ended() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (handle, mut rx) = sessions
            .create(ident(), launch(), Affinity::Sticky)
            .await
            .unwrap();
        let req = router.dispatched.lock().unwrap()[0];
        let started = Tokens::Chunk(ServerMessage::Response {
            corr_id: 4,
            ok: true,
            result: "p".into(),
        });
        for chunk in [started, event("return"), Tokens::Eos] {
            sessions.feed(req, chunk).await;
            rx.recv().await;
        }
        assert_eq!(
            *router.launches.lock().unwrap(),
            [(req, "p".to_string(), WorkerId(7))]
        );

        let failing = handle.turn(launch()).await.unwrap();
        let failed = router.dispatched.lock().unwrap()[1];
        sessions.feed(failed, event("error")).await;
        sessions.feed(failed, Tokens::Eos).await;
        let answered = handle.turn(launch()).await.unwrap();
        let answered_req = router.dispatched.lock().unwrap()[2];
        sessions.feed(answered_req, event("return")).await;
        let _cancelled = handle.turn(input()).await.unwrap();
        let cancelled = router.dispatched.lock().unwrap()[3];
        handle.cancel().await;
        drop((failing, answered));
        assert_eq!(
            *router.endings.lock().unwrap(),
            [
                (req, Outcome::Completed, By::Host),
                (failed, Outcome::Failed, By::Host),
                (answered_req, Outcome::Completed, By::Host),
                (cancelled, Outcome::Cancelled, By::Client),
            ]
        );
    }

    fn ident() -> Identity {
        Identity {
            tenant: TenantId("t".into()),
            user: "u".into(),
            client_ip: None,
            request_id: None,
        }
    }

    fn input() -> TurnInput {
        TurnInput {
            message: ClientMessage::Ping { corr_id: 1 },
            blobs: Vec::new(),
            priority: Priority::Normal,
        }
    }

    fn chunk() -> Tokens {
        Tokens::Chunk(ServerMessage::Response {
            corr_id: 1,
            ok: true,
            result: String::new(),
        })
    }

    #[tokio::test]
    async fn create_then_stream_chunk_and_eos() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (_h, mut rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();

        let req_id = router.dispatched.lock().unwrap()[0];
        assert_eq!(sessions.feed(req_id, chunk()).await, Control::Continue);
        assert!(matches!(rx.recv().await, Some(Tokens::Chunk(_))));

        assert_eq!(sessions.feed(req_id, Tokens::Eos).await, Control::Continue);
        assert!(matches!(rx.recv().await, Some(Tokens::Eos)));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn feed_unknown_req_aborts() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router);
        assert_eq!(sessions.feed(ReqId(999), chunk()).await, Control::Abort);
    }

    #[tokio::test]
    async fn dropped_consumer_aborts_feed() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router.clone());
        let (_h, rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let req_id = router.dispatched.lock().unwrap()[0];
        drop(rx);
        assert_eq!(sessions.feed(req_id, chunk()).await, Control::Abort);
    }

    #[tokio::test]
    async fn closing_a_session_ends_it_on_the_worker() {
        let router = MockRouter::new(Some(WorkerId(3)));
        let sessions = Sessions::new(router.clone());
        let (handle, _rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let session = handle.id();
        handle.close().await;
        assert!(router.ended.lock().unwrap().contains(&session));
    }

    #[tokio::test]
    async fn no_worker_surfaces() {
        let router = MockRouter::new(None);
        let sessions = Sessions::new(router);
        let res = sessions.create(ident(), input(), Affinity::Sticky).await;
        assert!(matches!(res, Err(SessionError::NoWorker)));
    }

    async fn settle() {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn a_turn_its_worker_dropped_goes_on_after_what_the_client_was_sent() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let mut turn = launch();
        if let ClientMessage::LaunchProcess { input, .. } = &mut turn.message {
            *input = r#"{"prompt":"hi"}"#.into();
        }
        let (_h, mut rx) = sessions
            .create(ident(), turn, Affinity::Sticky)
            .await
            .unwrap();
        let req = router.dispatched.lock().unwrap()[0];
        let answer = || {
            Tokens::Chunk(ServerMessage::Response {
                corr_id: 4,
                ok: true,
                result: "p".into(),
            })
        };
        let frame = |value: &str| {
            Tokens::Chunk(ServerMessage::ProcessEvent {
                process_id: "p".into(),
                event: "message".into(),
                value: value.into(),
            })
        };
        let note = |note: client_api::resume::Note| {
            Tokens::Chunk(ServerMessage::ProcessEvent {
                process_id: "p".into(),
                event: client_api::resume::EVENT.into(),
                value: serde_json::to_string(&note).unwrap(),
            })
        };
        sessions.feed_from(req, answer(), Some(WorkerId(7))).await;
        sessions
            .feed_from(
                req,
                note(client_api::resume::Note {
                    state: Some("cmpl-1\n".into()),
                    ..Default::default()
                }),
                Some(WorkerId(7)),
            )
            .await;
        sessions
            .feed_from(
                req,
                note(client_api::resume::Note {
                    more: "1 2 ".into(),
                    message: Some(r#"{"text":"ab"}"#.into()),
                    ..Default::default()
                }),
                Some(WorkerId(7)),
            )
            .await;
        rx.recv().await;
        let Some(Tokens::Chunk(ServerMessage::ProcessEvent { event, value, .. })) = rx.recv().await
        else {
            panic!("the frame");
        };
        assert_eq!(
            (event.as_str(), value.as_str()),
            ("message", r#"{"text":"ab"}"#),
            "the client sees what a note carries, and not a note carrying nothing"
        );

        // Worker 7 goes; 8 is all there is.
        *router.dispatch_worker.lock().unwrap() = Some(WorkerId(8));
        router
            ._connected_tx
            .send_replace(Arc::new([WorkerId(8)].into_iter().collect()));
        settle().await;
        assert_eq!(router.dispatched.lock().unwrap().len(), 2);
        let relaunch = router.messages.lock().unwrap()[1].clone();
        let ClientMessage::LaunchProcess { input, resume, .. } = &relaunch else {
            panic!("a launch");
        };
        assert_eq!(input, r#"{"prompt":"hi"}"#, "the program's own input");
        assert_eq!(
            resume.as_deref(),
            Some("cmpl-1\n1 2 "),
            "its log as the client had it"
        );
        assert_eq!(*router.moves.lock().unwrap(), [(req, Some(WorkerId(8)))]);
        assert!(
            router.endings.lock().unwrap().is_empty(),
            "the turn goes on"
        );

        // The new launch's answer is not heard again; its frames are.
        sessions.feed_from(req, answer(), Some(WorkerId(8))).await;
        sessions
            .feed_from(req, frame(r#"{"text":"c"}"#), Some(WorkerId(8)))
            .await;
        let Some(Tokens::Chunk(ServerMessage::ProcessEvent { value, .. })) = rx.recv().await else {
            panic!("the next frame");
        };
        assert_eq!(value, r#"{"text":"c"}"#);
    }

    #[tokio::test]
    async fn a_turn_that_said_what_it_cannot_resume_from_fails_with_its_worker() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (_h, _rx) = sessions
            .create(ident(), launch(), Affinity::Sticky)
            .await
            .unwrap();
        let req = router.dispatched.lock().unwrap()[0];
        sessions
            .feed_from(req, event("message"), Some(WorkerId(7)))
            .await;
        router._connected_tx.send_replace(Arc::new(HashSet::new()));
        settle().await;
        assert_eq!(
            *router.endings.lock().unwrap(),
            [(req, Outcome::Failed, By::Host)]
        );
        assert_eq!(router.dispatched.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_turn_that_sent_without_a_resume_log_is_not_moved() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (_h, _rx) = sessions
            .create(ident(), launch(), Affinity::Sticky)
            .await
            .unwrap();
        let req = router.dispatched.lock().unwrap()[0];
        sessions
            .feed_from(req, event("message"), Some(WorkerId(7)))
            .await;
        assert!(
            !sessions.migrate(req, WorkerId(8)).await,
            "run again, it would send what the client has"
        );
        assert_eq!(router.dispatched.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn cancelled_dispatches_are_cancelled_where_they_land() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        router.held.store(true, Ordering::Release);

        let create = tokio::spawn({
            let sessions = sessions.clone();
            async move { sessions.create(ident(), input(), Affinity::Sticky).await }
        });
        router.entered.notified().await;
        create.abort();
        let _ = create.await;
        router.release.notify_one();
        settle().await;
        assert_eq!(
            *router.cancels.lock().unwrap(),
            vec![(WorkerId(7), ReqId(0))]
        );
        assert!(router.ended.lock().unwrap().contains(&SessionId(0)));
        assert_eq!(sessions.live(), 0);

        router.held.store(false, Ordering::Release);
        let (handle, _rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        router.held.store(true, Ordering::Release);
        {
            let mut turn = Box::pin(handle.turn(input()));
            tokio::select! {
                _ = router.entered.notified() => {}
                _ = &mut turn => panic!("turn dispatched while held"),
            }
        }
        router.release.notify_one();
        settle().await;
        assert!(
            router
                .cancels
                .lock()
                .unwrap()
                .contains(&(WorkerId(7), ReqId(2)))
        );
        assert!(!sessions.inner.turns.lock().unwrap().contains_key(&ReqId(2)));
        assert!(!router.ended.lock().unwrap().contains(&handle.id()));
    }
}
