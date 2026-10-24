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

use crate::policy::By;
use controller_api::Outcome;

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
    lost: Arc<AtomicBool>,
}

impl TokenRx {
    pub async fn recv(&mut self) -> Option<Tokens> {
        self.rx.recv().await
    }

    /// The turn ended because its worker was lost after it had sent the
    /// client output: what it sent cannot be unsent, so it was not run
    /// again. An ingress that knows the API may continue it
    /// (`ingress::compat`); any other client is told to retry.
    #[must_use]
    pub fn lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }
}

/// What a client hears when the worker running its turn is lost after the
/// turn sent output.
pub const LOST_AFTER_OUTPUT: &str = "the worker running this turn was lost after it sent output; \
                                     what it sent stands, and the turn did not finish: retry it";

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
    /// The client has been handed output: a process event (a message, a
    /// return, stdout, stderr, an error) or a file, not just the launch's
    /// answer. From then on the turn is not run again. It is marked as the
    /// output goes to the client's pipe, before the send waits on it, so a
    /// slow reader cannot leave the turn looking unsent.
    emitted: bool,
    sink: mpsc::Sender<Tokens>,
    open: Arc<AtomicBool>,
    outcome: Option<Outcome>,
    /// Run again elsewhere before it sent output: the worker it left, whose
    /// late frames are dropped, and the new launch's answer, which the
    /// client does not see again.
    left: Option<WorkerId>,
    relaunched: bool,
    /// The client has had the launch's answer, naming this process: what a
    /// launch run again elsewhere is called to the client too.
    answered: bool,
    process: Option<String>,
    lost: Arc<AtomicBool>,
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

/// Readies a turn that has sent no output to be launched again away from
/// `old`, from its own input: `old`'s late frames are kept from the client,
/// and so is the new launch's answer if it had the first one.
fn relaunch(turn: &mut TurnState, old: WorkerId) {
    turn.worker = None;
    turn.left = Some(old);
    turn.relaunched = turn.answered;
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
        user: &str,
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
            user: user.to_string(),
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
        let lost = Arc::new(AtomicBool::new(false));
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
                    left: None,
                    relaunched: false,
                    answered: false,
                    process: None,
                    lost: lost.clone(),
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
            Ok(Ok(_)) => Ok((req_id, TokenRx { rx, lost })),
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
            user: ident.user,
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
    /// dropped.
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
                // A launch run again is another process to the worker, and
                // the one the client knows to the client.
                if let Some(known) = &turn.process
                    && let ServerMessage::ProcessEvent { process_id, .. }
                    | ServerMessage::File { process_id, .. } = message
                {
                    process_id.clone_from(known);
                }
                // The client knows this process: a launch run again answers
                // it again, which it hears only if the launch failed.
                let relaunch = turn.relaunched && is_launch_answer(turn, message);
                if relaunch {
                    turn.relaunched = false;
                }
                self.note(turn, req_id, message);
                if relaunch && matches!(message, ServerMessage::Response { ok: true, .. }) {
                    return Control::Continue;
                }
                if let ServerMessage::ProcessEvent { process_id, .. }
                | ServerMessage::File { process_id, .. } = message
                {
                    turn.emitted = true;
                    // What the client calls it, if it never had the answer.
                    if turn.process.is_none() {
                        turn.process = Some(process_id.clone());
                    }
                }
            }
            turn.sink.clone()
        };

        let is_eos = matches!(chunk, Tokens::Eos);
        match sink.send(chunk).await {
            Ok(()) => {
                if is_eos {
                    self.inner.end(req_id, Outcome::Completed, By::Host);
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
                if !turn.answered && *ok {
                    turn.process = Some(result.clone());
                }
                turn.answered = true;
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

    /// Moves a running turn to `target` while it has sent no output: ends it
    /// where it runs and launches it again from its own input. One that has
    /// sent output stays where it is. The placement table already names
    /// `target`, so the dispatch goes there.
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
            // Run again, it would send what the client already has.
            if turn.emitted {
                Err(turn.request.clone())
            } else {
                relaunch(turn, old);
                Ok((old, turn.request.clone(), turn.affinity, turn.open.clone()))
            }
        };
        let entry = match entry {
            Ok(entry) => entry,
            Err(request) => {
                tracing::debug!(req = %req_id, "a turn that sent output is not moved");
                self.inner.router.moved(
                    &request,
                    None,
                    "it has sent output, so it finishes where it runs",
                );
                return false;
            }
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
    user: String,
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
                &self.user,
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
                    .map(|(id, t)| (*id, t.affinity, t.open.clone()))
                    .collect()
            };

            // A turn that has sent no output is launched again from its own
            // input; one that has is told why it ends, and ends: what it sent
            // cannot be unsent.
            for (req_id, affinity, open) in affected {
                enum Next {
                    Again(Box<Request>),
                    Lost(mpsc::Sender<Tokens>, Option<String>),
                    End,
                }
                let next = {
                    let mut turns = inner.turns.lock().unwrap();
                    let Some(turn) = turns.get_mut(&req_id) else {
                        continue;
                    };
                    let Some(old) = turn.worker else {
                        continue;
                    };
                    if turn.outcome.is_some() {
                        Next::End
                    } else if turn.emitted {
                        turn.lost.store(true, Ordering::Release);
                        tracing::info!(req = %req_id, from = %old, "a turn that sent output lost its worker");
                        Next::Lost(turn.sink.clone(), turn.process.clone())
                    } else {
                        relaunch(turn, old);
                        tracing::info!(req = %req_id, from = %old, "running a turn again: its worker dropped it before it sent output");
                        Next::Again(Box::new(turn.request.clone()))
                    }
                };
                match next {
                    Next::Again(request) => {
                        let _ = inner.dispatch(req_id, &request, affinity, &open).await;
                    }
                    Next::Lost(sink, process_id) => {
                        // Ended first, so nothing the lost worker still sends
                        // reaches the client after the error; told on its
                        // own, so a client that does not read holds no other
                        // turn's recovery.
                        inner.end(req_id, Outcome::Failed, By::Host);
                        if let Some(process_id) = process_id {
                            tokio::spawn(async move {
                                let _ = sink
                                    .send(Tokens::Chunk(ServerMessage::ProcessEvent {
                                        process_id,
                                        event: "error".to_string(),
                                        value: LOST_AFTER_OUTPUT.to_string(),
                                    }))
                                    .await;
                            });
                        }
                    }
                    Next::End => {
                        inner.end(req_id, Outcome::Failed, By::Host);
                    }
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
    async fn a_turn_its_worker_dropped_before_it_sent_output_runs_again_from_its_input() {
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
        sessions.feed_from(req, answer(), Some(WorkerId(7))).await;
        assert!(matches!(
            rx.recv().await,
            Some(Tokens::Chunk(ServerMessage::Response { .. }))
        ));

        // Worker 7 goes; 8 is all there is.
        *router.dispatch_worker.lock().unwrap() = Some(WorkerId(8));
        router
            ._connected_tx
            .send_replace(Arc::new([WorkerId(8)].into_iter().collect()));
        settle().await;
        assert_eq!(router.dispatched.lock().unwrap().len(), 2);
        let relaunch = router.messages.lock().unwrap()[1].clone();
        let ClientMessage::LaunchProcess { input, .. } = &relaunch else {
            panic!("a launch");
        };
        assert_eq!(input, r#"{"prompt":"hi"}"#, "the program's own input");
        assert!(
            router.endings.lock().unwrap().is_empty(),
            "the turn goes on"
        );

        // The new launch's answer is not heard again; its frames are, and
        // late frames from the worker it left are not.
        sessions
            .feed_from(req, event("message"), Some(WorkerId(7)))
            .await;
        sessions.feed_from(req, answer(), Some(WorkerId(8))).await;
        let again = Tokens::Chunk(ServerMessage::ProcessEvent {
            process_id: "q".into(),
            event: "message".into(),
            value: String::new(),
        });
        sessions.feed_from(req, again, Some(WorkerId(8))).await;
        let Some(Tokens::Chunk(ServerMessage::ProcessEvent {
            event, process_id, ..
        })) = rx.recv().await
        else {
            panic!("the next frame");
        };
        assert_eq!(event, "message");
        assert_eq!(process_id, "p", "under the process the client knows");
        assert!(!rx.lost());
    }

    #[tokio::test]
    async fn a_turn_that_sent_output_fails_with_its_worker_and_says_why() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (_h, mut rx) = sessions
            .create(ident(), launch(), Affinity::Sticky)
            .await
            .unwrap();
        let req = router.dispatched.lock().unwrap()[0];
        sessions
            .feed_from(req, event("message"), Some(WorkerId(7)))
            .await;
        rx.recv().await;
        router._connected_tx.send_replace(Arc::new(HashSet::new()));
        settle().await;
        assert_eq!(
            *router.endings.lock().unwrap(),
            [(req, Outcome::Failed, By::Host)]
        );
        assert_eq!(router.dispatched.lock().unwrap().len(), 1, "not run again");
        let Some(Tokens::Chunk(ServerMessage::ProcessEvent { event, value, .. })) = rx.recv().await
        else {
            panic!("the reason");
        };
        assert_eq!(
            (event.as_str(), value.as_str()),
            ("error", LOST_AFTER_OUTPUT)
        );
        assert!(rx.lost());
    }

    #[tokio::test]
    async fn a_turn_that_sent_output_is_not_moved() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (_h, _rx) = sessions
            .create(ident(), launch(), Affinity::Sticky)
            .await
            .unwrap();
        let req = router.dispatched.lock().unwrap()[0];
        assert!(
            sessions.migrate(req, WorkerId(8)).await,
            "before output a turn moves"
        );
        let req = router.dispatched.lock().unwrap()[1];
        sessions
            .feed_from(req, event("message"), Some(WorkerId(8)))
            .await;
        assert!(
            !sessions.migrate(req, WorkerId(7)).await,
            "run again, it would send what the client has"
        );
        assert_eq!(router.dispatched.lock().unwrap().len(), 2);
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
