use std::collections::HashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use client_api::ClientMessage;
use ids::{ReqId, SessionId, TenantId, WorkerId};
use tokio::sync::{mpsc, watch};
use worker_api::{BlobRef, Priority, Request, Tokens};

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
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Admission(r) => write!(f, "admission rejected: {r}"),
            SessionError::NoWorker => f.write_str("no worker available"),
            SessionError::Draining => f.write_str("gateway draining"),
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
pub struct AdmitReject(pub String);

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

    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>>;
}

struct TurnState {
    worker: Option<WorkerId>,
    request: Request,
    affinity: Option<u64>,
    emitted: bool,
    sink: mpsc::Sender<Tokens>,
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

impl Inner {
    async fn run_turn(
        &self,
        session: SessionId,
        tenant: &TenantId,
        input: TurnInput,
        affinity: Option<u64>,
    ) -> Result<(ReqId, TokenRx), SessionError> {
        let req_id = ReqId(self.next_req.fetch_add(1, Ordering::Relaxed));
        let request = Request {
            req_id,
            session,
            tenant: tenant.clone(),
            priority: input.priority,
            blobs: input.blobs,
            message: input.message,
        };

        self.router
            .admit(&request)
            .await
            .map_err(|r| SessionError::Admission(r.0))?;

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
                },
            );
        }

        match self.router.dispatch(&request, affinity).await {
            Ok(worker) => {
                let mut turns = self.turns.lock().unwrap();
                if let Some(turn) = turns.get_mut(&req_id) {
                    turn.worker = Some(worker);
                }
                Ok((req_id, TokenRx { rx }))
            }
            Err(DispatchFail) => {
                self.turns.lock().unwrap().remove(&req_id);
                Err(SessionError::NoWorker)
            }
        }
    }

    fn abort_turn(&self, req_id: ReqId) -> Option<WorkerId> {
        self.turns
            .lock()
            .unwrap()
            .remove(&req_id)
            .and_then(|t| t.worker)
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
        let (req_id, rx) = match self
            .inner
            .run_turn(session, &ident.tenant, first, affinity_key)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                self.inner.live.fetch_sub(1, Ordering::Relaxed);
                return Err(e);
            }
        };
        let handle = SessionHandle {
            inner: self.inner.clone(),
            session,
            tenant: ident.tenant,
            affinity_key,
            current: Mutex::new(vec![req_id]),
        };
        Ok((handle, rx))
    }

    pub async fn feed(&self, req_id: ReqId, chunk: Tokens) -> worker_api::Control {
        use worker_api::Control;

        let sink = {
            let turns = self.inner.turns.lock().unwrap();
            match turns.get(&req_id) {
                Some(turn) => turn.sink.clone(),
                None => return Control::Abort,
            }
        };

        let is_eos = matches!(chunk, Tokens::Eos);
        match sink.send(chunk).await {
            Ok(()) => {
                {
                    let mut turns = self.inner.turns.lock().unwrap();
                    if is_eos {
                        turns.remove(&req_id);
                    } else if let Some(turn) = turns.get_mut(&req_id) {
                        turn.emitted = true;
                    }
                }
                Control::Continue
            }
            Err(_) => {
                self.inner.turns.lock().unwrap().remove(&req_id);
                Control::Abort
            }
        }
    }

    pub fn redirect(&self, req_id: ReqId) {
        let inner = self.inner.clone();
        let entry = {
            let mut turns = inner.turns.lock().unwrap();
            match turns.get_mut(&req_id) {
                Some(turn) => {
                    turn.worker = None;
                    Some((turn.request.clone(), turn.affinity))
                }
                None => None,
            }
        };
        let Some((request, affinity)) = entry else {
            return;
        };
        tokio::spawn(async move {
            match inner.router.dispatch(&request, affinity).await {
                Ok(worker) => {
                    let mut turns = inner.turns.lock().unwrap();
                    if let Some(turn) = turns.get_mut(&req_id) {
                        turn.worker = Some(worker);
                    }
                }
                Err(DispatchFail) => {
                    inner.turns.lock().unwrap().remove(&req_id);
                }
            }
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
}

impl SessionHandle {
    pub async fn turn(&self, next: TurnInput) -> Result<TokenRx, SessionError> {
        let (req_id, rx) = self
            .inner
            .run_turn(self.session, &self.tenant, next, self.affinity_key)
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

            let affected: Vec<(ReqId, bool, Request, Option<u64>)> = {
                let turns = inner.turns.lock().unwrap();
                turns
                    .iter()
                    .filter(|(_, t)| t.worker.is_some_and(|w| !live.contains(&w)))
                    .map(|(id, t)| (*id, t.emitted, t.request.clone(), t.affinity))
                    .collect()
            };

            for (req_id, emitted, request, affinity) in affected {
                if emitted {
                    inner.turns.lock().unwrap().remove(&req_id);
                    continue;
                }
                {
                    let mut turns = inner.turns.lock().unwrap();
                    match turns.get_mut(&req_id) {
                        Some(turn) => turn.worker = None,
                        None => continue,
                    }
                }
                match inner.router.dispatch(&request, affinity).await {
                    Ok(worker) => {
                        let mut turns = inner.turns.lock().unwrap();
                        if let Some(turn) = turns.get_mut(&req_id) {
                            turn.worker = Some(worker);
                        }
                    }
                    Err(DispatchFail) => {
                        inner.turns.lock().unwrap().remove(&req_id);
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
                Err(AdmitReject("full".into()))
            }
        }
        async fn dispatch(
            &self,
            req: &Request,
            affinity: Option<u64>,
        ) -> Result<WorkerId, DispatchFail> {
            self.dispatched.lock().unwrap().push(req.req_id);
            self.affinities.lock().unwrap().push(affinity);
            match *self.dispatch_worker.lock().unwrap() {
                Some(w) => Ok(w),
                None => Err(DispatchFail),
            }
        }
        async fn cancel(&self, worker: WorkerId, req: ReqId) {
            self.cancels.lock().unwrap().push((worker, req));
        }
        fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
            self.connected_rx.clone()
        }
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
    async fn no_worker_surfaces() {
        let router = MockRouter::new(None);
        let sessions = Sessions::new(router);
        let res = sessions.create(ident(), input(), Affinity::Sticky).await;
        assert!(matches!(res, Err(SessionError::NoWorker)));
    }
}
