//! The cluster policies' client: this gateway's processes, told to the
//! controller in order, and the controller's decisions about them, applied.
//! The gate, the order launches leave in, and where each goes stay here, on
//! the request path; what decides them runs once, in the controller.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use client_api::{ClientMessage, Props};
use controller_api::{GatewayDecisions, Outcome, PlacementView, ProcessEvent, ReplicateOrder};
use ids::{GatewayId, ReqId, WorkerId};
use tokio::sync::{Notify, mpsc, watch};
use worker_api::Request;

use crate::admission::{AdmissionDecision, RejectReason};
use crate::controller::GatewayControl;
use crate::route::RoutingHandle;
use crate::session::AdmitReject;

/// Who caused a process's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum By {
    Policy,
    Host,
    Client,
}

/// A copy asked of a worker and not yet seen done is asked again after this.
const REPLICATE_RETRY: Duration = Duration::from_secs(30);

/// The controller's decisions about this gateway's processes, as last heard.
#[derive(Default)]
struct Decided {
    active: bool,
    schedule: Option<(HashMap<u64, i64>, Option<i64>)>,
    placements: HashMap<u64, PlacementView>,
}

/// What this gateway did about the decisions, so each acts once.
#[derive(Default)]
struct Acted {
    /// Where each of its processes runs.
    running: HashMap<u64, WorkerId>,
    cancelled: HashSet<u64>,
    /// Moves asked of the sessions: (process, target).
    moves: HashSet<(u64, WorkerId)>,
    /// Copies asked of workers, and when.
    asked: HashMap<(String, WorkerId, bool), Instant>,
}

struct Shared {
    decided: RwLock<Decided>,
    /// Launches past the gate, by arrival: their request ids.
    waiting: Mutex<BTreeMap<u64, u64>>,
    /// Launches at the gate: the reason one was refused.
    refused: Mutex<HashMap<u64, Option<String>>>,
    cancels: mpsc::UnboundedSender<ReqId>,
    /// Running processes the placement moved, and where to.
    migrations: mpsc::UnboundedSender<(ReqId, WorkerId)>,
    /// The highest arrival every program has heard.
    heard: watch::Sender<u64>,
    gate_ms: AtomicU64,
    acted: Mutex<Acted>,
    changed: Notify,
    /// What a controller new to this gateway is told first: each live
    /// process's arrival, and each session's props.
    told: Mutex<Told>,
}

#[derive(Default)]
struct Told {
    workers: Vec<WorkerId>,
    arrived: BTreeMap<u64, ProcessEvent>,
    sessions: HashMap<String, Props>,
}

impl Shared {
    /// The events that bring a new controller up to date: the workers
    /// linked here, each live process's arrival, then where it runs, then
    /// each session's props.
    fn retell(&self) -> Vec<ProcessEvent> {
        let told = self.told.lock().unwrap();
        let running = self.acted.lock().unwrap().running.clone();
        let mut events = vec![ProcessEvent::Workers(told.workers.clone())];
        events.extend(told.arrived.values().cloned());
        events.extend(told.arrived.keys().filter_map(|req| {
            Some(ProcessEvent::Dispatched {
                req: *req,
                worker: *running.get(req)?,
            })
        }));
        events.extend(
            told.sessions
                .iter()
                .map(|(id, props)| ProcessEvent::SessionProps {
                    id: id.clone(),
                    changes: json_props(props),
                }),
        );
        events
    }
}

fn json_props(props: &Props) -> Vec<(String, String)> {
    props
        .iter()
        .map(|(name, value)| (name.clone(), value.to_string()))
        .collect()
}

pub struct Cluster {
    events: mpsc::UnboundedSender<ProcessEvent>,
    shared: Arc<Shared>,
    next_seq: AtomicU64,
}

/// A started client, the launches the policies cancelled, and the moves
/// they asked for: (process, the worker it is to run on).
pub type Started = (
    Arc<Cluster>,
    mpsc::UnboundedReceiver<ReqId>,
    mpsc::UnboundedReceiver<(ReqId, WorkerId)>,
);

impl Cluster {
    /// Takes part in the cluster policies through `control`, once this
    /// gateway has an id (`me`): tells them of its processes and the workers
    /// linked to it (`connected`), and applies their decisions; as the
    /// routing table's author, it carries their copies to workers
    /// (`fleet`).
    pub fn start<C: GatewayControl>(
        control: C,
        mut me: watch::Receiver<Option<GatewayId>>,
        mut connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
        fleet: crate::worker::WorkerRegistry,
    ) -> Started {
        let (cancels, cancelled) = mpsc::unbounded_channel();
        let (migrations, moves) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared {
            decided: RwLock::default(),
            waiting: Mutex::default(),
            refused: Mutex::default(),
            cancels,
            migrations,
            heard: watch::Sender::new(0),
            gate_ms: AtomicU64::new(0),
            acted: Mutex::default(),
            changed: Notify::new(),
            told: Mutex::default(),
        });
        let (events, mut rx) = mpsc::unbounded_channel::<ProcessEvent>();
        let workers = events.clone();
        let decisions_shared = shared.clone();
        let events_shared = shared.clone();
        let workers_shared = shared.clone();
        tokio::spawn(async move {
            let Ok(gateway) = me
                .wait_for(Option::is_some)
                .await
                .map(|id| id.expect("some"))
            else {
                return;
            };
            // The decisions.
            let mut decisions = control.decisions_watch(gateway);
            let shared = decisions_shared;
            tokio::spawn(async move {
                loop {
                    let view = decisions.borrow_and_update().clone();
                    apply(&shared, view, &fleet);
                    if decisions.changed().await.is_err() {
                        return;
                    }
                }
            });
            // The workers linked here.
            let told = workers_shared;
            tokio::spawn(async move {
                loop {
                    let now: Vec<WorkerId> =
                        connected.borrow_and_update().iter().copied().collect();
                    told.told.lock().unwrap().workers = now.clone();
                    if workers.send(ProcessEvent::Workers(now)).is_err()
                        || connected.changed().await.is_err()
                    {
                        return;
                    }
                }
            });
            // The events, one batch at a time, in order; a controller new to
            // this gateway hears first what the last one had.
            let mut batch = Vec::new();
            let mut generation = control.generation();
            // A quiet gateway still finds a new controller out.
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    got = rx.recv_many(&mut batch, 1024) => if got == 0 {
                        return;
                    },
                    _ = tick.tick() => {}
                }
                let mut events = std::mem::take(&mut batch);
                if control.generation() != generation {
                    generation = control.generation();
                    let mut retold = events_shared.retell();
                    tracing::info!(
                        processes = retold.len(),
                        "telling a new controller this gateway's processes"
                    );
                    retold.append(&mut events);
                    events = retold;
                }
                if events.is_empty() {
                    continue;
                }
                if let Err(error) = control.cluster_events(gateway, events).await {
                    tracing::warn!(error = %format!("{error:#}"), "cluster policy events not delivered");
                }
            }
        });
        (
            Arc::new(Self {
                events,
                shared,
                next_seq: AtomicU64::new(0),
            }),
            cancelled,
            moves,
        )
    }

    /// Whether cluster policies run: only then does this gateway's gate and
    /// placement follow them.
    pub fn active(&self) -> bool {
        self.shared.decided.read().unwrap().active
    }

    /// Tells the policies of `event`, while any run.
    fn send(&self, event: ProcessEvent) {
        if self.active() {
            let _ = self.events.send(event);
        }
    }

    pub async fn admit(&self, req: &Request, routing: &RoutingHandle) -> Result<(), AdmitReject> {
        let id = req.req_id.0;
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let (props, session) = match &req.message {
            ClientMessage::LaunchProcess { props, session, .. } => (props.clone(), session.clone()),
            _ => Default::default(),
        };
        // The prompt goes as the blocks its text hashes to.
        let prompt =
            client_api::props::prompt_of(props.iter()).map(client_api::props::prompt_blocks);
        let props = props
            .iter()
            .filter(|(name, _)| name.as_str() != client_api::props::PROMPT)
            .map(|(name, value)| (name.clone(), value.to_string()))
            .collect();
        self.shared.refused.lock().unwrap().insert(id, None);
        let arrived = ProcessEvent::Arrived {
            req: id,
            seq,
            tenant: req.tenant.0.clone(),
            session,
            props,
            prompt,
        };
        if self.active() {
            self.shared
                .told
                .lock()
                .unwrap()
                .arrived
                .insert(id, arrived.clone());
        }
        self.send(arrived);
        let admitted = {
            let mut waiting = Waiting {
                router: self,
                seq,
                id,
                left: false,
            };
            self.pass(&mut waiting, req, routing).await
        };
        if admitted.is_err() {
            self.shared.told.lock().unwrap().arrived.remove(&id);
            self.send(ProcessEvent::Exited {
                req: id,
                outcome: Outcome::Cancelled,
            });
        }
        admitted
    }

    async fn pass(
        &self,
        waiting: &mut Waiting<'_>,
        req: &Request,
        routing: &RoutingHandle,
    ) -> Result<(), AdmitReject> {
        let (id, seq) = (waiting.id, waiting.seq);
        let gate = Duration::from_millis(self.shared.gate_ms.load(Ordering::Acquire));
        if !gate.is_zero() {
            // Unanswered, the launch goes by the gateway's default.
            let mut heard = self.shared.heard.subscribe();
            let _ = tokio::time::timeout(gate, heard.wait_for(|&h| h >= seq)).await;
        }

        {
            let mut queue = self.shared.waiting.lock().unwrap();
            if queue.len() >= routing.max_queued_launches() {
                drop(queue);
                return waiting.leave(Err(AdmitReject::Full(RejectReason::QueueFull.to_string())));
            }
            queue.insert(seq, id);
        }
        let mut table = routing.table_watch();
        loop {
            let changed = self.shared.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if matches!(self.shared.refused.lock().unwrap().get(&id), Some(Some(_))) {
                return waiting.leave(Ok(()));
            }
            if self.next(seq) {
                match routing.admit(req) {
                    AdmissionDecision::Admit => return waiting.leave(Ok(())),
                    AdmissionDecision::Reject(RejectReason::ClusterSaturated) => {}
                    AdmissionDecision::Reject(reason) => {
                        return waiting.leave(Err(AdmitReject::Full(reason.to_string())));
                    }
                }
            }
            tokio::select! {
                _ = changed => {}
                closed = table.changed() => if closed.is_err() {
                    return waiting.leave(Err(AdmitReject::Full(
                        RejectReason::ClusterSaturated.to_string(),
                    )));
                },
            }
        }
    }

    /// The highest key the schedule does not hold, the earliest among equals.
    fn next(&self, seq: u64) -> bool {
        let decided = self.shared.decided.read().unwrap();
        let schedule = decided.schedule.as_ref();
        let key = |id: u64| schedule.map_or(0, |(keys, _)| keys.get(&id).copied().unwrap_or(0));
        let held = |id: u64| schedule.is_some_and(|(_, floor)| floor.is_some_and(|f| key(id) < f));
        self.shared
            .waiting
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, id)| !held(**id))
            .max_by_key(|(s, id)| (key(**id), Reverse(**s)))
            .is_some_and(|(s, _)| *s == seq)
    }

    /// Where the placement puts `req`: its worker, its props, and the
    /// workers its passes may run on, best first.
    pub fn placement(&self, req: &Request) -> (Option<WorkerId>, Props, Vec<WorkerId>) {
        let decided = self.shared.decided.read().unwrap();
        let Some(placement) = decided.placements.get(&req.req_id.0) else {
            return (None, Props::new(), Vec::new());
        };
        let props = placement
            .props
            .iter()
            .filter_map(|(name, json)| Some((name.clone(), serde_json::from_str(json).ok()?)))
            .collect();
        (placement.worker, props, placement.executors.clone())
    }

    pub fn dispatched(&self, req: &Request, worker: WorkerId) {
        self.shared
            .acted
            .lock()
            .unwrap()
            .running
            .insert(req.req_id.0, worker);
        move_misplaced(&self.shared);
        self.send(ProcessEvent::Dispatched {
            req: req.req_id.0,
            worker,
        });
    }

    pub fn counted(&self, req: &Request, prompt: u64, generated: u64) {
        self.send(ProcessEvent::Counted {
            req: req.req_id.0,
            prompt,
            generated,
        });
    }

    pub fn session_props(&self, id: String, changes: Props) {
        if self.active() {
            let mut told = self.shared.told.lock().unwrap();
            let current = told.sessions.remove(&id).unwrap_or_default();
            if let Ok((merged, _)) = client_api::props::merge(&current, changes.clone())
                && !merged.is_empty()
            {
                told.sessions.insert(id.clone(), merged);
            }
        }
        self.send(ProcessEvent::SessionProps {
            id,
            changes: json_props(&changes),
        });
    }

    pub fn session_closed(&self, id: String) {
        self.shared.told.lock().unwrap().sessions.remove(&id);
        self.send(ProcessEvent::SessionClosed { id });
    }

    /// A running process moved to `worker`, or (`None`) could not move.
    pub fn moved(&self, req: &Request, worker: Option<WorkerId>, why: &str) {
        if let Some(worker) = worker {
            self.shared
                .acted
                .lock()
                .unwrap()
                .running
                .insert(req.req_id.0, worker);
            move_misplaced(&self.shared);
        }
        self.send(ProcessEvent::Moved {
            req: req.req_id.0,
            worker,
            why: why.to_string(),
        });
    }

    pub fn exited(&self, req: &Request, outcome: Outcome) {
        let id = req.req_id.0;
        {
            let mut acted = self.shared.acted.lock().unwrap();
            acted.running.remove(&id);
            acted.cancelled.remove(&id);
            acted.moves.retain(|(of, _)| *of != id);
        }
        self.shared.told.lock().unwrap().arrived.remove(&id);
        self.send(ProcessEvent::Exited { req: id, outcome });
    }
}

/// Applies the controller's view: the gate and the placement as it says;
/// a process it ended refused at the gate or cancelled once admitted; a
/// running process it placed elsewhere moved, once per target; and, for the
/// gateway that carries them, its copies asked of workers.
fn apply(shared: &Shared, view: GatewayDecisions, fleet: &crate::worker::WorkerRegistry) {
    shared.gate_ms.store(view.gate_ms, Ordering::Release);
    {
        let mut decided = shared.decided.write().unwrap();
        decided.active = view.active;
        decided.schedule = view
            .schedule
            .map(|schedule| (schedule.keys.into_iter().collect(), schedule.floor));
        decided.placements = view.placements.into_iter().collect();
    }
    let mut acted = shared.acted.lock().unwrap();
    {
        let mut refused = shared.refused.lock().unwrap();
        for (id, reason) in view.terminated {
            if let Some(at_gate) = refused.get_mut(&id) {
                at_gate.get_or_insert(reason);
            } else if acted.cancelled.insert(id) {
                let _ = shared.cancels.send(ReqId(id));
            }
        }
    }
    carry(&mut acted.asked, view.replicate, fleet);
    drop(acted);
    move_misplaced(shared);
    shared.heard.send_replace(view.heard);
    shared.changed.notify_waiters();
}

/// Moves each running process its placement names another worker for,
/// once per target: whenever the placement or where it runs changes.
fn move_misplaced(shared: &Shared) {
    let decided = shared.decided.read().unwrap();
    let mut acted = shared.acted.lock().unwrap();
    let moves: Vec<(u64, WorkerId)> = acted
        .running
        .iter()
        .filter_map(|(id, at)| {
            let target = decided.placements.get(id)?.worker?;
            (target != *at).then_some((*id, target))
        })
        .collect();
    for (id, target) in moves {
        if acted.moves.insert((id, target)) {
            let _ = shared.migrations.send((ReqId(id), target));
        }
    }
}

/// Asks each worker for the copies `orders` names, again after
/// [`REPLICATE_RETRY`] while the order stands.
fn carry(
    asked: &mut HashMap<(String, WorkerId, bool), Instant>,
    orders: Vec<ReplicateOrder>,
    fleet: &crate::worker::WorkerRegistry,
) {
    let now = Instant::now();
    asked.retain(|_, at| now.duration_since(*at) < REPLICATE_RETRY);
    for order in orders {
        let key = (order.group.clone(), order.worker, order.disk);
        if asked.contains_key(&key) {
            continue;
        }
        let Some(client) = fleet.client(order.worker) else {
            continue;
        };
        asked.insert(key, now);
        tokio::spawn(async move {
            let mut ctx = tarpc::context::current();
            ctx.deadline = Instant::now() + Duration::from_secs(120);
            let ReplicateOrder {
                group,
                worker,
                keys,
                from,
                retention,
                disk,
                table,
            } = order;
            match client.replicate(ctx, keys, from, retention, disk).await {
                Ok(Ok(pages)) => tracing::info!(%group, %worker, table, pages, disk, "replicated"),
                Ok(Err(why)) => {
                    tracing::warn!(%group, %worker, table, %why, disk, "replication refused")
                }
                Err(error) => {
                    tracing::warn!(%group, %worker, table, %error, disk, "replication not delivered")
                }
            }
        });
    }
}

struct Waiting<'a> {
    router: &'a Cluster,
    seq: u64,
    id: u64,
    left: bool,
}

impl Waiting<'_> {
    fn leave(&mut self, outcome: Result<(), AdmitReject>) -> Result<(), AdmitReject> {
        self.left = true;
        let shared = &self.router.shared;
        let refused = {
            let mut waiting = shared.waiting.lock().unwrap();
            waiting.remove(&self.seq);
            shared.refused.lock().unwrap().remove(&self.id).flatten()
        };
        shared.changed.notify_waiters();
        refused.map_or(outcome, |reason| Err(AdmitReject::Refused(reason)))
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        if self.left {
            return;
        }
        let _ = self.leave(Ok(()));
        self.router
            .shared
            .told
            .lock()
            .unwrap()
            .arrived
            .remove(&self.id);
        self.router.send(ProcessEvent::Exited {
            req: self.id,
            outcome: Outcome::Cancelled,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use controller::cluster::{Host, LoadError};
    use controller_api::{
        Ack, DirectoryView, GatewayInfo, Health, Role, RoutableWorker, RoutingTable, ScaleEntry,
        StateReport, WorkerStatus,
    };
    use ids::{NodeId, SessionId, TenantId};
    use worker_api::Priority;

    use super::*;

    fn table(kv: u8) -> RoutingTable {
        RoutingTable {
            epoch: 1,
            scales: false,
            author: None,
            workers: vec![RoutableWorker {
                id: WorkerId(1),
                addr: "w1".into(),
                role: Role::Decode,
                model: "m".into(),
                engine: String::new(),
                health: Health::Healthy,
                coarse_load: WorkerStatus {
                    kv_pressure_bucket: kv,
                    inflight: 0,
                    running: 2,
                    waiting: 5,
                    capacity_tokens: 1000,
                    used_tokens: 250,
                    draining: false,
                },
            }],
        }
    }

    fn launch(req: u64, props: serde_json::Value) -> Request {
        Request {
            req_id: ReqId(req),
            session: SessionId(1),
            tenant: TenantId("t".into()),
            user: "u".into(),
            priority: Priority::Normal,
            blobs: Vec::new(),
            message: ClientMessage::LaunchProcess {
                corr_id: 1,
                inferlet: "chat".into(),
                input: String::new(),
                capture_outputs: true,
                props: serde_json::from_value(props).unwrap(),
                session: None,
                files: Default::default(),
                cluster_props: Props::new(),
                executors: Vec::new(),
            },
            cluster_props: Props::new(),
            executors: Vec::new(),
        }
    }

    /// A controller of the cluster policies alone, in process.
    #[derive(Clone)]
    struct HostControl(Arc<controller::cluster::Host>);

    impl GatewayControl for HostControl {
        async fn register_gateway(&self, _info: GatewayInfo) -> anyhow::Result<GatewayId> {
            Ok(GatewayId(0))
        }

        async fn heartbeat(&self, _id: NodeId) -> anyhow::Result<Ack> {
            Ok(Ack::Ok)
        }

        fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
            watch::channel(table(0)).1
        }

        fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
            watch::channel(DirectoryView::default()).1
        }

        async fn report_state(
            &self,
            _worker: WorkerId,
            _report: StateReport,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn forget_state(&self, _worker: WorkerId) -> anyhow::Result<()> {
            Ok(())
        }

        async fn cluster_events(
            &self,
            gateway: GatewayId,
            events: Vec<ProcessEvent>,
        ) -> anyhow::Result<()> {
            self.0.events(gateway, events);
            Ok(())
        }

        fn decisions_watch(&self, gateway: GatewayId) -> watch::Receiver<GatewayDecisions> {
            self.0.decisions(gateway)
        }
    }

    struct Rig {
        router: Arc<Cluster>,
        host: Arc<Host>,
        routing: RoutingHandle,
        table: watch::Sender<RoutingTable>,
        cancelled: mpsc::UnboundedReceiver<ReqId>,
        moves: mpsc::UnboundedReceiver<(ReqId, WorkerId)>,
        _connected: watch::Sender<Arc<HashSet<WorkerId>>>,
        _me: watch::Sender<Option<GatewayId>>,
    }

    fn programs(names: &[&str]) -> Vec<(String, Vec<u8>)> {
        names
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    builtins::policy(name).unwrap().component.to_vec(),
                )
            })
            .collect()
    }

    async fn rig(kv: u8) -> Rig {
        rig_of(kv, &["test-route"]).await
    }

    /// A gateway, its policies `names` running in a controller beside it,
    /// once the gateway has heard they run.
    async fn rig_of(kv: u8, names: &[&str]) -> Rig {
        let (table, table_rx) = watch::channel(table(kv));
        let (connected, connected_rx) = watch::channel(Arc::new(HashSet::from([WorkerId(1)])));
        let routing = RoutingHandle::new(table_rx.clone(), connected_rx.clone());
        let host = Host::start(
            programs(names),
            table_rx,
            watch::channel(DirectoryView::default()).1,
        )
        .unwrap();
        let (me, me_rx) = watch::channel(Some(GatewayId(0)));
        let (router, cancelled, moves) = Cluster::start(
            HostControl(host.clone()),
            me_rx,
            connected_rx,
            crate::worker::WorkerRegistry::new(),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while !router.active() {
            assert!(
                Instant::now() < deadline,
                "the gateway never heard the policies run"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Rig {
            router,
            host,
            routing,
            table,
            cancelled,
            moves,
            _connected: connected,
            _me: me,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_reload_swaps_the_programs_and_withdraws_the_old_ones_tables() {
        let rig = rig(0).await;
        let placed = launch(1, serde_json::json!({"key": 1, "worker": "1"}));
        rig.router.admit(&placed, &rig.routing).await.unwrap();
        assert_eq!(rig.router.placement(&placed).0, Some(WorkerId(1)));
        // Below test-route's floor, a launch waits.
        let held = launch(2, serde_json::json!({"key": 0}));
        let waiting = {
            let (router, routing) = (rig.router.clone(), rig.routing.clone());
            tokio::spawn(async move { router.admit(&held, &routing).await })
        };
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!waiting.is_finished(), "held below the floor");

        rig.host.reload(programs(&["replicate"])).await.unwrap();
        // test-route's schedule and placement are withdrawn with it: the
        // held launch leaves, and nothing places the first any more.
        tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("let through once the floor is withdrawn")
            .unwrap()
            .unwrap();
        assert_eq!(rig.router.placement(&placed).0, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_reload_that_does_not_load_leaves_the_running_set() {
        let rig = rig(0).await;
        let mut broken = programs(&["replicate"]);
        broken.push(("garbled".to_string(), vec![0, 1, 2, 3]));
        assert!(matches!(
            rig.host.reload(broken).await,
            Err(LoadError::Of(name, _)) if name == "garbled"
        ));
        assert!(matches!(
            rig.host
                .reload(programs(&["test-route", "cache-aware"]))
                .await,
            Err(LoadError::Shared { .. })
        ));
        // test-route still places.
        let placed = launch(1, serde_json::json!({"key": 1, "worker": "1"}));
        rig.router.admit(&placed, &rig.routing).await.unwrap();
        assert_eq!(rig.router.placement(&placed).0, Some(WorkerId(1)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_reload_tells_the_new_programs_what_the_old_ones_withdrew() {
        // test-share publishes the placement test-route made; test-route
        // names it as the process's executors.
        let rig = rig_of(0, &["test-share", "test-route"]).await;
        let placed = launch(1, serde_json::json!({"key": 1, "worker": "1"}));
        rig.router.admit(&placed, &rig.routing).await.unwrap();
        let until = |want: &[WorkerId]| {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            let (router, want) = (rig.router.clone(), want.to_vec());
            let placed = placed.clone();
            async move {
                loop {
                    let (worker, _, executors) = router.placement(&placed);
                    if executors == want && worker == Some(WorkerId(1)) {
                        return;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "executors {executors:?}, wanted {want:?}"
                    );
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        };
        until(&[WorkerId(1)]).await;
        // Without test-share, its value is withdrawn; the new test-route,
        // told so, places the process again and names no executors.
        rig.host.reload(programs(&["test-route"])).await.unwrap();
        until(&[]).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_launch_is_held_below_the_floor_and_placed_where_its_policy_says() {
        let rig = rig(0).await;
        let held = launch(
            1,
            serde_json::json!({"key": 0, "worker": "1", "executors": "3,2"}),
        );
        let free = launch(2, serde_json::json!({}));

        let waiting = tokio::spawn({
            let (router, routing, held) = (rig.router.clone(), rig.routing.clone(), held.clone());
            async move { router.admit(&held, &routing).await }
        });
        rig.router.admit(&free, &rig.routing).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!waiting.is_finished(), "key 0 is below the floor of 1");

        let (worker, props, executors) = rig.router.placement(&held);
        assert_eq!(worker, Some(WorkerId(1)));
        assert_eq!(
            props["from"],
            serde_json::json!("0.1"),
            "the policies name it by its gateway and its request"
        );
        assert_eq!(
            executors,
            [WorkerId(3), WorkerId(2)],
            "its passes' workers, best first"
        );
        assert_eq!(rig.router.placement(&free).0, None, "the router's default");
        waiting.abort();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_policy_beside_another_places_what_it_places() {
        let rig = rig_of(0, &["replicate", "test-route"]).await;
        let placed = launch(1, serde_json::json!({"key": 1, "worker": "1"}));
        rig.router.admit(&placed, &rig.routing).await.unwrap();
        assert_eq!(rig.router.placement(&placed).0, Some(WorkerId(1)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_scale_table_comes_only_from_the_policy_granted_it() {
        let ask = serde_json::json!({
            "key": 1,
            "scale-workers": 2,
            "scale-engine": "cuda",
            "scale-model": "m",
        });
        let granted = rig_of(0, &["scale-by-prop", "test-route"]).await;
        let mut scale = granted.host.scale_watch();
        granted
            .router
            .admit(&launch(1, ask.clone()), &granted.routing)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), scale.wait_for(Option::is_some))
            .await
            .expect("the scale table is installed")
            .unwrap();
        assert_eq!(
            scale.borrow().as_deref(),
            Some(
                &[ScaleEntry {
                    model: Some("m".into()),
                    engine: "cuda".into(),
                    workers: 2,
                }][..]
            )
        );

        // test-route installs one too, and holds no `scale` grant.
        let ungranted = rig_of(0, &["test-route"]).await;
        let scale = ungranted.host.scale_watch();
        ungranted
            .router
            .admit(&launch(1, ask), &ungranted.routing)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            scale.borrow().is_none(),
            "refused: a program granted no `scale` speaks for no size"
        );
    }

    #[tokio::test]
    async fn waiting_launches_leave_in_key_order_when_room_opens() {
        let rig = rig(255).await;
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut waits = Vec::new();
        for (req, key) in [(1, 1), (2, 5), (3, 3)] {
            let (router, routing, order) = (rig.router.clone(), rig.routing.clone(), order.clone());
            waits.push(tokio::spawn(async move {
                router
                    .admit(&launch(req, serde_json::json!({ "key": key })), &routing)
                    .await
                    .unwrap();
                order.lock().unwrap().push(req);
            }));
            // Each is heard (the gate is 50 ms) before the next arrives.
            tokio::time::sleep(Duration::from_millis(60)).await;
        }
        assert!(order.lock().unwrap().is_empty(), "the cluster is saturated");
        rig.table.send_replace(table(0));
        for wait in waits {
            wait.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), [2, 3, 1]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_terminated_launch_is_refused_while_it_waits_and_cancelled_once_it_runs() {
        let mut rig = rig(0).await;
        let refused = launch(1, serde_json::json!({"refuse": "not today"}));
        match rig.router.admit(&refused, &rig.routing).await {
            Err(AdmitReject::Refused(why)) => assert_eq!(why, "not today"),
            other => panic!("refused while it waits, got {other:?}"),
        }

        let stopped = launch(2, serde_json::json!({"stop": true}));
        rig.router.admit(&stopped, &rig.routing).await.unwrap();
        rig.router.dispatched(&stopped, WorkerId(1));
        let cancelled = tokio::time::timeout(Duration::from_secs(5), rig.cancelled.recv())
            .await
            .expect("the gateway is told to cancel it");
        assert_eq!(cancelled, Some(ReqId(2)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_program_answers_another_s_placement_and_that_one_acts_on_the_answer() {
        // test-route places the process on worker 1; test-share, told the
        // placement changed, publishes it; test-route, told of that, names
        // what it read as the process's executors.
        let rig = rig_of(0, &["test-share", "test-route"]).await;
        let placed = launch(1, serde_json::json!({"key": 1, "worker": "1"}));
        rig.router.admit(&placed, &rig.routing).await.unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let (worker, _, executors) = rig.router.placement(&placed);
            if executors == [WorkerId(1)] {
                assert_eq!(worker, Some(WorkerId(1)));
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "executors never followed the published placement: {executors:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_running_process_placed_elsewhere_moves_there_once() {
        let mut rig = rig_of(0, &["relocate"]).await;
        let placed = launch(
            1,
            serde_json::json!({"worker": 1, "move-to": 2, "move-after-ms": 0}),
        );
        rig.router.admit(&placed, &rig.routing).await.unwrap();
        rig.router.dispatched(&placed, WorkerId(1));
        let moved = tokio::time::timeout(Duration::from_secs(5), rig.moves.recv())
            .await
            .expect("the gateway is told to move it");
        assert_eq!(moved, Some((ReqId(1), WorkerId(2))));
        assert!(
            tokio::time::timeout(Duration::from_millis(300), rig.moves.recv())
                .await
                .is_err(),
            "asked once while the placement stands"
        );
        rig.router.moved(&placed, Some(WorkerId(2)), "");
        rig.router.exited(&placed, Outcome::Completed);
    }

    /// A controller that records what it hears, and can be swapped for a
    /// new one.
    #[derive(Clone, Default)]
    struct Recorder {
        heard: Arc<Mutex<Vec<ProcessEvent>>>,
        generation: Arc<AtomicU64>,
    }

    impl GatewayControl for Recorder {
        async fn register_gateway(&self, _info: GatewayInfo) -> anyhow::Result<GatewayId> {
            Ok(GatewayId(0))
        }
        async fn heartbeat(&self, _id: NodeId) -> anyhow::Result<Ack> {
            Ok(Ack::Ok)
        }
        fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
            watch::channel(table(0)).1
        }
        fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
            watch::channel(DirectoryView::default()).1
        }
        async fn report_state(&self, _w: WorkerId, _r: StateReport) -> anyhow::Result<()> {
            Ok(())
        }
        async fn forget_state(&self, _w: WorkerId) -> anyhow::Result<()> {
            Ok(())
        }
        async fn cluster_events(&self, _g: GatewayId, e: Vec<ProcessEvent>) -> anyhow::Result<()> {
            self.heard.lock().unwrap().extend(e);
            Ok(())
        }
        fn decisions_watch(&self, _g: GatewayId) -> watch::Receiver<GatewayDecisions> {
            watch::channel(GatewayDecisions {
                epoch: 1,
                active: true,
                heard: u64::MAX,
                ..GatewayDecisions::default()
            })
            .1
        }
        fn generation(&self) -> u64 {
            self.generation.load(Ordering::Acquire)
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_new_controller_hears_the_live_processes_and_sessions_first() {
        let recorder = Recorder::default();
        let (_table, table_rx) = watch::channel(table(0));
        let (_connected, connected_rx) = watch::channel(Arc::new(HashSet::from([WorkerId(1)])));
        let routing = RoutingHandle::new(table_rx, connected_rx.clone());
        let (_me, me_rx) = watch::channel(Some(GatewayId(0)));
        let (router, _, _) = Cluster::start(
            recorder.clone(),
            me_rx,
            connected_rx,
            crate::worker::WorkerRegistry::new(),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while !router.active() {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let (live, gone) = (
            launch(1, serde_json::json!({"key": 1})),
            launch(2, serde_json::json!({})),
        );
        router.admit(&live, &routing).await.unwrap();
        router.dispatched(&live, WorkerId(1));
        router.admit(&gone, &routing).await.unwrap();
        router.exited(&gone, Outcome::Completed);
        let changes: Props = serde_json::from_value(serde_json::json!({"eta": 9})).unwrap();
        router.session_props("t/s".into(), changes);
        tokio::time::sleep(Duration::from_millis(1200)).await;
        recorder.heard.lock().unwrap().clear();

        // The controller is dialed again: a new one.
        recorder.generation.fetch_add(1, Ordering::AcqRel);
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let heard = std::mem::take(&mut *recorder.heard.lock().unwrap());
        let kinds: Vec<String> = heard
            .iter()
            .filter(|event| !matches!(event, ProcessEvent::Workers(_)))
            .map(|event| match event {
                ProcessEvent::Arrived { req, .. } => format!("arrived {req}"),
                ProcessEvent::Dispatched { req, worker } => format!("dispatched {req} {worker}"),
                ProcessEvent::SessionProps { id, .. } => format!("session {id}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            kinds,
            ["arrived 1", "dispatched 1 worker#1", "session t/s"],
            "the live process, where it runs, and the session; not the one that exited"
        );
    }

    /// A controller that can die and be replaced: while down, nothing it is
    /// told is heard and it decides nothing; each one up is new, dialed
    /// afresh.
    #[derive(Clone, Default)]
    struct Replaced {
        up: Arc<Mutex<Option<Arc<Host>>>>,
        generation: Arc<AtomicU64>,
    }

    impl Replaced {
        fn down(&self) {
            *self.up.lock().unwrap() = None;
        }

        fn up(&self, host: Arc<Host>) {
            *self.up.lock().unwrap() = Some(host);
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    impl GatewayControl for Replaced {
        async fn register_gateway(&self, _info: GatewayInfo) -> anyhow::Result<GatewayId> {
            Ok(GatewayId(0))
        }
        async fn heartbeat(&self, _id: NodeId) -> anyhow::Result<Ack> {
            Ok(Ack::Ok)
        }
        fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
            watch::channel(table(0)).1
        }
        fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
            watch::channel(DirectoryView::default()).1
        }
        async fn report_state(&self, _w: WorkerId, _r: StateReport) -> anyhow::Result<()> {
            Ok(())
        }
        async fn forget_state(&self, _w: WorkerId) -> anyhow::Result<()> {
            Ok(())
        }
        async fn cluster_events(&self, g: GatewayId, e: Vec<ProcessEvent>) -> anyhow::Result<()> {
            let host = self.up.lock().unwrap().clone();
            let host = host.ok_or_else(|| anyhow::anyhow!("no controller"))?;
            host.events(g, e);
            Ok(())
        }
        /// Follows whichever controller is up, as the healing client does.
        fn decisions_watch(&self, g: GatewayId) -> watch::Receiver<GatewayDecisions> {
            let (tx, rx) = watch::channel(GatewayDecisions::default());
            let this = self.clone();
            tokio::spawn(async move {
                loop {
                    let generation = this.generation();
                    let host = this.up.lock().unwrap().clone();
                    let Some(host) = host else {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        continue;
                    };
                    let mut decisions = host.decisions(g);
                    decisions.mark_changed();
                    while this.generation() == generation && this.up.lock().unwrap().is_some() {
                        if decisions.has_changed().unwrap_or(false)
                            && tx.send(decisions.borrow_and_update().clone()).is_err()
                        {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                }
            });
            rx
        }
        fn generation(&self) -> u64 {
            self.generation.load(Ordering::Acquire)
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_controller_that_replaces_a_dead_one_takes_up_its_running_processes() {
        let (_table, table_rx) = watch::channel(table(0));
        let (_connected, connected_rx) = watch::channel(Arc::new(HashSet::from([WorkerId(1)])));
        let routing = RoutingHandle::new(table_rx.clone(), connected_rx.clone());
        let start = || {
            Host::start(
                programs(&["relocate"]),
                table_rx.clone(),
                watch::channel(DirectoryView::default()).1,
            )
            .unwrap()
        };
        let control = Replaced::default();
        control.up(start());
        let (_me, me_rx) = watch::channel(Some(GatewayId(0)));
        let (router, _, mut moves) = Cluster::start(
            control.clone(),
            me_rx,
            connected_rx,
            crate::worker::WorkerRegistry::new(),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while !router.active() {
            assert!(
                Instant::now() < deadline,
                "the gateway never heard the policies run"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let placed = launch(
            1,
            serde_json::json!({"worker": 1, "move-to": 2, "move-after-ms": 600}),
        );
        rig_admit(&router, &placed, &routing).await;
        router.dispatched(&placed, WorkerId(1));
        let deadline = Instant::now() + Duration::from_secs(5);
        while router.placement(&placed).0.is_none() {
            assert!(Instant::now() < deadline, "the placement never arrived");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(router.placement(&placed).0, Some(WorkerId(1)));

        // The controller dies before the move is due, and stays down past it.
        control.down();
        assert!(
            tokio::time::timeout(Duration::from_millis(900), moves.recv())
                .await
                .is_err(),
            "no controller, no move"
        );
        // While it is down, the gateway keeps to what it last heard.
        assert!(router.active());
        assert_eq!(router.placement(&placed).0, Some(WorkerId(1)));

        // A new one hears the process from the gateway and, its time come
        // again, moves it.
        control.up(start());
        let moved = tokio::time::timeout(Duration::from_secs(5), moves.recv())
            .await
            .expect("the new controller moves it");
        assert_eq!(moved, Some((ReqId(1), WorkerId(2))));
        router.moved(&placed, Some(WorkerId(2)), "");
        router.exited(&placed, Outcome::Completed);
    }

    async fn rig_admit(router: &Cluster, req: &Request, routing: &RoutingHandle) {
        tokio::time::timeout(Duration::from_secs(5), router.admit(req, routing))
            .await
            .expect("admitted in time")
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn with_no_policy_running_the_gateway_goes_by_its_defaults() {
        let (_table, table_rx) = watch::channel(table(0));
        let (_connected, connected_rx) = watch::channel(Arc::new(HashSet::new()));
        let controller = controller::embed(controller::Config::default());
        let (_me, me_rx) = watch::channel(Some(GatewayId(0)));
        #[derive(Clone)]
        struct Idle(controller::Handle);
        impl GatewayControl for Idle {
            async fn register_gateway(&self, _info: GatewayInfo) -> anyhow::Result<GatewayId> {
                Ok(GatewayId(0))
            }
            async fn heartbeat(&self, _id: NodeId) -> anyhow::Result<Ack> {
                Ok(Ack::Ok)
            }
            fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
                watch::channel(table(0)).1
            }
            fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
                self.0.directory_watch()
            }
            async fn report_state(&self, _w: WorkerId, _r: StateReport) -> anyhow::Result<()> {
                Ok(())
            }
            async fn forget_state(&self, _w: WorkerId) -> anyhow::Result<()> {
                Ok(())
            }
            async fn cluster_events(
                &self,
                g: GatewayId,
                e: Vec<ProcessEvent>,
            ) -> anyhow::Result<()> {
                self.0.cluster_events(g, e);
                Ok(())
            }
            fn decisions_watch(&self, g: GatewayId) -> watch::Receiver<GatewayDecisions> {
                self.0.decisions_watch(g)
            }
        }
        let (router, _, _) = Cluster::start(
            Idle(controller.clone()),
            me_rx,
            connected_rx,
            crate::worker::WorkerRegistry::new(),
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!router.active(), "nothing to tell, nothing to follow");
        let _ = table_rx;
        controller.shutdown().await;
    }
}
