use crate::rt::Instant;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ::engine::ChannelRegistration;

use crate::engine::{
    BoundInstance, ChannelJoin, EngineBox, RegisteredChannel, SubmissionCompletion,
};
use crate::scheduler::ProcessId;
use anyhow::{Result, anyhow};

use super::stats::SchedulerStats;
use crate::store::roles::Pump;

use super::worker::{BindRespond, HostKvOp, PoolOp, PreLaunchCopy, QueuedItem};

pub(super) type LaunchResult = std::result::Result<SubmissionCompletion, String>;

type LaunchReplyTx = tokio::sync::oneshot::Sender<LaunchResult>;

pub(super) type LaunchReplyRx = tokio::sync::oneshot::Receiver<LaunchResult>;

type ControlReplyTx = tokio::sync::oneshot::Sender<LaneCommit>;

pub(super) type ControlReplyRx = tokio::sync::oneshot::Receiver<LaneCommit>;

#[derive(Clone, Copy)]
enum LaneWork {
    Launch,
    Prefill,
    Control,
}

struct LaneCharge<'a> {
    stats: &'a SchedulerStats,
    began: Instant,
    work: Option<LaneWork>,
}

impl Drop for LaneCharge<'_> {
    fn drop(&mut self) {
        let Some(work) = self.work else {
            return;
        };
        use std::sync::atomic::Ordering::Relaxed;
        let us = self.began.elapsed().as_micros() as u64;
        let q = &self.stats.fire.quorum;
        match work {
            LaneWork::Control => {
                q.lane_control_us.fetch_add(us, Relaxed);
                q.lane_control_n.fetch_add(1, Relaxed);
                q.lane_control_max_us.fetch_max(us, Relaxed);
            }
            LaneWork::Prefill => {
                q.lane_prefill_us.fetch_add(us, Relaxed);
                q.lane_prefill_n.fetch_add(1, Relaxed);
            }
            LaneWork::Launch => {
                q.lane_launch_us.fetch_add(us, Relaxed);
                q.lane_launch_n.fetch_add(1, Relaxed);
            }
        }
    }
}

enum LaneRequest {
    Launch {
        submission: crate::engine::FrameFire,
        prefill: bool,
        reply: LaunchReplyTx,
    },
    Control {
        item: Box<QueuedItem>,
        reply: ControlReplyTx,
    },
    Shutdown {
        response: crate::rt::channel::Sender<(Option<EngineBox>, ChannelJoin)>,
    },
    Landed {
        frame: ::engine::FrameId,
        step: u32,
        outcome: ::engine::StepOutcome,
    },
}

pub(super) enum LaneCommit {
    None,
    BindInstance {
        pipeline_id: Option<ProcessId>,
        bound: BoundInstance,
        respond: BindRespond,
    },
    BindFinished {
        pipeline_id: Option<ProcessId>,
        /// The bind found no room for its guest state and registered
        /// nothing: its process waits for room
        /// ([`crate::store::roles::bind_when_resident`]), off the frame.
        waits: bool,
    },
    CloseInstance {
        id: ::engine::InstanceId,
    },
    AsyncControl {
        result: std::result::Result<SubmissionCompletion, String>,
    },
}

/// The lanes a frame settled inside `submit` faulted, step by step, as its
/// engine told them before `submit` returned.
type InlineFaults = Arc<std::sync::Mutex<Vec<(usize, Vec<::engine::LaneFault>)>>>;

const LANE_PANICKED: &str = "the engine lane panicked serving this request";
const NO_BACKEND: &str = "engine has no backend installed";

impl LaneCommit {
    fn refused(message: impl Into<String>) -> Self {
        Self::AsyncControl {
            result: Err(message.into()),
        }
    }
}

/// The instance and channel slots of engine `engine_idx`'s pool, where it
/// states their shapes (the serving model's stores, model 0).
/// A step on its way through the engine: the instances it attaches, and
/// what they name in the roles, read until it lands
/// ([`crate::store::roles::Roles::reading`]).
/// A fired step's attachments and what they name in the roles, until its
/// landing takes them (`None` once landed: steps land in any order).
type LandingReads = Option<(Vec<::engine::InstanceId>, Vec<crate::store::roles::Owner>)>;

/// Takes step `step` of `frame`'s reads at its landing, and the frame's
/// entry once every step of it has landed, in whatever order they land.
fn take_landing(
    landed: &mut HashMap<::engine::FrameId, Vec<LandingReads>>,
    frame: ::engine::FrameId,
    step: usize,
) -> LandingReads {
    let steps = landed.get_mut(&frame)?;
    let taken = steps.get_mut(step).and_then(Option::take);
    if steps.iter().all(Option::is_none) {
        landed.remove(&frame);
    }
    taken
}

/// Closes instance `id`'s books on the lane once the engine answered its
/// close (`closed`, the standing guest bytes it states after): its channels
/// unbound, its role items closed (its slots go back once no step on its
/// way names them, and a pair, channel or adapter it shares is held by it
/// no more), and the pool told. A close the engine refused (an instance a
/// failed fire left unusable) is closed all the same: in the engine's eyes
/// no fire of it can run again. Kept, a failed process's items stayed on
/// the device and in its pair for good (Mac Pro's M5: pairs of dead
/// processes on the device and the host, a 6-chunk bind short with 1850
/// chunks free).
pub(crate) fn close_books(
    id: ::engine::InstanceId,
    closed: anyhow::Result<u64>,
    channels: &mut crate::engine::channel::ChannelJoin,
    roles: Option<&crate::store::roles::RolesHandle>,
    shared: Option<&crate::store::shared::Shared>,
) {
    let standing = match closed {
        Ok(standing) => Some(standing),
        Err(err) => {
            tracing::warn!(
                instance_id = id.0,
                ?err,
                "scheduler close_instance failed; its slots go back all the same"
            );
            None
        }
    };
    channels.unbind(id);
    if let Some(roles) = roles {
        roles.lock().unwrap().close_instance(id);
    }
    if let Some(shared) = shared {
        let mut pool = shared.lock().unwrap();
        pool.instance_closed(id);
        if let Some(standing) = standing {
            pool.standing_stated(standing);
        }
    }
}

fn roles_of(engine_idx: usize) -> Option<crate::store::roles::RolesHandle> {
    crate::store::registry::try_get(0, engine_idx).and_then(|stores| stores.roles)
}

/// With `PIE_CHECK_SLOT_REFS` set, a fire whose lanes name a slot of the
/// engine's shared pool that is not handed out (its chunk not open for the
/// shape, or the slot free in it) is refused, saying which: a page table
/// entry the engine would read or write through unassigned memory.
/// A step of `frame` wider than the rung the engine holds, `held` rows
/// ([`crate::store::shared::engine_rows`]): its token rows or its lanes
/// past them. The engine lays a fire's inputs for the rung it was told, so
/// such a step would run its tables past their regions; it is refused by
/// name before the engine stages anything.
fn past_rung(held: u32, frame: &crate::engine::FrameFire) -> Option<String> {
    frame.steps.iter().enumerate().find_map(|(at, step)| {
        let rows = step.submission.rows();
        let lanes = u32::try_from(step.submission.lanes.len()).unwrap_or(u32::MAX);
        (rows > held || lanes > held).then(|| {
            format!(
                "step {at} of a fire takes {rows} token rows over {lanes} lanes, past the \
                 {held} rows of the activation rung the engine holds: no step wider than the \
                 rung the engine was told reaches it"
            )
        })
    })
}

fn unassigned_ref(engine_idx: usize, frame: &crate::engine::FrameFire) -> Option<String> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("PIE_CHECK_SLOT_REFS").is_some()) {
        return None;
    }
    let shared = crate::store::registry::try_get(0, engine_idx)?.shared?;
    let pool = shared.lock().unwrap();
    frame.steps.iter().find_map(|step| {
        step.submission
            .lanes
            .iter()
            .enumerate()
            .find_map(|(at, lane)| {
                let fixed: Vec<(::engine::chunk::ShapeId, Vec<u32>)> = lane
                    .record
                    .pieces()
                    .iter()
                    .map(|(shape, pairs)| (*shape, pairs.iter().flatten().copied().collect()))
                    .collect();
                pool.unassigned(&lane.tables)
                    .or_else(|| pool.unassigned(&fixed))
                    .map(|(shape, slot_ref)| {
                        format!(
                            "a fire's lane {at} names slot ref {slot_ref} of shape {shape:?}, \
                         which the pool has not handed out"
                        )
                    })
            })
    })
}

/// Fails the fires of `steps[at]` whose lanes the engine `refused`, each
/// by its first such lane's reason, and every later fire of their
/// instances, which reads what they would have left; and takes them out of
/// `rest` (the submissions of `steps[at..]`, which never ran), dropping a
/// step left with no lane. What stays of `rest` goes to the engine again:
/// nothing of it ran, so nothing of it is lost or run twice.
fn refuse_lanes(
    steps: &mut Vec<crate::engine::StepFire>,
    at: usize,
    mut rest: Vec<::engine::Step>,
    refused: &[::engine::LaneFault],
) -> std::result::Result<Vec<::engine::Step>, String> {
    use crate::engine::completion;

    let named = || {
        refused
            .iter()
            .map(|fault| format!("lane {}: {}", fault.lane, fault.reason))
            .collect::<Vec<_>>()
            .join("; ")
    };
    if rest.len() != steps.len().saturating_sub(at) || rest.is_empty() {
        return Err(format!(
            "the engine refused lanes of a step past the frame's end: {}",
            named()
        ));
    }
    let mut gone: Vec<(::engine::InstanceId, String)> = Vec::new();
    for (offset, step) in rest.iter_mut().enumerate() {
        let fire = &mut steps[at + offset];
        let members = fire.member_lanes.len().saturating_sub(1);
        let mut why: Vec<Option<String>> = vec![None; members];
        for (member, slot) in why.iter_mut().enumerate() {
            let lanes = fire.member_lanes[member]..fire.member_lanes[member + 1];
            let instance = fire.instances.get(lanes.start as usize).copied();
            if offset == 0
                && let Some(fault) = refused.iter().find(|fault| lanes.contains(&fault.lane))
            {
                *slot = Some(format!(
                    "engine: lane {} refused: {}",
                    fault.lane, fault.reason
                ));
            } else if let Some((_, reason)) = gone
                .iter()
                .find(|(gone, _)| instance.is_some_and(|instance| instance == *gone))
            {
                *slot = Some(format!(
                    "an earlier fire of this instance in its frame was refused: {reason}"
                ));
            }
        }
        if offset == 0 && why.iter().all(Option::is_none) {
            return Err(format!(
                "the engine refused lanes no fire of the step holds: {}",
                named()
            ));
        }
        for (member, reason) in why.iter().enumerate() {
            let Some(reason) = reason else { continue };
            let start = fire.member_lanes[member] as usize;
            if let Some(&instance) = fire.instances.get(start)
                && !gone.iter().any(|(gone, _)| *gone == instance)
            {
                gone.push((instance, reason.clone()));
            }
            tracing::warn!(%reason, "a fire's lanes were refused; it fails alone");
            if let Some(&cell) = fire.terminal_cells.get(member) {
                completion::settle(&[cell], completion::TERMINAL_OUTCOME_FAILED);
            }
            if let Some(completion) = fire.completions.get(member) {
                completion.reject_unsubmitted(reason.clone());
            }
        }
        strip_members(fire, step, &why);
    }
    let mut kept = Vec::with_capacity(rest.len());
    let mut index = at;
    for step in rest {
        if step.lanes.is_empty() {
            steps.remove(index);
        } else {
            kept.push(step);
            index += 1;
        }
    }
    Ok(kept)
}

/// Takes the members `drop` marks out of a step: their lanes, their cells
/// and completions, their attachments and media, every lane index after
/// them renumbered, and the lane groups left packed in order.
fn strip_members(
    fire: &mut crate::engine::StepFire,
    step: &mut ::engine::Step,
    drop: &[Option<String>],
) {
    if drop.iter().all(Option::is_none) {
        return;
    }
    let lanes = step.lanes.len();
    // Each old lane's new index, `None` where it goes.
    let mut to: Vec<Option<u32>> = vec![None; lanes];
    let mut member_lanes = vec![0u32];
    let mut next = 0u32;
    for (member, dropped) in drop.iter().enumerate() {
        let span = fire.member_lanes[member] as usize..fire.member_lanes[member + 1] as usize;
        if dropped.is_none() {
            for lane in span {
                if let Some(slot) = to.get_mut(lane) {
                    *slot = Some(next);
                    next += 1;
                }
            }
            member_lanes.push(next);
        }
    }
    let keep = |lane: u32| to.get(lane as usize).copied().flatten();
    let mut kept_lanes = Vec::with_capacity(next as usize);
    let mut instances = Vec::with_capacity(next as usize);
    for (lane, moved) in std::mem::take(&mut step.lanes).into_iter().enumerate() {
        if to[lane].is_some() {
            kept_lanes.push(moved);
            instances.extend(fire.instances.get(lane).copied());
        }
    }
    // A pass's group id is unique in its step: the kept ones, packed.
    let mut ids: Vec<u32> = kept_lanes.iter().filter_map(|lane| lane.group).collect();
    ids.sort_unstable();
    ids.dedup();
    for lane in &mut kept_lanes {
        lane.group = lane
            .group
            .and_then(|id| ids.binary_search(&id).ok())
            .map(|at| u32::try_from(at).unwrap_or(u32::MAX));
    }
    step.lanes = kept_lanes;
    step.attachments.retain_mut(|attachment| match keep(attachment.lane) {
        Some(lane) => {
            attachment.lane = lane;
            true
        }
        None => false,
    });
    step.media.retain_mut(|media| match keep(media.lane) {
        Some(lane) => {
            media.lane = lane;
            true
        }
        None => false,
    });
    step.voxels.retain_mut(|voxels| match keep(voxels.lane) {
        Some(lane) => {
            voxels.lane = lane;
            true
        }
        None => false,
    });
    let mut member = 0;
    fire.terminal_cells.retain(|_| {
        let kept = drop.get(member).is_none_or(Option::is_none);
        member += 1;
        kept
    });
    let mut member = 0;
    fire.completions.retain(|_| {
        let kept = drop.get(member).is_none_or(Option::is_none);
        member += 1;
        kept
    });
    fire.member_lanes = member_lanes;
    fire.instances = instances;
}

fn backend(engine: &mut Option<EngineBox>) -> Result<&mut EngineBox> {
    engine.as_mut().ok_or_else(|| anyhow!(NO_BACKEND))
}

fn fail_request(request: LaneRequest, why: &str) {
    match request {
        LaneRequest::Launch { reply, .. } => {
            let _ = reply.send(Err(why.to_string()));
        }
        LaneRequest::Control { item, reply } => {
            match &*item {
                QueuedItem::PreLaunchCopy {
                    logical_completion, ..
                } => logical_completion.reject_unsubmitted(why),
                QueuedItem::CopyTracked { completion, .. } => {
                    completion.resolve(&Err(anyhow!("{why}")));
                }
                _ => {}
            }
            let commit = Lane::commit_on_failure(&item, why);
            match *item {
                QueuedItem::HostKv { response, .. } => {
                    let _ = response.send(Err(anyhow!("{why}")));
                }
                QueuedItem::Pool { response, .. } => {
                    let _ = response.send(Err(anyhow!("{why}")));
                }
                QueuedItem::PrepareProgram { response, .. } => {
                    let _ = response.send(Err(anyhow!("{why}")));
                }
                QueuedItem::FinishProgram { response, .. } => {
                    let _ = response.send(Err(anyhow!("{why}")));
                }
                QueuedItem::FireFacts { response, .. } => {
                    let _ = response.send(Err(anyhow!("{why}")));
                }
                QueuedItem::StandingAfter { response, .. } => {
                    let _ = response.send(Err(anyhow!("{why}")));
                }
                _ => {}
            }
            let _ = reply.send(commit);
        }
        LaneRequest::Shutdown { .. } | LaneRequest::Landed { .. } => {}
    }
}

pub(super) struct Lane {
    launch_tx: crate::rt::channel::Sender<LaneRequest>,
    control_tx: crate::rt::channel::Sender<LaneRequest>,
    thread: Option<crate::rt::thread::JoinHandle<()>>,
}

#[derive(Default)]
struct LaneTurn {
    launch_run: u32,
    control_run: u32,
}

impl LaneTurn {
    const LAUNCH_RUN_BEFORE_CONTROL: u32 = 2;

    const CONTROL_RUN_MAX: u32 = 32;

    const fn control_due(&self) -> bool {
        self.launch_run >= Self::LAUNCH_RUN_BEFORE_CONTROL
            && self.control_run < Self::CONTROL_RUN_MAX
    }

    fn took_launch(&mut self) {
        self.launch_run = self.launch_run.saturating_add(1);
    }

    fn took_control(&mut self) {
        self.control_run = self.control_run.saturating_add(1);
        if self.control_run >= Self::CONTROL_RUN_MAX {
            self.end_control_turn();
        }
    }

    fn end_control_turn(&mut self) {
        self.launch_run = 0;
        self.control_run = 0;
    }
}

impl Lane {
    pub(super) fn spawn(
        engine_idx: usize,
        engine: Option<EngineBox>,
        stats: Arc<SchedulerStats>,
    ) -> Self {
        let (launch_tx, launch_rx) = crate::rt::channel::unbounded::<LaneRequest>();
        let (control_tx, control_rx) = crate::rt::channel::unbounded::<LaneRequest>();
        let landed_tx = control_tx.clone();
        let thread = crate::rt::thread::Builder::new()
            .name(format!("pie-engine-{engine_idx}"))
            .spawn(move || Self::run(engine_idx, engine, launch_rx, control_rx, landed_tx, stats))
            .expect("spawn pie-engine lane thread");
        Self {
            launch_tx,
            control_tx,
            thread: Some(thread),
        }
    }

    fn post(&self, request: LaneRequest) {
        let queue = match &request {
            LaneRequest::Launch { .. } => &self.launch_tx,
            _ => &self.control_tx,
        };
        let _ = queue.send(request);
    }

    pub(super) fn fire(
        &self,
        submission: crate::engine::FrameFire,
        prefill: bool,
    ) -> LaunchReplyRx {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.post(LaneRequest::Launch {
            submission,
            prefill,
            reply: reply_tx,
        });
        reply_rx
    }

    pub(super) fn control(&self, item: QueuedItem) -> ControlReplyRx {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.post(LaneRequest::Control {
            item: Box::new(item),
            reply: reply_tx,
        });
        reply_rx
    }

    pub(super) fn shutdown(&mut self) -> (Option<EngineBox>, ChannelJoin) {
        let (response_tx, response_rx) = crate::rt::channel::bounded(1);
        let _ = self.control_tx.send(LaneRequest::Shutdown {
            response: response_tx,
        });
        let state = response_rx
            .recv()
            .unwrap_or_else(|_| (None, ChannelJoin::new()));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        state
    }

    fn next_request(
        launch_rx: &crate::rt::channel::Receiver<LaneRequest>,
        control_rx: &crate::rt::channel::Receiver<LaneRequest>,
        turn: &mut LaneTurn,
    ) -> std::result::Result<LaneRequest, ()> {
        use crate::rt::channel::TryRecvError;
        #[cfg(target_arch = "wasm32")]
        const ENGINE_LANE_HOT_US: u64 = 0;
        #[cfg(not(target_arch = "wasm32"))]
        const ENGINE_LANE_HOT_US: u64 = 1_000_000;
        let hot_window = Duration::from_micros(ENGINE_LANE_HOT_US);
        let mut spin_until = Instant::now() + hot_window;
        loop {
            if turn.control_due() {
                match control_rx.try_recv() {
                    Ok(request) => {
                        turn.took_control();
                        return Ok(request);
                    }
                    Err(TryRecvError::Disconnected) => {
                        return launch_rx.try_recv().map_err(|_| ());
                    }
                    Err(TryRecvError::Empty) => turn.end_control_turn(),
                }
            }
            match launch_rx.try_recv() {
                Ok(request) => {
                    turn.took_launch();
                    return Ok(request);
                }
                Err(TryRecvError::Disconnected) => {
                    return control_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            match control_rx.try_recv() {
                Ok(request) => {
                    turn.took_control();
                    return Ok(request);
                }
                Err(TryRecvError::Disconnected) => {
                    return launch_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            if Instant::now() < spin_until {
                std::hint::spin_loop();
                continue;
            }
            let mut select = crate::rt::channel::Select::new();
            select.recv(launch_rx);
            select.recv(control_rx);
            select.ready();
            spin_until = Instant::now() + hot_window;
        }
    }

    fn run(
        engine_idx: usize,
        mut engine: Option<EngineBox>,
        launch_rx: crate::rt::channel::Receiver<LaneRequest>,
        control_rx: crate::rt::channel::Receiver<LaneRequest>,
        landed_tx: crate::rt::channel::Sender<LaneRequest>,
        stats: Arc<SchedulerStats>,
    ) {
        let mut channels = ChannelJoin::new();
        // The pool's instance and channel slots, looked up at the first
        // fire (the stores are registered before any process runs).
        let roles = std::cell::OnceCell::new();
        let mut landed_instances: HashMap<::engine::FrameId, Vec<LandingReads>> =
            HashMap::new();
        let broker = crate::engine::CompletionBroker::new();
        let settlements = crate::engine::completion::FrameSettlements::new();
        let inline_faults: InlineFaults = Arc::default();
        if let Some(engine) = engine.as_mut() {
            if let Err(error) = engine.bind_thread() {
                tracing::error!(engine_idx, %error, "engine lane could not bind its thread");
            }
            engine.on_tapped(crate::offload::stream::sink());
            // A copy still landing settles its tracked completion here: the
            // runtime gives its source slots back only then.
            let landed = broker.clone();
            engine.on_landed(std::sync::Arc::new(move |token: u64| {
                landed.notify(token, waker::FIRST_COMPLETION_EPOCH);
                // The scheduler retires the copy's completion on its next
                // turn: woken now, not at its idle backstop, so the move
                // waiting on it commits (and its pages come back) at once.
                if let Ok(handle) = crate::scheduler::scheduler_handle(engine_idx) {
                    let _ = handle.nudge();
                }
            }));
            if engine.settles_asynchronously() {
                engine.on_complete(std::sync::Arc::new(
                    move |at: engine::StepDone, mut outcome: engine::StepOutcome| {
                        // What lanes wrote is kept before their step is published
                        // settled, so the fires that asked find it.
                        let writes = std::mem::take(&mut outcome.wrote);
                        if !writes.is_empty() {
                            crate::pipeline::fire::record_kv_writes(writes);
                        }
                        let _ = landed_tx.send(LaneRequest::Landed {
                            frame: at.frame,
                            step: at.step,
                            outcome,
                        });
                    },
                ));
            } else {
                // A frame settled inside `submit` answers through its
                // result. What its lanes wrote, and which faulted, is told
                // here before `submit` returns, so it is kept before the
                // frame's cells settle.
                let faulted = Arc::clone(&inline_faults);
                engine.on_complete(std::sync::Arc::new(
                    move |at: engine::StepDone, outcome: engine::StepOutcome| {
                        if !outcome.wrote.is_empty() {
                            crate::pipeline::fire::record_kv_writes(outcome.wrote);
                        }
                        if !outcome.faulted.is_empty() {
                            faulted
                                .lock()
                                .unwrap()
                                .push((at.step as usize, outcome.faulted));
                        }
                    },
                ));
            }
        }
        let mut stash: Option<LaneRequest> = None;
        let mut lane_turn = LaneTurn::default();
        loop {
            let request = match stash.take() {
                Some(stashed) => {
                    lane_turn.took_launch();
                    stashed
                }
                None => match Self::next_request(&launch_rx, &control_rx, &mut lane_turn) {
                    Ok(request) => request,
                    Err(()) => break,
                },
            };
            let work = match &request {
                LaneRequest::Control { .. } => Some(LaneWork::Control),
                LaneRequest::Launch { prefill: true, .. } => Some(LaneWork::Prefill),
                LaneRequest::Launch { .. } => Some(LaneWork::Launch),
                LaneRequest::Shutdown { .. } | LaneRequest::Landed { .. } => None,
            };
            let _lane_charge = LaneCharge {
                stats: &stats,
                began: Instant::now(),
                work,
            };
            // A wasm32 host cannot unwind at all; a panic there ends the tab.
            #[cfg(all(panic = "abort", not(target_arch = "wasm32")))]
            compile_error!(
                "the engine lane answers its owed frame from the panic path, \
                 which requires unwinding; under `panic = \"abort\"` a panic \
                 in one lane takes down every session the runtime is serving"
            );
            let panicked = match request {
                LaneRequest::Launch {
                    submission: mut frame,
                    reply,
                    ..
                } => {
                    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let fault = past_rung(
                            crate::store::shared::engine_rows(engine_idx),
                            &frame,
                        )
                        .or_else(|| unassigned_ref(engine_idx, &frame));
                        let result = match engine.as_mut() {
                            // A fire wider than the rung the engine holds,
                            // or naming a slot nobody was handed, is refused
                            // before the engine reads it.
                            _ if fault.is_some() => Err(fault.unwrap_or_default()),
                            Some(engine) => {
                                crate::probe_fire!(stats.fire.execute.engine_fire_us, {
                                    Self::fire_frame(
                                        engine_idx,
                                        engine,
                                        &channels,
                                        roles.get_or_init(|| roles_of(engine_idx)).as_ref(),
                                        &mut frame,
                                        &launch_rx,
                                        &mut stash,
                                        &broker,
                                        &settlements,
                                        &inline_faults,
                                        &mut landed_instances,
                                    )
                                })
                            }
                            None => Err(NO_BACKEND.to_string()),
                        };
                        if result.is_err() {
                            let cells: Vec<_> = frame.terminal_cells().collect();
                            crate::engine::completion::settle(
                                &cells,
                                crate::engine::completion::TERMINAL_OUTCOME_FAILED,
                            );
                        }
                        result
                    }));
                    let panicked = served.is_err();
                    let _ = reply.send(served.unwrap_or_else(|_| Err(LANE_PANICKED.to_string())));
                    panicked
                }
                LaneRequest::Control { item, reply } => {
                    let on_panic = Self::commit_on_failure(&item, LANE_PANICKED);
                    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        Self::execute_control(
                            engine_idx,
                            &mut engine,
                            &mut channels,
                            &broker,
                            *item,
                        )
                    }));
                    let panicked = served.is_err();
                    let _ = reply.send(served.unwrap_or(on_panic));
                    panicked
                }
                LaneRequest::Shutdown { response } => {
                    let _ = response.send((engine.take(), std::mem::take(&mut channels)));
                    return;
                }
                LaneRequest::Landed {
                    frame,
                    step,
                    outcome,
                } => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Taken, not read: a frame's steps land in any order (an
                    // engine tells a step with no instance from its device's
                    // completion, one with instances at a later harvest), and
                    // each step's reads go back at its own landing.
                    let (instances, reads) =
                        take_landing(&mut landed_instances, frame, step as usize)
                            .unwrap_or_default();
                    if let Some(engine) = engine.as_mut() {
                        if let Err(error) = engine.publish_landed() {
                            tracing::error!(frame = frame.0, step, %error, "publishing a landed step");
                        }
                        let roles = roles.get_or_init(|| roles_of(engine_idx)).as_ref();
                        for instance in instances {
                            if let Err(error) =
                                crate::store::roles::pump(roles, instance, Pump::Out, || {
                                    channels.pump_out(engine.as_mut(), instance)
                                })
                            {
                                tracing::error!(frame = frame.0, step, %error, "channel take after landing");
                            }
                        }
                    }
                    // What the step named is read no more: let go under it,
                    // it goes back now.
                    if let Some(roles) = roles.get_or_init(|| roles_of(engine_idx)).as_ref() {
                        roles.lock().unwrap().landed(&reads);
                    }
                    tracing::debug!(frame = frame.0, step, "channels pumped out after landing");
                    settlements.settled(frame, step as usize, &outcome, &broker);

                }))
                .is_err(),
            };
            if panicked {
                tracing::error!(
                    "engine lane panicked mid-request; failing it and every request \
                     behind it rather than leaving them in flight"
                );
                settlements.close_all(&broker);
                // No step of the lane's lands now: what they named is read
                // no more, and goes back as their closes come.
                if let Some(roles) = roles.get_or_init(|| roles_of(engine_idx)).as_ref() {
                    let mut roles = roles.lock().unwrap();
                    for (_, reads) in landed_instances.drain().flat_map(|(_, steps)| steps).flatten()
                    {
                        roles.landed(&reads);
                    }
                }
                if let Some(stashed) = stash.take() {
                    fail_request(stashed, "the engine lane is down after a panic");
                }
                Self::drain_poisoned(&launch_rx, &control_rx, engine, channels);
                return;
            }
        }
        drop(engine.take());
    }

    fn commit_on_failure(item: &QueuedItem, why: &str) -> LaneCommit {
        match item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::RegisterChannelsBind { pipeline_id, .. } => LaneCommit::BindFinished {
                pipeline_id: *pipeline_id,
                waits: false,
            },
            QueuedItem::PreLaunchCopy { .. } | QueuedItem::CopyTracked { .. } => {
                LaneCommit::AsyncControl {
                    result: Err(why.to_string()),
                }
            }
            QueuedItem::CloseInstance { .. }
            | QueuedItem::CloseChannels { .. }
            | QueuedItem::HostKv { .. }
            | QueuedItem::Pool { .. }
            | QueuedItem::PrepareProgram { .. }
            | QueuedItem::FinishProgram { .. }
            | QueuedItem::FireFacts { .. }
            | QueuedItem::StandingAfter { .. } => LaneCommit::None,
        }
    }

    fn drain_poisoned(
        launch_rx: &crate::rt::channel::Receiver<LaneRequest>,
        control_rx: &crate::rt::channel::Receiver<LaneRequest>,
        engine: Option<EngineBox>,
        channels: ChannelJoin,
    ) {
        std::mem::forget(engine);
        drop(channels);
        let mut turn = LaneTurn::default();
        while let Ok(request) = Self::next_request(launch_rx, control_rx, &mut turn) {
            if let LaneRequest::Shutdown { response } = request {
                let _ = response.send((None, ChannelJoin::new()));
                return;
            }
            fail_request(request, "the engine lane is down after a panic");
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::too_many_arguments,
        reason = "the lane's state a frame reads, each its own"
    )]
    fn fire_frame(
        engine_idx: usize,
        engine: &mut EngineBox,
        channels: &ChannelJoin,
        roles: Option<&crate::store::roles::RolesHandle>,
        frame: &mut crate::engine::FrameFire,
        launch_rx: &crate::rt::channel::Receiver<LaneRequest>,
        stash: &mut Option<LaneRequest>,
        broker: &crate::engine::CompletionBroker,
        settlements: &std::sync::Arc<crate::engine::completion::FrameSettlements>,
        inline_faults: &InlineFaults,
        landed: &mut HashMap<::engine::FrameId, Vec<LandingReads>>,
    ) -> std::result::Result<SubmissionCompletion, String> {
        use crate::engine::completion;

        inline_faults.lock().unwrap().clear();

        if stash.is_none()
            && let Ok(queued) = launch_rx.try_recv()
        {
            if let LaneRequest::Launch { submission, .. } = &queued
                && let Some(first) = submission.steps.first()
            {
                engine.expect_fire(&first.submission);
            }
            *stash = Some(queued);
        }

        let mut pending: Vec<::engine::Step> = frame
            .steps
            .iter_mut()
            .map(|step| std::mem::take(&mut step.submission))
            .collect();

        for step in &pending {
            for attachment in &step.attachments {
                let instance = attachment.instance;
                if let Err(error) = crate::store::roles::pump(roles, instance, Pump::In, || {
                    channels.pump_in(engine.as_mut(), instance)
                }) {
                    return Err(format!("channel publish: {error}"));
                }
            }
        }

        let asynchronous = engine.settles_asynchronously();
        // The engine frames this one took, where they settle off the
        // submit: each one's id, the steps of `frame` it fired, and the
        // waiters on what each step puts.
        let mut booked: Vec<(::engine::FrameId, std::ops::Range<usize>, Vec<Vec<u64>>)> =
            Vec::new();
        // Where `pending` starts in `frame.steps`.
        let mut at = 0usize;
        while !pending.is_empty() {
            let submitted = ::engine::FrameSubmission { steps: pending };
            // What each step's attachments name (their instances, and the
            // pair, channels and adapter each shares) is read until the step
            // lands: let go meanwhile (a last close, a park), it waits for
            // that. A step the engine did not fire is read no more.
            let mut reads: Vec<Vec<crate::store::roles::Owner>> = submitted
                .steps
                .iter()
                .map(|step| {
                    roles.map_or_else(Vec::new, |roles| {
                        roles
                            .lock()
                            .unwrap()
                            .reading(step.attachments.iter().map(|a| a.instance))
                    })
                })
                .collect();
            let let_go = |reads: &[Vec<crate::store::roles::Owner>]| {
                if let Some(roles) = roles {
                    let mut roles = roles.lock().unwrap();
                    for reads in reads {
                        roles.landed(reads);
                    }
                }
            };
            tracing::debug!(
                steps = submitted.steps.len(),
                "submitting a frame to the engine"
            );
            // A refusal of lanes names lanes of the first step that did not
            // fire: the frame fired the steps before it (`Engine::submit`).
            let submit_start = crate::rt::Instant::now();
            let submit_result = engine.submit(&submitted);
            tracing::debug!(
                target: "pie::launch",
                steps = submitted.steps.len(),
                submit_us = submit_start.elapsed().as_micros() as u64,
                "engine submit"
            );
            let (ticket, refused) = match submit_result {
                Ok(mut ticket) => {
                    let refused = std::mem::take(&mut ticket.refused);
                    (Some(ticket), refused)
                }
                Err(::engine::Error::Lanes(refused)) => (None, refused),
                Err(error) if error.is_retryable() => {
                    let_go(&reads);
                    return Err(format!(
                        "frame admission answered a retryable refusal past static \
                         admission, which the frame contract forbids: {error}"
                    ));
                }
                Err(error) => {
                    let_go(&reads);
                    if matches!(error, ::engine::Error::Impossible(_)) {
                        crate::store::shared::programs::impossible_submission(engine_idx, &error);
                    }
                    return Err(format!("{error}"));
                }
            };
            let fired = ticket.as_ref().map_or(0, |ticket| ticket.steps.len());
            let mut rest = submitted.steps;
            let ran: Vec<::engine::Step> = rest.drain(..fired.min(rest.len())).collect();
            let ran_at = at..at + ran.len();
            let unfired = reads.split_off(ran.len().min(reads.len()));
            let_go(&unfired);

            match &ticket {
                Some(ticket) if asynchronous && !ran.is_empty() => {
                    landed.insert(
                        ticket.id,
                        ran.iter()
                            .zip(reads)
                            .map(|(step, reads)| {
                                Some((step.attachments.iter().map(|a| a.instance).collect(), reads))
                            })
                            .collect(),
                    );
                }
                // Settled inside `submit` (or nothing fired): the device is
                // done with what it named.
                _ => let_go(&reads),
            }
            let mut wakes: Vec<Vec<u64>> = vec![Vec::new(); ran.len()];
            for (step, woken) in ran.iter().zip(&mut wakes) {
                for attachment in &step.attachments {
                    let deferred = asynchronous.then_some(&mut *woken);
                    let instance = attachment.instance;
                    if let Err(error) = crate::store::roles::pump(roles, instance, Pump::Out, || {
                        channels.pump_out_with(engine.as_mut(), instance, deferred)
                    }) {
                        return Err(format!("channel take: {error}"));
                    }
                }
            }

            if !asynchronous {
                // The lanes the engine told faulted fail their own fires;
                // every other fire of the frame commits.
                let faulted = std::mem::take(&mut *inline_faults.lock().unwrap());
                for (index, step) in frame.steps[ran_at.clone()].iter().enumerate() {
                    let faults: Vec<::engine::LaneFault> = faulted
                        .iter()
                        .filter(|(step, _)| *step == index)
                        .flat_map(|(_, faults)| faults.iter().cloned())
                        .collect();
                    if faults.is_empty() {
                        completion::settle(
                            &step.terminal_cells,
                            completion::TERMINAL_OUTCOME_SUCCESS,
                        );
                    } else {
                        completion::settle_fires(
                            &step.terminal_cells,
                            &step.completions,
                            &step.member_lanes,
                            &faults,
                            "frame settle",
                        );
                    }
                }
            } else if let Some(ticket) = &ticket
                && !ran.is_empty()
            {
                booked.push((ticket.id, ran_at.clone(), wakes));
            }

            at = ran_at.end;
            if refused.is_empty() {
                if !rest.is_empty() {
                    return Err(format!(
                        "the engine fired {fired} of a frame's {} steps and named no lane it \
                         refused",
                        fired + rest.len()
                    ));
                }
                break;
            }
            // The refused lanes' fires fail alone, with every later fire of
            // their instances; the rest go to the engine again.
            pending = refuse_lanes(&mut frame.steps, at, rest, &refused)?;
        }

        if !asynchronous || booked.is_empty() {
            return Ok(SubmissionCompletion::ready());
        }
        let completion = broker.submission_completion(waker::FIRST_COMPLETION_EPOCH);
        let last = booked.len() - 1;
        // A launch of several frames is checked to land in order.
        let split = (last > 0).then(|| completion::SplitLaunch::new(booked.len()));
        for (index, (id, steps, wakes)) in booked.into_iter().enumerate() {
            // Each step publishes its own cells and waiters as it lands.
            let steps = frame.steps[steps]
                .iter()
                .zip(wakes)
                .map(|(step, wakes)| completion::LandingStep {
                    cells: step.terminal_cells.clone(),
                    completions: step.completions.clone(),
                    member_lanes: step.member_lanes.clone(),
                    wakes,
                })
                .collect();
            // A lane's frames land in the order they were submitted, which
            // the split launch checks: the submission is in once its last
            // frame is.
            settlements.expect(
                id,
                steps,
                (index == last).then_some(&completion),
                split.as_ref().map(|split| (Arc::clone(split), index)),
                broker,
            );
        }
        Ok(completion)
    }

    fn execute_control(
        engine_idx: usize,
        engine: &mut Option<EngineBox>,
        channels: &mut ChannelJoin,
        broker: &crate::engine::CompletionBroker,
        item: QueuedItem,
    ) -> LaneCommit {
        match item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.is_settled() => {
                LaneCommit::refused("pre-launch copy already settled")
            }
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
                LaneCommit::refused("logical fire cancelled before pre-launch copy")
            }
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                ..
            } => {
                let operation = plan.label();
                let submitted =
                    backend(engine).and_then(|engine| copy_settled_on_landing(engine, broker, plan));
                match submitted {
                    Ok(completion) => LaneCommit::AsyncControl {
                        result: Ok(completion),
                    },
                    Err(error) => {
                        let message = format!("pre-launch {operation} rejected: {error:#}");
                        logical_completion.reject_unsubmitted(message.clone());
                        LaneCommit::refused(message)
                    }
                }
            }
            QueuedItem::PrepareProgram { program, response } => {
                // Its host codegen ran off this thread before it was sent.
                let prepared =
                    backend(engine).and_then(|engine| Ok(engine.prepare_program(&program)?));
                let _ = response.send(prepared);
                LaneCommit::None
            }
            QueuedItem::FinishProgram { prepared, response } => {
                let registered = backend(engine).and_then(|engine| {
                    let program_id = engine.finish_program(prepared)?;
                    // What an instance of it holds in the pool, asked once
                    // at registration. A program whose facts are not
                    // learned is released: the registration fails whole.
                    if let Some(roles) = roles_of(engine_idx)
                        && let Err(error) =
                            crate::store::roles::learn_program(&roles, &**engine, program_id)
                    {
                        let released = engine.close_program(program_id);
                        return Err(match released {
                            Ok(()) => anyhow!("its facts: {error}; released"),
                            Err(also) => anyhow!(
                                "its facts: {error}; and releasing program {program_id}: {also}"
                            ),
                        });
                    }
                    Ok(program_id)
                });
                let _ = response.send(registered);
                LaneCommit::None
            }
            QueuedItem::FireFacts { key, response } => {
                let asked = backend(engine).and_then(|engine| {
                    let facts = engine.fire_facts(key.program, &key.extents)?;
                    if let Some(shared) =
                        crate::store::registry::try_get(0, engine_idx).and_then(|s| s.shared)
                    {
                        shared.lock().unwrap().fire_facts_learned(key, &facts);
                    }
                    Ok(())
                });
                let _ = response.send(asked);
                LaneCommit::None
            }
            QueuedItem::StandingAfter { binding, response } => {
                let asked = backend(engine).and_then(|engine| {
                    let total = engine.standing_after(&binding)?;
                    // The booked total is the engine's as it stands: every
                    // bind and close books its total on this lane.
                    Ok(crate::store::registry::try_get(0, engine_idx)
                        .and_then(|s| s.shared)
                        .map_or(0, |shared| shared.lock().unwrap().standing_added(total)))
                });
                let _ = response.send(asked);
                LaneCommit::None
            }
            QueuedItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                bind,
                response,
            } => {
                if response.is_closed() {
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    tracing::warn!(
                        operation = "register_channels_bind",
                        program_id = bind.program_id().0,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished {
                        pipeline_id,
                        waits: false,
                    };
                }
                let engine = match backend(engine) {
                    Ok(engine) => engine,
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        return LaneCommit::BindFinished {
                            pipeline_id,
                            waits: false,
                        };
                    }
                };
                let roles = roles_of(engine_idx);
                let held = crate::store::roles::held_by(&**engine, &plans);
                // Everything it takes of the pool first, its rings, its
                // guest state, its tables: a bind short of room is answered
                // so before it registers anything, and waits for room
                // (`bind_when_resident`) with nothing rolled back.
                let taken = crate::store::roles::take_bind(
                    roles.as_ref(),
                    &mut **engine,
                    &bind.binding,
                    |id| channels.engine_registered(id) || held.contains(&id),
                    pipeline_id,
                )
                .and_then(|taken| {
                    match crate::store::roles::take_rings(roles.as_ref(), &mut **engine, &plans) {
                        Ok(rings) => Ok((taken, rings)),
                        Err(error) => {
                            if let (Some(taken), Some(roles)) = (taken, roles.as_ref()) {
                                roles.lock().unwrap().untake(taken);
                            }
                            Err(error)
                        }
                    }
                });
                let (taken, rings) = match taken {
                    Ok(taken) => taken,
                    Err(error) => {
                        let waits = crate::store::roles::waits_for_room(&error);
                        if response.send(Err(error)).is_err() {
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        return LaneCommit::BindFinished { pipeline_id, waits };
                    }
                };
                let untake = |taken: Option<crate::store::roles::Taken>| {
                    if let (Some(taken), Some(roles)) = (taken, roles.as_ref()) {
                        roles.lock().unwrap().untake(taken);
                    }
                };
                let registered = match Self::register_channel_set(
                    engine,
                    engine_idx,
                    channels,
                    &plans,
                    rings,
                    pipeline_id,
                ) {
                    Ok(registered) => registered,
                    Err(error) => {
                        untake(taken);
                        if response.send(Err(error)).is_err() {
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        return LaneCommit::BindFinished {
                            pipeline_id,
                            waits: false,
                        };
                    }
                };
                if response.is_closed() {
                    untake(taken);
                    Self::rollback_channel_set(
                        engine,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    Lane::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    return LaneCommit::BindFinished {
                        pipeline_id,
                        waits: false,
                    };
                }
                match Self::bind(engine, channels, &bind, roles.as_ref(), pipeline_id, taken) {
                    Ok((bound, standing)) => {
                        // Its program can be in a fire until it closes, and
                        // the engine's standing guest bytes are its total.
                        if let Some(shared) =
                            crate::store::registry::try_get(0, engine_idx).and_then(|s| s.shared)
                        {
                            let mut pool = shared.lock().unwrap();
                            pool.instance_bound(
                                bound.instance_id,
                                crate::store::shared::programs::FireKey {
                                    program: bind.program_id(),
                                    extents: bind.binding.extents,
                                },
                            );
                            pool.standing_stated(standing);
                        }
                        LaneCommit::BindInstance {
                            pipeline_id,
                            bound,
                            respond: BindRespond {
                                registered,
                                program_id: bind.program_id(),
                                response,
                            },
                        }
                    }
                    Err(error) => {
                        Self::rollback_channel_set(
                            engine,
                            channels,
                            &registered,
                            "register_channels_bind",
                            false,
                        );
                        if response.send(Err(error)).is_err() {
                            Self::release_registered_channel_wait_slots(&registered);
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        LaneCommit::BindFinished {
                            pipeline_id,
                            waits: false,
                        }
                    }
                }
            }
            QueuedItem::CopyTracked { plan, completion } => {
                match backend(engine).and_then(|engine| copy_settled_on_landing(engine, broker, plan)) {
                    Ok(native_completion) => LaneCommit::AsyncControl {
                        result: Ok(native_completion),
                    },
                    Err(error) => {
                        let message = format!("{error:#}");
                        completion.resolve(&Err(error));
                        LaneCommit::refused(message)
                    }
                }
            }
            QueuedItem::Pool { op, response } => {
                let result = backend(engine).and_then(|engine| {
                    match op {
                        PoolOp::CommitSegments(segments) => engine.commit_segments(segments),
                        PoolOp::ReleaseSegments(segments) => engine.release_segments(segments),
                        PoolOp::CommitSpan(segments) => engine.commit_span(segments),
                        PoolOp::Rung { from, to } => engine.set_rung(from, to),
                        PoolOp::ReleaseSpan(segments) => engine.release_span(segments),
                        PoolOp::Evacuate(evacuation) => engine.evacuate(&evacuation),
                        PoolOp::SettleCopies => engine.settle_copies(),
                        PoolOp::Rebind(rebind) => crate::store::roles::rebind(
                            roles_of(engine_idx).as_ref(),
                            &mut **engine,
                            rebind,
                            |engine, instance, pump| match pump {
                                Pump::In => channels.pump_in(engine, instance),
                                Pump::Out => channels.pump_out(engine, instance),
                            },
                        ),
                    }
                    .map_err(anyhow::Error::from)
                });
                let _ = response.send(result);
                LaneCommit::None
            }
            QueuedItem::HostKv { op, response } => {
                let result = backend(engine).and_then(|engine| match op {
                    HostKvOp::Write(slots, bytes) => engine
                        .write_host_kv(&slots, &bytes)
                        .map(|()| Vec::new())
                        .map_err(Into::into),
                    HostKvOp::CopyRead(copy, slots) => engine
                        .copy_kv(&copy)
                        .and_then(|()| engine.read_host_kv(&slots))
                        .map_err(Into::into),
                    HostKvOp::Read(slots) => engine.read_host_kv(&slots).map_err(Into::into),
                    HostKvOp::ReadState(slots, pages) => {
                        engine.read_host_state(&slots, &pages).map_err(Into::into)
                    }
                    HostKvOp::WriteState(slots, pages, bytes) => engine
                        .write_host_state(&slots, &pages, &bytes)
                        .map(|()| Vec::new())
                        .map_err(Into::into),
                    HostKvOp::ReadTier(slots) => engine.read_tier_slots(&slots).map_err(Into::into),
                    HostKvOp::WriteTier(slots, bytes) => engine
                        .write_tier_slots(&slots, &bytes)
                        .map(|()| Vec::new())
                        .map_err(Into::into),
                    HostKvOp::TapGroups => {
                        Ok(serde_json::to_vec(&engine.tap_groups()).unwrap_or_default())
                    }
                    HostKvOp::WeightResidency(residency) => engine
                        .weight_residency(&residency)
                        .map(|held| serde_json::to_vec(&held).unwrap_or_default())
                        .map_err(Into::into),
                    HostKvOp::WeightUses => engine
                        .weight_uses()
                        .map(|uses| serde_json::to_vec(&uses).unwrap_or_default())
                        .map_err(Into::into),
                    HostKvOp::BeamList(instance, beam) => engine
                        .beam_list(instance, beam)
                        .map(|list| list.iter().flat_map(|page| page.to_le_bytes()).collect())
                        .map_err(Into::into),
                    HostKvOp::Layout => Ok(engine
                        .host_kv_layout()
                        .and_then(|layout| serde_json::to_vec(&layout).ok())
                        .unwrap_or_default()),
                });
                let _ = response.send(result);
                LaneCommit::None
            }
            QueuedItem::CloseInstance { id, .. } => {
                let closed = backend(engine)
                    .and_then(|engine| engine.close_instance(id).map_err(anyhow::Error::from));
                let shared = crate::store::registry::try_get(0, engine_idx).and_then(|s| s.shared);
                close_books(
                    id,
                    closed,
                    channels,
                    roles_of(engine_idx).as_ref(),
                    shared.as_ref(),
                );
                LaneCommit::CloseInstance { id }
            }
            QueuedItem::CloseChannels { ids } => {
                for id in ids {
                    let result = if !channels.contains(id) {
                        Err(anyhow!("channel {id} is unknown or stale"))
                    } else {
                        backend(engine).and_then(|engine| {
                            Self::close_channel(engine, id).map(|()| {
                                channels.remove(id);
                                if let Some(roles) = roles_of(engine_idx) {
                                    roles.lock().unwrap().close_channel(id);
                                }
                            })
                        })
                    };
                    if let Err(err) = result {
                        tracing::warn!(channel_id = id.0, ?err, "scheduler close_channel failed");
                    }
                }
                LaneCommit::None
            }
        }
    }

    fn bind(
        engine: &mut EngineBox,
        channels: &mut ChannelJoin,
        plan: &crate::engine::InstanceBindingPlan,
        roles: Option<&crate::store::roles::RolesHandle>,
        holder: Option<ProcessId>,
        taken: Option<crate::store::roles::Taken>,
    ) -> Result<(BoundInstance, u64)> {
        let bound =
            crate::store::roles::bind_taken(roles, &mut **engine, &plan.binding, holder, taken)?;
        let standing = bound.standing_guest_bytes;
        let bound = BoundInstance::new(&bound, plan.pacing_wait_id);
        channels.bind(bound.instance_id, plan.binding.channels.clone());
        Ok((bound, standing))
    }

    fn close_channel(engine: &mut EngineBox, id: ::engine::ChannelId) -> Result<()> {
        match engine.close_channel(id) {
            Ok(()) | Err(engine::Error::Unsupported { .. }) => Ok(()),
            Err(error) => Err(anyhow::Error::from(error)),
        }
    }

    fn register_channel_set(
        engine: &mut EngineBox,
        engine_idx: usize,
        channels: &mut ChannelJoin,
        plans: &[ChannelRegistration],
        rings: Vec<Option<crate::store::roles::TakenRing>>,
        holder: Option<ProcessId>,
    ) -> Result<Vec<RegisteredChannel>> {
        let roles = roles_of(engine_idx);
        // Each plan's ring, taken before anything was registered.
        let mut rings = rings.into_iter();
        let mut registered = Vec::with_capacity(plans.len());
        let mut registered_ids = Vec::with_capacity(plans.len());
        let undo =
            |engine: &mut EngineBox, channels: &mut ChannelJoin, ids: &[::engine::ChannelId]| {
                for &channel_id in ids.iter().rev() {
                    let _ = engine.close_channel(channel_id);
                    channels.remove(channel_id);
                    if let Some(roles) = &roles {
                        roles.lock().unwrap().close_channel(channel_id);
                    }
                }
            };
        for plan in plans {
            let ring = rings.next().flatten();
            if channels.contains(plan.id) {
                undo(engine, channels, &registered_ids);
                crate::store::roles::untake_rings(
                    roles.as_ref(),
                    std::iter::once(ring).chain(rings).collect(),
                );
                return Err(anyhow!("channel {} is already registered", plan.id));
            }
            match crate::store::roles::register_taken(
                roles.as_ref(),
                &mut **engine,
                plan,
                holder,
                ring,
                |engine, plan| crate::engine::verbs::register_channel(engine, engine_idx, plan),
            ) {
                Ok(channel) => {
                    channels.insert(channel.clone(), plan.host_role);
                    registered_ids.push(plan.id);
                    registered.push(channel);
                }
                Err(cause) => {
                    undo(engine, channels, &registered_ids);
                    crate::store::roles::untake_rings(roles.as_ref(), rings.collect());
                    return Err(cause);
                }
            }
        }
        Ok(registered)
    }

    fn rollback_channel_set(
        engine: &mut EngineBox,
        channels: &mut ChannelJoin,
        registered: &[RegisteredChannel],
        operation: &'static str,
        cancellation: bool,
    ) {
        for channel in registered.iter().rev() {
            let channel_id = channel.binding.channel_id;
            match Self::close_channel(engine, channel_id) {
                Ok(()) => {
                    channels.remove(channel_id);
                    if let Some(roles) = roles_of(channel.engine_id) {
                        roles.lock().unwrap().close_channel(channel_id);
                    }
                }
                Err(error) => {
                    tracing::error!(
                        operation,
                        cancellation,
                        channel_id = channel_id.0,
                        ?error,
                        "scheduler cancellation rollback close_channel failed"
                    );
                }
            }
        }
        tracing::warn!(
            operation,
            cancellation,
            channel_count = registered.len(),
            "scheduler registration rollback closed registered channels"
        );
    }

    pub(super) fn release_registered_channel_wait_slots(registered: &[RegisteredChannel]) {
        Self::release_wait_slots(
            registered
                .iter()
                .flat_map(|channel| [channel.reader_wait_id, channel.writer_wait_id]),
        );
    }

    pub(super) fn release_wait_slots(wait_ids: impl IntoIterator<Item = u64>) {
        let wait_ids: Vec<u64> = wait_ids.into_iter().collect();
        let table = waker::WakerTable::global();
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

/// Runs `plan` on `engine`: its completion settles once the copy has read
/// its sources, so what waits on it gives them back only then, not while the
/// copy may still read slots another owner now writes
/// ([`::engine::Engine::copy_slots_landing`]).
fn copy_settled_on_landing(
    engine: &mut EngineBox,
    broker: &crate::engine::CompletionBroker,
    plan: PreLaunchCopy,
) -> anyhow::Result<SubmissionCompletion> {
    match plan {
        PreLaunchCopy::Kv(plan) => crate::engine::verbs::settled(engine.copy_kv(&plan)),
        PreLaunchCopy::State(plan) => crate::engine::verbs::settled(engine.copy_state(&plan)),
        PreLaunchCopy::Slots(plan) => {
            let landing = broker.landing_completion(waker::FIRST_COMPLETION_EPOCH);
            if engine.copy_slots_landing(&plan, landing.wait_id())? {
                Ok(SubmissionCompletion::ready())
            } else {
                Ok(landing)
            }
        }
    }
}

#[cfg(test)]
mod landing_tests {
    use super::*;

    /// An engine tells a frame's steps in any order (one with no instance
    /// from its device's completion, one with instances at a later
    /// harvest): each step's reads come back at its own landing, the
    /// frame's entry goes only once every step has landed. Taken by
    /// position, the last step landing first dropped the frame, and the
    /// earlier step's reads were never let go (its items never freed).
    #[test]
    fn a_frame_s_steps_landing_out_of_order_each_give_their_reads_back() {
        let frame = ::engine::FrameId(7);
        let owner = |n: u64| crate::store::roles::Owner::Instance(::engine::InstanceId(n));
        let mut landed: HashMap<::engine::FrameId, Vec<LandingReads>> = HashMap::new();
        landed.insert(
            frame,
            vec![
                Some((vec![::engine::InstanceId(1)], vec![owner(1)])),
                Some((Vec::new(), Vec::new())),
            ],
        );
        assert_eq!(take_landing(&mut landed, frame, 1), Some((Vec::new(), Vec::new())));
        assert!(landed.contains_key(&frame), "step 0 has not landed");
        assert_eq!(
            take_landing(&mut landed, frame, 0),
            Some((vec![::engine::InstanceId(1)], vec![owner(1)]))
        );
        assert!(landed.is_empty(), "every step landed");
        assert_eq!(take_landing(&mut landed, frame, 0), None, "told once");
    }

    use ::engine::chunk::{ShapeId, SlotCopy};
    use ::engine::fire::{FrameSubmission, FrameTicket};
    use ::engine::load::{LoadRequest, Loaded};
    use ::engine::transfer::MemoryDomain;
    use ::engine::{Engine, LandingSink};

    type Told = Arc<std::sync::Mutex<Option<LandingSink>>>;

    /// A tier copy that lands inside the call, or lands only when the test
    /// says so, through the sink the lane gave it.
    struct LateCopies {
        lands_in_call: bool,
        sink: Told,
        tokens: Arc<std::sync::Mutex<Vec<u64>>>,
    }

    impl Engine for LateCopies {
        fn kind(&self) -> &'static str {
            "fake-late-copies"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
            Err(::engine::Error::Load("no model".into()))
        }

        fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
            unreachable!("nothing fires here")
        }

        fn copy_slots(&mut self, _copy: &SlotCopy) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        fn copy_slots_landing(&mut self, copy: &SlotCopy, token: u64) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            self.tokens.lock().unwrap().push(token);
            Ok(self.lands_in_call)
        }

        fn on_landed(&mut self, sink: LandingSink) {
            *self.sink.lock().unwrap() = Some(sink);
        }
    }

    fn park() -> PreLaunchCopy {
        PreLaunchCopy::Slots(SlotCopy {
            shape: ShapeId(0),
            src: MemoryDomain::CudaDevice(0),
            dst: MemoryDomain::HostPinned,
            slots: vec![(3, 0), (4, 1)],
            runs: Vec::new(),
        })
    }

    /// The engine, the tokens of its copies, and the sink the lane gave it.
    fn engine(
        lands_in_call: bool,
        broker: &crate::engine::CompletionBroker,
    ) -> (EngineBox, Arc<std::sync::Mutex<Vec<u64>>>, Told) {
        let tokens: Arc<std::sync::Mutex<Vec<u64>>> = Arc::default();
        let sink: Told = Arc::default();
        let mut engine = LateCopies {
            lands_in_call,
            sink: Arc::clone(&sink),
            tokens: Arc::clone(&tokens),
        };
        let told = broker.clone();
        engine.on_landed(Arc::new(move |token| {
            told.notify(token, waker::FIRST_COMPLETION_EPOCH);
        }));
        (Box::new(engine), tokens, sink)
    }

    /// A park still landing past the call settles only when the engine
    /// tells it landed: what waits on it (the move's commit, which gives the
    /// source slots back) waits for the copy to have read them.
    #[test]
    fn a_copy_still_landing_settles_only_when_it_has_landed() {
        let broker = crate::engine::CompletionBroker::new();
        let (mut late, tokens, sink) = engine(false, &broker);
        let completion =
            copy_settled_on_landing(&mut late, &broker, park()).expect("the copy is taken");
        assert!(
            !completion.is_settled(),
            "the source slots stay out while the copy may still read them"
        );
        let token = tokens.lock().unwrap()[0];
        let sink = sink.lock().unwrap().clone().expect("the lane gave a sink");
        assert!(!completion.is_settled());
        sink(token);
        assert!(
            completion.is_settled(),
            "told landed, the copy's sources may go back"
        );
        assert!(completion.check().is_some_and(|settled| settled.is_ok()));
    }

    /// An engine whose copies land inside the call settles at once, as
    /// before.
    #[test]
    fn a_copy_landed_in_the_call_settles_at_once() {
        let broker = crate::engine::CompletionBroker::new();
        let (mut inline, _, _) = engine(true, &broker);
        let completion =
            copy_settled_on_landing(&mut inline, &broker, park()).expect("the copy is taken");
        assert!(completion.is_settled());
    }
}
