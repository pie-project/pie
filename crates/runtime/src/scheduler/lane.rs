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

use super::worker::{BindRespond, PreLaunchCopy, QueuedItem};

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
    },
    CloseInstance {
        id: u64,
    },
    AsyncControl {
        result: std::result::Result<SubmissionCompletion, String>,
    },
}

const LANE_PANICKED: &str = "the engine lane panicked serving this request";
const NO_BACKEND: &str = "engine has no backend installed";

impl LaneCommit {
    fn refused(message: impl Into<String>) -> Self {
        Self::AsyncControl {
            result: Err(message.into()),
        }
    }
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
                QueuedItem::CopyKvTracked { completion, .. } => {
                    completion.resolve(&Err(anyhow!("{why}")));
                }
                _ => {}
            }
            let _ = reply.send(Lane::commit_on_failure(&item, why));
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
        let mut landed_instances: HashMap<::engine::FrameId, Vec<Vec<u64>>> = HashMap::new();
        let broker = crate::engine::CompletionBroker::new();
        let settlements = crate::engine::completion::FrameSettlements::new();
        if let Some(engine) = engine.as_mut() {
            if let Err(error) = engine.bind_thread() {
                tracing::error!(engine_idx, %error, "engine lane could not bind its thread");
            }
            if engine.settles_asynchronously() {
                engine.on_complete(std::sync::Arc::new(move |at: engine::StepDone, outcome| {
                    let _ = landed_tx.send(LaneRequest::Landed {
                        frame: at.frame,
                        step: at.step,
                        outcome,
                    });
                }));
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
                        let result = match engine.as_mut() {
                            Some(engine) => {
                                crate::probe_fire!(stats.fire.execute.engine_fire_us, {
                                    Self::fire_frame(
                                        engine,
                                        &channels,
                                        &mut frame,
                                        &launch_rx,
                                        &mut stash,
                                        &broker,
                                        &settlements,
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
                        Self::execute_control(engine_idx, &mut engine, &mut channels, *item)
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
                    let instances = landed_instances
                        .get(&frame)
                        .and_then(|steps| steps.get(step as usize))
                        .cloned()
                        .unwrap_or_default();
                    if let Some(engine) = engine.as_mut() {
                        for instance in instances {
                            if let Err(error) = channels.pump_out(engine.as_mut(), instance) {
                                tracing::error!(frame, step, %error, "channel take after landing");
                            }
                        }
                    }
                    tracing::debug!(frame, step, "channels pumped out after landing");
                    settlements.settled(frame, &outcome, &broker);
                    if landed_instances
                        .get(&frame)
                        .is_some_and(|steps| step as usize + 1 >= steps.len())
                    {
                        landed_instances.remove(&frame);
                    }
                }))
                .is_err(),
            };
            if panicked {
                tracing::error!(
                    "engine lane panicked mid-request; failing it and every request \
                     behind it rather than leaving them in flight"
                );
                settlements.close_all(&broker);
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
            QueuedItem::BindInstance { pipeline_id, .. }
            | QueuedItem::RegisterChannelsBind { pipeline_id, .. } => LaneCommit::BindFinished {
                pipeline_id: *pipeline_id,
            },
            QueuedItem::PreLaunchCopy { .. }
            | QueuedItem::CopyKv { .. }
            | QueuedItem::CopyKvTracked { .. }
            | QueuedItem::CopyState { .. } => LaneCommit::AsyncControl {
                result: Err(why.to_string()),
            },
            QueuedItem::RegisterProgram { .. }
            | QueuedItem::RegisterChannel { .. }
            | QueuedItem::RegisterChannels { .. }
            | QueuedItem::CloseInstance { .. }
            | QueuedItem::CloseChannels { .. } => LaneCommit::None,
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
    fn fire_frame(
        engine: &mut EngineBox,
        channels: &ChannelJoin,
        frame: &mut crate::engine::FrameFire,
        launch_rx: &crate::rt::channel::Receiver<LaneRequest>,
        stash: &mut Option<LaneRequest>,
        broker: &crate::engine::CompletionBroker,
        settlements: &std::sync::Arc<crate::engine::completion::FrameSettlements>,
        landed: &mut HashMap<::engine::FrameId, Vec<Vec<u64>>>,
    ) -> std::result::Result<SubmissionCompletion, String> {
        use crate::engine::completion;

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

        let submitted = ::engine::FrameSubmission {
            steps: frame
                .steps
                .iter_mut()
                .map(|step| std::mem::take(&mut step.submission))
                .collect(),
        };

        for step in &submitted.steps {
            for attachment in &step.attachments {
                if let Err(error) = channels.pump_in(engine.as_mut(), attachment.instance) {
                    return Err(format!("channel publish: {error}"));
                }
            }
        }
        tracing::debug!(
            steps = submitted.steps.len(),
            "submitting a frame to the engine"
        );
        let ticket = match engine.submit(&submitted) {
            Ok(ticket) => ticket,
            Err(error) if error.is_retryable() => {
                return Err(format!(
                    "frame admission answered a retryable refusal past static \
                     admission, which the frame contract forbids: {error}"
                ));
            }
            Err(error) => return Err(format!("{error}")),
        };

        let asynchronous = engine.settles_asynchronously();
        if asynchronous {
            landed.insert(
                ticket.id,
                submitted
                    .steps
                    .iter()
                    .map(|step| step.attachments.iter().map(|a| a.instance).collect())
                    .collect(),
            );
        }
        let mut wakes: Vec<u64> = Vec::new();
        for step in &submitted.steps {
            for attachment in &step.attachments {
                let deferred = asynchronous.then_some(&mut wakes);
                if let Err(error) =
                    channels.pump_out_with(engine.as_mut(), attachment.instance, deferred)
                {
                    return Err(format!("channel take: {error}"));
                }
            }
        }

        if !asynchronous {
            for step in &frame.steps {
                completion::settle(&step.terminal_cells, completion::TERMINAL_OUTCOME_SUCCESS);
            }
            return Ok(SubmissionCompletion::ready());
        }

        let completion = broker.submission_completion(waker::FIRST_COMPLETION_EPOCH);
        let cells: Vec<_> = frame.terminal_cells().collect();
        settlements.expect(
            ticket.id,
            ticket.steps.len(),
            cells,
            wakes,
            &completion,
            broker,
        );
        Ok(completion)
    }

    fn execute_control(
        engine_idx: usize,
        engine: &mut Option<EngineBox>,
        channels: &mut ChannelJoin,
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
                let submitted = backend(engine).and_then(|engine| {
                    crate::engine::verbs::settled(match plan {
                        PreLaunchCopy::Kv(plan) => engine.copy_kv(&plan),
                        PreLaunchCopy::State(plan) => engine.copy_state(&plan),
                    })
                });
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
            QueuedItem::RegisterProgram { plan, response } => {
                if response.is_closed() {
                    tracing::warn!(
                        operation = "register_program",
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = backend(engine).and_then(|engine| {
                    let backend = crate::engine::verbs::codegen_backend(engine);
                    let plan = crate::pipeline::program::with_host_codegen(&plan, backend);
                    engine.register_program(&plan).map_err(anyhow::Error::from)
                });
                match result {
                    Ok(program_id) => {
                        if response.send(Ok(program_id)).is_err() {
                            tracing::warn!(
                                operation = "register_program",
                                program_hash = format_args!("0x{:016x}", plan.program_hash),
                                "scheduler RPC cancelled after program registration; retaining engine-lifetime program"
                            );
                        }
                    }
                    Err(error) => {
                        let _ = response.send(Err(error));
                    }
                }
                LaneCommit::None
            }
            QueuedItem::RegisterChannel { plan, response } => {
                if response.is_closed() {
                    tracing::warn!(
                        operation = "register_channel",
                        channel_id = plan.id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = if channels.contains(plan.id) {
                    Err(anyhow!("channel {} is already registered", plan.id))
                } else {
                    backend(engine).and_then(|engine| {
                        crate::engine::verbs::register_channel(engine, engine_idx, &plan).inspect(
                            |channel| {
                                channels.insert(channel.clone(), plan.host_role);
                            },
                        )
                    })
                };
                match result {
                    Ok(channel) => {
                        if let Err(Ok(channel)) = response.send(Ok(channel)) {
                            if let Some(engine) = engine.as_mut() {
                                Self::rollback_channel_set(
                                    engine,
                                    channels,
                                    std::slice::from_ref(&channel),
                                    "register_channel",
                                    true,
                                );
                            }
                            Self::release_registered_channel_wait_slots(std::slice::from_ref(
                                &channel,
                            ));
                        }
                    }
                    Err(error) => {
                        let _ = response.send(Err(error));
                    }
                }
                LaneCommit::None
            }
            QueuedItem::RegisterChannels { plans, response } => {
                if response.is_closed() {
                    tracing::warn!(
                        operation = "register_channels",
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = backend(engine).and_then(|engine| {
                    Self::register_channel_set(engine, engine_idx, channels, &plans)
                });
                match result {
                    Ok(registered) => {
                        if let Err(Ok(registered)) = response.send(Ok(registered)) {
                            if let Some(engine) = engine.as_mut() {
                                Self::rollback_channel_set(
                                    engine,
                                    channels,
                                    &registered,
                                    "register_channels",
                                    true,
                                );
                            }
                            Self::release_registered_channel_wait_slots(&registered);
                        }
                    }
                    Err(error) => {
                        let _ = response.send(Err(error));
                    }
                }
                LaneCommit::None
            }
            QueuedItem::BindInstance {
                pipeline_id,
                plan,
                response,
            } => {
                if response.is_closed() {
                    Lane::release_wait_slots([plan.pacing_wait_id]);
                    tracing::warn!(
                        operation = "bind_instance",
                        program_id = plan.program_id(),
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match backend(engine).and_then(|engine| Self::bind(engine, channels, &plan)) {
                    Ok(bound) => LaneCommit::BindInstance {
                        pipeline_id,
                        bound,
                        respond: BindRespond::Bind(response),
                    },
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_wait_slots([plan.pacing_wait_id]);
                        }
                        LaneCommit::BindFinished { pipeline_id }
                    }
                }
            }
            QueuedItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program,
                mut bind,
                response,
            } => {
                if response.is_closed() {
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    tracing::warn!(
                        operation = "register_channels_bind",
                        program_id = bind.program_id(),
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let engine = match backend(engine) {
                    Ok(engine) => engine,
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        return LaneCommit::BindFinished { pipeline_id };
                    }
                };
                let registered =
                    match Self::register_channel_set(engine, engine_idx, channels, &plans) {
                        Ok(registered) => registered,
                        Err(error) => {
                            if response.send(Err(error)).is_err() {
                                Self::release_wait_slots([bind.pacing_wait_id]);
                            }
                            return LaneCommit::BindFinished { pipeline_id };
                        }
                    };
                if response.is_closed() {
                    Self::rollback_channel_set(
                        engine,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    Lane::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let program_registered = program.is_some();
                if let Some(plan) = &program {
                    let backend = crate::engine::verbs::codegen_backend(engine);
                    let plan = &crate::pipeline::program::with_host_codegen(plan, backend);
                    match engine.register_program(plan).map_err(anyhow::Error::from) {
                        Ok(program_id) => bind.binding.program = program_id,
                        Err(error) => {
                            tracing::error!(
                                ?error,
                                "register_program refused; rolling the bind's channels back"
                            );
                            Self::rollback_channel_set(
                                engine,
                                channels,
                                &registered,
                                "register_channels_bind",
                                false,
                            );
                            if response.send(Err(error)).is_err() {
                                Lane::release_registered_channel_wait_slots(&registered);
                                Self::release_wait_slots([bind.pacing_wait_id]);
                            }
                            return LaneCommit::BindFinished { pipeline_id };
                        }
                    }
                }
                if response.is_closed() {
                    Self::rollback_channel_set(
                        engine,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    Self::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    if program_registered {
                        tracing::warn!(
                            operation = "register_channels_bind",
                            program_id = bind.program_id(),
                            "scheduler RPC cancelled after program registration; retaining engine-lifetime program"
                        );
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match Self::bind(engine, channels, &bind) {
                    Ok(bound) => LaneCommit::BindInstance {
                        pipeline_id,
                        bound,
                        respond: BindRespond::ChannelsBind {
                            registered,
                            program_id: bind.program_id(),
                            program_registered,
                            response,
                        },
                    },
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
                        LaneCommit::BindFinished { pipeline_id }
                    }
                }
            }
            QueuedItem::CopyKv { plan, response } => Self::answer_copy(
                backend(engine)
                    .and_then(|engine| crate::engine::verbs::settled(engine.copy_kv(&plan))),
                response,
            ),
            QueuedItem::CopyState { plan, response } => Self::answer_copy(
                backend(engine)
                    .and_then(|engine| crate::engine::verbs::settled(engine.copy_state(&plan))),
                response,
            ),
            QueuedItem::CopyKvTracked { plan, completion } => {
                match backend(engine)
                    .and_then(|engine| crate::engine::verbs::settled(engine.copy_kv(&plan)))
                {
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
            QueuedItem::CloseInstance { id, .. } => {
                match backend(engine)
                    .and_then(|engine| engine.close_instance(id).map_err(anyhow::Error::from))
                {
                    Ok(()) => {
                        channels.unbind(id);
                        LaneCommit::CloseInstance { id }
                    }
                    Err(err) => {
                        tracing::warn!(instance_id = id, ?err, "scheduler close_instance failed");
                        LaneCommit::None
                    }
                }
            }
            QueuedItem::CloseChannels { ids } => {
                for id in ids {
                    let result = if !channels.contains(id) {
                        Err(anyhow!("channel {id} is unknown or stale"))
                    } else {
                        backend(engine).and_then(|engine| {
                            Self::close_channel(engine, id).map(|()| {
                                channels.remove(id);
                            })
                        })
                    };
                    if let Err(err) = result {
                        tracing::warn!(channel_id = id, ?err, "scheduler close_channel failed");
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
    ) -> Result<BoundInstance> {
        let bound = engine.bind_instance(&plan.binding)?;
        let bound = BoundInstance::new(plan.engine_id, &bound, plan.pacing_wait_id);
        channels.bind(bound.instance_id, plan.binding.channels.clone());
        Ok(bound)
    }

    fn answer_copy(
        submitted: Result<SubmissionCompletion>,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    ) -> LaneCommit {
        match submitted {
            Ok(completion) => {
                let _ = response.send(Ok(completion.clone()));
                LaneCommit::AsyncControl {
                    result: Ok(completion),
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                let _ = response.send(Err(error));
                LaneCommit::refused(message)
            }
        }
    }

    fn close_channel(engine: &mut EngineBox, id: u64) -> Result<()> {
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
    ) -> Result<Vec<RegisteredChannel>> {
        let mut registered = Vec::with_capacity(plans.len());
        let mut registered_ids = Vec::with_capacity(plans.len());
        for plan in plans {
            if channels.contains(plan.id) {
                for channel_id in registered_ids.iter().rev() {
                    let _ = engine.close_channel(*channel_id);
                    channels.remove(*channel_id);
                }
                return Err(anyhow!("channel {} is already registered", plan.id));
            }
            match crate::engine::verbs::register_channel(engine, engine_idx, plan) {
                Ok(channel) => {
                    channels.insert(channel.clone(), plan.host_role);
                    registered_ids.push(plan.id);
                    registered.push(channel);
                }
                Err(cause) => {
                    for channel_id in registered_ids.iter().rev() {
                        let _ = engine.close_channel(*channel_id);
                        channels.remove(*channel_id);
                    }
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
                }
                Err(error) => {
                    tracing::error!(
                        operation,
                        cancellation,
                        channel_id,
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
