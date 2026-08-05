//! Per-driver direct batch scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::driver::{
    BoundInstance, ChannelRegistrationPlan, DriverBackend, DriverId, InstanceBindingPlan,
    PoolResizePlan, ProgramRegistration, RegisteredChannel, SchedulerLimits, StateCopyPlan,
    SubmissionCompletion, WorkItemAttemptOutcome, WorkItemCompletion,
};
use crate::scheduler::ProcessId;
use anyhow::{Result, anyhow};

use super::ControlCompletion;
use super::batch::{self, AdmissionLimits};
use super::frame::{self, FramePlan, FramePolicy, FrameStamp};
use super::stats::{self, SchedulerStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    Terminate,
    /// The planner is evicting the process: its lanes stop being awaited
    /// (boundaries seal without it) while already-submitted frames stay
    /// sealable — the tail drains untracked and its fire leases release,
    /// which the eviction's quiescence wait depends on. No purge.
    Suspend,
    /// A pipeline closed or dropped. Its wait-set row is released immediately,
    /// while every request already accepted by the scheduler continues
    /// untracked to settlement. Later guest submissions are rejected by the
    /// pipeline resource before they reach the scheduler.
    Close,
}

/// Post one pipeline-leave to EVERY registered driver's scheduler thread (a
/// pipeline's requests may have landed on any of them) so each thread's
/// local [`FramePolicy`] drops the leaver from its wait-set. Fire-and-forget:
/// a shutting-down/closed scheduler channel is silently skipped.
///
/// The `id` KEY SPACE DEPENDS ON `kind` — Close is lane-keyed (pipeline
/// scope id), Suspend/Terminate are process-keyed. Callers use the typed
/// wrappers below so the key space is fixed by the function name instead of
/// per-call-site convention (the §15.2 seal-wedge bug class).
fn post_pipeline_leave(id: ProcessId, owner: Option<ProcessId>, kind: LeaveKind) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::PipelineLeave(id, owner, kind, None));
    }
}

/// One LANE (pipeline scope) leaves the wait-all quorum gracefully; its
/// accepted fires drain to settlement untracked. `owner` is the owning
/// process when known — the KV allocation-wait park posts it, because the
/// process may block before its first fire, and without the owner the
/// policy cannot drop it from the process-keyed `staged` /
/// `joins_in_flight` (the frame seal would wait forever for a join that
/// cannot come).
pub(crate) fn notify_lane_close(scope: ProcessId, owner: Option<ProcessId>) {
    post_pipeline_leave(scope, owner, LeaveKind::Close);
}

/// EVERY lane `pid` owns leaves the wait-all quorum (planner suspend); the
/// submitted tail stays sealable and drains untracked. Process-keyed.
pub(crate) fn notify_process_suspend(pid: ProcessId) {
    post_pipeline_leave(pid, Some(pid), LeaveKind::Suspend);
}

/// `pid` is runnable again after a suspend (restore committed, or the
/// eviction rolled back). Undoes the wait-set consequences of
/// [`notify_process_suspend`]. Process-keyed, fire-and-forget: a missed
/// resume is fail-safe (the fleet just stops waiting for the process).
pub(crate) fn notify_process_resume(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ProcessResume(pid));
    }
}

/// Terminate `pid`'s lanes, fire-and-forget (queued fires are rejected).
/// Process-keyed. The waited sibling is [`notify_process_terminate`].
pub(crate) fn post_process_terminate(pid: ProcessId) {
    post_pipeline_leave(pid, None, LeaveKind::Terminate);
}

/// Post `pid`'s Terminate leave to every driver and hand back the fences that
/// resolve once each scheduler has PROCESSED it — the posting half of
/// [`notify_process_terminate`], split from its await.
///
/// The split is what lets a retiring process release its execution seat
/// SYNCHRONOUSLY, in `ProcessCtx::drop`, instead of carrying the permit into
/// the spawned teardown task: the leave is already queued ahead of the
/// release broadcast on the same producer, so every driver still observes
/// leave-then-release, while the deferred teardown keeps the awaited fence it
/// needs before recycling pooled resources. Measured at conc 512, holding the
/// permit until the teardown task ran cost 27.8 ms p50 per retiree — 16.7 ms
/// of it purely waiting for the spawn to be scheduled behind 511 siblings.
pub(crate) fn post_process_terminate_fenced(pid: ProcessId) -> Vec<TerminateFence> {
    let handles = super::handle_registry().read().unwrap();
    handles
        .iter()
        .flatten()
        .filter_map(|handle| {
            let (response, received) = tokio::sync::oneshot::channel();
            handle
                .send(SchedulerItem::PipelineLeave(
                    pid,
                    None,
                    LeaveKind::Terminate,
                    Some(response),
                ))
                .ok()
                .map(|_| received)
        })
        .collect()
}

/// One driver's acknowledgement that it has processed a posted Terminate
/// leave (see [`post_process_terminate_fenced`]).
pub(crate) type TerminateFence = tokio::sync::oneshot::Receiver<()>;

/// Await fences from [`post_process_terminate_fenced`]. Equivalent to the
/// tail of [`notify_process_terminate`]: once this resolves, every driver's
/// scheduler has purged the pid's queued work and cancelled its protected
/// in-flight control, so pooled resources can be recycled.
pub(crate) async fn await_terminate_fences(fences: Vec<TerminateFence>) {
    for fence in fences {
        let _ = fence.await;
    }
}

async fn notify_pipeline_leave_and_wait(pid: ProcessId, kind: LeaveKind) {
    let responses = {
        let handles = super::handle_registry().read().unwrap();
        handles
            .iter()
            .flatten()
            .filter_map(|handle| {
                let (response, received) = tokio::sync::oneshot::channel();
                handle
                    .send(SchedulerItem::PipelineLeave(
                        pid,
                        None,
                        kind,
                        Some(response),
                    ))
                    .ok()
                    .map(|_| received)
            })
            .collect::<Vec<_>>()
    };
    for response in responses {
        let _ = response.await;
    }
}

pub(crate) async fn notify_pipeline_close(pid: ProcessId) {
    notify_pipeline_leave_and_wait(pid, LeaveKind::Close).await;
}

/// `forward.park()`: the lane is leaving the frame wait-set until it fires
/// again. Broadcast to every driver's scheduler and fire-and-forget — a
/// policy that has never seen the lane fire ignores it, and the exit is
/// ordered by `seq` against that lane's own submits rather than against this
/// call. Deliberately NOT routed through the control path: park exists to
/// release a gather, and the control slot is depth 1, so a park queued behind
/// the dispatch the gather is holding could never arrive.
pub(crate) fn notify_lane_park(pid: ProcessId, seq: u64) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::LanePark { lane: pid, seq });
    }
}

/// A retiring process released its capped execution permit (capped
/// deployments only). Broadcast to every driver's scheduler (mirrors
/// [`notify_pipeline_leave`]): a policy with no staged successor ignores it,
/// the policy holding the successor's staged bind earmarks the join.
/// Carries the retiree's identity so the policy resolves exactly that
/// holder's departure. Fire-and-forget: the caller posted the holder's
/// Terminate leave first, on this same producer, so every driver sees
/// leave-then-release.
pub(crate) fn notify_execution_slot_released(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ExecutionSlotReleased(pid));
    }
}

/// The deferred teardown finished: every event the process can ever
/// produce is already in each driver's mailbox ahead of this one (the
/// teardown task is the process's last producer and sends this after its
/// final drop). The worker retires the pid's terminate tombstone on
/// receipt, bounding `terminated_processes` by live-plus-draining
/// processes instead of every process that ever ran.
pub(crate) fn notify_process_quiesced(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ProcessQuiesced(pid));
    }
}

/// A parked process acquired its execution permit: its first fire is
/// imminent, so it is a named join in flight and the cohort-boundary window
/// stays open until it lands. Sent BEFORE the
/// process's first fire enters the mailbox (same producer), so the policy
/// sees consume-then-fire; a reordered arrival is harmless (the policy's
/// staged guard skips a lane that already fired).
pub(crate) fn notify_execution_slot_consumed(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ExecutionSlotConsumed(pid));
    }
}

/// A process joined the execution-admission FIFO. Announced before the
/// permit wait so the frame policy can earmark it by identity.
pub(crate) fn notify_admission_queued(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::AdmissionQueued(pid));
    }
}

/// A process left the FIFO -- it took its permit, or it was cancelled.
pub(crate) fn notify_admission_dequeued(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::AdmissionDequeued(pid));
    }
}

/// No-op: wait-set rejoin is implicit on the pipeline's next scheduler
/// submission, so a join event has nothing to do here (the planner's
/// restore path relies on the same implicit rejoin).
#[allow(dead_code)] // no live caller — see doc.
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

/// Wake-class counter (plan §16.2): completions that the 250 ms hang backstop
/// discovered already settled — a lost nudge. Steady state stays at zero; any
/// increment is a wake-path regression worth a warning.
pub(crate) static BACKSTOP_RETIREMENTS: AtomicU64 = AtomicU64::new(0);
static NEXT_LOGICAL_FIRE_ID: AtomicU64 = AtomicU64::new(1);

/// Total backstop-path retirements since process start (test observability).
#[cfg(test)]
pub(crate) fn backstop_retirements() -> u64 {
    BACKSTOP_RETIREMENTS.load(Ordering::Relaxed)
}

pub(crate) struct PendingRequest {
    pub(crate) logical_fire_id: u64,
    pub(crate) request: crate::driver::LaunchPlan,
    pub(crate) instance_id: u64,
    pub(crate) completion: WorkItemCompletion,
    pub(crate) last_page_len: u32,
    /// The owning process. Process-wide suspend/terminate acts on every
    /// request with this identity.
    pub(crate) process_id: Option<ProcessId>,
    /// The submitting pipeline resource's stable scope identity, or `None`
    /// for an untracked/prebuilt fire. This is the wait-set key: the frame
    /// lane (and at k = 1 the synthesized single-slot stamp's lane).
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) prebuilt: bool,
    pub(crate) prelaunch_copy: Option<crate::driver::KvCopyPlan>,
    pub(crate) prelaunch_state_copy: Option<StateCopyPlan>,
    /// Vesuvius frame identity: which lane/frame/slot this fire belongs to.
    /// At k = 1 the worker synthesizes a single-slot stamp at admission for
    /// every tracked fire (`lane` = `pipeline_id`, `seq` = the fire id).
    /// `None` = an untracked/prebuilt rider — dispatched outside the
    /// sealed-wave order, never awaited.
    pub(crate) frame: Option<FrameStamp>,
    /// tart (0.3 re-port step 1): whether this fire's program carries
    /// attention-stage hooks (OnAttnProj/OnAttn). Stamped at the pipeline
    /// submit from the bound container; fire planning sorts hook rows
    /// last so the driver's hook-free fast prefix is maximal.
    pub(crate) hook_program: bool,
    /// The pass-wide adapter sink (dormant on 0.3 until the adapter
    /// re-port; the region table's LORA bit reads it).
    pub(crate) lora_program: bool,
    pub(super) timing: Option<FireTimingState>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FireTimingState {
    submitted_us: u64,
    enqueued_us: Option<u64>,
    ready_us: Option<u64>,
}

impl FireTimingState {
    fn new() -> Self {
        Self {
            submitted_us: super::fire_timing_now_us(),
            enqueued_us: None,
            ready_us: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WaveTimingState {
    wave_id: u64,
    membership_hash: u64,
    dispatch_started_us: u64,
    batch_built_us: u64,
    driver_started_us: u64,
    launch_returned_us: u64,
    decision_us: u64,
    active_pipelines: usize,
    missing_pipelines: usize,
    candidate_count: usize,
    deferred_pipelines: usize,
    depth_capped_pipelines: usize,
}

#[derive(Clone, Copy, Debug)]
struct FireTimingSnapshot {
    outcome_index: usize,
    logical_fire_id: u64,
    instance_id: u64,
    process_id: Option<ProcessId>,
    sampled_rows: usize,
    timing: FireTimingState,
}

impl PendingRequest {
    fn direct(
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
        prebuilt: bool,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        frame: Option<FrameStamp>,
        timing_enabled: bool,
        hook_program: bool,
        lora_program: bool,
    ) -> Self {
        let logical_fire_id = NEXT_LOGICAL_FIRE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            logical_fire_id,
            request,
            instance_id,
            completion,
            last_page_len,
            process_id,
            pipeline_id,
            prebuilt,
            prelaunch_copy,
            prelaunch_state_copy,
            frame,
            hook_program,
            lora_program,
            timing: timing_enabled.then(FireTimingState::new),
        }
    }

    pub(crate) fn wire_row_count(&self) -> usize {
        self.request.qo_indptr.len().saturating_sub(1)
    }

    fn requires_solo_submission(&self) -> bool {
        (self.prebuilt && self.pipeline_id.is_none())
            || (self.preserves_inner_rows() && self.request.qo_indptr.last().copied() == Some(0))
            || self.rs_batch_kind() == RsBatchKind::Solo
    }

    /// How this fire's recurrent-state rows constrain the wave it joins.
    ///
    /// The driver's RS execution mode is read off `rs_slot_flags` and the
    /// buffered CSR for the WHOLE composed batch, so a fire that touches the
    /// RS buffer used to go out alone unconditionally. It no longer has to:
    /// a row that appends to its buffer and a row that folds in-forward run
    /// the identical dispatch and differ only in whether the recurrence
    /// persists, which the driver now expresses per row. What still cannot
    /// share a batch is a pure COMMIT — it gathers its activations out of the
    /// slabs instead of computing them, a wholly different dispatch — and an
    /// RS row cannot share with a row that has no RS binding at all, because
    /// the RS arrays are one-per-request and a partial batch does not resolve.
    fn rs_batch_kind(&self) -> RsBatchKind {
        if self.request.rs_slot_ids.is_empty() {
            return RsBatchKind::None;
        }
        let indptr = &self.request.rs_buffer_slot_indptr;
        let replays = self
            .request
            .rs_slot_flags
            .iter()
            .enumerate()
            .any(|(row, flags)| {
                let span = indptr
                    .get(row + 1)
                    .zip(indptr.get(row))
                    .is_some_and(|(end, begin)| end > begin);
                span && flags & pie_driver_abi::RS_FLAG_FOLD != 0
                    && flags & pie_driver_abi::RS_FLAG_BUFFER_WRITE == 0
            });
        if replays {
            RsBatchKind::Solo
        } else {
            RsBatchKind::Composable
        }
    }

    pub(crate) fn preserves_inner_rows(&self) -> bool {
        self.wire_row_count() > 1
    }
}

/// See [`PendingRequest::rs_batch_kind`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RsBatchKind {
    /// No recurrent-state rows.
    None,
    /// Recurrent rows that compute their own tokens: a plain in-forward fold,
    /// a buffered append, or a write-and-fold. These compose with each other.
    Composable,
    /// A pure commit, which replays buffered activations instead of computing
    /// them. Goes out alone.
    Solo,
}

fn fire_membership_hash<'a>(logical_fire_ids: impl IntoIterator<Item = &'a u64>) -> u64 {
    logical_fire_ids
        .into_iter()
        .fold(14_695_981_039_346_656_037u64, |hash, logical_fire_id| {
            (hash ^ logical_fire_id).wrapping_mul(1_099_511_628_211)
        })
}

#[derive(Default)]
pub(crate) struct LaunchGrouping {
    instances: HashSet<u64>,
    /// Tracked pipelines already contributing to this wave. ONE wave-member
    /// per pipeline per wave: fires of one pipeline are ORDERED (B3), and
    /// the driver's compose-time geometry/containment validation reads the
    /// device state as of wave entry — a decode fire composed into the same
    /// wave as the prefill it depends on (R4-4 device-carried handoff
    /// submits it run-ahead) validates against KV the prefill has not
    /// committed yet and FAIL-STOPs the lane. Same-instance dedup already
    /// enforced this within an instance; the single-pipeline stream makes
    /// the cross-instance case reachable. Also aligns the composer with the
    /// one-frame-per-lane-per-boundary seal rule.
    pipelines: HashSet<ProcessId>,
    count: usize,
    forward_tokens: usize,
    page_refs: usize,
    has_solo_submission: bool,
    has_rs_rows: bool,
    has_user_mask: bool,
    has_device_geometry: bool,
    has_hook_program: bool,
    has_multi_token: bool,
}

/// A fire the driver will resolve a DENSE per-cell attention mask for out of
/// a descriptor channel. Such a fire composes only SOLO: the driver's
/// multi-program batch has no way to merge one program's dense mask with
/// another's geometry (v1 scope), and it throws
/// `RetryableLaunchError("dense device mask in a multi-program batch")`
/// rather than execute a wrong one.
///
/// `dense_device_mask` is the program's own binding — an `AttnMask` port
/// sourced from a channel — and is the SAME predicate the driver resolves
/// on. The second clause is the older inference (a user mask with no wire
/// BRLE rows must be device-carried); it is kept because it covers fires
/// whose mask is device-carried without the port binding being visible here.
///
/// The inference alone was not enough. `cuda_runahead_concurrent` runs 8
/// pipelines of a sink/sliding-window decode program: every fire carried
/// BOTH wire BRLE rows (`masks` non-empty, so the second clause is false)
/// AND a channel-bound `AttnMask`, so the batcher merged them and the first
/// concurrent step failed the driver's contract, poisoned descriptor
/// channel 0, and lost all 8 streams.
fn has_dense_device_mask(request: &crate::driver::LaunchPlan) -> bool {
    request.dense_device_mask || (request.has_user_mask && request.masks.is_empty())
}

impl LaunchGrouping {
    pub(crate) fn accepts(
        &self,
        request: &PendingRequest,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> bool {
        if self.instances.contains(&request.instance_id) {
            return false;
        }
        // Wire-geometry (chunk) and device-resolved (chained decode) fires
        // CO-BATCH as ordered sub-batches of one step (true sub-batches,
        // Venus second landing): `build_frame_submission` orders the group
        // wire-first and the driver composes the envelope suffix on device
        // via the offset fixed-decode compose. The mask/solo exclusions
        // below still keep dense-masked and wire-masked device-geometry
        // fires out of shared batches.
        if request
            .pipeline_id
            .is_some_and(|pid| self.pipelines.contains(&pid))
        {
            return false;
        }
        if self.count != 0 && (request.requires_solo_submission() || self.has_solo_submission) {
            return false;
        }
        // RS rows are one per request across the whole composed batch, so a
        // fire that binds recurrent state and one that does not cannot share
        // a wave: the driver would see fewer slot ids than requests.
        if self.count != 0 && (request.rs_batch_kind() == RsBatchKind::None) != !self.has_rs_rows {
            return false;
        }
        // Custom wire masks co-batch freely with other wire-geometry fires —
        // the wire layer emits a mask row per request (synthesized causal for
        // the unmasked ones) and the driver predicates per row. They cannot
        // ride a composed device-geometry batch: wire masks index the wire
        // request layout, which composition replaces (driver fails loud).
        // A DENSE-masked device-resolved fire is stricter still: unlike a
        // host-derived channel mask, it has no wire BRLE rows and the composed
        // path cannot merge it with another program.
        let masked_device_geometry = has_dense_device_mask(&request.request);
        let wire_mask_on_device_geometry = request.request.has_user_mask
            && !request.request.masks.is_empty()
            && request.request.device_resolved_geometry;
        if self.count != 0
            && (masked_device_geometry
                || wire_mask_on_device_geometry
                || (self.has_user_mask && self.has_device_geometry)
                || (request.request.has_user_mask && self.has_device_geometry)
                || (request.request.device_resolved_geometry && self.has_user_mask))
        {
            return false;
        }
        // Hook-program fires: the driver executes ONE hook program per
        // launch (the sideband arena is singular), and a page-list
        // substitution written from a hook needs the PAGED DECODE path —
        // which a batch loses the moment it carries a multi-token row
        // (driver fail: "attn_page_mask was written but this layer does
        // not take the paged decode path"). So a hook fire joins only
        // all-single-token groups with no other hook member, and a
        // multi-token fire never joins past a hook member.
        if self.count != 0
            && ((request.hook_program
                && (self.has_hook_program || self.has_multi_token))
                || (!request.request.single_token_mode && self.has_hook_program))
        {
            return false;
        }
        if self.count == 0 {
            return true;
        }
        let usage = batch::request_capacity_usage(request, page_size);
        self.count.saturating_add(usage.forward_requests) <= limits.max_forward_requests
            && self.forward_tokens.saturating_add(usage.forward_tokens) <= limits.max_forward_tokens
            && self.page_refs.saturating_add(usage.page_refs) <= limits.max_page_refs
    }

    pub(crate) fn push(
        &mut self,
        request: &PendingRequest,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> bool {
        let usage = batch::request_capacity_usage(request, page_size);
        self.instances.insert(request.instance_id);
        if let Some(pid) = request.pipeline_id {
            self.pipelines.insert(pid);
        }
        self.count = self.count.saturating_add(usage.forward_requests);
        self.forward_tokens = self.forward_tokens.saturating_add(usage.forward_tokens);
        self.page_refs = self.page_refs.saturating_add(usage.page_refs);
        self.has_solo_submission |= request.requires_solo_submission();
        self.has_rs_rows |= request.rs_batch_kind() != RsBatchKind::None;
        self.has_user_mask |= request.request.has_user_mask;
        self.has_device_geometry |= request.request.device_resolved_geometry;
        self.has_hook_program |= request.hook_program;
        self.has_multi_token |= !request.request.single_token_mode;
        request.requires_solo_submission()
            || has_dense_device_mask(&request.request)
            || self.count >= limits.max_forward_requests
            || self.forward_tokens >= limits.max_forward_tokens
            || self.page_refs >= limits.max_page_refs
    }
}

/// Mailbox-census buckets (diagnostic; see the epoch-drain loop).
const ITEM_CENSUS_KINDS: [&str; 12] = [
    "launch",
    "reg_chan_bind",
    "close_instance",
    "close_channel",
    "lane_reply",
    "leave",
    "slot_released",
    "slot_consumed",
    "nudge",
    "copy",
    "register_other",
    "other",
];

fn item_census_idx(item: &SchedulerItem) -> usize {
    match item {
        SchedulerItem::Launch { .. } => 0,
        SchedulerItem::RegisterChannelsBind { .. } => 1,
        SchedulerItem::CloseInstance { .. } => 2,
        SchedulerItem::CloseChannel { .. } | SchedulerItem::CloseChannels { .. } => 3,
        SchedulerItem::Lane(_) => 4,
        SchedulerItem::PipelineLeave(..) => 5,
        SchedulerItem::ExecutionSlotReleased(_) | SchedulerItem::ProcessQuiesced(_) => 6,
        SchedulerItem::ExecutionSlotConsumed(_) => 7,
        SchedulerItem::Nudge => 8,
        SchedulerItem::CopyKv { .. }
        | SchedulerItem::CopyKvTracked { .. }
        | SchedulerItem::CopyState { .. }
        | SchedulerItem::ResizePool { .. } => 9,
        SchedulerItem::RegisterProgram { .. }
        | SchedulerItem::RegisterChannel { .. }
        | SchedulerItem::RegisterChannels { .. }
        | SchedulerItem::BindInstance { .. } => 10,
        _ => 11,
    }
}

enum SchedulerItem {
    Launch {
        pending: PendingRequest,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistrationPlan,
        response: tokio::sync::oneshot::Sender<Result<RegisteredChannel>>,
    },
    RegisterChannels {
        plans: Vec<ChannelRegistrationPlan>,
        response: tokio::sync::oneshot::Sender<Result<Vec<RegisteredChannel>>>,
    },
    BindInstance {
        pipeline_id: Option<ProcessId>,
        plan: InstanceBindingPlan,
        response: tokio::sync::oneshot::Sender<Result<BoundInstance>>,
    },
    /// One dispatch registering an instance's channels AND binding it —
    /// the two per-join controls always run back-to-back with only an
    /// ordering dependency, and dispatching them separately doubled the
    /// turnover control convoy (V6 iteration 25 attribution).
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistrationPlan>,
        /// Some on the program cache's first sight (the driver requires the
        /// instance's channels registered BEFORE the program — status -5
        /// otherwise — so registration must ride between channels and bind
        /// inside the one dispatch); None when the hash is already
        /// registered, with `bind.program_id` carrying the cached id.
        program: Option<ProgramRegistration>,
        bind: InstanceBindingPlan,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
    CopyKv {
        plan: crate::driver::KvCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: crate::driver::KvCopyPlan,
        completion: ControlCompletion,
    },
    // Only reached via `SchedulerHandle::copy_state`/`resize_pool`, which
    // the mock-driver fire path doesn't call yet (`scheduler::resize_pool`
    // is exercised by this module's unit tests) — see `scheduler::dispatch`'s
    // module doc for the full driver-ABI-completeness rationale.
    #[allow(dead_code)]
    CopyState {
        plan: StateCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    #[allow(dead_code)]
    ResizePool {
        plan: PoolResizePlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
    },
    CloseChannel {
        id: u64,
    },
    /// A whole cohort of channel closes in one mailbox item — posted by
    /// process teardown, which owns every id it retires. One item per
    /// departing process instead of one per channel keeps a teardown herd
    /// from inflating the epoch a worker pass has to drain.
    CloseChannels {
        ids: Vec<u64>,
    },
    /// Event-driven retirement wake: sent by [`NudgeWaker`] when an in-flight
    /// driver submission completion publishes. Carries no work; it only unblocks the
    /// scheduler's wait so the retire pass runs immediately.
    Nudge,
    /// A pipeline left the fleet ([`notify_pipeline_leave`]'s broadcast).
    /// Handled immediately on dequeue (like [`SchedulerItem::Nudge`]): it
    /// only mutates the run-loop's local [`FramePolicy`] (and, for
    /// Terminate, rejects the pid's queued work), so it can't reorder
    /// control ops or launches.
    /// `.0` is the leaving lane's PIPELINE SCOPE id; `.1` names the owning
    /// PROCESS when the caller knows it. The two are different key spaces
    /// (`FramePolicy::lanes` vs `staged`/`joins_in_flight`/`pending_binds`),
    /// and a leaver that has not fired yet has no lane to recover the owner
    /// from — so the owner must travel with the notification.
    PipelineLeave(
        ProcessId,
        Option<ProcessId>,
        LeaveKind,
        Option<tokio::sync::oneshot::Sender<()>>,
    ),
    /// A capped execution slot was released: the named retiree's deferred
    /// teardown dropped its execution permit ([`notify_execution_slot_released`]'s
    /// broadcast). While the freed slot has a staged taker the frame seal
    /// waits, so a cohort turnover gathers the incoming herd instead of
    /// sealing narrow epochs. Uncapped deployments never send this.
    ExecutionSlotReleased(ProcessId),
    /// The named process's deferred teardown finished; no event from it can
    /// follow. Retires its terminate tombstone.
    ProcessQuiesced(ProcessId),
    /// A parked process acquired its execution permit
    /// ([`notify_execution_slot_consumed`]'s broadcast): the frame seal
    /// waits for this exact process's first fire (identity-paired with the
    /// release above — the two race through the mailbox in either order).
    ExecutionSlotConsumed(ProcessId),
    /// A process is queued for an execution permit; it is the identified
    /// taker of the next slot to free (the semaphore is FIFO-fair).
    AdmissionQueued(ProcessId),
    /// It took the permit, or went away before it could.
    AdmissionDequeued(ProcessId),
    /// The planner concluded a suspended process is runnable again (restore
    /// committed, or the eviction rolled back): its lanes may rejoin the
    /// wait-set and batch full frames again. Process-keyed.
    ProcessResume(ProcessId),
    /// A frame submit failed mid-way host-side: only `submitted` of the
    /// declared fires exist. The frame policy adjusts the lane frame's
    /// expected count so it can still seal (frame mode only; a no-op
    /// otherwise).
    FrameTruncate {
        lane: ProcessId,
        seq: u64,
        submitted: u32,
    },
    /// `forward.park()`: the guest is leaving the seal's wait-set until it
    /// fires again. Ordered against that lane's submits by `seq` — the exit
    /// lands once every frame submitted before it has sealed, so a guest may
    /// park with fires still outstanding (frame mode only; a no-op
    /// otherwise).
    LanePark {
        lane: ProcessId,
        seq: u64,
    },
    /// Snapshot the run loop's state as a human-readable dump (queue
    /// composition, in-flight work, barrier membership). Answered inline on
    /// dequeue — a held wave must be inspectable from outside the thread.
    DebugDump {
        response: tokio::sync::oneshot::Sender<String>,
    },
    /// A driver-lane reply (launch accepted/rejected, control commit).
    /// Handled immediately on dequeue, like `Nudge` — it mutates only
    /// in-flight bookkeeping, never queue order.
    Lane(LaneReply),
    Stop,
}

/// Wakes the scheduler thread through its own queue when a registered driver
/// submission completion publishes, so batch/control retirement is event-driven instead
/// of timeout-polled (plan §5.1).
struct NudgeWaker {
    tx: crossbeam::channel::Sender<SchedulerItem>,
}

impl std::task::Wake for NudgeWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.tx.send(SchedulerItem::Nudge);
    }
}

/// Register the nudge waker on a pending completion's wait slot with
/// register-then-recheck. Returns false when the completion has already
/// settled (or its slot is gone) and the caller should retire immediately.
fn arm_completion_nudge(completion: &SubmissionCompletion, waker: &std::task::Waker) -> bool {
    if completion.is_settled() {
        return false;
    }
    let table = pie_waker::WakerTable::global();
    let slot = completion.wait_id();
    let observed = table.published(slot).unwrap_or_default();
    if !table.register(slot, waker, observed) {
        return false;
    }
    if completion.is_settled() {
        table.deregister(slot);
        return false;
    }
    true
}

#[derive(Clone)]
enum PreLaunchCopy {
    Kv(crate::driver::KvCopyPlan),
    State(StateCopyPlan),
}

impl PreLaunchCopy {
    fn label(&self) -> &'static str {
        match self {
            Self::Kv(_) => "KV copy",
            Self::State(_) => "recurrent-state copy",
        }
    }
}

// =============================================================================
// Driver lane (V6 iteration 48)
// =============================================================================
//
// A dedicated thread owns the `DriverBackend` and executes EVERY driver call
// in FIFO post order, so the driver keeps the exact single-threaded
// serialization it has always had — no driver-side concurrency is introduced.
// What changes is who blocks: launch submit (1.2–2.5 ms) and lifecycle
// controls (0.16 ms p50 with 2–10 ms allocator tails) leave the scheduler
// worker's critical path, which must otherwise fit inside the run-ahead
// pipelining window (the measured 283–437 ms/run of
// sched-lag-after-wait-all — the fast/slow boot-mode spread — and the
// 66–88 ms/run of control-occupancy wave gaps).
//
// Division of state:
// - The lane owns the driver and the `channels` registry set (only control
//   execution ever touched it).
// - The worker keeps ALL policy and admission state: the frame policy, `pending`,
//   `instances` (read by launch admission and gather on every pass). Control
//   arms that used to mutate `instances` inline are split: the driver half
//   runs on the lane, and the map mutation + response happen back on the
//   worker when the lane's reply arrives (`apply_lane_reply`) — keeping the
//   invariant that a bind's response is sent only AFTER the instance is
//   admissible, on the same thread that admits launches.
// - Replies ride the scheduler's own channel (`SchedulerItem::Lane`), so a
//   reply wakes the worker exactly like any other event.

/// A [`FrameSubmission`] in transit to the driver lane.
///
/// SAFETY: the submission is `!Send` only through its
/// `Vec<*mut PieTerminalCell>` — raw pointers into the driver's pinned
/// terminal-cell slots, which are process-stable allocations with no thread
/// affinity (the driver itself reads them from its own threads today). The
/// submission is built complete on the worker, moved to the lane, and
/// consumed exactly once by `driver.launch` — the same single-consumer
/// discipline as the worker-inline call this replaces, with the backing
/// requests kept alive in `in_flight_launches` until the frame retires
/// (retire happens strictly after the lane's reply).
struct LaneLaunch(crate::driver::FrameSubmission);
unsafe impl Send for LaneLaunch {}

/// Worker → lane requests, executed strictly in FIFO order.
enum LaneRequest {
    Launch {
        token: u64,
        submission: LaneLaunch,
    },
    /// A control `QueuedItem` (never `Launch`): the lane runs the driver
    /// half of the old `dispatch_ordered_item` arm.
    Control {
        token: u64,
        item: QueuedItem,
    },
    /// Drain marker: the lane replies with the driver and its channel set so
    /// the worker can run shutdown teardown with everything already quiesced.
    Shutdown {
        response: crossbeam::channel::Sender<(Option<DriverBackend>, HashSet<u64>)>,
    },
}

/// Lane → worker replies (via `SchedulerItem::Lane`).
enum LaneReply {
    LaunchDone {
        token: u64,
        result: std::result::Result<SubmissionCompletion, String>,
        driver_started_us: Option<u64>,
        launch_returned_us: Option<u64>,
    },
    ControlDone {
        token: u64,
        commit: LaneCommit,
    },
}

/// The worker-side half of a control that the lane finished executing.
enum LaneCommit {
    /// Nothing to commit — the lane already sent the response (pure driver
    /// ops that touch no worker state: program/channel registers, channel
    /// closes, failed binds after lane-side rollback).
    None,
    /// A successful bind: insert the instance, THEN respond (launch admission
    /// reads `instances` on the worker thread, so respond-after-insert is the
    /// ordering that makes the guest's first fire admissible).
    BindInstance {
        pipeline_id: Option<ProcessId>,
        bound: BoundInstance,
        respond: BindRespond,
    },
    /// A bind control completed without creating an instance.
    BindFinished { pipeline_id: Option<ProcessId> },
    /// A successful driver-side instance close: remove + close wait slots.
    CloseInstance { id: u64 },
    /// An async-completing control (copies / pool resizes): install the
    /// driver's completion into the pending control slot, or clear the slot
    /// on a synchronous driver rejection.
    AsyncControl {
        result: std::result::Result<SubmissionCompletion, String>,
    },
}

/// Which response shape a successful bind commits to.
enum BindRespond {
    Bind(tokio::sync::oneshot::Sender<Result<BoundInstance>>),
    ChannelsBind {
        registered: Vec<RegisteredChannel>,
        program_id: u64,
        program_registered: bool,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
}

struct DriverLane {
    /// Launch fast path: served before any queued control. A queued launch
    /// and a queued control are ALWAYS mutually independent — a close only
    /// posts once its instance is quiesced (in-flight counts from post), and
    /// a fire can only exist after its bind COMMITTED on the worker — so
    /// preferring launches never reorders a dependent pair. Without the
    /// split, control bursts (a prefix cohort turnover posts hundreds of
    /// closes + binds faster than the lane drains them) head-of-line block
    /// the wave train: measured +7 % prefix regression on the single-FIFO
    /// variant, gaps doubling while sched-lag stayed low.
    launch_tx: crossbeam::channel::Sender<LaneRequest>,
    control_tx: crossbeam::channel::Sender<LaneRequest>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DriverLane {
    fn spawn(
        driver_idx: usize,
        driver: Option<DriverBackend>,
        reply_tx: crossbeam::channel::Sender<SchedulerItem>,
        stats: Arc<SchedulerStats>,
    ) -> Self {
        let (launch_tx, launch_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let (control_tx, control_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let thread = std::thread::Builder::new()
            .name(format!("pie-driver-{driver_idx}"))
            .spawn(move || Self::run(driver, launch_rx, control_rx, reply_tx, stats))
            .expect("spawn pie-driver lane thread");
        Self {
            launch_tx,
            control_tx,
            thread: Some(thread),
        }
    }

    fn post(&self, request: LaneRequest) {
        // The lane outlives every poster (shutdown joins it last); a send
        // failure means the lane thread panicked, which the join reports.
        let _ = match &request {
            LaneRequest::Launch { .. }
            | LaneRequest::Control {
                item: QueuedItem::ResizePool { .. },
                ..
            } => self.launch_tx.send(request),
            LaneRequest::Control { .. } | LaneRequest::Shutdown { .. } => {
                self.control_tx.send(request)
            }
        };
    }

    /// Drain both queues and take the driver + channel set back for
    /// teardown. The worker only calls this with `lane_inflight == 0`, so
    /// both queues are empty and the Shutdown marker is the sole item.
    fn shutdown(&mut self) -> (Option<DriverBackend>, HashSet<u64>) {
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        let _ = self.control_tx.send(LaneRequest::Shutdown {
            response: response_tx,
        });
        let state = response_rx
            .recv()
            .unwrap_or_else(|_| (None, HashSet::new()));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        state
    }

    /// Receive the next request, launches first. Blocks on both queues when
    /// idle; between waves (a cadence of idle gaps) queued controls drain,
    /// so control progress rides the wave rhythm instead of competing with
    /// it.
    fn next_request(
        launch_rx: &crossbeam::channel::Receiver<LaneRequest>,
        control_rx: &crossbeam::channel::Receiver<LaneRequest>,
    ) -> std::result::Result<LaneRequest, ()> {
        use crossbeam::channel::TryRecvError;
        // Stay hot briefly after going empty before parking.
        // The lane hop sits on the enqueue-ahead path: a parked lane pays a
        // thread wake (µs–ms under the box's wake-burst contention) on
        // every submit, which measurably broke run-ahead pipelining
        // (81 % → ~30 % of transitions enqueued ahead on the parked
        // variant). At wave cadence the spin window always covers the next
        // post, so the submit hop costs a cache-hot poll instead; the lane
        // parks only after true idleness (between requests / at shutdown).
        // This is a lane wake optimization, independent of the fire policy.
        const DRIVER_LANE_HOT_US: u64 = 1_000_000;
        let hot_window = Duration::from_micros(DRIVER_LANE_HOT_US);
        let mut spin_until = Instant::now() + hot_window;
        loop {
            match launch_rx.try_recv() {
                Ok(request) => return Ok(request),
                // Both senders live in the worker's `DriverLane` handle and
                // drop together (the graceful path is the Shutdown marker):
                // drain what remains on the other queue, then stop.
                Err(TryRecvError::Disconnected) => {
                    return control_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            match control_rx.try_recv() {
                Ok(request) => return Ok(request),
                Err(TryRecvError::Disconnected) => {
                    return launch_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            if Instant::now() < spin_until {
                std::hint::spin_loop();
                continue;
            }
            let mut select = crossbeam::channel::Select::new();
            select.recv(launch_rx);
            select.recv(control_rx);
            // Only wait; the loop re-runs the launch-first try_recv order
            // (and the disconnect handling above) once something is ready,
            // with a fresh spin window.
            select.ready();
            spin_until = Instant::now() + hot_window;
        }
    }

    fn run(
        mut driver: Option<DriverBackend>,
        launch_rx: crossbeam::channel::Receiver<LaneRequest>,
        control_rx: crossbeam::channel::Receiver<LaneRequest>,
        reply_tx: crossbeam::channel::Sender<SchedulerItem>,
        stats: Arc<SchedulerStats>,
    ) {
        let mut channels: HashSet<u64> = HashSet::new();
        while let Ok(request) = Self::next_request(&launch_rx, &control_rx) {
            match request {
                LaneRequest::Launch { token, submission } => {
                    let LaneLaunch(submission) = submission;
                    let timing_enabled = super::fire_timing_enabled();
                    let driver_started_us = timing_enabled.then(super::fire_timing_now_us);
                    // Folded admission (ABI v14): EXHAUSTED retries in place —
                    // the lane is FIFO, so retrying here preserves global
                    // frame order (later frames must not overtake), and the
                    // physical pool frees resolve on the driver's own
                    // completion threads, never on this lane. Bounded so a
                    // wedged pool converges to a loud failure.
                    const EXHAUSTED_RETRY_SLEEP: Duration = Duration::from_micros(200);
                    const EXHAUSTED_RETRY_MAX: u32 = 25_000; // ~5 s
                    let result = match driver.as_mut() {
                        Some(driver) => crate::probe_fire!(stats.fire.execute.driver_fire_us, {
                            let mut attempts = 0u32;
                            loop {
                                match driver.launch(&submission) {
                                    Ok(crate::driver::FrameLaunchOutcome::Launched(completion)) => {
                                        break Ok(completion);
                                    }
                                    Ok(crate::driver::FrameLaunchOutcome::Exhausted) => {
                                        attempts += 1;
                                        if attempts == 1 || attempts % 1000 == 0 {
                                            tracing::warn!(
                                                attempts,
                                                "frame admission exhausted; lane retrying"
                                            );
                                        }
                                        if attempts > EXHAUSTED_RETRY_MAX {
                                            break Err("frame admission exhausted beyond deadline"
                                                .to_string());
                                        }
                                        std::thread::sleep(EXHAUSTED_RETRY_SLEEP);
                                    }
                                    Ok(crate::driver::FrameLaunchOutcome::Impossible) => {
                                        break Err(
                                            "frame exceeds the driver's physical budget ceiling"
                                                .to_string(),
                                        );
                                    }
                                    Err(err) => break Err(format!("{err:#}")),
                                }
                            }
                        }),
                        None => Err("driver has no backend installed".to_string()),
                    };
                    let launch_returned_us = timing_enabled.then(super::fire_timing_now_us);
                    let _ = reply_tx.send(SchedulerItem::Lane(LaneReply::LaunchDone {
                        token,
                        result,
                        driver_started_us,
                        launch_returned_us,
                    }));
                }
                LaneRequest::Control { token, item } => {
                    let control_timing = super::fire_timing_full().then(|| {
                        (
                            BatchScheduler::item_kind(&item),
                            super::fire_timing_now_us(),
                        )
                    });
                    let commit = Self::execute_control(&mut driver, &mut channels, item);
                    if let Some((kind, started_us)) = control_timing {
                        let finished_us = super::fire_timing_now_us();
                        super::fire_timing_write(&serde_json::json!({
                            "schema": 1,
                            "source": "scheduler",
                            "event": "control_dispatched",
                            "kind": kind,
                            "started_us": started_us,
                            "occupancy_us": finished_us.saturating_sub(started_us),
                        }));
                    }
                    let _ = reply_tx.send(SchedulerItem::Lane(LaneReply::ControlDone {
                        token,
                        commit,
                    }));
                }
                LaneRequest::Shutdown { response } => {
                    let _ = response.send((driver.take(), std::mem::take(&mut channels)));
                    return;
                }
            }
        }
        // Worker dropped its sender without a shutdown handshake (panic
        // path): release the driver here.
        drop(driver.take());
    }

    /// The driver half of the old `dispatch_ordered_item`: everything a
    /// control does against the driver and the lane-owned `channels` set,
    /// with worker-map effects returned as a [`LaneCommit`]. Failures respond
    /// directly from here (after lane-side rollback) — only effects that
    /// must be ordered with worker state travel back.
    fn execute_control(
        driver: &mut Option<DriverBackend>,
        channels: &mut HashSet<u64>,
        item: QueuedItem,
    ) -> LaneCommit {
        match item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.is_settled() => LaneCommit::AsyncControl {
                result: Err("pre-launch copy already settled".to_string()),
            },
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
                LaneCommit::AsyncControl {
                    result: Err("logical fire cancelled before pre-launch copy".to_string()),
                }
            }
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                ..
            } => {
                let operation = plan.label();
                match driver.as_mut() {
                    Some(driver) => {
                        let submitted = match plan {
                            PreLaunchCopy::Kv(plan) => driver.copy_kv(&plan),
                            PreLaunchCopy::State(plan) => driver.copy_state(&plan),
                        };
                        match submitted {
                            Ok(completion) => LaneCommit::AsyncControl {
                                result: Ok(completion),
                            },
                            Err(error) => {
                                let message = format!("pre-launch {operation} rejected: {error:#}");
                                logical_completion.reject_unsubmitted(message.clone());
                                LaneCommit::AsyncControl {
                                    result: Err(message),
                                }
                            }
                        }
                    }
                    None => {
                        logical_completion.reject_unsubmitted("driver has no backend installed");
                        LaneCommit::AsyncControl {
                            result: Err("driver has no backend installed".to_string()),
                        }
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
                let result = match driver.as_mut() {
                    Some(driver) => driver.register_program(&plan),
                    None => Err(anyhow!("driver has no backend installed")),
                };
                match result {
                    Ok(program_id) => {
                        if response.send(Ok(program_id)).is_err() {
                            tracing::warn!(
                                operation = "register_program",
                                program_hash = format_args!("0x{:016x}", plan.program_hash),
                                "scheduler RPC cancelled after program registration; retaining driver-lifetime program"
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
                    Self::release_channel_plan_wait_slots(std::slice::from_ref(&plan));
                    tracing::warn!(
                        operation = "register_channel",
                        channel_id = plan.channel_id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = if channels.contains(&plan.channel_id) {
                    Err(anyhow!("channel {} is already registered", plan.channel_id))
                } else {
                    match driver.as_mut() {
                        Some(driver) => driver.register_channel(&plan).map(|channel| {
                            channels.insert(plan.channel_id);
                            channel
                        }),
                        None => Err(anyhow!("driver has no backend installed")),
                    }
                };
                match result {
                    Ok(channel) => {
                        if let Err(Ok(channel)) = response.send(Ok(channel)) {
                            if let Some(driver) = driver.as_mut() {
                                Self::rollback_channel_set(
                                    driver,
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
                        if response.send(Err(error)).is_err() {
                            Self::release_channel_plan_wait_slots(std::slice::from_ref(&plan));
                        }
                    }
                }
                LaneCommit::None
            }
            QueuedItem::RegisterChannels { plans, response } => {
                if response.is_closed() {
                    Self::release_channel_plan_wait_slots(&plans);
                    tracing::warn!(
                        operation = "register_channels",
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = match driver.as_mut() {
                    Some(driver) => Self::register_channel_set(driver, channels, &plans),
                    None => Err(anyhow!("driver has no backend installed")),
                };
                match result {
                    Ok(registered) => {
                        if let Err(Ok(registered)) = response.send(Ok(registered)) {
                            if let Some(driver) = driver.as_mut() {
                                Self::rollback_channel_set(
                                    driver,
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
                        if response.send(Err(error)).is_err() {
                            Self::release_channel_plan_wait_slots(&plans);
                        }
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
                    DriverLane::release_wait_slots([plan.pacing_wait_id]);
                    tracing::warn!(
                        operation = "bind_instance",
                        requested_instance_id = plan.requested_instance_id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match driver.as_mut() {
                    Some(driver) => match driver.bind_instance(&plan) {
                        Ok(bound) => LaneCommit::BindInstance {
                            pipeline_id,
                            bound,
                            respond: BindRespond::Bind(response),
                        },
                        Err(error) => {
                            if response.send(Err(error)).is_err() {
                                DriverLane::release_wait_slots([plan.pacing_wait_id]);
                            }
                            LaneCommit::BindFinished { pipeline_id }
                        }
                    },
                    None => {
                        if response
                            .send(Err(anyhow!("driver has no backend installed")))
                            .is_err()
                        {
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
                    DriverLane::release_channel_plan_wait_slots(&plans);
                    DriverLane::release_wait_slots([bind.pacing_wait_id]);
                    tracing::warn!(
                        operation = "register_channels_bind",
                        requested_instance_id = bind.requested_instance_id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let Some(driver) = driver.as_mut() else {
                    if response
                        .send(Err(anyhow!("driver has no backend installed")))
                        .is_err()
                    {
                        DriverLane::release_channel_plan_wait_slots(&plans);
                        DriverLane::release_wait_slots([bind.pacing_wait_id]);
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                };
                // Section timing (diagnostic, PIE_FIRE_TIMING): the boundary
                // grinds ~1k of these controls against the frame seal; this
                // breakdown names the engine-side payer next to the driver's
                // own `cuda_bind` record.
                let bind_probe = super::fire_timing_full();
                let probe_t0 = bind_probe.then(Instant::now);
                let registered = match Self::register_channel_set(driver, channels, &plans) {
                    Ok(registered) => registered,
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_channel_plan_wait_slots(&plans);
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        return LaneCommit::BindFinished { pipeline_id };
                    }
                };
                if response.is_closed() {
                    Self::rollback_channel_set(
                        driver,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    DriverLane::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let probe_t1 = bind_probe.then(Instant::now);
                let program_registered = program.is_some();
                if let Some(plan) = &program {
                    match driver.register_program(plan) {
                        Ok(program_id) => bind.program_id = program_id,
                        Err(error) => {
                            Self::rollback_channel_set(
                                driver,
                                channels,
                                &registered,
                                "register_channels_bind",
                                false,
                            );
                            if response.send(Err(error)).is_err() {
                                DriverLane::release_registered_channel_wait_slots(&registered);
                                Self::release_wait_slots([bind.pacing_wait_id]);
                            }
                            return LaneCommit::BindFinished { pipeline_id };
                        }
                    }
                }
                if response.is_closed() {
                    Self::rollback_channel_set(
                        driver,
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
                            program_id = bind.program_id,
                            "scheduler RPC cancelled after program registration; retaining driver-lifetime program"
                        );
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let probe_t2 = bind_probe.then(Instant::now);
                match driver.bind_instance(&bind) {
                    Ok(bound) => {
                        if let (Some(t0), Some(t1), Some(t2)) = (probe_t0, probe_t1, probe_t2) {
                            super::fire_timing_write(&serde_json::json!({
                                "schema": 1,
                                "source": "scheduler",
                                "event": "engine_bind_breakdown",
                                "channels": plans.len(),
                                "set_us": t1.duration_since(t0).as_micros() as u64,
                                "program_us": t2.duration_since(t1).as_micros() as u64,
                                "bind_us": t2.elapsed().as_micros() as u64,
                            }));
                        }
                        LaneCommit::BindInstance {
                            pipeline_id,
                            bound,
                            respond: BindRespond::ChannelsBind {
                                registered,
                                program_id: bind.program_id,
                                program_registered,
                                response,
                            },
                        }
                    }
                    Err(error) => {
                        Self::rollback_channel_set(
                            driver,
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
            QueuedItem::CopyKv { plan, response } => match driver.as_mut() {
                Some(driver) => match driver.copy_kv(&plan) {
                    Ok(completion) => {
                        let _ = response.send(Ok(completion.clone()));
                        LaneCommit::AsyncControl {
                            result: Ok(completion),
                        }
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        let _ = response.send(Err(err));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    let _ = response.send(Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CopyKvTracked { plan, completion } => match driver.as_mut() {
                Some(driver) => match driver.copy_kv(&plan) {
                    Ok(native_completion) => LaneCommit::AsyncControl {
                        result: Ok(native_completion),
                    },
                    Err(error) => {
                        let message = format!("{error:#}");
                        completion.resolve(&Err(error));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    completion.resolve(&Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CopyState { plan, response } => match driver.as_mut() {
                Some(driver) => match driver.copy_state(&plan) {
                    Ok(completion) => {
                        let _ = response.send(Ok(completion.clone()));
                        LaneCommit::AsyncControl {
                            result: Ok(completion),
                        }
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        let _ = response.send(Err(err));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    let _ = response.send(Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::ResizePool { plan, response } => match driver.as_mut() {
                Some(driver) => match driver.resize_pool(&plan) {
                    Ok(completion) => {
                        let _ = response.send(Ok(completion.clone()));
                        LaneCommit::AsyncControl {
                            result: Ok(completion),
                        }
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        let _ = response.send(Err(err));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    let _ = response.send(Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CloseInstance { id, .. } => match driver.as_mut() {
                // The worker already gated existence/pacing/quiescence before
                // posting; the map removal happens at commit.
                Some(driver) => match driver.close_instance(id) {
                    Ok(()) => LaneCommit::CloseInstance { id },
                    Err(err) => {
                        tracing::warn!(instance_id = id, ?err, "scheduler close_instance failed");
                        LaneCommit::None
                    }
                },
                None => {
                    tracing::warn!(instance_id = id, "scheduler has no backend installed");
                    LaneCommit::None
                }
            },
            QueuedItem::CloseChannels { ids } => {
                for id in ids {
                    let result = if !channels.contains(&id) {
                        Err(anyhow!("channel {id} is unknown or stale"))
                    } else {
                        match driver.as_mut() {
                            Some(driver) => driver.close_channel(id).map(|()| {
                                channels.remove(&id);
                            }),
                            None => Err(anyhow!("driver has no backend installed")),
                        }
                    };
                    if let Err(err) = result {
                        tracing::warn!(channel_id = id, ?err, "scheduler close_channel failed");
                    }
                }
                LaneCommit::None
            }
        }
    }

    /// Register a set of channels with all-or-nothing rollback (the shared
    /// body of `RegisterChannels` and `RegisterChannelsBind`).
    fn register_channel_set(
        driver: &mut DriverBackend,
        channels: &mut HashSet<u64>,
        plans: &[ChannelRegistrationPlan],
    ) -> Result<Vec<RegisteredChannel>> {
        let mut registered = Vec::with_capacity(plans.len());
        let mut registered_ids = Vec::with_capacity(plans.len());
        for plan in plans {
            if channels.contains(&plan.channel_id) {
                for channel_id in registered_ids.iter().rev() {
                    let _ = driver.close_channel(*channel_id);
                    channels.remove(channel_id);
                }
                return Err(anyhow!("channel {} is already registered", plan.channel_id));
            }
            match driver.register_channel(plan) {
                Ok(channel) => {
                    channels.insert(plan.channel_id);
                    registered_ids.push(plan.channel_id);
                    registered.push(channel);
                }
                Err(cause) => {
                    for channel_id in registered_ids.iter().rev() {
                        let _ = driver.close_channel(*channel_id);
                        channels.remove(channel_id);
                    }
                    return Err(cause);
                }
            }
        }
        Ok(registered)
    }

    fn rollback_channel_set(
        driver: &mut DriverBackend,
        channels: &mut HashSet<u64>,
        registered: &[RegisteredChannel],
        operation: &'static str,
        cancellation: bool,
    ) {
        for channel in registered.iter().rev() {
            let channel_id = channel.binding.channel_id;
            match driver.close_channel(channel_id) {
                Ok(()) => {
                    channels.remove(&channel_id);
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

    fn release_channel_plan_wait_slots(plans: &[ChannelRegistrationPlan]) {
        Self::release_wait_slots(
            plans
                .iter()
                .flat_map(|plan| [plan.reader_wait_id, plan.writer_wait_id]),
        );
    }

    fn release_registered_channel_wait_slots(registered: &[RegisteredChannel]) {
        Self::release_wait_slots(
            registered
                .iter()
                .flat_map(|channel| [channel.reader_wait_id, channel.writer_wait_id]),
        );
    }

    fn release_wait_slots(wait_ids: impl IntoIterator<Item = u64>) {
        let wait_ids: Vec<u64> = wait_ids.into_iter().collect();
        let table = pie_waker::WakerTable::global();
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

enum QueuedItem {
    /// Boxed: the queue is rotated and compacted item-by-item on every
    /// dispatcher pass, and an inline `PendingRequest` made `QueuedItem`
    /// 1280 bytes — a cohort boundary moved ~800 MB through `VecDeque`
    /// rotations alone. The indirection makes every queue move a pointer
    /// move; the payload itself never moves.
    Launch(QueuedLaunch),
    PreLaunchCopy {
        plan: PreLaunchCopy,
        logical_completion: WorkItemCompletion,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistrationPlan,
        response: tokio::sync::oneshot::Sender<Result<RegisteredChannel>>,
    },
    RegisterChannels {
        plans: Vec<ChannelRegistrationPlan>,
        response: tokio::sync::oneshot::Sender<Result<Vec<RegisteredChannel>>>,
    },
    BindInstance {
        pipeline_id: Option<ProcessId>,
        plan: InstanceBindingPlan,
        response: tokio::sync::oneshot::Sender<Result<BoundInstance>>,
    },
    /// One dispatch registering an instance's channels AND binding it —
    /// the two per-join controls always run back-to-back with only an
    /// ordering dependency, and dispatching them separately doubled the
    /// turnover control convoy (V6 iteration 25 attribution).
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistrationPlan>,
        /// Some on the program cache's first sight (the driver requires the
        /// instance's channels registered BEFORE the program — status -5
        /// otherwise — so registration must ride between channels and bind
        /// inside the one dispatch); None when the hash is already
        /// registered, with `bind.program_id` carrying the cached id.
        program: Option<ProgramRegistration>,
        bind: InstanceBindingPlan,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
    CopyKv {
        plan: crate::driver::KvCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: crate::driver::KvCopyPlan,
        completion: ControlCompletion,
    },
    CopyState {
        plan: StateCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    ResizePool {
        plan: PoolResizePlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
    },
    /// A coalesced run of channel closes: one lane round trip retires the
    /// whole batch (per-id driver calls inside), so a 512-lane cohort's
    /// ~9.2k teardown closes cost ~512 control posts instead of ~9.2k.
    CloseChannels {
        ids: Vec<u64>,
    },
}

/// A queued launch, plus the only two fields the dispatcher's queue scan
/// reads, mirrored inline next to the box.
///
/// [`BatchScheduler::scan_queue`] walks the whole queue and previously read
/// both fields *through* the box, which costs one cache miss per queued item.
/// The scan runs a fixed ~250 times per 1000 tokens no matter how many
/// processes are admitted, so that per-item miss made the scan's cost linear
/// in queue depth and therefore the host's scheduling cost linear in
/// concurrency while the work stayed constant: measured 13.9 us/scan at 256
/// admitted processes against 24.1 us at 512 (mixed-phase shape, same token
/// count both sides), 3.9 s vs 6.5 s of loop time for identical work.
///
/// The mirror cannot go stale, structurally: `QueuedLaunch` hands out only
/// `&PendingRequest` (there is deliberately no `DerefMut`), so neither field
/// can be reassigned while the item is queued. Both are already final by the
/// time they are mirrored — `logical_fire_id` is assigned at construction and
/// the frame stamp is synthesized at ACCEPT, before `queue_attempt` hands the
/// item to the queue.
struct QueuedLaunch {
    fire_id: u64,
    framed: bool,
    request: Box<PendingRequest>,
}

impl QueuedLaunch {
    fn new(request: Box<PendingRequest>) -> Self {
        Self {
            fire_id: request.logical_fire_id,
            framed: request.frame.is_some(),
            request,
        }
    }

    fn into_request(self) -> Box<PendingRequest> {
        self.request
    }
}

impl std::ops::Deref for QueuedLaunch {
    type Target = PendingRequest;
    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

/// A posted launch's lane lifecycle: the batch enters `in_flight_launches`
/// (and the run-ahead depth) at POST; the driver's verdict arrives as a
/// `LaneReply::LaunchDone` and upgrades the state. Retirement only ever
/// consumes `Accepted` (settled) or `Failed` heads — a `Posted` head is
/// simply not ready yet.
enum LaunchState {
    Posted { token: u64 },
    Accepted(SubmissionCompletion),
    Failed(String),
}

struct PendingLaunchBatch {
    state: LaunchState,
    requests: Vec<Box<PendingRequest>>,
    started: Instant,
    batch_size: u64,
    total_tokens: usize,
    timing: Option<WaveTimingState>,
}

/// The control slot's lane lifecycle (async-completing controls only —
/// copies and pool resizes; lifecycle controls never occupy the slot).
enum ControlSlotState {
    Posted { token: u64 },
    Ready(SubmissionCompletion),
}

struct PendingControl {
    state: ControlSlotState,
    logical_completion: Option<WorkItemCompletion>,
    process_id: Option<ProcessId>,
    pipeline_id: Option<ProcessId>,
    tracked_completion: Option<ControlCompletion>,
    operation: &'static str,
    /// Whether launches must wait for this control to settle. True for a
    /// `PreLaunchCopy` (its consumer fire is queued right behind it) and a
    /// pool resize (its pipe drain must not admit new frames under it);
    /// false for standalone copies — their pages are grant-pinned and no
    /// queued fire references them, so frames keep posting while
    /// suspend/restore traffic settles.
    ///
    /// It is also the exclusivity test: a control that holds launches needs
    /// the whole in-flight set empty and blocks every other control while it
    /// settles, exactly as the original single slot did.
    holds_launches: bool,
}

/// The async-completing controls the worker is waiting on — copies and pool
/// resizes; lifecycle controls execute on the lane without ever entering
/// here.
///
/// Two classes share this set. An **exclusive** control (a `PreLaunchCopy`,
/// whose consumer fire is queued directly behind it, and a pool resize, whose
/// pipe drain IS its ordering mechanism) keeps the original rule: it needs
/// the set empty and, once posted, nothing else may join it.
///
/// **Standalone copies** — the residency planner's suspend/restore traffic —
/// instead settle concurrently with one another. Nothing queued orders
/// against them: their pages are grant-pinned and no queued fire can name
/// one, which `pipe_concurrent_control` already relies on. So a single slot
/// bought no safety, only a queue, and the queue was on the planner's
/// critical path. Measured at 512-way KV contention: up to 7 restores wanted
/// the slot at once (`restoring` p90 = 4, max = 7) and each H2D copy took
/// 22.8 ms end to end against ~3.3 ms of transfer — 1.528 ms/page, versus
/// 0.227 ms/page on the D2H side, which the planner itself issues strictly
/// one at a time and which therefore never queued. The 6.7x asymmetry was
/// the wait for this slot, not the device.
///
/// Nothing here needs a concurrency ceiling: the pending queue is the bound.
/// Only copies the planner has already enqueued can be in flight, and the
/// planner enqueues at most one per suspending or restoring process.
#[derive(Default)]
struct InFlightControls {
    settling: Vec<PendingControl>,
}

impl InFlightControls {
    fn is_empty(&self) -> bool {
        self.settling.is_empty()
    }

    /// Whether anything is still settling.
    fn is_settling(&self) -> bool {
        !self.settling.is_empty()
    }

    fn iter(&self) -> std::slice::Iter<'_, PendingControl> {
        self.settling.iter()
    }

    /// Whether a standalone copy may be posted now: only an exclusive
    /// control can refuse one.
    fn admits_copy(&self) -> bool {
        self.settling.iter().all(|control| !control.holds_launches)
    }

    /// Whether `item` may be posted into this set now. A standalone copy and
    /// a lifecycle control are each refused only by an EXCLUSIVE control: the
    /// copy addresses grant-pinned pages nothing queued can name, and a
    /// lifecycle control never enters this set at all — it executes on the
    /// lane, its driver order guaranteed by the lane FIFO. Blocking lifecycle
    /// controls on a settling copy wedged the fleet: the planner's copies are
    /// in flight almost continuously under churn, so every bind waited out the
    /// whole strict-watchdog window (measured on `churn`: 270 binds at 1.0-2.4 s
    /// end to end against a 59 us driver bind).
    fn admits(&self, item: &QueuedItem) -> bool {
        if BatchScheduler::standalone_copy(item) || BatchScheduler::lifecycle_control(item) {
            !self.holds_launches()
        } else {
            self.is_empty()
        }
    }

    /// Whether any settling control makes queued launches wait.
    fn holds_launches(&self) -> bool {
        self.settling.iter().any(|control| control.holds_launches)
    }

    fn push(&mut self, control: PendingControl) {
        self.settling.push(control);
    }

    fn position_posted(&self, token: u64) -> Option<usize> {
        self.settling.iter().position(
            |control| matches!(control.state, ControlSlotState::Posted { token: t } if t == token),
        )
    }
}

/// What one pass over the pending queue tells the frame dispatcher — see
/// [`BatchScheduler::scan_queue`].
#[derive(Default)]
struct QueueScan {
    /// Stamped fire ids still in the queue (the frame policy resolves
    /// sealed ids that vanished against this set).
    queued_ids: frame::QueuedFireIds,
    /// Lanes a frame post must hold for: only lanes with a queued
    /// `PreLaunchCopy` (order-coupled to its consumer fire).
    blocked_lanes: HashSet<ProcessId>,
    /// The oldest unstamped rider, dispatched as its own batch.
    untracked: Option<u64>,
    /// Every queued fire id in queue order, for the shutdown drain. Only
    /// filled while `stopping` — the steady-state scan never allocates it.
    drain_eligible: Vec<u64>,
}

impl QueueScan {
    /// Reset for reuse, keeping the allocations.
    fn clear(&mut self) {
        self.queued_ids.clear();
        self.blocked_lanes.clear();
        self.untracked = None;
        self.drain_eligible.clear();
    }
}

/// The worker's pending queue, plus an epoch that changes on every mutation.
///
/// The epoch exists so [`BatchScheduler::scan_queue`] can skip a pass whose
/// answer cannot have changed. `DerefMut` bumps it, which is what makes the
/// invalidation total: every `&mut` reach into the queue counts, including
/// rotations that leave the length alone, in-place edits through `iter_mut`,
/// and the rebuild in `post_frame`. A length or endpoint fingerprint would
/// have missed all three. Over-invalidation (a `&mut` taken but not used) is
/// merely a wasted scan.
#[derive(Default)]
struct PendingQueue {
    items: VecDeque<QueuedItem>,
    epoch: u64,
    /// `(epoch, index of the first non-`Launch` item)`.
    first_other: Option<(u64, Option<usize>)>,
    /// `(epoch, index of the first queued close)`.
    first_close: Option<(u64, usize)>,
}

impl PendingQueue {
    fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Replace the contents wholesale, preserving the epoch counter.
    fn replace(&mut self, items: VecDeque<QueuedItem>) {
        self.items = items;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Offset of the first item that is not a `Launch`, or `None` when the
    /// queue is all launches.
    ///
    /// Both this and [`Self::first_close`] answer "where does the launch run
    /// end", which every worker pass asks at least once. At a cohort boundary
    /// the queue is a run of thousands of launches held back by an unsealed
    /// frame, so the linear search — run once per pass and once per queued
    /// control — was measured as the loop's single largest per-pass cost
    /// (~18 us/pass against ~3 us elsewhere). The scan itself is unavoidable;
    /// repeating it while the queue is unchanged is not, so both are cached
    /// against the same epoch that guards [`ScanCache`].
    fn first_other(&mut self) -> Option<usize> {
        if let Some((epoch, idx)) = self.first_other
            && epoch == self.epoch
        {
            return idx;
        }
        let idx = self
            .items
            .iter()
            .position(|item| !matches!(item, QueuedItem::Launch(_)));
        self.first_other = Some((self.epoch, idx));
        idx
    }

    /// Offset of the first queued close, or the length when there is none.
    fn first_close(&mut self) -> usize {
        if let Some((epoch, idx)) = self.first_close
            && epoch == self.epoch
        {
            return idx;
        }
        let idx = self
            .items
            .iter()
            .position(|item| {
                matches!(
                    item,
                    QueuedItem::CloseInstance { .. } | QueuedItem::CloseChannels { .. }
                )
            })
            .unwrap_or(self.items.len());
        self.first_close = Some((self.epoch, idx));
        idx
    }

    /// Move the leading run of launches behind the rest of the queue.
    ///
    /// Equivalent to popping each leading launch off the front and pushing it
    /// back, but as one rotation rather than thousands of individual moves.
    fn rotate_launch_run_to_back(&mut self, run_len: usize) {
        self.items.rotate_left(run_len);
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Insert a bring-up control ahead of the trailing close run.
    fn insert_before_closes(&mut self, item: QueuedItem) {
        let index = self.first_close();
        self.items.insert(index, item);
        self.epoch = self.epoch.wrapping_add(1);
        // The insert shifted the close run one to the right and put a
        // non-close at `index`, so the next control lands after this one and
        // bind-vs-bind order is preserved without rescanning.
        self.first_close = Some((self.epoch, index + 1));
        self.first_other = None;
    }
}

impl std::ops::Deref for PendingQueue {
    type Target = VecDeque<QueuedItem>;
    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

impl std::ops::DerefMut for PendingQueue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.epoch = self.epoch.wrapping_add(1);
        &mut self.items
    }
}

impl From<VecDeque<QueuedItem>> for PendingQueue {
    fn from(items: VecDeque<QueuedItem>) -> Self {
        Self {
            items,
            epoch: 0,
            first_other: None,
            first_close: None,
        }
    }
}

impl FromIterator<QueuedItem> for PendingQueue {
    fn from_iter<T: IntoIterator<Item = QueuedItem>>(iter: T) -> Self {
        Self {
            items: iter.into_iter().collect(),
            epoch: 0,
            first_other: None,
            first_close: None,
        }
    }
}

/// A [`QueueScan`] plus the queue epoch it was taken at.
#[derive(Default)]
struct ScanCache {
    scan: QueueScan,
    /// `None` until the first scan; otherwise the (epoch, stopping) the
    /// cached scan is valid for.
    taken_at: Option<(u64, bool)>,
}

/// Reused across frames: `post_frame` places each picked fire into its
/// sealed slot, and a fresh `Vec` per frame would be a ~650 KB allocation on
/// the loop's critical path. Compaction `take()`s every slot, so the buffer
/// comes back empty and only ever grows.
type SlotBuffer = Vec<Vec<Option<Box<PendingRequest>>>>;

struct SchedulerControl {
    tx: crossbeam::channel::Sender<SchedulerItem>,
    active_senders: AtomicUsize,
    shutdown_wait: Condvar,
    shutdown_gate: Mutex<()>,
    program_ids: Mutex<HashMap<u64, (u64, pie_driver_abi::plan::LaunchPackage)>>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
}

#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    inner: Arc<SchedulerControl>,
}

impl SchedulerHandle {
    fn send(&self, item: SchedulerItem) -> Result<()> {
        if !self.inner.accepting.load(Ordering::SeqCst) {
            return Err(anyhow!("scheduler shutting down"));
        }
        self.inner.active_senders.fetch_add(1, Ordering::SeqCst);
        if !self.inner.accepting.load(Ordering::SeqCst) {
            self.finish_send();
            return Err(anyhow!("scheduler shutting down"));
        }
        let result = self
            .inner
            .tx
            .send(item)
            .map_err(|_| anyhow!("scheduler channel closed"));
        self.finish_send();
        result
    }

    fn finish_send(&self) {
        if self.inner.active_senders.fetch_sub(1, Ordering::SeqCst) == 1
            && !self.inner.accepting.load(Ordering::SeqCst)
        {
            let _guard = self.inner.shutdown_gate.lock().unwrap();
            self.inner.shutdown_wait.notify_all();
        }
    }

    fn begin_shutdown(&self) {
        if !self.inner.accepting.swap(false, Ordering::SeqCst) {
            return;
        }
        let mut guard = self.inner.shutdown_gate.lock().unwrap();
        while self.inner.active_senders.load(Ordering::SeqCst) != 0 {
            guard = self.inner.shutdown_wait.wait(guard).unwrap();
        }
        let _ = self.inner.tx.send(SchedulerItem::Stop);
    }

    async fn request<T>(
        &self,
        make: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> SchedulerItem,
    ) -> Result<T> {
        let (response, receiver) = tokio::sync::oneshot::channel();
        self.send(make(response))?;
        receiver
            .await
            .map_err(|_| anyhow!("scheduler channel closed"))
    }

    /// This driver's lock-free stats snapshot (read by
    /// `scheduler::get_stats`'s cross-driver aggregation).
    pub(crate) fn stats(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.inner.stats)
    }

    pub fn submit_with_identity_and_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        pipeline_id: Option<ProcessId>,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        timing_enabled: bool,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                last_page_len,
                pipeline_id,
                pipeline_id,
                false,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                timing_enabled,
                /*hook_program=*/false,
                /*lora_program=*/false,
            ),
        })
    }

    pub fn submit_prebuilt_with_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                last_page_len,
                None,
                None,
                true,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                super::fire_timing_full(),
                /*hook_program=*/false,
                /*lora_program=*/false),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_prebuilt_tracked_with_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        process_id: ProcessId,
        pipeline_id: ProcessId,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        frame: Option<FrameStamp>,
        timing_enabled: bool,
        hook_program: bool,
        lora_program: bool,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                last_page_len,
                Some(process_id),
                Some(pipeline_id),
                true,
                prelaunch_copy,
                prelaunch_state_copy,
                frame,
                timing_enabled,
                hook_program,
                lora_program),
        })
    }

    /// Fire-and-forget frame truncation notice (frame mode): a host frame
    /// submit failed mid-way, so only `submitted` fires of (lane, seq) exist.
    pub fn frame_truncate(&self, lane: ProcessId, seq: u64, submitted: u32) -> Result<()> {
        self.send(SchedulerItem::FrameTruncate {
            lane,
            seq,
            submitted,
        })
    }

    pub(crate) fn nudge(&self) -> Result<()> {
        self.send(SchedulerItem::Nudge)
    }
    pub async fn register_program(&self, plan: ProgramRegistration) -> Result<u64> {
        let program_hash = plan.program_hash;
        {
            let program_ids = self.inner.program_ids.lock().unwrap();
            if let Some((program_id, launch)) = program_ids.get(&program_hash) {
                if launch != &plan.launch {
                    return Err(anyhow!("program hash collision for 0x{program_hash:016x}"));
                }
                return Ok(*program_id);
            }
        }
        let launch = plan.launch.clone();
        let program_id = self
            .request(|response| SchedulerItem::RegisterProgram { plan, response })
            .await??;
        self.inner
            .program_ids
            .lock()
            .unwrap()
            .insert(program_hash, (program_id, launch));
        Ok(program_id)
    }

    pub async fn register_channel(
        &self,
        plan: ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        self.request(|response| SchedulerItem::RegisterChannel { plan, response })
            .await?
    }

    pub async fn register_channels(
        &self,
        plans: Vec<ChannelRegistrationPlan>,
    ) -> Result<Vec<RegisteredChannel>> {
        self.request(|response| SchedulerItem::RegisterChannels { plans, response })
            .await?
    }

    pub async fn bind_instance(
        &self,
        pipeline_id: Option<ProcessId>,
        plan: InstanceBindingPlan,
    ) -> Result<BoundInstance> {
        self.request(|response| SchedulerItem::BindInstance {
            pipeline_id,
            plan,
            response,
        })
        .await?
    }

    pub async fn register_channels_bind(
        &self,
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistrationPlan>,
        program: ProgramRegistration,
        mut bind: InstanceBindingPlan,
    ) -> Result<(Vec<RegisteredChannel>, BoundInstance)> {
        let program_hash = program.program_hash;
        let cached = {
            let program_ids = self.inner.program_ids.lock().unwrap();
            match program_ids.get(&program_hash) {
                Some((program_id, launch)) => {
                    if launch != &program.launch {
                        return Err(anyhow!("program hash collision for 0x{program_hash:016x}"));
                    }
                    Some(*program_id)
                }
                None => None,
            }
        };
        let (program_field, cache_fill) = match cached {
            Some(program_id) => {
                bind.program_id = program_id;
                (None, None)
            }
            None => (Some(program.clone()), Some(program.launch)),
        };
        let (registered, program_id, bound) = self
            .request(|response| SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program: program_field,
                bind,
                response,
            })
            .await??;
        if let Some(launch) = cache_fill {
            self.inner
                .program_ids
                .lock()
                .unwrap()
                .insert(program_hash, (program_id, launch));
        }
        Ok((registered, bound))
    }

    pub async fn copy_kv(&self, plan: crate::driver::KvCopyPlan) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::CopyKv { plan, response })
            .await?
    }

    pub(crate) fn copy_kv_tracked(
        &self,
        plan: crate::driver::KvCopyPlan,
    ) -> Result<ControlCompletion> {
        let completion = ControlCompletion::new();
        self.send(SchedulerItem::CopyKvTracked {
            plan,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    /// Human-readable snapshot of the run loop's state (see
    /// [`SchedulerItem::DebugDump`]).
    pub(crate) async fn debug_dump(&self) -> Result<String> {
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.request(|response| SchedulerItem::DebugDump { response }),
        )
        .await
        .map_err(|_| anyhow!("scheduler did not answer the debug dump"))?
    }

    // Only called from `scheduler::dispatch::copy_rs_d2d`/`resize_pool`
    // (not yet issued by the mock-driver fire path) and this module's own
    // unit tests — see `scheduler::dispatch`'s module doc.
    #[allow(dead_code)]
    pub async fn copy_state(&self, plan: StateCopyPlan) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::CopyState { plan, response })
            .await?
    }

    #[allow(dead_code)]
    pub async fn resize_pool(&self, plan: PoolResizePlan) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::ResizePool { plan, response })
            .await?
    }

    pub fn close_instance(&self, id: u64, pacing_wait_id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseInstance { id, pacing_wait_id })
    }

    pub fn close_channel(&self, id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseChannel { id })
    }

    /// Batched form of [`Self::close_channel`] for callers that retire a
    /// whole cohort of channels at once (process teardown).
    pub fn close_channels(&self, ids: Vec<u64>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.send(SchedulerItem::CloseChannels { ids })
    }
}

pub struct BatchScheduler {
    driver_id: DriverId,
    handle: SchedulerHandle,
    thread: Option<std::thread::JoinHandle<()>>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    pub fn new(
        driver_id: DriverId,
        driver_idx: usize,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
        frame_size: usize,
    ) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded::<SchedulerItem>();
        let stats = Arc::new(SchedulerStats::default());
        let handle = SchedulerHandle {
            inner: Arc::new(SchedulerControl {
                tx,
                active_senders: AtomicUsize::new(0),
                shutdown_wait: Condvar::new(),
                shutdown_gate: Mutex::new(()),
                program_ids: Mutex::new(HashMap::new()),
                accepting: AtomicBool::new(true),
                stats: Arc::clone(&stats),
            }),
        };
        crate::scheduler::install_scheduler_handle(driver_id, handle.clone());
        let stats_for_loop = Arc::clone(&stats);
        let nudge_tx = handle.inner.tx.clone();
        let thread = std::thread::Builder::new()
            .name(format!("pie-sched-{driver_idx}"))
            .spawn(move || {
                let _request_timeout = Duration::from_secs(request_timeout_secs);
                Self::run(
                    driver_id,
                    rx,
                    nudge_tx,
                    page_size,
                    limits,
                    stats_for_loop,
                    frame_size,
                );
            })
            .expect("spawn pie-sched thread");
        Self {
            driver_id,
            handle,
            thread: Some(thread),
            stats,
        }
    }

    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    fn shutdown(&mut self) {
        self.handle.begin_shutdown();
        crate::scheduler::clear_scheduler_handle(self.driver_id);
        if let Some(thread) = self.thread.take() {
            if let Err(err) = thread.join() {
                tracing::error!(
                    driver_id = self.driver_id,
                    ?err,
                    "scheduler thread panicked"
                );
            }
        }
    }

    fn run(
        driver_id: DriverId,
        rx: crossbeam::channel::Receiver<SchedulerItem>,
        nudge_tx: crossbeam::channel::Sender<SchedulerItem>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: Arc<SchedulerStats>,
        frame_size: usize,
    ) {
        let lane_reply_tx = nudge_tx.clone();
        let nudge_waker = std::task::Waker::from(Arc::new(NudgeWaker {
            tx: nudge_tx.clone(),
        }));
        let driver = crate::driver::take_driver_backend(driver_id).ok();
        let mut lane = DriverLane::spawn(driver_id, driver, lane_reply_tx, Arc::clone(&stats));
        // Worker→lane requests not yet replied to (launch posts + control
        // posts). Shutdown may only tear down once this drains — every lane
        // request produces exactly one reply.
        let mut lane_inflight: u64 = 0;
        let mut lane_token: u64 = 0;
        let mut instances = HashMap::new();
        let mut pending = PendingQueue::default();
        let mut scan_cache = ScanCache::default();
        let mut slot_buffer = SlotBuffer::new();
        let mut terminated_processes: HashSet<ProcessId> = HashSet::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = InFlightControls::default();
        let mut stopping = false;
        // THE fire rule: the wait-for-all-active-lanes frame policy, one
        // instance per driver thread, mirroring `instances`/`channels` above.
        // Every backend and every k schedules the same way — at the default
        // `PIE_FRAME_SIZE=1` a frame is one wave (each tracked fire admits as
        // a synthesized single-slot frame); density comes from the sealed
        // epoch, throughput from run-ahead depth within it.
        let mut frame_policy = FramePolicy::new(
            frame_size,
            limits.max_forward_requests,
            limits.max_forward_tokens,
            Some(Arc::clone(&stats)),
        );
        frame_policy
            .preload_free_slots(crate::inferlet::process::execution_slot_capacity().unwrap_or(0));
        // Stall self-diagnosis: a scheduler that spins on the backstop with
        // queued or in-flight work and zero progress is deadlocked from the
        // caller's point of view, and every wait in this loop is silent. After
        // 10s of that, print the full state dump so the wedge names itself
        // (then re-print every 60s while it persists).
        let mut stall_since: Option<std::time::Instant> = None;
        let mut stall_dumps: u32 = 0;

        loop {
            let mut progress = false;
            // Worker-pass timing probe (PIE_FIRE_TIMING): the generation
            // boundary showed fires arriving within ~28 ms while waves
            // started at ~+170 ms — this names which pass phase eats the
            // difference (mailbox drain vs retire vs dispatch) and how the
            // pending queue length scales it.
            let probe = super::fire_timing_enabled();
            let pass_started = Instant::now();
            let mut mailbox_items: u32 = 0;
            // Epoch drain: a pass consumes only what was queued when it began.
            // A sustained producer flood (the next cohort's bring-up at a
            // generation boundary) otherwise keeps `try_recv` non-empty for
            // tens of milliseconds and holds retire/dispatch hostage behind
            // the live stream — the seal-opening events (leaves, slot
            // releases, first fires) then reach the policy tail-late and the
            // boundary wave dispatches when the mailbox finally runs dry
            // instead of the pass after the last join lands.
            let mailbox_epoch = rx.len();
            // Per-variant census (diagnostic, PIE_FIRE_TIMING): which item
            // class the boundary epochs are made of, count and time — the
            // flood's composition decides the next lever.
            let mut census_n = [0u32; ITEM_CENSUS_KINDS.len()];
            let mut census_ns = [0u64; ITEM_CENSUS_KINDS.len()];
            for _ in 0..mailbox_epoch {
                let Ok(item) = rx.try_recv() else { break };
                progress = true;
                mailbox_items += 1;
                let kind = if probe { item_census_idx(&item) } else { 0 };
                let item_started = if probe { Some(Instant::now()) } else { None };
                match item {
                    SchedulerItem::DebugDump { response } => {
                        let _ = response.send(Self::render_debug_dump(
                            &pending,
                            &in_flight_launches,
                            &in_flight_control,
                            &instances,
                            &frame_policy,
                        ));
                    }
                    SchedulerItem::Lane(reply) => {
                        Self::apply_lane_reply(
                            reply,
                            &mut lane_inflight,
                            &mut in_flight_launches,
                            &mut in_flight_control,
                            &mut instances,
                            &mut frame_policy,
                            &nudge_tx,
                        );
                    }
                    item => {
                        Self::enqueue_item(
                            &mut pending,
                            &mut terminated_processes,
                            &mut in_flight_control,
                            &instances,
                            limits,
                            page_size,
                            &mut stopping,
                            &mut frame_policy,
                            item,
                        );
                    }
                }
                if let Some(item_started) = item_started {
                    census_n[kind] += 1;
                    census_ns[kind] += item_started.elapsed().as_nanos() as u64;
                }
            }
            let mailbox_done = Instant::now();
            progress |= Self::retire_ready_launches(
                &mut in_flight_launches,
                &mut instances,
                &mut pending,
                &stats,
                &mut frame_policy,
            );
            progress |= Self::retire_ready_control(&mut in_flight_control);
            let retire_done = Instant::now();
            let (dispatched, wait_hint) = Self::dispatch_ready_items(
                &lane,
                &mut lane_inflight,
                &mut lane_token,
                &mut instances,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                page_size,
                limits,
                &stats,
                &mut frame_policy,
                &mut scan_cache,
                &mut slot_buffer,
                stopping,
            );
            progress |= dispatched;
            if probe {
                let dispatch_ns = retire_done.elapsed().as_nanos() as u64;
                let acc = &super::LOOP_PHASES;
                acc.mailbox_ns.fetch_add(
                    mailbox_done.duration_since(pass_started).as_nanos() as u64,
                    Ordering::Relaxed,
                );
                acc.retire_ns.fetch_add(
                    retire_done.duration_since(mailbox_done).as_nanos() as u64,
                    Ordering::Relaxed,
                );
                acc.dispatch_ns.fetch_add(dispatch_ns, Ordering::Relaxed);
                acc.passes.fetch_add(1, Ordering::Relaxed);
                acc.pass_max_ns
                    .fetch_max(pass_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                acc.mailbox_items
                    .fetch_add(mailbox_items as u64, Ordering::Relaxed);
            }
            if probe {
                let dispatch_us = retire_done.elapsed().as_micros() as u64;
                let mailbox_us = mailbox_done.duration_since(pass_started).as_micros() as u64;
                let retire_us = retire_done.duration_since(mailbox_done).as_micros() as u64;
                if mailbox_us + retire_us + dispatch_us > 2_000 {
                    let census: serde_json::Map<String, serde_json::Value> = ITEM_CENSUS_KINDS
                        .iter()
                        .zip(census_n.iter().zip(census_ns.iter()))
                        .filter(|(_, (n, _))| **n > 0)
                        .map(|(name, (n, ns))| {
                            ((*name).to_string(), serde_json::json!([n, ns / 1_000]))
                        })
                        .collect();
                    super::fire_timing_write(&serde_json::json!({
                        "schema": 1,
                        "source": "scheduler",
                        "event": "worker_pass",
                        "pass_started_us": super::fire_timing_now_us(),
                        "mailbox_us": mailbox_us,
                        "mailbox_items": mailbox_items,
                        "census": census,
                        "retire_us": retire_us,
                        "dispatch_us": dispatch_us,
                        "pending_len": pending.len(),
                        "progress": progress,
                    }));
                }
            }
            if stopping
                && pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_empty()
                && lane_inflight == 0
            {
                break;
            }

            // Cohort-boundary bind deferral: while a successor's arrival is
            // imminent, hold back the bind permits that retiring
            // processes return, so the staged cohort's working-set
            // declaration and prefill construction do not compete with the
            // boundary's own bring-up. Cleared the moment this pass has
            // nothing left to do, which is what makes the hold incapable of
            // being the last thing standing.
            crate::inferlet::process::set_bind_release_hold(
                !stopping
                    && frame_policy.is_joining()
                    && (progress
                        || !in_flight_launches.is_empty()
                        || in_flight_control.is_settling()),
            );

            if progress {
                stall_since = None;
                stall_dumps = 0;
                continue;
            }

            let item = if pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_empty()
                && !stopping
            {
                match rx.recv() {
                    Ok(item) => Some(item),
                    Err(_) => {
                        stopping = true;
                        None
                    }
                }
            } else {
                // Event-driven retirement: park the nudge waker on the oldest
                // in-flight completions so the driver callback wakes this
                // thread the moment one publishes. The timeout is only a hang
                // backstop, never the steady-state wake path.
                let mut armed = true;
                if let Some(front) = in_flight_launches.front() {
                    match &front.state {
                        // A posted launch's reply arrives on the scheduler
                        // channel itself — recv() IS the wake path.
                        LaunchState::Posted { .. } => {}
                        LaunchState::Accepted(completion) => {
                            armed &= arm_completion_nudge(completion, &nudge_waker);
                        }
                        // A failed launch is retire-ready right now.
                        LaunchState::Failed(_) => armed = false,
                    }
                }
                for control in in_flight_control.iter() {
                    match &control.state {
                        ControlSlotState::Posted { .. } => {}
                        ControlSlotState::Ready(completion) => {
                            armed &= arm_completion_nudge(completion, &nudge_waker);
                        }
                    }
                }
                if !armed {
                    // Something already settled; retire it on the next pass.
                    continue;
                }
                // A pending wait-all hold (cold gather / seal barrier /
                // depth-cap poll) re-arms the backstop at its own
                // cadence — never longer than the 250ms hang backstop, so a
                // held wave still fires on time even with no new arrival or
                // completion nudge in between.
                let backstop = Duration::from_millis(250);
                let recv_wait = wait_hint.map(|hold| hold.min(backstop)).unwrap_or(backstop);
                let park_started = probe.then(Instant::now);
                let parked = rx.recv_timeout(recv_wait);
                if let Some(park_started) = park_started {
                    super::LOOP_PHASES
                        .park_ns
                        .fetch_add(park_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                match parked {
                    Ok(item) => Some(item),
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                        // A settled completion discovered by the backstop
                        // means a wake was lost somewhere — the steady-state
                        // count stays zero (plan §16.2). Shutdown races are
                        // excluded: teardown may legitimately cross a tick.
                        // A wait-all-hold timeout is NOT a lost wake (it is
                        // the wait's own cadence), so it never counts here.
                        let missed = in_flight_launches.front().is_some_and(|front| {
                            matches!(&front.state, LaunchState::Accepted(c) if c.is_settled())
                        }) || in_flight_control.iter().any(|control| {
                            matches!(&control.state, ControlSlotState::Ready(c) if c.is_settled())
                        });
                        if missed && !stopping && wait_hint.is_none() {
                            let total = BACKSTOP_RETIREMENTS.fetch_add(1, Ordering::Relaxed) + 1;
                            tracing::warn!(
                                driver_id,
                                total,
                                "completion retired by the backstop poll, not the nudge"
                            );
                        }
                        let stalled_for = stall_since
                            .get_or_insert_with(std::time::Instant::now)
                            .elapsed();
                        if stalled_for
                            >= Duration::from_secs(10)
                                .saturating_add(Duration::from_secs(60) * stall_dumps)
                        {
                            stall_dumps += 1;
                            eprintln!(
                                "[pie-sched] driver {driver_id} stalled for {stalled_for:?} \
                                 (no progress, work queued or in flight); state:\n{}",
                                Self::render_debug_dump(
                                    &pending,
                                    &in_flight_launches,
                                    &in_flight_control,
                                    &instances,
                                    &frame_policy,
                                ),
                            );
                        }
                        None
                    }
                    Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                        stopping = true;
                        None
                    }
                }
            };

            if let Some(item) = item {
                if let SchedulerItem::DebugDump { response } = item {
                    let _ = response.send(Self::render_debug_dump(
                        &pending,
                        &in_flight_launches,
                        &in_flight_control,
                        &instances,
                        &frame_policy,
                    ));
                    continue;
                }
                if let SchedulerItem::Lane(reply) = item {
                    Self::apply_lane_reply(
                        reply,
                        &mut lane_inflight,
                        &mut in_flight_launches,
                        &mut in_flight_control,
                        &mut instances,
                        &mut frame_policy,
                        &nudge_tx,
                    );
                    continue;
                }
                Self::enqueue_item(
                    &mut pending,
                    &mut terminated_processes,
                    &mut in_flight_control,
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
                    &mut frame_policy,
                    item,
                );
            }
        }

        // The lane has no pending requests here (`lane_inflight == 0` gates
        // the loop exit), so shutdown returns the quiesced driver and the
        // channel registry for teardown.
        let (mut driver, mut channels) = lane.shutdown();
        Self::shutdown_instances(&mut driver, &mut instances);
        Self::shutdown_channels(&mut driver, &mut channels);
        drop(driver.take());
    }

    #[allow(clippy::too_many_arguments)]
    fn render_debug_dump(
        pending: &VecDeque<QueuedItem>,
        in_flight_launches: &VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        instances: &HashMap<u64, TrackedInstance>,
        frame_policy: &FramePolicy,
    ) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let describe = |request: &PendingRequest| {
            format!(
                "fire {} instance {} pipeline {:?} tracked={} settled={} cancelled={}",
                request.logical_fire_id,
                request.instance_id,
                request.pipeline_id,
                instances.contains_key(&request.instance_id),
                request.completion.is_settled(),
                request.completion.cancel_requested(),
            )
        };
        let _ = writeln!(out, "pending ({}):", pending.len());
        for item in pending {
            let line = match item {
                QueuedItem::Launch(request) => format!("Launch: {}", describe(request)),
                QueuedItem::PreLaunchCopy {
                    plan, pipeline_id, ..
                } => format!("PreLaunchCopy({}) pipeline {pipeline_id:?}", plan.label()),
                QueuedItem::RegisterProgram { .. } => "RegisterProgram".to_string(),
                QueuedItem::RegisterChannel { .. } => "RegisterChannel".to_string(),
                QueuedItem::RegisterChannels { plans, .. } => {
                    format!("RegisterChannels({})", plans.len())
                }
                QueuedItem::BindInstance { .. } => "BindInstance".to_string(),
                QueuedItem::RegisterChannelsBind { .. } => "RegisterChannelsBind".to_string(),
                QueuedItem::CopyKv { .. } => "CopyKv".to_string(),
                QueuedItem::CopyKvTracked { .. } => "CopyKvTracked".to_string(),
                QueuedItem::CopyState { .. } => "CopyState".to_string(),
                QueuedItem::ResizePool { .. } => "ResizePool".to_string(),
                QueuedItem::CloseInstance { id, .. } => format!("CloseInstance {id}"),
                QueuedItem::CloseChannels { ids } => format!("CloseChannels x{}", ids.len()),
            };
            let _ = writeln!(out, "  {line}");
        }
        let _ = writeln!(out, "in_flight_launches ({}):", in_flight_launches.len());
        for batch in in_flight_launches {
            let state = match &batch.state {
                LaunchState::Posted { token } => format!("posted(token={token})"),
                LaunchState::Accepted(c) => format!("settled={}", c.is_settled()),
                LaunchState::Failed(msg) => format!("failed({msg})"),
            };
            let _ = writeln!(
                out,
                "  batch of {} ({state}, age={:?})",
                batch.requests.len(),
                batch.started.elapsed(),
            );
        }
        if in_flight_control.is_empty() {
            let _ = writeln!(out, "in_flight_control: none");
        }
        for control in in_flight_control.iter() {
            let state = match &control.state {
                ControlSlotState::Posted { token } => format!("posted(token={token})"),
                ControlSlotState::Ready(c) => format!("settled={}", c.is_settled()),
            };
            let _ = writeln!(
                out,
                "in_flight_control: {} pipeline {:?} {state}",
                control.operation, control.pipeline_id,
            );
        }
        let _ = write!(out, "{}", frame_policy.debug_summary());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_item(
        pending: &mut PendingQueue,
        terminated_processes: &mut HashSet<ProcessId>,
        in_flight_control: &mut InFlightControls,
        instances: &HashMap<u64, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
        stopping: &mut bool,
        frame_policy: &mut FramePolicy,
        item: SchedulerItem,
    ) {
        match item {
            SchedulerItem::Stop => {
                *stopping = true;
            }
            // Answered inline at both dequeue sites in `run` — it never
            // reaches this queue-mutating path.
            SchedulerItem::DebugDump { .. } => {
                unreachable!("DebugDump is intercepted before enqueue_item")
            }
            // A nudge only unblocks the wait; the retire pass at the top of
            // the loop does the work.
            SchedulerItem::Nudge => {}
            // Immediate, not queued. Termination rejects queued work; graceful
            // pipeline close instead releases the wait-set and lets every
            // already-admitted request drain untracked.
            SchedulerItem::ExecutionSlotReleased(pid) => {
                frame_policy.on_execution_slot_released(pid);
            }
            SchedulerItem::ProcessQuiesced(pid) => {
                terminated_processes.remove(&pid);
            }
            SchedulerItem::ExecutionSlotConsumed(pid) => {
                frame_policy.on_execution_slot_consumed(pid);
            }
            SchedulerItem::AdmissionQueued(pid) => {
                frame_policy.on_admission_queued(pid);
            }
            SchedulerItem::AdmissionDequeued(pid) => {
                frame_policy.on_admission_dequeued(pid);
            }
            SchedulerItem::PipelineLeave(pid, owner, kind, response) => {
                if kind == LeaveKind::Terminate {
                    if !terminated_processes.insert(pid) {
                        // Duplicate Terminate (the exit funnel notifies from
                        // the terminate entry point and again from deferred
                        // teardown): the first leave did all the work and
                        // every step below is a no-op for it — skip straight
                        // to the ack a waiting sender may hold.
                        if let Some(response) = response {
                            let _ = response.send(());
                        }
                        return;
                    }
                    // A departing slot holder's release broadcast is now in
                    // flight; the seal keeps gathering its successor (the
                    // ragged-boundary guard — see on_slotted_terminate).
                    frame_policy.on_slotted_terminate(pid);
                    let protected = in_flight_control
                        .iter()
                        .find(|control| control.process_id == Some(pid))
                        .and_then(|control| control.logical_completion.clone());
                    if let Some(completion) = &protected {
                        completion.request_cancel();
                    }
                    Self::reject_pipeline_queued(pending, pid, protected.as_ref());
                }
                match kind {
                    LeaveKind::Close => {
                        // Graceful close keeps queued frames: their accepted
                        // fires drain to settlement like any submitted work.
                        frame_policy.on_lane_leave(pid, owner, false);
                    }
                    LeaveKind::Suspend => {
                        // Process-wide graceful leave (see the variant doc);
                        // `pid` names the process here.
                        frame_policy.on_process_suspend(pid);
                    }
                    LeaveKind::Terminate => {
                        // Terminate rejected the lane's queued fires. Both
                        // ids are the process here, so the owner is `pid`.
                        frame_policy.on_lane_leave(pid, owner.or(Some(pid)), true);
                        frame_policy.on_process_leave(pid);
                    }
                }
                if let Some(response) = response {
                    let _ = response.send(());
                }
            }
            SchedulerItem::Launch {
                pending: mut launch,
            } => {
                if let Some(timing) = launch.timing.as_mut() {
                    let now = super::fire_timing_now_us();
                    timing.enqueued_us = Some(now);
                    let lag = now.saturating_sub(timing.submitted_us) * 1_000;
                    let acc = &super::LOOP_PHASES;
                    acc.lag_ns.fetch_add(lag, Ordering::Relaxed);
                    acc.lag_n.fetch_add(1, Ordering::Relaxed);
                    acc.lag_max_ns.fetch_max(lag, Ordering::Relaxed);
                }
                let validation = AdmissionLimits::new(limits, page_size);
                let rejection = if launch.completion.cancel_requested() {
                    Some("logical fire cancelled before scheduler admission".to_string())
                } else if launch
                    .process_id
                    .is_some_and(|pid| terminated_processes.contains(&pid))
                {
                    Some("process terminated before scheduler admission".to_string())
                } else if !instances.contains_key(&launch.instance_id) {
                    Some(format!(
                        "instance {} is unknown or stale",
                        launch.instance_id
                    ))
                } else if let Some(message) = validation.single_request_limit_error(&launch) {
                    Some(message)
                } else if *stopping {
                    Some("scheduler shutting down".to_string())
                } else {
                    None
                };
                if let Some(message) = rejection {
                    // A rejected mid-frame fire still counts toward its
                    // frame's arrival completeness (the surviving fires
                    // execute; the guest observed the rejection) — UNLESS
                    // its process already terminated: the terminate purge
                    // removed the lane and rejected its queued siblings, so
                    // recording this straggler would resurrect a ghost lane
                    // no future event releases.
                    if let Some(stamp) = launch.frame
                        && !launch
                            .process_id
                            .is_some_and(|pid| terminated_processes.contains(&pid))
                    {
                        frame_policy.on_fire_rejected_at_admission(stamp, launch.process_id);
                    }
                    launch.completion.reject_unsubmitted(message);
                } else {
                    // The default single-slot deployment: every tracked fire
                    // IS a one-fire frame. Synthesizing the stamp at accept
                    // (lane = the pipeline scope, seq = the globally
                    // monotonic fire id) makes stamp coverage exactly the
                    // old wait-set membership; an untracked/prebuilt fire
                    // stays an unstamped rider, dispatched outside sealed
                    // waves with no bookkeeping. Synthesis happens only on
                    // the accept path so a rejected fire never touches the
                    // wait-set (mirroring the per-wave rule it replaces).
                    if frame_policy.single_slot()
                        && launch.frame.is_none()
                        && let Some(lane) = launch.pipeline_id
                    {
                        launch.frame = Some(FrameStamp {
                            lane,
                            seq: launch.logical_fire_id,
                            slot: 0,
                            fires: 1,
                        });
                    }
                    // The gather starts at acceptance, not dispatch: a
                    // stamped fire counts toward its lane's frame arrival
                    // even while it sits in `pending` behind an
                    // in-flight-depth or seal hold.
                    if std::env::var_os("PIE_WAVE_TRACE").is_some() {
                        eprintln!(
                            "[wave-trace] enq fire={} framed={} mask={} masks={} stm={} pipe={}",
                            launch.logical_fire_id,
                            launch.frame.is_some(),
                            launch.request.has_user_mask,
                            launch.request.masks.len(),
                            launch.request.single_token_mode,
                            launch.pipeline_id.is_some()
                        );
                    }
                    if let Some(stamp) = launch.frame {
                        frame_policy.on_fire_enqueued(
                            stamp,
                            launch.process_id,
                            launch.logical_fire_id,
                            launch.request.token_ids.len(),
                            launch.wire_row_count(),
                        );
                    }
                    if let Some(timing) = launch.timing.as_mut() {
                        timing.ready_us = Some(super::fire_timing_now_us());
                    }
                    Self::queue_attempt(pending, launch);
                }
            }

            SchedulerItem::ProcessResume(pid) => {
                frame_policy.on_process_resume(pid);
            }
            SchedulerItem::FrameTruncate {
                lane,
                seq,
                submitted,
            } => {
                frame_policy.on_frame_truncated(lane, seq, submitted);
            }
            SchedulerItem::LanePark { lane, seq } => {
                frame_policy.on_lane_park(lane, seq);
            }
            SchedulerItem::RegisterProgram { plan, response } => {
                pending.push_back(QueuedItem::RegisterProgram { plan, response });
            }
            SchedulerItem::RegisterChannel { plan, response } => {
                pending.push_back(QueuedItem::RegisterChannel { plan, response });
            }
            SchedulerItem::RegisterChannels { plans, response } => {
                pending.push_back(QueuedItem::RegisterChannels { plans, response });
            }
            SchedulerItem::BindInstance {
                pipeline_id,
                plan,
                response,
            } => {
                if pipeline_id.is_some_and(|pid| terminated_processes.contains(&pid)) {
                    DriverLane::release_wait_slots([plan.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before instance bind admission"
                    )));
                    return;
                }
                frame_policy.on_bind_enqueued(pipeline_id);
                Self::queue_bind_control(
                    pending,
                    QueuedItem::BindInstance {
                        pipeline_id,
                        plan,
                        response,
                    },
                );
            }
            SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program,
                bind,
                response,
            } => {
                if pipeline_id.is_some_and(|pid| terminated_processes.contains(&pid)) {
                    DriverLane::release_channel_plan_wait_slots(&plans);
                    DriverLane::release_wait_slots([bind.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before channel bind admission"
                    )));
                    return;
                }
                // Binds do not hold the seal (a live rebinder is wait-set-held
                // through its lane; a bring-up process cannot fire). The
                // policy stages bring-up processes here, and a retiring
                // execution slot earmarks one staged successor — that earmark
                // is what gathers a cohort turnover into a dense epoch. The
                // bare `holds_seal` predecessor (gating the hold on execution
                // admission with NO successor earmarking) fragmented k>1
                // boundaries into narrow epochs — the earmark is the
                // gathering mechanism that was missing.
                frame_policy.on_bind_enqueued(pipeline_id);
                Self::queue_bind_control(
                    pending,
                    QueuedItem::RegisterChannelsBind {
                        pipeline_id,
                        plans,
                        program,
                        bind,
                        response,
                    },
                );
            }
            SchedulerItem::CopyKv { plan, response } => {
                pending.push_back(QueuedItem::CopyKv { plan, response });
            }
            SchedulerItem::CopyKvTracked { plan, completion } => {
                pending.push_back(QueuedItem::CopyKvTracked { plan, completion });
            }
            SchedulerItem::CopyState { plan, response } => {
                pending.push_back(QueuedItem::CopyState { plan, response });
            }
            SchedulerItem::ResizePool { plan, response } => {
                pending.push_back(QueuedItem::ResizePool { plan, response });
            }
            SchedulerItem::CloseInstance { id, pacing_wait_id } => {
                pending.push_back(QueuedItem::CloseInstance { id, pacing_wait_id });
            }
            SchedulerItem::CloseChannel { id } => Self::queue_close_channel(pending, id),
            SchedulerItem::CloseChannels { ids } => {
                for id in ids {
                    Self::queue_close_channel(pending, id);
                }
            }
            // Handled on dequeue in the run loop (like DebugDump) before
            // enqueue_item is reached.
            SchedulerItem::Lane(_) => unreachable!(),
        }
    }

    fn reject_pipeline_queued(
        pending: &mut PendingQueue,
        pid: ProcessId,
        protected: Option<&WorkItemCompletion>,
    ) {
        // Common case first: a naturally-completed process has nothing
        // queued, and rebuilding the deque unconditionally moved every
        // pending item per leave (~37 us x ~2 leaves x 512 exits = the
        // largest single slice of the boundary mailbox time). One
        // field-read scan decides; only an actual purge pays the rebuild.
        let has_queued = pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.process_id == Some(pid),
            QueuedItem::PreLaunchCopy { process_id, .. } => *process_id == Some(pid),
            _ => false,
        });
        if !has_queued {
            return;
        }
        let mut kept = VecDeque::with_capacity(pending.len());
        while let Some(item) = pending.pop_front() {
            let reject = match &item {
                QueuedItem::Launch(request) => {
                    request.process_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !request.completion.same_request(completion))
                }
                QueuedItem::PreLaunchCopy {
                    process_id,
                    logical_completion,
                    ..
                } => {
                    *process_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !logical_completion.same_request(completion))
                }
                _ => false,
            };
            if reject {
                match item {
                    QueuedItem::Launch(request) => {
                        request
                            .completion
                            .reject_unsubmitted("pipeline left while queued");
                    }
                    // A pre-launch copy is order-coupled to its consumer
                    // launch (one fire, one book entry — the Launch arm
                    // resolves it).
                    QueuedItem::PreLaunchCopy {
                        logical_completion, ..
                    } => logical_completion
                        .reject_unsubmitted("pipeline left before pre-launch copy"),
                    _ => unreachable!("rejected item kind checked above"),
                }
            } else {
                kept.push_back(item);
            }
        }
        pending.replace(kept);
    }

    fn queue_attempt(pending: &mut PendingQueue, request: PendingRequest) {
        let mut copies = Vec::with_capacity(2);
        if let Some(plan) = request.prelaunch_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        if let Some(plan) = request.prelaunch_state_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::State(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        for copy in copies {
            pending.push_back(copy);
        }
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(Box::new(request))));
    }

    /// Whether any queued fire still targets `instance_id` (a queued
    /// `PreLaunchCopy` is covered by its consumer launch queued behind it).
    /// Together with `TrackedInstance::in_flight` this is the close gate:
    /// an instance with neither queued nor in-flight work is quiesced.
    fn instance_has_queued_work(pending: &VecDeque<QueuedItem>, instance_id: u64) -> bool {
        pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.instance_id == instance_id,
            _ => false,
        })
    }

    /// A standalone KV/state copy: suspend D2H, restore H2D, graft/CAS
    /// copies. These touch pages no queued fire references (suspend takes
    /// only unpinned drained pages; restore writes freshly reserved ones),
    /// so a held wave must NEVER starve them — the planner's eviction and
    /// restore traffic is what unsticks a held wave in the first place. They therefore
    /// dispatch out-of-band from any queue position once the control slot
    /// frees (`dispatch_ready_items` tail sweep), never barrier a queued
    /// fire, and never hold frame posting while they settle.
    /// `PreLaunchCopy` is NOT in this class: it is order-coupled to its
    /// own launch (queued directly in front of it) and must keep queue
    /// order.
    const fn standalone_copy(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::CopyKv { .. }
                | QueuedItem::CopyKvTracked { .. }
                | QueuedItem::CopyState { .. }
        )
    }

    /// Items that a rotation can usefully expose at the queue FRONT — the
    /// only reason to move a held close backward.
    ///
    /// Three kinds dispatch without ever reaching the front, so rotating for
    /// them is pure churn. A `Launch` is picked by fire id in
    /// `dispatch_frame_work`, which reads the whole queue. A standalone copy
    /// is pulled from any position by the tail sweep below. And a close is
    /// excluded because this predicate is only asked while the WHOLE close
    /// run is held, where exposing one held close behind another dispatches
    /// nothing.
    ///
    /// Measured cost of asking the wrong question (`any non-close behind`,
    /// which a queue of held closes followed by the next cohort's launches
    /// always answers yes): 0.5M rotations and 103 ms of the loop thread in
    /// ONE cohort boundary at 512 lanes, and 6.6M rotations over a 1024-lane
    /// run whose GPU then idled 6.4 s. Each rotation also bumps the queue
    /// epoch, so it invalidated `ScanCache` and re-walked the queue on top.
    const fn rotation_target(item: &QueuedItem) -> bool {
        !matches!(
            item,
            QueuedItem::Launch(_)
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannels { .. }
        ) && !Self::standalone_copy(item)
    }

    /// Controls that dispatch without draining in-flight launches. The
    /// registrations are synchronous and create entities nothing in flight
    /// can reference yet — with one caveat: a channel registration that
    /// grows the driver's shared slot table would reallocate arrays whose
    /// pointers in-flight kernels hold, so the CUDA registry quiesces the
    /// device inside `grow()` (RV-27; capacity is driver knowledge, so the
    /// drain lives there, not here). The copies (standalone and pre-launch)
    /// address only committed or quiesced extents, which in-flight launches
    /// never rewrite (append-only ledger) — a copy's coupled consumer launch
    /// still holds behind `in_flight_control` until the copy settles.
    /// A channel close only ever follows its instance closes (the guest
    /// awaits each control's response) and the driver rejects a close with
    /// live attachments, so no in-flight kernel can reference the closing
    /// channel — it needs no drain. `CloseInstance` has its own per-instance
    /// quiescence gate in `dispatch_ready_items`. Only pool resizes keep the
    /// empty-pipe requirement: drain IS their ordering mechanism.
    const fn pipe_concurrent_control(item: &QueuedItem) -> bool {
        Self::standalone_copy(item)
            || matches!(
                item,
                QueuedItem::PreLaunchCopy { .. }
                    | QueuedItem::RegisterProgram { .. }
                    | QueuedItem::RegisterChannel { .. }
                    | QueuedItem::RegisterChannels { .. }
                    | QueuedItem::BindInstance { .. }
                    | QueuedItem::RegisterChannelsBind { .. }
                    | QueuedItem::CloseChannels { .. }
            )
    }

    /// Move held launches behind work that can make the current wave
    /// denser. The ENTIRE contiguous launch prefix rotates to the back in one
    /// call: per-instance launch order is a dispatch invariant
    /// (`launch_has_earlier_instance_member` defers an out-of-order head, and
    /// a head whose earlier sibling sits beyond a non-launch item is
    /// unreachable — a permanent stall), so a run-ahead sibling group must
    /// never be split by a partial rotation.
    ///
    /// A `PreLaunchCopy` is valid rotate-target work under `allow_controls`:
    /// it occupies the free control slot exactly like a lifecycle control.
    /// Rotating front launches past it cannot break copy→consumer coupling —
    /// a consumer launch is enqueued behind its copy and stays behind it (a
    /// Lifecycle controls (registers, binds, closes) never order against a
    /// launch already in the queue: a fire can only be submitted after its
    /// own bind returned to the guest, so every queued launch's lifecycle
    /// dependencies have already dispatched. Only `PreLaunchCopy` (channel
    /// data feeding a later queued launch of the same pipeline) and pool
    /// resizes (pipe drains) order against queued launches.
    const fn lifecycle_control(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::RegisterProgram { .. }
                | QueuedItem::RegisterChannel { .. }
                | QueuedItem::RegisterChannels { .. }
                | QueuedItem::BindInstance { .. }
                | QueuedItem::RegisterChannelsBind { .. }
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannels { .. }
        )
    }

    fn queue_close_channel(pending: &mut PendingQueue, id: u64) {
        // Coalesce teardown runs: consecutive channel closes ride one
        // control post. Bounded so a batch's lane occupancy stays a
        // fraction of a wave (~3-6 us per close driver-side).
        const CLOSE_CHANNEL_BATCH_MAX: usize = 512;
        if let Some(QueuedItem::CloseChannels { ids }) = pending.back_mut()
            && ids.len() < CLOSE_CHANNEL_BATCH_MAX
        {
            ids.push(id);
        } else {
            pending.push_back(QueuedItem::CloseChannels { ids: vec![id] });
        }
    }

    fn queue_bind_control(pending: &mut PendingQueue, item: QueuedItem) {
        // Queue-priority invariant: execution outranks bring-up outranks
        // teardown. A queued LAUNCH never depends on a queued bind — a fire
        // exists only after its own lane's bind control completed and the
        // bind RPC returned to the guest — so a bind may never delay one:
        // with staged admission the binds arriving at a turnover belong to
        // a cohort BEHIND the queued launches, and inserting them ahead
        // starves the sealed wave behind the whole bring-up stream. A bind
        // and a QUEUED close always target different instances/channels
        // (ids are never reused, a close only posts after its own bind
        // committed), so binds still jump the close tail and closes drain
        // during the next generation's execution. Bind-vs-bind order is
        // preserved (insertion before the trailing close run).
        pending.insert_before_closes(item);
    }

    /// launch that reached the queue front has no queued copy left).
    ///
    /// `allow_lifecycle` is the wider of the two flags: a lifecycle control
    /// needs no control slot, so a standalone copy in flight does not stop
    /// it from being worth exposing at the front.
    fn rotate_launch_for_wave_work(
        pending: &mut PendingQueue,
        allow_slot: bool,
        allow_lifecycle: bool,
    ) -> bool {
        if !matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            return false;
        }
        let Some(run_len) = pending.first_other() else {
            return false;
        };
        let work = &pending[run_len];
        if !(Self::standalone_copy(work)
            || (allow_lifecycle && Self::lifecycle_control(work))
            || (allow_slot && matches!(work, QueuedItem::PreLaunchCopy { .. })))
        {
            return false;
        }
        pending.rotate_launch_run_to_back(run_len);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_ready_items(
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut InFlightControls,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let probe_disp = super::fire_timing_enabled();
        let disp_started = probe_disp.then(Instant::now);
        let (mut progress, wait_hint) = Self::dispatch_frame_work(
            scan_cache,
            slot_buffer,
            frame_policy,
            driver_lane,
            lane_inflight,
            lane_token,
            instances,
            pending,
            in_flight_launches,
            in_flight_control,
            page_size,
            limits,
            stats,
            stopping,
        );
        if let Some(started) = disp_started {
            super::LOOP_PHASES
                .disp_frame_ns
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let mut rot_ns = 0u64;
        let mut rot_n = 0u64;
        let mut busy_ns = 0u64;
        let mut busy_n = 0u64;
        // Busy-close rotations this pass: bounded so a queue of nothing but
        // busy closes breaks out instead of spinning.
        let mut close_rotations = 0usize;
        // Cohort-boundary close hold: while any bind is in assembly (the
        // seal is bind-held), teardown closes yield the driver lane to the
        // fresh cohort's registrations — queue-position reordering alone
        // cannot do this, because closes ARRIVE interleaved with binds and
        // each worker pass flushes its controls to the lane FIFO. Held
        // closes rotate and drain during the next generation's execution.
        // Shutdown never holds (the drain must retire everything).
        let hold_closes = !stopping && frame_policy.has_pending_binds();
        // The control SLOT (depth 1) exists for controls that settle
        // asynchronously; lifecycle controls execute on the lane FIFO and
        // never take it (see `post_control`). So a standalone copy holding
        // the slot must not block them: it addresses grant-pinned pages no
        // bind, register or close can reference, and the lane FIFO already
        // fixes their driver order. Blocking them on it wedged the fleet —
        // the planner's suspend/restore copies are in flight almost
        // continuously under churn, so every bind waited out the whole
        // strict-watchdog window, its process sat in `staged` for the
        // duration, and the cohort-boundary window it therefore held open
        // (which then still held the seal) stalled the very traffic the copy
        // was waiting behind. Measured on
        // `churn`: every one of 270 binds took 1.0-2.4 s end to end against
        // a 59 us driver bind, and the probe found the slot held by a
        // tracked KV copy in 100% of the samples.
        let slot_blocks_lifecycle = in_flight_control.holds_launches();
        loop {
            let Some(item) = pending.front() else {
                break;
            };
            match item {
                QueuedItem::Launch(_) => {
                    // Launches are dispatched by `dispatch_frame_work` (by
                    // id, not queue position). A launch at the queue front
                    // only needs to yield to any dispatchable control behind
                    // it.
                    if Self::rotate_launch_for_wave_work(
                        pending,
                        in_flight_control.is_empty(),
                        !slot_blocks_lifecycle,
                    ) {
                        progress = true;
                        continue;
                    }
                    break;
                }
                QueuedItem::CloseInstance { id, .. } => {
                    // A close needs only ITS OWN instance quiesced — never a
                    // global pipe drain. Inferlets submit passes upfront, so
                    // an idle instance's close is always safe to overlap
                    // with other instances' in-flight launches; the old
                    // whole-pipe drain stalled every launch queued behind a
                    // front close during cohort swaps and made freshly-bound
                    // pipelines' credits ragged (V6 iteration 3).
                    // A close needs no control slot either (see
                    // `slot_blocks_lifecycle`), only its own instance
                    // quiesced — a settling standalone copy addresses
                    // grant-pinned pages no close can name.
                    if slot_blocks_lifecycle {
                        break;
                    }
                    let id = *id;
                    if hold_closes {
                        // Held for the boundary: rotate WITHOUT claiming
                        // progress (a rotation changes nothing dispatchable;
                        // the bind-completed lane reply that empties
                        // `pending_binds` is the wake that re-checks).
                        let rot_t = probe_disp.then(Instant::now);
                        let rot_stop = close_rotations >= pending.len()
                            || !pending.iter().skip(1).any(Self::rotation_target);
                        if let Some(t) = rot_t {
                            rot_ns += t.elapsed().as_nanos() as u64;
                            rot_n += 1;
                        }
                        if rot_stop {
                            break;
                        }
                        close_rotations += 1;
                        let item = pending.pop_front().expect("close front");
                        pending.push_back(item);
                        continue;
                    }
                    let busy_t = probe_disp.then(Instant::now);
                    let busy = instances
                        .get(&id)
                        .is_some_and(|tracked| tracked.in_flight != 0)
                        || Self::instance_has_queued_work(pending, id);
                    if let Some(t) = busy_t {
                        busy_ns += t.elapsed().as_nanos() as u64;
                        busy_n += 1;
                    }
                    if !busy {
                        let item = pending.pop_front().expect("close front");
                        Self::post_control(
                            driver_lane,
                            lane_inflight,
                            lane_token,
                            instances,
                            in_flight_control,
                            frame_policy,
                            item,
                        );
                        progress = true;
                        continue;
                    }
                    // Busy: rotate the close behind the queue so the fires
                    // that will quiesce it (and everything unrelated) keep
                    // flowing; its own retirement re-checks it. A close can
                    // only move BACKWARD, so it never overtakes its own
                    // instance's queued work.
                    //
                    // No progress claim: a rotation dispatches nothing, and
                    // `busy` is precisely the condition that guarantees a
                    // later wake (in-flight work completes, or queued work
                    // dispatches and then completes). Claiming progress here
                    // instead makes the run loop re-enter immediately and
                    // spin: at a cohort boundary the queue front is hundreds
                    // of busy closes, so every pass rotated the whole queue
                    // and the loop paid it thousands of times over.
                    let rot_t = probe_disp.then(Instant::now);
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(|item| {
                            !matches!(
                                item,
                                QueuedItem::CloseInstance { .. } | QueuedItem::CloseChannels { .. }
                            )
                        });
                    if let Some(t) = rot_t {
                        rot_ns += t.elapsed().as_nanos() as u64;
                        rot_n += 1;
                    }
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                QueuedItem::CloseChannels { .. } if hold_closes => {
                    // Same bounded rotation as a held instance close; no
                    // progress claim (see the CloseInstance hold branch).
                    let rot_t = probe_disp.then(Instant::now);
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(Self::rotation_target);
                    if let Some(t) = rot_t {
                        rot_ns += t.elapsed().as_nanos() as u64;
                        rot_n += 1;
                    }
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                // A settling exclusive control (a `PreLaunchCopy` or a
                // pool resize) blocks the next control. Standalone copies and
                // lifecycle controls are refused by nothing else — see
                // `InFlightControls::admits`. The front-rotation this arm used
                // to need is gone with the single slot: a copy that cannot post
                // is blocked by an exclusive control, and so is everything
                // behind it, so giving up its position buys nothing.
                _ if !in_flight_control.admits(item) => break,
                _ if !in_flight_launches.is_empty() && !Self::pipe_concurrent_control(item) => {
                    break;
                }
                _ => {
                    let item = pending.pop_front().expect("front item present");
                    Self::post_control(
                        driver_lane,
                        lane_inflight,
                        lane_token,
                        instances,
                        in_flight_control,
                        frame_policy,
                        item,
                    );
                    progress = true;
                }
            }
        }
        // Standalone copies dispatch from ANY queue position once the
        // control slot frees: nothing queued orders against them (their
        // pages are grant-pinned — see the queue scan in
        // `dispatch_frame_work`), while the queue front can be legitimately
        // immovable for a long stretch (a gathering frame's fires, a resize
        // waiting out the pipe). Under contention the suspend/restore
        // copies ARE the residency planner's forward progress — leaving
        // them positional starved the very traffic that unsticks a held
        // frame (CONTENTION_FOLLOWUP.md §12).
        //
        // They also pipeline: the sweep keeps posting while no exclusive
        // control holds the set, so a restore never waits out an unrelated
        // copy's device time. Serialized, the wait WAS the cost — 22.8 ms
        // per H2D restore against ~3.3 ms of transfer at 512-way
        // contention. The queue bounds the depth: only what the planner
        // enqueued can be posted.
        let copy_t = probe_disp.then(Instant::now);
        while in_flight_control.admits_copy() {
            let Some(index) = pending.iter().position(|item| Self::standalone_copy(item)) else {
                break;
            };
            let Some(item) = pending.remove(index) else {
                break;
            };
            Self::post_control(
                driver_lane,
                lane_inflight,
                lane_token,
                instances,
                in_flight_control,
                frame_policy,
                item,
            );
            progress = true;
        }
        if let Some(t) = copy_t {
            let acc = &super::LOOP_PHASES;
            acc.disp_copy_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            acc.disp_rot_ns.fetch_add(rot_ns, Ordering::Relaxed);
            acc.disp_rot_n.fetch_add(rot_n, Ordering::Relaxed);
            acc.disp_busy_ns.fetch_add(busy_ns, Ordering::Relaxed);
            acc.disp_busy_n.fetch_add(busy_n, Ordering::Relaxed);
        }
        (progress, wait_hint)
    }

    /// Static item-kind label for the control-occupancy fire-timing probe.
    const fn item_kind(item: &QueuedItem) -> &'static str {
        match item {
            QueuedItem::Launch(_) => "launch",
            QueuedItem::PreLaunchCopy { .. } => "pre_launch_copy",
            QueuedItem::RegisterProgram { .. } => "register_program",
            QueuedItem::RegisterChannel { .. } => "register_channel",
            QueuedItem::RegisterChannels { .. } => "register_channels",
            QueuedItem::BindInstance { .. } => "bind_instance",
            QueuedItem::RegisterChannelsBind { .. } => "register_channels_bind",
            QueuedItem::CopyKv { .. } => "copy_kv",
            QueuedItem::CopyKvTracked { .. } => "copy_kv_tracked",
            QueuedItem::CopyState { .. } => "copy_state",
            QueuedItem::ResizePool { .. } => "resize_pool",
            QueuedItem::CloseInstance { .. } => "close_instance",
            QueuedItem::CloseChannels { .. } => "close_channels",
        }
    }

    /// Post a control to the driver lane after the worker-side pre-checks
    /// that read scheduler state. The driver half runs on the lane in FIFO
    /// order; worker-map effects come back as a [`LaneCommit`]. Async
    /// controls (copies / pool resizes) occupy the single control slot from
    /// the moment they post.
    fn post_control(
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
        frame_policy: &mut FramePolicy,
        item: QueuedItem,
    ) {
        match &item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                logical_completion, ..
            } if logical_completion.is_settled() => return,
            QueuedItem::PreLaunchCopy {
                logical_completion, ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
                return;
            }
            QueuedItem::CloseInstance {
                id, pacing_wait_id, ..
            } => {
                let error = match instances.get(id) {
                    Some(instance) if instance.pacing_wait_id == *pacing_wait_id => {
                        (instance.in_flight != 0).then(|| format!("instance {id} is busy"))
                    }
                    _ => Some(format!("instance {id} is unknown or stale")),
                };
                if let Some(message) = error {
                    tracing::warn!(
                        instance_id = id,
                        error = %message,
                        "scheduler close_instance skipped"
                    );
                    return;
                }
            }
            QueuedItem::BindInstance { plan, .. }
                if plan.requested_instance_id != 0
                    && instances.contains_key(&plan.requested_instance_id) =>
            {
                let QueuedItem::BindInstance {
                    pipeline_id,
                    plan,
                    response,
                } = item
                else {
                    unreachable!();
                };
                if response
                    .send(Err(anyhow!(
                        "instance {} is already bound",
                        plan.requested_instance_id
                    )))
                    .is_err()
                {
                    DriverLane::release_wait_slots([plan.pacing_wait_id]);
                }
                frame_policy.on_bind_completed(pipeline_id);
                return;
            }
            QueuedItem::RegisterChannelsBind { bind, .. }
                if bind.requested_instance_id != 0
                    && instances.contains_key(&bind.requested_instance_id) =>
            {
                let QueuedItem::RegisterChannelsBind {
                    pipeline_id,
                    plans,
                    bind,
                    response,
                    ..
                } = item
                else {
                    unreachable!();
                };
                if response
                    .send(Err(anyhow!(
                        "instance {} is already bound",
                        bind.requested_instance_id
                    )))
                    .is_err()
                {
                    DriverLane::release_channel_plan_wait_slots(&plans);
                    DriverLane::release_wait_slots([bind.pacing_wait_id]);
                }
                frame_policy.on_bind_completed(pipeline_id);
                return;
            }
            _ => {}
        }
        *lane_token += 1;
        let token = *lane_token;
        // Async-completing controls enter the in-flight set from POST: an
        // exclusive one (the copy's coupled consumer launch, a resize) must
        // not be passed by any later control, exactly as before the lane
        // existed. Exactly the standalone copies do NOT hold launches — one
        // classification, shared with the out-of-band dispatch and the
        // concurrency rule in `InFlightControls` that both rely on it.
        let holds_launches = !Self::standalone_copy(&item);
        match &item {
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                process_id,
                pipeline_id,
            } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: Some(logical_completion.clone()),
                    process_id: *process_id,
                    pipeline_id: *pipeline_id,
                    tracked_completion: None,
                    operation: plan.label(),
                    holds_launches,
                });
            }
            QueuedItem::CopyKv { .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "KV copy",
                    holds_launches,
                });
            }
            QueuedItem::CopyKvTracked { completion, .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: Some(completion.clone()),
                    operation: "tracked KV copy",
                    holds_launches,
                });
            }
            QueuedItem::CopyState { .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "state copy",
                    holds_launches,
                });
            }
            QueuedItem::ResizePool { .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "pool resize",
                    holds_launches,
                });
            }
            _ => {}
        }
        *lane_inflight += 1;
        driver_lane.post(LaneRequest::Control { token, item });
    }

    /// One queue pass: the stamped ids still queued, the oldest unstamped
    /// rider, and the lanes a frame post must hold for.
    ///
    /// Only a queued `PreLaunchCopy` blocks a lane — it is order-coupled to
    /// its consumer fire by construction. Standalone copies and pool
    /// resizes never barrier fires: reservations pin every page they touch
    /// and no queued fire can reference those pages (the planner's eviction
    /// fences and quiesces a victim's working sets before its D2H, and a
    /// restored process is only readmitted after its H2D copy retired —
    /// `planner::exec` awaits the tracked completion before the commit).
    /// Resizes were exempted first, on the same pinning argument (~45x on
    /// gen-boundary teardown); the copy barrier that remained composed with
    /// frame atomicity and the resize rotation refusal into a three-party
    /// queue-order deadlock under contention — a sealed frame straddling a
    /// {resize, copy} pair never posted (CONTENTION_FOLLOWUP.md §12).
    fn scan_queue<'a>(
        cache: &'a mut ScanCache,
        pending: &PendingQueue,
        stopping: bool,
    ) -> &'a QueueScan {
        // The scan is a pure function of (queue contents, stopping), so a
        // pass at an unchanged epoch would rebuild exactly what is already
        // here. This matters: the worker scans once per pass and passes run
        // ~50x per wave while the queue changes only a couple of times, and
        // walking `pending` drags every large `QueuedItem` through cache
        // (~25us per scan at 128 requests, about half of all dispatch time).
        if cache.taken_at == Some((pending.epoch(), stopping)) {
            return &cache.scan;
        }
        let scan = &mut cache.scan;
        scan.clear();
        for item in pending.iter() {
            match item {
                QueuedItem::Launch(launch) => {
                    if stopping {
                        scan.drain_eligible.push(launch.fire_id);
                    }
                    if launch.framed {
                        scan.queued_ids.push(launch.fire_id);
                    } else if scan.untracked.is_none() {
                        scan.untracked = Some(launch.fire_id);
                    }
                }
                QueuedItem::PreLaunchCopy { pipeline_id, .. } => {
                    if let Some(pipeline_id) = pipeline_id {
                        scan.blocked_lanes.insert(*pipeline_id);
                    }
                }
                _ => {}
            }
        }
        scan.queued_ids.seal();
        cache.taken_at = Some((pending.epoch(), stopping));
        &cache.scan
    }

    /// Launch dispatch: post WHOLE sealed frames to the driver lane at the
    /// run-ahead depth (frames in seal order; the driver executes the
    /// frame's waves in slot order as one closed system with a single
    /// completion). At the default k = 1 a sealed frame is one wave, so
    /// this degenerates to the per-wave wait-all dispatch.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_frame_work(
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        frame_policy: &mut FramePolicy,
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let mut progress = false;
        let mut wait_hint: Option<Duration> = None;
        let merge_hint = |hint: &mut Option<Duration>, hold: Duration| {
            *hint = Some(hint.map_or(hold, |old| old.min(hold)));
        };
        loop {
            // A settling control holds launches only when a launch could
            // depend on it: a `PreLaunchCopy`'s consumer fire is queued
            // right behind it, and a resize's pipe drain must not admit new
            // frames under it. A settling standalone copy holds nothing
            // (`PendingControl::holds_launches`) — frames keep posting
            // while suspend/restore traffic settles.
            if in_flight_control.holds_launches() {
                break;
            }
            // Run-ahead depth in FRAMES: the enqueue horizon. Retirement
            // frees a slot; posting never waits on completion beyond this
            // backpressure.
            if in_flight_launches.len() >= frame::configured_max_in_flight() {
                break;
            }
            let now = Instant::now();
            let probe = super::fire_timing_enabled();
            let scan = Self::scan_queue(scan_cache, pending, stopping);
            if probe {
                let acc = &super::LOOP_PHASES;
                acc.scan_ns
                    .fetch_add(now.elapsed().as_nanos() as u64, Ordering::Relaxed);
                acc.scans.fetch_add(1, Ordering::Relaxed);
            }
            let plan_started = probe.then(Instant::now);
            let mut rider_batch = false;
            let waves: Vec<Vec<u64>> = if stopping {
                // Shutdown drain: the boundary gate waits for arrivals that
                // will never come once the host stops, so bypass it and post
                // every accepted fire in queue order (queue order IS each
                // lane's submission order, which the device tickets require);
                // repeated instances and budget overflows split into
                // successive steps at build.
                if scan.drain_eligible.is_empty() {
                    break;
                }
                vec![scan.drain_eligible.clone()]
            } else if let Some(untracked) = scan.untracked {
                rider_batch = true;
                if std::env::var_os("PIE_WAVE_TRACE").is_some() {
                    eprintln!("[wave-trace] rider fire={untracked}");
                }
                vec![vec![untracked]]
            } else {
                match frame_policy.plan_dispatch(
                    &scan.queued_ids,
                    &scan.blocked_lanes,
                    !in_flight_launches.is_empty(),
                    now,
                ) {
                    FramePlan::Dispatch(waves) => {
                        if std::env::var_os("PIE_WAVE_TRACE").is_some() {
                            eprintln!(
                                "[wave-trace] dispatch waves={:?}",
                                waves.iter().map(Vec::len).collect::<Vec<_>>()
                            );
                        }
                        waves
                    }
                    FramePlan::Hold(hold) => {
                        merge_hint(&mut wait_hint, hold);
                        break;
                    }
                    FramePlan::Park => break,
                    FramePlan::Terminate(pids) => {
                        // Abandoned pipeline. This is NOT the submit
                        // deadline: that one only leashes (drops the lane
                        // from the wait-set and lets it rejoin), so a guest
                        // that is merely slow never lands here. Reaching this
                        // means the lane was silent for the whole silence
                        // timeout without ever calling `forward.park()`, so
                        // nothing but a wedged process is being reclaimed.
                        // The policy has already dropped these lanes, so the
                        // `continue` re-plans a gather that no longer waits
                        // on them; the terminate is asynchronous and arrives
                        // back as the usual leave.
                        for pid in pids {
                            tracing::error!(
                                pid = %pid,
                                "scheduler: terminating abandoned pipeline (silent for the \
                                 whole silence timeout without submitting and without \
                                 calling forward.park())"
                            );
                            crate::inferlet::process::terminate(
                                pid,
                                Err("pipeline abandoned: silent past the silence timeout \
                                     without submitting and without parking"
                                    .to_string()),
                            );
                        }
                        continue;
                    }
                }
            };
            if let Some(plan_started) = plan_started {
                super::LOOP_PHASES
                    .plan_ns
                    .fetch_add(plan_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            let post_started = probe.then(Instant::now);
            #[allow(clippy::let_and_return)]
            let (frame_progress, posted) = Self::post_frame(
                slot_buffer,
                driver_lane,
                lane_inflight,
                lane_token,
                instances,
                pending,
                in_flight_launches,
                page_size,
                limits,
                stats,
                &waves,
            );
            if let Some(post_started) = post_started {
                super::LOOP_PHASES
                    .post_ns
                    .fetch_add(post_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            progress |= frame_progress;
            if !posted {
                if stopping || !frame_progress {
                    break;
                }
                continue;
            }
            if rider_batch {
                frame_policy.record_rider_wave();
            }
        }
        (progress, wait_hint)
    }

    /// Extract the frame's fires (all waves) from the queue, drop the
    /// settled/stale, assemble the v14 frame submission, and post it as ONE
    /// launch. Returns (progress, posted-a-frame).
    #[allow(clippy::too_many_arguments)]
    /// Cached so the frame path does not pay an environment lookup per frame.
    fn frame_shape_trace() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("PIE_FRAME_SHAPE").is_some())
    }

    /// Whether a queued fire still belongs in the frame being built; settles
    /// it with a rejection if not.
    fn admits_to_frame(
        request: &PendingRequest,
        instances: &HashMap<u64, TrackedInstance>,
    ) -> bool {
        if request.completion.is_settled() || request.completion.cancel_requested() {
            if !request.completion.is_settled() {
                request
                    .completion
                    .reject_unsubmitted("logical fire cancelled before native launch");
            }
            return false;
        }
        if !instances.contains_key(&request.instance_id) {
            request.completion.reject_unsubmitted(format!(
                "instance {} is unknown or stale",
                request.instance_id
            ));
            return false;
        }
        true
    }

    fn post_frame(
        slot_buffer: &mut SlotBuffer,
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        waves: &[Vec<u64>],
    ) -> (bool, bool) {
        let mut progress = false;
        let sub = super::fire_timing_enabled().then(Instant::now);
        // One map, carrying BOTH the wave and the in-wave position: the
        // position is the sealed wave's id order (lane admission order), so
        // carrying it here lets the sort below compare plain integers. The
        // previous shape hashed twice per queued launch (`contains_key` then
        // index) and rebuilt a second id->position map per wave that the sort
        // comparator then hashed into once per comparison — n log n hash
        // lookups on the loop's hottest per-fire path at 512 fires a frame.
        let mut slot_of: HashMap<u64, (usize, usize)> =
            HashMap::with_capacity(waves.iter().map(Vec::len).sum());
        for (index, wave) in waves.iter().enumerate() {
            for (position, &fire_id) in wave.iter().enumerate() {
                slot_of.insert(fire_id, (index, position));
            }
        }
        let t_drain = sub.map(|t| {
            super::LOOP_PHASES
                .post_map_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Instant::now()
        });
        let mut kept: VecDeque<QueuedItem> = VecDeque::with_capacity(pending.len());
        let drained = pending.len() as u64;
        // Place by slot rather than push-then-sort. `position` is already a
        // permutation of `0..wave.len()`, so the sealed order is recovered by
        // writing each request straight into its slot: one move per fire,
        // against the n log n swaps of a 1288-byte element that sorting cost.
        // The buffer is caller-owned and comes back empty from the last
        // frame, so steady state neither allocates nor refills.
        slot_buffer.resize_with(waves.len(), Vec::new);
        for (slots, wave) in slot_buffer.iter_mut().zip(waves) {
            debug_assert!(slots.iter().all(Option::is_none));
            if slots.len() < wave.len() {
                slots.resize_with(wave.len(), || None);
            }
        }
        // A fire id repeated across the queue cannot be placed twice; it is
        // degenerate, but it must still be dispatched rather than dropped.
        let mut collisions: Vec<(usize, Box<PendingRequest>)> = Vec::new();
        while let Some(item) = pending.pop_front() {
            match item {
                QueuedItem::Launch(launch) => match slot_of.get(&launch.fire_id) {
                    Some(&(wave, position)) => {
                        let slot = &mut slot_buffer[wave][position];
                        if slot.is_none() {
                            *slot = Some(launch.into_request());
                        } else {
                            collisions.push((wave, launch.into_request()));
                        }
                    }
                    None => kept.push_back(QueuedItem::Launch(launch)),
                },
                item => kept.push_back(item),
            }
        }
        pending.replace(kept);
        let t_filter = t_drain.map(|t| {
            let acc = &super::LOOP_PHASES;
            acc.post_drain_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            acc.post_drain_n.fetch_add(drained, Ordering::Relaxed);
            Instant::now()
        });
        // Compact the slots and drop settled/cancelled/stale fires in one
        // pass — the frame posts without them.
        let mut survivors: Vec<Vec<Box<PendingRequest>>> = Vec::with_capacity(waves.len());
        for slots in slot_buffer.iter_mut() {
            let mut kept_wave = Vec::with_capacity(slots.len());
            for request in slots.iter_mut().filter_map(Option::take) {
                if Self::admits_to_frame(&request, instances) {
                    kept_wave.push(request);
                } else {
                    progress = true;
                }
            }
            survivors.push(kept_wave);
        }
        for (wave, request) in collisions {
            if Self::admits_to_frame(&request, instances) {
                survivors[wave].push(request);
            } else {
                progress = true;
            }
        }
        if let Some(t) = t_filter {
            super::LOOP_PHASES
                .post_filter_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if survivors.iter().all(Vec::is_empty) {
            return (progress, false);
        }
        let t_tail = sub.map(|_| Instant::now());
        let timing_enabled = super::fire_timing_enabled();
        let dispatch_started_us = timing_enabled.then(super::fire_timing_now_us);
        if let Some(now_us) = dispatch_started_us {
            for request in survivors.iter_mut().flatten() {
                if let Some(timing) = request.timing.as_mut()
                    && timing.ready_us.is_none()
                {
                    timing.ready_us = Some(now_us);
                }
            }
        }
        let nonempty_waves = survivors.iter().filter(|w| !w.is_empty()).count();
        let (submission, requests) =
            batch::build_frame_submission(survivors, limits, page_size, stats);
        // How many waves a sealed frame actually carries. ABI v14 has a frame
        // carry k steps the driver runs as one closed system, and the guest does
        // submit `live_slots` fires per frame -- but this reports
        // `nonempty_waves=1` at every k, so each fire becomes its own frame and
        // every decode step pays a host round trip (2.31ms of a 25.6ms step at
        // 32 lanes, measured driver-side with PIE_METAL_GPU_METER).
        if Self::frame_shape_trace() {
            use std::sync::atomic::{AtomicU64, Ordering as O};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, O::Relaxed) + 1;
            if n % 256 == 0 {
                eprintln!(
                    "[frame-shape] n={n} nonempty_waves={nonempty_waves} steps={}",
                    submission.steps.len()
                );
            }
        }
        let batch_size = requests.len() as u64;
        let total_tokens = requests
            .iter()
            .map(|req| req.request.token_ids.len())
            .sum::<usize>();
        let batch_built_us = timing_enabled.then(super::fire_timing_now_us);
        let membership_hash = if timing_enabled {
            fire_membership_hash(requests.iter().map(|request| &request.logical_fire_id))
        } else {
            0
        };
        let wave_timing = dispatch_started_us.map(|dispatch_started_us| WaveTimingState {
            wave_id: 0,
            membership_hash,
            dispatch_started_us,
            batch_built_us: batch_built_us.unwrap_or(dispatch_started_us),
            driver_started_us: dispatch_started_us,
            launch_returned_us: dispatch_started_us,
            decision_us: 0,
            active_pipelines: 0,
            missing_pipelines: 0,
            candidate_count: requests.len(),
            deferred_pipelines: 0,
            depth_capped_pipelines: 0,
        });
        if let Some(t) = t_tail {
            super::LOOP_PHASES
                .post_tail_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let t_tail = sub.map(|_| Instant::now());
        for request in &requests {
            if let Some(instance) = instances.get_mut(&request.instance_id) {
                instance.in_flight += 1;
            }
        }
        *lane_token += 1;
        let token = *lane_token;
        in_flight_launches.push_back(PendingLaunchBatch {
            state: LaunchState::Posted { token },
            requests,
            started: Instant::now(),
            batch_size,
            total_tokens,
            timing: wave_timing,
        });
        *lane_inflight += 1;
        driver_lane.post(LaneRequest::Launch {
            token,
            submission: LaneLaunch(submission),
        });
        if super::sched_trace_enabled() {
            super::sched_trace_write(format_args!(
                "frame dispatched={batch_size} tokens={total_tokens} pending={} in_flight={}",
                pending.len(),
                in_flight_launches.len(),
            ));
        }
        if let Some(t) = t_tail {
            super::LOOP_PHASES
                .post_tail_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        (true, true)
    }

    fn retire_ready_launches(
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
    ) -> bool {
        let mut progress = false;
        while let Some(front) = in_flight_launches.front() {
            // A lane-rejected launch retires like a wave: it entered the
            // pipe (depth, instance accounting) at post, so the common
            // unwind below applies; only its requests' settlement differs
            // (rejected, never submitted).
            let launch_failure = match &front.state {
                LaunchState::Posted { .. } => break,
                LaunchState::Failed(message) => Some(message.clone()),
                LaunchState::Accepted(_) => None,
            };
            let result = match &front.state {
                LaunchState::Accepted(completion) => {
                    let Some(result) = completion.check() else {
                        break;
                    };
                    Some(result)
                }
                _ => None,
            };
            let mut retired = in_flight_launches.pop_front().expect("front batch exists");
            // The engine has answered these lanes: re-arm their submit
            // deadline from here so the wave they waited on is not charged
            // to them (see `FramePolicy::on_frame_retired`).
            frame_policy.on_frame_retired(retired.requests.iter().filter_map(|r| r.pipeline_id));
            let native_complete_us = retired.timing.as_ref().map(|_| super::fire_timing_now_us());
            let timing_snapshots = retired
                .timing
                .as_ref()
                .map(|_| Self::fire_timing_snapshots(&retired.requests));
            let sub = super::fire_timing_enabled().then(Instant::now);
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            if let Some(mark) = sub {
                let acc = &super::LOOP_PHASES;
                acc.retire_instances_ns
                    .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
                acc.retire_n
                    .fetch_add(retired.requests.len() as u64, Ordering::Relaxed);
            }
            if let Some(message) = launch_failure {
                if let (Some(timing), Some(native_complete_us), Some(snapshots)) =
                    (retired.timing, native_complete_us, timing_snapshots)
                {
                    let settled_us = super::fire_timing_now_us();
                    Self::emit_fire_timing(
                        &snapshots,
                        timing,
                        false,
                        native_complete_us,
                        settled_us,
                        &vec!["launch_error"; snapshots.len()],
                        retired.batch_size,
                        retired.total_tokens,
                        Self::queued_untracked_riders(pending),
                        &[],
                    );
                }
                let message = format!("direct launch rejected: {message}");
                for request in &retired.requests {
                    request.completion.reject_unsubmitted(message.clone());
                }
                progress = true;
                continue;
            }
            let result = result.expect("accepted batch carries a settled result");
            match result {
                Ok(()) => {
                    let t_mark = sub.map(|_| Instant::now());
                    for request in &retired.requests {
                        request.completion.mark_native_retired();
                    }
                    let t_resolve = t_mark.map(|t| {
                        super::LOOP_PHASES
                            .retire_mark_ns
                            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        Instant::now()
                    });
                    if retired.timing.is_some() {
                        super::LAST_RESOLVE_US
                            .store(super::fire_timing_now_us(), Ordering::Relaxed);
                    }
                    let requests = std::mem::take(&mut retired.requests);
                    let mut outcomes = Vec::with_capacity(requests.len());
                    let mut token_instance_ids = Vec::new();
                    for request in &requests {
                        match request.completion.resolve_from_terminal() {
                            Ok(WorkItemAttemptOutcome::Committed) => {
                                outcomes.push("committed");
                                if !request.request.sampling_indices.is_empty() {
                                    token_instance_ids.push(request.instance_id);
                                }
                            }
                            Ok(WorkItemAttemptOutcome::Failed) => {
                                outcomes.push("failed");
                            }
                            Ok(WorkItemAttemptOutcome::Retry) => {
                                // Venus (ABI v14): admitted frames are
                                // atomic and stream work is SUCCESS-only —
                                // ring capacity is admission-bounded and
                                // deterministic compose kills latch FAILED.
                                // A surviving RETRY terminal therefore
                                // violates the driver contract: fail loudly
                                // instead of replaying (the makeup machinery
                                // is deleted).
                                outcomes.push("retry");
                                request.completion.reject(
                                    "driver published RETRY at frame settle; \
                                     retry is not a v14 outcome (frame admission \
                                     bounds every in-frame gate)",
                                );
                            }
                            Err(err) => {
                                outcomes.push("settlement_error");
                                tracing::warn!(
                                    instance_id = request.instance_id,
                                    ?err,
                                    "direct launch terminal settlement failed"
                                );
                            }
                        }
                    }
                    let t_drop = t_resolve.map(|t| {
                        super::LOOP_PHASES
                            .retire_resolve_ns
                            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        Instant::now()
                    });
                    drop(requests);
                    let t_emit = t_drop.map(|t| {
                        super::LOOP_PHASES
                            .retire_drop_ns
                            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        Instant::now()
                    });
                    if let (Some(timing), Some(native_complete_us), Some(snapshots)) =
                        (retired.timing, native_complete_us, timing_snapshots)
                    {
                        let settled_us = super::fire_timing_now_us();
                        Self::emit_fire_timing(
                            &snapshots,
                            timing,
                            true,
                            native_complete_us,
                            settled_us,
                            &outcomes,
                            retired.batch_size,
                            retired.total_tokens,
                            Self::queued_untracked_riders(pending),
                            &token_instance_ids,
                        );
                    }
                    if let Some(t) = t_emit {
                        super::LOOP_PHASES
                            .retire_emit_ns
                            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    stats::record_fire_stats(
                        stats,
                        retired.started.elapsed(),
                        retired.batch_size,
                        retired.total_tokens,
                    )
                }

                Err(err) => {
                    if let (Some(timing), Some(native_complete_us), Some(snapshots)) =
                        (retired.timing, native_complete_us, timing_snapshots)
                    {
                        let settled_us = super::fire_timing_now_us();
                        Self::emit_fire_timing(
                            &snapshots,
                            timing,
                            true,
                            native_complete_us,
                            settled_us,
                            &vec!["completion_error"; snapshots.len()],
                            retired.batch_size,
                            retired.total_tokens,
                            Self::queued_untracked_riders(pending),
                            &[],
                        );
                    }
                    tracing::warn!(?err, "direct launch completion closed before callback");
                    for request in &retired.requests {
                        request.completion.reject(format!(
                            "direct launch batch callback closed before terminal settlement: {err:#}"
                        ));
                        if let Some(instance) = instances.get(&request.instance_id) {
                            instance.wait_slots.close();
                        }
                    }
                }
            }
            progress = true;
        }
        progress
    }

    /// Unstamped (rider) launches currently queued — the fire-timing
    /// `untracked_ready` gauge. Riders dispatch outside sealed waves, so a
    /// monotonic climb here means untracked work is starving behind the
    /// fleet (the successor of the old quorum W1 leak gate).
    fn queued_untracked_riders(pending: &VecDeque<QueuedItem>) -> usize {
        pending
            .iter()
            .filter(|item| matches!(item, QueuedItem::Launch(request) if request.frame.is_none()))
            .count()
    }

    fn fire_timing_snapshots(requests: &[Box<PendingRequest>]) -> Vec<FireTimingSnapshot> {
        requests
            .iter()
            .enumerate()
            .filter_map(|(outcome_index, request)| {
                request.timing.map(|timing| FireTimingSnapshot {
                    outcome_index,
                    logical_fire_id: request.logical_fire_id,
                    instance_id: request.instance_id,
                    process_id: request.process_id,
                    sampled_rows: request.request.sampling_indices.len(),
                    timing,
                })
            })
            .collect()
    }

    fn emit_fire_timing(
        requests: &[FireTimingSnapshot],
        timing: WaveTimingState,
        cuda_submitted: bool,
        native_complete_us: u64,
        settled_us: u64,
        outcomes: &[&str],
        batch_size: u64,
        total_tokens: usize,
        untracked_ready: usize,
        token_instance_ids: &[u64],
    ) {
        let committed = outcomes
            .iter()
            .filter(|&&outcome| outcome == "committed")
            .count();
        let retried = outcomes
            .iter()
            .filter(|&&outcome| outcome == "retry")
            .count();
        let failed = outcomes.len().saturating_sub(committed + retried);
        let acc = &super::LOOP_PHASES;
        let take = |cell: &std::sync::atomic::AtomicU64| cell.swap(0, Ordering::Relaxed);
        let (loop_mailbox, loop_retire, loop_dispatch, loop_park) = (
            take(&acc.mailbox_ns) / 1_000,
            take(&acc.retire_ns) / 1_000,
            take(&acc.dispatch_ns) / 1_000,
            take(&acc.park_ns) / 1_000,
        );
        let (loop_passes, loop_items) = (take(&acc.passes), take(&acc.mailbox_items));
        let (loop_lag, loop_lag_max, loop_lag_n, loop_pass_max) = (
            take(&acc.lag_ns) / 1_000,
            take(&acc.lag_max_ns) / 1_000,
            take(&acc.lag_n),
            take(&acc.pass_max_ns) / 1_000,
        );
        let mut subs: Vec<u64> = requests.iter().map(|r| r.timing.submitted_us).collect();
        subs.sort_unstable();
        let pick = |q: usize| subs.get(subs.len() * q / 100).copied().unwrap_or(0);
        let sub_min = subs.first().copied().unwrap_or(0);
        let sub_max = subs.last().copied().unwrap_or(0);
        let (sub_p50, sub_p90) = (pick(50), pick(90));
        let sub_lanes: Vec<serde_json::Value> = requests
            .iter()
            .map(|r| {
                serde_json::json!([
                    r.instance_id,
                    timing
                        .dispatch_started_us
                        .saturating_sub(r.timing.submitted_us)
                ])
            })
            .collect();
        let enq_max = requests
            .iter()
            .filter_map(|r| r.timing.enqueued_us)
            .max()
            .unwrap_or(0);
        let gacc = &super::GUEST_PHASES;
        let (wake_woken, wake_empty) = (take(&gacc.wake_woken), take(&gacc.wake_empty));
        let (guest_resume, guest_resume_max, guest_resume_n) = (
            take(&gacc.resume_ns) / 1_000,
            take(&gacc.resume_max_ns) / 1_000,
            take(&gacc.resume_n),
        );
        let (guest_wake, guest_work, guest_work_max, guest_n) = (
            take(&gacc.wake_ns) / 1_000,
            take(&gacc.work_ns) / 1_000,
            take(&gacc.work_max_ns) / 1_000,
            take(&gacc.n),
        );
        let (retire_instances, retire_mark, retire_resolve, retire_drop, retire_emit, retire_n) = (
            take(&acc.retire_instances_ns) / 1_000,
            take(&acc.retire_mark_ns) / 1_000,
            take(&acc.retire_resolve_ns) / 1_000,
            take(&acc.retire_drop_ns) / 1_000,
            take(&acc.retire_emit_ns) / 1_000,
            take(&acc.retire_n),
        );
        let (loop_scan, loop_plan, loop_post, loop_scans) = (
            take(&acc.scan_ns) / 1_000,
            take(&acc.plan_ns) / 1_000,
            take(&acc.post_ns) / 1_000,
            take(&acc.scans),
        );
        let (post_map, post_drain, post_filter, post_tail, post_drain_n) = (
            take(&acc.post_map_ns) / 1_000,
            take(&acc.post_drain_ns) / 1_000,
            take(&acc.post_filter_ns) / 1_000,
            take(&acc.post_tail_ns) / 1_000,
            take(&acc.post_drain_n),
        );
        let (disp_frame, disp_rot, disp_rot_n, disp_busy, disp_busy_n, disp_copy) = (
            take(&acc.disp_frame_ns) / 1_000,
            take(&acc.disp_rot_ns) / 1_000,
            take(&acc.disp_rot_n),
            take(&acc.disp_busy_ns) / 1_000,
            take(&acc.disp_busy_n),
            take(&acc.disp_copy_ns) / 1_000,
        );
        let mut record = serde_json::json!({
            "schema": 1,
            "source": "scheduler",
            "event": "scheduler_wave",
            "planner_parks_total": crate::planner::planner()
                .map(|planner| planner.park_census().0)
                .unwrap_or(0),
            "planner_parked_now": crate::planner::planner()
                .map(|planner| planner.park_census().1)
                .unwrap_or(0),
            "wave_id": timing.wave_id,
            "membership_hash": timing.membership_hash,
            "cuda_submitted": cuda_submitted,
            "fire_count": batch_size,
            "batch_size": batch_size,
            "tokens": total_tokens,
            "committed": committed,
            "retried": retried,
            "failed": failed,
            "dispatch_started_us": timing.dispatch_started_us,
            "batch_built_us": timing.batch_built_us,
            "driver_started_us": timing.driver_started_us,
            "launch_returned_us": timing.launch_returned_us,
            "native_complete_us": native_complete_us,
            "settled_us": settled_us,
            "batch_build_us": timing
                .batch_built_us
                .saturating_sub(timing.dispatch_started_us),
            "driver_submit_us": timing
                .launch_returned_us
                .saturating_sub(timing.driver_started_us),
            "native_inflight_us": native_complete_us
                .saturating_sub(timing.launch_returned_us),
            "retire_settle_us": settled_us.saturating_sub(native_complete_us),
            "decision_us": timing.decision_us,
            "active_pipelines": timing.active_pipelines,
            "missing_pipelines": timing.missing_pipelines,
            "candidate_count": timing.candidate_count,
            "deferred_pipelines": timing.deferred_pipelines,
            "depth_capped_pipelines": timing.depth_capped_pipelines,
            // Queued rider gauge: unstamped launches awaiting dispatch
            // outside the sealed waves (0 in an all-tracked fleet).
            "untracked_ready": untracked_ready,
        });
        if let Some(object) = record.as_object_mut() {
            for (key, value) in [
                ("loop_mailbox_us", loop_mailbox),
                ("loop_retire_us", loop_retire),
                ("loop_dispatch_us", loop_dispatch),
                ("loop_park_us", loop_park),
                ("loop_passes", loop_passes),
                ("loop_items", loop_items),
                ("loop_scan_us", loop_scan),
                ("loop_plan_us", loop_plan),
                ("loop_post_us", loop_post),
                ("post_map_us", post_map),
                ("post_drain_us", post_drain),
                ("post_filter_us", post_filter),
                ("post_tail_us", post_tail),
                ("post_drain_n", post_drain_n),
                ("retire_instances_us", retire_instances),
                ("retire_mark_us", retire_mark),
                ("retire_resolve_us", retire_resolve),
                ("retire_drop_us", retire_drop),
                ("retire_emit_us", retire_emit),
                ("retire_n", retire_n),
                ("loop_scans", loop_scans),
                ("disp_frame_us", disp_frame),
                ("disp_rot_us", disp_rot),
                ("disp_rot_n", disp_rot_n),
                ("disp_busy_us", disp_busy),
                ("disp_busy_n", disp_busy_n),
                ("disp_copy_us", disp_copy),
                (
                    "loop_lag_us",
                    if loop_lag_n > 0 {
                        loop_lag / loop_lag_n
                    } else {
                        0
                    },
                ),
                ("loop_lag_max_us", loop_lag_max),
                ("loop_lag_n", loop_lag_n),
                ("loop_pass_max_us", loop_pass_max),
                ("sub_min_us", sub_min),
                ("sub_max_us", sub_max),
                (
                    "guest_wake_us",
                    if guest_n > 0 { guest_wake / guest_n } else { 0 },
                ),
                (
                    "guest_work_us",
                    if guest_n > 0 { guest_work / guest_n } else { 0 },
                ),
                ("guest_work_max_us", guest_work_max),
                ("guest_n", guest_n),
                (
                    "guest_resume_us",
                    if guest_resume_n > 0 {
                        guest_resume / guest_resume_n
                    } else {
                        0
                    },
                ),
                ("guest_resume_max_us", guest_resume_max),
                ("wake_woken", wake_woken),
                ("wake_empty", wake_empty),
                ("sub_p50_us", sub_p50),
                ("sub_p90_us", sub_p90),
                ("enq_max_us", enq_max),
            ] {
                object.insert(key.to_string(), serde_json::json!(value));
            }
        }
        if std::env::var_os("PIE_WAVE_LANES").is_some() {
            record["sub_lanes"] = serde_json::Value::Array(sub_lanes);
        }
        if super::ledger_timing_enabled() {
            record["token_instance_ids"] = serde_json::json!(token_instance_ids);
        }
        super::fire_timing_write(&record);
        if !super::fire_timing_per_fire() {
            return;
        }
        for request in requests {
            let outcome = outcomes
                .get(request.outcome_index)
                .copied()
                .unwrap_or("unknown");
            let fire = request.timing;
            super::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "scheduler",
                "event": "scheduler_fire",
                "wave_id": timing.wave_id,
                "logical_fire_id": request.logical_fire_id,
                "instance_id": request.instance_id,
                "process_id": request.process_id,
                "sampled_rows": request.sampled_rows,
                "attempt": 1,
                "preparation_retries": 0,
                "outcome": outcome,
                "submitted_us": fire.submitted_us,
                "enqueued_us": fire.enqueued_us,
                "prepare_started_us": null,
                "prepared_us": null,
                "ready_us": fire.ready_us,
                "native_complete_us": native_complete_us,
                "settled_us": settled_us,
                "submit_to_enqueue_us": fire
                    .enqueued_us
                    .map(|value| value.saturating_sub(fire.submitted_us)),
                "prepare_us": 0,
                "ready_to_dispatch_us": fire.ready_us
                    .map(|ready| timing.dispatch_started_us.saturating_sub(ready)),
            }));
        }
    }

    /// Retire every settled control this pass. Concurrent standalone copies
    /// settle in device order, not post order, so the sweep cannot stop at
    /// the first control that is still outstanding.
    fn retire_ready_control(in_flight_control: &mut InFlightControls) -> bool {
        let mut retired = false;
        let mut index = 0;
        while index < in_flight_control.settling.len() {
            let ready = match &in_flight_control.settling[index].state {
                // Still waiting for the lane's reply to install the driver
                // completion (or drop the entry on rejection).
                ControlSlotState::Posted { .. } => None,
                ControlSlotState::Ready(completion) => completion.check(),
            };
            let Some(result) = ready else {
                index += 1;
                continue;
            };
            let pending = in_flight_control.settling.remove(index);
            let operation = pending.operation;
            if let Some(tracked) = pending.tracked_completion.as_ref() {
                tracked.resolve(&result);
            }
            if let Err(ref err) = result {
                tracing::warn!(
                    ?err,
                    operation,
                    "direct control completion closed before callback"
                );
                if let Some(logical) = pending.logical_completion.as_ref() {
                    logical.reject_unsubmitted(format!("pre-launch {operation} failed: {err:#}"));
                }
            }
            retired = true;
        }
        retired
    }

    /// Apply a driver-lane reply on the worker thread: fill in a posted
    /// launch's verdict, commit a control's worker-map effects, or install an
    /// async control's driver completion. Replies arrive in lane FIFO order.
    fn apply_lane_reply(
        reply: LaneReply,
        lane_inflight: &mut u64,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut InFlightControls,
        instances: &mut HashMap<u64, TrackedInstance>,
        frame_policy: &mut FramePolicy,
        rollback_tx: &crossbeam::channel::Sender<SchedulerItem>,
    ) {
        *lane_inflight = lane_inflight.saturating_sub(1);
        match reply {
            LaneReply::LaunchDone {
                token,
                result,
                driver_started_us,
                launch_returned_us,
            } => {
                let Some(batch) = in_flight_launches.iter_mut().find(
                    |batch| matches!(batch.state, LaunchState::Posted { token: t } if t == token),
                ) else {
                    // The batch can only leave the deque by retiring, and a
                    // Posted batch never retires — a missing token is a bug.
                    tracing::error!(token, "lane launch reply for an unknown batch");
                    return;
                };
                match result {
                    Ok(completion) => {
                        // Commit target epochs AT ACCEPT: lane replies arrive
                        // in post order — the driver's launch acceptance
                        // order — so the per-instance ledger stays gapless
                        // (a rejected launch commits nothing) and each
                        // completion's target matches the ordinal the
                        // instance slot will publish.
                        for request in &batch.requests {
                            if let Some(instance) = instances.get_mut(&request.instance_id) {
                                let epoch = instance.next_target_epoch;
                                request.completion.commit_target_epoch(epoch);
                                instance.next_target_epoch = epoch + 1;
                            }
                        }
                        if let Some(timing) = batch.timing.as_mut() {
                            timing.wave_id = completion.wait_id();
                            if let Some(at) = driver_started_us {
                                timing.driver_started_us = at;
                            }
                            if let Some(at) = launch_returned_us {
                                timing.launch_returned_us = at;
                            }
                        }
                        batch.state = LaunchState::Accepted(completion);
                    }
                    Err(message) => {
                        if let (Some(timing), Some(at)) =
                            (batch.timing.as_mut(), launch_returned_us)
                        {
                            timing.launch_returned_us = at;
                            if let Some(started) = driver_started_us {
                                timing.driver_started_us = started;
                            }
                        }
                        batch.state = LaunchState::Failed(message);
                    }
                }
            }
            LaneReply::ControlDone { token, commit } => match commit {
                LaneCommit::None => {}
                LaneCommit::BindFinished { pipeline_id } => {
                    frame_policy.on_bind_completed(pipeline_id);
                }
                LaneCommit::BindInstance {
                    pipeline_id,
                    bound,
                    respond,
                } => {
                    frame_policy.on_bind_completed(pipeline_id);
                    if instances.contains_key(&bound.instance_id) {
                        // Practically unreachable: driver-assigned ids are
                        // unique and requested ids are pre-checked at post
                        // (a guest awaits its bind response before it could
                        // reuse an id). Refuse loudly; the legit instance in
                        // the map stays untouched.
                        tracing::error!(
                            instance_id = bound.instance_id,
                            "bind committed an already-bound instance id"
                        );
                        let error = anyhow!("instance {} is already bound", bound.instance_id);
                        match respond {
                            BindRespond::Bind(response) => {
                                let _ = response.send(Err(error));
                            }
                            BindRespond::ChannelsBind { response, .. } => {
                                let _ = response.send(Err(error));
                            }
                        }
                        return;
                    }
                    let instance_id = bound.instance_id;
                    instances.insert(instance_id, TrackedInstance::from_bound(&bound));
                    // Respond AFTER the insert: launch admission reads
                    // `instances` on this thread, so the guest's first fire
                    // (sent only after this response) is always admissible.
                    match respond {
                        BindRespond::Bind(response) => {
                            if let Err(Ok(bound)) = response.send(Ok(bound)) {
                                tracing::warn!(
                                    operation = "bind_instance",
                                    instance_id = bound.instance_id,
                                    "scheduler cancellation rollback enqueued bound instance"
                                );
                                if rollback_tx
                                    .send(SchedulerItem::CloseInstance {
                                        id: bound.instance_id,
                                        pacing_wait_id: bound.pacing_wait_id,
                                    })
                                    .is_err()
                                {
                                    tracing::error!(
                                        operation = "bind_instance",
                                        instance_id = bound.instance_id,
                                        "scheduler cancellation rollback enqueue failed"
                                    );
                                }
                            }
                        }
                        BindRespond::ChannelsBind {
                            registered,
                            program_id,
                            program_registered,
                            response,
                        } => {
                            if let Err(Ok((registered, _, bound))) =
                                response.send(Ok((registered, program_id, bound)))
                            {
                                tracing::warn!(
                                    operation = "register_channels_bind",
                                    instance_id = bound.instance_id,
                                    channel_count = registered.len(),
                                    "scheduler cancellation rollback enqueued bound instance and channels"
                                );
                                if program_registered {
                                    tracing::warn!(
                                        operation = "register_channels_bind",
                                        program_id,
                                        "scheduler RPC cancelled after program registration; retaining driver-lifetime program"
                                    );
                                }
                                DriverLane::release_registered_channel_wait_slots(&registered);
                                let instance_id = bound.instance_id;
                                if rollback_tx
                                    .send(SchedulerItem::CloseInstance {
                                        id: instance_id,
                                        pacing_wait_id: bound.pacing_wait_id,
                                    })
                                    .is_err()
                                {
                                    tracing::error!(
                                        operation = "register_channels_bind",
                                        instance_id,
                                        "scheduler cancellation rollback close_instance enqueue failed"
                                    );
                                }
                                for channel in registered {
                                    let channel_id = channel.binding.channel_id;
                                    if rollback_tx
                                        .send(SchedulerItem::CloseChannel { id: channel_id })
                                        .is_err()
                                    {
                                        tracing::error!(
                                            operation = "register_channels_bind",
                                            channel_id,
                                            "scheduler cancellation rollback close_channel enqueue failed"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                LaneCommit::CloseInstance { id } => {
                    if let Some(instance) = instances.remove(&id) {
                        instance.close_wait_slots();
                    }
                }
                LaneCommit::AsyncControl { result } => {
                    // Replies arrive in lane FIFO order, but several
                    // standalone copies can be posted at once, so the reply
                    // is matched to its own entry by token.
                    let Some(index) = in_flight_control.position_posted(token) else {
                        tracing::error!(
                            token,
                            "lane async-control reply without a matching control slot"
                        );
                        return;
                    };
                    match result {
                        Ok(completion) => {
                            in_flight_control.settling[index].state =
                                ControlSlotState::Ready(completion);
                        }
                        // The lane already rejected/resolved the control's
                        // completions; the entry just leaves.
                        Err(_) => {
                            in_flight_control.settling.remove(index);
                        }
                    }
                }
            },
        }
    }

    fn shutdown_instances(
        driver: &mut Option<DriverBackend>,
        instances: &mut HashMap<u64, TrackedInstance>,
    ) {
        let outstanding = std::mem::take(instances);
        for (instance_id, instance) in outstanding {
            if let Some(driver) = driver.as_mut() {
                if let Err(err) = driver.close_instance(instance_id) {
                    tracing::warn!(
                        instance_id,
                        ?err,
                        "scheduler shutdown close_instance failed"
                    );
                }
            }
            instance.close_wait_slots();
        }
    }

    fn shutdown_channels(driver: &mut Option<DriverBackend>, channels: &mut HashSet<u64>) {
        let outstanding = std::mem::take(channels);
        for channel_id in outstanding {
            if let Some(driver) = driver.as_mut()
                && let Err(err) = driver.close_channel(channel_id)
            {
                tracing::warn!(channel_id, ?err, "scheduler shutdown close_channel failed");
            }
        }
    }
}

impl Drop for BatchScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct TrackedInstance {
    pacing_wait_id: u64,
    wait_slots: Arc<crate::driver::instance::BoundWaitSlots>,
    in_flight: usize,
    next_target_epoch: u64,
}

impl TrackedInstance {
    fn from_bound(bound: &BoundInstance) -> Self {
        Self {
            pacing_wait_id: bound.pacing_wait_id,
            wait_slots: bound.wait_slots(),
            in_flight: 0,
            next_target_epoch: pie_waker::FIRST_COMPLETION_EPOCH,
        }
    }

    fn close_wait_slots(self) {
        self.wait_slots.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{
        self, ChannelValue, DriverSpec, LaunchPlan, ProgramRegistration, SchedulerLimits,
    };
    use pie_driver_abi::{PieInstanceBinding, PieKvMoveCell, PiePoolRange};
    use pie_driver_dummy_lib::DummyDriverOptions;
    use pie_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use pie_ir::op::Op;
    use pie_ir::registry::Stage;
    use pie_ir::types::{DType, Literal, Shape};
    use tokio::time::{Duration, timeout};

    async fn setup_scheduler(
        operation_log: Arc<Mutex<Vec<String>>>,
    ) -> anyhow::Result<(
        usize,
        BatchScheduler,
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        setup_scheduler_with_options(DummyDriverOptions {
            operation_log: Some(operation_log),
            ..DummyDriverOptions::default()
        })
        .await
    }

    fn dummy_launch() -> LaunchPlan {
        LaunchPlan {
            token_ids: vec![1],
            position_ids: vec![0],
            kv_page_indptr: vec![0, 0],
            kv_last_page_lens: vec![0],
            qo_indptr: vec![0, 1],
            sampling_indices: vec![0],
            sampling_indptr: vec![0, 1],
            mask_indptr: vec![0, 0],
            single_token_mode: true,
            ..LaunchPlan::default()
        }
    }

    fn dummy_prefill(tokens: usize) -> LaunchPlan {
        let mut launch = dummy_launch();
        launch.token_ids = vec![1; tokens];
        launch.position_ids = (0..tokens as u32).collect();
        launch.qo_indptr = vec![0, tokens as u32];
        launch.sampling_indices = vec![tokens.saturating_sub(1) as u32];
        launch.single_token_mode = false;
        launch
    }

    /// Test lane over a driverless backend plus the reply stream the worker
    /// loop would normally drain.
    fn test_lane(
        driver: Option<DriverBackend>,
    ) -> (DriverLane, crossbeam::channel::Receiver<SchedulerItem>) {
        let (reply_tx, reply_rx) = crossbeam::channel::unbounded();
        let lane = DriverLane::spawn(
            usize::MAX,
            driver,
            reply_tx,
            Arc::new(SchedulerStats::default()),
        );
        (lane, reply_rx)
    }

    async fn wait_for_operation_count(
        operation_log: &Arc<Mutex<Vec<String>>>,
        operation: &str,
        count: usize,
    ) {
        timeout(Duration::from_secs(5), async {
            loop {
                if operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.as_str() == operation)
                    .count()
                    >= count
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("scheduler operation must complete");
    }

    fn chan(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 2,
            host_role: role,
            seeded,
        }
    }

    fn dummy_program() -> ProgramRegistration {
        let bytes = TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::None, true),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Const(Literal::U32(1)),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 0, value: 2 },
                    Op::ChanPut { chan: 1, value: 2 },
                ],
            }],
        }
        .encode();
        ProgramRegistration {
            program_hash: pie_ir::container_hash(&bytes),
            reference_ptir: bytes,
            ..Default::default()
        }
    }

    async fn register_test_channels(
        driver_id: usize,
        channel_ids: [u64; 2],
    ) -> anyhow::Result<Vec<Arc<crate::driver::ChannelEndpoint>>> {
        let mut endpoints = Vec::new();
        for (channel_id, host_role, seeded) in [
            (channel_ids[0], HostRole::None, true),
            (channel_ids[1], HostRole::Reader, false),
        ] {
            endpoints.push(
                crate::scheduler::register_channel(
                    driver_id,
                    ChannelRegistrationPlan {
                        driver_id,
                        channel_id,
                        shape: vec![1],
                        dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                        host_role: host_role as u8,
                        seeded,
                        extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                        capacity: 2,
                        reader_wait_id: 0,
                        writer_wait_id: 0,
                        extern_name: Vec::new(),
                    },
                )
                .await?,
            );
        }
        Ok(endpoints)
    }

    async fn setup_scheduler_with_options(
        options: DummyDriverOptions,
    ) -> anyhow::Result<(
        usize,
        BatchScheduler,
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        setup_scheduler_with_limits(
            options,
            SchedulerLimits {
                max_forward_requests: 1,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
        )
        .await
    }

    /// Like [`setup_scheduler_with_options`], but with a caller-chosen
    /// `SchedulerLimits` — the wait-all rule's structural cap
    /// (`max_forward_requests`) short-circuits any cold-hold/wait-all
    /// delay once a wave saturates it (see `frame::tests::
    /// structural_cap_seals_immediately_even_cold`), so every other test in
    /// this module runs at cap 1 and never observes the wait-all hold.
    /// Tests that need to actually exercise the hold (coalescing/leave)
    /// use this with a cap > 1 instead.
    async fn setup_scheduler_with_limits(
        options: DummyDriverOptions,
        limits: SchedulerLimits,
    ) -> anyhow::Result<(
        usize,
        BatchScheduler,
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        let driver_id = driver::register_driver_backend(
            DriverSpec {
                num_kv_pages: 16,
                limits,
                device_geometry_port_mask: 0,
            },
            DriverBackend::Dummy(crate::driver::DummyDriver::new(options)),
        );
        let scheduler = BatchScheduler::new(driver_id, driver_id, 16, limits, 1, 1);
        let program_id = crate::scheduler::register_program(driver_id, dummy_program()).await?;
        let endpoints = register_test_channels(driver_id, [7, 8]).await?;
        let bound = crate::scheduler::bind_instance(
            driver_id,
            None,
            program_id,
            41,
            vec![7, 8],
            vec![ChannelValue {
                channel: 7,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;
        Ok((driver_id, scheduler, bound, endpoints))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn typed_copy_paths_dispatch_to_distinct_driver_methods() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let copy_kv = crate::scheduler::copy_kv_cells(
            driver_id,
            vec![PieKvMoveCell {
                dst_page_id: 1,
                dst_token_offset: 0,
                src_page_id: 2,
                src_token_offset: 0,
            }],
        )
        .await?;
        timeout(Duration::from_secs(5), copy_kv).await??;
        let copy_state = crate::scheduler::copy_rs_d2d(driver_id, &[3], &[4]).await?;
        timeout(Duration::from_secs(5), copy_state).await??;
        crate::scheduler::close_instance(&bound)?;

        let log = operation_log.lock().unwrap().clone();
        let copy_kv_idx = log
            .iter()
            .position(|entry| entry == "copy_kv")
            .expect("copy_kv logged");
        let copy_state_idx = log
            .iter()
            .position(|entry| entry == "copy_state")
            .expect("copy_state logged");
        assert!(
            copy_kv_idx < copy_state_idx,
            "copy_kv should precede copy_state: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn resize_ops_run_before_queued_launches() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let resize = crate::scheduler::resize_pool(
            driver_id,
            7,
            32,
            vec![PiePoolRange {
                page_index: 0,
                page_count: 4,
            }],
            Vec::new(),
        )
        .await?;
        let launch = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            launch.clone(),
        )?;

        timeout(Duration::from_secs(5), resize).await??;
        timeout(Duration::from_secs(5), launch).await??;
        crate::scheduler::close_instance(&bound)?;

        let log = operation_log.lock().unwrap().clone();
        let resize_idx = log
            .iter()
            .position(|entry| entry == "resize_pool")
            .expect("resize_pool logged");
        let launch_idx = log
            .iter()
            .position(|entry| entry == "launch")
            .expect("launch logged");
        assert!(
            resize_idx < launch_idx,
            "resize should precede launch: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_instance_retires_bound_wait_slots() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pacing_wait_id = bound.pacing_wait_id;
        crate::scheduler::close_instance(&bound)?;

        timeout(Duration::from_secs(5), async {
            while pie_waker::WakerTable::global()
                .published(pacing_wait_id)
                .is_some()
            {
                tokio::task::yield_now().await;
            }
        })
        .await?;
        Ok(())
    }

    #[test]
    fn cancelled_register_channel_releases_wait_slots_before_creation() {
        let table = pie_waker::WakerTable::global();
        let reader_wait_id = table.alloc();
        let writer_wait_id = table.alloc();
        let (response, receiver) = tokio::sync::oneshot::channel();
        drop(receiver);
        let mut driver = None;
        let mut channels = HashSet::new();

        let commit = DriverLane::execute_control(
            &mut driver,
            &mut channels,
            QueuedItem::RegisterChannel {
                plan: ChannelRegistrationPlan {
                    driver_id: 0,
                    channel_id: 91,
                    shape: vec![1],
                    dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                    host_role: HostRole::None as u8,
                    seeded: false,
                    extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                    capacity: 1,
                    reader_wait_id,
                    writer_wait_id,
                    extern_name: Vec::new(),
                },
                response,
            },
        );

        assert!(matches!(commit, LaneCommit::None));
        assert!(table.published(reader_wait_id).is_none());
        assert!(table.published(writer_wait_id).is_none());
        assert!(channels.is_empty());
    }

    #[test]
    fn cancelled_bind_response_enqueues_instance_rollback() {
        let pacing_wait_id = pie_waker::WakerTable::global().alloc();
        let bound = BoundInstance::new(
            7,
            11,
            PieInstanceBinding {
                instance_id: 41,
                geometry_class: pie_driver_abi::GeometryClass::Host as u32,
                reserved0: 0,
            },
            pacing_wait_id,
        );
        let (response, receiver) = tokio::sync::oneshot::channel();
        drop(receiver);
        let (rollback_tx, rollback_rx) = crossbeam::channel::unbounded();
        let mut lane_inflight = 1;
        let mut launches = VecDeque::new();
        let mut control = InFlightControls::default();
        let mut instances = HashMap::new();
        let mut frame_policy = FramePolicy::new(1, 1, 4096, None);

        BatchScheduler::apply_lane_reply(
            LaneReply::ControlDone {
                token: 1,
                commit: LaneCommit::BindInstance {
                    pipeline_id: None,
                    bound,
                    respond: BindRespond::Bind(response),
                },
            },
            &mut lane_inflight,
            &mut launches,
            &mut control,
            &mut instances,
            &mut frame_policy,
            &rollback_tx,
        );

        assert!(matches!(
            rollback_rx.try_recv(),
            Ok(SchedulerItem::CloseInstance {
                id: 41,
                pacing_wait_id: wait_id,
            }) if wait_id == pacing_wait_id
        ));
        instances
            .remove(&41)
            .expect("cancelled bind remains tracked until ordered rollback")
            .close_wait_slots();
        assert!(
            pie_waker::WakerTable::global()
                .published(pacing_wait_id)
                .is_none()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn duplicate_bind_preserves_original_instance() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let error = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound.program_id,
            bound.instance_id,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await
        .expect_err("duplicate requested instance id must be rejected");
        assert!(error.to_string().contains("already bound"));

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        crate::scheduler::close_instance(&bound)?;

        let log = operation_log.lock().unwrap();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "bind_instance")
                .count(),
            1,
            "duplicate bind must be rejected before entering the backend"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_defers_slot_retirement_until_outstanding_completion_drops() -> anyhow::Result<()>
    {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pacing_wait_id = bound.pacing_wait_id;
        let outstanding = bound.reserve_completion();

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || crate::scheduler::close_instance(&bound)
        });

        std::thread::sleep(Duration::from_millis(10));
        assert!(
            close_bound.is_finished(),
            "close must not block the scheduler on an externally held completion"
        );
        close_bound.join().unwrap()?;
        assert!(
            !matches!(
                pie_waker::WakerTable::global().publish(pacing_wait_id, 1),
                pie_waker::WakeOutcome::Stale
            ),
            "bound wait slots remain leased until the completion drops"
        );
        drop(outstanding);

        assert!(matches!(
            pie_waker::WakerTable::global().publish(pacing_wait_id, 2),
            pie_waker::WakeOutcome::Stale
        ));
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn published_completion_survives_nonblocking_close_before_late_poll() -> anyhow::Result<()>
    {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;

        timeout(Duration::from_secs(5), async {
            loop {
                if pie_waker::WakerTable::global()
                    .published(completion.wait_id())
                    .is_some_and(|epoch| epoch >= completion.target_epoch())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await?;

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || crate::scheduler::close_instance(&bound)
        });
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            close_bound.is_finished(),
            "published terminal cells do not require close to wait for a late poll"
        );
        close_bound.join().unwrap()?;

        timeout(Duration::from_secs(5), completion).await??;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn one_instance_multi_row_rs_launch_reaches_dummy_intact() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) = setup_scheduler_with_limits(
            DummyDriverOptions {
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            },
            SchedulerLimits {
                max_forward_requests: 2,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
        )
        .await?;
        let mut launch = dummy_launch();
        launch.token_ids = vec![1, 2];
        launch.position_ids = vec![0, 0];
        launch.qo_indptr = vec![0, 1, 2];
        launch.kv_page_indptr = vec![0, 0, 0];
        launch.kv_last_page_lens = vec![0, 0];
        launch.sampling_indices = vec![0, 1];
        launch.sampling_indptr = vec![0, 1, 2];
        launch.mask_indptr = vec![0, 0, 0];
        launch.rs_slot_ids = vec![7, 9];
        launch.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET, 0];

        let completion = bound.reserve_completion();
        crate::scheduler::submit_async(
            launch,
            driver_id,
            bound.instance_id,
            0,
            None,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;

        assert!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .any(|entry| entry.starts_with("launch-shape tokens=2 programs=1"))
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn synchronous_launch_rejection_has_no_callback_or_epoch_gap() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                reject_launches_remaining: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let rejected = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            rejected.clone(),
        )?;
        let err = timeout(Duration::from_secs(5), rejected.clone())
            .await?
            .expect_err("rejected launch must fail");
        assert!(err.to_string().contains("direct launch rejected"));
        assert_eq!(
            rejected.target_epoch(),
            0,
            "rejected launch must not commit an epoch"
        );

        let accepted = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            accepted.clone(),
        )?;
        timeout(Duration::from_secs(5), accepted.clone()).await??;
        assert_eq!(
            accepted.target_epoch(),
            pie_waker::FIRST_COMPLETION_EPOCH,
            "the first accepted launch must still claim the first completion epoch"
        );

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "callback")
                .count(),
            1,
            "only the accepted launch may emit a callback: {log:?}"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exhausted_admission_preserves_wave_books_and_wakes_later() -> anyhow::Result<()> {
        // Folded admission (ABI v14): EXHAUSTED retries on the lane in
        // place — FIFO order holds and the fire completes once the pool
        // frees; the engine never observes the transient denial.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                prepare_exhaustions_remaining: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let stats = Arc::clone(scheduler.stats());
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion.clone()).await??;

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "launch-exhausted")
                .count(),
            1,
            "{log:?}"
        );
        assert_eq!(stats.total_batches.load(Ordering::Relaxed), 1);
        assert_eq!(
            stats.fire.quorum.wave_fires.load(Ordering::Relaxed),
            1,
            "the denied attempt must not count as a wave"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn impossible_admission_fails_without_parking() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                prepare_impossible_above_kv_pages: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let mut launch = dummy_launch();
        launch.required_kv_pages = 2;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            launch,
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let error = timeout(Duration::from_secs(1), completion)
            .await?
            .expect_err("impossible demand must fail explicitly");
        assert!(
            error.to_string().contains("physical budget ceiling"),
            "unexpected error: {error:#}"
        );
        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "launch-impossible")
                .count(),
            1,
            "{log:?}"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retry_terminal_fails_loudly_as_a_contract_violation() -> anyhow::Result<()> {
        // Venus (ABI v14): admitted frames are atomic and stream work is
        // SUCCESS-only, so a RETRY terminal surviving to frame settle is a
        // driver-contract violation — the fire fails loudly instead of
        // replaying (the makeup machinery is deleted).
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: 1,
                ..DummyDriverOptions::default()
            })
            .await?;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let error = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("a RETRY terminal must reject the fire");
        assert!(
            error.to_string().contains("RETRY"),
            "unexpected error: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn termination_preserves_launch_until_its_inflight_prelaunch_copy_retires() {
        let pid = ProcessId::new_v4();
        let completion = WorkItemCompletion::deferred_with_guard(None);
        let request = PendingRequest::direct(
            dummy_launch(),
            1,
            completion.clone(),
            0,
            Some(pid),
            Some(pid),
            false,
            None,
            None,
            None,
            false,
                /*hook_program=*/false,
                /*lora_program=*/false);
        let mut pending: PendingQueue =
            VecDeque::from([QueuedItem::Launch(QueuedLaunch::new(Box::new(request)))]).into();
        completion.request_cancel();
        BatchScheduler::reject_pipeline_queued(&mut pending, pid, Some(&completion));
        assert_eq!(pending.len(), 1);
        assert!(completion.cancel_requested());
        assert!(!completion.is_settled());
    }

    #[tokio::test]
    async fn tracked_control_completion_wakes_multiple_waiters() {
        let completion = ControlCompletion::new();
        let first = completion.clone();
        let second = completion.clone();
        let first = tokio::spawn(async move { first.wait().await });
        let second = tokio::spawn(async move { second.wait().await });
        tokio::task::yield_now().await;
        completion.resolve(&Ok(()));
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    #[test]
    fn aggregated_rs_copy_is_queued_before_its_launch() {
        let completion = WorkItemCompletion::deferred_with_guard(None);
        let state_copy = StateCopyPlan {
            slot_ranges: vec![
                pie_driver_abi::PieStateCopyRange {
                    src_slot_id: 3,
                    dst_slot_id: 5,
                    src_token_offset: 0,
                    dst_token_offset: 0,
                    token_count: 0,
                },
                pie_driver_abi::PieStateCopyRange {
                    src_slot_id: 3,
                    dst_slot_id: 6,
                    src_token_offset: 0,
                    dst_token_offset: 0,
                    token_count: 0,
                },
            ],
        };
        let request = PendingRequest::direct(
            dummy_launch(),
            1,
            completion,
            0,
            None,
            None,
            false,
            None,
            Some(state_copy),
            None,
            false,
                /*hook_program=*/false,
                /*lora_program=*/false);
        let mut pending = PendingQueue::default();
        BatchScheduler::queue_attempt(&mut pending, request);

        let QueuedItem::PreLaunchCopy {
            plan: PreLaunchCopy::State(plan),
            ..
        } = pending.pop_front().unwrap()
        else {
            panic!("aggregated state copy must precede the launch");
        };
        assert_eq!(plan.slot_ranges.len(), 2);
        assert_eq!(plan.slot_ranges[0].src_slot_id, 3);
        assert_eq!(plan.slot_ranges[1].dst_slot_id, 6);
        assert!(matches!(pending.pop_front(), Some(QueuedItem::Launch(_))));
        assert!(pending.is_empty());
    }

    #[test]
    fn unresolved_multi_row_prebuilt_request_remains_solo() {
        let mut launch = dummy_launch();
        launch.qo_indptr = vec![0, 0, 0];
        let pid = ProcessId::new_v4();
        let request = PendingRequest::direct(
            launch,
            1,
            WorkItemCompletion::deferred_with_guard(None),
            0,
            Some(pid),
            Some(pid),
            true,
            None,
            None,
            None,
            false,
                /*hook_program=*/false,
                /*lora_program=*/false);
        assert!(request.preserves_inner_rows());
        assert!(request.requires_solo_submission());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_terminal_outcome_rejects_launch_completion() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                fail_launches_after_accept: true,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let err = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("failed launch terminal outcome must fail");
        assert!(err.to_string().contains("Failed terminal outcome"));

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "callback")
                .count(),
            1,
            "failed accepted launches still publish exactly one callback: {log:?}"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn launches_can_overlap_before_prior_callback_when_fifo_allows() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 50,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let first = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            first.clone(),
        )?;

        let second = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            second.clone(),
        )?;

        let overlapping_launches = timeout(Duration::from_secs(5), async {
            loop {
                let launches = operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.as_str() == "launch")
                    .count();
                if launches >= 2 {
                    return launches;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        assert_eq!(
            overlapping_launches, 2,
            "launch 2 should submit before callback 1 when overlap is allowed"
        );

        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn same_instance_launches_can_run_ahead_across_batches() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 50,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let first = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            first.clone(),
        )?;

        let second = bound.reserve_completion();
        let second_for_submit = second.clone();
        let instance_id = bound.instance_id;
        let second_submit = std::thread::spawn(move || {
            crate::scheduler::submit_prebuilt_async(
                dummy_launch(),
                driver_id,
                instance_id,
                0,
                second_for_submit,
            )
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            2,
            "same-instance launch 2 should be accepted before launch 1 callback"
        );
        assert!(
            second_submit.is_finished(),
            "same-instance acceptance should not wait for launch 1 callback"
        );

        second_submit.join().unwrap()?;
        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queued_resize_does_not_gate_launches_and_dispatches_at_drain() -> anyhow::Result<()> {
        // Venus: ResizePool is a pure capacity operation — the driver's
        // quiescence gate holds correctness, so fires never wait for a
        // queued resize (the old FIFO barrier paced gen-boundary teardown
        // to one frame per resize cycle). The resize itself dispatches
        // once the launch pipe drains.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 50,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let first = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            first.clone(),
        )?;

        let resize_join = tokio::spawn(async move {
            crate::scheduler::resize_pool(
                driver_id,
                7,
                32,
                vec![PiePoolRange {
                    page_index: 0,
                    page_count: 4,
                }],
                Vec::new(),
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        let second = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            second.clone(),
        )?;

        // Both launches complete without waiting for the queued resize.
        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        // The resize dispatches once the pipe drains, and completes.
        let resize = resize_join.await??;
        timeout(Duration::from_secs(5), resize).await??;

        let log = operation_log.lock().unwrap().clone();
        let launches: Vec<usize> = log
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.as_str() == "launch")
            .map(|(index, _)| index)
            .collect();
        let resize_idx = log
            .iter()
            .position(|entry| entry == "resize_pool")
            .expect("resize dispatched");
        assert_eq!(launches.len(), 2, "{log:?}");
        assert!(
            launches.iter().all(|&launch| launch < resize_idx),
            "the resize dispatches only at pipe drain, after both launches: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_enqueues_before_accepted_launch_retires() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 75,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let launch = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            bound.driver_id,
            bound.instance_id,
            0,
            launch.clone(),
        )?;

        let started = std::time::Instant::now();
        crate::scheduler::close_instance(&bound)?;
        assert!(
            started.elapsed() < Duration::from_millis(10),
            "fire-and-forget close must return after enqueue"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(
            !operation_log
                .lock()
                .unwrap()
                .iter()
                .any(|entry| entry == "close_instance"),
            "native close still waits for the accepted launch to retire"
        );
        timeout(Duration::from_secs(5), launch).await??;
        wait_for_operation_count(&operation_log, "close_instance", 1).await;

        let log = operation_log.lock().unwrap().clone();
        let launch_idx = log.iter().position(|entry| entry == "launch").unwrap();
        let close_idx = log
            .iter()
            .position(|entry| entry == "close_instance")
            .unwrap();
        assert!(
            launch_idx < close_idx,
            "close should happen after launch retires: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stale_instance_close_is_fire_and_forget() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, scheduler, bound, endpoints) =
            setup_scheduler(operation_log.clone()).await?;
        crate::scheduler::close_instance(&bound)?;
        crate::scheduler::close_instance(&bound)?;
        drop(endpoints);
        drop(scheduler);
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "close_instance")
                .count(),
            2,
            "both fire-and-forget requests are attempted; stale-close diagnostics are scheduler-owned"
        );
        Ok(())
    }

    /// A close needs only ITS OWN instance quiesced: instance B's close
    /// completes while instance A's launch is still in flight. The old
    /// behavior held every close hostage to a global pipe drain, which at
    /// cohort swaps stalled all queued launches behind a front close.
    #[tokio::test(flavor = "current_thread")]
    async fn close_of_idle_instance_overlaps_in_flight_launches() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 200,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let program_id = crate::scheduler::register_program(driver_id, dummy_program()).await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let launch = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            bound_a.driver_id,
            bound_a.instance_id,
            0,
            launch.clone(),
        )?;

        // Give the worker a moment to dispatch A's launch into flight.
        std::thread::sleep(Duration::from_millis(20));
        let started = std::time::Instant::now();
        crate::scheduler::close_instance(&bound_b)?;
        assert!(
            started.elapsed() < Duration::from_millis(120),
            "idle-instance close must not wait for the pipe to drain"
        );
        wait_for_operation_count(&operation_log, "close_instance", 1).await;

        let log = operation_log.lock().unwrap().clone();
        let launch_idx = log.iter().position(|entry| entry == "launch");
        let close_idx = log.iter().position(|entry| entry == "close_instance");
        assert!(
            launch_idx.is_some() && close_idx.is_some() && launch_idx < close_idx,
            "B's close must overlap A's in-flight launch: {log:?}"
        );
        timeout(Duration::from_secs(5), launch).await??;
        Ok(())
    }

    /// Strict wait-all lets a synchronous lifecycle burst fill the lane in one
    /// pass; no wave window can age while the worker drains its mailbox.
    #[tokio::test(flavor = "current_thread")]
    async fn synchronous_control_burst_dispatches_in_one_pass() {
        let (tx_a, mut rx_a) = tokio::sync::oneshot::channel();
        let (tx_b, mut rx_b) = tokio::sync::oneshot::channel();
        let mut pending: PendingQueue = VecDeque::from([
            QueuedItem::RegisterProgram {
                plan: dummy_program(),
                response: tx_a,
            },
            QueuedItem::RegisterProgram {
                plan: dummy_program(),
                response: tx_b,
            },
        ])
        .into();
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = InFlightControls::default();
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());
        let mut frame_policy = FramePolicy::new(
            1,
            limits.max_forward_requests,
            limits.max_forward_tokens,
            None,
        );

        let (progress, _) = BatchScheduler::dispatch_ready_items(
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            &mut in_flight_control,
            16,
            limits,
            &stats,
            &mut frame_policy,
            &mut ScanCache::default(),
            &mut SlotBuffer::new(),
            false,
        );
        assert!(progress);
        assert!(
            timeout(Duration::from_secs(5), &mut rx_a).await.is_ok(),
            "the first control dispatches this pass"
        );
        assert!(
            timeout(Duration::from_secs(5), &mut rx_b).await.is_ok(),
            "the second control dispatches in the same pass"
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn instance_queued_work_gate_sees_launches() {
        let pid = ProcessId::new_v4();
        let pending = VecDeque::from([QueuedItem::Launch(QueuedLaunch::new(
            dummy_launch_request(pid, 7),
        ))]);
        assert!(BatchScheduler::instance_has_queued_work(&pending, 7));
        assert!(!BatchScheduler::instance_has_queued_work(&pending, 8));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scheduler_shutdown_drains_instances_and_destroys_once() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 40,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let program_id = crate::scheduler::register_program(driver_id, dummy_program()).await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let resize =
            crate::scheduler::resize_pool(driver_id, 9, 16, Vec::new(), Vec::new()).await?;
        let a = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            a,
        )?;
        let b = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            b,
        )?;
        drop(resize);
        drop(scheduler);

        let log = operation_log.lock().unwrap().clone();
        // The shutdown drain posts both instances' fires as ONE frame
        // (a single driver launch with a two-member step).
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1
        );
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "close_instance")
                .count(),
            2
        );
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "destroy")
                .count(),
            1
        );
        let destroy_idx = log.iter().position(|entry| entry == "destroy").unwrap();
        let last_callback_idx = log
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| (entry == "callback").then_some(idx))
            .max()
            .unwrap();
        assert!(
            last_callback_idx < destroy_idx,
            "destroy must be last after callbacks: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_retirement_is_event_driven() -> anyhow::Result<()> {
        // Plan §14 gate 6: the driver callback's nudge retires the batch, not
        // the backstop poll. A retirement that misses the nudge waits out the
        // 250 ms backstop and trips the bound below.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 30,
                operation_log: Some(operation_log),
                ..DummyDriverOptions::default()
            })
            .await?;
        let backstops_before = backstop_retirements();
        let completion = bound.reserve_completion();
        let started = Instant::now();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "retirement must ride the completion nudge, not the backstop poll (took {elapsed:?})"
        );
        assert_eq!(
            backstop_retirements(),
            backstops_before,
            "steady state retires with zero backstop-path wakeups (plan §16.2)"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parked_reader_wakes_straight_from_the_driver_callback() -> anyhow::Result<()> {
        // Plan §14 gates 2/3: a task that never submitted (and drains no
        // pipeline FIFO) parks on the channel's reader wait slot and wakes
        // straight from the driver's per-channel notify, with the published
        // tail word already visible.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) = setup_scheduler(operation_log).await?;
        let waiter = tokio::spawn({
            let endpoint = Arc::clone(&endpoints[1]);
            async move { endpoint.wait_for_reader_change(0).await }
        });
        tokio::task::yield_now().await;

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), waiter)
            .await??
            .expect("reader wake surfaces the new tail, not an error");
        let binding = endpoints[1].registered().binding;
        let tail = unsafe {
            (&*((binding.word_base as *const std::sync::atomic::AtomicU64)
                .add(binding.tail_word_index as usize)))
                .load(Ordering::Acquire)
        };
        assert_eq!(tail, 1, "the tail word is published before the wake");
        timeout(Duration::from_secs(5), completion).await??;
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parked_reader_wakes_into_poisoned_not_empty() -> anyhow::Result<()> {
        // Plan §14 gate 7: a failed fire release-stores the poison word BEFORE
        // the channel notify, so a parked reader wakes into Poisoned — never
        // into a spurious Empty retry.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                fail_launches_after_accept: true,
                operation_log: Some(operation_log),
                ..DummyDriverOptions::default()
            })
            .await?;
        let waiter = tokio::spawn({
            let endpoint = Arc::clone(&endpoints[1]);
            async move { endpoint.wait_for_reader_change(0).await }
        });
        tokio::task::yield_now().await;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let woke = timeout(Duration::from_secs(5), waiter).await??;
        assert!(
            matches!(
                woke,
                Err(crate::driver::channel::ChannelWaitError::Poisoned(_))
            ),
            "a parked take classifies the failed fire as Poisoned, got {woke:?}"
        );
        let _ = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("the failed fire's terminal outcome is surfaced");
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn extern_export_flows_into_importing_instance() -> anyhow::Result<()> {
        // Plan §14 gate 3: instance A's fire fills a shared extern channel;
        // instance B's fire consumes it and publishes to its host reader —
        // cross-instance dataflow over one global channel registration.
        use pie_ir::container::{ExternDecl, ExternDir};
        let driver_id = driver::register_driver_backend(
            DriverSpec {
                num_kv_pages: 16,
                limits: SchedulerLimits {
                    max_forward_requests: 1,
                    max_forward_tokens: 64,
                    max_page_refs: 64,
                },
                device_geometry_port_mask: 0,
            },
            DriverBackend::Dummy(crate::driver::DummyDriver::new(
                DummyDriverOptions::default(),
            )),
        );
        let _scheduler = BatchScheduler::new(
            driver_id,
            driver_id,
            16,
            SchedulerLimits {
                max_forward_requests: 1,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
            1,
            1,
        );
        let exporter_bytes = TraceContainer {
            names: vec!["shared".to_string()],
            externs: vec![ExternDecl {
                name: 0,
                dir: ExternDir::Export,
                chan: 0,
            }],
            channels: vec![chan(Shape::vector(1), DType::U32, HostRole::None, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::Const(Literal::U32(7)),
                    Op::Broadcast {
                        value: 0,
                        shape: Shape::vector(1),
                    },
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
        }
        .encode();
        let importer_bytes = TraceContainer {
            names: vec!["shared".to_string()],
            externs: vec![ExternDecl {
                name: 0,
                dir: ExternDir::Import,
                chan: 0,
            }],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::None, false),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
            }],
        }
        .encode();
        let exporter_program = crate::scheduler::register_program(
            driver_id,
            ProgramRegistration {
                program_hash: pie_ir::container_hash(&exporter_bytes),
                reference_ptir: exporter_bytes,
                ..Default::default()
            },
        )
        .await?;
        let importer_program = crate::scheduler::register_program(
            driver_id,
            ProgramRegistration {
                program_hash: pie_ir::container_hash(&importer_bytes),
                reference_ptir: importer_bytes,
                ..Default::default()
            },
        )
        .await?;
        let shared = crate::scheduler::register_channel(
            driver_id,
            ChannelRegistrationPlan {
                driver_id,
                channel_id: 91,
                shape: vec![1],
                dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                host_role: HostRole::None as u8,
                seeded: false,
                extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_EXPORT,
                capacity: 2,
                reader_wait_id: 0,
                writer_wait_id: 0,
                extern_name: b"shared".to_vec(),
            },
        )
        .await?;
        let reader = crate::scheduler::register_channel(
            driver_id,
            ChannelRegistrationPlan {
                driver_id,
                channel_id: 92,
                shape: vec![1],
                dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                host_role: HostRole::Reader as u8,
                seeded: false,
                extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                capacity: 2,
                reader_wait_id: 0,
                writer_wait_id: 0,
                extern_name: Vec::new(),
            },
        )
        .await?;
        let _ = shared;
        let exporter = crate::scheduler::bind_instance(
            driver_id,
            None,
            exporter_program,
            61,
            vec![91],
            Vec::new(),
        )
        .await?;
        let importer = crate::scheduler::bind_instance(
            driver_id,
            None,
            importer_program,
            62,
            vec![91, 92],
            Vec::new(),
        )
        .await?;

        // A parked take on the importer's reader — a task that never
        // submitted anything — observes the cross-instance flow end to end.
        let waiter = tokio::spawn({
            let endpoint = Arc::clone(&reader);
            async move { endpoint.wait_for_reader_change(0).await }
        });
        tokio::task::yield_now().await;
        let export_fire = exporter.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            exporter.instance_id,
            0,
            export_fire.clone(),
        )?;
        let import_fire = importer.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            importer.instance_id,
            0,
            import_fire.clone(),
        )?;
        timeout(Duration::from_secs(5), export_fire).await??;
        timeout(Duration::from_secs(5), import_fire).await??;
        timeout(Duration::from_secs(5), waiter)
            .await??
            .expect("the importer's publish wakes the parked reader");
        let binding = reader.registered().binding;
        let value = unsafe { std::ptr::read_unaligned(binding.mirror_base as *const u32) };
        assert_eq!(value, 7, "the exported value crossed instances");
        crate::scheduler::close_instance(&exporter)?;
        crate::scheduler::close_instance(&importer)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn timeout_bounded_shutdown_stress() -> anyhow::Result<()> {
        timeout(Duration::from_secs(5), async {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (driver_id, scheduler, bound, _endpoints) =
                setup_scheduler_with_options(DummyDriverOptions {
                    callback_delay_ms: 5,
                    operation_log: Some(operation_log),
                    ..DummyDriverOptions::default()
                })
                .await?;
            for _ in 0..16 {
                let completion = bound.reserve_completion();
                crate::scheduler::submit_prebuilt_async(
                    dummy_launch(),
                    driver_id,
                    bound.instance_id,
                    0,
                    completion,
                )?;
            }
            drop(scheduler);
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    /// Every wait-all-hold test below needs a structural cap big enough
    /// that a single request never trivially saturates it (else the
    /// wait-all rule short-circuits straight to a seal — see
    /// `frame::tests::structural_cap_seals_immediately_even_cold`).
    fn coalescing_limits() -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 4,
            max_forward_tokens: 64,
            max_page_refs: 64,
        }
    }

    /// Binds a second instance on the same program/driver as `bound_a`, for
    /// tests that need two independent pipelines' fires in flight at once.
    async fn bind_second_instance(
        driver_id: usize,
        bound_a: &crate::driver::BoundInstance,
        channel_ids: [u64; 2],
        requested_instance_id: u64,
    ) -> anyhow::Result<(
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        let endpoints = register_test_channels(driver_id, channel_ids).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound_a.program_id,
            requested_instance_id,
            channel_ids.to_vec(),
            vec![ChannelValue {
                channel: channel_ids[0],
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;
        Ok((bound_b, endpoints))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_pipelines_coalesce_into_one_wave() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) = setup_scheduler_with_limits(
            DummyDriverOptions {
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            },
            coalescing_limits(),
        )
        .await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [27, 28], 52).await?;

        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        // Submitted back-to-back, no await in between: both land in the
        // scheduler's queue before it next drains, so both `on_pipeline_
        // request` calls land in the SAME wave-gather.
        let first = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            Some(pid_a),
            first.clone(),
        )?;
        let second = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            Some(pid_b),
            second.clone(),
        )?;

        // The wait-all gate holds the seal until every member is ready, so
        // both pipelines' first requests land in ONE dense wave
        // (`requests=2`) instead of two solo fires — the dummy driver's
        // launch-shape trace names the program count directly.
        let coalesced = timeout(Duration::from_secs(5), async {
            loop {
                let hit = operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|entry| entry.starts_with("launch-shape tokens=2 programs=2"));
                if hit {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await?;
        assert!(
            coalesced,
            "both pipelines' first requests should coalesce into one programs=2 wave: {:?}",
            operation_log.lock().unwrap()
        );

        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn token_capacity_partitions_wait_all_wave_without_deadlock() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let limits = SchedulerLimits {
            max_forward_requests: 4,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let (driver_id, _scheduler, bound_a, _endpoints) = setup_scheduler_with_limits(
            DummyDriverOptions {
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            },
            limits,
        )
        .await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [29, 30], 53).await?;
        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        for _ in 0..2 {
            let first = bound_a.reserve_completion();
            crate::scheduler::submit_async(
                dummy_prefill(40),
                driver_id,
                bound_a.instance_id,
                0,
                Some(pid_a),
                first.clone(),
            )?;
            let second = bound_b.reserve_completion();
            crate::scheduler::submit_async(
                dummy_prefill(40),
                driver_id,
                bound_b.instance_id,
                0,
                Some(pid_b),
                second.clone(),
            )?;

            timeout(Duration::from_secs(5), first).await??;
            timeout(Duration::from_secs(5), second).await??;
        }

        let launches = operation_log
            .lock()
            .unwrap()
            .iter()
            .filter(|entry| entry.starts_with("launch-shape tokens=40 programs=1"))
            .count();
        assert_eq!(
            launches,
            4,
            "each logical wave should split into two capacity-limited launches: {:?}",
            operation_log.lock().unwrap()
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn leave_unblocks_a_wave_holding_for_a_missing_member() -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [27, 28], 54).await?;

        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        // Wave 1: both pipelines seen, both in the wait-set.
        let first_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            Some(pid_a),
            first_a.clone(),
        )?;
        let first_b = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            Some(pid_b),
            first_b.clone(),
        )?;
        timeout(Duration::from_secs(5), first_a).await??;
        timeout(Duration::from_secs(5), first_b).await??;

        // Wave 2: only `a` resubmits; `b` instead leaves the fleet. The
        // quorum drops it from the wait-set and releases `a`.
        let started = Instant::now();
        let second_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            Some(pid_a),
            second_a.clone(),
        )?;
        post_process_terminate(pid_b);
        timeout(Duration::from_secs(5), second_a).await??;
        assert!(
            started.elapsed() < Duration::from_millis(8),
            "leave should unblock the wait-all hold promptly, took {:?}",
            started.elapsed()
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scoped_leave_does_not_remove_a_sibling_pipeline_of_the_same_process()
    -> anyhow::Result<()> {
        let (driver_id, scheduler, bound_a, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [31, 32], 55).await?;
        let process_id = ProcessId::new_v4();
        let pipeline_a = ProcessId::new_v4();
        let pipeline_b = ProcessId::new_v4();

        for (bound, pipeline_id) in [(&bound_a, pipeline_a), (&bound_b, pipeline_b)] {
            let completion = bound.reserve_completion();
            scheduler.handle.submit_prebuilt_tracked_with_copy(
                dummy_launch(),
                bound.instance_id,
                completion.clone(),
                0,
                process_id,
                pipeline_id,
                None,
                None,
                None,
                false,
                /*hook_program=*/false,
                /*lora_program=*/false)?;
            if pipeline_id == pipeline_b {
                timeout(Duration::from_secs(5), completion).await??;
            }
        }

        // Wait for the first wave's other completion before starting wave 2.
        // Both scopes now belong to the same process but have independent
        // quorum membership.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let sibling = bound_b.reserve_completion();
        scheduler.handle.submit_prebuilt_tracked_with_copy(
            dummy_launch(),
            bound_b.instance_id,
            sibling.clone(),
            0,
            process_id,
            pipeline_b,
            None,
            None,
            None,
            false,
                /*hook_program=*/false,
                /*lora_program=*/false)?;
        notify_pipeline_close(pipeline_a).await;
        timeout(Duration::from_secs(5), sibling).await??;

        let dump = scheduler.handle.debug_dump().await?;
        assert!(
            dump.contains(&pipeline_b.to_string()),
            "sibling pipeline must remain in the quorum:\n{dump}"
        );
        assert!(
            !dump.contains(&format!("pipeline {pipeline_a}")),
            "only the departed scope should be removed:\n{dump}"
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pipeline_close_drains_the_already_submitted_run_ahead_tail() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 25,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let pid = ProcessId::new_v4();
        let mut completions = Vec::new();
        for _ in 0..3 {
            let completion = bound.reserve_completion();
            crate::scheduler::submit_async(
                dummy_launch(),
                driver_id,
                bound.instance_id,
                0,
                Some(pid),
                completion.clone(),
            )?;
            completions.push(completion);
        }

        // FIFO receipt puts this after all three launches. At least one launch
        // remains queued behind the scheduler's run-ahead depth while close
        // releases the wait-set; none may be cancelled.
        notify_pipeline_close(pid).await;

        // Drain the reader ring CONCURRENTLY, as a real host reader
        // (`channel.take`) does. The third fire is dispatched the instant
        // run-ahead frees a slot, which is the same moment the first fire's
        // completion resolves — so a test that only advances `head` after
        // awaiting completions races the scheduler, and a fire that lands on
        // a full 2-cell ring latches RETRY (a v14 contract violation). Nothing
        // else bounds it here: `submit_async` is the raw scheduler entry and
        // bypasses the pipeline's submit-time ring-occupancy admission
        // (`validate_frame`), which is what keeps this in range in production.
        // On a `current_thread` runtime this task interleaves at exactly the
        // awaits below, i.e. whenever the test is blocked on a completion.
        let binding = endpoints[1].registered().binding;
        let words = binding.word_base as usize;
        let drainer = tokio::task::spawn(async move {
            let mut drained = 0u64;
            loop {
                {
                    // Derived inside the loop body: a raw pointer held across
                    // the await below would make this future !Send.
                    let words = words as *const std::sync::atomic::AtomicU64;
                    let tail = unsafe {
                        (&*words.add(binding.tail_word_index as usize)).load(Ordering::Acquire)
                    };
                    if tail > drained {
                        drained = tail;
                        unsafe {
                            (&*words.add(binding.head_word_index as usize))
                                .store(tail, Ordering::Release);
                        }
                        crate::scheduler::nudge(driver_id);
                    }
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        for index in 0..3 {
            timeout(Duration::from_secs(5), completions.remove(0))
                .await
                .unwrap_or_else(|_| panic!("fire {index} did not complete after close"))?;
        }
        drainer.abort();

        // Every settled output stayed visible across the close: none was
        // poisoned or discarded, so the ring published all three.
        let tail = unsafe {
            (&*(words as *const std::sync::atomic::AtomicU64).add(binding.tail_word_index as usize))
                .load(Ordering::Acquire)
        };
        assert_eq!(tail, 3, "settled outputs remain visible after close");

        assert!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count()
                >= 3,
            "close must preserve queued, preparing, and dispatched fires"
        );

        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn untracked_prebuilt_fire_never_blocks_on_the_quorum() -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;

        // `submit_prebuilt_async` always carries `pipeline_id: None` — it
        // never joins the wait-set, so it must fire promptly even though
        // nothing else is active to gather with it (bootstrap cold-hold at
        // most).
        let started = Instant::now();
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "an untracked prebuilt fire must never hold for the quorum, took {:?}",
            started.elapsed()
        );

        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    fn dummy_launch_request(pipeline_id: ProcessId, instance_id: u64) -> Box<PendingRequest> {
        Box::new(PendingRequest::direct(
            dummy_launch(),
            instance_id,
            WorkItemCompletion::deferred_with_guard(None),
            0,
            Some(pipeline_id),
            Some(pipeline_id),
            false,
            None,
            None,
            None,
            false,
                /*hook_program=*/false,
                /*lora_program=*/false))
    }

    #[test]
    fn launch_grouping_uses_driver_token_capacity() {
        let limits = SchedulerLimits {
            max_forward_requests: 4,
            max_forward_tokens: 4096,
            max_page_refs: 4096,
        };
        let mut first = dummy_launch_request(ProcessId::new_v4(), 1);
        first.request = dummy_prefill(1536);
        let mut second = dummy_launch_request(ProcessId::new_v4(), 2);
        second.request = dummy_prefill(1536);

        let mut grouping = LaunchGrouping::default();
        assert!(grouping.accepts(&first, limits, 16));
        grouping.push(&first, limits, 16);
        assert!(
            grouping.accepts(&second, limits, 16),
            "the scheduler must not impose a token cap below the driver limit"
        );
    }

    #[test]
    fn launch_grouping_only_solos_device_derived_masks() {
        let limits = SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let mut host_mask = dummy_launch_request(ProcessId::new_v4(), 1);
        host_mask.request.has_user_mask = true;
        host_mask.request.masks = vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        host_mask.request.mask_indptr = vec![0, 1];
        let causal = dummy_launch_request(ProcessId::new_v4(), 2);

        let mut grouping = LaunchGrouping::default();
        assert!(grouping.accepts(&host_mask, limits, 16));
        assert!(
            !grouping.push(&host_mask, limits, 16),
            "a host-derived wire mask must not close the batch"
        );
        assert!(
            grouping.accepts(&causal, limits, 16),
            "host-derived custom and causal fires should co-batch"
        );

        let mut dense = dummy_launch_request(ProcessId::new_v4(), 3);
        dense.request.has_user_mask = true;
        dense.request.device_resolved_geometry = true;
        let mut grouping = LaunchGrouping::default();
        assert!(grouping.accepts(&dense, limits, 16));
        assert!(
            grouping.push(&dense, limits, 16),
            "a device-derived dense mask remains a solo batch"
        );

        let mut host_on_device = dummy_launch_request(ProcessId::new_v4(), 4);
        host_on_device.request.has_user_mask = true;
        host_on_device.request.device_resolved_geometry = true;
        host_on_device.request.masks =
            vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        host_on_device.request.mask_indptr = vec![0, 1];
        let mut grouping = LaunchGrouping::default();
        assert!(
            !grouping.push(&host_on_device, limits, 16),
            "wire rows distinguish a host-derived mask from dense device lowering"
        );
        let mut ordinary_group = LaunchGrouping::default();
        ordinary_group.push(&dummy_launch_request(ProcessId::new_v4(), 5), limits, 16);
        assert!(
            !ordinary_group.accepts(&host_on_device, limits, 16),
            "resolved-geometry host masks remain incompatible with reordered wire rows"
        );

        // The wire rows above are an INFERENCE, not the binding. A program
        // that binds `AttnMask` to a channel gets its dense mask resolved on
        // device whether or not this fire also lowered BRLE rows, so the
        // binding itself has to keep the fire solo. `cuda_runahead_concurrent`
        // is the case: 8 pipelines of a sink/sliding-window decode program,
        // every fire carrying BOTH wire rows and the channel binding, batched
        // into one step that the driver rejects — the failed prepare poisons
        // descriptor channel 0 and every stream is lost.
        let mut bound_dense = dummy_launch_request(ProcessId::new_v4(), 6);
        bound_dense.request.dense_device_mask = true;
        bound_dense.request.has_user_mask = true;
        bound_dense.request.masks = vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        bound_dense.request.mask_indptr = vec![0, 1];
        let mut grouping = LaunchGrouping::default();
        assert!(
            grouping.push(&bound_dense, limits, 16),
            "a channel-bound dense mask seals its step even with wire rows"
        );
        let mut ordinary_group = LaunchGrouping::default();
        ordinary_group.push(&dummy_launch_request(ProcessId::new_v4(), 7), limits, 16);
        assert!(
            !ordinary_group.accepts(&bound_dense, limits, 16),
            "and never joins a step that already has a member"
        );
    }

    /// Rotating held launches behind wave work must move the WHOLE contiguous
    /// launch prefix in one call. A partial rotation reorders a pipeline's
    /// run-ahead siblings; dispatch then defers the out-of-order head
    /// (`launch_has_earlier_instance_member`) and, with the earlier sibling
    /// sitting beyond a non-launch item, can never reach it — a permanent
    /// scheduler stall (the V5 benchmark deadlock, 2026-07-15).
    #[test]
    fn launch_rotation_preserves_per_instance_order() {
        let pipeline_a = ProcessId::new_v4();
        let pipeline_b = ProcessId::new_v4();
        let mut pending = PendingQueue::default();
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(dummy_launch_request(
            pipeline_a, 1,
        ))));
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(dummy_launch_request(
            pipeline_a, 1,
        ))));
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(dummy_launch_request(
            pipeline_b, 2,
        ))));
        pending.push_back(QueuedItem::CloseInstance {
            id: 9,
            pacing_wait_id: 0,
        });

        assert!(BatchScheduler::rotate_launch_for_wave_work(
            &mut pending,
            true,
            true
        ));

        assert!(
            matches!(pending.front(), Some(QueuedItem::CloseInstance { .. })),
            "the rotate-target work must reach the queue front"
        );
        let launches: Vec<(u64, u64)> = pending
            .iter()
            .filter_map(|item| match item {
                QueuedItem::Launch(request) => Some((request.instance_id, request.logical_fire_id)),
                _ => None,
            })
            .collect();
        assert_eq!(
            launches
                .iter()
                .map(|(instance, _)| *instance)
                .collect::<Vec<_>>(),
            vec![1, 1, 2],
            "rotation must not interleave the launch prefix"
        );
        assert!(
            launches[0].1 < launches[1].1,
            "same-instance run-ahead fires must stay FIFO across rotation"
        );
    }

    /// A `PreLaunchCopy` queued behind held launches is dispatchable control
    /// work: rotation must treat it as a valid target when controls are
    /// allowed (it occupies the free control slot exactly like a lifecycle
    /// control), or a held front launch starves the copy — and the copy's
    /// consumer launch — forever.
    #[test]
    fn launch_rotation_reaches_a_pre_launch_copy() {
        let make_pending = || {
            let mut pending = PendingQueue::default();
            pending.push_back(QueuedItem::Launch(QueuedLaunch::new(dummy_launch_request(
                ProcessId::new_v4(),
                1,
            ))));
            pending.push_back(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(crate::driver::KvCopyPlan::default()),
                logical_completion: WorkItemCompletion::deferred_with_guard(None),
                process_id: None,
                pipeline_id: Some(ProcessId::new_v4()),
            });
            pending
        };

        let mut pending = make_pending();
        assert!(
            !BatchScheduler::rotate_launch_for_wave_work(&mut pending, false, false),
            "a settling control slot (controls disallowed) must keep launch order"
        );

        let mut pending = make_pending();
        assert!(BatchScheduler::rotate_launch_for_wave_work(
            &mut pending,
            true,
            true
        ));
        assert!(
            matches!(pending.front(), Some(QueuedItem::PreLaunchCopy { .. })),
            "the copy must reach the front so it can occupy the control slot"
        );
    }

    /// A queued fire cancelled before native launch drops at dispatch
    /// WITHOUT corrupting the wave books: its sealed wave resolves without
    /// it, nothing launches, and the lane stays awaited for the next epoch
    /// (the frame-policy successor of the old RV-20 credit guard).
    #[test]
    fn cancelled_fire_drops_and_resolves_out_of_its_sealed_wave() {
        let pid = ProcessId::new_v4();
        let completion = WorkItemCompletion::deferred_with_guard(None);
        let stamp = FrameStamp {
            lane: pid,
            seq: 1,
            slot: 0,
            fires: 1,
        };
        let request = PendingRequest::direct(
            dummy_launch(),
            7,
            completion.clone(),
            0,
            Some(pid),
            Some(pid),
            false,
            None,
            None,
            Some(stamp),
            false,
                /*hook_program=*/false,
                /*lora_program=*/false);
        let fire_id = request.logical_fire_id;
        // Row budget 1: the single-fire wave is structurally full and seals
        // with no cold hold.
        let mut frame_policy = FramePolicy::new(1, 1, 4096, None);
        frame_policy.on_fire_enqueued(stamp, Some(pid), fire_id, 1, 1);

        let mut pending: PendingQueue =
            VecDeque::from([QueuedItem::Launch(QueuedLaunch::new(Box::new(request)))]).into();
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());

        let queued: frame::QueuedFireIds = [fire_id].into_iter().collect();
        let FramePlan::Dispatch(waves) =
            frame_policy.plan_dispatch(&queued, &HashSet::new(), false, Instant::now())
        else {
            panic!("the single-fire frame must seal");
        };
        assert_eq!(waves, vec![vec![fire_id]]);
        completion.request_cancel();

        let (progress, posted) = BatchScheduler::post_frame(
            &mut SlotBuffer::new(),
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            16,
            limits,
            &stats,
            &waves,
        );
        assert!(progress, "the drop is progress");
        assert!(!posted, "nothing launches for a cancelled fire");
        assert!(pending.is_empty());
        assert!(completion.is_settled(), "the cancelled fire must reject");
        assert_eq!(
            frame_policy.plan_dispatch(
                &frame::QueuedFireIds::default(),
                &HashSet::new(),
                false,
                Instant::now()
            ),
            FramePlan::Park,
            "the frame resolved without the fire; the lane stays awaited"
        );
    }

    /// §12 regression, sweep half: a suspend/restore copy dispatches even
    /// when the queue front cannot move — front is a stamped fire whose
    /// frame is still gathering, behind it a ResizePool (not a valid
    /// rotate target), and only then the copy. The front-only scan starved
    /// the copy forever; the captured production wedge froze exactly here
    /// for ~70 s (sealed=1, in-flight empty, 8 copies queued).
    #[test]
    fn standalone_copy_dispatches_out_of_band_past_an_immovable_front() {
        let pid = ProcessId::new_v4();
        let stamp = FrameStamp {
            lane: pid,
            seq: 1,
            slot: 0,
            fires: 2,
        };
        let request = PendingRequest::direct(
            dummy_launch(),
            7,
            WorkItemCompletion::deferred_with_guard(None),
            0,
            Some(pid),
            Some(pid),
            false,
            None,
            None,
            Some(stamp),
            false,
                /*hook_program=*/false,
                /*lora_program=*/false);
        let fire_id = request.logical_fire_id;
        // fires=2 with one arrival: the frame is still gathering, so the
        // front launch is immovable.
        let mut frame_policy = FramePolicy::new(1, 2, 4096, None);
        frame_policy.on_fire_enqueued(stamp, Some(pid), fire_id, 1, 1);

        let (resize_tx, _resize_rx) = tokio::sync::oneshot::channel();
        let mut pending: PendingQueue = VecDeque::from([
            QueuedItem::Launch(QueuedLaunch::new(Box::new(request))),
            QueuedItem::ResizePool {
                plan: PoolResizePlan::default(),
                response: resize_tx,
            },
            QueuedItem::CopyKvTracked {
                plan: crate::driver::KvCopyPlan::default(),
                completion: ControlCompletion::new(),
            },
        ])
        .into();
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = InFlightControls::default();
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());

        let (progress, _) = BatchScheduler::dispatch_ready_items(
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            &mut in_flight_control,
            16,
            limits,
            &stats,
            &mut frame_policy,
            &mut ScanCache::default(),
            &mut SlotBuffer::new(),
            false,
        );
        assert!(progress, "the copy dispatch is progress");
        assert!(
            in_flight_launches.is_empty(),
            "the gathering frame must not post"
        );
        assert_eq!(
            in_flight_control
                .iter()
                .next()
                .map(|control| control.operation),
            Some("tracked KV copy"),
            "the copy dispatches out-of-band past the launch and the resize"
        );
        assert_eq!(pending.len(), 2, "launch and resize keep their positions");
    }

    /// Standalone copies pipeline instead of queueing for one slot. Nothing
    /// queued orders against them, so the single control slot only ever made
    /// each restore wait out the ones ahead of it: measured at 512-way KV
    /// contention, up to 7 restores wanted the slot at once and each H2D
    /// copy billed 22.8 ms against ~3.3 ms of transfer, while the D2H side —
    /// which the planner already issues one at a time, so it never queued —
    /// ran 6.7x cheaper per page. An exclusive control still takes the whole
    /// set (the next test).
    #[test]
    fn concurrent_standalone_copies_all_dispatch_in_one_pass() {
        let (copy_tx, _copy_rx) = tokio::sync::oneshot::channel();
        let (resize_tx, _resize_rx) = tokio::sync::oneshot::channel();
        let mut pending: PendingQueue = VecDeque::from([
            QueuedItem::CopyKvTracked {
                plan: crate::driver::KvCopyPlan::default(),
                completion: ControlCompletion::new(),
            },
            QueuedItem::CopyKv {
                plan: crate::driver::KvCopyPlan::default(),
                response: copy_tx,
            },
            QueuedItem::CopyKvTracked {
                plan: crate::driver::KvCopyPlan::default(),
                completion: ControlCompletion::new(),
            },
            QueuedItem::ResizePool {
                plan: PoolResizePlan::default(),
                response: resize_tx,
            },
        ])
        .into();
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = InFlightControls::default();
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());
        let mut frame_policy = FramePolicy::new(1, 1, 4096, None);

        let (progress, _) = BatchScheduler::dispatch_ready_items(
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            &mut in_flight_control,
            16,
            limits,
            &stats,
            &mut frame_policy,
            &mut ScanCache::default(),
            &mut SlotBuffer::new(),
            false,
        );

        assert!(progress);
        assert_eq!(
            in_flight_control.settling.len(),
            3,
            "every queued standalone copy is in flight after one pass"
        );
        assert_eq!(lane_inflight, 3, "each copy was posted to the lane");
        assert_eq!(
            pending.len(),
            1,
            "the resize stays queued behind the settling copies"
        );
    }

    /// The exclusivity half: an exclusive control (a `PreLaunchCopy`, whose
    /// consumer fire is queued behind it, or a pool resize, whose pipe drain
    /// IS its ordering mechanism) keeps the original single-slot rule in
    /// both directions.
    #[test]
    fn an_exclusive_control_never_shares_the_in_flight_set() {
        let settling = |holds_launches: bool| {
            let mut controls = InFlightControls::default();
            controls.push(PendingControl {
                state: ControlSlotState::Posted { token: 1 },
                logical_completion: None,
                process_id: None,
                pipeline_id: None,
                tracked_completion: None,
                operation: "settling",
                holds_launches,
            });
            controls
        };
        let run = |mut in_flight_control: InFlightControls, item: QueuedItem| {
            let mut pending: PendingQueue = VecDeque::from([item]).into();
            let (lane, _lane_rx) = test_lane(None);
            let mut lane_inflight = 1u64;
            let mut lane_token = 1u64;
            let mut instances = HashMap::new();
            let mut in_flight_launches = VecDeque::new();
            let limits = SchedulerLimits {
                max_forward_requests: 64,
                max_forward_tokens: 64,
                max_page_refs: 64,
            };
            let stats = Arc::new(SchedulerStats::default());
            let mut frame_policy = FramePolicy::new(1, 1, 4096, None);
            BatchScheduler::dispatch_ready_items(
                &lane,
                &mut lane_inflight,
                &mut lane_token,
                &mut instances,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                16,
                limits,
                &stats,
                &mut frame_policy,
                &mut ScanCache::default(),
                &mut SlotBuffer::new(),
                false,
            );
            (pending.len(), in_flight_control.settling.len())
        };

        assert_eq!(
            run(
                settling(true),
                QueuedItem::CopyKvTracked {
                    plan: crate::driver::KvCopyPlan::default(),
                    completion: ControlCompletion::new(),
                }
            ),
            (1, 1),
            "a settling exclusive control admits no standalone copy"
        );

        let (resize_tx, _resize_rx) = tokio::sync::oneshot::channel();
        assert_eq!(
            run(
                settling(false),
                QueuedItem::ResizePool {
                    plan: PoolResizePlan::default(),
                    response: resize_tx,
                }
            ),
            (1, 1),
            "a resize waits for the settling copies to drain"
        );
    }

    /// A lifecycle control never enters the in-flight set, so a settling
    /// standalone copy must not delay one. Under churn the planner's
    /// suspend/restore copies are in flight nearly continuously, and gating
    /// binds on them made every bind wait out the strict-watchdog window: the
    /// process stayed in `staged`, which pinned the cohort-boundary window
    /// open (back when that still held the seal) and stalled the very traffic
    /// the copy was settling behind.
    #[test]
    fn a_bind_dispatches_past_a_settling_standalone_copy() {
        let mut pending: PendingQueue = VecDeque::from([QueuedItem::BindInstance {
            pipeline_id: Some(ProcessId::new_v4()),
            plan: InstanceBindingPlan {
                driver_id: 0,
                program_id: 0,
                requested_instance_id: 0,
                pacing_wait_id: 0,
                channel_ids: Vec::new(),
                seed_values: Vec::new(),
                geometry_class: pie_driver_abi::GeometryClass::Host,
            },
            response: tokio::sync::oneshot::channel().0,
        }])
        .into();
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = InFlightControls::default();
        in_flight_control.push(PendingControl {
            state: ControlSlotState::Posted { token: 1 },
            logical_completion: None,
            process_id: None,
            pipeline_id: None,
            tracked_completion: Some(ControlCompletion::new()),
            operation: "tracked KV copy",
            holds_launches: false,
        });
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());
        let mut frame_policy = FramePolicy::new(1, 1, 4096, None);

        let (progress, _) = BatchScheduler::dispatch_ready_items(
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            &mut in_flight_control,
            16,
            limits,
            &stats,
            &mut frame_policy,
            &mut ScanCache::default(),
            &mut SlotBuffer::new(),
            false,
        );

        assert!(progress, "the bind dispatched");
        assert!(
            pending.is_empty(),
            "the bind must not wait out a copy it shares nothing with"
        );
        assert_eq!(
            in_flight_control.settling.len(),
            1,
            "the bind enters no in-flight set, so the copy is still alone"
        );
    }

    /// §12 regression, barrier half: queued standalone copies and resizes
    /// contribute NOTHING to `blocked_lanes` — a sealed frame straddling
    /// them posts whole. Only a `PreLaunchCopy` blocks, and only its own
    /// lane.
    #[test]
    fn queued_standalone_copies_and_resizes_never_block_lanes() {
        let lane_a = ProcessId::new_v4();
        let lane_b = ProcessId::new_v4();
        let coupled = ProcessId::new_v4();
        let stamped = |lane: ProcessId, instance: u64| {
            PendingRequest::direct(
                dummy_launch(),
                instance,
                WorkItemCompletion::deferred_with_guard(None),
                0,
                Some(lane),
                Some(lane),
                false,
                None,
                None,
                Some(FrameStamp {
                    lane,
                    seq: 1,
                    slot: 0,
                    fires: 1,
                }),
                false,
                /*hook_program=*/false,
                /*lora_program=*/false)
        };
        let request_a = stamped(lane_a, 7);
        let request_b = stamped(lane_b, 8);
        let fire_a = request_a.logical_fire_id;
        let fire_b = request_b.logical_fire_id;
        let (resize_tx, _resize_rx) = tokio::sync::oneshot::channel();
        let (copy_tx, _copy_rx) = tokio::sync::oneshot::channel();
        let pending: PendingQueue = VecDeque::from([
            QueuedItem::Launch(QueuedLaunch::new(Box::new(request_a))),
            QueuedItem::ResizePool {
                plan: PoolResizePlan::default(),
                response: resize_tx,
            },
            QueuedItem::CopyKvTracked {
                plan: crate::driver::KvCopyPlan::default(),
                completion: ControlCompletion::new(),
            },
            QueuedItem::CopyKv {
                plan: crate::driver::KvCopyPlan::default(),
                response: copy_tx,
            },
            QueuedItem::Launch(QueuedLaunch::new(Box::new(request_b))),
            QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(crate::driver::KvCopyPlan::default()),
                logical_completion: WorkItemCompletion::deferred_with_guard(None),
                process_id: Some(coupled),
                pipeline_id: Some(coupled),
            },
        ])
        .into();

        let mut scan_cache = ScanCache::default();
        let scan = BatchScheduler::scan_queue(&mut scan_cache, &pending, false);
        assert_eq!(
            scan.queued_ids,
            [fire_a, fire_b]
                .into_iter()
                .collect::<frame::QueuedFireIds>()
        );
        assert_eq!(
            scan.blocked_lanes,
            [coupled].into_iter().collect::<HashSet<ProcessId>>(),
            "only the pre-launch copy's lane blocks; the fire behind the \
             resize/copy run stays dispatchable (the deadlock's broken edge)"
        );
        assert_eq!(
            scan.drain_eligible,
            Vec::<u64>::new(),
            "the steady-state scan never builds the drain list"
        );
        assert_eq!(scan.untracked, None);

        // `stopping` is part of the cache key, so flipping it must re-scan
        // even though the queue itself never moved.
        let draining = BatchScheduler::scan_queue(&mut scan_cache, &pending, true);
        assert_eq!(draining.drain_eligible, vec![fire_a, fire_b]);
    }

    /// The cached scan is keyed on a queue epoch that every `&mut` reach
    /// bumps. A rotation is the case a length or endpoint fingerprint would
    /// miss: same length, same id set, different answer for `untracked`.
    #[test]
    fn a_mutated_queue_invalidates_the_cached_scan() {
        let lane = ProcessId::new_v4();
        let make = |frame: Option<FrameStamp>| {
            PendingRequest::direct(
                dummy_launch(),
                1,
                WorkItemCompletion::deferred_with_guard(None),
                0,
                Some(lane),
                Some(lane),
                false,
                None,
                None,
                frame,
                false,
                /*hook_program=*/false,
                /*lora_program=*/false)
        };
        let stamped = make(Some(FrameStamp {
            lane,
            seq: 1,
            slot: 0,
            fires: 1,
        }));
        let rider = make(None);
        let (stamped_id, rider_id) = (stamped.logical_fire_id, rider.logical_fire_id);
        let mut pending: PendingQueue = VecDeque::from([
            QueuedItem::Launch(QueuedLaunch::new(Box::new(stamped))),
            QueuedItem::Launch(QueuedLaunch::new(Box::new(rider))),
        ])
        .into();

        let mut cache = ScanCache::default();
        let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert_eq!(scan.untracked, Some(rider_id));
        assert!(scan.queued_ids.contains(&stamped_id));

        // A repeat scan at an unchanged epoch is the whole point: it must be
        // the cached one, and it must still be right.
        let hit = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert_eq!(hit.untracked, Some(rider_id));

        let before = pending.epoch();
        let front = pending.pop_front().expect("stamped front");
        pending.push_back(front);
        assert_ne!(before, pending.epoch(), "a rotation bumps the epoch");
        assert_eq!(pending.len(), 2, "a rotation keeps the length");

        let rescan = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert_eq!(
            rescan.untracked,
            Some(rider_id),
            "the rider is still the oldest unstamped fire"
        );
        assert!(rescan.queued_ids.contains(&stamped_id));

        // Dropping the stamped fire (now at the back, after the rotation)
        // must drop it from the cached id set.
        let _ = pending.pop_back();
        let after = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert!(
            !after.queued_ids.contains(&stamped_id),
            "a scan cached at an older epoch must never be reused"
        );
    }

    /// A settling standalone copy does not hold frame posting; a settling
    /// pre-launch copy or resize still does. Observable without a bound
    /// instance: when frame work RUNS, the sealed fire is extracted from
    /// the queue (and rejected as unknown-instance); when held, it stays
    /// queued.
    #[test]
    fn a_settling_standalone_copy_does_not_hold_frame_posting() {
        let run = |holds_launches: bool| {
            let pid = ProcessId::new_v4();
            let stamp = FrameStamp {
                lane: pid,
                seq: 1,
                slot: 0,
                fires: 1,
            };
            let request = PendingRequest::direct(
                dummy_launch(),
                7,
                WorkItemCompletion::deferred_with_guard(None),
                0,
                Some(pid),
                Some(pid),
                false,
                None,
                None,
                Some(stamp),
                false,
                /*hook_program=*/false,
                /*lora_program=*/false);
            let fire_id = request.logical_fire_id;
            let mut frame_policy = FramePolicy::new(1, 1, 4096, None);
            frame_policy.on_fire_enqueued(stamp, Some(pid), fire_id, 1, 1);
            let mut pending: PendingQueue =
                VecDeque::from([QueuedItem::Launch(QueuedLaunch::new(Box::new(request)))]).into();
            let (lane, _lane_rx) = test_lane(None);
            let mut lane_inflight = 0u64;
            let mut lane_token = 1u64;
            let mut instances = HashMap::new();
            let mut in_flight_launches = VecDeque::new();
            let mut in_flight_control = InFlightControls::default();
            in_flight_control.push(PendingControl {
                state: ControlSlotState::Posted { token: 1 },
                logical_completion: None,
                process_id: None,
                pipeline_id: None,
                tracked_completion: None,
                operation: "tracked KV copy",
                holds_launches,
            });
            let limits = SchedulerLimits {
                max_forward_requests: 64,
                max_forward_tokens: 64,
                max_page_refs: 64,
            };
            let stats = Arc::new(SchedulerStats::default());
            BatchScheduler::dispatch_ready_items(
                &lane,
                &mut lane_inflight,
                &mut lane_token,
                &mut instances,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                16,
                limits,
                &stats,
                &mut frame_policy,
                &mut ScanCache::default(),
                &mut SlotBuffer::new(),
                false,
            );
            pending.len()
        };
        assert_eq!(
            run(false),
            0,
            "frame work proceeds past a settling standalone copy"
        );
        assert_eq!(
            run(true),
            1,
            "a settling pre-launch copy or resize still holds launches"
        );
    }
}
