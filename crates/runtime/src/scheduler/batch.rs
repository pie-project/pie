use std::collections::HashSet;

use crate::engine::completion::CellPtr;
use crate::engine::{FrameFire, SchedulerLimits, StepFire};
use crate::scheduler::ProcessId;

use super::fire_plan;
use super::stats::SchedulerStats;
use super::worker::PendingRequest;

pub(crate) struct StepBuild {
    pub(crate) lanes: Vec<::engine::Lane>,
    pub(crate) media: Vec<engine::fire::StepMedia>,
    pub(crate) voxels: Vec<engine::fire::StepVoxels>,
    pub(crate) instance_ids: Vec<::engine::InstanceId>,
    pub(crate) boundary_programs: Vec<bool>,
    pub(crate) member_lane_indptr: Vec<u32>,
    pub(crate) terminal_cells: Vec<CellPtr>,
    pub(crate) completions: Vec<crate::engine::WorkItemCompletion>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_requests: usize,
    pub(crate) forward_tokens: usize,
    /// The recurrent rows the request carries unfolded into its fire: a
    /// step carries no more than its rung's tokens of them, as of its own
    /// rows ([`::engine::fire::rung_holds`]).
    pub(crate) carried_rows: usize,
    /// The buffered recurrent rows its lanes replay, a decode lane's
    /// included ([`crate::engine::FireRequest::replayed`]).
    pub(crate) replayed_rows: usize,
    pub(crate) page_refs: usize,
    /// The rows the request's sequence lanes read out
    /// ([`crate::engine::FireRequest::sequence_readouts`]): a step reads out
    /// no more than the engine's `max_readouts` of them, a fire bound. A
    /// float lane's rows land through the engine's readout landing and are
    /// not counted.
    pub(crate) readouts: usize,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let forward_requests = req.wire_row_count().max(1);
    let _ = page_size;

    RequestCapacityUsage {
        forward_requests,
        forward_tokens: req.request.tokens(),
        carried_rows: req.request.carried as usize,
        replayed_rows: req.request.replayed as usize,
        page_refs: req.request.pages().count(),
        readouts: req.request.sequence_readouts(),
    }
}

pub(crate) struct AdmissionLimits {
    page_size: u32,
    limits: SchedulerLimits,
}

impl AdmissionLimits {
    pub(crate) fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self { page_size, limits }
    }

    pub(crate) fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
        if usage.forward_requests > self.limits.max_forward_requests {
            return Some(format!(
                "forward request has {} resolved rows, exceeding engine limit {}",
                usage.forward_requests, self.limits.max_forward_requests
            ));
        }
        if usage.forward_tokens > self.limits.max_forward_tokens {
            return Some(format!(
                "forward request has {} forward tokens, exceeding engine limit {}",
                usage.forward_tokens, self.limits.max_forward_tokens
            ));
        }
        // Sequence lanes reading out more than the readout store holds: no
        // fire, however narrow, seats them. Their bind refuses them first
        // (`SharedPool::readouts_refused`); this names a request that came
        // past it. A float lane's rows (an encoder's, an embedding's every
        // row) are not counted: they land through the engine's readout
        // landing whatever the store holds.
        if self.limits.readout_store > 0 && usage.readouts > self.limits.readout_store {
            return Some(format!(
                "forward request of instance {} reads out {} rows, more than the {} rows \
                 the engine's readout store holds for a fire",
                req.instance_id.0, usage.readouts, self.limits.readout_store
            ));
        }
        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding engine limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }
        // A request whose lanes alone make a step outside the set the load
        // stated. A request here is one piece of its pass, already cut to a
        // fire (`pipeline::fire::pass_pieces`): a long prompt arrives as
        // pieces, each inside the set. What is refused is a piece the cut
        // could not bring inside it, a step no bound above names (its
        // sequence readouts past the rows one fire reads out, a buffered
        // step on a load with no rs list, a lane attachment the load does
        // not serve); the gap names which.
        if let Some(shapes) = self.limits.shapes
            && let Err(gap) = shapes.holds(&composition(req))
        {
            return Some(format!(
                "forward request of instance {} is a step no cut makes smaller and no stated \
                 shape holds: {gap}",
                req.instance_id.0
            ));
        }
        None
    }
}

/// The step composition `request`'s lanes make ([`::engine::StepComposition`]).
fn composition(request: &PendingRequest) -> ::engine::StepComposition {
    ::engine::StepComposition::of(request.request.lanes.iter().map(|lane| &lane.lane))
}

/// The stated buffered rung `request`'s step pads up to where its lanes
/// buffer ([`crate::engine::FireRequest::pads_to`]): the rung of the stated
/// shape holding it, buffered steps running at the load's buffered rungs
/// ([`::engine::StepFacts::buffered_rungs`]). 0 where its lanes fold or the
/// limits state no set.
pub(crate) fn pads_to(request: &PendingRequest, shapes: Option<&::engine::StepShapes>) -> u32 {
    let step = composition(request);
    if step.treatment != ::engine::Treatment::Buffers {
        return 0;
    }
    shapes
        .and_then(|shapes| shapes.shape_of(&step).map(|at| shapes.shapes[at].rung))
        .unwrap_or(0)
}

/// The set a step is checked against under the contract check: the load's
/// stated one; in a test building its limits by hand, the one the runtime's
/// rules form over those limits, every rung one fire wide, buffering.
fn checked_shapes(limits: &SchedulerLimits) -> Option<std::borrow::Cow<'static, ::engine::StepShapes>> {
    if let Some(shapes) = limits.shapes {
        return Some(std::borrow::Cow::Borrowed(shapes));
    }
    #[cfg(test)]
    {
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let facts = ::engine::StepFacts {
            rungs: Vec::new(),
            max_lanes: count(limits.max_forward_requests),
            max_tokens: count(limits.max_forward_tokens),
            max_readouts: count(limits.max_readouts),
            buffers: true,
            attachments: ::engine::Attachments::served(true, true, true),
            mixed: ::engine::MixedLanes::Any,
        };
        Some(std::borrow::Cow::Owned(::engine::StepShapes::of(
            &::engine::StepRules::runtime(),
            &facts,
        )))
    }
    #[cfg(not(test))]
    None
}

/// The contract check on a formed step ([`::engine::shapes::check_on`]):
/// a composition outside the stated set is a contract bug, named with the
/// composition and the nearest stated shapes.
fn check_step(lanes: &[::engine::Lane], limits: &SchedulerLimits) {
    if !::engine::shapes::check_on() {
        return;
    }
    let Some(shapes) = checked_shapes(limits) else {
        return;
    };
    if let Err(gap) = shapes.holds(&::engine::StepComposition::of(lanes)) {
        panic!("{gap}");
    }
}

/// The stated shape a step of `group` is of, as the fire carries it
/// (CONTRACT-BC 24): what [`::engine::StepShapes::shape_of`] answers for the
/// composition its members make, the one `StepGroup::accepts` held to the
/// set. `Ok(None)` where the limits state no set. A step no stated shape
/// holds is a runtime bug (the builder forms none): its gap, named.
fn stated_shape(
    group: &[Box<PendingRequest>],
    limits: &SchedulerLimits,
) -> Result<Option<::engine::FireShape>, ::engine::StepGap> {
    let Some(shapes) = limits.shapes else {
        return Ok(None);
    };
    let step = group
        .iter()
        .fold(::engine::StepComposition::default(), |step, request| {
            step.with(composition(request))
        });
    shapes.fire_shape(&step).map(Some)
}

/// The stated shape a step of `lanes` is of in `shapes`, stated again once
/// members are stripped from a formed step (`scheduler::lane`); `Ok(None)`
/// where no set is stated.
pub(crate) fn restated_shape(
    shapes: Option<&::engine::StepShapes>,
    lanes: &[::engine::Lane],
) -> Result<Option<::engine::FireShape>, ::engine::StepGap> {
    shapes.map_or(Ok(None), |shapes| {
        shapes.fire_shape(&::engine::StepComposition::of(lanes)).map(Some)
    })
}

/// The verify classes of `request`'s lanes ([`crate::model::Model::verify_class`]),
/// sorted: a step holds requests of one such set, the empty one for every
/// request that verifies nothing.
fn verifies(request: &PendingRequest) -> Vec<usize> {
    let Some(model) = crate::model::registered() else {
        return Vec::new();
    };
    let mut classes: Vec<usize> = request
        .request
        .lanes
        .iter()
        .filter_map(|lane| model.verify_class(lane.word))
        .collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

#[derive(Default)]
pub(crate) struct StepGroup {
    instances: HashSet<::engine::InstanceId>,
    verifies: Vec<usize>,
    pipelines: HashSet<ProcessId>,
    /// The ring ends its fires move: a second fire moving one of them goes
    /// a step later, so a ring's cursor moves once a step at each end.
    ring_ends: HashSet<RingEnd>,
    forward_requests: usize,
    forward_tokens: usize,
    carried_rows: usize,
    replayed_rows: usize,
    page_refs: usize,
    /// The rows its members read out: no more than the engine's
    /// `max_readouts`, its fixed readout store's rows under its ceiling
    /// ([`SchedulerLimits::read_out_by`]).
    readouts: usize,
    /// Lanes taken, and whether any decodes: a step of prompts only holds
    /// no more than `prefill_lanes`.
    lanes: usize,
    decoding: bool,
    /// Every member's rs scratch compact, and the widest rung any asked
    /// ([`crate::engine::FireRequest::replay_rung`]): the step's members
    /// together fit that rung's replay scratch.
    incompact: bool,
    replay_rungs: Option<crate::store::shared::Replays>,
    /// The composition its members make ([`::engine::StepComposition`]): a
    /// member joins only where the step stays inside the load's stated set.
    composition: ::engine::StepComposition,
}

/// Whether `request` has a lane decoding (a row of its own, or none the
/// host states), rather than prompts alone.
fn decodes(request: &PendingRequest) -> bool {
    request
        .request
        .lanes
        .iter()
        .any(|lane| lane.tokens.len() <= 1)
}

/// One end of a ring: where its fires take (the head) or put (the tail).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RingEnd {
    channel: ::engine::ChannelId,
    tail: bool,
}

/// The ring ends `request` moves.
fn ring_ends(request: &PendingRequest) -> impl Iterator<Item = RingEnd> + '_ {
    request.request.ring_moves.iter().flat_map(|ring| {
        [(false, ring.takes), (true, ring.puts)]
            .into_iter()
            .filter(|&(_, moved)| moved)
            .map(|(tail, _)| RingEnd {
                channel: ring.channel,
                tail,
            })
    })
}

impl StepGroup {
    pub(crate) fn accepts(
        &self,
        request: &PendingRequest,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> bool {
        if self.instances.contains(&request.instance_id) {
            return false;
        }
        if request
            .pipeline_id
            .is_some_and(|pid| self.pipelines.contains(&pid))
        {
            return false;
        }
        if ring_ends(request).any(|end| self.ring_ends.contains(&end)) {
            return false;
        }
        if self.forward_requests == 0 {
            return true;
        }
        if verifies(request) != self.verifies {
            return false;
        }
        let usage = request_capacity_usage(request, page_size);
        // Prompts only, past what the engine carries a body for: refused
        // until a decoding lane joins.
        let lanes = self.lanes.saturating_add(request.request.lanes.len());
        if limits.prefill_lanes > 0
            && !self.decoding
            && !decodes(request)
            && lanes > limits.prefill_lanes
        {
            return false;
        }
        // Its members' replay together fits the rung their rows together
        // pick (the step's own rung, by width as ever): a fire whose rs
        // scratch that rung would not hold goes in a later step.
        if self.forward_requests > 0 && !self.replay_holds(request) {
            return false;
        }
        // The step it would make is one the load stated: else it goes in a
        // later step, and this one is formed without it.
        // A buffered step pads up to its stated buffered rung: it stays
        // within the activation rung the step may take, else the member goes
        // in a later step (each member's own pad it asked at admission,
        // `FireRequest::pads_to`).
        if let Some(shapes) = limits.shapes {
            match shapes.shape_of(&self.composition.with(composition(request))) {
                None => return false,
                Some(at)
                    if shapes.shapes[at].treatment == ::engine::Treatment::Buffers
                        && shapes.shapes[at].rung as usize > limits.max_forward_tokens =>
                {
                    return false;
                }
                Some(_) => {}
            }
        }
        self.forward_requests.saturating_add(usage.forward_requests) <= limits.max_forward_requests
            && self.forward_tokens.saturating_add(usage.forward_tokens) <= limits.max_forward_tokens
            && self.carried_rows.saturating_add(usage.carried_rows) <= limits.max_forward_tokens
            // What the engine's replay table holds a step: a fire that
            // would carry more goes in a step after (never alone: a step's
            // first request is admitted whatever it replays).
            && (limits.max_replayed_rows == 0
                || self.forward_requests == 0
                || self.replayed_rows.saturating_add(usage.replayed_rows)
                    <= limits.max_replayed_rows)
            && self.page_refs.saturating_add(usage.page_refs) <= limits.max_page_refs
            && (limits.max_readouts == 0
                || self.readouts.saturating_add(usage.readouts) <= limits.max_readouts)
    }

    /// Whether a step holding `request` too keeps its members' replay
    /// within the rung their rows together pick ([`::engine::fire::rung_holds`]'s
    /// carried and scratch bounds; its own rows are what picked it). A step
    /// none of whose members carries recurrent state holds any.
    fn replay_holds(&self, request: &PendingRequest) -> bool {
        let Some(rungs) = self
            .replay_rungs
            .as_ref()
            .or(request.request.replay_rungs.as_ref())
        else {
            return true;
        };
        let usage = request_capacity_usage(request, 0);
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let own = count(self.forward_tokens.saturating_add(usage.forward_tokens));
        let Some((_, rung)) = rungs.iter().find(|&&(tokens, _)| tokens >= own) else {
            // Past every rung: the step's other bounds hold it.
            return true;
        };
        let rows = ::engine::ReplayRows::of_sums(
            own,
            count(self.carried_rows.saturating_add(usage.carried_rows)),
            count(self.replayed_rows.saturating_add(usage.replayed_rows)),
            count(self.lanes.saturating_add(request.request.lanes.len())),
            !self.incompact && request.request.scratch_compact,
        );
        !matches!(
            ::engine::fire::rung_holds(rung, rows),
            Err(past) if past.bound != ::engine::ReplayBound::Own
        )
    }

    pub(crate) fn push(
        &mut self,
        request: &PendingRequest,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> bool {
        let usage = request_capacity_usage(request, page_size);
        if self.forward_requests == 0 {
            self.verifies = verifies(request);
        }
        self.instances.insert(request.instance_id);
        if let Some(pid) = request.pipeline_id {
            self.pipelines.insert(pid);
        }
        self.ring_ends.extend(ring_ends(request));
        self.forward_requests = self.forward_requests.saturating_add(usage.forward_requests);
        self.forward_tokens = self.forward_tokens.saturating_add(usage.forward_tokens);
        self.carried_rows = self.carried_rows.saturating_add(usage.carried_rows);
        self.replayed_rows = self.replayed_rows.saturating_add(usage.replayed_rows);
        self.page_refs = self.page_refs.saturating_add(usage.page_refs);
        self.readouts = self.readouts.saturating_add(usage.readouts);
        if self.replay_rungs.is_none() {
            self.replay_rungs.clone_from(&request.request.replay_rungs);
        }
        if request.request.replay_rungs.is_some() {
            self.incompact |= !request.request.scratch_compact;
        }
        self.lanes = self.lanes.saturating_add(request.request.lanes.len());
        self.composition = self.composition.with(composition(request));
        self.decoding |= decodes(request);
        (limits.prefill_lanes > 0 && !self.decoding && self.lanes >= limits.prefill_lanes)
            || self.forward_requests >= limits.max_forward_requests
            || self.forward_tokens >= limits.max_forward_tokens
            || self.carried_rows >= limits.max_forward_tokens
            || (limits.max_replayed_rows > 0 && self.replayed_rows >= limits.max_replayed_rows)
            || self.page_refs >= limits.max_page_refs
            || (limits.max_readouts > 0 && self.readouts >= limits.max_readouts)
    }
}

/// The step `requests` make, their lanes, media and voxels moved into it:
/// a posted request is kept for its completion and its counts alone, so a
/// lane's slot tables are neither copied for the fire nor freed as the frame
/// retires, both on the path from one decode step to the next.
pub(crate) fn build_batch_request(
    requests: &mut [Box<PendingRequest>],
    page_size: u32,
    stats: &SchedulerStats,
) -> StepBuild {
    let _ = page_size;
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut lanes = Vec::with_capacity(requests.len());
        let mut instance_ids = Vec::with_capacity(requests.len());
        let mut boundary_programs = Vec::with_capacity(requests.len());
        let mut terminal_cells = Vec::with_capacity(requests.len());
        let mut completions = Vec::with_capacity(requests.len());
        let mut member_lane_indptr = Vec::with_capacity(requests.len() + 1);
        member_lane_indptr.push(0);
        let mut media: Vec<engine::fire::StepMedia> = Vec::new();
        let mut voxels: Vec<engine::fire::StepVoxels> = Vec::new();
        // Group ids renumber past earlier members', so two passes' groups never merge.
        let mut groups = 0u32;
        for req in requests.iter_mut() {
            let fires_after = req
                .frame
                .map(|stamp| stamp.fires.saturating_sub(stamp.slot + 1));
            boundary_programs.push(req.request.boundary_program);
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            completions.push(req.completion.clone());
            let base = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
            let mut own: Vec<u32> = Vec::new();
            // Each lane moves into the step; its page ids stay with the
            // request, which a fire refused for placement goes again by
            // once its lanes come back from the step (`restore_lanes`).
            lanes.extend(
                req.request
                    .lanes
                    .iter_mut()
                    .map(|fire_lane| {
                        let mut lane = std::mem::take(&mut fire_lane.lane);
                        lane.fires_after = fires_after;
                        lane.group = lane.group.map(|id| {
                            let local =
                                own.iter().position(|&seen| seen == id).unwrap_or_else(|| {
                                    own.push(id);
                                    own.len() - 1
                                });
                            groups + u32::try_from(local).unwrap_or(u32::MAX)
                        });
                        lane
                    }),
            );
            groups = groups.saturating_add(u32::try_from(own.len()).unwrap_or(u32::MAX));
            member_lane_indptr.push(u32::try_from(lanes.len()).unwrap_or(u32::MAX));
            media.extend(
                std::mem::take(&mut req.request.media)
                    .into_iter()
                    .map(|mut row| {
                        row.lane = row.lane.saturating_add(base);
                        row
                    }),
            );
            voxels.extend(
                std::mem::take(&mut req.request.voxels)
                    .into_iter()
                    .map(|mut row| {
                        row.lane = row.lane.saturating_add(base);
                        row
                    }),
            );
        }
        StepBuild {
            lanes,
            media,
            voxels,
            instance_ids,
            boundary_programs,
            member_lane_indptr,
            terminal_cells,
            completions,
        }
    })
}

/// Puts back into `requests` what the build moved out of them into
/// `steps` (a frame the engine refused for where state lies before any of
/// its steps ran): each fire's lanes into the page shells it kept, and its
/// media and voxels, found by its own completion. Says, per request,
/// whether it came back: a fire a refused lane took out of the frame
/// (failed alone) is in no step, and goes nowhere again.
pub(crate) fn restore_lanes(requests: &mut [Box<PendingRequest>], steps: Vec<StepFire>) -> Vec<bool> {
    let mut restored = vec![false; requests.len()];
    for step in steps {
        let ::engine::Step {
            lanes,
            media,
            voxels,
            ..
        } = step.submission;
        let members = step.member_lanes.len().saturating_sub(1);
        let member_of = |lane: u32| {
            step.member_lanes[..members]
                .partition_point(|&start| start <= lane)
                .saturating_sub(1)
        };
        let mut media_of: Vec<Vec<engine::fire::StepMedia>> = vec![Vec::new(); members];
        for row in media {
            if members > 0 {
                media_of[member_of(row.lane)].push(row);
            }
        }
        let mut voxels_of: Vec<Vec<engine::fire::StepVoxels>> = vec![Vec::new(); members];
        for row in voxels {
            if members > 0 {
                voxels_of[member_of(row.lane)].push(row);
            }
        }
        let mut lanes = lanes.into_iter();
        for member in 0..members {
            let span = (step.member_lanes[member + 1] - step.member_lanes[member]) as usize;
            let own: Vec<::engine::Lane> = lanes.by_ref().take(span).collect();
            let Some(completion) = step.completions.get(member) else {
                continue;
            };
            let Some(at) = (0..requests.len())
                .find(|&at| !restored[at] && requests[at].completion.same_request(completion))
            else {
                continue;
            };
            let request = &mut requests[at].request;
            if request.lanes.len() != own.len() {
                debug_assert!(
                    false,
                    "a fire's lanes come back {} for the {} it was built with",
                    own.len(),
                    request.lanes.len()
                );
                continue;
            }
            for (shell, lane) in request.lanes.iter_mut().zip(own) {
                shell.lane = lane;
            }
            let base = step.member_lanes[member];
            request.media = std::mem::take(&mut media_of[member])
                .into_iter()
                .map(|mut row| {
                    row.lane -= base;
                    row
                })
                .collect();
            request.voxels = std::mem::take(&mut voxels_of[member])
                .into_iter()
                .map(|mut row| {
                    row.lane -= base;
                    row
                })
                .collect();
            restored[at] = true;
        }
    }
    restored
}

#[allow(
    clippy::vec_box,
    reason = "measured: `PendingRequest` is 1408 bytes. This vec is not a store but a \
              conveyor — requests are moved wave -> step_groups -> deferred -> back \
              out repeatedly in this function — and the box makes each of those moves \
              8 bytes instead of 1408. Unboxing would trade one allocation per request \
              for a 1408-byte memcpy on every shuffle and every Vec regrow"
)]
pub(crate) fn build_frame_submission(
    waves: Vec<Vec<Box<PendingRequest>>>,
    limits: SchedulerLimits,
    page_size: u32,
    stats: &SchedulerStats,
) -> (FrameFire, Vec<Box<PendingRequest>>) {
    let mut step_groups: Vec<Vec<Box<PendingRequest>>> = Vec::new();
    for wave in waves {
        if wave.is_empty() {
            continue;
        }
        let mut deferred = wave;
        while !deferred.is_empty() {
            let mut grouping = StepGroup::default();
            let mut group: Vec<Box<PendingRequest>> = Vec::new();
            let mut rest: Vec<Box<PendingRequest>> = Vec::new();
            let mut closed = false;
            for req in deferred {
                if closed || !grouping.accepts(&req, limits, page_size) {
                    rest.push(req);
                    continue;
                }
                closed = grouping.push(&req, limits, page_size);
                group.push(req);
            }
            debug_assert!(!group.is_empty(), "grouping always admits the head");
            if group.is_empty() {
                group.push(rest.remove(0));
            }
            step_groups.push(group);
            deferred = rest;
        }
    }

    let mut steps: Vec<StepFire> = Vec::new();
    let mut flattened: Vec<Box<PendingRequest>> = Vec::new();

    for group in step_groups {
        // The step's stated shape, carried by its fire. A step outside the
        // set is a runtime bug: a debug build stops on it; a release build
        // refuses its members by name, never posting an unstated shape.
        let shape = match stated_shape(&group, &limits) {
            Ok(shape) => shape,
            Err(gap) => {
                debug_assert!(false, "the step builder formed {gap}");
                let reason = format!(
                    "scheduler: the step builder formed {gap} (a runtime bug): its fires fail, \
                     never posted"
                );
                tracing::error!("{reason}");
                for request in &group {
                    request.completion.reject_unsubmitted(reason.clone());
                }
                continue;
            }
        };
        let facts: Vec<fire_plan::MemberFacts> = group
            .iter()
            .enumerate()
            .map(|(arrival, req)| fire_plan::MemberFacts {
                custom_mask: req.request.has_user_mask,
                multi_token: req.request.lanes.iter().any(|lane| lane.tokens.len() > 1),
                geometry_class: req.request.geometry,
                arrival,
            })
            .collect();
        let mut slots: Vec<Option<Box<PendingRequest>>> = group.into_iter().map(Some).collect();
        let mut group: Vec<Box<PendingRequest>> = fire_plan::member_order(&facts)
            .iter()
            .map(|&index| slots[index].take().expect("member_order is a permutation"))
            .collect();

        // Counted before the build moves the lanes out.
        let traced = tracing::enabled!(tracing::Level::DEBUG).then(|| {
            let (mut tokens, mut page_refs) = (0usize, 0usize);
            for req in &group {
                let usage = request_capacity_usage(req, page_size);
                tokens += usage.forward_tokens;
                page_refs += usage.page_refs;
            }
            let lanes = || group.iter().flat_map(|req| req.request.lanes.iter());
            let prefill_lanes = lanes().filter(|lane| lane.tokens.len() > 1).count();
            let kv_len_max = lanes().map(|lane| lane.pages.len()).max().unwrap_or(0);
            (tokens, page_refs, prefill_lanes, kv_len_max)
        });
        // Each lane's kv set, before the build moves the lanes out.
        let lane_sets: Vec<_> = group
            .iter()
            .flat_map(|req| req.request.lanes.iter().map(|lane| lane.ws))
            .collect();
        let build = build_batch_request(&mut group, page_size, stats);
        if let Some((tokens, page_refs, prefill_lanes, kv_len_max)) = traced {
            tracing::debug!(
                "step lanes: members={} prefill_lanes={prefill_lanes} lanes={} tokens={tokens}/{} \
                 page_refs={page_refs}/{} kv_len_max={kv_len_max}",
                group.len(),
                build.lanes.len(),
                limits.max_forward_tokens,
                limits.max_page_refs,
            );
        }
        let mut instances = Vec::with_capacity(build.lanes.len());
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            let span = build.member_lane_indptr[member + 1] - build.member_lane_indptr[member];
            instances.extend(std::iter::repeat_n(instance, span as usize));
        }
        let mut attachments = Vec::new();
        let lanes = build.lanes;
        check_step(&lanes, &limits);
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            // Only the fire ending a pass runs its program; earlier fires carry the adapter alone.
            let adapter_only = !build.boundary_programs[member];
            if adapter_only && !group[member].lora_program {
                continue;
            }
            let lane = build.member_lane_indptr[member];
            let lanes = build.member_lane_indptr[member + 1] - lane;
            attachments.push(::engine::Attachment {
                lane,
                lanes,
                instance,
                adapter_only,
                scratch: ::engine::ScratchPlace::None,
            });
        }
        steps.push(StepFire {
            fire_stamps: group
                .iter()
                .map(|request| (request.logical_fire_id, request.queued))
                .collect(),
            lane_sets,
            submission: ::engine::Step {
                lanes,
                attachments,
                media: build.media,
                voxels: build.voxels,
                shape,
            },
            terminal_cells: build.terminal_cells,
            completions: build.completions,
            member_lanes: build.member_lane_indptr,
            instances,
            shapes: limits.shapes,
        });
        flattened.extend(group);
    }

    (FrameFire { steps }, flattened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FireRequest, WorkItemCompletion};
    use ::engine::InstanceId;
    use eta_ir::registry::GeometryClass;

    fn pending(request: FireRequest, instance_id: InstanceId) -> Box<PendingRequest> {
        Box::new(PendingRequest {
            queued: std::time::Instant::now(),
            lora_program: false,
            logical_fire_id: 1,
            request,
            instance_id,
            completion: WorkItemCompletion::new(instance_id.0, 0),
            process_id: None,
            pipeline_id: None,
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: None,
        })
    }

    fn decode(token: u32, page: u32) -> FireRequest {
        let mut request =
            FireRequest::one(crate::engine::fire::lane_of(0, vec![token], 0, vec![page]));
        request.single_token_mode = true;
        request
    }

    fn limits() -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
            max_context: 0,
            prefill_rows: 0,
            prefill_lanes: 0,
            max_readouts: 0,
            readout_store: 0,
            max_replayed_rows: 0,
            shapes: None,
        }
    }

    /// Qwen3.5-0.8B's rs scratch on Metal at rung 256: eight scratch slots of
    /// 45,376 bytes a row together, 1 MiB a lane, 291,667,968 bytes reserved.
    fn qwen3_5_rung_256() -> ::engine::RungReplay {
        ::engine::RungReplay {
            own_max: Some(256),
            carried_max: Some(256),
            scratch: ::engine::ScratchBytes {
                slot_bytes: vec![12288, 64, 4096, 12288, 128, 128, 8192, 8192],
                lane_bytes: 262_144 * 4,
                room_bytes: 291_667_968,
                pad: 256,
            },
        }
    }

    /// Sixteen processes' replay fires of ten decode lanes each, every lane
    /// replaying 37 rows, laid non-compact, each asking rung 256 alone (380
    /// scratch rows over 10 lanes fit it). Composed into one step of 160 lanes
    /// they would take 443,658,240 bytes, past the rung's 291,667,968: the
    /// step takes no more of them than the rung holds together, the rest going
    /// in later steps.
    #[test]
    fn a_step_takes_no_more_replay_than_its_rung_s_scratch_holds() {
        let rung = qwen3_5_rung_256();
        assert_eq!(rung.scratch.bytes(6080, 160), 443_658_240, "Mac's 160-lane fire");
        let fire = |instance: u64| {
            let mut request = FireRequest {
                lanes: (0..10)
                    .map(|lane| crate::engine::fire::lane_of(lane, vec![9], 0, vec![lane]))
                    .collect(),
                ..FireRequest::default()
            };
            request.replayed = 370;
            request.scratch_compact = false;
            request.replay_rungs = Some(std::sync::Arc::new(vec![(256, rung.clone())]));
            pending(request, InstanceId(instance))
        };
        let wave: Vec<Box<PendingRequest>> = (1..=16).map(fire).collect();
        let limits = SchedulerLimits {
            max_forward_requests: 1024,
            max_forward_tokens: 4096,
            max_page_refs: 4096,
            ..limits()
        };
        let (frame, _) = build_frame_submission(vec![wave], limits, 16, &SchedulerStats::default());
        assert!(frame.steps.len() > 1, "one step would pass the rung's scratch");
        for step in &frame.steps {
            let lanes = u32::try_from(step.submission.lanes.len()).unwrap();
            let rows = lanes + lanes * 37;
            assert!(
                rung.scratch.bytes(rows, lanes) <= rung.scratch.room_bytes,
                "a step of {lanes} lanes takes {} bytes",
                rung.scratch.bytes(rows, lanes)
            );
        }
        let lanes: usize = frame.steps.iter().map(|step| step.submission.lanes.len()).sum();
        assert_eq!(lanes, 160, "every lane served");
    }

    #[test]
    fn batch_every_case() {
        every_member_owns_a_span_of_the_steps_lanes();
        a_multi_lane_member_arrives_lane_for_lane();
        members_cobatch_by_concatenation();
        two_passes_groups_never_meet();
        a_masked_lane_carries_its_own_mask_and_a_peer_carries_none();
        two_seated_members_batch_into_a_fire_the_contract_accepts();
        a_device_geometry_member_seriates_into_the_suffix();
        a_frames_cells_are_one_per_member_and_never_shared();
        a_device_resolved_lane_counts_against_the_token_cap();
        a_step_carries_no_more_than_its_rungs_tokens();
    }

    /// A step's lanes share one rung's replay scratch, which holds the
    /// rung's tokens of carried rows: requests carrying more between them
    /// take steps of their own, and ones carrying none step as before.
    fn a_step_carries_no_more_than_its_rungs_tokens() {
        let carrying = |instance: u64, carried: u32| {
            let mut request = decode(9, 1);
            request.carried = carried;
            pending(request, InstanceId(instance))
        };
        let cap = u32::try_from(limits().max_forward_tokens).unwrap();
        let mut group = StepGroup::default();
        assert!(!group.push(&carrying(1, cap - 10), limits(), 16));
        assert!(
            group.accepts(&carrying(2, 0), limits(), 16),
            "carrying none"
        );
        assert!(
            group.accepts(&carrying(2, 10), limits(), 16),
            "up to the rung"
        );
        assert!(
            !group.accepts(&carrying(2, 11), limits(), 16),
            "past the rung's tokens of carried rows"
        );
        assert!(
            group.push(&carrying(2, 10), limits(), 16),
            "full at the rung's tokens carried"
        );
        // Carrying none, a step fills by its rows alone.
        let mut plain = StepGroup::default();
        for at in 1..=4 {
            assert!(plain.accepts(&carrying(at, 0), limits(), 16));
            assert!(!plain.push(&carrying(at, 0), limits(), 16));
        }
    }

    fn every_member_owns_a_span_of_the_steps_lanes() {
        let placeholder = FireRequest::default();
        let mut requests = vec![
            pending(decode(11, 3), InstanceId(1)),
            pending(placeholder, InstanceId(2)),
            pending(decode(22, 4), InstanceId(3)),
        ];
        let step = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(step.member_lane_indptr, vec![0, 1, 1, 2]);
        assert_eq!(
            step.instance_ids,
            vec![InstanceId(1), InstanceId(2), InstanceId(3)]
        );
        assert_eq!(step.lanes.len(), 2, "the placeholder contributes no lane");
        assert_eq!(step.lanes[0].tokens, vec![11]);
        assert_eq!(step.lanes[1].tokens, vec![22]);
    }

    fn a_multi_lane_member_arrives_lane_for_lane() {
        let mut two = FireRequest {
            lanes: vec![
                crate::engine::fire::lane_of(0, vec![11], 0, vec![3]),
                crate::engine::fire::lane_of(1, vec![22], 0, vec![4]),
            ],
            ..FireRequest::default()
        };
        two.lanes[1].rs_reset = engine::fire::RsReset::Fresh;
        let expected: Vec<::engine::Lane> =
            two.lanes.iter().map(|lane| lane.lane.clone()).collect();

        let step = build_batch_request(
            &mut [pending(two, InstanceId(1))],
            16,
            &SchedulerStats::default(),
        );
        assert_eq!(step.member_lane_indptr, vec![0, 2]);
        assert_eq!(step.lanes, expected);
    }

    fn members_cobatch_by_concatenation() {
        let two = FireRequest {
            lanes: vec![
                crate::engine::fire::lane_of(0, vec![11], 0, vec![3]),
                crate::engine::fire::lane_of(1, vec![22], 0, vec![4]),
            ],
            ..FireRequest::default()
        };
        let mut requests = [
            pending(two, InstanceId(9)),
            pending(decode(33, 5), InstanceId(10)),
        ];
        let step = build_batch_request(&mut requests, 16, &SchedulerStats::default());

        assert_eq!(step.member_lane_indptr, vec![0, 2, 3]);
        assert_eq!(
            step.lanes
                .iter()
                .map(|lane| lane.tokens.clone())
                .collect::<Vec<_>>(),
            vec![vec![11], vec![22], vec![33]]
        );
    }

    fn two_passes_groups_never_meet() {
        let grouped = |first: u32| {
            let mut pass = FireRequest {
                lanes: (0..4)
                    .map(|at| crate::engine::fire::lane_of(at + first, vec![1], 0, vec![at]))
                    .collect(),
                ..FireRequest::default()
            };
            for (at, lane) in pass.lanes.iter_mut().enumerate() {
                lane.group = Some(at as u32 / 2);
            }
            pass
        };
        let mut requests = [
            pending(grouped(0), InstanceId(1)),
            pending(grouped(4), InstanceId(2)),
        ];
        let step = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        let groups: Vec<Option<u32>> = step.lanes.iter().map(|lane| lane.group).collect();
        assert_eq!(
            groups,
            [0, 0, 1, 1, 2, 2, 3, 3].map(Some).to_vec(),
            "each pass keeps its own groups"
        );
    }

    fn a_masked_lane_carries_its_own_mask_and_a_peer_carries_none() {
        let mut masked = decode(11, 3);
        masked.has_user_mask = true;
        masked.single_token_mode = false;
        masked.lanes[0].mask = Some(::engine::Masking::Extent(::engine::Mask::new(
            vec![0, 1],
            1,
        )));

        let mut requests = [
            pending(masked, InstanceId(20)),
            pending(decode(22, 4), InstanceId(21)),
        ];
        let step = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(
            step.lanes[0].mask,
            Some(::engine::Masking::Extent(::engine::Mask::new(
                vec![0, 1],
                1
            )))
        );
        assert_eq!(step.lanes[1].mask, None, "an unmasked peer stays unmasked");
    }

    fn two_seated_members_batch_into_a_fire_the_contract_accepts() {
        let model = crate::store::registry::register_model(16, &[8], &[4]);
        let stores = crate::store::registry::get(model, 0);
        let (first_ws, second_ws) =
            crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
                (kv.create_working_set(), kv.create_working_set())
            });

        let mut first = decode(11, 3);
        let mut second = decode(22, 4);
        assert_eq!(
            (first.lanes[0].slot, second.lanes[0].slot),
            (0, 0),
            "both fires arrive at the seat stamp unseated — this is the defect's shape"
        );
        crate::pipeline::fire::stamp_lane_slots(&mut first, &stores, &vec![(first_ws, vec![0])])
            .expect("a two-slot pool seats one sequence");
        crate::pipeline::fire::stamp_lane_slots(&mut second, &stores, &vec![(second_ws, vec![0])])
            .expect("and its peer");

        let (frame, _) = build_frame_submission(
            vec![vec![
                pending(first, InstanceId(41)),
                pending(second, InstanceId(42)),
            ]],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        let step = &frame.steps[0];
        assert_eq!(step.submission.lanes.len(), 2, "both members co-batch");
        assert_ne!(
            step.submission.lanes[0].slot, step.submission.lanes[1].slot,
            "two concurrent sequences, two pool slots"
        );
        step.submission
            .validate()
            .expect("the contract accepts a fire whose lanes are seated apart");
    }

    fn a_device_geometry_member_seriates_into_the_suffix() {
        let mut pooled = decode(11, 3);
        pooled.geometry = GeometryClass::DeviceGeometry;
        pooled.lanes[0].pages.clear();

        let (frame, flattened) = build_frame_submission(
            vec![vec![
                pending(pooled, InstanceId(13)),
                pending(decode(22, 4), InstanceId(12)),
            ]],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        let step = &frame.steps[0];
        assert_eq!(step.submission.lanes.len(), 2);
        assert_eq!(
            flattened
                .iter()
                .map(|req| req.request.geometry)
                .collect::<Vec<_>>(),
            vec![GeometryClass::Host, GeometryClass::DeviceGeometry],
            "host first, device-resolved as the suffix run"
        );
        assert_eq!(step.instances, vec![InstanceId(12), InstanceId(13)]);
    }

    fn a_device_resolved_lane_counts_against_the_token_cap() {
        // A decode whose token the device resolves carries no tokens here,
        // and the engine still runs its row: with a prefill filling the cap
        // they cannot share a step.
        let mut pooled = decode(11, 3);
        pooled.geometry = GeometryClass::DeviceGeometry;
        pooled.lanes[0].tokens.clear();
        let prefill = FireRequest::one(crate::engine::fire::lane_of(
            1,
            vec![7; limits().max_forward_tokens],
            0,
            vec![4, 5, 6, 7],
        ));
        let (frame, _) = build_frame_submission(
            vec![vec![
                pending(pooled, InstanceId(1)),
                pending(prefill, InstanceId(2)),
            ]],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        assert_eq!(frame.steps.len(), 2, "the prefill takes a step of its own");
        for step in &frame.steps {
            let rows: usize = step
                .submission
                .lanes
                .iter()
                .map(|lane| lane.tokens.len().max(1))
                .sum();
            assert!(rows <= limits().max_forward_tokens, "{rows} rows in a step");
        }
    }

    fn a_frames_cells_are_one_per_member_and_never_shared() {
        let (frame, flattened) = build_frame_submission(
            vec![
                vec![pending(decode(11, 3), InstanceId(1))],
                vec![pending(decode(22, 4), InstanceId(2))],
            ],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        let cells: Vec<_> = frame.terminal_cells().collect();
        assert_eq!(cells.len(), flattened.len());
        assert_eq!(
            cells
                .iter()
                .map(|cell| std::sync::Arc::as_ptr(&cell.0))
                .collect::<std::collections::HashSet<_>>()
                .len(),
            cells.len(),
            "no cell is owned by two members"
        );
    }

    /// The set the runtime's rules form over `limits` at `rungs`, as a load
    /// states it.
    fn stated(limits: SchedulerLimits, rungs: &[u32], buffers: bool) -> ::engine::StepShapes {
        ::engine::StepShapes::of(&::engine::StepRules::runtime(), &facts(limits, rungs, buffers))
    }

    /// The facts a load of `rungs` under `limits` states.
    fn facts(limits: SchedulerLimits, rungs: &[u32], buffers: bool) -> ::engine::StepFacts {
        ::engine::StepFacts {
            rungs: rungs.to_vec(),
            max_lanes: limits.max_forward_requests as u32,
            max_tokens: limits.max_forward_tokens as u32,
            max_readouts: limits.max_readouts as u32,
            buffers,
            attachments: ::engine::Attachments::served(true, true, true),
            mixed: ::engine::MixedLanes::Any,
        }
    }

    fn holding(limits: SchedulerLimits, shapes: ::engine::StepShapes) -> SchedulerLimits {
        SchedulerLimits {
            shapes: Some(Box::leak(Box::new(shapes))),
            ..limits
        }
    }

    fn prompt(instance: u64, rows: u32) -> Box<PendingRequest> {
        let lane = crate::engine::fire::lane_of(0, (0..rows).collect(), 0, vec![1]);
        pending(FireRequest::one(lane), InstanceId(instance))
    }

    fn buffered_decode(instance: u64) -> Box<PendingRequest> {
        let mut request = decode(1, 1);
        request.lanes[0].lane.rs = ::engine::RsVerb::Buffer {
            pages: vec![1],
            base: 0,
            folded: 0,
            fold: ::engine::fire::FoldLen::Host(0),
            lineage: ::engine::fire::Lineage::Set,
        };
        pending(request, InstanceId(instance))
    }

    fn compositions(frame: &FrameFire) -> Vec<::engine::StepComposition> {
        frame
            .steps
            .iter()
            .map(|step| ::engine::StepComposition::of(&step.submission.lanes))
            .collect()
    }

    /// Waves of decodes (folded and buffered), prompts of every width and
    /// both together, on a load of rungs 4, 16 and 64: every step the
    /// builder forms is a composition of the set the load stated, and
    /// every lane is served. A buffered step pads to a stated buffered rung
    /// (16 and 64: the lattice's widest spacing is 4), never past its
    /// folded rung by more than that spacing, and each buffered member
    /// asks the rung it pads to.
    #[test]
    fn every_step_the_builder_forms_is_in_the_stated_set() {
        let base = limits();
        let facts = facts(base, &[4, 16, 64], true);
        let shapes = stated(base, &[4, 16, 64], true);
        let limits = holding(base, shapes.clone());
        let buffered_rungs = facts.buffered_rungs();
        assert_eq!(buffered_rungs, vec![16, 64]);
        let (upper, lower) = facts.spacing();
        let lattice = facts.lattice();
        let mut buffered_steps = 0;
        let mut instance = 0u64;
        let mut next = || {
            instance += 1;
            instance
        };
        for decodes in [0usize, 1, 3, 7, 12] {
            for prompts in [0u32, 2, 5, 30, 63] {
                for buffered in [false, true] {
                    let mut wave: Vec<Box<PendingRequest>> = (0..decodes)
                        .map(|_| {
                            if buffered {
                                buffered_decode(next())
                            } else {
                                pending(decode(1, 1), InstanceId(next()))
                            }
                        })
                        .collect();
                    if prompts > 1 {
                        wave.push(prompt(next(), prompts));
                        wave.push(prompt(next(), 2));
                    }
                    let lanes: usize = wave.iter().map(|r| r.request.lanes.len()).sum();
                    if lanes == 0 {
                        continue;
                    }
                    for member in &mut wave {
                        member.request.pads_to = pads_to(member, limits.shapes);
                        if composition(member).treatment == ::engine::Treatment::Buffers {
                            assert!(
                                buffered_rungs.contains(&member.request.pads_to),
                                "a buffered member asks its buffered rung"
                            );
                            assert!(member.request.rung_rows() >= member.request.pads_to as usize);
                        } else {
                            assert_eq!(member.request.pads_to, 0);
                        }
                    }
                    let (frame, flattened) =
                        build_frame_submission(vec![wave], limits, 16, &SchedulerStats::default());
                    assert_eq!(flattened.len(), frame.steps.iter().map(|s| s.fire_stamps.len()).sum::<usize>());
                    let served: usize = frame.steps.iter().map(|s| s.submission.lanes.len()).sum();
                    assert_eq!(served, lanes, "every lane is served");
                    for step in compositions(&frame) {
                        shapes.holds(&step).unwrap_or_else(|gap| panic!("{gap}"));
                    }
                    for step in &frame.steps {
                        let shape = step.submission.shape.expect("a formed step carries its shape");
                        shapes
                            .agrees(shape, &step.submission.lanes)
                            .unwrap_or_else(|why| panic!("{why}"));
                        if shape.treatment != ::engine::Treatment::Buffers {
                            continue;
                        }
                        buffered_steps += 1;
                        let rung = shapes.shapes[shape.id as usize].rung;
                        assert!(buffered_rungs.contains(&rung), "r{rung} is a buffered rung");
                        let floor = ::engine::StepComposition::of(&step.submission.lanes).floor();
                        let folded = *lattice.iter().find(|&&r| r >= floor).unwrap();
                        assert!(
                            u64::from(rung) * u64::from(lower)
                                <= u64::from(upper) * u64::from(folded),
                            "{floor} rows: r{rung} over folded r{folded} past {upper}/{lower}"
                        );
                        assert!(rung as usize <= limits.max_forward_tokens);
                    }
                }
            }
        }
        assert!(buffered_steps > 0, "buffered steps were formed");
    }

    /// A load stating no buffered step: a buffered decode is deferred from a
    /// folded step and, alone outside the set, refused at admission by name.
    #[test]
    fn a_lone_request_outside_the_stated_set_is_refused_by_name() {
        let base = limits();
        let limits = holding(base, stated(base, &[64], false));
        let admission = AdmissionLimits::new(limits, 16);
        assert_eq!(admission.single_request_limit_error(&pending(decode(1, 1), InstanceId(1))), None);
        let refused = admission
            .single_request_limit_error(&buffered_decode(7))
            .expect("a buffered step the load never stated is refused");
        assert!(
            refused.contains("instance 7")
                && refused.contains("outside the set")
                && refused.contains("nearest stated"),
            "{refused}"
        );
    }

    /// A prompt of 21,250 tokens, past ten times the widest rung (2048), on
    /// a load stating its set: its pass is cut into pieces of a fire each
    /// (`pass_pieces`), every piece is admitted, the builder forms each in a
    /// step of the set, beside decodes too, and every row is served once,
    /// the last piece reading out. A long prompt is never refused.
    #[test]
    fn a_prompt_past_every_rung_is_cut_and_served_never_refused() {
        let base = SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 2048,
            max_page_refs: 4096,
            max_context: 32_768,
            ..limits()
        };
        let rungs = [16, 64, 256, 1024, 2048];
        for buffers in [false, true] {
            let shapes = stated(base, &rungs, buffers);
            let limits = holding(base, shapes.clone());
            let admission = AdmissionLimits::new(limits, 16);
            let rows = 21_250u32;
            let mut lane = crate::engine::fire::lane_of(0, (0..rows).map(|t| t + 100).collect(), 0, Vec::new());
            lane.pages = (0..rows.div_ceil(16)).collect();
            lane.readout = ::engine::Readout::Last;
            let mut full = FireRequest::one(lane);
            full.boundary_program = true;
            let whole = pending(full.clone(), InstanceId(1));
            assert!(
                admission.single_request_limit_error(&whole).is_some(),
                "uncut, the pass is no fire"
            );
            let pieces = crate::pipeline::fire::pass_pieces(&full, true, true, 2048, 16, &[], &[]);
            assert!(pieces.len() >= 11, "cut into fires: {}", pieces.len());
            let mut served = 0u32;
            let mut read = 0usize;
            for (at, piece) in pieces.iter().enumerate() {
                let last = at + 1 == pieces.len();
                let request = crate::pipeline::fire::rows_of_lanes(&full, piece, last, 16);
                let request = pending(request, InstanceId(1));
                assert_eq!(
                    admission.single_request_limit_error(&request),
                    None,
                    "piece {at} is admitted"
                );
                // Each piece steps beside a few decodes, as a frame would.
                let mut wave = vec![request];
                wave.extend((10..13).map(|i| pending(decode(1, 1), InstanceId(i))));
                let (frame, flattened) =
                    build_frame_submission(vec![wave], limits, 16, &SchedulerStats::default());
                assert_eq!(flattened.len(), 4, "no member is dropped");
                for step in &frame.steps {
                    let composition = ::engine::StepComposition::of(&step.submission.lanes);
                    shapes.holds(&composition).unwrap_or_else(|gap| panic!("piece {at}: {gap}"));
                    for lane in step.submission.lanes.iter().filter(|lane| lane.tokens.len() > 1 || lane.tokens.first() != Some(&1)) {
                        served += lane.rows();
                        read += match &lane.readout {
                            ::engine::Readout::Last => 1,
                            ::engine::Readout::Rows(rows) => rows.len(),
                            ::engine::Readout::None => 0,
                        };
                    }
                }
            }
            assert_eq!(served, rows, "every row of the prompt served once");
            assert_eq!(read, 1, "the last piece reads out");
        }
    }

    /// What admission refuses against the set is a piece no cut makes
    /// smaller: here one reading out more rows than one fire reads out
    /// (prompt logprobs past the engine's readout rows a fire), named with
    /// the bound it passes.
    #[test]
    fn a_piece_reading_out_past_one_fire_is_refused_naming_its_readouts() {
        let base = SchedulerLimits {
            max_forward_tokens: 64,
            max_readouts: 16,
            ..limits()
        };
        let limits = holding(base, stated(base, &[64], false));
        let mut lane = crate::engine::fire::lane_of(0, (0..40).collect(), 0, vec![1]);
        lane.readout = ::engine::Readout::Rows((0..40).collect());
        let refused = AdmissionLimits::new(limits, 16)
            .single_request_limit_error(&pending(FireRequest::one(lane), InstanceId(9)))
            .expect("40 readout rows past 16 a fire");
        assert!(
            refused.contains("instance 9") && refused.contains("readout rows 40 (the shape's 0-16)"),
            "{refused}"
        );
    }
}
