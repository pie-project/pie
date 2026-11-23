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
    /// rows ([`::engine::fire::rung_rows`]).
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
        None
    }
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
        self.lanes = self.lanes.saturating_add(request.request.lanes.len());
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
        let traced = super::worker::wave_trace().then(|| {
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
            super::worker::wave_trace_emit(format!(
                "[step-lanes] t={}us members={} prefill_lanes={prefill_lanes} lanes={} tokens={tokens}/{} page_refs={page_refs}/{} kv_len_max={kv_len_max}",
                super::worker::wave_trace_us(),
                group.len(),
                build.lanes.len(),
                limits.max_forward_tokens,
                limits.max_page_refs,
            ));
        }
        let mut instances = Vec::with_capacity(build.lanes.len());
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            let span = build.member_lane_indptr[member + 1] - build.member_lane_indptr[member];
            instances.extend(std::iter::repeat_n(instance, span as usize));
        }
        let mut attachments = Vec::new();
        let lanes = build.lanes;
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
            },
            terminal_cells: build.terminal_cells,
            completions: build.completions,
            member_lanes: build.member_lane_indptr,
            instances,
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
        }
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
        a_step_of_prompts_only_stops_at_the_engines_prefill_lanes();
        fires_moving_one_ring_end_go_a_step_apart_in_the_order_they_stand();
        a_seal_that_dispatches_the_later_putter_first_gives_it_the_earlier_cell();
        a_step_carries_no_more_than_its_rungs_tokens();
        a_step_replays_no_more_rows_than_the_engine_states();
        a_wave_of_prompts_past_one_fire_is_cut_and_its_decode_wave_is_one_fire();
    }

    /// A verify lane of `instance`: a block of 16 tokens, each row read out.
    fn verifying(instance: u64) -> Box<PendingRequest> {
        let mut lane = crate::engine::fire::lane_of(0, (0..16).collect(), 0, vec![1]);
        lane.readout = ::engine::Readout::Rows((0..16).collect());
        pending(FireRequest::one(lane), InstanceId(instance))
    }

    /// The limits of an engine whose fires carry up to 128 lanes, its
    /// readout ceiling sixteen a lane, with its readouts as `readouts`
    /// states them.
    fn reading(readouts: &::engine::ReadoutFacts) -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 128,
            max_forward_tokens: 4096,
            max_page_refs: 4096,
            max_readouts: 128 * 16,
            ..limits()
        }
        .read_out_by(readouts)
    }

    /// The readouts of each step of the one wave `requests` make, and the
    /// lanes the steps carry in all.
    fn read_out(
        requests: Vec<Box<PendingRequest>>,
        limits: SchedulerLimits,
    ) -> (Vec<usize>, usize) {
        let (frame, _) =
            build_frame_submission(vec![requests], limits, 16, &SchedulerStats::default());
        let readouts = frame
            .steps
            .iter()
            .map(|step| {
                step.submission
                    .lanes
                    .iter()
                    .map(|lane| match &lane.readout {
                        ::engine::Readout::Last => 1,
                        ::engine::Readout::Rows(rows) => rows.len(),
                        ::engine::Readout::None => 0,
                    })
                    .sum()
            })
            .collect();
        let lanes = frame
            .steps
            .iter()
            .map(|step| step.submission.lanes.len())
            .sum();
        (readouts, lanes)
    }

    /// A drafter's verify, 80 lanes of 16 readouts each, on an engine whose
    /// fixed readout store holds 128 rows: cut into fires of no more than
    /// 128 readout rows, every lane served.
    #[test]
    fn a_verify_past_the_readout_store_is_cut_into_fires_it_holds() {
        let store = ::engine::ReadoutFacts {
            store_rows: Some(128),
            ..::engine::ReadoutFacts::default()
        };
        let limits = reading(&store);
        assert_eq!(limits.max_readouts, 128, "the store bounds a fire");
        let (readouts, lanes) = read_out((0..80).map(verifying).collect(), limits);
        assert!(
            readouts.iter().all(|&rows| rows <= 128),
            "every fire reads out no more than the store holds: {readouts:?}"
        );
        assert_eq!(readouts.len(), 10, "eight lanes a fire: {readouts:?}");
        assert_eq!(lanes, 80, "every lane is served");
        assert_eq!(readouts.iter().sum::<usize>(), 80 * 16);
    }

    /// An engine that states nothing of its readouts, or states them inside
    /// the span (or beside it, unbounded), leaves the fires as wide as they
    /// were: the 80 lanes one fire.
    #[test]
    fn readouts_inside_the_span_leave_the_fires_as_they_were() {
        let stated = [
            ::engine::ReadoutFacts::default(),
            ::engine::ReadoutFacts {
                landing: ::engine::ReadoutLanding::InSpan,
                store_rows: None,
            },
            ::engine::ReadoutFacts {
                landing: ::engine::ReadoutLanding::Beside { row_bytes: 4096 },
                store_rows: None,
            },
        ];
        for readouts in stated {
            let limits = reading(&readouts);
            assert_eq!(limits.max_readouts, 128 * 16, "{readouts:?}");
            let (steps, lanes) = read_out((0..80).map(verifying).collect(), limits);
            assert_eq!(steps, vec![80 * 16], "{readouts:?}: one fire, as before");
            assert_eq!(lanes, 80);
        }
        // With no ceiling stated either, none comes of the facts.
        let open = SchedulerLimits {
            max_readouts: 0,
            ..limits()
        }
        .read_out_by(&::engine::ReadoutFacts::default());
        assert_eq!(open.max_readouts, 0);
    }

    /// A float lane (an embedding's, a cacheless encoder's) of `rows`
    /// rows, every one read out.
    fn embedding(instance: u64, rows: u32) -> Box<PendingRequest> {
        let lane = ::engine::Lane {
            tokens: vec![0; rows as usize],
            readout: ::engine::Readout::Rows((0..rows).collect()),
            kv_less: true,
            ..::engine::Lane::default()
        };
        pending(FireRequest::one(lane), InstanceId(instance))
    }

    /// A float pass reads out every row it embeds through the engine's
    /// readout landing, not its fixed store: a 512-row embedding against a
    /// 128-row store is admitted and fires whole, one lane in one fire (a
    /// cacheless encoder attends over all its rows). A sequence request
    /// reading out more than the store holds is still refused, by name.
    #[test]
    fn a_float_pass_past_the_readout_store_fires_whole() {
        let store = ::engine::ReadoutFacts {
            store_rows: Some(128),
            ..::engine::ReadoutFacts::default()
        };
        let limits = reading(&store);
        let admission = AdmissionLimits::new(limits, 16);
        assert_eq!(
            admission.single_request_limit_error(&embedding(42, 512)),
            None,
            "a float pass is not held to the store"
        );
        let (readouts, lanes) = read_out(vec![embedding(42, 512)], limits);
        assert_eq!(readouts, vec![512], "one fire, every row read out");
        assert_eq!(lanes, 1);

        let mut lane = crate::engine::fire::lane_of(0, (0..200).collect(), 0, vec![1]);
        lane.readout = ::engine::Readout::Rows((0..200).collect());
        let refused = admission
            .single_request_limit_error(&pending(FireRequest::one(lane), InstanceId(43)))
            .expect("200 sequence readouts past a store of 128 are refused");
        assert!(
            refused.contains("instance 43")
                && refused.contains("200 rows")
                && refused.contains("128 rows the engine's readout store holds"),
            "refused by name: {refused}"
        );
    }

    /// Prompts reading every row of themselves (prompt logprobs) beside
    /// decodes, against a 128-row readout store: every fire's sequence
    /// readouts are cut to the store, every lane served. The engine lays a
    /// fire with such a lane at the store's rows (XLA's warm ladder), so a
    /// fire past it is unreachable from serving, and one prompt alone past
    /// it is refused by name (`a_float_pass_past_the_readout_store_fires_whole`).
    #[test]
    fn prompts_reading_every_row_are_cut_to_the_readout_store() {
        let store = ::engine::ReadoutFacts {
            store_rows: Some(128),
            ..::engine::ReadoutFacts::default()
        };
        let limits = reading(&store);
        let logprobs = |instance: u64| {
            let mut lane = crate::engine::fire::lane_of(0, (0..40).collect(), 0, vec![1]);
            lane.readout = ::engine::Readout::Rows((0..40).collect());
            pending(FireRequest::one(lane), InstanceId(instance))
        };
        let mut requests: Vec<Box<PendingRequest>> = (0..6).map(logprobs).collect();
        requests.extend((10..20).map(|i| pending(decode(1, 1), InstanceId(i))));
        let (readouts, lanes) = read_out(requests, limits);
        assert!(
            readouts.iter().all(|&rows| rows <= 128),
            "every fire reads out no more than the store holds: {readouts:?}"
        );
        assert!(
            readouts.len() > 1,
            "the prompts are cut across fires: {readouts:?}"
        );
        assert_eq!(lanes, 16, "every lane is served");
        assert_eq!(readouts.iter().sum::<usize>(), 6 * 40 + 10);
    }

    /// Two unframed float requests whose rows together pass the readout
    /// bound share a step, as before the step counted readouts: their rows
    /// are no sequence readouts, whether the bound is the engine's ceiling
    /// or a fixed store's.
    #[test]
    fn two_unframed_float_requests_share_a_step_again() {
        let ceiling = SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 4096,
            max_page_refs: 4096,
            max_readouts: 8 * 16,
            ..limits()
        };
        let store = reading(&::engine::ReadoutFacts {
            store_rows: Some(128),
            ..::engine::ReadoutFacts::default()
        });
        for limits in [ceiling, store] {
            assert_eq!(limits.max_readouts, 128);
            let (readouts, lanes) = read_out(vec![embedding(1, 100), embedding(2, 100)], limits);
            assert_eq!(readouts, vec![200], "one step of both: {readouts:?}");
            assert_eq!(lanes, 2);
        }
        // Sequence lanes past the same bound are cut, as they were.
        let (readouts, _) = read_out((0..10).map(verifying).collect(), ceiling);
        assert!(readouts.iter().all(|&rows| rows <= 128), "{readouts:?}");
        assert!(readouts.len() > 1);
    }

    /// The fires of `dispatch`'s frames, each fire id standing for the
    /// fire `fire_of` makes, stepped as the engine would build them: the
    /// cell each put of ring 7 takes from the engine's cursor, by fire id.
    fn minted_puts(
        frames: Vec<Vec<Vec<u64>>>,
        fire_of: impl Fn(u64) -> Box<PendingRequest>,
    ) -> Vec<(u64, u64)> {
        let mut cursor = 0u64;
        let mut minted = Vec::new();
        for waves in frames {
            let waves = waves
                .iter()
                .map(|wave| wave.iter().map(|&id| fire_of(id)).collect())
                .collect();
            let (frame, _) =
                build_frame_submission(waves, limits(), 16, &SchedulerStats::default());
            for step in &frame.steps {
                let puts: Vec<u64> = step
                    .instances
                    .iter()
                    .map(|id| id.0)
                    .filter(|&id| fire_of(id).request.ring_moves.iter().any(|ring| ring.puts))
                    .collect();
                assert!(puts.len() <= 1, "one put of ring 7 a step: {puts:?}");
                for id in puts {
                    minted.push((id, cursor));
                    cursor += 1;
                }
            }
        }
        minted
    }

    /// Two lanes each put ring 7, lane `a`'s fire (1) reserved first and
    /// lane `b`'s (2) after it. The seal sends 2 first in two ways: `a`'s
    /// frame does not fit the wave `b`'s does, or `a`'s put is its lane's
    /// second frame. Each put takes the cell the engine's cursor stands at
    /// when its step is built, so 2 has the earlier cell: no order the fires
    /// reserved in reaches the engine, and none is needed.
    fn a_seal_that_dispatches_the_later_putter_first_gives_it_the_earlier_cell() {
        use crate::scheduler::frame::{FramePlan, FramePolicy, FrameStamp, QueuedFireIds};
        let stamp = |lane: ProcessId, seq: u64| FrameStamp {
            lane,
            seq,
            slot: 0,
            fires: 1,
        };
        let dispatch = |policy: &mut FramePolicy, queued: &[u64]| -> Vec<Vec<u64>> {
            let queued: QueuedFireIds = queued.iter().copied().collect();
            match policy.plan_dispatch(&queued, &HashSet::new(), false, crate::rt::Instant::now()) {
                FramePlan::Dispatch(waves) => waves,
                plan => panic!("expected a frame dispatch, got {plan:?}"),
            }
        };
        let fire_of = |id: u64| {
            let mut fire = ring_fire(id, if id < 3 { &[(7, false, true)] } else { &[] });
            fire.logical_fire_id = id;
            fire
        };
        // Lanes stand in id order: `c` (filling the wave), `b`, then `a`.
        let (c, b, a) = (
            ProcessId::from_u128(1),
            ProcessId::from_u128(2),
            ProcessId::from_u128(3),
        );

        let mut policy = FramePolicy::new(1, 64, 4, None);
        policy.on_fire_enqueued(stamp(a, 0), Some(a), 1, 2, 1);
        policy.on_fire_enqueued(stamp(b, 0), Some(b), 2, 2, 1);
        policy.on_fire_enqueued(stamp(c, 0), Some(c), 3, 2, 1);
        let first = dispatch(&mut policy, &[1, 2, 3]);
        assert_eq!(first, [vec![3, 2]], "a's frame does not fit the wave");
        policy.on_frame_retired([b, c]);
        let second = dispatch(&mut policy, &[1]);
        assert_eq!(second, [vec![1]]);
        assert_eq!(minted_puts(vec![first, second], fire_of), [(2, 0), (1, 1)]);

        let mut policy = FramePolicy::new(1, 64, 4096, None);
        policy.on_fire_enqueued(stamp(a, 0), Some(a), 3, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1), Some(a), 1, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0), Some(b), 2, 1, 1);
        let first = dispatch(&mut policy, &[1, 2, 3]);
        let mut sorted = first.clone();
        sorted[0].sort_unstable();
        assert_eq!(sorted, [vec![2, 3]], "only a's front frame seals");
        policy.on_frame_retired([a, b]);
        // a's second frame waits for every awaited lane's next one.
        policy.on_fire_enqueued(stamp(b, 1), Some(b), 4, 1, 1);
        let second = dispatch(&mut policy, &[1, 4]);
        let mut sorted = second.clone();
        sorted[0].sort_unstable();
        assert_eq!(sorted, [vec![1, 4]]);
        assert_eq!(minted_puts(vec![first, second], fire_of), [(2, 0), (1, 1)]);
    }

    /// A fire of instance `instance`, a pipeline of its own, moving the
    /// ends `(channel, takes, puts)`.
    fn ring_fire(instance: u64, moves: &[(u64, bool, bool)]) -> Box<PendingRequest> {
        let mut request = decode(instance as u32, instance as u32);
        request.ring_moves = moves
            .iter()
            .map(|&(channel, takes, puts)| crate::engine::fire::RingMove {
                channel: ::engine::ChannelId(channel),
                takes,
                puts,
            })
            .collect();
        let mut pending = pending(request, InstanceId(instance));
        pending.pipeline_id = Some(uuid::Uuid::from_u128(u128::from(instance)));
        pending
    }

    /// The instances of each step of the submission `waves` make.
    fn stepped(waves: Vec<Vec<Box<PendingRequest>>>) -> Vec<Vec<u64>> {
        let (frame, flattened) =
            build_frame_submission(waves, limits(), 16, &SchedulerStats::default());
        let steps: Vec<Vec<u64>> = frame
            .steps
            .iter()
            .map(|step| {
                let mut instances: Vec<u64> = step.instances.iter().map(|id| id.0).collect();
                instances.sort_unstable();
                instances
            })
            .collect();
        assert_eq!(
            flattened.len(),
            steps.iter().map(Vec::len).sum::<usize>(),
            "every fire is seated once"
        );
        steps
    }

    /// A ring's cursor moves once a step at each end: the engine reads no
    /// order among the fires of one step, so two movers of one end go a step
    /// apart, in the order they stand; a putter and a taker share a step.
    fn fires_moving_one_ring_end_go_a_step_apart_in_the_order_they_stand() {
        let first = || ring_fire(1, &[(7, false, true)]);
        let second = || ring_fire(2, &[(7, false, true)]);
        assert_eq!(stepped(vec![vec![first(), second()]]), [vec![1], vec![2]]);
        assert_eq!(stepped(vec![vec![second(), first()]]), [vec![2], vec![1]]);
        assert_eq!(
            stepped(vec![vec![
                ring_fire(2, &[(7, true, false)]),
                ring_fire(1, &[(7, true, false)]),
            ]]),
            [vec![2], vec![1]]
        );
        assert_eq!(
            stepped(vec![vec![
                first(),
                pending(decode(9, 9), InstanceId(9)),
                ring_fire(3, &[(7, true, false)]),
                ring_fire(4, &[(8, false, true)]),
                second(),
            ]]),
            [vec![1, 3, 4, 9], vec![2]]
        );
        let mut group = StepGroup::default();
        group.push(&first(), limits(), 16);
        assert!(!group.accepts(&second(), limits(), 16), "the same end");
        assert!(
            group.accepts(&ring_fire(3, &[(7, true, false)]), limits(), 16),
            "the other end"
        );
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

    /// Restored decode lanes catching up replay rows their rung counts
    /// none of (`lane_carried`): where the engine states the rows its
    /// replay table holds a step, a step takes no more replayed rows than
    /// that, every lane's summed, and a fire that would pass it goes in a
    /// later step, served, never sent whole; one past it alone still goes,
    /// a step's first. An engine stating no bound steps them by its rung.
    fn a_step_replays_no_more_rows_than_the_engine_states() {
        let stated = SchedulerLimits {
            max_replayed_rows: 1024,
            ..limits()
        };
        let replaying = |instance: u64, replayed: u32| {
            let mut request = decode(9, 1);
            request.replayed = replayed;
            pending(request, InstanceId(instance))
        };
        // Four lanes of 257 rows each: three fit a step, the fourth waits.
        let mut group = StepGroup::default();
        for at in 1..=3 {
            assert!(group.accepts(&replaying(at, 257), stated, 16), "lane {at} fits");
            assert!(!group.push(&replaying(at, 257), stated, 16));
        }
        assert!(
            !group.accepts(&replaying(4, 257), stated, 16),
            "1028 replayed rows pass the 1024 the engine states"
        );
        assert!(group.accepts(&replaying(4, 253), stated, 16), "up to the bound");
        // A lane past the bound alone is a step's first and goes.
        let alone = StepGroup::default();
        assert!(alone.accepts(&replaying(5, 2000), stated, 16));
        // With no bound stated, replayed rows step nothing.
        let mut free = StepGroup::default();
        for at in 1..=4 {
            assert!(free.accepts(&replaying(at, 257), limits(), 16));
            assert!(!free.push(&replaying(at, 257), limits(), 16));
        }
        // A frame of the four is cut into two steps, no member dropped.
        let requests: Vec<Box<PendingRequest>> = (1..=4).map(|at| replaying(at, 257)).collect();
        let (frame, flattened) =
            build_frame_submission(vec![requests], stated, 16, &SchedulerStats::default());
        assert_eq!(flattened.len(), 4, "no member is dropped");
        assert_eq!(frame.steps.len(), 2, "split to the stated bound");
    }

    /// A frame's wave of prompts past what one fire carries is cut into
    /// fires each within it; the lanes' first decode rows, the next wave,
    /// are one fire. A prompt past one fire alone takes a fire of its own,
    /// and no member is dropped.
    fn a_wave_of_prompts_past_one_fire_is_cut_and_its_decode_wave_is_one_fire() {
        let limits = limits();
        let lanes = limits.max_forward_requests as u64;
        // Two prompts pass one fire.
        let prompt_tokens = limits.max_forward_tokens / 2 + 1;
        let prompt = |instance: u64, tokens: usize| {
            pending(
                FireRequest::one(crate::engine::fire::lane_of(
                    0,
                    vec![7; tokens],
                    0,
                    vec![instance as u32],
                )),
                InstanceId(instance),
            )
        };
        let step_tokens = |step: &StepFire| -> usize {
            step.submission
                .lanes
                .iter()
                .map(|lane| lane.tokens.len())
                .sum()
        };
        let decodes = || -> Vec<Box<PendingRequest>> {
            (1..=lanes)
                .map(|at| pending(decode(9, at as u32), InstanceId(at)))
                .collect()
        };

        let prompts = (1..=lanes).map(|at| prompt(at, prompt_tokens)).collect();
        let (frame, flattened) = build_frame_submission(
            vec![prompts, decodes()],
            limits,
            16,
            &SchedulerStats::default(),
        );
        assert_eq!(flattened.len(), 2 * lanes as usize, "no member is dropped");
        let (decode_fire, prompt_fires) = frame.steps.split_last().expect("steps");
        assert_eq!(prompt_fires.len(), lanes as usize, "a fire a prompt");
        for step in prompt_fires {
            assert!(step_tokens(step) <= limits.max_forward_tokens);
            assert_eq!(step_tokens(step), prompt_tokens);
        }
        assert_eq!(
            decode_fire.submission.lanes.len(),
            lanes as usize,
            "the lanes' first decode rows are one fire"
        );
        assert!(
            decode_fire
                .submission
                .lanes
                .iter()
                .all(|lane| lane.tokens.len() == 1)
        );

        // Short prompts share fires as they did: the cut is by what a fire
        // carries, and the decode wave is one fire all the same.
        let short = limits.max_forward_tokens / lanes as usize;
        let prompts = (1..=lanes).map(|at| prompt(at, short)).collect();
        let (frame, _) = build_frame_submission(
            vec![prompts, decodes()],
            limits,
            16,
            &SchedulerStats::default(),
        );
        assert_eq!(frame.steps.len(), 2, "one fire of prompts, one of decodes");
        assert_eq!(step_tokens(&frame.steps[0]), short * lanes as usize);
        assert_eq!(frame.steps[1].submission.lanes.len(), lanes as usize);

        // A prompt past one fire alone still takes a fire of its own.
        let wide = limits.max_forward_tokens + 1;
        let prompts = vec![prompt(1, short), prompt(2, wide), prompt(3, short)];
        let (frame, flattened) =
            build_frame_submission(vec![prompts], limits, 16, &SchedulerStats::default());
        assert_eq!(flattened.len(), 3, "no member is dropped");
        let mut sizes: Vec<usize> = frame.steps.iter().map(step_tokens).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![2 * short, wide]);
    }

    /// The engine holds a body for no more than `prefill_lanes` lanes of
    /// prompts alone: such a step stops there, and a decoding lane lifts it.
    fn a_step_of_prompts_only_stops_at_the_engines_prefill_lanes() {
        let stated = SchedulerLimits {
            prefill_lanes: 2,
            ..limits()
        };
        let prompt = |instance: u64| {
            pending(
                FireRequest::one(crate::engine::fire::lane_of(
                    0,
                    vec![1, 2, 3, 4],
                    0,
                    vec![1],
                )),
                InstanceId(instance),
            )
        };
        let mut group = StepGroup::default();
        assert!(!group.push(&prompt(1), stated, 16));
        assert!(group.accepts(&prompt(2), stated, 16));
        assert!(
            group.push(&prompt(2), stated, 16),
            "full at two prompt lanes"
        );
        assert!(
            !group.accepts(&prompt(3), stated, 16),
            "a third prompt lane is past what the engine carries"
        );
        let decoding = pending(decode(9, 1), InstanceId(4));
        assert!(
            group.accepts(&decoding, stated, 16),
            "a decoding lane joins"
        );
        group.push(&decoding, stated, 16);
        assert!(
            group.accepts(&prompt(5), stated, 16),
            "with a lane decoding the step is no prompts-only one"
        );
        // Without the engine stating it, prompts take as many lanes as any.
        let mut open = StepGroup::default();
        for at in 1..=4 {
            assert!(open.accepts(&prompt(at), limits(), 16));
            open.push(&prompt(at), limits(), 16);
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
            cells.iter().collect::<std::collections::HashSet<_>>().len(),
            cells.len(),
            "no cell is owned by two members"
        );
    }
}
