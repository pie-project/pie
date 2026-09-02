//! Batch assembly: capacity accounting, and the lanes a step is made of.

use std::collections::HashSet;

use crate::engine::completion::TerminalCell;
use crate::engine::{FrameFire, SchedulerLimits, StepFire};
use crate::scheduler::ProcessId;

use super::fire_plan;
use super::stats::SchedulerStats;
use super::worker::PendingRequest;

/// One step's assembled lanes, before the frame is sealed.
pub(crate) struct StepBuild {
    /// The lanes, in member order.
    pub(crate) lanes: Vec<::engine::Lane>,
    /// Media rows, rebased onto the step's lane numbering (each member's
    /// own lane index is offset by its base lane in the step).
    pub(crate) media: Vec<engine::fire::StepMedia>,
    /// Which bound instance each member belongs to.
    pub(crate) instance_ids: Vec<u64>,
    /// Whether each member's instance runs a pass at the fire's boundary —
    /// [`FireRequest::boundary_program`](crate::engine::FireRequest).
    pub(crate) boundary_programs: Vec<bool>,
    /// Which lane each member's rows start at.
    pub(crate) member_lane_indptr: Vec<u32>,
    /// One cell per member.
    pub(crate) terminal_cells: Vec<*mut TerminalCell>,
    /// One id per member.
    pub(crate) logical_fire_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_requests: usize,
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    // A lane is one forward request; at least one even for an empty request.
    let forward_requests = req.wire_row_count().max(1);
    let _ = page_size;

    RequestCapacityUsage {
        forward_requests,
        forward_tokens: req.request.tokens(),
        page_refs: req.request.pages().count(),
    }
}

/// Admission-time shape gate: rejects a single fire whose resolved shape
/// exceeds the engine's launch limits before it can enter the queue.
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
        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding engine limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }
        None
    }
}

/// What one step may hold. Refuses a request that would reseat an instance already in it, shares a pipeline already in it, or would exceed the accumulated capacity caps.
#[derive(Default)]
pub(crate) struct StepGroup {
    /// Bound instances already in this step.
    instances: HashSet<u64>,
    /// Tracked pipelines already in this step.
    pipelines: HashSet<ProcessId>,
    forward_requests: usize,
    forward_tokens: usize,
    page_refs: usize,
}

impl StepGroup {
    /// May `request` join this step? An empty group admits anything, since
    /// a fire the caps refuse on its own is already refused at admission.
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
        if self.forward_requests == 0 {
            return true;
        }
        let usage = request_capacity_usage(request, page_size);
        self.forward_requests.saturating_add(usage.forward_requests) <= limits.max_forward_requests
            && self.forward_tokens.saturating_add(usage.forward_tokens) <= limits.max_forward_tokens
            && self.page_refs.saturating_add(usage.page_refs) <= limits.max_page_refs
    }

    /// Take `request` into this step. Answers whether the step is now full —
    /// the caller stops offering it members.
    pub(crate) fn push(
        &mut self,
        request: &PendingRequest,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> bool {
        let usage = request_capacity_usage(request, page_size);
        self.instances.insert(request.instance_id);
        if let Some(pid) = request.pipeline_id {
            self.pipelines.insert(pid);
        }
        self.forward_requests = self.forward_requests.saturating_add(usage.forward_requests);
        self.forward_tokens = self.forward_tokens.saturating_add(usage.forward_tokens);
        self.page_refs = self.page_refs.saturating_add(usage.page_refs);
        self.forward_requests >= limits.max_forward_requests
            || self.forward_tokens >= limits.max_forward_tokens
            || self.page_refs >= limits.max_page_refs
    }
}

/// Assemble one step's lanes out of its member requests: a concatenation,
/// in member order.
pub(crate) fn build_batch_request(
    requests: &[Box<PendingRequest>],
    page_size: u32,
    stats: &SchedulerStats,
) -> StepBuild {
    let _ = page_size;
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut lanes = Vec::with_capacity(requests.len());
        let mut instance_ids = Vec::with_capacity(requests.len());
        let mut boundary_programs = Vec::with_capacity(requests.len());
        let mut terminal_cells = Vec::with_capacity(requests.len());
        let mut logical_fire_ids = Vec::with_capacity(requests.len());
        let mut member_lane_indptr = Vec::with_capacity(requests.len() + 1);
        member_lane_indptr.push(0);
        let mut media: Vec<engine::fire::StepMedia> = Vec::new();
        for req in requests {
            boundary_programs.push(req.request.boundary_program);
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            logical_fire_ids.push(req.logical_fire_id);
            // base must be read before this member's lanes are appended.
            let base = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
            lanes.extend(req.request.lanes.iter().cloned());
            member_lane_indptr.push(u32::try_from(lanes.len()).unwrap_or(u32::MAX));
            // A text-only member contributes no media row.
            media.extend(req.request.media.iter().cloned().map(|mut row| {
                row.lane = row.lane.saturating_add(base);
                row
            }));
        }
        StepBuild {
            lanes,
            media,
            instance_ids,
            boundary_programs,
            member_lane_indptr,
            terminal_cells,
            logical_fire_ids,
        }
    })
}

/// Assemble one sealed frame's fires from its waves' picked requests, in slot order. Returns the frame plus the flattened requests in post order (step order, member order within a step) — the retirement set.
/// Within a wave, requests group under [`StepGroup`] and each group becomes one step.
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
        // Incompatible members defer to the wave's next step.
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
                // Defensive: never loop forever on a malformed head.
                group.push(rest.remove(0));
            }
            step_groups.push(group);
            deferred = rest;
        }
    }

    let mut steps: Vec<StepFire> = Vec::new();
    let mut flattened: Vec<Box<PendingRequest>> = Vec::new();

    for group in step_groups {
        // The fire planner's seriation decides submission order within the group.
        let facts: Vec<fire_plan::MemberFacts> = group
            .iter()
            .enumerate()
            .map(|(arrival, req)| fire_plan::MemberFacts {
                hook_program: req.hook_program,
                lora: req.lora_program,
                custom_mask: req.request.has_user_mask,
                truncated: req.request.max_layers.is_some(),
                max_layers: req.request.max_layers,
                multi_token: req.request.lanes.iter().any(|lane| lane.tokens.len() > 1),
                geometry_class: req.request.geometry,
                arrival,
            })
            .collect();
        let plan = fire_plan::plan_fire_with_model(&facts, &[]);
        let mut slots: Vec<Option<Box<PendingRequest>>> = group.into_iter().map(Some).collect();
        let group: Vec<Box<PendingRequest>> = plan
            .member_order
            .iter()
            .map(|&index| slots[index].take().expect("member_order is a permutation"))
            .collect();

        let build = build_batch_request(&group, page_size, stats);
        if super::worker::wave_trace() {
            eprintln!(
                "[step-lanes] members={} lanes={}",
                group.len(),
                build.lanes.len()
            );
        }
        // One instance id per lane, not per member.
        let mut instances = Vec::with_capacity(build.lanes.len());
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            let span = build.member_lane_indptr[member + 1] - build.member_lane_indptr[member];
            instances.extend(std::iter::repeat_n(instance, span as usize));
        }
        // one attachment per member, not per lane; the lane named is the member's first.
        let mut attachments = Vec::new();
        let lanes = build.lanes;
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            if !build.boundary_programs[member] {
                continue;
            }
            let lane = build.member_lane_indptr[member];
            attachments.push(::engine::Attachment {
                lane,
                instance,
                at: ::engine::Boundary::Epilogue,
            });
        }
        steps.push(StepFire {
            submission: ::engine::Step {
                lanes,
                attachments,
                media: build.media,
            },
            terminal_cells: build.terminal_cells,
            instances,
            logical_fire_ids: build.logical_fire_ids,
        });
        flattened.extend(group);
    }

    (FrameFire { steps }, flattened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FireRequest, WorkItemCompletion};
    use eta_ir::registry::GeometryClass;

    fn pending(request: FireRequest, instance_id: u64) -> Box<PendingRequest> {
        Box::new(PendingRequest {
            hook_program: false,
            lora_program: false,
            logical_fire_id: 1,
            request,
            instance_id,
            completion: WorkItemCompletion::new(instance_id, 0),
            process_id: None,
            pipeline_id: None,
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: None,
        })
    }

    /// One decode lane: one token, one page, last-row readout.
    fn decode(token: u32, page: u32) -> FireRequest {
        let mut request = FireRequest::one(crate::engine::fire::lane_of(0, vec![token], 0, vec![page]));
        request.single_token_mode = true;
        request
    }

    fn limits() -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
        }
    }

    /// Every member's lanes stand where `member_lane_indptr` says they do.
    #[test]
    fn every_member_owns_a_span_of_the_steps_lanes() {
        let placeholder = FireRequest::default();
        let requests = vec![
            pending(decode(11, 3), 1),
            pending(placeholder, 2),
            pending(decode(22, 4), 3),
        ];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());
        assert_eq!(step.member_lane_indptr, vec![0, 1, 1, 2]);
        assert_eq!(step.instance_ids, vec![1, 2, 3]);
        assert_eq!(step.lanes.len(), 2, "the placeholder contributes no lane");
        assert_eq!(step.lanes[0].tokens, vec![11]);
        assert_eq!(step.lanes[1].tokens, vec![22]);
    }

    /// A multi-lane member keeps every one of its lanes.
    #[test]
    fn a_multi_lane_member_arrives_lane_for_lane() {
        let mut two = FireRequest {
            lanes: vec![
                crate::engine::fire::lane_of(0, vec![11], 0, vec![3]),
                crate::engine::fire::lane_of(1, vec![22], 0, vec![4]),
            ],
            ..FireRequest::default()
        };
        // The recurrent verb and reset fact live on the lane.
        two.lanes[1].rs_reset = engine::fire::RsReset::Fresh;
        let expected = two.lanes.clone();

        let step = build_batch_request(&[pending(two, 1)], 16, &SchedulerStats::default());
        assert_eq!(step.member_lane_indptr, vec![0, 2]);
        assert_eq!(step.lanes, expected);
    }

    /// Two members co-batch by concatenation, in member order.
    #[test]
    fn members_cobatch_by_concatenation() {
        let two = FireRequest {
            lanes: vec![
                crate::engine::fire::lane_of(0, vec![11], 0, vec![3]),
                crate::engine::fire::lane_of(1, vec![22], 0, vec![4]),
            ],
            ..FireRequest::default()
        };
        let requests = [pending(two, 9), pending(decode(33, 5), 10)];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());

        assert_eq!(step.member_lane_indptr, vec![0, 2, 3]);
        assert_eq!(
            step.lanes
                .iter()
                .map(|lane| lane.tokens.clone())
                .collect::<Vec<_>>(),
            vec![vec![11], vec![22], vec![33]]
        );
    }

    /// A lane's mask is the lane's, and a peer with none gets none.
    #[test]
    fn a_masked_lane_carries_its_own_mask_and_a_peer_carries_none() {
        let mut masked = decode(11, 3);
        masked.has_user_mask = true;
        masked.single_token_mode = false;
        masked.lanes[0].mask = Some(::engine::Masking::Extent(
            ::engine::Mask::new(vec![0, 1], 1),
        ));

        let requests = [pending(masked, 20), pending(decode(22, 4), 21)];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());
        assert_eq!(
            step.lanes[0].mask,
            Some(::engine::Masking::Extent(::engine::Mask::new(
                vec![0, 1],
                1
            )))
        );
        assert_eq!(step.lanes[1].mask, None, "an unmasked peer stays unmasked");
    }

    /// A batched fire's lanes carry distinct pool slots: the seat each
    /// member states is what keeps two concurrent guests out of one
    /// another's pool slot.
    #[test]
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
        crate::pipeline::fire::stamp_lane_slots(&mut first, &stores, first_ws)
            .expect("a two-slot pool seats one sequence");
        crate::pipeline::fire::stamp_lane_slots(&mut second, &stores, second_ws)
            .expect("and its peer");

        let (frame, _) = build_frame_submission(
            vec![vec![pending(first, 41), pending(second, 42)]],
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

    /// A device-resolved member is a contiguous suffix, never interleaved
    /// with host members.
    #[test]
    fn a_device_geometry_member_seriates_into_the_suffix() {
        let mut pooled = decode(11, 3);
        pooled.geometry = GeometryClass::DeviceGeometry;
        // This fire states its pages in a channel, not in the submission.
        pooled.lanes[0].kv.pages.clear();

        let (frame, flattened) = build_frame_submission(
            vec![vec![pending(pooled, 13), pending(decode(22, 4), 12)]],
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
        assert_eq!(step.instances, vec![12, 13]);
    }

    /// A frame's steps carry one terminal cell per member, and no two steps
    /// share one.
    #[test]
    fn a_frames_cells_are_one_per_member_and_never_shared() {
        let (frame, flattened) = build_frame_submission(
            vec![
                vec![pending(decode(11, 3), 1)],
                vec![pending(decode(22, 4), 2)],
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
