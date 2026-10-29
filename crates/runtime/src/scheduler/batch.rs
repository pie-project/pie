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
    pub(crate) logical_fire_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_requests: usize,
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let forward_requests = req.wire_row_count().max(1);
    let _ = page_size;

    RequestCapacityUsage {
        forward_requests,
        forward_tokens: req.request.tokens(),
        page_refs: req.request.pages().count(),
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
        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding engine limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }
        None
    }
}

#[derive(Default)]
pub(crate) struct StepGroup {
    instances: HashSet<::engine::InstanceId>,
    pipelines: HashSet<ProcessId>,
    forward_requests: usize,
    forward_tokens: usize,
    page_refs: usize,
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
        if self.forward_requests == 0 {
            return true;
        }
        let usage = request_capacity_usage(request, page_size);
        self.forward_requests.saturating_add(usage.forward_requests) <= limits.max_forward_requests
            && self.forward_tokens.saturating_add(usage.forward_tokens) <= limits.max_forward_tokens
            && self.page_refs.saturating_add(usage.page_refs) <= limits.max_page_refs
    }

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
        let mut completions = Vec::with_capacity(requests.len());
        let mut logical_fire_ids = Vec::with_capacity(requests.len());
        let mut member_lane_indptr = Vec::with_capacity(requests.len() + 1);
        member_lane_indptr.push(0);
        let mut media: Vec<engine::fire::StepMedia> = Vec::new();
        let mut voxels: Vec<engine::fire::StepVoxels> = Vec::new();
        // Group ids renumber past earlier members', so two passes' groups never merge.
        let mut groups = 0u32;
        for req in requests {
            // Where this fire sits in its pipeline, on each of its lanes, for
            // an engine that runs ahead (a device token loop).
            let ahead = req.frame.as_ref().map(crate::scheduler::FrameStamp::ahead);
            boundary_programs.push(req.request.boundary_program);
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            completions.push(req.completion.clone());
            logical_fire_ids.push(req.logical_fire_id);
            let base = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
            let mut own: Vec<u32> = Vec::new();
            lanes.extend(req.request.lanes.iter().cloned().map(|mut lane| {
                lane.ahead.clone_from(&ahead);
                lane.group = lane.group.map(|id| {
                    let local = own.iter().position(|&seen| seen == id).unwrap_or_else(|| {
                        own.push(id);
                        own.len() - 1
                    });
                    groups + u32::try_from(local).unwrap_or(u32::MAX)
                });
                lane
            }));
            groups = groups.saturating_add(u32::try_from(own.len()).unwrap_or(u32::MAX));
            member_lane_indptr.push(u32::try_from(lanes.len()).unwrap_or(u32::MAX));
            media.extend(req.request.media.iter().cloned().map(|mut row| {
                row.lane = row.lane.saturating_add(base);
                row
            }));
            voxels.extend(req.request.voxels.iter().cloned().map(|mut row| {
                row.lane = row.lane.saturating_add(base);
                row
            }));
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
            logical_fire_ids,
        }
    })
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
        let group: Vec<Box<PendingRequest>> = fire_plan::member_order(&facts)
            .iter()
            .map(|&index| slots[index].take().expect("member_order is a permutation"))
            .collect();

        let build = build_batch_request(&group, page_size, stats);
        if super::worker::wave_trace() {
            let (mut tokens, mut page_refs) = (0usize, 0usize);
            for req in &group {
                let usage = request_capacity_usage(req, page_size);
                tokens += usage.forward_tokens;
                page_refs += usage.page_refs;
            }
            let prefill_lanes = build
                .lanes
                .iter()
                .filter(|lane| lane.tokens.len() > 1)
                .count();
            super::worker::wave_trace_emit(format!(
                "[step-lanes] t={}us members={} prefill_lanes={prefill_lanes} lanes={} tokens={tokens}/{} page_refs={page_refs}/{} kv_len_max={}",
                super::worker::wave_trace_us(),
                group.len(),
                build.lanes.len(),
                limits.max_forward_tokens,
                limits.max_page_refs,
                group
                    .iter()
                    .flat_map(|req| req.request.lanes.iter().map(|lane| lane.kv.pages.len()))
                    .max()
                    .unwrap_or(0)
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
            });
        }
        steps.push(StepFire {
            submission: ::engine::Step {
                lanes,
                attachments,
                media: build.media,
                voxels: build.voxels,
            },
            terminal_cells: build.terminal_cells,
            completions: build.completions,
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
    use ::engine::InstanceId;
    use eta_ir::registry::GeometryClass;

    fn pending(request: FireRequest, instance_id: InstanceId) -> Box<PendingRequest> {
        Box::new(PendingRequest {
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
            max_readouts: 0,
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
    }

    fn every_member_owns_a_span_of_the_steps_lanes() {
        let placeholder = FireRequest::default();
        let requests = vec![
            pending(decode(11, 3), InstanceId(1)),
            pending(placeholder, InstanceId(2)),
            pending(decode(22, 4), InstanceId(3)),
        ];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());
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
        let expected = two.lanes.clone();

        let step = build_batch_request(
            &[pending(two, InstanceId(1))],
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
        let requests = [
            pending(two, InstanceId(9)),
            pending(decode(33, 5), InstanceId(10)),
        ];
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
        let requests = [
            pending(grouped(0), InstanceId(1)),
            pending(grouped(4), InstanceId(2)),
        ];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());
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

        let requests = [
            pending(masked, InstanceId(20)),
            pending(decode(22, 4), InstanceId(21)),
        ];
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
        pooled.lanes[0].kv.pages.clear();

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
