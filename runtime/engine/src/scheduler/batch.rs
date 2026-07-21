//! Batch assembly: capacity accounting + the dense-batch accumulator.

use super::stats::SchedulerStats;
use super::wire;
use super::worker::PendingRequest;
use crate::driver::{LaunchSubmission, SchedulerLimits};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_requests: usize,
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let input_tokens = req.request.token_ids.len();
    let forward_requests = req
        .wire_row_count()
        .max(req.request.rs_slot_ids.len())
        .max(1);
    let forward_tokens = input_tokens;
    let page_refs = req.request.kv_page_indices.len();
    let _ = page_size;

    RequestCapacityUsage {
        forward_requests,
        forward_tokens,
        page_refs,
    }
}

pub(crate) struct BatchAccumulator {
    requests: Vec<PendingRequest>,
    page_size: u32,
    limits: SchedulerLimits,
}

impl BatchAccumulator {
    pub(crate) fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self {
            requests: Vec::new(),
            page_size,
            limits,
        }
    }

    pub(crate) fn push(&mut self, req: PendingRequest) {
        self.requests.push(req);
    }

    pub(crate) fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
        if usage.forward_requests > self.limits.max_forward_requests {
            return Some(format!(
                "forward request has {} resolved rows, exceeding driver limit {}",
                usage.forward_requests, self.limits.max_forward_requests
            ));
        }
        if usage.forward_tokens > self.limits.max_forward_tokens {
            return Some(format!(
                "forward request has {} forward tokens, exceeding driver limit {}",
                usage.forward_tokens, self.limits.max_forward_tokens
            ));
        }
        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding driver limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }
        // Malformed shapes reject the FIRE at admission; batch build treats
        // them as invariants (a panic there would take down the scheduler
        // thread — RV-20).
        if req.request.kv_write_lower_bounds.len() > 1
            || req.request.kv_write_upper_bounds.len() > 1
            || req.request.kv_write_lower_bounds.len() != req.request.kv_write_upper_bounds.len()
        {
            return Some("per-fire KV containment bounds must be empty or scalar".to_string());
        }
        let rows = req.request.qo_indptr.len().saturating_sub(1);
        if rows > 1 && !req.request.masks.is_empty() && req.request.mask_indptr.len() != rows + 1 {
            return Some(format!(
                "multi-row fire carries {} masks without a row CSR \
                 ({} mask boundaries for {} rows)",
                req.request.masks.len(),
                req.request.mask_indptr.len(),
                rows
            ));
        }
        None
    }

    pub(crate) fn take(&mut self) -> Vec<PendingRequest> {
        std::mem::take(&mut self.requests)
    }
}

pub(crate) fn build_batch_request(
    requests: &[PendingRequest],
    page_size: u32,
    stats: &SchedulerStats,
) -> LaunchSubmission {
    if requests.len() == 1 && (requests[0].prebuilt || requests[0].preserves_inner_rows()) {
        // Keep the logical-fire payload intact across attempts. RETRY builds a
        // fresh native launch from this same immutable request. Ordinary
        // multi-row programs take this path too: flattening them through
        // `wire::append_request` would collapse their inner CSR to one row
        // while incorrectly retaining B recurrent-state slots.
        let req = &requests[0];
        let mut plan = req.request.clone();
        let kv_translation = std::mem::take(&mut plan.kv_translation);
        plan.required_kv_pages = plan.required_kv_pages.max(
            kv_translation
                .iter()
                .copied()
                .max()
                .map_or(0, |page| page.saturating_add(1)),
        );
        let channel_expected_head = plan.channel_expected_head.clone();
        let channel_expected_tail = plan.channel_expected_tail.clone();
        let channel_ticket_len = channel_expected_head.len() as u32;
        let rows = plan.qo_indptr.len().saturating_sub(1) as u32;
        return LaunchSubmission {
            kv_translation_indptr: vec![0, kv_translation.len() as u32],
            kv_translation,
            program_row_indptr: vec![0, rows],
            plan,
            instance_ids: vec![req.instance_id],
            terminal_cells: vec![req.completion.terminal_cell_ptr()],
            logical_fire_ids: vec![req.logical_fire_id],
            channel_expected_head,
            channel_expected_tail,
            channel_ticket_indptr: vec![0, channel_ticket_len],
        };
    }
    let elide_decode_masks = requests.iter().all(|req| {
        req.request.single_token_mode
            && !req.request.has_user_mask
            && req.request.token_ids.len() <= 1
    });
    let use_kv_write_bounds = requests.iter().any(|req| {
        !req.request.kv_write_lower_bounds.is_empty()
            || !req.request.kv_write_upper_bounds.is_empty()
    });
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut batch_req = wire::new_batched_forward_request_with_capacity(requests.len());
        let mut instance_ids = Vec::with_capacity(requests.len());
        let mut terminal_cells = Vec::with_capacity(requests.len());
        let mut kv_translation = Vec::new();
        let mut kv_translation_indptr = Vec::with_capacity(requests.len() + 1);
        kv_translation_indptr.push(0);
        let mut logical_fire_ids = Vec::with_capacity(requests.len());
        let mut channel_expected_head = Vec::new();
        let mut channel_expected_tail = Vec::new();
        let mut channel_ticket_indptr = Vec::with_capacity(requests.len() + 1);
        channel_ticket_indptr.push(0);
        let mut program_row_indptr = Vec::with_capacity(requests.len() + 1);
        program_row_indptr.push(0);
        for req in requests {
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            logical_fire_ids.push(req.logical_fire_id);
            wire::append_request_with_options(
                &mut batch_req,
                &req.request,
                req.last_page_len,
                page_size,
                elide_decode_masks,
            );
            if use_kv_write_bounds {
                match (
                    req.request.kv_write_lower_bounds.as_slice(),
                    req.request.kv_write_upper_bounds.as_slice(),
                ) {
                    ([], []) => {
                        batch_req.kv_write_lower_bounds.push(0);
                        batch_req.kv_write_upper_bounds.push(u64::MAX);
                    }
                    ([lower], [upper]) => {
                        batch_req.kv_write_lower_bounds.push(*lower);
                        batch_req.kv_write_upper_bounds.push(*upper);
                    }
                    _ => panic!("per-fire KV containment bounds must be empty or scalar"),
                }
            }
            kv_translation.extend(req.request.kv_translation.iter().copied());
            kv_translation_indptr.push(kv_translation.len() as u32);
            channel_expected_head.extend(req.request.channel_expected_head.iter().copied());
            channel_expected_tail.extend(req.request.channel_expected_tail.iter().copied());
            channel_ticket_indptr.push(channel_expected_head.len() as u32);
            program_row_indptr.push(
                program_row_indptr.last().copied().unwrap_or(0)
                    + req.wire_row_count().max(1) as u32,
            );
        }
        LaunchSubmission {
            plan: batch_req,
            instance_ids,
            terminal_cells,
            kv_translation,
            kv_translation_indptr,
            program_row_indptr,
            logical_fire_ids,
            channel_expected_head,
            channel_expected_tail,
            channel_ticket_indptr,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{LaunchPlan, WorkItemCompletion};

    fn pending(request: LaunchPlan, instance_id: u64, prebuilt: bool) -> PendingRequest {
        PendingRequest {
            logical_fire_id: 1,
            last_page_len: 1,
            request,
            instance_id,
            completion: WorkItemCompletion::new(instance_id, 0),
            process_id: None,
            pipeline_id: None,
            prebuilt,
            retry_count: 0,
            retry_after: None,
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            retry_classifier: None,
            credit_published: false,
            quorum_generation: 0,
            timing: None,
        }
    }

    fn wire_decode(token: u32, page: u32) -> LaunchPlan {
        LaunchPlan {
            token_ids: vec![token],
            position_ids: vec![0],
            kv_page_indices: vec![page],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![1],
            qo_indptr: vec![0, 1],
            sampling_indices: vec![0],
            sampling_indptr: vec![0, 1],
            mask_indptr: vec![0, 0],
            single_token_mode: true,
            ..LaunchPlan::default()
        }
    }

    #[test]
    fn batched_fires_attribute_one_row_each() {
        // Two wire decodes around a device-geometry placeholder: every fire
        // owns exactly one wire request row; the placeholder's row is empty
        // of tokens/sampling but still holds its boundary.
        let dg = LaunchPlan {
            kv_translation: vec![7, 8],
            ..LaunchPlan::default()
        };
        let requests = vec![
            pending(wire_decode(11, 3), 1, false),
            pending(dg, 2, true),
            pending(wire_decode(22, 4), 3, false),
        ];
        let mut requests = requests;
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 1, 2, 3]);
        assert_eq!(
            sub.plan.qo_indptr,
            vec![0, 1, 1, 2],
            "placeholder row is empty"
        );
        assert_eq!(sub.plan.sampling_indptr, vec![0, 1, 1, 2]);
        assert_eq!(sub.kv_translation, vec![7, 8]);
        assert_eq!(sub.kv_translation_indptr, vec![0, 0, 2, 2]);
    }

    #[test]
    fn batch_preserves_largest_required_kv_high_water() {
        let mut first = wire_decode(11, 3);
        first.kv_translation = vec![3, 16];
        let mut second = wire_decode(22, 4);
        second.kv_translation = vec![8, 28];
        let mut requests = [pending(first, 1, false), pending(second, 2, false)];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());

        assert_eq!(sub.plan.required_kv_pages, 29);
    }

    #[test]
    fn prebuilt_solo_owns_all_wire_rows() {
        // A prebuilt wire plan with two lanes: its single program owns both
        // rows. A device-geometry prebuilt (empty qo) owns zero rows.
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.rs_slot_ids = vec![31, 32];
        two_lane.rs_slot_flags = vec![0, crate::driver::RS_FLAG_RESET];
        let mut solo = [pending(two_lane, 1, true)];
        let sub = build_batch_request(&mut solo, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 2]);
        assert_eq!(sub.plan.qo_indptr, vec![0, 1, 2]);
        assert_eq!(sub.plan.rs_slot_ids, vec![31, 32]);

        let dg = LaunchPlan::default();
        let mut solo_dg = [pending(dg, 2, true)];
        let sub = build_batch_request(&mut solo_dg, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 0]);
    }

    #[test]
    fn ordinary_multi_row_submission_remains_intact() {
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.kv_page_indices = vec![3, 4];
        two_lane.kv_page_indptr = vec![0, 1, 2];
        two_lane.kv_last_page_lens = vec![1, 1];
        two_lane.sampling_indices = vec![0, 0];
        two_lane.sampling_indptr = vec![0, 1, 2];
        two_lane.mask_indptr = vec![0, 0, 0];
        two_lane.rs_slot_ids = vec![17, 23];
        two_lane.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET, 0];
        let expected = two_lane.clone();

        let mut ordinary = [pending(two_lane, 9, false)];
        let sub = build_batch_request(&mut ordinary, 16, &SchedulerStats::default());

        assert_eq!(sub.instance_ids, vec![9]);
        assert_eq!(sub.program_row_indptr, vec![0, 2]);
        assert_eq!(sub.plan.qo_indptr, expected.qo_indptr);
        assert_eq!(sub.plan.kv_page_indptr, expected.kv_page_indptr);
        assert_eq!(sub.plan.sampling_indptr, expected.sampling_indptr);
        assert_eq!(sub.plan.rs_slot_ids, vec![17, 23]);
        assert_eq!(
            sub.plan.rs_slot_flags,
            vec![crate::driver::RS_FLAG_RESET, 0]
        );
    }

    /// Multi-row masks without a row CSR reject the FIRE at admission —
    /// the batch build treats the shape as an invariant, and a panic there
    /// would take down the scheduler thread (RV-20).
    #[test]
    fn multi_row_masks_without_a_row_csr_reject_at_admission() {
        let accumulator = BatchAccumulator::new(
            SchedulerLimits {
                max_forward_requests: 64,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
            16,
        );
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.kv_page_indices = vec![3, 4];
        two_lane.kv_page_indptr = vec![0, 1, 2];
        two_lane.kv_last_page_lens = vec![1, 1];
        two_lane.sampling_indices = vec![0, 0];
        two_lane.sampling_indptr = vec![0, 1, 2];
        two_lane.masks = vec![crate::driver::command::EncodedMask::new(vec![1], 1)];
        two_lane.mask_indptr = vec![0, 1];

        let message = accumulator
            .single_request_limit_error(&pending(two_lane.clone(), 9, false))
            .expect("multi-row masks without a row CSR must reject the fire");
        assert!(message.contains("without a row CSR"), "{message}");

        // The same fire with a row CSR is admitted.
        two_lane.mask_indptr = vec![0, 1, 1];
        assert_eq!(
            accumulator.single_request_limit_error(&pending(two_lane, 9, false)),
            None
        );
    }

    #[test]
    fn multi_row_submission_cobatches_without_collapsing_csrs() {
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.kv_page_indices = vec![3, 4];
        two_lane.kv_page_indptr = vec![0, 1, 2];
        two_lane.kv_last_page_lens = vec![1, 1];
        two_lane.sampling_indices = vec![0, 0];
        two_lane.sampling_indptr = vec![0, 1, 2];
        two_lane.mask_indptr = vec![0, 0, 0];
        two_lane.rs_slot_ids = vec![17, 23];
        two_lane.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET, 0];

        let mut requests = [
            pending(two_lane, 9, false),
            pending(wire_decode(33, 5), 10, false),
        ];
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());

        assert_eq!(sub.program_row_indptr, vec![0, 2, 3]);
        assert_eq!(sub.plan.qo_indptr, vec![0, 1, 2, 3]);
        assert_eq!(sub.plan.kv_page_indptr, vec![0, 1, 2, 3]);
        assert_eq!(sub.plan.sampling_indptr, vec![0, 1, 2, 3]);
        assert_eq!(sub.plan.sampling_indices, vec![0, 0, 0]);
        assert_eq!(sub.plan.rs_slot_ids, vec![17, 23]);
        assert_eq!(sub.plan.image_indptr, vec![0, 0, 0, 0]);
        assert_eq!(sub.plan.audio_indptr, vec![0, 0, 0, 0]);
        assert_eq!(sub.plan.embed_block_indptr, vec![0, 0, 0, 0]);
    }

    #[test]
    fn device_resolved_multitoken_geometry_skips_placeholder_mask_trim() {
        let mut request = wire_decode(11, 3);
        request.token_ids = vec![11, 0, 0, 0];
        request.position_ids = vec![0; 4];
        request.qo_indptr = vec![0, 4];
        request.kv_page_indices = vec![3, 4, 5];
        request.kv_page_indptr = vec![0, 3];
        request.kv_last_page_lens = vec![6];
        request.sampling_indices = vec![0, 1, 2, 3];
        request.sampling_indptr = vec![0, 4];
        request.single_token_mode = false;
        request.device_resolved_geometry = true;

        let mut requests = [pending(request, 12, false)];
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.plan.kv_page_indices, vec![3, 4, 5]);
        assert!(sub.plan.masks.is_empty());
    }

    #[test]
    fn host_custom_mask_cobatches_with_causal_fire() {
        let mut custom = wire_decode(11, 3);
        custom.has_user_mask = true;
        custom.single_token_mode = false;
        custom.masks = vec![crate::driver::command::EncodedMask::new(vec![1], 1)];
        custom.mask_indptr = vec![0, 1];
        let mut requests = [
            pending(custom, 20, false),
            pending(wire_decode(22, 4), 21, false),
        ];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.instance_ids, vec![20, 21]);
        assert_eq!(sub.plan.mask_indptr, vec![0, 1, 2]);
        assert_eq!(sub.plan.masks.len(), 2);
        assert_eq!(sub.plan.masks[0].runs, vec![1], "explicit custom row");
        assert_eq!(
            sub.plan.masks[1].runs,
            vec![0, 1],
            "causal peer receives the synthesized compatible row"
        );
        assert!(sub.plan.has_user_mask);
        assert!(!sub.plan.single_token_mode);
    }

    #[test]
    fn host_mask_on_device_geometry_is_not_elided_as_dense() {
        let mut request = wire_decode(11, 3);
        request.device_resolved_geometry = true;
        request.has_user_mask = true;
        request.single_token_mode = false;
        request.masks = vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        request.mask_indptr = vec![0, 1];

        let mut requests = [pending(request, 12, false)];
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.plan.masks.len(), 1);
        assert_eq!(sub.plan.mask_indptr, vec![0, 1]);
        assert!(sub.plan.has_user_mask);
    }

    #[test]
    fn deferred_multi_row_geometry_cobatches_as_zero_kv_spans() {
        let mut plan = wire_decode(11, 3);
        plan.token_ids = vec![11, u32::MAX];
        plan.position_ids = vec![0, 0];
        plan.qo_indptr = vec![0, 1, 2];
        plan.kv_page_indices.clear();
        plan.kv_page_indptr.clear();
        plan.kv_last_page_lens.clear();
        plan.sampling_indices = vec![0, 0];
        plan.sampling_indptr = vec![0, 1, 2];
        plan.mask_indptr = vec![0, 0, 0];
        plan.device_resolved_geometry = true;
        plan.kv_write_lower_bounds = vec![7];
        plan.kv_write_upper_bounds = vec![15];
        let mut requests = [pending(plan.clone(), 20, false), pending(plan, 21, false)];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 2, 4]);
        assert_eq!(sub.plan.kv_page_indices, Vec::<u32>::new());
        assert_eq!(sub.plan.kv_page_indptr, vec![0, 0, 0, 0, 0]);
        assert_eq!(sub.plan.kv_last_page_lens, vec![0, 0, 0, 0]);
        assert_eq!(sub.plan.kv_write_lower_bounds, vec![7, 7]);
        assert_eq!(sub.plan.kv_write_upper_bounds, vec![15, 15]);
    }

    #[test]
    fn mixed_batches_fill_unbounded_containment_entries() {
        let plain = wire_decode(11, 3);
        let mut bounded = wire_decode(22, 4);
        bounded.kv_write_lower_bounds = vec![7];
        bounded.kv_write_upper_bounds = vec![9];
        let mut requests = [pending(plain, 20, false), pending(bounded, 21, false)];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.plan.kv_write_lower_bounds, vec![0, 7]);
        assert_eq!(sub.plan.kv_write_upper_bounds, vec![u64::MAX, 9]);
    }
}
