//! Batch assembly: capacity accounting + the dense-batch accumulator.

use super::request;
use super::scheduler::PendingRequest;
use super::stats::SchedulerStats;
use crate::driver::{LaunchSubmission, SchedulerLimits};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
    pub(crate) is_single_token_decode: bool,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let input_tokens = req.request.token_ids.len();
    let forward_tokens = input_tokens;
    let is_single_token_decode =
        input_tokens == 1 && req.request.single_token_mode && !req.request.has_user_mask;
    let page_refs = req.physical_page_ids.len();
    let _ = page_size;

    RequestCapacityUsage {
        forward_tokens,
        page_refs,
        is_single_token_decode,
    }
}

pub(crate) struct BatchAccumulator {
    requests: Vec<PendingRequest>,
    total_tokens: usize,
    total_pages: usize,
    page_size: u32,
    limits: SchedulerLimits,
}

impl BatchAccumulator {
    pub(crate) fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
            total_pages: 0,
            page_size,
            limits,
        }
    }

    pub(crate) fn push(&mut self, req: PendingRequest) {
        let usage = request_capacity_usage(&req, self.page_size);
        self.total_tokens = self.total_tokens.saturating_add(usage.forward_tokens);
        self.total_pages = self.total_pages.saturating_add(usage.page_refs);
        self.requests.push(req);
    }

    pub(crate) fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
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
        if self.limits.max_forward_requests == 0 {
            return Some("driver max forward requests is zero".to_string());
        }
        None
    }

    pub(crate) fn would_exceed(&self, req: &PendingRequest) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        let usage = request_capacity_usage(req, self.page_size);
        self.requests.len() + 1 > self.limits.max_forward_requests
            || self.total_tokens.saturating_add(usage.forward_tokens)
                > self.limits.max_forward_tokens
            || self.total_pages.saturating_add(usage.page_refs) > self.limits.max_page_refs
    }

    pub(crate) fn is_full(&self) -> bool {
        self.requests.len() >= self.limits.max_forward_requests
            || self.total_tokens >= self.limits.max_forward_tokens
            || self.total_pages >= self.limits.max_page_refs
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.requests.len()
    }

    pub(crate) fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub(crate) fn take(&mut self) -> Vec<PendingRequest> {
        self.total_tokens = 0;
        self.total_pages = 0;
        std::mem::take(&mut self.requests)
    }
}

pub(crate) fn build_batch_request(
    requests: &mut [PendingRequest],
    page_size: u32,
    stats: &SchedulerStats,
) -> LaunchSubmission {
    if requests.len() == 1 && requests[0].prebuilt {
        // Nothing reads a prebuilt plan after submission, so move it into the
        // launch instead of cloning its geometry vectors. Its single program
        // owns every wire request row it carries (possibly zero for a
        // device-geometry fire, possibly several lanes for a wire beam).
        let req = &mut requests[0];
        let mut plan = std::mem::take(&mut req.request);
        let kv_translation = std::mem::take(&mut plan.kv_translation);
        let rows = plan.qo_indptr.len().saturating_sub(1) as u32;
        return LaunchSubmission {
            kv_translation_indptr: vec![0, kv_translation.len() as u32],
            kv_translation,
            program_row_indptr: vec![0, rows],
            plan,
            instance_ids: vec![req.instance_id],
            terminal_cells: vec![req.completion.terminal_cell_ptr()],
        };
    }
    let elide_decode_masks = requests.iter().all(|req| {
        req.request.single_token_mode
            && !req.request.has_user_mask
            && req.request.token_ids.len() <= 1
    });
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut batch_req = request::new_batched_forward_request_with_capacity(requests.len());
        let mut instance_ids = Vec::with_capacity(requests.len());
        let mut terminal_cells = Vec::with_capacity(requests.len());
        let mut kv_translation = Vec::new();
        let mut kv_translation_indptr = Vec::with_capacity(requests.len() + 1);
        kv_translation_indptr.push(0);
        // Batched fires contribute exactly one wire request row each (a
        // device-geometry fire's row is an empty placeholder).
        let program_row_indptr: Vec<u32> = (0..=requests.len() as u32).collect();
        for req in requests {
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            request::append_request_with_options(
                &mut batch_req,
                &req.request,
                &req.physical_page_ids,
                req.last_page_len,
                page_size,
                elide_decode_masks,
            );
            kv_translation.extend(req.request.kv_translation.iter().copied());
            kv_translation_indptr.push(kv_translation.len() as u32);
        }
        LaunchSubmission {
            plan: batch_req,
            instance_ids,
            terminal_cells,
            kv_translation,
            kv_translation_indptr,
            program_row_indptr,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{InstanceCompletion, LaunchPlan};

    fn pending(request: LaunchPlan, instance_id: u64, prebuilt: bool) -> PendingRequest {
        PendingRequest {
            physical_page_ids: request.kv_page_indices.clone(),
            last_page_len: 1,
            request,
            instance_id,
            completion: InstanceCompletion::new(instance_id, 0),
            program_identity_hashes: Vec::new(),
            pipeline_id: None,
            submitted_at_us: 0,
            prebuilt,
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
        assert_eq!(sub.plan.qo_indptr, vec![0, 1, 1, 2], "placeholder row is empty");
        assert_eq!(sub.plan.sampling_indptr, vec![0, 1, 1, 2]);
        assert_eq!(sub.kv_translation, vec![7, 8]);
        assert_eq!(sub.kv_translation_indptr, vec![0, 0, 2, 2]);
    }

    #[test]
    fn prebuilt_solo_owns_all_wire_rows() {
        // A prebuilt wire plan with two lanes: its single program owns both
        // rows. A device-geometry prebuilt (empty qo) owns zero rows.
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        let mut solo = [pending(two_lane, 1, true)];
        let sub = build_batch_request(&mut solo, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 2]);

        let dg = LaunchPlan::default();
        let mut solo_dg = [pending(dg, 2, true)];
        let sub = build_batch_request(&mut solo_dg, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 0]);
    }
}
