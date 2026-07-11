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
        // launch instead of cloning its geometry vectors.
        let req = &mut requests[0];
        return LaunchSubmission {
            plan: std::mem::take(&mut req.request),
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
        }
        LaunchSubmission {
            plan: batch_req,
            instance_ids,
            terminal_cells,
        }
    })
}
