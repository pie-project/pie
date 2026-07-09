//! Batch assembly: capacity accounting + the dense-batch accumulator.
//!
//! `BatchAccumulator` folds `PendingRequest`s into one driver batch under the
//! `SchedulerLimits` caps (`RequestCapacityUsage` per request), and the
//! `prepare_pending_*` helpers resolve a pending request into wire form. The
//! scheduler run loop drives this; the fire decision lives in `policy`.

use crate::driver::SchedulerLimits;
use super::scheduler::{Completion, PendingRequest};
use super::stats::SchedulerStats;
use super::request;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
    pub(crate) logit_rows: usize,
    pub(crate) prob_rows: usize,
    pub(crate) sampler_rows: usize,
    pub(crate) logprob_labels: usize,
    pub(crate) user_custom_mask_bytes: usize,
    pub(crate) spec_custom_mask_bytes: usize,
    pub(crate) has_spec_drafts: bool,
    pub(crate) has_dense_logit_requirement: bool,
    pub(crate) has_prob_sampling: bool,
    pub(crate) is_single_token_decode: bool,
    pub(crate) all_samplers_token: bool,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let input_tokens = req.request.token_ids.len();
    let spec_tokens = req.request.spec_token_ids.len();
    let forward_tokens = input_tokens.saturating_add(spec_tokens);
    let mut sampler_rows = req.request.sampling_indices.len();
    if spec_tokens > 0 {
        sampler_rows = sampler_rows.saturating_add(spec_tokens.saturating_add(1));
    }
    let mut all_samplers_token = true;
    let mut has_prob_sampling = false;
    for i in 0..req.request.n_samplers() {
        let sampler = req.request.sampler_at(i).expect("slot in range");
        if !is_token_sampler(&sampler) {
            all_samplers_token = false;
        }
        if sampler_needs_prob_rows(&sampler) {
            has_prob_sampling = true;
        }
    }
    let has_dense_logit_requirement = req.request.has_user_mask
        || !req.request.logit_masks.is_empty()
        || spec_tokens > 0
        || !all_samplers_token;
    let is_single_token_decode = input_tokens == 1
        && spec_tokens == 0
        && req.request.single_token_mode
        && !req.request.has_user_mask;
    let page_refs = req.physical_page_ids.len();
    let spec_custom_mask_bytes =
        packed_mask_bytes(forward_tokens, page_refs, req.last_page_len, page_size);
    let user_custom_mask_bytes = if req.request.has_user_mask && input_tokens > 1 {
        packed_mask_bytes(input_tokens, page_refs, req.last_page_len, page_size)
    } else {
        0
    };

    RequestCapacityUsage {
        forward_tokens,
        page_refs,
        logit_rows: 0,
        prob_rows: 0,
        sampler_rows,
        logprob_labels: request_logprob_labels(&req.request),
        user_custom_mask_bytes,
        spec_custom_mask_bytes,
        has_spec_drafts: spec_tokens > 0,
        has_dense_logit_requirement,
        has_prob_sampling,
        is_single_token_decode,
        all_samplers_token,
    }
}

pub(crate) fn is_token_sampler(sampler: &pie_driver_abi::Sampler) -> bool {
    matches!(
        sampler,
        pie_driver_abi::Sampler::Multinomial { .. }
            | pie_driver_abi::Sampler::TopK { .. }
            | pie_driver_abi::Sampler::TopP { .. }
            | pie_driver_abi::Sampler::MinP { .. }
            | pie_driver_abi::Sampler::TopKTopP { .. }
    )
}

pub(crate) fn sampler_needs_prob_rows(sampler: &pie_driver_abi::Sampler) -> bool {
    match sampler {
        pie_driver_abi::Sampler::TopK { temperature, k } => *temperature > 0.0 && *k > 0,
        pie_driver_abi::Sampler::TopP { temperature, p } => *temperature > 0.0 && *p < 1.0,
        pie_driver_abi::Sampler::TopKTopP { temperature, k, p } => {
            *temperature > 0.0 && (*k > 0 || *p < 1.0)
        }
        _ => false,
    }
}

pub(crate) fn packed_mask_bytes(
    query_tokens: usize,
    page_refs: usize,
    last_page_len: u32,
    page_size: u32,
) -> usize {
    if query_tokens == 0 || page_refs == 0 || page_size == 0 {
        return 0;
    }
    let kv_len = page_refs
        .saturating_sub(1)
        .saturating_mul(page_size as usize)
        .saturating_add(last_page_len as usize);
    query_tokens.saturating_mul(kv_len).saturating_add(7) / 8
}

pub(crate) fn request_logprob_labels(req: &pie_driver_abi::ForwardRequest) -> usize {
    // Logprob → 1 label, Logprobs → its token_ids count (from the CSR), else 0.
    (0..req.n_samplers())
        .map(|s| match req.sampler_kinds[s] {
            pie_driver_abi::PIE_SAMPLER_LOGPROB => 1,
            pie_driver_abi::PIE_SAMPLER_LOGPROBS => (req.sampler_token_ids_indptr[s + 1]
                - req.sampler_token_ids_indptr[s])
                as usize,
            _ => 0,
        })
        .sum()
}

// =============================================================================
// BatchAccumulator
// =============================================================================

/// Accumulates pending requests into a batch.
///
/// Pure synchronous struct — no async, no channels. Can be tested
/// independently from the scheduling loop.
pub(crate) struct BatchAccumulator {
    requests: Vec<PendingRequest>,
    total_tokens: usize,
    total_pages: usize,
    total_logit_rows: usize,
    total_prob_rows: usize,
    total_sampler_rows: usize,
    total_logprob_labels: usize,
    total_user_custom_mask_bytes: usize,
    total_spec_custom_mask_bytes: usize,
    has_spec_drafts: bool,
    has_dense_logit_requirement: bool,
    has_prob_sampling: bool,
    all_single_token_decode: bool,
    all_samplers_token: bool,
    page_size: u32,
    limits: SchedulerLimits,
}

impl BatchAccumulator {
    pub(crate) fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
            total_pages: 0,
            total_logit_rows: 0,
            total_prob_rows: 0,
            total_sampler_rows: 0,
            total_logprob_labels: 0,
            total_user_custom_mask_bytes: 0,
            total_spec_custom_mask_bytes: 0,
            has_spec_drafts: false,
            has_dense_logit_requirement: false,
            has_prob_sampling: false,
            all_single_token_decode: true,
            all_samplers_token: true,
            page_size,
            limits,
        }
    }

    pub(crate) fn projected_rows(
        &self,
        extra: Option<&RequestCapacityUsage>,
    ) -> (usize, usize, bool, bool, bool) {
        let total_tokens = self
            .total_tokens
            .saturating_add(extra.map(|usage| usage.forward_tokens).unwrap_or(0));
        let total_sampler_rows = self
            .total_sampler_rows
            .saturating_add(extra.map(|usage| usage.sampler_rows).unwrap_or(0));
        let has_dense_logit_requirement = self.has_dense_logit_requirement
            || extra
                .map(|usage| usage.has_dense_logit_requirement)
                .unwrap_or(false);
        let has_prob_sampling =
            self.has_prob_sampling || extra.map(|usage| usage.has_prob_sampling).unwrap_or(false);
        let all_samplers_token =
            self.all_samplers_token && extra.map(|usage| usage.all_samplers_token).unwrap_or(true);
        let all_single_token_decode = self.all_single_token_decode
            && extra
                .map(|usage| usage.is_single_token_decode)
                .unwrap_or(true);
        let compact_logit_rows = !all_single_token_decode
            && !has_dense_logit_requirement
            && !has_prob_sampling
            && all_samplers_token
            && total_sampler_rows > 0
            && total_sampler_rows < total_tokens;
        let logit_rows = if total_sampler_rows == 0 {
            // Pure KV-fill / encode fire (e.g. a multimodal image-token
            // prefill): no token is sampled, so the driver computes no
            // logits. Without this, the fire would project `total_tokens`
            // logit rows and trip `max_logit_rows` for large image spans.
            0
        } else if compact_logit_rows {
            total_sampler_rows
        } else {
            total_tokens
        };
        let prob_rows = if has_prob_sampling { total_tokens } else { 0 };
        (
            logit_rows,
            prob_rows,
            has_dense_logit_requirement,
            has_prob_sampling,
            all_single_token_decode,
        )
    }

    pub(crate) fn push(&mut self, req: PendingRequest) {
        let usage = request_capacity_usage(&req, self.page_size);
        self.push_with(req, usage);
    }

    pub(crate) fn push_with(&mut self, req: PendingRequest, mut usage: RequestCapacityUsage) {
        let (logit_rows, prob_rows, _, _, _) = self.projected_rows(Some(&usage));
        usage.logit_rows = logit_rows;
        usage.prob_rows = prob_rows;
        self.total_tokens = self.total_tokens.saturating_add(usage.forward_tokens);
        self.total_pages = self.total_pages.saturating_add(usage.page_refs);
        self.total_logit_rows = usage.logit_rows;
        self.total_prob_rows = usage.prob_rows;
        self.total_sampler_rows = self.total_sampler_rows.saturating_add(usage.sampler_rows);
        self.total_logprob_labels = self
            .total_logprob_labels
            .saturating_add(usage.logprob_labels);
        self.total_user_custom_mask_bytes = self
            .total_user_custom_mask_bytes
            .saturating_add(usage.user_custom_mask_bytes);
        self.total_spec_custom_mask_bytes = self
            .total_spec_custom_mask_bytes
            .saturating_add(usage.spec_custom_mask_bytes);
        self.has_spec_drafts |= usage.has_spec_drafts;
        self.has_dense_logit_requirement |= usage.has_dense_logit_requirement;
        self.has_prob_sampling |= usage.has_prob_sampling;
        self.all_single_token_decode &= usage.is_single_token_decode;
        self.all_samplers_token &= usage.all_samplers_token;
        self.requests.push(req);
    }

    pub(crate) fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
        let (logit_rows, prob_rows, _, _, _) =
            BatchAccumulator::new(self.limits, self.page_size).projected_rows(Some(&usage));
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

        if logit_rows > self.limits.max_logit_rows {
            return Some(format!(
                "forward request needs {} logit rows, exceeding driver limit {}",
                logit_rows, self.limits.max_logit_rows
            ));
        }

        if prob_rows > self.limits.max_prob_rows {
            return Some(format!(
                "forward request needs {} probability rows, exceeding driver limit {}",
                prob_rows, self.limits.max_prob_rows
            ));
        }

        if usage.sampler_rows > self.limits.max_sampler_rows {
            return Some(format!(
                "forward request has {} sampler rows, exceeding driver limit {}",
                usage.sampler_rows, self.limits.max_sampler_rows
            ));
        }

        if usage.logprob_labels > self.limits.max_logprob_labels {
            return Some(format!(
                "forward request has {} logprob labels, exceeding driver limit {}",
                usage.logprob_labels, self.limits.max_logprob_labels
            ));
        }

        let custom_mask_bytes = if usage.has_spec_drafts {
            usage.spec_custom_mask_bytes
        } else {
            usage.user_custom_mask_bytes
        };
        if custom_mask_bytes > self.limits.max_custom_mask_bytes {
            return Some(format!(
                "forward request needs {custom_mask_bytes} custom mask bytes, exceeding driver limit {}",
                self.limits.max_custom_mask_bytes
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
        self.would_exceed_with(&usage)
    }


    pub(crate) fn would_exceed_with(&self, usage: &RequestCapacityUsage) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        // rs_cache spec-decode (MTP for hybrid GDN models) no longer needs a
        // per-batch cap: the driver runs a frozen verify (committed slot stays
        // at its pre-verify value) and a single batched repair forward over
        // [input | accepted] to advance state. There is no per-request snapshot
        // buffer, so rs-spec batches grow to the normal forward limits below,
        // exactly like non-spec batches.
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(usage));
        self.requests.len() + 1 > self.limits.max_forward_requests
            || self.total_tokens.saturating_add(usage.forward_tokens)
                > self.limits.max_forward_tokens
            || self.total_pages.saturating_add(usage.page_refs) > self.limits.max_page_refs
            || next_logit_rows > self.limits.max_logit_rows
            || next_prob_rows > self.limits.max_prob_rows
            || self.total_sampler_rows.saturating_add(usage.sampler_rows)
                > self.limits.max_sampler_rows
            || self
                .total_logprob_labels
                .saturating_add(usage.logprob_labels)
                > self.limits.max_logprob_labels
            || next_custom_mask_bytes > self.limits.max_custom_mask_bytes
    }

    pub(crate) fn would_exceed_reason(&self, req: &PendingRequest) -> Option<String> {
        if self.requests.is_empty() {
            return None;
        }
        let usage = request_capacity_usage(req, self.page_size);
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(&usage));
        let checks = [
            (
                "requests",
                self.requests.len().saturating_add(1),
                self.limits.max_forward_requests,
            ),
            (
                "tokens",
                self.total_tokens.saturating_add(usage.forward_tokens),
                self.limits.max_forward_tokens,
            ),
            (
                "pages",
                self.total_pages.saturating_add(usage.page_refs),
                self.limits.max_page_refs,
            ),
            ("logit_rows", next_logit_rows, self.limits.max_logit_rows),
            ("prob_rows", next_prob_rows, self.limits.max_prob_rows),
            (
                "sampler_rows",
                self.total_sampler_rows.saturating_add(usage.sampler_rows),
                self.limits.max_sampler_rows,
            ),
            (
                "logprob_labels",
                self.total_logprob_labels
                    .saturating_add(usage.logprob_labels),
                self.limits.max_logprob_labels,
            ),
            (
                "custom_mask_bytes",
                next_custom_mask_bytes,
                self.limits.max_custom_mask_bytes,
            ),
        ];
        checks
            .into_iter()
            .find(|(_, have, limit)| have > limit)
            .map(|(name, have, limit)| {
                format!(
                    "{name} {have}>{limit} pending_tokens={} pending_pages={} pending_sampler_rows={} pending_has_spec={} pending_dense={} pending_prob={}",
                    usage.forward_tokens,
                    usage.page_refs,
                    usage.sampler_rows,
                    usage.has_spec_drafts,
                    usage.has_dense_logit_requirement,
                    usage.has_prob_sampling,
                )
            })
    }

    pub(crate) fn is_full(&self) -> bool {
        let active_custom_mask_bytes = if self.has_spec_drafts {
            self.total_spec_custom_mask_bytes
        } else {
            self.total_user_custom_mask_bytes
        };
        self.requests.len() >= self.limits.max_forward_requests
            // rs-spec batches (frozen verify + batched repair) carry no
            // per-request buffer, so they fill to the normal forward limits
            // like any other batch — no rs-spec-specific early fire.
            || self.total_tokens >= self.limits.max_forward_tokens
            || self.total_pages >= self.limits.max_page_refs
            || self.total_logit_rows >= self.limits.max_logit_rows
            || self.total_prob_rows >= self.limits.max_prob_rows
            || self.total_sampler_rows >= self.limits.max_sampler_rows
            || self.total_logprob_labels >= self.limits.max_logprob_labels
            || active_custom_mask_bytes >= self.limits.max_custom_mask_bytes
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// True if any accumulated request is a G2 prebuilt-passthrough fire (a
    /// complete pre-assembled multi-lane batch that must fire solo). Forces a
    /// batch boundary so a prebuilt fire is never co-batched.
    pub(crate) fn has_prebuilt(&self) -> bool {
        self.requests.iter().any(|r| r.prebuilt)
    }

    pub(crate) fn len(&self) -> usize {
        self.requests.len()
    }

    pub(crate) fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub(crate) fn should_prefill_coalesce(&self) -> bool {
        !self.has_spec_drafts && self.total_tokens > self.requests.len()
    }

    pub(crate) fn take(&mut self) -> Vec<PendingRequest> {
        self.total_tokens = 0;
        self.total_pages = 0;
        self.total_logit_rows = 0;
        self.total_prob_rows = 0;
        self.total_sampler_rows = 0;
        self.total_logprob_labels = 0;
        self.total_user_custom_mask_bytes = 0;
        self.total_spec_custom_mask_bytes = 0;
        self.has_spec_drafts = false;
        self.has_dense_logit_requirement = false;
        self.has_prob_sampling = false;
        self.all_single_token_decode = true;
        self.all_samplers_token = true;
        std::mem::take(&mut self.requests)
    }
}

pub(crate) fn prepare_pending_for_batch(
    batch: &BatchAccumulator,
    pending: PendingRequest,
) -> Option<PendingRequest> {
    prepare_pending_with_usage(batch, pending).map(|(p, _)| p)
}

/// Same as `prepare_pending_for_batch` but also returns the computed
/// `RequestCapacityUsage`, avoiding a recompute when the caller will
/// immediately consult `would_exceed_with` + `push_with`.
///
/// The pure-decode fast path skips `maybe_start_chunking` and
/// `single_request_limit_error`: for `single_token_mode` requests with
/// 1 token, no spec drafts, no user mask, and no logit masks, both are
/// no-ops (chunk_size is never reached; per-request limits hold trivially
/// given the BatchAccumulator's `would_exceed_with` check will still gate
/// page_refs / sampler_rows etc. when batching).
pub(crate) fn prepare_pending_with_usage(
    batch: &BatchAccumulator,
    pending: PendingRequest,
) -> Option<(PendingRequest, RequestCapacityUsage)> {
    if is_pure_decode_pending(&pending) {
        let usage = request_capacity_usage(&pending, batch.page_size);
        let limits = batch.limits;
        // Fields that COULD still trip the single-request limit for decode:
        // page_refs (long-context decode) and logprob_labels. Token/sampler/
        // mask limits hold trivially for 1-token single_token_mode requests.
        if usage.page_refs <= limits.max_page_refs
            && usage.logprob_labels <= limits.max_logprob_labels
            && limits.max_forward_requests > 0
        {
            return Some((pending, usage));
        }
        // Fall through to the slow path so the proper error message is
        // surfaced via `single_request_limit_error`.
    }
    let pending = match pending.maybe_start_chunking(batch.limits, batch.page_size) {
        Ok(pending) => pending,
        Err((pending, msg)) => {
            pending.send_error(msg);
            return None;
        }
    };
    if let Some(msg) = batch.single_request_limit_error(&pending) {
        pending.send_error(msg);
        return None;
    }
    let usage = request_capacity_usage(&pending, batch.page_size);
    Some((pending, usage))
}

#[inline]
pub(crate) fn is_pure_decode_pending(p: &PendingRequest) -> bool {
    matches!(&p.completion, Completion::Direct(_)) && p.request.token_ids.len() == 1
        && p.request.spec_token_ids.is_empty()
        && p.request.single_token_mode
        && !p.request.has_user_mask
        && p.request.logit_masks.is_empty()
}

// =============================================================================
// Batched ForwardRequest assembly
// =============================================================================

/// Build the batched `pie_driver_abi::ForwardRequest` by folding each
/// per-request shape into one batch. Runs on the scheduler thread (so it
/// overlaps the GPU of any in-flight batch); the caller then enqueues it in
/// fire-order via [`driver::fire_batch_deferred`].
pub(crate) fn build_batch_request(
    requests: &[PendingRequest],
    page_size: u32,
    stats: &SchedulerStats,
) -> pie_driver_abi::ForwardRequest {
    // Build batched request — a single `pie_driver_abi::ForwardRequest`
    // populated by folding each per-request shape into the batch.
    let elide_decode_masks = requests.iter().all(|req| {
        req.request.single_token_mode
            && !req.request.has_user_mask
            && req.request.token_ids.len() <= 1
            && req.request.spec_token_ids.is_empty()
    });
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut batch_req =
            request::new_batched_forward_request_with_capacity(requests.len());
        for req in requests {
            request::append_request_with_options(
                &mut batch_req,
                &req.request,
                &req.physical_page_ids,
                req.last_page_len,
                page_size,
                elide_decode_masks,
            );
        }
        batch_req
    })
}
