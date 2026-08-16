//! The memory budget a plan states: what the forward may carry and what that
//! costs, plus the fold that reconciles one plan per rank into one for the
//! group.

/// Upper bounds on per-fire shapes, sized once by the planner.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlannedForwardLimits {
    /// Most tokens one forward may carry.
    pub max_forward_tokens: i32,
    /// Most requests one forward may carry.
    pub max_forward_requests: i32,
    /// Most page-table entries one forward may reference.
    pub max_page_refs: i32,
    /// Rows the logit buffer must hold.
    pub max_logit_rows: i32,
    /// Rows the probability buffer must hold.
    pub max_prob_rows: i32,
    /// Bytes the custom-mask buffer must hold.
    pub max_custom_mask_bytes: i32,
    /// Rows the sampler must hold.
    pub max_sampler_rows: i32,
    /// Labels the log-probability path must hold.
    pub max_logprob_labels: i32,
}

/// One end-to-end memory plan for the CUDA driver.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaMemoryPlan {
    /// Tokens per KV page.
    pub kv_page_size: i32,
    /// Token capacity of the forward workspace.
    pub max_workspace_tokens: i32,
    /// Request capacity of one forward.
    pub max_requests: i32,
    /// Page-reference capacity of one forward.
    pub max_page_refs: i32,
    /// Device bytes one KV page costs, envelopes included.
    pub kv_page_bytes: u64,
    /// Float section of the attention workspace.
    pub attn_float_workspace_bytes: u64,
    /// Scratch the runtime-quantised GEMM path needs.
    pub runtime_quant_scratch_bytes: u64,
    /// Persistent per-fire input buffers.
    pub persistent_input_bytes: u64,
    /// The bounds the executor passes downstream.
    pub capacity: PlannedForwardLimits,
}

impl CudaMemoryPlan {
    /// Fold `src` into `self`, keeping whatever every rank can satisfy.
    ///
    /// Shape limits take the minimum (a bound only one rank meets is not a
    /// group bound); byte sizes take the maximum (allocations must fit the
    /// hungriest rank). `kv_page_size` is a discrete choice that must agree
    /// across ranks, so the smaller page wins and drags its own
    /// `kv_page_bytes` with it — taking min page size and max page bytes
    /// independently would describe a layout no rank has.
    pub fn min_into(&mut self, src: &Self) {
        if src.kv_page_size < self.kv_page_size {
            self.kv_page_size = src.kv_page_size;
            self.kv_page_bytes = src.kv_page_bytes;
        } else if src.kv_page_size == self.kv_page_size {
            self.kv_page_bytes = self.kv_page_bytes.max(src.kv_page_bytes);
        }
        self.max_workspace_tokens = self.max_workspace_tokens.min(src.max_workspace_tokens);
        self.max_requests = self.max_requests.min(src.max_requests);
        self.max_page_refs = self.max_page_refs.min(src.max_page_refs);
        self.attn_float_workspace_bytes = self
            .attn_float_workspace_bytes
            .max(src.attn_float_workspace_bytes);
        self.runtime_quant_scratch_bytes = self
            .runtime_quant_scratch_bytes
            .max(src.runtime_quant_scratch_bytes);
        self.persistent_input_bytes = self.persistent_input_bytes.max(src.persistent_input_bytes);

        let (d, s) = (&mut self.capacity, &src.capacity);
        d.max_forward_tokens = d.max_forward_tokens.min(s.max_forward_tokens);
        d.max_forward_requests = d.max_forward_requests.min(s.max_forward_requests);
        d.max_page_refs = d.max_page_refs.min(s.max_page_refs);
        d.max_logit_rows = d.max_logit_rows.min(s.max_logit_rows);
        d.max_prob_rows = d.max_prob_rows.min(s.max_prob_rows);
        d.max_custom_mask_bytes = d.max_custom_mask_bytes.min(s.max_custom_mask_bytes);
        d.max_sampler_rows = d.max_sampler_rows.min(s.max_sampler_rows);
        d.max_logprob_labels = d.max_logprob_labels.min(s.max_logprob_labels);
    }
}
