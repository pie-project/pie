//! The plan the memory planner produces, and the tensor-parallel rendezvous
//! that reconciles one per rank into one for the group.
//!
//! Ported from the `CudaMemoryPlan` / `PlannedForwardLimits` structs in
//! `store/memory_planner.hpp` and the `min_into` / `tp_min_plan` pair in
//! `store/memory_planner.cpp`.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Upper bounds on per-fire shapes.
///
/// Sized once by the planner so persistent device buffers can be reserved
/// ahead of time and shared across every call.
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
    /// The direction is **not** uniform, and that is the whole content of this
    /// function:
    ///
    ///   * *Shape* limits take the **minimum**. A bound only one rank can meet
    ///     is not a bound the group can meet.
    ///   * *Byte* sizes take the **maximum**. These are allocations, so the
    ///     group must reserve enough for its hungriest rank.
    ///
    /// `kv_page_size` is neither: it is a discrete choice that has to agree
    /// across ranks, so the smaller page wins and drags its own
    /// `kv_page_bytes` along with it. Taking the minimum page size and the
    /// maximum page bytes independently would describe a layout no rank has.
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
        self.attn_float_workspace_bytes =
            self.attn_float_workspace_bytes.max(src.attn_float_workspace_bytes);
        self.runtime_quant_scratch_bytes =
            self.runtime_quant_scratch_bytes.max(src.runtime_quant_scratch_bytes);
        self.persistent_input_bytes =
            self.persistent_input_bytes.max(src.persistent_input_bytes);

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

/// Shared state for one tensor-parallel group's rendezvous.
struct Rendezvous {
    inner: Mutex<RendezvousState>,
    ready: Condvar,
}

struct RendezvousState {
    arrived: i32,
    ready: bool,
    plan: CudaMemoryPlan,
}

/// The registry keyed by NCCL unique id, matching the C++'s function-static
/// `unordered_map`.
fn registry() -> &'static Mutex<HashMap<String, Arc<Rendezvous>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Rendezvous>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Reduce one rank's plan to the plan the whole tensor-parallel group can run.
///
/// The ranks of a TP group are **threads of one process**, so this is an
/// in-process barrier rather than a collective: every rank contributes its
/// plan, the last to arrive releases the rest, and all of them leave with the
/// same answer. They must, because the plan sizes buffers whose shapes appear
/// in collective operations -- two ranks disagreeing about `max_requests`
/// deadlock at the first all-reduce rather than failing here.
///
/// Returns `local` unchanged when there is nothing to reconcile: a single rank,
/// or a group with no `nccl_unique_id_hex` to key on.
///
/// # Panics
///
/// If the registry lock is poisoned by a rank that panicked mid-rendezvous. A
/// poisoned barrier cannot be completed -- the ranks still waiting will never
/// be released -- so there is nothing to recover to.
#[must_use]
pub fn tp_min_plan(
    tp_size: i32,
    nccl_unique_id_hex: &str,
    local: &CudaMemoryPlan,
) -> CudaMemoryPlan {
    if tp_size <= 1 || nccl_unique_id_hex.is_empty() {
        return local.clone();
    }

    let shared = {
        let mut reg = registry().lock().expect("planner rendezvous registry poisoned");
        Arc::clone(reg.entry(nccl_unique_id_hex.to_owned()).or_insert_with(|| {
            Arc::new(Rendezvous {
                inner: Mutex::new(RendezvousState {
                    arrived: 0,
                    ready: false,
                    plan: CudaMemoryPlan::default(),
                }),
                ready: Condvar::new(),
            })
        }))
    };

    let mut state = shared.inner.lock().expect("planner rendezvous poisoned");
    if state.arrived == 0 {
        state.plan = local.clone();
    } else {
        let src = local.clone();
        state.plan.min_into(&src);
    }
    state.arrived += 1;
    if state.arrived >= tp_size {
        state.ready = true;
        shared.ready.notify_all();
    } else {
        // `wait_while` rather than a bare `wait`: a condvar may wake
        // spuriously, and a rank that returned from a spurious wake would
        // read a half-reduced plan.
        state = shared
            .ready
            .wait_while(state, |s| !s.ready)
            .expect("planner rendezvous poisoned");
    }
    state.plan.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(page: i32, page_bytes: u64, n: i32, r: i32) -> CudaMemoryPlan {
        CudaMemoryPlan {
            kv_page_size: page,
            max_workspace_tokens: n,
            max_requests: r,
            max_page_refs: r * 512,
            kv_page_bytes: page_bytes,
            attn_float_workspace_bytes: u64::from(n.unsigned_abs()) * 16,
            runtime_quant_scratch_bytes: u64::from(n.unsigned_abs()),
            persistent_input_bytes: u64::from(r.unsigned_abs()) * 64,
            capacity: PlannedForwardLimits {
                max_forward_tokens: n,
                max_forward_requests: r,
                max_page_refs: r * 512,
                max_logit_rows: r,
                max_prob_rows: r,
                max_custom_mask_bytes: n * 8,
                max_sampler_rows: r,
                max_logprob_labels: r,
            },
        }
    }

    #[test]
    fn shapes_take_the_minimum_and_allocations_the_maximum() {
        let mut a = plan(16, 4096, 8192, 512);
        a.min_into(&plan(16, 8192, 4096, 256));
        // Shapes: the smaller of each.
        assert_eq!(a.max_workspace_tokens, 4096);
        assert_eq!(a.max_requests, 256);
        assert_eq!(a.capacity.max_custom_mask_bytes, 4096 * 8);
        // Allocations: the larger of each.
        assert_eq!(a.attn_float_workspace_bytes, 8192 * 16);
        assert_eq!(a.persistent_input_bytes, 512 * 64);
        // Equal page size keeps the larger page bytes.
        assert_eq!(a.kv_page_size, 16);
        assert_eq!(a.kv_page_bytes, 8192);
    }

    #[test]
    fn a_smaller_page_brings_its_own_byte_count() {
        // The one field pair that must move together: taking min(page_size)
        // and max(page_bytes) independently would describe a layout no rank
        // proposed.
        let mut a = plan(32, 9000, 4096, 256);
        a.min_into(&plan(16, 4096, 4096, 256));
        assert_eq!(a.kv_page_size, 16);
        assert_eq!(a.kv_page_bytes, 4096, "page bytes must follow the page size down");

        let mut b = plan(16, 4096, 4096, 256);
        b.min_into(&plan(32, 9000, 4096, 256));
        assert_eq!(b.kv_page_size, 16);
        assert_eq!(b.kv_page_bytes, 4096, "a larger page must not raise the byte count");
    }

    #[test]
    fn a_single_rank_or_an_unkeyed_group_is_returned_unchanged() {
        let p = plan(16, 4096, 8192, 512);
        assert_eq!(tp_min_plan(1, "abc", &p), p);
        assert_eq!(tp_min_plan(4, "", &p), p);
    }

    #[test]
    fn every_rank_leaves_the_rendezvous_with_the_same_plan() {
        let key = format!("test-{}-{:?}", std::process::id(), std::thread::current().id());
        let shapes = [(16, 4096, 8192, 512), (32, 9000, 4096, 256), (16, 5000, 16384, 1024)];
        let handles: Vec<_> = shapes
            .iter()
            .map(|&(pg, pb, n, r)| {
                let key = key.clone();
                std::thread::spawn(move || tp_min_plan(3, &key, &plan(pg, pb, n, r)))
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().expect("rank")).collect();
        assert!(
            results.windows(2).all(|w| w[0] == w[1]),
            "ranks disagreed, which deadlocks at the first collective"
        );
        let got = &results[0];
        assert_eq!(got.max_workspace_tokens, 4096);
        assert_eq!(got.max_requests, 256);
        assert_eq!(got.kv_page_size, 16);
        // 5000 wins over 4096 because both are 16-token pages.
        assert_eq!(got.kv_page_bytes, 5000);
        assert_eq!(got.attn_float_workspace_bytes, 16384 * 16);
    }
}
