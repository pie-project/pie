//! The tensor-parallel rendezvous: the one plan every rank leaves with.
//!
//! The ranks of a TP group are THREADS OF ONE PROCESS, so this is an
//! in-process barrier rather than a collective. It is separate from
//! [`super::budget`] because it is not arithmetic about memory at all —
//! it is a synchronisation primitive that happens to reduce plans, and
//! the two were one file only because the C++ `plan.hpp` was.

use super::budget::CudaMemoryPlan;

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

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
        let mut reg = registry()
            .lock()
            .expect("planner rendezvous registry poisoned");
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
    use super::super::budget::PlannedForwardLimits;
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
        assert_eq!(
            a.kv_page_bytes, 4096,
            "page bytes must follow the page size down"
        );

        let mut b = plan(16, 4096, 4096, 256);
        b.min_into(&plan(32, 9000, 4096, 256));
        assert_eq!(b.kv_page_size, 16);
        assert_eq!(
            b.kv_page_bytes, 4096,
            "a larger page must not raise the byte count"
        );
    }

    #[test]
    fn a_single_rank_or_an_unkeyed_group_is_returned_unchanged() {
        let p = plan(16, 4096, 8192, 512);
        assert_eq!(tp_min_plan(1, "abc", &p), p);
        assert_eq!(tp_min_plan(4, "", &p), p);
    }

    #[test]
    fn every_rank_leaves_the_rendezvous_with_the_same_plan() {
        let key = format!(
            "test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let shapes = [
            (16, 4096, 8192, 512),
            (32, 9000, 4096, 256),
            (16, 5000, 16384, 1024),
        ];
        let handles: Vec<_> = shapes
            .iter()
            .map(|&(pg, pb, n, r)| {
                let key = key.clone();
                std::thread::spawn(move || tp_min_plan(3, &key, &plan(pg, pb, n, r)))
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("rank"))
            .collect();
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
