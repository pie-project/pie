//! The tensor-parallel rendezvous: the one plan every rank leaves with, and
//! the host all-gather the P2P plane bootstraps through.
//!
//! The ranks of a TP group are THREADS OF ONE PROCESS, so both of these are
//! in-process barriers rather than collectives. This file is separate from
//! [`super::budget`] because it is not arithmetic about memory at all —
//! it is a synchronisation primitive that happens to reduce plans, and
//! the two were one file only because the C++ `plan.hpp` was.
//!
//! # Two rendezvous, one key
//!
//! [`tp_min_plan`] reduces plans; [`tp_host_allgather`] concatenates bytes.
//! They are separate registries keyed on the same string, and they must be:
//! the plan reduction happens ONCE per rank and the all-gather happens once
//! per bootstrap exchange (`CustomAllReduce` runs at least four, and one more
//! per `register_buffer`), so a single barrier could not serve both without
//! the plan's arrival count and the exchange's round count meaning the same
//! thing.
//!
//! **`tp_host_allgather` is the `HostAllgather::gather` half of
//! `fire::all_reduce::CustomAllReduce::new`.** That constructor reads exactly
//! two things off a collective — the world size, and one bootstrap-time
//! all-gather of pointers or IPC handles — and this is the second, for the
//! deployment shape this driver has: N ranks, N threads, one process.

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

// ── the byte all-gather ──────────────────────────────────────────────────

/// Shared state for one tensor-parallel group's host all-gather.
struct Exchange {
    inner: Mutex<ExchangeState>,
    /// Signalled when a round closes. Waiters test [`ExchangeState::round`]
    /// rather than a flag, so a round that closes and reopens between a
    /// notify and a wake cannot be missed.
    departed: Condvar,
}

struct ExchangeState {
    /// How many rounds have CLOSED. A rank waits for this to move past the
    /// value it read on arrival.
    round: u64,
    /// How many ranks have contributed to the round now open.
    arrived: usize,
    /// The round now open, one slot per rank.
    slots: Vec<Vec<u8>>,
    /// The last round to close, whole.
    ///
    /// **Publishing rather than reading [`Self::slots`] in place is what makes
    /// this safe to call in a loop.** The rank that closes a round returns
    /// without waiting and may re-enter for the next one before a slower rank
    /// has copied anything out; reading `slots` directly, that rank would see
    /// the next round's contribution in the previous round's answer. It cannot
    /// see a stale `published`, because round N+1 cannot close until every
    /// rank has ENTERED it, and a rank still waiting in round N has not.
    published: Arc<Vec<Vec<u8>>>,
}

/// The all-gather registry. Separate from [`registry`] — see the module
/// header for why one barrier cannot serve both.
fn exchanges() -> &'static Mutex<HashMap<String, Arc<Exchange>>> {
    static EXCHANGES: OnceLock<Mutex<HashMap<String, Arc<Exchange>>>> = OnceLock::new();
    EXCHANGES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// This rank's half of its group's host all-gather, or `None` when there is
/// no group to gather across.
///
/// The returned closure is the `HostAllgather::gather` contract verbatim:
/// `send` is this rank's contribution, `recv` is `send.len() * tp_size` bytes
/// and comes back **rank-major**. It may be called any number of times; each
/// call is one round, and every rank must make the same number of calls with
/// the same `send.len()` in the same order — which is the same discipline the
/// collective it bootstraps demands, so it is not a new obligation.
///
/// `None` for a single rank (nothing to gather) and for an empty
/// `group_key` (nothing to gather ON). The second is not a degenerate case
/// that can be served by copying `send` into `recv`: a group of two whose
/// ranks cannot find each other would then bootstrap a plane pointing at its
/// own memory twice, and every reduction it ran would be silently a no-op.
/// The caller has to be told, so it is an `Option` rather than a fallback.
///
/// # Panics
///
/// The returned closure panics if the exchange registry lock is poisoned by a
/// rank that panicked mid-round, for [`tp_min_plan`]'s reason: the ranks still
/// waiting will never be released, so there is nothing to recover to.
#[must_use]
pub fn tp_host_allgather(
    tp_size: i32,
    group_key: &str,
    rank: i32,
) -> Option<Arc<dyn Fn(&[u8], &mut [u8]) + Send + Sync>> {
    if tp_size <= 1 || group_key.is_empty() {
        return None;
    }
    let world = tp_size.unsigned_abs() as usize;
    let me = rank.max(0).unsigned_abs() as usize;
    if me >= world {
        return None;
    }

    let shared = {
        let mut reg = exchanges().lock().expect("tp all-gather registry poisoned");
        Arc::clone(reg.entry(group_key.to_owned()).or_insert_with(|| {
            Arc::new(Exchange {
                inner: Mutex::new(ExchangeState {
                    round: 0,
                    arrived: 0,
                    slots: vec![Vec::new(); world],
                    published: Arc::new(Vec::new()),
                }),
                departed: Condvar::new(),
            })
        }))
    };

    Some(Arc::new(move |send: &[u8], recv: &mut [u8]| {
        gather_round(&shared, world, me, send, recv);
    }))
}

/// One round of [`tp_host_allgather`].
fn gather_round(shared: &Exchange, world: usize, me: usize, send: &[u8], recv: &mut [u8]) {
    let published = {
        let mut state = shared.inner.lock().expect("tp all-gather poisoned");
        let opened = state.round;
        // A group whose first caller sized the slot list is the only one that
        // ever sizes it; a later rank with a different `tp_size` on the same
        // key is a misconfiguration, and growing here rather than indexing
        // out of bounds keeps it a wrong answer the caller can see instead of
        // a panic in a barrier.
        if state.slots.len() < world {
            state.slots.resize(world, Vec::new());
        }
        state.slots[me] = send.to_vec();
        state.arrived += 1;
        if state.arrived >= world {
            state.arrived = 0;
            state.round = opened.wrapping_add(1);
            let closed = std::mem::replace(&mut state.slots, vec![Vec::new(); world]);
            state.published = Arc::new(closed);
            shared.departed.notify_all();
        } else {
            // `wait_while` on the round counter, not on a flag: a condvar may
            // wake spuriously, and a rank that returned from a spurious wake
            // would read the PREVIOUS round's answer as this round's.
            state = shared
                .departed
                .wait_while(state, |s| s.round == opened)
                .expect("tp all-gather poisoned");
        }
        Arc::clone(&state.published)
    };

    // Rank-major, and sized by what each rank actually sent rather than by
    // `send.len()`, so a caller that mis-sized `recv` truncates instead of
    // writing past it.
    let stride = send.len();
    for (r, slot) in published.iter().enumerate() {
        let at = r * stride;
        let end = (at + slot.len()).min(recv.len());
        if at >= end {
            continue;
        }
        recv[at..end].copy_from_slice(&slot[..end - at]);
    }
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

    fn unique_key(what: &str) -> String {
        format!("test-{what}-{}-{:?}", std::process::id(), std::thread::current().id())
    }

    #[test]
    fn a_group_of_one_and_an_unkeyed_group_have_no_all_gather() {
        assert!(tp_host_allgather(1, "k", 0).is_none(), "one rank has nothing to gather");
        assert!(tp_host_allgather(4, "", 0).is_none(), "no key is nothing to gather ON");
        assert!(tp_host_allgather(2, "k", 7).is_none(), "a rank outside its own group");
    }

    /// The `HostAllgather` contract: `recv` is `send.len() * world` bytes and
    /// comes back rank-major.
    #[test]
    fn every_rank_leaves_with_every_ranks_bytes_in_rank_order() {
        let key = unique_key("gather");
        let world = 4;
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let key = key.clone();
                std::thread::spawn(move || {
                    let gather =
                        tp_host_allgather(world, &key, rank).expect("a group of four gathers");
                    let mut recv = vec![0u8; 8 * world.unsigned_abs() as usize];
                    gather(&(0xAA00_u64 + rank.unsigned_abs() as u64).to_ne_bytes(), &mut recv);
                    recv
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().expect("rank")).collect();
        assert!(results.windows(2).all(|w| w[0] == w[1]), "every rank sees the same answer");
        for r in 0..world.unsigned_abs() as usize {
            let mut word = [0u8; 8];
            word.copy_from_slice(&results[0][r * 8..r * 8 + 8]);
            assert_eq!(u64::from_ne_bytes(word), 0xAA00 + r as u64, "rank-major, rank {r}");
        }
    }

    /// Round N's answer must not pick up round N+1's contributions.
    ///
    /// This is the property `ExchangeState::published` exists for, and the
    /// one a barrier that read its slots in place gets wrong: the rank that
    /// CLOSES a round returns without waiting, so it can re-enter and
    /// overwrite its own slot before a slower rank has copied anything out.
    /// Ten rounds with a distinct payload per round is enough to hit it —
    /// each rank checks every round's answer, so a single leaked byte fails.
    #[test]
    fn consecutive_rounds_do_not_bleed_into_each_other() {
        let key = unique_key("rounds");
        let world = 3;
        let rounds = 10u64;
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let key = key.clone();
                std::thread::spawn(move || {
                    let gather = tp_host_allgather(world, &key, rank).expect("a group of three");
                    for round in 0..rounds {
                        let mine = round * 100 + rank.unsigned_abs() as u64;
                        let mut recv = vec![0u8; 8 * world.unsigned_abs() as usize];
                        gather(&mine.to_ne_bytes(), &mut recv);
                        for r in 0..world.unsigned_abs() as usize {
                            let mut word = [0u8; 8];
                            word.copy_from_slice(&recv[r * 8..r * 8 + 8]);
                            assert_eq!(
                                u64::from_ne_bytes(word),
                                round * 100 + r as u64,
                                "rank {rank} saw the wrong value for rank {r} in round {round}"
                            );
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("no rank saw another round's bytes");
        }
    }
}
