//! The tests that run the ranks of one group in one process: a binary of
//! their own. Rank `r` runs on device `r % count` (`PIE_EP_RANKS` ranks,
//! two by default), so a pair runs on two peers and a larger group puts
//! several ranks on a device. A rank spins in a barrier until its peers
//! arrive, and a legacy-stream call from a test running beside it (in this
//! process, on its device) waits on that spin while holding back a peer's
//! launch: the group's ranks and nothing else, one test at a time.
#![cfg(feature = "cuda")]

// Strict kernels for the whole binary, as `engine_cuda`'s.
#[cfg(target_os = "linux")]
#[used]
#[allow(unsafe_code)]
#[unsafe(link_section = ".init_array")]
static STRICT_KERNELS: extern "C" fn() = {
    extern "C" fn on() {
        kernels_cuda::jit::strict_kernels();
    }
    on
};

// Every work queue a context may have, before CUDA's first call reads the
// count and before any thread runs: a group of four or eight ranks on one
// device needs more than the default eight (`ep::wire` refuses it).
#[cfg(target_os = "linux")]
#[used]
#[allow(unsafe_code)]
#[unsafe(link_section = ".init_array")]
static WORK_QUEUES: extern "C" fn() = {
    extern "C" fn queues() {
        if std::env::var_os("CUDA_DEVICE_MAX_CONNECTIONS").is_none() {
            // SAFETY: a constructor, before main and before any thread.
            unsafe { std::env::set_var("CUDA_DEVICE_MAX_CONNECTIONS", "32") };
        }
    }
    queues
};

/// Held by each test for its whole run. A test that fails leaves nothing
/// behind it guards, so the next still runs.
#[allow(clippy::disallowed_methods)]
pub fn alone() -> std::sync::MutexGuard<'static, ()> {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

mod an_expert_parallel_group_decodes_at_its_buckets;
mod an_expert_parallel_group_rebalances_its_experts;
mod an_expert_parallel_group_serves_through_its_engines;
mod an_expert_parallel_group_that_arms_little_serves_eagerly;
mod an_expert_parallel_pair_answers_what_one_rank_answers;
mod an_expert_parallel_wire_carries_tokens_out_and_back;
mod micro;
mod peer_collectives_answer_what_every_rank_holds;
