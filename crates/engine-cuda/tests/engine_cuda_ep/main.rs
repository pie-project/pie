//! The tests that run several ranks of one group on one device: a binary of
//! their own. A rank spins in a barrier until its peers arrive, and a
//! legacy-stream call from a test running beside it (in this process, on
//! this device) waits on that spin while holding back a peer's launch: one
//! device's ranks and nothing else, one test at a time.
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

/// Held by each test for its whole run. A test that fails leaves nothing
/// behind it guards, so the next still runs.
#[allow(clippy::disallowed_methods)]
pub fn alone() -> std::sync::MutexGuard<'static, ()> {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

mod an_expert_parallel_pair_answers_what_one_rank_answers;
mod an_expert_parallel_wire_carries_tokens_out_and_back;
mod peer_collectives_answer_what_every_rank_holds;
