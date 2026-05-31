//! Tensor-parallel orchestration — replaces the rank-0 broadcast loop and
//! `tp_follower_serve` in `driver/cuda/src/executor/executor.cpp`.
//!
//! Rank 0 broadcasts each fire's header + inputs; followers run the same
//! `pie_body` so the all-reduces inside complete against rank 0. The
//! broadcast plumbing is Rust (NCCL via bindings or a thin device-lib
//! entry); the all-reduce kernels stay C++ (custom_all_reduce.cu).

/// Per-fire control header broadcast from rank 0 to followers. Mirrors the
/// shape executor.cpp packs before `ncclBroadcast(root=0)`.
#[derive(Copy, Clone, Debug)]
pub struct FireHeader {
    pub total_tokens: i32,
    pub num_requests: i32,
    pub is_pure_decode: bool,
    pub shutdown: bool, // the explicit teardown sentinel
}

/// Follower serve loop (ranks > 0): block on broadcast, run body, repeat;
/// exit on the shutdown header. Phase 3 (after executor.rs).
pub fn follower_serve(/* executor, comm, stop */) {
    unimplemented!("tp::follower_serve — phase 3")
}
