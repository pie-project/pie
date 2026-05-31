//! The per-fire forward loop — replaces the core of
//! `driver/cuda/src/executor/executor.cpp::handle_fire_batch`.
//!
//! Everything *decided* here is Rust; everything *computed* is one coarse
//! FFI call into libpie_cuda_device. A steady decode step is:
//!
//!   1. classify the batch (pure-decode vs prefill)        [Rust]
//!   2. pick the CUDA-graph bucket, if graph-safe          [Rust]  (forward_graph.hpp)
//!   3. upload inputs → prepare → body | graph_launch      [FFI]
//!   4. sample (policy in Rust, kernel in C++)             [Rust + FFI]
//!   5. build the response view                            [Rust, pie-bridge]
//!
//! The MTP/spec-decode path (spec.rs) and TP broadcast (tp.rs) wrap this;
//! they are *not* tangled into it the way executor.cpp interleaves them.

use anyhow::Result;

use crate::arch::Capabilities;
use crate::device::Device;
use crate::ffi::PieArchId;

/// vLLM-style decode graph lattice (ported from forward_graph.hpp:63).
/// Batches pad up to one of these request counts before capture/replay:
/// 1, 2, 4, multiples of 8 up to 256, then multiples of 16.
pub fn graph_request_bucket(requests: i32, max_requests: i32) -> i32 {
    if requests <= 0 || max_requests <= 0 || requests > max_requests {
        return 0;
    }
    let bucket = if requests <= 1 {
        1
    } else if requests <= 2 {
        2
    } else if requests <= 4 {
        4
    } else if requests < 256 {
        ((requests + 7) / 8) * 8
    } else {
        ((requests + 15) / 16) * 16
    };
    bucket.min(max_requests)
}

/// Stable cross-fire references the loop threads through. The Rust analog
/// of driver/cuda's `Executor` struct (executor.hpp:261) — but holding
/// only what the *control plane* needs; device state lives behind handles.
pub struct Executor<'a> {
    pub device: &'a Device,
    pub arch: PieArchId,
    pub caps: Capabilities,
    pub max_forward_requests: i32,
    // Phase 2/3 add: Weights, KvCache, Workspace handles; graph cache;
    // sampler; spec drafter; tp comm.
}

impl<'a> Executor<'a> {
    /// Run one fire_batch request. The batched-forward CORE now lives in
    /// `builder::Model::fire_batch` (assemble R requests across the paged KV
    /// cache → one forward → per-request next token, bit-equivalent to single
    /// runs). This `Executor::fire` is the serving wrapper that will add the
    /// transport request-view decode (steps 1/2/5) + sampling policy around it
    /// once `pie-bridge` is wired; until then drive `Model::fire_batch` directly.
    pub fn fire(&mut self /* view: PieForwardRequestView, out: &mut ... */) -> Result<()> {
        // 1. classify; 2. bucket; 3. upload/prepare/body|launch (→ Model::fire_batch);
        // 4. sample; 5. build response (pie-bridge).
        unimplemented!("executor.fire — serving wrapper (transport pending); core = Model::fire_batch")
    }
}

#[cfg(test)]
mod tests {
    use super::graph_request_bucket;

    // Same invariants as the static_asserts in forward_graph.hpp:85-91.
    #[test]
    fn bucket_lattice() {
        assert_eq!(graph_request_bucket(1, 512), 1);
        assert_eq!(graph_request_bucket(3, 512), 4);
        assert_eq!(graph_request_bucket(5, 512), 8);
        assert_eq!(graph_request_bucket(255, 512), 256);
        assert_eq!(graph_request_bucket(257, 512), 272);
        assert_eq!(graph_request_bucket(506, 512), 512);
        assert_eq!(graph_request_bucket(129, 130), 130);
    }
}
