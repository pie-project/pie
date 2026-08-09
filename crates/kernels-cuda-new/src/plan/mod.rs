//! Layer 2½: FlashInfer's attention scheduler, as host arithmetic, in Rust.
//!
//! # Why this exists
//!
//! `flashinfer/attention/scheduler.cuh` is 1827 lines of **host C++** with no
//! device code in it at all. Given a batch's sequence lengths, page counts,
//! head counts and a workspace size it decides how to spread attention work
//! across CTAs, writes a handful of index arrays into a page-locked staging
//! buffer, and fills in a `PlanInfo` struct the kernel reads. Arithmetic, a
//! `memcpy`, and a struct.
//!
//! That file is the reason `driver-cuda` compiles C++ for attention.
//! `attention_flashinfer_common.cuh` calls `DecodePlan`, `attention_flashinfer.cu`
//! calls `PrefillPlan` three times, `attention_mla.cu` calls `MLAPlan`, and
//! `attention_flashinfer_hopper.cu` calls `PrefillSM90Plan` — and each of those
//! `#include`s drags in the FlashInfer tree, which drags in CMake, CCCL,
//! CUTLASS and nvcc, into a crate that otherwise needs none of them. NVRTC does
//! not help: §4.4 of `new-horizon.md` names the split, and it is the sharpest
//! sentence in that document — *"launch from Rust" and "compile at run time"
//! are orthogonal axes.* A JIT does nothing for a `std::sort` on the host.
//!
//! # What this buys, and what is still missing
//!
//! With this module in place, the only remaining reason `driver-cuda` compiles
//! C++ for attention is **the kernels themselves** — and those are the thing
//! NVRTC is for. Named, so the next person does not re-derive it:
//!
//! * **The FA2/FA3/MLA `__global__` templates** still go through nvcc.
//!   §13.6 prices that separately and correctly: it is a FlashInfer patch set
//!   plus ~39 bit-exact device intrinsics, not arithmetic we own.
//! * **The occupancy query.** [`decode::estimate`] takes `max_grid_size`
//!   because upstream gets it from
//!   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` **on the decode kernel**
//!   — a per-cubin fact, not a per-device one. When the decode kernel is
//!   JIT-compiled, that becomes `cuOccupancyMaxActiveBlocksPerMultiprocessor`
//!   on the resulting `CUfunction`, in layer 3. Until then the caller passes
//!   the number it already has.
//! * **The upload.** This module returns the bytes; it does not move them.
//!   The `cudaMemcpyAsync(int_buffer, page_locked, used, H2D, stream)` at the
//!   end of every upstream planner is layer 3's, and belongs beside the launch
//!   that reads it.
//! * **`TwoStageHolisticPlan`** (the persistent-kernel scheduler,
//!   `HolisticPlanInfo`) is **not ported** — we do not call it. If the
//!   persistent kernels are ever adopted it comes with them, and it is the
//!   most intricate function in the file.
//! * **`DecodePlanCache`/`PrefillPlanCache`** — our own host logic wrapped
//!   around these calls, including the graph-layout and page-count-independence
//!   predicates — is still C++ in `kernels-cuda/csrc`. It is ours, it is small,
//!   and it is the next thing to move.
//!
//! # This module takes numbers and returns numbers
//!
//! No `cudarc`, no `#[cfg(feature = ...)]`, no device allocation, no stream.
//! Every device fact the upstream code queries — SM count, compute capability,
//! occupancy — is a parameter here ([`Device`]), because a scheduler that
//! queries a device cannot be tested without one, and the whole argument for
//! moving this code is that it is arithmetic. It sits in layer 2 with
//! [`crate::table`] and [`crate::source`]: data in, data out, on a machine that
//! has never seen a GPU. `tests/plan.rs` is the proof — it compiles the real
//! `scheduler.cuh` with nvcc, runs both, and compares bytes.
//!
//! # The ABI is a contract, and it is asserted rather than described
//!
//! [`info`] mirrors four structs that **device kernels read**. A field at the
//! wrong offset is not a wrong answer this crate can see: it is
//! `params.request_indices` pointing at `padded_batch_size`, a kernel indexing
//! a page table with a byte count, and a model that answers fluent nonsense. No
//! test that only checks the Rust side can catch it, so the layout is pinned
//! with `const _: () = assert!(offset_of!(..) == ..)` on every field of every
//! struct — a mismatch is a compile error, in this crate, on a machine with no
//! CUDA. The numbers came from `offsetof` on the real headers; `tests/plan.rs`
//! re-derives them from the same C++ every time it runs.
//!
//! # Faithful first, idiomatic second
//!
//! A work partitioner's integer division IS its behaviour. `ceil_div` written
//! as `(a + b - 1) / b` where upstream widened to `int64_t` first changes which
//! CTA gets which row, and the symptom is a throughput regression or a wrong
//! logit — never a crash. So [`arith`] ports the arithmetic with the C++
//! integer conversions spelled out, [`heap`] reimplements libstdc++'s
//! `push_heap`/`pop_heap` rather than reaching for `BinaryHeap` (whose
//! tie-break differs, and ties are the common case: equal-cost CTAs), and
//! [`sort`] reimplements libstdc++'s introsort because `std::sort` is
//! **unstable** and the order of equal-length requests decides their CTA
//! assignment. Where this port deviates, the deviation is in a comment with the
//! reason — there are three, all of them places where upstream has undefined
//! behaviour ([`Error::EmptyBatch`] and the two overflow refusals).
//!
//! The surface is Rust: real return types instead of out-parameters,
//! [`Result`] instead of `cudaError_t`, slices instead of raw pointers. None of
//! that changes a number.

pub mod alloc;
pub mod arith;
pub mod decode;
pub mod error;
pub mod heap;
pub mod info;
pub mod mla;
pub mod prefill;
pub mod sm90;
pub mod sort;

pub use error::Error;
pub use info::{DecodePlanInfo, MlaPlanInfo, PrefillPlanInfo, PrefillPlanSm90Info};

/// The device facts a planner reads, hoisted into a parameter.
///
/// Upstream calls `cudaGetDevice` + `cudaDeviceGetAttribute` in the middle of
/// the arithmetic. Doing the same here would make this module need a driver to
/// answer a question about integers, and would make every test need a GPU —
/// which is precisely the coupling this port exists to break. The caller asks
/// once, at the layer that already holds a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    /// `cudaDevAttrMultiProcessorCount`.
    ///
    /// Every planner scales its grid by this; MLA and SM90 size arrays by it,
    /// so it is load-bearing for the *layout* as well as for the schedule.
    pub num_sm: u32,
    /// `cudaDevAttrComputeCapabilityMajor`.
    ///
    /// Read by exactly one decision — [`arith::fa2_determine_cta_tile_q`]'s
    /// Ampere branch — and it changes `cta_tile_q`, which changes every
    /// subsequent tile count. Kept as a whole number rather than folded into a
    /// bool so the next upstream branch on it does not need a new field.
    pub cc_major: i32,
}

impl Device {
    /// A device, named by the two attributes the planners read.
    #[must_use]
    pub const fn new(num_sm: u32, cc_major: i32) -> Self {
        Self { num_sm, cc_major }
    }
}

/// The two workspace buffers, as sizes — because a planner only ever needs
/// their sizes.
///
/// Upstream takes four pointers and two lengths. Three of the pointers exist
/// to be written or `memcpy`'d, and the fourth (`float_buffer`) is never
/// dereferenced at all: the float allocator only ever hands back *offsets*. So
/// the honest signature is the sizes, and the honest return is the bytes to
/// upload — which is what [`Plan`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workspace {
    /// `float_workspace_size_in_bytes` — the partial-output arena, carved but
    /// never written by the planner.
    pub float_bytes: usize,
    /// `int_workspace_size_in_bytes` — the descriptor arena, which the planner
    /// fills and the caller uploads.
    pub int_bytes: usize,
}

impl Workspace {
    /// A workspace of the two given sizes.
    #[must_use]
    pub const fn new(float_bytes: usize, int_bytes: usize) -> Self {
        Self { float_bytes, int_bytes }
    }

    /// The workspace a sizing pass uses: unbounded, so nothing refuses.
    ///
    /// This is upstream's default-constructed `AlignedAllocator`, whose
    /// `remaining_space` is `SIZE_MAX` — which is why `PrefillPlanWorkspaceSize`
    /// can never report an overflow, and why asking for sizes and then
    /// allocating them is the only way to be sure a plan fits.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self { float_bytes: usize::MAX, int_bytes: usize::MAX }
    }
}

/// A finished plan: the struct the kernel reads, and the bytes to put under it.
///
/// The C++ writes into a page-locked buffer and issues the H2D copy itself. We
/// return the buffer instead, for two reasons. It keeps the module free of a
/// stream — the copy belongs with the launch — and it makes the result
/// *comparable*: `tests/plan.rs` diffs this `Vec` against the bytes upstream
/// staged, which is the only honest gate on a port like this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan<I> {
    /// The struct the device kernel reads. Its layout is the contract; see
    /// [`info`].
    pub info: I,
    /// Exactly the bytes upstream would have copied H2D: the int workspace
    /// from offset 0 to [`Plan::int_bytes`].
    ///
    /// Zero-filled where the allocator padded for alignment. Upstream copies
    /// whatever was in the page-locked buffer there — usually a previous
    /// plan's bytes — so a caller that compares gaps is comparing garbage; a
    /// caller that uploads them is uploading padding either way.
    pub int_upload: Vec<u8>,
    /// `num_allocated_bytes()` of the int allocator: the length of
    /// [`Plan::int_upload`], and the length upstream hands `cudaMemcpyAsync`.
    pub int_bytes: usize,
    /// `num_allocated_bytes()` of the float allocator — carved, never written.
    ///
    /// Zero when the plan does not split KV, because the float arena exists
    /// only to hold the partial outputs a split produces.
    pub float_bytes: usize,
}

/// What a sizing pass answers: how big the two arenas must be.
///
/// The counting mode of upstream's `AlignedAllocator`, which
/// `DecodePlanWorkspaceSize` and `PrefillPlanWorkspaceSize` run the whole
/// planner in with `MATERIALIZE = false`. Same arithmetic, same order, no
/// buffer — so the sizes are exactly what the materialising pass will use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// Bytes the float arena needs.
    pub float_bytes: usize,
    /// Bytes the int arena needs.
    pub int_bytes: usize,
}
