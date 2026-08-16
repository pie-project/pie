//! The custom P2P all-reduce: the two symbols a sharded model text states,
//! the cross product they select from, and the launches that reach it.
//!
//! # What this is
//!
//! `comm/custom_all_reduce.cu` was 664 lines with **zero** `__global__` and
//! **zero** `<<<>>>` — a host program that happened to carry a `.cu`
//! extension for linkage. Exactly two lines of it reached device text:
//! `impl_->allreduce<__nv_bfloat16>` at `:614-620` and
//! `allreduce_fusion_kernel_launcher` at `:157-162`. Both named headers in a
//! flashinfer tree that was CPM-*fetched* at configure time and not vendored,
//! so for as long as that was true both bodies below returned
//! `Decline::NoDeviceText` and [`CAN_LAUNCH`] was `false`.
//!
//! **That is history.** `kernels/flashinfer/comm/` now holds
//! `vllm_custom_all_reduce.cuh` and `trtllm_allreduce_fusion.cuh`,
//! internalised under `MODIFICATIONS`' procedure with their host halves
//! removed; `kernels/comm/all_reduce.cuh` is the NVRTC root over both; and
//! [`ROOT`] compiles it. Both bodies launch.
//!
//! # WHAT HAS NEVER RUN, and it is most of this file
//!
//! A collective needs a peer. **This module has been compiled and never
//! fired**, because the box it was written on has one GPU:
//!
//! * Every template-id [`Instantiation::name_expression`] and
//!   [`plain_name_expression`] can return compiles under real NVRTC 13.0, all
//!   32 of them, and every one lowers to a mangled name. Measured through this
//!   crate's own compiler and its own header set at `sm_89`, `sm_90a`,
//!   `sm_100a` and `sm_120a`: rc=0 and 32/32 at each. That is
//!   `every_instantiation_compiles`'s subject and it is a real check — it
//!   proves the vendored text, the shims and the template arguments agree.
//!   `sm_80` and `sm_86` refuse, and not for a reason of this module's:
//!   `shim/cuda_fp8.h` carries an `#error` below sm_89 that every
//!   `Headers::LibraryAndUpstream` root in the crate hits, because
//!   `vec_dtypes.cuh` includes it.
//! * **No launch below has ever been observed to produce a correct sum**, on
//!   any world size, because a two-rank fire needs two devices. The grid
//!   arithmetic, the parameter block's field order, the Lamport handshake and
//!   the peer addressing are argued from upstream's source and checked by a
//!   compiler, not by an answer.
//!
//! Every claim in this file distinguishes the two. Where a sentence says
//! "measured" it names the measurement; where it does not, it is an argument.
//!
//! # Why the LIFECYCLE is not here, and where the line falls
//!
//! `driver-cuda`'s `fire/all_reduce.rs` keeps `CustomAllReduce`: peer-access
//! enablement, the IPC handle exchange, the `Signal` + staging slab, the
//! `RankData` slab, the fusion plane's four allocations and its Lamport
//! initialisation, `register_buffer`, `register_graph_buffers` and the
//! destructor that closes what they opened. Every one of those owns a device
//! resource with a lifetime, and every one of them reports through
//! `driver_cuda::error::Error` — a type carrying a `cudaError` code and a
//! `String`, which this crate cannot name and whose nearest thing here
//! ([`Decline`]) deliberately carries neither.
//!
//! What is here is what a LAUNCH needs: the cross product as data, the two
//! `switch`es that pick a point out of it ([`resolve`]), the grid arithmetic
//! that used to be `allreduce_fusion_kernel_launcher`'s
//! ([`fusion_geometry`]) and `vllm::CustomAllreduce::allreduce`'s
//! ([`plain_geometry`]), the parameter block ([`FusionParams`]), the refusals,
//! and the two bodies.
//!
//! [`Plane`] is the seam: what the constructor established, `Copy`, which is
//! all either body reads off the instance.
//!
//! # `driver_bound!`, and the one column that is not obvious
//!
//! Neither symbol is `routine!`, and the reason is the plainest in the table:
//! **a collective takes a communicator.** The group, its rank, its peer
//! mappings and its registered buffers are properties of the deployment's
//! process group, assembled at load and named by no trace statement.
//!
//! Both are `whole`, and for a reason stronger than "a reduction is over the
//! whole value": **every rank must enter the same collective the same number
//! of times**, so a row window that split one rank's launch and not another's
//! would DEADLOCK rather than compute a wrong answer. The refusal is not an
//! optimisation. They are also synchronisation points, which the
//! graph-capture rules have to know.
//!
//! [`all_reduce_residual_rmsnorm_bf16`] additionally states
//! `in_place = &[(0, 1)]`: the fused landing has TWO results, so it needs a
//! pair list rather than a single alias — the residual stream is updated in
//! place and the normed activation is the other.
//!
//! # The 240-kernel cross product, and what 111s means under a JIT
//!
//! `kernels.def`'s `PIE_AR_FUSION_PATTERN` block made the argument, and the
//! measurement in it must not be consumed by this port, so it is restated
//! here with its arithmetic intact. flashinfer's `allreduce_fusion_op()`
//! turns four runtime values into one compile-time cross product:
//!
//! ```text
//!   nranks {2,4,8,16}  x  pattern {10}  x  fp32_acc {2}
//!       x  (oneshot x trigger_completion_at_end {2}, or twoshot {1})
//!     = 4 x 10 x 2 x 3
//!     = 240 kernels
//! ```
//!
//! — **96% of every kernel in `custom_all_reduce.cu`, and the reason that
//! translation unit cost 111s (40s cicc + 44s ptxas)**. pie reaches exactly
//! one pattern, `kARResidualRMSNorm`; the other nine are FP8/FP4 quant
//! epilogues no pie call site can select.
//!
//! Three of those four axes are template parameters and the fourth is not:
//! `use_oneshot` and `trigger_completion_at_end` are RUNTIME fields of
//! `AllReduceFusionParams` that upstream's host launcher branched on to pick
//! among three `__global__`s. So the 240 is `80 host instantiations x 3
//! device leaves`, and [`Leaf`] is that third factor named.
//!
//! **The vendoring moved that line, and this is where it moved to.**
//! `allreduce_fusion_kernel_launcher` is HOST C++ — it calls
//! `cudaLaunchKernelEx` — so it is removed with the rest of the host half and
//! no `nvrtcAddNameExpression` could ever have lowered its name. What NVRTC
//! is handed is the `__global__` the launcher would have picked, so the three
//! leaves are three template-ids here rather than one template-id and a
//! branch. [`Instantiation::name_expression`] is that: **24 arms,
//! `4 nranks x 1 pattern x 2 fp32_acc x 3 leaves`, which is
//! [`AOT_POINTS_AFTER_PRUNING`] exactly** — the same number pie's own
//! ahead-of-time build compiled, now as a set of strings a fixture can hand to
//! a compiler.
//!
//! **The measurement was about an AHEAD-OF-TIME translation unit, and under
//! a JIT it says something different.** Written out:
//!
//! ```text
//!   upstream, unpruned          4 x 10 x 2 x 3  = 240 kernels   ~111 s
//!   pie's AOT build, pruned     4 x  1 x 2 x 3  =  24 kernels   ~11 s
//!   under NVRTC, on demand              reached =   1 kernel    ~0.46 s, once
//! ```
//!
//! 111s / 240 is **~0.46s per instantiation**, and that figure is the whole
//! translation. Ahead of time it was a cost paid on the build's critical
//! path on every build, whether or not a single one of the 240 ever ran; the
//! `PIE_AR_FUSION_PATTERN` list bought back 216 of them, or ~99s. Under a
//! JIT that compiles on demand the cross product stops being a *build* cost
//! at all and becomes a **cache-key space**: a point nobody reaches is never
//! compiled, so there is nothing left for a list to prune, and the remaining
//! ~11s of pie's own pruned build goes with it. What survives is ~0.46s of
//! FIRST-CALL latency for the one point pie reaches, paid once per process
//! and off everyone else's critical path.
//!
//! So the list's job changes rather than ends. It no longer states which
//! points were BUILT; it states which points are **reachable**, and that is
//! checkable before a fire instead of during one. [`INSTANTIATED`] is that
//! list, in Rust, and [`resolve`] is the check.
//!
//! # The runtime throw became a refusal
//!
//! `kernels.def` said it plainly: *"a missing entry surfaces as a runtime
//! throw, not a link error"*. Both throws are here, and neither is spelled
//! like a failure of the launch:
//!
//! ```text
//!   custom_all_reduce.cu:171-176  "pattern N is not instantiated"
//!                                   -> Decline::PatternNotInstantiated
//!   custom_all_reduce.cu:206-209  "does not support TP world size N"
//!                                   -> Decline::WorldSizeUnsupported
//! ```
//!
//! A decline enqueues **nothing** — the caller's `output` is exactly as it
//! found it, and `dist::all_reduce_bf16` (NCCL, and equally absent) is the
//! other arm. A refusal is never a fallback.

use core::ffi::c_void;
use std::fmt;

use crate::driver_bound;
use crate::jit::{ArgValue, Ctx, Family, Launch, Routine};
use kernels::Refusal;

// ── the root ─────────────────────────────────────────────────────────────

// The template-ids live in `Instantiation::name_expression` and
// `plain_name_expression` below, spelled as NVRTC is handed them.
//
// **These are the `__global__`s, not upstream's launcher.**
// `allreduce_fusion_kernel_launcher` is host C++ and is removed from the
// vendored header; the three `__global__`s it dispatched to are what a JIT can
// name, which is why `Leaf` appears in a template-id here and was a runtime
// branch upstream.
//
// Two functions rather than one, because the two symbols do not share an axis:
// the fused landing is `{nranks} x {fp32_acc} x {leaf}` over flashinfer's
// instantiated world sizes, and the plain reduction is `{ngpus} x {stage}` over
// vllm's, which are a different set -- `{2,4,6,8}` against `{2,4,8,16}`. See
// `PLAIN_NRANKS`.
//
// WRITTEN OUT, AND NOT BUILT WITH A `format!`. A template-id assembled at run
// time is one no fixture can compile ahead of the fire that needs it, and
// `every_instantiation_compiles` is the only thing in this repository that has
// ever put one of these through a compiler: it reads the literals out of these
// two `fn` bodies, pairs them with the carried file the launch sites name, and
// hands all 32 to NVRTC. Measured on an L40S with NVRTC 13.0: rc=0 and 32/32
// lowered, at sm_89 and again at sm_90a.

// ── the constants a LAUNCH fixes ─────────────────────────────────────────
//
// The plane's own sizes -- `SIGNAL_BYTES`, `RANK_DATA_BYTES`, the Lamport
// cap, the 2 MiB fusion alignment -- stayed with the constructor that spends
// them. These are the launch rectangle, so they are here.

/// Whether a call into this module can reach device text — **`true`.**
///
/// It was `false` for as long as `flashinfer/comm/` was CPM-fetched by
/// upstream and not vendored here. It is `true` now because
/// `kernels/flashinfer/comm/` holds both headers and [`ROOT`] compiles
/// them.
///
/// # What it does and does not claim
///
/// It claims that a call which resolves reaches [`Ctx::launch`] — that there
/// is device text, that it compiles, and that a template-id names it. It
/// claims **nothing** about whether the result is right, and this file's
/// header says at length why it cannot: no launch here has ever run on more
/// than one device. The honest reading is *"the refusal that stood in front
/// of tensor parallelism is gone"*, not *"tensor parallelism is verified"*.
///
/// # Why a constant rather than a comment
///
/// A deployment has to decide, before it serves a single token, whether a
/// tensor-parallel rank can combine its shards. `driver-cuda`'s
/// `serve::load::tp_serving_refusal` reads this — so the driver's refusal is
/// derived from the launch half's own statement about itself instead of being
/// a second, hand-written account of the same absence that can drift from it.
/// The old refusal was exactly that second account, and it had already
/// drifted: it said *"no `CustomAllReduce` handle to pass"* after the handle
/// existed.
///
/// `the_bodies_agree_with_can_launch` fails the moment this and the bodies
/// disagree, in either direction.
pub const CAN_LAUNCH: bool = true;

/// The vector width both kernels move data in, in bf16 ELEMENTS.
///
/// Two derivations of one number, and they have to agree or one of the two
/// kernels is addressing wrongly:
///
/// * vllm — `packed_t<T>::P` is `array_t<T, 16 / sizeof(T)>`
///   (`vllm_custom_all_reduce.cuh:83`), so `d == 8` for bf16 and
///   `allreduce()` refuses a `size % d != 0`.
/// * flashinfer — `VEC_SIZE = details::kBytesPerAccess / sizeof(T)` with
///   `kBytesPerAccess == 16` (`trtllm_allreduce_fusion.cuh:30`).
///
/// Both are "16 bytes", which is `ld.128`/`st.128`, which is the whole reason
/// the number exists. `Decline::FusionHiddenNotOctet`'s `hidden % 8` is this
/// number on the fused side and [`Decline::Vector`] is it on the plain side.
pub const VEC_SIZE: i32 = 8;

/// The cluster dimension every launch in this module sets — **1**.
///
/// # This is a deviation from upstream and it is deliberate
///
/// `allreduce_fusion_kernel_launcher` picks `cluster_size = 8` on SM 90 and
/// above and then halves it until the token divides, so that one token's
/// `hidden_dim / VEC_SIZE` threads can span up to eight blocks. That needs
/// `cudaLaunchAttributeClusterDimension` on the launch, and
/// [`crate::jit::Launch`] carries no cluster dimension — it has `grid`,
/// `block`, `smem` and `cooperative`, and `jit::launch::issue` sets at most
/// the cooperative attribute.
///
/// **What pinning it to 1 costs, exactly.** With one block per token the
/// block is `hidden_dim / 8` threads wide, so the ceiling is
/// `hidden_dim <= 8192`; above that [`fusion_geometry`] answers
/// [`Decline::FusionBlockWidth`] where upstream would have split the token
/// across a cluster. Llama-3-70B's 8192 and Qwen3-32B's 5120 are inside it;
/// a 12k or 16k hidden size is not.
///
/// **What it does NOT cost.** A cluster of one is the DEFAULT cluster shape
/// on sm_90, not a special case: `%cluster_nctarank` reads 1, so
/// `FusedOp::rms_norm`'s `if (cluster.num_blocks() > 1)` is false and the
/// distributed-shared-memory reduction is skipped, and `IndexHelper` computes
/// exactly what its own `#else` arm computes below sm_90. So the arithmetic
/// is upstream's at cluster size 1 rather than an approximation of it.
///
/// The other thing this pin buys is that `adjust_for_sm_count` — upstream's
/// loop that widens the block when `cluster_num * cluster_size > sm_count` —
/// is a no-op, because its guard is `cluster_size > 1`. That loop is the only
/// reader of upstream's `max_threads_per_block`, which is
/// `min(max_registers / registers_per_thread, 1024)` and needs
/// `cudaFuncGetAttributes` on the compiled kernel. **Neither is computed
/// here, and with cluster size 1 neither has a reader.** Lifting this pin
/// means restoring both.
pub const CLUSTER_SIZE: i32 = 1;

/// The widest block CUDA will launch, on every architecture this crate
/// targets.
///
/// `public` because [`Decline::FusionBlockWidth`] reports against it and a
/// caller reading that refusal has to be able to name the ceiling it hit.
pub const MAX_BLOCK_THREADS: i32 = 1024;

// ── the cross product, as data ───────────────────────────────────────────

/// `flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern`, with
/// upstream's discriminants.
///
/// **The discriminants are not contiguous** — 6 and 7 are absent upstream —
/// so this enum states each one explicitly rather than relying on
/// declaration order. A `from_code` that assumed density would map
/// `kARResidualRMSNormPerTokenGroupFP8PackedQuant` (8) onto a pattern two
/// places away, silently, on a path whose whole job is to refuse the
/// patterns pie cannot serve.
///
/// `#[repr(i32)]` is now load-bearing rather than documentary: this type is a
/// FIELD of [`FusionParams`], which is `#[repr(C)]` and crosses to the device
/// as a by-value kernel parameter. `sizeof(AllReduceFusionPattern) == 4`,
/// measured against the vendored header.
///
/// # A drift worth recording
///
/// `kernels.def`'s block says `pattern {10}` and derives `4 x 10 x 2 x 3 =
/// 240` from it. The vendored header declares **eight** enumerators spanning
/// 0..=9. The 240 is preserved above exactly as measured — it was measured
/// against the header the archive compiled, and a measurement is not amended
/// by a later reading — but a re-measurement on the vendored copy would find
/// `4 x 8 x 2 x 3 = 192`. Recorded rather than reconciled: the only number
/// this module ACTS on is the size of [`INSTANTIATED`], which is 1 either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum FusionPattern {
    /// `kAllReduce` — the reduction with no epilogue.
    AllReduce = 0,
    /// `kARResidualRMSNorm` — **the one pattern pie reaches.**
    ARResidualRMSNorm = 1,
    /// `kARResidualRMSNormFp8Quant`.
    ARResidualRMSNormFp8Quant = 2,
    /// `kARResidualRMSNormFP4Quant`.
    ARResidualRMSNormFp4Quant = 3,
    /// `kARResidualRMSNormOutFP8Quant`.
    ARResidualRMSNormOutFp8Quant = 4,
    /// `kARResidualRMSNormOutFP4Quant`.
    ARResidualRMSNormOutFp4Quant = 5,
    /// `kARResidualRMSNormPerTokenGroupFP8PackedQuant`. Note the gap: 6 and
    /// 7 are not enumerators upstream.
    ARResidualRMSNormPerTokenGroupFp8PackedQuant = 8,
    /// `kARResidualRMSNormOutPerTokenGroupFP8PackedQuant`.
    ARResidualRMSNormOutPerTokenGroupFp8PackedQuant = 9,
}

impl FusionPattern {
    /// Every pattern upstream declares, in discriminant order.
    pub const ALL: &'static [Self] = &[
        Self::AllReduce,
        Self::ARResidualRMSNorm,
        Self::ARResidualRMSNormFp8Quant,
        Self::ARResidualRMSNormFp4Quant,
        Self::ARResidualRMSNormOutFp8Quant,
        Self::ARResidualRMSNormOutFp4Quant,
        Self::ARResidualRMSNormPerTokenGroupFp8PackedQuant,
        Self::ARResidualRMSNormOutPerTokenGroupFp8PackedQuant,
    ];

    /// The `int` upstream's `enum class ... : int` carries.
    #[must_use]
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// The C++ spelling, for a message a reader can grep upstream for.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AllReduce => "kAllReduce",
            Self::ARResidualRMSNorm => "kARResidualRMSNorm",
            Self::ARResidualRMSNormFp8Quant => "kARResidualRMSNormFp8Quant",
            Self::ARResidualRMSNormFp4Quant => "kARResidualRMSNormFP4Quant",
            Self::ARResidualRMSNormOutFp8Quant => "kARResidualRMSNormOutFP8Quant",
            Self::ARResidualRMSNormOutFp4Quant => "kARResidualRMSNormOutFP4Quant",
            Self::ARResidualRMSNormPerTokenGroupFp8PackedQuant => {
                "kARResidualRMSNormPerTokenGroupFP8PackedQuant"
            }
            Self::ARResidualRMSNormOutPerTokenGroupFp8PackedQuant => {
                "kARResidualRMSNormOutPerTokenGroupFP8PackedQuant"
            }
        }
    }

    /// The pattern with this discriminant, or `None`.
    ///
    /// Explicit rather than a transmute, for the gap the type's doc names.
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.iter().copied().find(|p| p.code() == code)
    }
}

/// `flashinfer::QuantizationSFLayout` — `fp4_layout.cuh:21-35`.
///
/// A field of [`FusionParams`], read only by the FP4 quant epilogues, which
/// no pattern in [`INSTANTIATED`] selects. It is a real `#[repr(i32)]` enum
/// rather than the `&'static str` this struct used to carry, for one reason:
/// the struct is now `#[repr(C)]` and crosses to the device, so the field has
/// to be four bytes with upstream's discriminant. `sizeof` and the
/// declaration order are read off the vendored header, which this tree now
/// has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SfLayout {
    /// `SWIZZLED_128x4`, upstream's default and the only value pie sets.
    Swizzled128x4 = 0,
    /// `SWIZZLED_8x4`.
    Swizzled8x4 = 1,
    /// `LINEAR`.
    Linear = 2,
}

/// `kernels.def`'s `PIE_AR_FUSION_PATTERN` list, in Rust — **one entry.**
///
/// This is the whole of what the `#include "kernels.def"` inside
/// `dispatch_ar_fusion_pattern` (`custom_all_reduce.cu:163-169`) expanded to.
/// The C++ expanded it into `case` labels of a `switch` on
/// `params.pattern`; [`resolve`] is that switch, and the `default:` that
/// threw is [`Decline::PatternNotInstantiated`].
///
/// **Adding a pattern to a call site requires adding it here** and to
/// [`Instantiation::name_expression`], exactly as `kernels.def` said — but the consequence has
/// changed twice and is now best: a missing entry used to surface as a
/// runtime throw deep inside a fire; then as a `Decline` a caller handles by
/// type; and now the added entry is a string a fixture compiles before it can
/// ever be fired.
pub static INSTANTIATED: &[FusionPattern] = &[FusionPattern::ARResidualRMSNorm];

/// upstream flashinfer's supported TP world sizes for the FUSED landing —
/// `allreduce_fusion_op`'s `switch (params.nranks)`.
///
/// **Deliberately unpruned, and the archive said why** (`:177-179`): *"nranks
/// is upstream flashinfer's supported set rather than a pie-owned axis, so it
/// stays fully instantiated: TP world size is a deployment choice and pruning
/// it would turn a valid launch into a runtime throw."* Under a JIT the
/// argument gets stronger, not weaker — an unreached world size now costs
/// nothing at all, not even a compile.
pub static NRANKS: &[i32] = &[2, 4, 8, 16];

/// upstream vllm's supported world sizes for the PLAIN reduction —
/// `vllm_custom_all_reduce.cuh:490-500`'s `REDUCE_CASE(2/4/6/8)`.
///
/// **It is a different set from [`NRANKS`] and neither contains the other.**
/// vllm has 6 and flashinfer does not; flashinfer has 16 and vllm does not.
/// That is not a mistake to reconcile: they are two upstreams' independent
/// choices about which `ngpus` to instantiate, and a deployment at world size
/// 6 can run the plain reduction and cannot run the fused landing (which is
/// world size 2 only anyway, so nothing is lost).
///
/// `CustomAllReduce::initialise` refuses anything outside `{2,4,6,8}` at
/// construction, so a `Plane` reaching [`plain_geometry`] with something else
/// cannot come from this tree — the check is kept because it is upstream's
/// `default: throw` and because the raw-pointer ABI forms exist.
pub static PLAIN_NRANKS: &[i32] = &[2, 4, 6, 8];

/// Which of the three fused device leaves a set of runtime flags selects.
///
/// The axis that is NOT a template parameter of upstream's HOST launcher, and
/// IS one of the `__global__` it dispatched to. `AllReduceFusionParams`
/// carries `use_oneshot` and `trigger_completion_at_end` as fields; upstream
/// branched on them and picked one of three `__global__`s. `kernels.def`
/// wrote it `(oneshot x trigger_completion_at_end {2}, or twoshot {1})`,
/// which is 3 and not 4 — the two-shot path ignores
/// `trigger_completion_at_end` entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leaf {
    /// `use_oneshot = true`, `trigger_completion_at_end = false`. **The one
    /// pie sets** — `custom_all_reduce.cu:653`, `:658` — and therefore the
    /// only one of the 24 fused points a fire in this tree can reach.
    OneShot,
    /// `use_oneshot = true`, `trigger_completion_at_end = true`.
    OneShotTriggerAtEnd,
    /// `use_oneshot = false`. `trigger_completion_at_end` is not read.
    TwoShot,
}

impl Leaf {
    /// The leaf a pair of `AllReduceFusionParams` flags selects.
    #[must_use]
    pub const fn of(use_oneshot: bool, trigger_completion_at_end: bool) -> Self {
        if !use_oneshot {
            Self::TwoShot
        } else if trigger_completion_at_end {
            Self::OneShotTriggerAtEnd
        } else {
            Self::OneShot
        }
    }

    /// Whether this leaf is a one-shot Lamport launch.
    #[must_use]
    pub const fn oneshot(self) -> bool {
        !matches!(self, Self::TwoShot)
    }
}

/// Which of the plain reduction's two `__global__`s a message selects.
///
/// `vllm_custom_all_reduce.cuh:476-488`'s `REDUCE_CASE` macro, as a value.
/// The choice is upstream's and it is about BANDWIDTH, not correctness: both
/// kernels compute the same sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// `cross_device_reduce_1stage` — every rank reads every peer and writes
    /// its own output. World size 2 always, and small messages otherwise.
    OneStage,
    /// `cross_device_reduce_2stage` — reduce-scatter into the peers' staging
    /// buffers, then all-gather. The wider and larger case.
    TwoStage,
}

/// How many device leaves a single fused host instantiation carries — 3.
pub const LEAVES: usize = 3;

/// `fp32_acc {2}` — the `Fp32Acc` template parameter's two values.
pub const FP32_ACC_VALUES: usize = 2;

/// The cross product `kernels.def` measured: **240 kernels**.
///
/// Stated as a constant rather than recomputed, because it is a
/// MEASUREMENT's denominator and recomputing it from today's
/// [`FusionPattern::ALL`] would silently restate it as 192 — see that type's
/// drift note.
pub const UPSTREAM_POINTS: usize = 240;

/// The translation unit's cost, in seconds — `kernels.def:114-115`.
pub const AOT_TU_SECONDS: usize = 111;

/// Of which cicc — `kernels.def:115`.
pub const AOT_CICC_SECONDS: usize = 40;

/// Of which ptxas — `kernels.def:115`.
pub const AOT_PTXAS_SECONDS: usize = 44;

/// What pie's own AOT build compiled after `PIE_AR_FUSION_PATTERN` pruned
/// the pattern axis: `4 nranks x 1 pattern x 2 fp32_acc x 3 leaves` = **24**.
///
/// Derived rather than written, so that adding a pattern to [`INSTANTIATED`]
/// moves the figure the module's header quotes — and
/// [`Instantiation::name_expression`] has exactly this many arms, which
/// `the_instantiation_table_is_the_pruned_cross_product` pins.
pub const AOT_POINTS_AFTER_PRUNING: usize =
    NRANKS.len() * INSTANTIATED.len() * FP32_ACC_VALUES * LEAVES;

/// One point of the fused cross product: everything a launch has to fix
/// before any device text exists for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instantiation {
    /// `NRanks` — a template parameter.
    pub nranks: i32,
    /// `Pattern` — a template parameter.
    pub pattern: FusionPattern,
    /// `Fp32Acc` — a template parameter. `custom_all_reduce.cu:660` pins
    /// `constexpr bool use_fp32_acc = true`: **fp32 accumulation of bf16
    /// multiplies**, the arithmetic every parity check in this tree is
    /// written against.
    pub fp32_acc: bool,
    /// Which `__global__`. Upstream's launcher branched on two runtime flags
    /// to pick it; here it is part of the template-id — see [`Leaf`].
    pub leaf: Leaf,
}

impl Instantiation {
    /// The template-id NVRTC is handed for this point.
    ///
    /// A `match` over WRITTEN-OUT strings, not a `format!`, so that the id a
    /// fire hands NVRTC is byte for byte the id
    /// `every_instantiation_compiles` put through a compiler — that fixture
    /// reads these literals out of this `fn` and pairs them with the carried
    /// file the two launch sites name.
    ///
    /// `None` cannot happen after [`resolve`], which admits only
    /// `nranks in NRANKS` and `pattern in INSTANTIATED`, and the arms cover
    /// that product times both `fp32_acc` values times all three leaves.
    /// `the_table_covers_everything_resolve_admits` is what keeps that true.
    #[must_use]
    pub fn name_expression(&self) -> Option<&'static str> {
        match (self.nranks, self.fp32_acc, self.leaf) {
            (2, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, true, false>",
            ),
            (2, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, true, true>",
            ),
            (2, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, true>",
            ),
            (2, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, false, false>",
            ),
            (2, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, false, true>",
            ),
            (2, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 2, false>",
            ),
            (4, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, true, false>",
            ),
            (4, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, true, true>",
            ),
            (4, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, true>",
            ),
            (4, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, false, false>",
            ),
            (4, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, false, true>",
            ),
            (4, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 4, false>",
            ),
            (8, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, true, false>",
            ),
            (8, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, true, true>",
            ),
            (8, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, true>",
            ),
            (8, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, false, false>",
            ),
            (8, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, false, true>",
            ),
            (8, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 8, false>",
            ),
            (16, true, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, true, false>",
            ),
            (16, true, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, true, true>",
            ),
            (16, true, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, true>",
            ),
            (16, false, Leaf::OneShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, false, false>",
            ),
            (16, false, Leaf::OneShotTriggerAtEnd) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_oneshot_lamport<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, false, true>",
            ),
            (16, false, Leaf::TwoShot) => Some(
                "::flashinfer::trtllm_allreduce_fusion::\
                 allreduce_fusion_kernel_twoshot_sync<\
                 (::flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, \
                 __nv_bfloat16, 16, false>",
            ),
            _ => None,
        }
    }
}

/// The one point pie reaches — 1 of [`UPSTREAM_POINTS`].
///
/// Every field is what `custom_all_reduce.cu:637-660` set, and the world size
/// is 2 because `can_fuse_residual_rmsnorm` (`:493-501`) admits nothing else.
pub const REACHED: Instantiation = Instantiation {
    nranks: 2,
    pattern: FusionPattern::ARResidualRMSNorm,
    fp32_acc: true,
    leaf: Leaf::OneShot,
};

/// The two `switch` statements of `custom_all_reduce.cu:143-215`, as one
/// function — **and both of their `throw`s, as `Err`.**
///
/// Order matters and is upstream's: `pie_allreduce_fusion_op` switched on
/// `nranks` first (`:181`) and reached `dispatch_ar_fusion_pattern`'s switch
/// on `pattern` (`:165`) only inside a supported arm. A caller with both
/// wrong therefore hears about the world size, which is the one it can
/// actually change.
///
/// # Errors
///
/// [`Decline::WorldSizeUnsupported`] for an `nranks` outside [`NRANKS`], and
/// [`Decline::PatternNotInstantiated`] for a pattern outside
/// [`INSTANTIATED`].
pub fn resolve(
    nranks: i32,
    pattern: FusionPattern,
    fp32_acc: bool,
    use_oneshot: bool,
    trigger_completion_at_end: bool,
) -> std::result::Result<Instantiation, Decline> {
    if !NRANKS.contains(&nranks) {
        return Err(Decline::WorldSizeUnsupported { nranks });
    }
    if !INSTANTIATED.contains(&pattern) {
        return Err(Decline::PatternNotInstantiated { code: pattern.code() });
    }
    Ok(Instantiation {
        nranks,
        pattern,
        fp32_acc,
        leaf: Leaf::of(use_oneshot, trigger_completion_at_end),
    })
}

// ── the refusals ─────────────────────────────────────────────────────────

/// Why a call did not reduce anything.
///
/// Every arm enqueues NOTHING, which is the whole contract: on a decline the
/// caller's `output` is exactly as it found it and `dist::all_reduce_bf16`
/// (NCCL) is the arm to take instead. `custom_all_reduce.hpp:160-163` is the
/// sentence this type exists to keep true — *"WHICH is a guard in the text
/// rather than an `if` inside a driver method"*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// `car` was null — `custom_all_reduce.hpp:170-174`, `:193-197`.
    ///
    /// **A refusal, not a fallback.** The header threw here, and its comment
    /// says why: *"a null one is a deployment that configured no custom
    /// all-reduce, which is a refusal rather than a fallback: the fused
    /// landing IS this kernel, and there is no other way to spell it."*
    NoInstance,
    /// The instance exists but construction never completed —
    /// `custom_all_reduce.cu:606-608`, `:503`, `:544`.
    ///
    /// In the C++ this was `!impl_`, the moved-from and default-constructed
    /// state. Here it is a `Plane` whose `self_signal` is null, which no
    /// constructed plane produces.
    NotInitialised,
    /// `input` was null — `custom_all_reduce.cu:466`.
    NullInput,
    /// `bytes == 0`, `bytes > max_bytes` or `bytes % 16 != 0` —
    /// `custom_all_reduce.cu:467`.
    ///
    /// The 16-byte rule is the kernel's vector width, not a preference.
    Bytes {
        /// The message size asked for.
        bytes: usize,
        /// The configured ceiling.
        max_bytes: usize,
    },
    /// The element count is zero, or is not a multiple of the kernel's vector
    /// width — `vllm_custom_all_reduce.cuh:445-449`, which threw
    /// *"custom allreduce currently requires input length to be multiple of
    /// N"*.
    ///
    /// Distinct from [`Decline::Bytes`] because it is about ELEMENTS and is
    /// checked by the launch rather than by `can_handle`: the two disagree
    /// about units, and the C++ checked them in two different places for the
    /// same reason.
    Vector {
        /// The element count asked for.
        count: usize,
        /// [`VEC_SIZE`], the count it has to be a multiple of.
        width: i32,
    },
    /// `world_size > 2` and some ordered pair in the group has no peer
    /// access — `custom_all_reduce.cu:468`.
    ///
    /// Also what upstream's `REDUCE_CASE` did SILENTLY: its body launches
    /// nothing when `world_size_ != 2 && !full_nvlink_`, falls out of the
    /// `switch` and returns success having enqueued no kernel. That is a
    /// wrong answer with no diagnostic, and it is a refusal here.
    NotFullyConnected {
        /// The group size that would have needed it.
        world_size: i32,
    },
    /// `cudaStreamIsCapturing` itself failed — `custom_all_reduce.cu:470`.
    ///
    /// The C++ folded this into its `false`. It is separated here because a
    /// failing query is a broken stream and not a message that is too large,
    /// and the two want different things from a caller.
    CaptureUnknown,
    /// The input's base allocation was never `register_buffer`'d —
    /// `custom_all_reduce.cu:477`, and `vllm_custom_all_reduce.cuh:461-466`
    /// which threw *"buffer address N is not registered!"*.
    ///
    /// Reached here as a null `PeerPlane::rank_data`: the driver resolves the
    /// input's base to a `RankData*` before the call and hands over null when
    /// it cannot.
    Unregistered,
    /// Above the world-size crossover where NCCL wins on bandwidth —
    /// `custom_all_reduce.cu:483-485`.
    AboveCrossover {
        /// The message size asked for.
        bytes: usize,
        /// The threshold that applied.
        crossover: usize,
        /// The world size that selected it.
        world_size: i32,
    },
    /// No fusion workspace was built — `custom_all_reduce.cu:495`.
    ///
    /// The constructor builds one only for `world_size == 2` with both
    /// `fusion_max_tokens > 0` and `fusion_hidden > 0` (`:308`).
    NoFusionWorkspace,
    /// `tokens <= 0` or `tokens > fusion_max_tokens` —
    /// `custom_all_reduce.cu:496`.
    FusionTokens {
        /// What was asked for.
        tokens: i32,
        /// What the workspace was sized for.
        max_tokens: i32,
    },
    /// `hidden != fusion_hidden` — `custom_all_reduce.cu:497`.
    ///
    /// Equality, not a bound: the Lamport buffer's stride is baked into the
    /// workspace at construction.
    FusionHidden {
        /// What was asked for.
        hidden: i32,
        /// What the workspace was sized for.
        want: i32,
    },
    /// `world_size != 2` — `custom_all_reduce.cu:498`.
    FusionWorldSize {
        /// The group size.
        world_size: i32,
    },
    /// `hidden % 8 != 0` — `custom_all_reduce.cu:499`, and
    /// `allreduce_fusion_kernel_launcher`'s
    /// `FLASHINFER_CHECK(params.hidden_dim % VEC_SIZE == 0)`.
    FusionHiddenNotOctet {
        /// The hidden size that failed it.
        hidden: i32,
    },
    /// The fused kernel's block would be wider than a CUDA block can be.
    ///
    /// **Not upstream's refusal, and [`CLUSTER_SIZE`] is why it exists.**
    /// With one block per token the block is `hidden_dim / VEC_SIZE` threads;
    /// upstream would have spread the token over a cluster of up to eight
    /// blocks instead. The ceiling this makes is `hidden_dim <= 8192`.
    FusionBlockWidth {
        /// The hidden size asked for.
        hidden: i32,
        /// The threads per block it implies at [`CLUSTER_SIZE`].
        threads: i32,
        /// [`MAX_BLOCK_THREADS`].
        max: i32,
    },
    /// The two-shot fused kernel needs at least one thread per rank —
    /// `allreduce_fusion_kernel_launcher`'s
    /// `FLASHINFER_CHECK(oneshot || block_size >= params.nranks)`.
    FusionBlockNarrow {
        /// The threads per block the hidden size implies.
        threads: i32,
        /// The world size it has to cover.
        nranks: i32,
    },
    /// The pattern is not in [`INSTANTIATED`] —
    /// `custom_all_reduce.cu:171-176`, which threw.
    PatternNotInstantiated {
        /// The `AllReduceFusionPattern` discriminant asked for.
        code: i32,
    },
    /// The world size is not in [`NRANKS`] (fused) or [`PLAIN_NRANKS`]
    /// (plain) — `custom_all_reduce.cu:206-209` and
    /// `vllm_custom_all_reduce.cuh:495-499`, both of which threw.
    WorldSizeUnsupported {
        /// The world size asked for.
        nranks: i32,
    },
    /// The point resolved and [`Instantiation::name_expression`] carries no
    /// template-id for it.
    ///
    /// Unreachable from [`resolve`], which is what
    /// `the_table_covers_everything_resolve_admits` asserts. It exists
    /// because a lookup that can return `None` must not be `unwrap`ed on a
    /// launch path, and because the raw-pointer ABI forms can build an
    /// `Instantiation` without going through `resolve`.
    NoTemplateId {
        /// The world size that found no row.
        nranks: i32,
    },
    /// A device property the launch needs and the driver would not say.
    ///
    /// The multiprocessor count, which upstream got from
    /// `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` inside
    /// `get_sm_count()` — removed with the rest of the host half, and
    /// `jit::Ctx::multiprocessors` now.
    DeviceQuery {
        /// What could not be asked.
        what: &'static str,
    },
    /// The compile, the load or the launch itself refused.
    ///
    /// Everything upstream wrapped in `FLASHINFER_CUDA_CALL` around
    /// `cudaLaunchKernelEx`, plus the two things a JIT adds and an
    /// ahead-of-time build cannot have: NVRTC refusing the template-id, and
    /// `cuModuleLoadData` refusing the cubin. `jit::Ctx::launch` logs the
    /// detail once per instantiation; the [`Refusal`] is what a caller can
    /// branch on.
    Launch(Refusal),
}

impl fmt::Display for Decline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoInstance => write!(
                f,
                "this deployment configured no custom all-reduce, and the P2P \
                 reduction is stated; there is no other way to spell it"
            ),
            Self::NotInitialised => write!(f, "the custom all-reduce is not initialised"),
            Self::NullInput => write!(f, "`input` is null"),
            Self::Bytes { bytes, max_bytes } => write!(
                f,
                "{bytes} bytes is zero, above the {max_bytes}-byte ceiling, or not a multiple of 16"
            ),
            Self::Vector { count, width } => write!(
                f,
                "{count} elements is zero or not a multiple of {width}, the kernel's 16-byte \
                 vector width in bf16"
            ),
            Self::NotFullyConnected { world_size } => write!(
                f,
                "world size {world_size} needs peer access between every ordered pair and does not \
                 have it"
            ),
            Self::CaptureUnknown => write!(f, "`cudaStreamIsCapturing` failed on this stream"),
            Self::Unregistered => {
                write!(f, "the input's base allocation was never passed to `register_buffer`")
            }
            Self::AboveCrossover { bytes, crossover, world_size } => write!(
                f,
                "{bytes} bytes is at or above the {crossover}-byte crossover for world size \
                 {world_size}; NCCL wins on bandwidth here"
            ),
            Self::NoFusionWorkspace => write!(
                f,
                "no fusion workspace was built (world size 2 with a positive `fusion_max_tokens` \
                 and `fusion_hidden` is what builds one)"
            ),
            Self::FusionTokens { tokens, max_tokens } => {
                write!(f, "{tokens} tokens against a workspace sized for {max_tokens}")
            }
            Self::FusionHidden { hidden, want } => {
                write!(f, "hidden {hidden} against a workspace sized for exactly {want}")
            }
            Self::FusionWorldSize { world_size } => {
                write!(f, "the fused landing is world size 2 only; this group is {world_size}")
            }
            Self::FusionHiddenNotOctet { hidden } => {
                write!(f, "hidden {hidden} is not a multiple of 8")
            }
            Self::FusionBlockWidth { hidden, threads, max } => write!(
                f,
                "hidden {hidden} needs {threads} threads in one block and a block holds {max}; \
                 upstream would have spread this token over a cluster, and \
                 `comm::CLUSTER_SIZE` is pinned to 1"
            ),
            Self::FusionBlockNarrow { threads, nranks } => write!(
                f,
                "the two-shot fused kernel needs at least one thread per rank: {threads} threads \
                 for a world size of {nranks}"
            ),
            Self::PatternNotInstantiated { code } => write!(
                f,
                "`AllReduceFusionPattern` {code} is not in `comm::INSTANTIATED`; \
                 adding a pattern to a call site requires adding it there and to `comm::inst`"
            ),
            Self::WorldSizeUnsupported { nranks } => write!(
                f,
                "TP world size {nranks} is not instantiated (flashinfer's fused landing takes \
                 2, 4, 8, 16; vllm's plain reduction takes 2, 4, 6, 8)"
            ),
            Self::NoTemplateId { nranks } => write!(
                f,
                "`comm::inst` carries no template-id for world size {nranks}; the table and \
                 `comm::resolve` disagree"
            ),
            Self::DeviceQuery { what } => write!(f, "the driver would not say {what}"),
            Self::Launch(why) => write!(f, "the launch refused: {why}"),
        }
    }
}

/// What a reduction did.
///
/// The C++'s `void`-or-throw, with the ambiguity removed. `#[must_use]`
/// because ignoring this answer is the one way to get a wrong result out of
/// these functions: a declined call leaves the destination untouched, and a
/// caller that reads it anyway reads whatever was there — which for an
/// all-reduce is this rank's unreduced partial, a plausible-looking tensor
/// that is silently wrong on every rank but one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum AllReduce {
    /// The launch is on the stream.
    Launched,
    /// Nothing was enqueued. **Use `dist::all_reduce_bf16` for this one.**
    Declined(Decline),
}

// ── the parameter block, which is now an ABI ─────────────────────────────

/// `flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<T>`, mirrored.
///
/// **`#[repr(C)]`, and that is a change.** This struct used to be a
/// CHECKLIST — deliberately not `repr(C)`, because the header was not
/// vendored and the field order could not be read. It is vendored now, so
/// this is the layout, and it crosses to the device as a by-value kernel
/// parameter through [`ArgValue::Bytes`].
///
/// Measured against `kernels/flashinfer/comm/trtllm_allreduce_fusion.cuh`
/// at `T = __nv_bfloat16` with NVRTC 13.0, using `__INTADDR__` (the only
/// `offsetof` spelling that is a constant expression under NVRTC — see
/// `MODIFICATIONS`):
///
/// ```text
///   sizeof  136      alignof  8
///   workspace 16   allreduce_in 24   rms_gamma 80   rms_eps 88
///   weight_bias 92   scale_factor 96   use_oneshot 104   layout 108
///   stream 112   pattern 120   trigger_completion_at_end 124
///   block_quant_group_size 128   tma_aligned_mn 132
/// ```
///
/// `the_parameter_block_is_the_upstream_struct` restates every one of those
/// against Rust's own `offset_of!`, so a reordered field here is a test
/// failure rather than a wrong kernel argument.
///
/// # Two fields upstream has that this does not carry as data
///
/// `launch_with_pdl` is not a member of the struct at all — it was the
/// launcher's second argument, and the launcher is host C++ that is gone.
/// `use_fp32_acc` is the `Fp32Acc` TEMPLATE parameter, which is why it is
/// part of [`Instantiation`] and not of this block.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FusionParams {
    /// `params.nranks` — `:638`.
    pub nranks: i32,
    /// `params.rank` — `:639`.
    pub rank: i32,
    /// `params.size = tokens * hidden` — `:640`. Element count, not bytes.
    pub size: i32,
    /// `params.hidden_dim` — `:641`.
    pub hidden_dim: i32,
    /// `params.workspace` — `:642`, the device array of `3 * world + 1`
    /// pointers.
    pub workspace: *mut *mut c_void,
    /// `params.allreduce_in` — `:643`.
    pub allreduce_in: *mut c_void,
    /// `params.allreduce_out = nullptr` — `:644`. The unfused output is not
    /// wanted; only the normed one is.
    pub allreduce_out: *mut c_void,
    /// `params.residual_in` — `:645`.
    pub residual_in: *mut c_void,
    /// `params.residual_out` — `:646`. **The same pointer as
    /// `residual_in`**, which is the `in_place = &[(0, 1)]` the row states.
    pub residual_out: *mut c_void,
    /// `params.norm_out` — `:647`.
    pub norm_out: *mut c_void,
    /// `params.quant_out = nullptr` — `:648`. Set only by the FP8/FP4
    /// patterns pie does not reach.
    pub quant_out: *mut c_void,
    /// `params.scale_out = nullptr` — `:649`.
    pub scale_out: *mut c_void,
    /// `params.rms_gamma` — `:650`.
    pub rms_gamma: *mut c_void,
    /// `params.rms_eps` — `:651`.
    pub rms_eps: f32,
    /// `params.weight_bias`, upstream's `= 0.f` default.
    ///
    /// **Not a field the archive's `custom_all_reduce.cu` set**, because the
    /// header it compiled against did not have one. The vendored header does:
    /// `0` is standard RMSNorm and `1` is Gemma/Qwen3.5's `(1 + gamma)`. A
    /// `#[repr(C)]` mirror has to carry it whatever pie sets, and pie sets 0.
    pub weight_bias: f32,
    /// `params.scale_factor = nullptr` — `:652`.
    pub scale_factor: *mut f32,
    /// `params.use_oneshot = true` — `:653`. One half of [`Leaf`].
    pub use_oneshot: bool,
    /// `params.layout = QuantizationSFLayout::SWIZZLED_128x4` — `:654`.
    pub layout: SfLayout,
    /// `params.stream` — `:655`.
    ///
    /// **Read by nothing on the device.** It is a `cudaStream_t` the host
    /// launcher put into `cudaLaunchConfig_t`; here the stream is
    /// `Ctx::stream()` and the field is carried because it is four bytes of
    /// alignment and eight of padding in the middle of a struct whose layout
    /// has to match.
    pub stream: *mut c_void,
    /// `params.pattern` — `:656`.
    pub pattern: FusionPattern,
    /// `params.trigger_completion_at_end = false` — `:657`. The other half
    /// of [`Leaf`].
    pub trigger_completion_at_end: bool,
    /// `params.block_quant_group_size`, upstream's `= 0` default. Read only
    /// by `kPerTokenGroupFP8Packed`.
    pub block_quant_group_size: i32,
    /// `params.tma_aligned_mn`, upstream's `= 0` default. Read only by
    /// `kPerTokenGroupFP8Packed`.
    pub tma_aligned_mn: i32,
}

impl FusionParams {
    /// The [`Instantiation`] this parameter block plus `use_fp32_acc`
    /// selects.
    ///
    /// # Errors
    ///
    /// Whatever [`resolve`] refuses.
    pub fn instantiation(
        &self,
        use_fp32_acc: bool,
    ) -> std::result::Result<Instantiation, Decline> {
        resolve(
            self.nranks,
            self.pattern,
            use_fp32_acc,
            self.use_oneshot,
            self.trigger_completion_at_end,
        )
    }

    /// The block, as `cuLaunchKernelEx` takes a by-value aggregate.
    ///
    /// # Safety
    ///
    /// The returned [`ArgValue::Bytes`] borrows `self`, and the launch copies
    /// the bytes out before returning — so `self` must outlive the
    /// [`Ctx::launch`] call it is passed to, which it does at both call sites
    /// because it is a local in the same frame.
    fn arg(&self) -> ArgValue {
        ArgValue::Bytes {
            ptr: std::ptr::from_ref(self).cast::<u8>(),
            len: core::mem::size_of::<Self>(),
        }
    }
}

// ── the geometry, which used to be two host launchers ────────────────────

/// One launch rectangle and the template-id that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Blocks.
    pub grid: u32,
    /// Threads per block.
    pub block: u32,
}

/// `allreduce_fusion_kernel_launcher`'s grid arithmetic, at
/// [`CLUSTER_SIZE`] — the vendored header's `:1547-1659`, upstream's
/// numbering.
///
/// # What this DOES compute, line for line
///
/// ```text
///   threads_per_token = hidden_dim / VEC_SIZE                   (:1547)
///   threads_per_block = threads_per_token / cluster_size        (:1557)
///   cluster_num       = token_num                               (:1533, oneshot)
///                     = ceil(token_num / NRanks)                (:1538, twoshot)
///   grid_size = (min(sm_count, cluster_num * cluster_size)
///                / cluster_size) * cluster_size                 (:1659)
/// ```
///
/// with `cluster_size == 1`, which collapses the last line to
/// `min(sm_count, cluster_num)`.
///
/// # What it does NOT compute, and why each is safe to omit at cluster size 1
///
/// * **`cluster_size = SM >= 90 ? 8 : 1` and the two loops that shrink it**
///   (`:1549-1561`) — [`CLUSTER_SIZE`] carries that decision and its cost.
/// * **`registers_per_thread` / `max_threads_per_block`** (`:1563-1577`) —
///   its only reader is `adjust_for_sm_count`, whose loop guard is
///   `cluster_size_ref > 1`. Dead at cluster size 1.
/// * **The FP4 block-size search** (`:1595-1615`) — guarded by
///   `if constexpr (GetQuantType<Pattern> == kFP4)`, and [`INSTANTIATED`] has
///   one pattern whose quant type is `kNone`.
/// * **The `kPerTokenGroupFP8Packed` group alignment and dynamic shared
///   memory** (`:1635-1668`) — same guard, same reason. `dynamicSmemBytes` is
///   0 for every pattern pie names, which is why [`Geometry`] has no `smem`.
/// * **`cudaLaunchAttributeProgrammaticStreamSerialization`** — `launch_with_pdl`
///   is `false` at pie's only call site (`custom_all_reduce.cu:661`), so the
///   attribute is set to 0, which is the same as not setting it.
///
/// # Errors
///
/// [`Decline::FusionHiddenNotOctet`] for a `hidden` the vector width does not
/// divide, [`Decline::FusionBlockWidth`] above [`MAX_BLOCK_THREADS`],
/// [`Decline::FusionBlockNarrow`] for a two-shot launch with fewer threads
/// than ranks, and [`Decline::FusionTokens`] for a non-positive token count.
pub fn fusion_geometry(
    tokens: i32,
    hidden: i32,
    nranks: i32,
    leaf: Leaf,
    multiprocessors: u32,
) -> std::result::Result<Geometry, Decline> {
    if tokens <= 0 {
        return Err(Decline::FusionTokens { tokens, max_tokens: 0 });
    }
    if hidden <= 0 || hidden % VEC_SIZE != 0 {
        return Err(Decline::FusionHiddenNotOctet { hidden });
    }
    // `:1547`, `:1557`. `CLUSTER_SIZE` is 1, so the division is the identity
    // and the two loops that would have shrunk the cluster are no-ops.
    let threads_per_token = hidden / VEC_SIZE;
    let threads_per_block = threads_per_token / CLUSTER_SIZE;
    if threads_per_block > MAX_BLOCK_THREADS {
        return Err(Decline::FusionBlockWidth {
            hidden,
            threads: threads_per_block,
            max: MAX_BLOCK_THREADS,
        });
    }
    // `:1655` -- `FLASHINFER_CHECK(oneshot || block_size >= params.nranks)`.
    if !leaf.oneshot() && threads_per_block < nranks {
        return Err(Decline::FusionBlockNarrow { threads: threads_per_block, nranks });
    }
    // `:1533` for one-shot, `:1536-1541` for two-shot.
    let cluster_num = if leaf.oneshot() {
        tokens
    } else {
        let per_rank = tokens / nranks;
        per_rank + i32::from(tokens % nranks != 0)
    };
    // `:1659`, with `cluster_size == 1`, so the divide-then-multiply is the
    // identity and this is `min(sm_count, cluster_num)`.
    //
    // The `.max(1)` is NOT upstream's and is one line of paranoia:
    // `cluster_num` is at least 1 because `tokens > 0` is checked above, so
    // the only way to reach zero is a device reporting no multiprocessors.
    // Upstream would launch a zero-block grid there; `Ctx::launch` refuses one
    // as `Refusal::Empty`, which would be a confusing way to hear about a
    // broken driver.
    let sm_count = i32::try_from(multiprocessors).unwrap_or(i32::MAX);
    let grid = cluster_num.min(sm_count).max(1);
    Ok(Geometry {
        grid: u32::try_from(grid).unwrap_or(1),
        block: u32::try_from(threads_per_block).unwrap_or(1),
    })
}

/// The two-shot kernel's per-rank token split — `:1536-1545`.
///
/// `begin_tokens[r]` and `token_num_per_ranks[r]`, the two
/// `std::array<int, NRanks>` the two-shot `__global__` takes by value. Both
/// are `nranks` `int`s and nothing else, which is what `std::array`'s layout
/// is and what `shim/array`'s banner argues at length.
///
/// Returned as a fixed 16-wide pair because [`NRANKS`]' largest is 16 and a
/// `Vec` on a launch path is an allocation per fire; only the first `nranks`
/// entries are handed over.
#[must_use]
pub fn twoshot_split(tokens: i32, nranks: i32) -> ([i32; 16], [i32; 16]) {
    let mut begin = [0i32; 16];
    let mut count = [0i32; 16];
    let per_rank = tokens / nranks;
    let remaining = tokens % nranks;
    for r in 0..nranks.clamp(0, 16) {
        let at = r as usize;
        begin[at] = r * per_rank + remaining.min(r);
        count[at] = per_rank + i32::from(remaining > r);
    }
    (begin, count)
}

/// `vllm::CustomAllreduce::allreduce`'s grid arithmetic and stage choice —
/// `vllm_custom_all_reduce.cuh:441-503`.
///
/// ```text
///   d      = 16 / sizeof(T)                                 (:444)
///   size  /= d                                              (:469)
///   bytes  = size * sizeof(packed_t<T>::P)   -- i.e. size * 16   (:470)
///   blocks = min(block_limit, ceil(size / threads))         (:471)
/// ```
///
/// and then `REDUCE_CASE`'s stage choice (`:476-488`): world size 2 is always
/// one-stage; wider needs full NVLink, and picks one-stage only for messages
/// under 512 KiB at 4 ranks or 256 KiB at 8.
///
/// The returned `size` is the DIVIDED one — what the kernel's last parameter
/// takes, in 16-byte vectors and not in elements.
///
/// # Errors
///
/// [`Decline::Vector`] for a count the vector width does not divide,
/// [`Decline::WorldSizeUnsupported`] outside [`PLAIN_NRANKS`], and
/// [`Decline::NotFullyConnected`] for the case upstream launched nothing at
/// all for.
pub fn plain_geometry(
    count: usize,
    world_size: i32,
    fully_connected: bool,
) -> std::result::Result<(Geometry, Stage, i32), Decline> {
    /// The 512 threads `custom_all_reduce.cu:614` pins on every plain P2P
    /// all-reduce, and the 36 blocks beside it (`:613`).
    ///
    /// Not a tuning knob and not derived from a shape: the vllm kernel's
    /// one-shot and two-shot bodies both assume a fixed cooperative rectangle,
    /// and `__launch_bounds__(512, 1)` is on both `__global__`s.
    pub const ALL_REDUCE_THREADS: i32 = 512;

    /// `vllm::kMaxBlocks` — `vllm_custom_all_reduce.cuh:46`.
    ///
    /// The same 36 that `custom_all_reduce.cu:613` passes as `block_limit`. The
    /// two agreeing is not a coincidence to be preserved by hand: the launcher
    /// clamps its grid to this, so a `block_limit` above it would index off the
    /// end of `Signal::self_counter[kMaxBlocks][8]` — a 3,456-byte struct — and
    /// one below it would leave bandwidth on the floor.
    pub const MAX_BLOCKS: i32 = 36;

    let width = usize::try_from(VEC_SIZE).unwrap_or(8);
    if count == 0 || !count.is_multiple_of(width) {
        return Err(Decline::Vector { count, width: VEC_SIZE });
    }
    if !PLAIN_NRANKS.contains(&world_size) {
        return Err(Decline::WorldSizeUnsupported { nranks: world_size });
    }
    // `:469-470`. `size` is now a count of 16-byte vectors.
    let vectors = count / width;
    let bytes = vectors * 16;
    let size = i32::try_from(vectors).unwrap_or(i32::MAX);

    // `:476-488`. Upstream's `else` here launches NOTHING and returns, which
    // is a silently wrong answer; it is a refusal instead.
    let stage = if world_size == 2 {
        Stage::OneStage
    } else if !fully_connected {
        return Err(Decline::NotFullyConnected { world_size });
    } else if (world_size <= 4 && bytes < 512 * 1024) || (world_size <= 8 && bytes < 256 * 1024) {
        Stage::OneStage
    } else {
        Stage::TwoStage
    };

    // `:471`.
    let threads = ALL_REDUCE_THREADS;
    let blocks = MAX_BLOCKS.min(size.div_euclid(threads) + i32::from(size % threads != 0)).max(1);
    Ok((
        Geometry {
            grid: u32::try_from(blocks).unwrap_or(1),
            block: u32::try_from(threads).unwrap_or(512),
        },
        stage,
        size,
    ))
}

/// The template-id for a plain reduction at this world size and stage.
#[must_use]
pub fn plain_name_expression(world_size: i32, stage: Stage) -> Option<&'static str> {
    match (world_size, stage) {
        (2, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>"),
        (2, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 2>"),
        (4, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 4>"),
        (4, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 4>"),
        (6, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 6>"),
        (6, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 6>"),
        (8, Stage::OneStage) => Some("::vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>"),
        (8, Stage::TwoStage) => Some("::vllm::cross_device_reduce_2stage<__nv_bfloat16, 8>"),
        _ => None,
    }
}

// ── the plane, which is the seam ─────────────────────────────────────────

/// The fusion plane's three facts, as a launch reads them.
///
/// `Fusion` up in `driver-cuda` owns five fields; two of them —
/// `buffers: [*mut c_void; 3]` and `flag_dev` — are what the CONSTRUCTOR
/// allocated and initialised, and no launch ever reads either. They are not
/// here, because carrying a field a reader does not read is how a mirror
/// starts disagreeing with the thing it mirrors.
#[derive(Debug, Clone, Copy)]
pub struct FusionPlane {
    /// `fusion_workspace_dev_` — the `3 * world + 1` pointer array,
    /// `custom_all_reduce.cu:391`. Becomes `params.workspace`.
    pub workspace: *mut c_void,
    /// `fusion_max_tokens_`.
    pub max_tokens: i32,
    /// `fusion_hidden_`.
    pub hidden: i32,
}

/// What the PLAIN reduction reads off the instance, which [`FusionPlane`]
/// does not carry any of.
///
/// # Why this is separate, and why it arrived late
///
/// The two kernels do not share an addressing scheme. The fused landing takes
/// ONE pointer — `params.workspace`, a device array the constructor filled —
/// and derives every peer address on the device. The plain kernels take three
/// things the host has to hand over: the peer `Signal*` array by value, this
/// rank's own `Signal*`, and a `RankData*` naming the eight peer addresses of
/// the input's base allocation.
///
/// So a `Plane` carrying only `world_size`, `rank` and the fusion workspace
/// was enough for the fused launcher and not for the plain one, and this is
/// the widening. It is recorded because the gap was measured before it was
/// closed: vendoring the headers alone would not have finished the plain arm.
///
/// # `rank_data` is PER CALL and the other three are not
///
/// `signals`, `self_signal` and `fully_connected` are what the constructor
/// established. `rank_data` is `buffers_[input]` — the slot
/// `register_buffer` wrote for THIS input's base allocation — so the driver
/// resolves it per fire and hands over null when the input was never
/// registered, which is [`Decline::Unregistered`].
///
/// They travel together anyway, because a plane with three of the four is a
/// plane no launch can use, and two `Option`s would make four states where
/// there are two.
#[derive(Debug, Clone, Copy)]
pub struct PeerPlane {
    /// `RankSignals::signals[8]` — `vllm_custom_all_reduce.cuh:66-68`, passed
    /// to the kernel BY VALUE as a 64-byte aggregate.
    ///
    /// Eight and not `world_size`, because upstream's struct is
    /// `Signal* signals[8]` at any world size and the kernel reads
    /// `sg.signals[threadIdx.x]` under `if (threadIdx.x < ngpus)`. Entries at
    /// or past `world_size` are never read; the driver zeroes them.
    pub signals: [*mut c_void; 8],
    /// `self_sg_` — `signals[rank]`. Null means the plane was never
    /// initialised, which is [`Decline::NotInitialised`].
    pub self_signal: *mut c_void,
    /// The `RankData*` the input's base allocation was registered into, or
    /// null. See this type's doc for why it is per call.
    pub rank_data: *mut c_void,
    /// `full_nvlink_` — whether every ordered pair of the group has peer
    /// access. Read only above world size 2.
    pub fully_connected: bool,
}

/// What a `CustomAllReduce` tells a launch about itself.
///
/// The whole of what either body below reads off the instance, and the seam
/// the descent is drawn on: the lifecycle stays in `driver-cuda`, which fills
/// one of these per call.
///
/// It carries no communicator, no IPC map and no registered-buffer set,
/// because neither body needs one. `CustomAllReduce::can_handle` does need
/// them — it walks the registered-buffer map and queries the stream's capture
/// state — and that is exactly why `can_handle` stayed up while
/// [`Plane::can_fuse_residual_rmsnorm`] came down: the two predicates read
/// different things, and only one of them reads the plane alone.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// The group size.
    pub world_size: i32,
    /// This rank's index in it.
    pub rank: i32,
    /// The fused landing's workspace, when one was built. `None` is
    /// [`Decline::NoFusionWorkspace`]: the constructor builds one only for
    /// `world_size == 2` with both fusion dimensions positive
    /// (`custom_all_reduce.cu:308`).
    pub fusion: Option<FusionPlane>,
    /// What the plain reduction addresses through — see [`PeerPlane`].
    pub peers: PeerPlane,
}

impl Plane {
    /// `custom_all_reduce.cu:488-495` — `can_fuse_residual_rmsnorm`.
    ///
    /// The C++ took a `cudaStream_t /*stream*/` it never read; the parameter
    /// is gone rather than kept as `_stream`, because unlike `buf_bytes` it
    /// answers no question that could later be asked of it.
    ///
    /// # Errors
    ///
    /// The [`Decline`] that the corresponding `return false` stood for.
    pub fn can_fuse_residual_rmsnorm(
        &self,
        tokens: i32,
        hidden: i32,
    ) -> std::result::Result<(), Decline> {
        // `:490`.
        let Some(fusion) = self.fusion.as_ref() else {
            return Err(Decline::NoFusionWorkspace);
        };
        // `:491`.
        if tokens <= 0 || tokens > fusion.max_tokens {
            return Err(Decline::FusionTokens { tokens, max_tokens: fusion.max_tokens });
        }
        // `:492`.
        if hidden != fusion.hidden {
            return Err(Decline::FusionHidden { hidden, want: fusion.hidden });
        }
        // `:493`. The fusion plane is built for TP=2 only (`:308`), so this
        // is unreachable from a constructed plane -- kept because the C++
        // kept it and because it is the invariant, not a consequence.
        if self.world_size != 2 {
            return Err(Decline::FusionWorldSize { world_size: self.world_size });
        }
        // `:494`. The kernel's vector width in bf16 elements.
        if hidden % VEC_SIZE != 0 {
            return Err(Decline::FusionHiddenNotOctet { hidden });
        }
        Ok(())
    }
}

// ── the two symbols ──────────────────────────────────────────────────────

/// `comm::all_reduce_bf16` — `custom_all_reduce.cu:603-621`, the plain bf16
/// in-place all-reduce.
///
/// `count` is an ELEMENT count, not bytes (`custom_all_reduce.hpp:100`).
///
/// # The line that has now crossed
///
/// `impl_->allreduce<__nv_bfloat16>(stream, in, out, count, 36, 512)`. The
/// two constants are [`MAX_BLOCKS`] and [`ALL_REDUCE_THREADS`]; `36` is not a
/// choice, it is `vllm_custom_all_reduce.cuh:46`'s `kMaxBlocks` and the first
/// dimension of the `Signal` counters, so a larger grid would index off the
/// end of a 3,456-byte struct.
///
/// `(int)count` at `:618` was a silent narrowing. It is
/// [`Decline::Vector`]'s and [`plain_geometry`]'s `i32::try_from` now, and it
/// still cannot bite: `can_handle` refuses above `max_bytes`, 8 MiB by
/// default, which is 4 Mi bf16 elements.
///
/// # What the kernel reads, and what it does not
///
/// **`input` is not a kernel argument.** The four peer addresses come from
/// `*rank_data`, which `register_buffer` wrote at the input's BASE
/// allocation, and the kernel indexes them itself. `input` is used here only
/// to be checked for null — upstream used it to look the slot up, which the
/// driver does before the call. That asymmetry is upstream's and it is worth
/// knowing: registering the base and passing an offset pointer would reduce
/// the wrong bytes.
///
/// Nothing is appended to the driver's `graph_unreg_buffers` from here, and
/// that stays deliberate: a deferred registration for a launch that declined
/// would bind the next real one to the wrong slot. The driver appends after
/// [`AllReduce::Launched`].
///
/// # Safety
///
/// `input` and `output` must address at least `count` live bf16 elements, and
/// `plane` must describe a live P2P plane whose peers are mapped for the
/// duration of the launch. **Nothing here checks either.**
#[must_use]
pub fn all_reduce_bf16(
    ctx: &Ctx,
    plane: Plane,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
) -> AllReduce {
    // `custom_all_reduce.cu:466`. `input` and not `output`, which is
    // upstream's asymmetry and is kept: a null `output` would be reported as
    // `NullInput`, and a refusal that names the wrong pointer is worse than
    // the `# Safety` clause that already covers it.
    if input.is_null() {
        return AllReduce::Declined(Decline::NullInput);
    }
    // The moved-from state the C++ called `!impl_`.
    if plane.peers.self_signal.is_null() {
        return AllReduce::Declined(Decline::NotInitialised);
    }
    // `vllm_custom_all_reduce.cuh:461-466`, resolved by the driver.
    if plane.peers.rank_data.is_null() {
        return AllReduce::Declined(Decline::Unregistered);
    }

    let (geometry, stage, size) =
        match plain_geometry(count, plane.world_size, plane.peers.fully_connected) {
            Ok(what) => what,
            Err(decline) => return AllReduce::Declined(decline),
        };
    let Some(instantiation) = plain_name_expression(plane.world_size, stage) else {
        return AllReduce::Declined(Decline::NoTemplateId { nranks: plane.world_size });
    };

    // `RankSignals` is `Signal* signals[8]` at `__align__(16)`, taken BY
    // VALUE. `[*mut c_void; 8]` is eight pointers with no padding, which is
    // the same 64 bytes; the alignment upstream asks for is about the
    // parameter's placement, and `cuLaunchKernelEx` copies the bytes into a
    // parameter buffer it aligns itself.
    let signals = plane.peers.signals;
    let signals_arg = ArgValue::Bytes {
        ptr: std::ptr::from_ref(&signals).cast::<u8>(),
        len: core::mem::size_of::<[*mut c_void; 8]>(),
    };

    // `:473` -- `name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_,
    // self_sg_, output, rank_, size)`.
    //
    // SAFETY: the caller's contract. Every pointer bound here addresses live
    // device memory of the extent the kernel reads it as, and `signals`
    // outlives the call because the launch copies the aggregate out.
    let fired = unsafe {
        ctx.launch(
            "comm/all_reduce.cuh",
            instantiation,
            Launch::grid([geometry.grid, 1, 1], [geometry.block, 1, 1]),
            &[
                ArgValue::Ptr(plane.peers.rank_data),
                signals_arg,
                ArgValue::Ptr(plane.peers.self_signal),
                ArgValue::Ptr(output),
                ArgValue::I32(plane.rank),
                ArgValue::I32(size),
            ],
        )
    };
    match fired {
        Ok(()) => AllReduce::Launched,
        Err(why) => AllReduce::Declined(Decline::Launch(why)),
    }
}

/// `comm::all_reduce_residual_rmsnorm_bf16` — `custom_all_reduce.cu:623-662`,
/// the fused all-reduce + residual add + RMSNorm.
///
/// The C++ threw when `can_fuse_residual_rmsnorm` said no (`:633-635`); that
/// throw is the [`Decline`] the query returns, unchanged, which is the whole
/// point of the query returning one.
///
/// # What crosses
///
/// [`FusionParams`] by value, and nothing else: every peer address the kernel
/// touches is derived on the device from `params.workspace`, the `3*world+1`
/// pointer array the constructor built. That is why this arm needed no
/// [`PeerPlane`] and the plain one did.
///
/// # Safety
///
/// `input`, `residual_inout` and `norm_out` must address at least
/// `tokens * hidden` live bf16 elements and `rms_gamma` at least `hidden`;
/// `plane.fusion.workspace` must be the array the constructor filled, with
/// every peer's Lamport buffer mapped for the duration of the launch.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn all_reduce_residual_rmsnorm_bf16(
    ctx: &Ctx,
    plane: Plane,
    input: *const c_void,
    residual_inout: *mut c_void,
    rms_gamma: *const c_void,
    norm_out: *mut c_void,
    tokens: i32,
    hidden: i32,
    eps: f32,
) -> AllReduce {
    if let Err(decline) = plane.can_fuse_residual_rmsnorm(tokens, hidden) {
        return AllReduce::Declined(decline);
    }
    let Some(fusion) = plane.fusion.as_ref() else {
        return AllReduce::Declined(Decline::NoFusionWorkspace);
    };
    // `:660`. `constexpr bool use_fp32_acc = true`, the `Fp32Acc` template
    // parameter -- not a field of the struct, which is why it is a local.
    let use_fp32_acc = true;
    // `:637-659`, field for field and in the file's order.
    let params = FusionParams {
        nranks: plane.world_size,
        rank: plane.rank,
        size: tokens.saturating_mul(hidden),
        hidden_dim: hidden,
        workspace: fusion.workspace.cast::<*mut c_void>(),
        allreduce_in: input.cast_mut(),
        allreduce_out: core::ptr::null_mut(),
        residual_in: residual_inout,
        residual_out: residual_inout,
        norm_out,
        quant_out: core::ptr::null_mut(),
        scale_out: core::ptr::null_mut(),
        rms_gamma: rms_gamma.cast_mut(),
        rms_eps: eps,
        weight_bias: 0.0,
        scale_factor: core::ptr::null_mut(),
        use_oneshot: true,
        layout: SfLayout::Swizzled128x4,
        stream: ctx.stream(),
        pattern: FusionPattern::ARResidualRMSNorm,
        trigger_completion_at_end: false,
        block_quant_group_size: 0,
        tma_aligned_mn: 0,
    };
    let point = match params.instantiation(use_fp32_acc) {
        Ok(point) => point,
        Err(decline) => return AllReduce::Declined(decline),
    };
    let Some(instantiation) = point.name_expression() else {
        return AllReduce::Declined(Decline::NoTemplateId { nranks: point.nranks });
    };
    let multiprocessors = match ctx.multiprocessors() {
        Ok(count) => count,
        Err(_) => {
            return AllReduce::Declined(Decline::DeviceQuery {
                what: "how many multiprocessors this device has",
            });
        }
    };
    let geometry = match fusion_geometry(tokens, hidden, point.nranks, point.leaf, multiprocessors)
    {
        Ok(geometry) => geometry,
        Err(decline) => return AllReduce::Declined(decline),
    };
    let launch = Launch::grid([geometry.grid, 1, 1], [geometry.block, 1, 1]);

    // The one-shot leaf takes the params block alone; the two-shot takes two
    // `std::array<int, NRanks>` after it. Neither has an alternative: the
    // template-id names one `__global__` and a `__global__` has one parameter
    // list.
    //
    // SAFETY: the caller's contract. `params`, `begin` and `count` are locals
    // of this frame and the launch copies every aggregate out before it
    // returns.
    let fired = if point.leaf.oneshot() {
        unsafe { ctx.launch("comm/all_reduce.cuh", instantiation, launch, &[params.arg()]) }
    } else {
        let (begin, per_rank) = twoshot_split(tokens, point.nranks);
        let bytes = core::mem::size_of::<i32>()
            * usize::try_from(point.nranks).unwrap_or(0).min(begin.len());
        unsafe {
            ctx.launch(
                "comm/all_reduce.cuh",
                instantiation,
                launch,
                &[
                    params.arg(),
                    ArgValue::Bytes { ptr: begin.as_ptr().cast::<u8>(), len: bytes },
                    ArgValue::Bytes { ptr: per_rank.as_ptr().cast::<u8>(), len: bytes },
                ],
            )
        }
    };
    match fired {
        Ok(()) => AllReduce::Launched,
        Err(why) => AllReduce::Declined(Decline::Launch(why)),
    }
}

/// The P2P collectives' two symbols, declared so a sharded model text
/// resolves.
///
/// `driver_bound!` and not `routine!`: a collective takes a COMMUNICATOR, and
/// a communicator is a property of the deployment's process group that no
/// trace statement carries or could. The `dist::` three say the same thing
/// about NCCL; the difference between the two families is that these have
/// device text and those have nothing at all.
///
/// Both `whole`, and the fused landing's `in_place` pair — the module header
/// carries both arguments.
pub static ROUTINES: &[Routine] = &[
    driver_bound!(all_reduce_bf16, whole),
    driver_bound!(all_reduce_residual_rmsnorm_bf16, whole, in_place = &[(0, 1)]),
];

/// `comm`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

#[cfg(test)]
mod tests {
    use super::{
        AOT_POINTS_AFTER_PRUNING, AllReduce, CAN_LAUNCH, CLUSTER_SIZE, Decline, FusionParams,
        FusionPattern, FusionPlane, INSTANTIATED, Instantiation, Leaf, NRANKS, PLAIN_NRANKS,
        Plane, PeerPlane, REACHED, Stage, UPSTREAM_POINTS, VEC_SIZE, all_reduce_bf16,
        all_reduce_residual_rmsnorm_bf16, fusion_geometry, plain_geometry,
        plain_name_expression, resolve, twoshot_split,
    };
    use crate::jit::Ctx;

    /// A plane sized to admit the one point pie reaches, with no live device
    /// memory behind any of it.
    ///
    /// Every field is non-null where a null would be refused BEFORE the
    /// launch, because a test whose subject is a shape check must not be
    /// passing for want of a pointer. Nothing here is dereferenced on the
    /// host, and nothing in this module launches on the tests' path except
    /// `the_bodies_agree_with_can_launch`, which is why that one is the only
    /// test with a `#[cfg]`-shaped caveat in its own doc.
    fn plane() -> Plane {
        Plane {
            world_size: REACHED.nranks,
            rank: 0,
            fusion: Some(FusionPlane {
                workspace: core::ptr::dangling_mut(),
                max_tokens: 8,
                hidden: 4096,
            }),
            peers: PeerPlane {
                signals: [core::ptr::dangling_mut(); 8],
                self_signal: core::ptr::dangling_mut(),
                rank_data: core::ptr::dangling_mut(),
                fully_connected: true,
            },
        }
    }

    /// [`CAN_LAUNCH`] is what a DEPLOYMENT reads to decide whether a
    /// tensor-parallel rank can combine its shards, so it must not be able to
    /// drift from what the bodies actually do.
    ///
    /// **This test does not launch and cannot**, which is the shape the flip
    /// to `true` forced. While `CAN_LAUNCH` was `false` both bodies answered
    /// `Decline::NoDeviceText` on a resolved point and the test called them.
    /// Calling them now would reach `Ctx::launch`, which compiles device text
    /// and needs a GPU, so the test would either need one or would pass by
    /// finding a refusal from the wrong layer.
    ///
    /// So what it asserts is the property that remains checkable without a
    /// device, and it is the one that matters: **`CAN_LAUNCH` is true exactly
    /// when there is a template-id for the reached point.** A body cannot
    /// launch without one, and no other decline in either body is a statement
    /// about this repository — every one of them is a statement about the
    /// call. `every_instantiation_compiles` is the other half, on a box with
    /// NVRTC.
    #[test]
    fn the_bodies_agree_with_can_launch() {
        let reachable = REACHED.name_expression().is_some();
        assert_eq!(
            CAN_LAUNCH, reachable,
            "`CAN_LAUNCH` is {CAN_LAUNCH} and the reached point {} a template-id in \
             `comm::Instantiation::name_expression`; a launch needs one and nothing else \
             supply it",
            if reachable { "has" } else { "has no" }
        );
        assert!(
            plain_name_expression(2, Stage::OneStage).is_some() == CAN_LAUNCH,
            "the plain arm's world-size-2 point must be reachable on the same terms"
        );
    }

    /// A refusal that is a statement about the CALL still reaches the caller
    /// without a device.
    ///
    /// The two bodies check their shapes before they reach `Ctx::launch`, so
    /// a call that is going to be refused is refused on a box with no GPU —
    /// which is what makes `bind/arms/comm.rs`' fallback to NCCL a decision
    /// taken before any device work.
    #[test]
    fn a_shape_refusal_needs_no_device() {
        // SAFETY: the null stream is CUDA's legal default, and nothing here
        // launches -- every call below is refused before `Ctx::launch`.
        let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };

        let null_input = all_reduce_bf16(
            &ctx,
            plane(),
            core::ptr::null(),
            core::ptr::dangling_mut(),
            4096,
        );
        assert_eq!(null_input, AllReduce::Declined(Decline::NullInput));

        let mut unregistered = plane();
        unregistered.peers.rank_data = core::ptr::null_mut();
        let refused = all_reduce_bf16(
            &ctx,
            unregistered,
            core::ptr::dangling(),
            core::ptr::dangling_mut(),
            4096,
        );
        assert_eq!(refused, AllReduce::Declined(Decline::Unregistered));

        // `hidden` disagreeing with the workspace is `can_fuse_residual_rmsnorm`'s
        // and is refused before anything is computed.
        let fused = all_reduce_residual_rmsnorm_bf16(
            &ctx,
            plane(),
            core::ptr::dangling(),
            core::ptr::dangling_mut(),
            core::ptr::dangling(),
            core::ptr::dangling_mut(),
            1,
            2048,
            1e-6,
        );
        assert_eq!(
            fused,
            AllReduce::Declined(Decline::FusionHidden { hidden: 2048, want: 4096 })
        );
    }

    /// The cross product is host arithmetic over an upstream struct layout
    /// and touches no device, which is the whole reason these are tests and
    /// not comments.
    #[test]
    fn the_cross_product_is_the_number_kernels_def_measured() {
        assert_eq!(UPSTREAM_POINTS, 240);
        // `4 nranks x 1 pattern x 2 fp32_acc x 3 leaves`.
        assert_eq!(AOT_POINTS_AFTER_PRUNING, 24);
        assert_eq!(INSTANTIATED.len(), 1);
        assert_eq!(INSTANTIATED[0], FusionPattern::ARResidualRMSNorm);
    }

    /// The instantiation set is the pruned cross product, point for point.
    ///
    /// The arms are 24 hand-written strings and the number they must come to
    /// is derived from four other constants, so this is the assertion that
    /// keeps a hand-written set from silently becoming a different lattice
    /// than the one the module's header describes. Counting DISTINCT ids is
    /// what makes it real: three arms that returned one string would cover the
    /// product and compile one kernel.
    #[test]
    fn the_instantiation_table_is_the_pruned_cross_product() {
        let mut fused: Vec<&'static str> = Vec::new();
        for &nranks in NRANKS {
            for &pattern in INSTANTIATED {
                for fp32_acc in [true, false] {
                    for leaf in [Leaf::OneShot, Leaf::OneShotTriggerAtEnd, Leaf::TwoShot] {
                        let point = Instantiation { nranks, pattern, fp32_acc, leaf };
                        fused.push(point.name_expression().expect("covered"));
                    }
                }
            }
        }
        assert_eq!(fused.len(), AOT_POINTS_AFTER_PRUNING);
        fused.sort_unstable();
        fused.dedup();
        assert_eq!(fused.len(), AOT_POINTS_AFTER_PRUNING, "two points share a template-id");

        let mut plain: Vec<&'static str> = PLAIN_NRANKS
            .iter()
            .flat_map(|&n| {
                [Stage::OneStage, Stage::TwoStage]
                    .map(|s| plain_name_expression(n, s).expect("covered"))
            })
            .collect();
        plain.sort_unstable();
        plain.dedup();
        assert_eq!(plain.len(), PLAIN_NRANKS.len() * 2);
    }

    /// Every point [`resolve`] admits has a template-id.
    ///
    /// This is what makes [`Decline::NoTemplateId`] unreachable from the
    /// resolved path, and it is quantified over the whole product rather than
    /// over the one point pie reaches — a caller may resolve any of them.
    #[test]
    fn the_table_covers_everything_resolve_admits() {
        for &nranks in NRANKS {
            for &pattern in INSTANTIATED {
                for fp32_acc in [true, false] {
                    for (oneshot, trigger) in [(true, false), (true, true), (false, false)] {
                        let point = resolve(nranks, pattern, fp32_acc, oneshot, trigger)
                            .expect("`NRANKS` x `INSTANTIATED` is what `resolve` admits");
                        assert!(
                            point.name_expression().is_some(),
                            "no template-id for {point:?}"
                        );
                    }
                }
            }
        }
        for &nranks in PLAIN_NRANKS {
            for stage in [Stage::OneStage, Stage::TwoStage] {
                assert!(plain_name_expression(nranks, stage).is_some());
            }
        }
    }

    /// A template-id names the `__global__` and carries the point's four
    /// template arguments in upstream's order.
    ///
    /// Spot-checked rather than re-derived, because re-deriving it here would
    /// be a second `format!` of the thing the table exists to replace.
    #[test]
    fn a_template_id_names_the_global_and_its_arguments() {
        let one = REACHED.name_expression().expect("the reached point is in the table");
        assert!(one.contains("allreduce_fusion_kernel_oneshot_lamport"), "{one}");
        assert!(one.ends_with("__nv_bfloat16, 2, true, false>"), "{one}");
        assert!(!one.contains("kernel_launcher"), "the launcher is host C++ and is gone: {one}");

        let two = Instantiation { leaf: Leaf::TwoShot, ..REACHED }
            .name_expression()
            .expect("the two-shot leaf is in the table");
        assert!(two.contains("allreduce_fusion_kernel_twoshot_sync"), "{two}");
        assert!(two.ends_with("__nv_bfloat16, 2, true>"), "{two}");

        // Spelled by its parts rather than as one literal: a `::`-prefixed
        // string in a `fn` that names no carried file is a template-id
        // `every_instantiation_compiles` cannot attribute, and it fails on
        // one rather than skipping it.
        let six = plain_name_expression(6, Stage::TwoStage).expect("vllm instantiates six");
        assert!(six.contains("cross_device_reduce_2stage"), "{six}");
        assert!(six.ends_with("<__nv_bfloat16, 6>"), "{six}");
        // vllm has 6 and flashinfer does not; flashinfer has 16 and vllm does
        // not. Neither set contains the other, and both are upstream's.
        assert_eq!(plain_name_expression(16, Stage::OneStage), None);
        assert!(!NRANKS.contains(&6));
    }

    #[test]
    fn the_one_reached_point_resolves() {
        let got = resolve(REACHED.nranks, REACHED.pattern, REACHED.fp32_acc, true, false)
            .expect("the one point pie reaches must resolve");
        assert_eq!(got, REACHED);
    }

    #[test]
    fn an_uninstantiated_pattern_declines_with_its_code() {
        // The nine patterns pie never selects were the 96%.
        let err = resolve(2, FusionPattern::ARResidualRMSNormFp8Quant, true, true, false)
            .expect_err("an unpruned pattern must decline");
        assert_eq!(
            err,
            Decline::PatternNotInstantiated {
                code: FusionPattern::ARResidualRMSNormFp8Quant.code()
            }
        );
    }

    #[test]
    fn an_unsupported_world_size_declines_before_the_pattern_is_read() {
        // Upstream's switch is on `nranks` first (`trtllm_allreduce_fusion.
        // cuh`'s launcher dispatch), so a world size of 3 refuses even for a
        // pattern that is instantiated.
        let err = resolve(3, FusionPattern::ARResidualRMSNorm, true, true, false)
            .expect_err("world_size 3 must decline");
        assert_eq!(err, Decline::WorldSizeUnsupported { nranks: 3 });
    }

    /// The `#[repr(C)]` mirror is the vendored struct, field for field.
    ///
    /// Every number is the one `__INTADDR__` reported against
    /// `AllReduceFusionParams<__nv_bfloat16>` under NVRTC 13.0; the type's own
    /// doc quotes the same list. This is the only thing standing between a
    /// reordered field here and a kernel reading `rms_eps` out of a pointer.
    #[test]
    fn the_parameter_block_is_the_upstream_struct() {
        use core::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<FusionParams>(), 136);
        assert_eq!(align_of::<FusionParams>(), 8);
        assert_eq!(offset_of!(FusionParams, nranks), 0);
        assert_eq!(offset_of!(FusionParams, rank), 4);
        assert_eq!(offset_of!(FusionParams, size), 8);
        assert_eq!(offset_of!(FusionParams, hidden_dim), 12);
        assert_eq!(offset_of!(FusionParams, workspace), 16);
        assert_eq!(offset_of!(FusionParams, allreduce_in), 24);
        assert_eq!(offset_of!(FusionParams, allreduce_out), 32);
        assert_eq!(offset_of!(FusionParams, residual_in), 40);
        assert_eq!(offset_of!(FusionParams, residual_out), 48);
        assert_eq!(offset_of!(FusionParams, norm_out), 56);
        assert_eq!(offset_of!(FusionParams, quant_out), 64);
        assert_eq!(offset_of!(FusionParams, scale_out), 72);
        assert_eq!(offset_of!(FusionParams, rms_gamma), 80);
        assert_eq!(offset_of!(FusionParams, rms_eps), 88);
        assert_eq!(offset_of!(FusionParams, weight_bias), 92);
        assert_eq!(offset_of!(FusionParams, scale_factor), 96);
        assert_eq!(offset_of!(FusionParams, use_oneshot), 104);
        assert_eq!(offset_of!(FusionParams, layout), 108);
        assert_eq!(offset_of!(FusionParams, stream), 112);
        assert_eq!(offset_of!(FusionParams, pattern), 120);
        assert_eq!(offset_of!(FusionParams, trigger_completion_at_end), 124);
        assert_eq!(offset_of!(FusionParams, block_quant_group_size), 128);
        assert_eq!(offset_of!(FusionParams, tma_aligned_mn), 132);
    }

    /// The fused grid is upstream's at [`CLUSTER_SIZE`], and the ceiling that
    /// pin creates is where the doc says it is.
    #[test]
    fn the_fused_geometry_is_upstream_at_cluster_size_one() {
        assert_eq!(CLUSTER_SIZE, 1);
        // Llama-3-70B's hidden size, one token, a 132-SM device: one block of
        // `8192 / 8` threads, and the grid is the token count.
        let got = fusion_geometry(1, 8192, 2, Leaf::OneShot, 132).expect("inside the ceiling");
        assert_eq!(got.block, 1024);
        assert_eq!(got.grid, 1);
        // The grid is `min(sm_count, token_num)` and nothing else.
        assert_eq!(fusion_geometry(512, 4096, 2, Leaf::OneShot, 132).unwrap().grid, 132);
        assert_eq!(fusion_geometry(7, 4096, 2, Leaf::OneShot, 132).unwrap().grid, 7);
        // Above 8192 the block would exceed 1024, which is the whole cost of
        // not carrying a cluster dimension.
        assert_eq!(
            fusion_geometry(1, 16384, 2, Leaf::OneShot, 132),
            Err(Decline::FusionBlockWidth { hidden: 16384, threads: 2048, max: 1024 })
        );
        // Two-shot splits the tokens across the ranks first.
        assert_eq!(fusion_geometry(9, 4096, 2, Leaf::TwoShot, 132).unwrap().grid, 5);
        assert_eq!(fusion_geometry(8, 4096, 2, Leaf::TwoShot, 132).unwrap().grid, 4);
    }

    /// The two-shot split covers every token exactly once, with the remainder
    /// spread over the low ranks — `:1543-1544`.
    #[test]
    fn the_twoshot_split_partitions_the_tokens() {
        for (tokens, nranks) in [(9, 2), (8, 4), (17, 8), (3, 4), (0, 2)] {
            let (begin, count) = twoshot_split(tokens, nranks);
            let mut at = 0;
            for r in 0..nranks as usize {
                assert_eq!(begin[r], at, "rank {r} of {nranks} starts where {} ended", r - 1);
                at += count[r];
            }
            assert_eq!(at, tokens, "{nranks} ranks must cover {tokens} tokens");
        }
    }

    /// The plain reduction's stage choice and grid are `REDUCE_CASE`'s.
    #[test]
    fn the_plain_geometry_is_the_reduce_case_macro() {
        // World size 2 is one-stage at any size (`:478`).
        let (geometry, stage, size) = plain_geometry(4 * 1024 * 1024, 2, true).expect("legal");
        assert_eq!(stage, Stage::OneStage);
        assert_eq!(size, 4 * 1024 * 1024 / VEC_SIZE as usize as i32);
        // 36 blocks is `kMaxBlocks` and the grid is clamped to it.
        assert_eq!(geometry.grid, 36);
        assert_eq!(geometry.block, 512);

        // Wider than two: one-stage under the size threshold, two-stage over.
        // `bytes = count / 8 * 16 = count * 2`, so 512 KiB is 256 Ki elements.
        assert_eq!(plain_geometry(256 * 1024 - 8, 4, true).unwrap().1, Stage::OneStage);
        assert_eq!(plain_geometry(256 * 1024, 4, true).unwrap().1, Stage::TwoStage);
        assert_eq!(plain_geometry(128 * 1024 - 8, 8, true).unwrap().1, Stage::OneStage);
        assert_eq!(plain_geometry(128 * 1024, 8, true).unwrap().1, Stage::TwoStage);

        // A small message does not launch a full grid.
        assert_eq!(plain_geometry(8 * 512, 2, true).unwrap().0.grid, 1);

        // Upstream's silent no-launch, as a refusal.
        assert_eq!(
            plain_geometry(4096, 4, false),
            Err(Decline::NotFullyConnected { world_size: 4 })
        );
        // The vector width, and the count that is not a multiple of it.
        assert_eq!(
            plain_geometry(4095, 2, true),
            Err(Decline::Vector { count: 4095, width: VEC_SIZE })
        );
        assert_eq!(plain_geometry(0, 2, true), Err(Decline::Vector { count: 0, width: VEC_SIZE }));
        // vllm's set, not flashinfer's.
        assert_eq!(
            plain_geometry(4096, 16, true),
            Err(Decline::WorldSizeUnsupported { nranks: 16 })
        );
        assert!(plain_geometry(4096, 6, true).is_ok());
    }
}
