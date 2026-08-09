//! `comm/custom_all_reduce.cu`'s HOST PROGRAM, in Rust — 664 lines with
//! **zero** `__global__` and **zero** `<<<>>>`, which is what made it a host
//! program that happened to carry a `.cu` extension.
//!
//! The file was named for linkage, not for content. Its whole body is peer
//! access, an IPC handle exchange, two slabs of workspace with a geometry
//! computed on the host, two predicates, and a template dispatch. Exactly
//! **two lines** of it reached device text — `impl_->allreduce<__nv_bfloat16>`
//! at `custom_all_reduce.cu:614-620` and `allreduce_fusion_kernel_launcher`
//! at `:157-162` — and both live in headers this repository does not carry.
//!
//! # The four terms
//!
//! **What it fires.** Nothing, today, and that is a statement about the
//! headers rather than about this module. The two launch points name
//! `flashinfer/comm/vllm_custom_all_reduce.cuh` and
//! `flashinfer/comm/trtllm_allreduce_fusion.cuh`, which are CPM-*fetched*
//! at configure time and are **not** vendored:
//! `crates/kernels-cuda-new/csrc/vendor/flashinfer/` holds `attention/` and
//! has no `comm/` directory at all, so NVRTC has nothing to compile. Both
//! points therefore answer [`Decline::NoDeviceText`], and each answer carries
//! the resolved [`Instantiation`] and the exact header that would supply it —
//! the refusal IS the specification for the vendoring step.
//!
//! **What sits between.** Four device allocations whose sizes are host
//! arithmetic and appear in no row's operand list: the `Signal` + staging
//! slab ([`SIGNAL_BYTES`] + `max_bytes`), the `RankData` slab (8 MiB of
//! 64-byte slots), the three fusion buffers (`buffer`, `flag`, `lamport`,
//! each 2 MiB-aligned), and the `3 * world_size + 1` pointer workspace that
//! is copied to the device once at construction. Every one of those is
//! computed by [`CustomAllReduce::new`] from the constructor block at
//! `custom_all_reduce.cu:222-401`.
//!
//! **What it decides on the host.** Whether peer access exists in both
//! directions between every pair in the group; whether the group is fully
//! connected; whether a message is worth the NVLink kernel at all
//! ([`CustomAllReduce::can_handle`], a four-term test ending in a world-size
//! crossover table); whether the fused landing is available
//! ([`CustomAllReduce::can_fuse_residual_rmsnorm`]); and which of upstream's
//! 240 compile-time instantiations a call selects ([`resolve`]).
//!
//! **What vocabulary is missing.** Only the device text. There is no
//! `LaunchRule` here and no `Launch`: this module never builds a grid,
//! because the kernels it would fire are launched by upstream's own host
//! launcher templates, which take an `AllReduceFusionParams` and compute
//! their own geometry. [`FusionParams`] mirrors that struct field for field
//! so the thing a vendored path needs is written down rather than inferred.
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
//! Three of those four axes are template parameters of
//! `allreduce_fusion_kernel_launcher<Pattern, T, NRanks, Fp32Acc>` and the
//! fourth is not: `use_oneshot` and `trigger_completion_at_end` are RUNTIME
//! fields of `AllReduceFusionParams` that the launcher itself branches on to
//! pick among three `__global__`s. So the 240 is `80 host instantiations x 3
//! device leaves`, and [`Leaf`] is that third factor named. Getting this
//! decomposition wrong is how a pruning argument turns into a wrong count.
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
//! [`fire::gemv`] is the established shape and this module follows it:
//! [`AllReduce`] is `#[must_use]`, *"it declined"* cannot be spelled like
//! *"it ran"*, and a decline enqueues **nothing** — the caller's `output` is
//! exactly as it found it, and `dist::all_reduce_bf16` (NCCL) is the other
//! arm. A refusal is never a fallback: a null `car` is
//! [`Decline::NoInstance`], matching `custom_all_reduce.hpp:170-174` and
//! `:193-197`, which threw rather than quietly reducing nothing.
//!
//! [`fire::gemv`]: crate::fire::gemv
//!
//! # What is NOT here, and why the lifecycle is
//!
//! `.wiki/driver/new-horizon.md` §43.4 records that the reachability audit
//! reports the `CustomAllReduce` lifecycle dead **and is wrong**. It is kept
//! whole. The half of `vllm::CustomAllreduce` this module absorbs —
//! `open_ipc_handle`, `get_graph_buffer_ipc_meta`, `check_rank_data_capacity`,
//! `register_buffer(void**)` and `register_graph_buffers` — is pure host code
//! in upstream too; only `allreduce<T>()` launches. So the Rust owns the
//! entire lifecycle natively and declines at exactly the two points where
//! device text is required, which is a smaller surface than the C++ had.
//!
//! One thing got *better* in the crossing rather than merely equal.
//! `custom_all_reduce.cu:340-342` initialised the Lamport buffer by calling
//! `flashinfer::trtllm_allreduce::lamportInitialize<__nv_bfloat16>`, which
//! launches a kernel to write negative-zero into every slot. Negative zero
//! in bf16 is `0x8000`, a 16-bit pattern and not a byte pattern, so
//! `cudaMemset` cannot express it — but `cuMemsetD16_v2` can, exactly, and
//! it is a driver-API call with no device text behind it. See
//! [`CustomAllReduce::new`]. That is one launch point removed rather than
//! deferred.
//!
//! # Reachability today
//!
//! `serve/load.rs:120-147` refuses `tp_size > 1` outright — *"there is no
//! NCCL in this tree and no `CustomAllReduce` handle to pass"* — so nothing
//! in this module runs in any configuration this driver currently accepts.
//! It is written whole anyway, because the alternative to writing it is
//! leaving 664 lines of C++ in an archive that is being deleted, and because
//! the thing that is missing (a vendored `comm/`) is named precisely enough
//! here that closing it is a vendoring step and not a re-derivation.

// `initialise` prints ONE line to stderr at construction, which the archive
// did at `custom_all_reduce.cu:398-404` and which is the only trace a
// deployment gets that the P2P plane came up at all. `serve/mod.rs` and
// `fire/attn_score.rs` carry the same allow for the same reason.
#![allow(clippy::print_stderr)]

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::fmt;
use std::sync::Arc;

use cudarc::driver::sys::{CUdeviceptr, CUpointer_attribute, cuMemsetD16_v2, cuPointerGetAttribute};
use cudarc::runtime::sys::{
    cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess, cudaDeviceSynchronize, cudaError, cudaFree,
    cudaGetDevice, cudaGetLastError, cudaIpcCloseMemHandle, cudaIpcGetMemHandle, cudaIpcMemHandle_t,
    cudaIpcMemLazyEnablePeerAccess, cudaIpcOpenMemHandle, cudaMalloc, cudaMemcpy, cudaMemcpyKind,
    cudaMemset, cudaStreamCaptureStatus, cudaStreamIsCapturing, cudaStream_t,
};

use crate::error::{Error, check_cu, check_rt, ignore_in_drop};

// ── the constants the archive carried, each cited to its line ────────────

/// `sizeof(vllm::Signal)` — 3,456 bytes.
///
/// `flashinfer/comm/vllm_custom_all_reduce.cuh:52-60`:
///
/// ```text
///   struct Signal {
///     alignas(128) FlagType self_counter[kMaxBlocks][8];      // 36*8*4 = 1152
///     alignas(128) FlagType peer_counter[2][kMaxBlocks][8];   // 2*36*8*4 = 2304
///   };
/// ```
///
/// with `kMaxBlocks = 36` (`:46`) and `using FlagType = uint32_t` (`:51`).
/// Both members are already whole multiples of the 128-byte alignment, so
/// the struct needs no tail padding and the sum is exact. **This number is
/// an ABI fact of a header this tree does not vendor**, which is why it is
/// quoted here with the derivation rather than referenced.
pub const SIGNAL_BYTES: usize = 1152 + 2304;

/// `vllm::kMaxBlocks` — `vllm_custom_all_reduce.cuh:46`.
///
/// The same 36 that `custom_all_reduce.cu:613` passes as `block_limit`. The
/// two agreeing is not a coincidence to be preserved by hand: the launcher
/// clamps its grid to this, so a `block_limit` above it would silently do
/// nothing and one below it would leave bandwidth on the floor.
pub const MAX_BLOCKS: i32 = 36;

/// `sizeof(vllm::RankData)` — `vllm_custom_all_reduce.cuh:62-64`,
/// `struct __align__(16) RankData { void* ptrs[8]; }`.
pub const RANK_DATA_BYTES: usize = 8 * 8;

/// The 512 threads `custom_all_reduce.cu:614` pins on every plain P2P
/// all-reduce, and the 36 blocks beside it (`:613`).
///
/// Not a tuning knob and not derived from a shape: the vllm kernel's
/// one-shot and two-shot bodies both assume a fixed cooperative rectangle,
/// and the launcher takes the pair as arguments only so a caller can shrink
/// it. Nothing in pie ever did.
pub const ALL_REDUCE_THREADS: i32 = 512;

/// `custom_all_reduce.hpp:69` — the default `max_bytes`, 8 MiB.
pub const DEFAULT_MAX_BYTES: usize = 8 * 1024 * 1024;

/// `custom_all_reduce.hpp:70` — the default `rank_data_bytes`, 8 MiB.
///
/// `custom_all_reduce.cu:302` calls that *"enough for ~131k graph
/// addresses"*, which is `8 MiB / 64 B` exactly.
pub const DEFAULT_RANK_DATA_BYTES: usize = 8 * 1024 * 1024;

/// `custom_all_reduce.cu:313` — `constexpr std::size_t kAlign = 1ull << 21`.
///
/// Every fusion allocation is rounded up to 2 MiB, which is the large-page
/// granularity the Lamport protocol's address arithmetic assumes.
pub const FUSION_ALIGN: usize = 1 << 21;

/// `custom_all_reduce.cu:314` — `constexpr std::size_t kBarrierFlagCount = 256`.
pub const BARRIER_FLAG_COUNT: usize = 256;

/// `custom_all_reduce.cu:329-333` — the Lamport communication cap,
/// 2,145,386,496 bytes.
///
/// The archive wrote it as a bare literal. It is `2^31 - 2 MiB`: the largest
/// 2 MiB-aligned byte count that still fits a SIGNED 32-bit integer, which
/// is what the flag word at index 3 of the five-word flag block is
/// (`custom_all_reduce.cu:345-349` casts it to `std::uint32_t`, and the
/// device side reads it as an offset). A cap chosen for the width of the
/// field that carries it, and stated that way so the next reader does not
/// have to factor it.
pub const LAMPORT_COMM_CAP: usize = 2_145_386_496;

/// bf16 negative zero — the Lamport "empty slot" sentinel.
///
/// `custom_all_reduce.cu:339-342` got this by launching
/// `flashinfer::trtllm_allreduce::lamportInitialize<__nv_bfloat16>`. It is a
/// 16-bit fill, so `cuMemsetD16_v2` writes it with no kernel at all.
const LAMPORT_EMPTY_BF16: u16 = 0x8000;

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
/// # A drift worth recording
///
/// `kernels.def`'s block says `pattern {10}` and derives `4 x 10 x 2 x 3 =
/// 240` from it. Today's upstream header
/// (`trtllm_allreduce_fusion.cuh:720-733`) declares **eight** enumerators
/// spanning 0..=9. The 240 is preserved above exactly as measured — it was
/// measured against the header the archive compiled, and a measurement is
/// not amended by a later reading — but a re-measurement on today's upstream
/// would find `4 x 8 x 2 x 3 = 192`. Recorded rather than reconciled: the
/// only number this module ACTS on is the size of [`INSTANTIATED`], which is
/// 1 either way.
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

/// `kernels.def`'s `PIE_AR_FUSION_PATTERN` list, in Rust — **one entry.**
///
/// This is the whole of what the `#include "kernels.def"` inside
/// `dispatch_ar_fusion_pattern` (`custom_all_reduce.cu:163-169`) expanded to.
/// The C++ expanded it into `case` labels of a `switch` on
/// `params.pattern`; [`resolve`] is that switch, and the `default:` that
/// threw is [`Decline::PatternNotInstantiated`].
///
/// **Adding a pattern to a call site requires adding it here**, exactly as
/// `kernels.def` said — but the consequence has changed and is now better:
/// a missing entry used to surface as a runtime throw deep inside a fire,
/// and now surfaces as a `Decline` a caller must handle by type.
pub static INSTANTIATED: &[FusionPattern] = &[FusionPattern::ARResidualRMSNorm];

/// upstream flashinfer's supported TP world sizes —
/// `custom_all_reduce.cu:181-200`'s `switch (params.nranks)`.
///
/// **Deliberately unpruned, and the archive said why** (`:177-179`): *"nranks
/// is upstream flashinfer's supported set rather than a pie-owned axis, so it
/// stays fully instantiated: TP world size is a deployment choice and pruning
/// it would turn a valid launch into a runtime throw."* Under a JIT the
/// argument gets stronger, not weaker — an unreached world size now costs
/// nothing at all, not even a compile.
pub static NRANKS: &[i32] = &[2, 4, 8, 16];

/// Which of the three device leaves a set of runtime flags selects.
///
/// The axis that is NOT a template parameter. `AllReduceFusionParams` carries
/// `use_oneshot` and `trigger_completion_at_end` as fields; the launcher
/// branches on them and picks one of three `__global__`s. `kernels.def`
/// wrote it `(oneshot x trigger_completion_at_end {2}, or twoshot {1})`,
/// which is 3 and not 4 — the two-shot path ignores
/// `trigger_completion_at_end` entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leaf {
    /// `use_oneshot = true`, `trigger_completion_at_end = false`. **The one
    /// pie sets** — `custom_all_reduce.cu:653`, `:658`.
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
}

/// How many device leaves a single host instantiation carries — 3.
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
/// moves the figure the module's header quotes.
pub const AOT_POINTS_AFTER_PRUNING: usize =
    NRANKS.len() * INSTANTIATED.len() * FP32_ACC_VALUES * LEAVES;

/// One point of the cross product: everything a launch has to fix before
/// any device text exists for it.
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
    /// The runtime leaf. NOT a template parameter — see [`Leaf`].
    pub leaf: Leaf,
}

impl Instantiation {
    /// The name expression a vendored `comm/` would hand NVRTC.
    ///
    /// The **host launcher** specialisation, which is what the archive named
    /// (`custom_all_reduce.cu:157-162`) and what `T = __nv_bfloat16` is fixed
    /// in. The `__global__` NVRTC would actually compile is one of the three
    /// [`Leaf`] bodies this specialisation dispatches to, and its own
    /// template parameter list is NOT transcribed here because this tree
    /// cannot see it: `csrc/vendor/flashinfer/` has no `comm/`. Inventing it
    /// would be inventing an ABI.
    #[must_use]
    pub fn name_expression(&self) -> String {
        format!(
            "flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_launcher<\
             (flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern){}, \
             __nv_bfloat16, {}, {}>",
            self.pattern.code(),
            self.nranks,
            self.fp32_acc
        )
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
        return Err(Decline::PatternNotInstantiated {
            code: pattern.code(),
        });
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
///
/// Not `Copy`: [`Decline::NoDeviceText`] carries the resolved instantiation's
/// name expression, and that string is the specification for the vendoring
/// step. Losing it to keep the type two words wide would be losing the only
/// thing that makes the refusal actionable.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// state. Rust has no default-constructed `CustomAllReduce`, so this is
    /// reachable only through the raw-pointer ABI forms.
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
    /// `world_size > 2` and some ordered pair in the group has no peer
    /// access — `custom_all_reduce.cu:468`.
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
    /// `custom_all_reduce.cu:477`.
    ///
    /// Also covers the `cuPointerGetAttribute` failure the C++ caught and
    /// turned into `false` at `:471-476`: a pointer with no queryable base
    /// cannot have been registered.
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
    /// `hidden % 8 != 0` — `custom_all_reduce.cu:499`.
    FusionHiddenNotOctet {
        /// The hidden size that failed it.
        hidden: i32,
    },
    /// The pattern is not in [`INSTANTIATED`] —
    /// `custom_all_reduce.cu:171-176`, which threw.
    PatternNotInstantiated {
        /// The `AllReduceFusionPattern` discriminant asked for.
        code: i32,
    },
    /// The world size is not in [`NRANKS`] — `custom_all_reduce.cu:206-209`,
    /// which threw.
    WorldSizeUnsupported {
        /// The world size asked for.
        nranks: i32,
    },
    /// The instantiation resolved and **there is no device text for it in
    /// this tree.**
    ///
    /// The only arm that is not a port of a C++ branch, and the only one that
    /// is a statement about the repository rather than about the call. It
    /// carries the name expression so that vendoring the header is a
    /// mechanical step: the point is already resolved.
    NoDeviceText {
        /// What would have launched.
        what: &'static str,
        /// The header that defines it, as an `#include` path.
        header: &'static str,
        /// [`Instantiation::name_expression`] for the resolved point.
        name_expression: String,
    },
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
            Self::NotFullyConnected { world_size } => write!(
                f,
                "world size {world_size} needs peer access between every ordered pair and does not \
                 have it"
            ),
            Self::CaptureUnknown => write!(f, "`cudaStreamIsCapturing` failed on this stream"),
            Self::Unregistered => write!(
                f,
                "the input's base allocation was never passed to `register_buffer`"
            ),
            Self::AboveCrossover {
                bytes,
                crossover,
                world_size,
            } => write!(
                f,
                "{bytes} bytes is at or above the {crossover}-byte crossover for world size \
                 {world_size}; NCCL wins on bandwidth here"
            ),
            Self::NoFusionWorkspace => write!(
                f,
                "no fusion workspace was built (world size 2 with a positive `fusion_max_tokens` \
                 and `fusion_hidden` is what builds one)"
            ),
            Self::FusionTokens { tokens, max_tokens } => write!(
                f,
                "{tokens} tokens against a workspace sized for {max_tokens}"
            ),
            Self::FusionHidden { hidden, want } => write!(
                f,
                "hidden {hidden} against a workspace sized for exactly {want}"
            ),
            Self::FusionWorldSize { world_size } => write!(
                f,
                "the fused landing is world size 2 only; this group is {world_size}"
            ),
            Self::FusionHiddenNotOctet { hidden } => {
                write!(f, "hidden {hidden} is not a multiple of 8")
            }
            Self::PatternNotInstantiated { code } => write!(
                f,
                "`AllReduceFusionPattern` {code} is not in `fire::all_reduce::INSTANTIATED`; \
                 adding a pattern to a call site requires adding it there"
            ),
            Self::WorldSizeUnsupported { nranks } => write!(
                f,
                "the fused all-reduce does not support TP world size {nranks} (flashinfer supports \
                 2, 4, 8, 16)"
            ),
            Self::NoDeviceText {
                what,
                header,
                name_expression,
            } => write!(
                f,
                "{what} has no device text in this tree: `{header}` is CPM-fetched, not vendored, \
                 and `crates/kernels-cuda-new/csrc/vendor/flashinfer/` has no `comm/`. The point \
                 is resolved -- NVRTC would need `{name_expression}`"
            ),
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum AllReduce {
    /// The launch is on the stream.
    Launched,
    /// Nothing was enqueued. **Use `dist::all_reduce_bf16` for this one.**
    Declined(Decline),
}

// ── the parameter block a vendored path would fill ───────────────────────

/// `flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<T>`, mirrored.
///
/// Every field is one `custom_all_reduce.cu:637-659` set, in that order, and
/// nothing is added. It is here because it is **the specification of what a
/// vendored or NVRTC path needs**: the launcher takes this struct and derives
/// its own grid from it, so a future `comm/` port has to fill exactly these
/// and no reader should have to reconstruct the list from a deleted file.
///
/// **It is deliberately NOT `#[repr(C)]`**, and that is also a claim this
/// tree cannot check: the upstream struct's field ORDER is not transcribed
/// here (the header is not vendored), so this is a checklist, not an ABI. A
/// path that actually passes it across must re-derive the layout from the
/// vendored header, and a `#[repr(C)]` here would look like it already had.
#[derive(Debug, Clone, Copy)]
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
    pub workspace: *mut c_void,
    /// `params.allreduce_in` — `:643`.
    pub allreduce_in: *const c_void,
    /// `params.allreduce_out = nullptr` — `:644`. The unfused output is not
    /// wanted; only the normed one is.
    pub allreduce_out: *mut c_void,
    /// `params.residual_in` — `:645`.
    pub residual_in: *const c_void,
    /// `params.residual_out` — `:646`. **The same pointer as
    /// `residual_in`**, which is the `in_place = &[(0, 1)]` the row states.
    pub residual_out: *mut c_void,
    /// `params.norm_out` — `:647`.
    pub norm_out: *mut c_void,
    /// `params.quant_out = nullptr` — `:648`. Set only by the nine FP8/FP4
    /// patterns pie does not reach.
    pub quant_out: *mut c_void,
    /// `params.scale_out = nullptr` — `:649`.
    pub scale_out: *mut c_void,
    /// `params.rms_gamma` — `:650`.
    pub rms_gamma: *const c_void,
    /// `params.rms_eps` — `:651`.
    pub rms_eps: f32,
    /// `params.scale_factor = nullptr` — `:652`.
    pub scale_factor: *const c_void,
    /// `params.use_oneshot = true` — `:653`. One half of [`Leaf`].
    pub use_oneshot: bool,
    /// `params.layout = QuantizationSFLayout::SWIZZLED_128x4` — `:654`.
    ///
    /// Carried as the enumerator NAME rather than a number: it is read only
    /// by the quant epilogues, so pie's value is never load-bearing, and a
    /// discriminant transcribed from an unvendored header would be an
    /// invention with no consumer to catch it.
    pub layout: &'static str,
    /// `params.stream` — `:655`.
    pub stream: *mut c_void,
    /// `params.pattern` — `:656`.
    pub pattern: FusionPattern,
    /// `params.trigger_completion_at_end = false` — `:657`. The other half
    /// of [`Leaf`].
    pub trigger_completion_at_end: bool,
    /// `launch_with_pdl = false` — `:661`. Not a field of the struct
    /// upstream; the launcher's second argument. Kept beside the fields it
    /// travels with.
    pub launch_with_pdl: bool,
    /// `use_fp32_acc = true` — `:660`. The `Fp32Acc` template parameter,
    /// which is why it is not a struct field upstream either.
    pub use_fp32_acc: bool,
}

impl FusionParams {
    /// The [`Instantiation`] this parameter block selects.
    ///
    /// # Errors
    ///
    /// Whatever [`resolve`] refuses.
    pub fn instantiation(&self) -> std::result::Result<Instantiation, Decline> {
        resolve(
            self.nranks,
            self.pattern,
            self.use_fp32_acc,
            self.use_oneshot,
            self.trigger_completion_at_end,
        )
    }
}

// ── host helpers, one per anonymous-namespace function ───────────────────

/// `custom_all_reduce.cu:84-86` — `align_up(n, a)`.
fn align_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

/// `custom_all_reduce.cu:71-82` — `get_base_ptr`.
///
/// *"The vllm kernel needs the base pointer for the IPC handle exchange —
/// sub-allocation pointers won't round-trip across processes correctly."*
///
/// # Errors
///
/// The driver's code, when the pointer has no queryable range.
fn base_ptr(ptr: *const c_void) -> crate::error::Result<*mut c_void> {
    let mut base: *mut c_void = std::ptr::null_mut();
    check_cu(
        // SAFETY: `&mut base` is a live `void*` slot, which is the width
        // `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` writes.
        unsafe {
            cuPointerGetAttribute(
                std::ptr::addr_of_mut!(base).cast::<c_void>(),
                CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                ptr as usize as CUdeviceptr,
            )
        },
        "cuPointerGetAttribute(RANGE_START_ADDR)",
    )?;
    Ok(base)
}

/// `custom_all_reduce.cu:31-56` — `enable_peer_access`.
///
/// Idempotent: `cudaErrorPeerAccessAlreadyEnabled` is swallowed and the
/// sticky error reset, exactly as `:53` did with its `(void)cudaGetLastError()`.
///
/// **`peers` holds real device ORDINALS, never rank indices** — the archive's
/// warning at `:28-30`, kept because it is the kind of mistake that works on
/// every single-group box and corrupts the second group on a four-GPU one:
/// *"a TP group is not necessarily devices 0..world_size-1 (a second group on
/// a 4-GPU box runs on devices 2 and 3)"*.
///
/// # Errors
///
/// [`Error::Invalid`] naming the ordered pair, when peer access is
/// unavailable or cannot be enabled.
fn enable_peer_access(self_device: i32, peers: &[i32]) -> crate::error::Result<()> {
    for &peer in peers {
        if peer == self_device {
            continue;
        }
        let mut can_access: c_int = 0;
        // SAFETY: both ordinals are plain integers; the out-parameter is live.
        let can_err = unsafe { cudaDeviceCanAccessPeer(&mut can_access, self_device, peer) };
        if can_err != cudaError::cudaSuccess || can_access == 0 {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "peer access unavailable from {self_device} to {peer}{}",
                    if can_err == cudaError::cudaSuccess {
                        String::new()
                    } else {
                        format!(": {can_err:?}")
                    }
                ),
            ));
        }
        // SAFETY: as above.
        let err = unsafe { cudaDeviceEnablePeerAccess(peer, 0) };
        if err != cudaError::cudaSuccess && err != cudaError::cudaErrorPeerAccessAlreadyEnabled {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!("cudaDeviceEnablePeerAccess {self_device}->{peer} failed: {err:?}"),
            ));
        }
        // Reset the sticky error -- `custom_all_reduce.cu:53`.
        // SAFETY: no arguments, no aliasing.
        let _ = unsafe { cudaGetLastError() };
    }
    Ok(())
}

/// `custom_all_reduce.cu:58-69` — `has_full_peer_access`.
///
/// Both directions of every ordered pair, because peer access is not
/// symmetric on every topology.
fn has_full_peer_access(group_devices: &[i32]) -> bool {
    for &src in group_devices {
        for &dst in group_devices {
            if src == dst {
                continue;
            }
            let mut can_access: c_int = 0;
            // SAFETY: plain integers and a live out-parameter.
            if unsafe { cudaDeviceCanAccessPeer(&mut can_access, src, dst) }
                != cudaError::cudaSuccess
            {
                return false;
            }
            if can_access == 0 {
                return false;
            }
        }
    }
    true
}

/// `cudaIpcMemHandle_t` from its 64 opaque bytes.
fn to_handle(bytes: &[u8; 64]) -> cudaIpcMemHandle_t {
    let mut handle = cudaIpcMemHandle_t { reserved: [0; 64] };
    for (dst, src) in handle.reserved.iter_mut().zip(bytes.iter()) {
        *dst = *src as c_char;
    }
    handle
}

/// The 64 opaque bytes of a `cudaIpcMemHandle_t`.
///
/// Handles travel as bytes here, not as the struct, because they are the
/// payload of an all-gather written by the caller and the caller has no
/// business naming a CUDA type. It is also what makes the handle hashable:
/// upstream's `open_ipc_handle` memo is keyed on the raw bytes for exactly
/// this reason.
fn from_handle(handle: &cudaIpcMemHandle_t) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (dst, src) in bytes.iter_mut().zip(handle.reserved.iter()) {
        *dst = *src as u8;
    }
    bytes
}

// ── the seam ─────────────────────────────────────────────────────────────

/// One bootstrap-time all-gather over HOST buffers.
///
/// `send` is this rank's contribution; `recv` is `send.len() *
/// world_size` bytes, **rank-major**. `custom_all_reduce.hpp:55` states the
/// same contract in C++.
pub type Allgather = Arc<dyn Fn(&[u8], &mut [u8]) + Send + Sync>;

/// What this needs from the collective, and nothing more —
/// `custom_all_reduce.hpp:39-57`, kept whole because the reasoning is the
/// reason the type exists.
///
/// > The wrapper used to take an `NcclComm&`. It reads exactly two things off
/// > it — the world size, and one bootstrap-time all-gather of IPC handles —
/// > and taking the class instead of those two made a compute kernel depend
/// > on the driver's comm plane. It is a compute kernel:
/// > `all_reduce_residual_rmsnorm_bf16` fuses a reduction with a residual add
/// > and an RMSNorm, and the unfused halves of that live in `kernels-cuda`.
/// >
/// > So the seam is a callback. `gather` takes HOST buffers; whatever H2D
/// > dance a given collective needs is the caller's business, which is where
/// > NCCL knowledge belongs. Who decides custom-vs-NCCL by message size stays
/// > the caller's too — `can_handle()` only reports.
#[derive(Clone)]
pub struct HostAllgather {
    /// This rank's index in the group.
    pub rank: i32,
    /// The group size.
    pub world_size: i32,
    /// The collective.
    pub gather: Allgather,
}

impl fmt::Debug for HostAllgather {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostAllgather")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish_non_exhaustive()
    }
}

/// The constructor's remaining arguments — `custom_all_reduce.hpp:66-72`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Skip IPC entirely and exchange raw pointers, for a single-process
    /// multi-GPU deployment. `custom_all_reduce.cu:264-272`.
    pub same_process: bool,
    /// **The CUDA device ordinal of every rank, indexed by rank** —
    /// `custom_all_reduce.hpp:62-65`. Required, and required to be
    /// `world_size` long.
    pub group_devices: Vec<i32>,
    /// The largest message the plain P2P path will take.
    pub max_bytes: usize,
    /// The `RankData` slab. Floored at one slot.
    pub rank_data_bytes: usize,
    /// Zero disables the fused landing entirely.
    pub fusion_max_tokens: i32,
    /// Zero disables the fused landing entirely.
    pub fusion_hidden: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            same_process: false,
            group_devices: Vec::new(),
            max_bytes: DEFAULT_MAX_BYTES,
            rank_data_bytes: DEFAULT_RANK_DATA_BYTES,
            fusion_max_tokens: 0,
            fusion_hidden: 0,
        }
    }
}

/// The fusion plane, built only for `world_size == 2` with both fusion
/// dimensions positive — `custom_all_reduce.cu:308-397`.
#[derive(Debug)]
struct Fusion {
    /// `fusion_buffers_` — `[buffer, flag, lamport]`, `:336-338`.
    buffers: [*mut c_void; 3],
    /// `fusion_workspace_dev_` — the `3 * world + 1` pointer array, `:391`.
    workspace_dev: *mut c_void,
    /// `fusion_flag_dev_` — the five-word flag block, `:344`.
    flag_dev: *mut c_void,
    /// `fusion_max_tokens_`.
    max_tokens: i32,
    /// `fusion_hidden_`.
    hidden: i32,
    /// `fusion_lamport_comm_bytes_`, `:329-333`.
    lamport_comm_bytes: usize,
}

// ── the lifecycle ────────────────────────────────────────────────────────

/// The custom P2P all-reduce's whole host state —
/// `custom_all_reduce.hpp:59-141` and the half of `vllm::CustomAllreduce`
/// that never launched anything.
///
/// # Why the vllm half is absorbed rather than wrapped
///
/// The C++ held a `std::unique_ptr<vllm::CustomAllreduce> impl_` and reached
/// THROUGH it for state it also duplicated: `registered_bases_` beside
/// `impl_->buffers_`, and `register_graph_buffers` poking
/// `impl_->d_rank_data_base_` and `impl_->graph_unreg_buffers_` directly
/// (`custom_all_reduce.cu:557-575`). Every one of those members is plain
/// host bookkeeping — `open_ipc_handle`, `get_graph_buffer_ipc_meta`,
/// `check_rank_data_capacity`, `register_buffer(void**)` and
/// `register_graph_buffers` are host functions in upstream too, and only
/// `allreduce<T>()` launches. Absorbing them removes the duplication and
/// leaves exactly one thing on the other side of the seam: a launch.
///
/// # Two leaks the C++ had, closed by the crossing
///
/// `custom_all_reduce.cu:520-538` and `:355-375` opened peer handles with
/// bare `cudaIpcOpenMemHandle` calls and recorded them nowhere, so the
/// destructor (`:403-427`, which walks only `signal_peers_`) could not close
/// them. Every open here goes through [`CustomAllReduce::open_ipc_handle`],
/// which memoises by handle bytes — upstream's own shape — and [`Drop`]
/// closes the memo. A throwing C++ constructor also freed nothing;
/// [`CustomAllReduce::new`] builds the value first and lets [`Drop`] run on
/// the error path.
///
/// # Not `Send`, not `Sync`
///
/// The raw device pointers make it neither, which matches the C++: the class
/// is non-copyable and move-only, and every method assumes the calling thread
/// holds the device context the constructor ran on.
#[derive(Debug)]
pub struct CustomAllReduce {
    rank: i32,
    world_size: i32,
    fully_connected: bool,
    same_process: bool,
    max_bytes: usize,
    ag: HostAllgather,

    /// `signal_self_` — `sizeof(Signal) + max_bytes` of zeroed device
    /// memory, `custom_all_reduce.cu:259-261`.
    signal_self: *mut c_void,
    /// `signal_peers_`, one per rank, self included.
    signal_peers: Vec<*mut c_void>,

    /// `rank_data_` — the `RankData` slab.
    rank_data: *mut c_void,
    /// `d_rank_data_end_ - d_rank_data_base_` at construction, in slots.
    rank_data_slots: usize,
    /// upstream's `d_rank_data_base_`, as an index rather than a pointer.
    rank_data_next: usize,

    /// upstream's `buffers_` merged with the wrapper's `registered_bases_`:
    /// local base address -> the device `RankData*` it was registered into.
    buffers: HashMap<usize, *mut c_void>,
    /// upstream's `ipc_handles_`, keyed on the handle's 64 opaque bytes.
    ipc_handles: HashMap<[u8; 64], *mut c_void>,
    /// upstream's `graph_unreg_buffers_`.
    ///
    /// Appended to by the vllm host launcher when an all-reduce runs on a
    /// CAPTURING stream against a buffer that was never registered; drained
    /// by [`CustomAllReduce::register_graph_buffers`]. **Nothing appends to
    /// it in this tree**, because the launcher is the one thing that has not
    /// crossed, and a decline must not append: a deferred registration for a
    /// launch that never happened would bind the next real one to the wrong
    /// slot.
    graph_unreg_buffers: Vec<*mut c_void>,

    fusion: Option<Fusion>,
}

impl CustomAllReduce {
    /// `custom_all_reduce.cu:222-401`, the constructor.
    ///
    /// # Errors
    ///
    /// Every `throw` of the C++ constructor, as an [`Error`]: an unsupported
    /// world size, a missing or mis-sized `group_devices`, unavailable peer
    /// access, and any failing CUDA call.
    pub fn new(ag: HostAllgather, cfg: &Config) -> crate::error::Result<Self> {
        let mut me = Self {
            rank: ag.rank,
            world_size: ag.world_size,
            fully_connected: false,
            same_process: cfg.same_process,
            max_bytes: cfg.max_bytes,
            ag,
            signal_self: std::ptr::null_mut(),
            signal_peers: Vec::new(),
            rank_data: std::ptr::null_mut(),
            rank_data_slots: 0,
            rank_data_next: 0,
            buffers: HashMap::new(),
            ipc_handles: HashMap::new(),
            graph_unreg_buffers: Vec::new(),
            fusion: None,
        };
        // Built first so that a failure part-way returns through `Drop` and
        // frees whatever was allocated. The C++ constructor threw and freed
        // nothing.
        me.initialise(cfg)?;
        Ok(me)
    }

    fn initialise(&mut self, cfg: &Config) -> crate::error::Result<()> {
        // `custom_all_reduce.cu:231-235`.
        if self.world_size < 2 || self.world_size > 8 || (self.world_size % 2) != 0 {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "the vllm kernel supports world_size in {{2,4,6,8}}; got {}",
                    self.world_size
                ),
            ));
        }

        // `:239-251`.
        let mut dev: c_int = 0;
        // SAFETY: a live out-parameter.
        check_rt(unsafe { cudaGetDevice(&mut dev) }, "cudaGetDevice")?;
        if cfg.group_devices.is_empty() {
            return Err(Error::invalid(
                "custom_all_reduce",
                "group device ordinals are required",
            ));
        }
        if cfg.group_devices.len() != self.world_size as usize {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "group device list has {} entries for world_size {}",
                    cfg.group_devices.len(),
                    self.world_size
                ),
            ));
        }
        enable_peer_access(dev, &cfg.group_devices)?;

        // `:252-254`: the vllm kernel handles larger TP groups only when
        // every rank has direct peer access to every other rank.
        self.fully_connected = self.world_size <= 2 || has_full_peer_access(&cfg.group_devices);

        // `:256-261`. The staging region past the `Signal` is what
        // flashinfer's 2-stage algorithm needs; TP=2 takes the 1-stage path
        // and never touches it, and the layout is matched anyway so this
        // wrapper stays valid for fully-connected larger groups.
        let signal_bytes = SIGNAL_BYTES + self.max_bytes;
        // SAFETY: a live out-parameter and a positive size.
        check_rt(
            unsafe { cudaMalloc(&mut self.signal_self, signal_bytes) },
            "cudaMalloc(signal)",
        )?;
        // SAFETY: `signal_self` now addresses `signal_bytes` writable bytes.
        check_rt(
            unsafe { cudaMemset(self.signal_self, 0, signal_bytes) },
            "cudaMemset(signal)",
        )?;

        // `:263-297`, the signal exchange.
        self.signal_peers = self.exchange_pointers(self.signal_self)?;

        // `:299-304`. vLLM uses 8 MiB, "enough for ~131k graph addresses".
        let rank_data_bytes = cfg.rank_data_bytes.max(RANK_DATA_BYTES);
        // SAFETY: a live out-parameter and a positive size.
        check_rt(
            unsafe { cudaMalloc(&mut self.rank_data, rank_data_bytes) },
            "cudaMalloc(rank_data)",
        )?;
        self.rank_data_slots = rank_data_bytes / RANK_DATA_BYTES;
        self.rank_data_next = 0;

        if self.world_size == 2 && cfg.fusion_max_tokens > 0 && cfg.fusion_hidden > 0 {
            self.build_fusion(cfg)?;
        }

        // `:398-404`, kept verbatim. One line on stderr at construction, and
        // the four facts a support ticket needs.
        eprintln!(
            "[custom_all_reduce] initialised (world={}, rank={}, mode={}, fully_connected={})",
            self.world_size,
            self.rank,
            if self.same_process {
                "same-process"
            } else {
                "ipc"
            },
            if self.fully_connected { "yes" } else { "no" }
        );
        Ok(())
    }

    /// One all-gather of `local`'s address across the group, by whichever of
    /// the two mechanisms this deployment configured.
    ///
    /// `custom_all_reduce.cu:263-297` for the signal and `:355-375` for the
    /// fusion buffers were the same twelve lines twice; this is them once.
    /// In `same_process` mode the raw `u64` crosses; otherwise a
    /// `cudaIpcMemHandle_t` does and each peer's is opened.
    fn exchange_pointers(&mut self, local: *mut c_void) -> crate::error::Result<Vec<*mut c_void>> {
        let world = self.world_size as usize;
        if self.same_process {
            let send = (local as usize as u64).to_ne_bytes();
            let gathered = self.allgather(&send);
            let mut out = Vec::with_capacity(world);
            for r in 0..world {
                let mut word = [0u8; 8];
                word.copy_from_slice(&gathered[r * 8..r * 8 + 8]);
                out.push(u64::from_ne_bytes(word) as usize as *mut c_void);
            }
            return Ok(out);
        }

        let mut self_handle = cudaIpcMemHandle_t { reserved: [0; 64] };
        // SAFETY: `local` is a base allocation of this process.
        check_rt(
            unsafe { cudaIpcGetMemHandle(&mut self_handle, local) },
            "cudaIpcGetMemHandle",
        )?;
        let gathered = self.allgather(&from_handle(&self_handle));

        let mut out = Vec::with_capacity(world);
        for r in 0..world {
            if r == self.rank as usize {
                out.push(local);
                continue;
            }
            let mut key = [0u8; 64];
            key.copy_from_slice(&gathered[r * 64..r * 64 + 64]);
            out.push(self.open_ipc_handle(key)?);
        }
        Ok(out)
    }

    /// The collective, with the rank-major receive buffer sized here so no
    /// caller can get it wrong.
    fn allgather(&self, send: &[u8]) -> Vec<u8> {
        let mut recv = vec![0u8; send.len() * self.world_size as usize];
        (self.ag.gather)(send, &mut recv);
        recv
    }

    /// upstream `vllm::CustomAllreduce::open_ipc_handle`, memoised by the
    /// handle's bytes.
    ///
    /// The memo is not an optimisation: `cudaIpcOpenMemHandle` on a handle
    /// already open in this process returns the SAME mapping and increments
    /// nothing a second `cudaIpcCloseMemHandle` would balance, so opening
    /// twice and closing twice is a double free. Keying on the bytes is how
    /// upstream avoids it and how [`Drop`] here knows the exact set to close.
    fn open_ipc_handle(&mut self, key: [u8; 64]) -> crate::error::Result<*mut c_void> {
        if let Some(existing) = self.ipc_handles.get(&key) {
            return Ok(*existing);
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `key` is 64 bytes produced by `cudaIpcGetMemHandle` on a
        // peer, and the out-parameter is live.
        check_rt(
            unsafe {
                cudaIpcOpenMemHandle(
                    &mut ptr,
                    to_handle(&key),
                    cudaIpcMemLazyEnablePeerAccess as c_uint,
                )
            },
            "cudaIpcOpenMemHandle",
        )?;
        self.ipc_handles.insert(key, ptr);
        Ok(ptr)
    }

    /// `custom_all_reduce.cu:308-397` — the fusion plane's four allocations,
    /// its Lamport initialisation and its device workspace.
    fn build_fusion(&mut self, cfg: &Config) -> crate::error::Result<()> {
        let world = self.world_size as usize;
        let max_tokens = cfg.fusion_max_tokens;
        let hidden = cfg.fusion_hidden;

        // `:315-326`. `elem_bytes` is `sizeof(__nv_bfloat16)`.
        const ELEM_BYTES: usize = 2;
        let span = world * max_tokens as usize * hidden as usize * ELEM_BYTES;
        let buffer_bytes = align_up(span, FUSION_ALIGN);
        let flag_bytes = align_up(world * BARRIER_FLAG_COUNT * 4, FUSION_ALIGN);
        let lamport_comm_bytes = span.min(LAMPORT_COMM_CAP);
        let lamport_bytes = align_up(lamport_comm_bytes * 3, FUSION_ALIGN);

        let mut buffers = [std::ptr::null_mut::<c_void>(); 3];
        for (slot, bytes) in buffers
            .iter_mut()
            .zip([buffer_bytes, flag_bytes, lamport_bytes])
        {
            // SAFETY: a live out-parameter and a positive size.
            check_rt(unsafe { cudaMalloc(slot, bytes) }, "cudaMalloc(fusion)")?;
        }

        let mut flag_dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: a live out-parameter.
        check_rt(
            unsafe { cudaMalloc(&mut flag_dev, 5 * 4) },
            "cudaMalloc(fusion flags)",
        )?;

        // Everything allocated: park it on `self` so `Drop` owns it from
        // here, before anything else can fail.
        self.fusion = Some(Fusion {
            buffers,
            workspace_dev: std::ptr::null_mut(),
            flag_dev,
            max_tokens,
            hidden,
            lamport_comm_bytes,
        });

        // `:339-342`, and the one place this port does LESS device work than
        // the archive. `lamportInitialize<__nv_bfloat16>` launched a kernel
        // to write bf16 negative zero into every slot; negative zero is the
        // 16-bit pattern `0x8000`, so `cuMemsetD16_v2` writes it directly and
        // there is no device text to compile, fetch or vendor. The archive
        // passed a null stream and synchronised at `:396`; this is the
        // synchronous form, which is the same thing without the pair.
        // SAFETY: `buffers[2]` addresses `lamport_bytes` writable bytes, and
        // `lamport_bytes` is 2 MiB-aligned so the 16-bit count is exact.
        check_cu(
            unsafe {
                cuMemsetD16_v2(
                    buffers[2] as usize as CUdeviceptr,
                    LAMPORT_EMPTY_BF16,
                    lamport_bytes / ELEM_BYTES,
                )
            },
            "cuMemsetD16_v2(lamport)",
        )?;

        // `:345-351`. Index 3 carries the Lamport communication size; the
        // other four words start at zero.
        let flags: [u32; 5] = [0, 0, 0, lamport_comm_bytes as u32, 0];
        // SAFETY: `flag_dev` addresses 20 writable bytes and `flags` is 20
        // bytes of initialised host memory.
        check_rt(
            unsafe {
                cudaMemcpy(
                    flag_dev,
                    flags.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(&flags),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(fusion flags)",
        )?;

        // `:353-390`. `3 * world + 1` pointers: every rank's view of each of
        // the three buffers, rank-major within buffer, then the local flag
        // block.
        let mut workspace: Vec<*mut c_void> = Vec::with_capacity(3 * world + 1);
        for i in 0..3 {
            let peers = self.exchange_pointers(buffers[i])?;
            workspace.extend_from_slice(&peers);
        }
        workspace.push(flag_dev);

        let mut workspace_dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: a live out-parameter and a positive size.
        check_rt(
            unsafe {
                cudaMalloc(
                    &mut workspace_dev,
                    workspace.len() * std::mem::size_of::<*mut c_void>(),
                )
            },
            "cudaMalloc(fusion workspace)",
        )?;
        if let Some(fusion) = self.fusion.as_mut() {
            fusion.workspace_dev = workspace_dev;
        }
        // SAFETY: both sides address `workspace.len()` pointers.
        check_rt(
            unsafe {
                cudaMemcpy(
                    workspace_dev,
                    workspace.as_ptr().cast::<c_void>(),
                    workspace.len() * std::mem::size_of::<*mut c_void>(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(fusion workspace)",
        )?;
        // `:396`. The archive synchronised here because `lamportInitialize`
        // was asynchronous on the null stream; kept because the workspace
        // copy is too and every peer must see it before the first fire.
        // SAFETY: no arguments.
        check_rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")?;
        Ok(())
    }

    /// This rank's index.
    #[must_use]
    pub const fn rank(&self) -> i32 {
        self.rank
    }

    /// The group size.
    #[must_use]
    pub const fn world_size(&self) -> i32 {
        self.world_size
    }

    /// Whether every ordered pair of the group has direct peer access —
    /// `custom_all_reduce.cu:252-254`.
    #[must_use]
    pub const fn fully_connected(&self) -> bool {
        self.fully_connected
    }

    /// The largest message the plain P2P path will take.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Whether the fusion plane was built — `custom_all_reduce.cu:308`.
    #[must_use]
    pub const fn has_fusion(&self) -> bool {
        self.fusion.is_some()
    }

    /// `custom_all_reduce.cu:498-541` — `register_buffer`.
    ///
    /// `buf_bytes` was `/*buf_bytes*/` in the C++ too (`:499`): the vllm
    /// kernel registers a BASE address and does its own offset arithmetic
    /// against the registered `RankData`, so the extent never mattered. It
    /// stays in the signature because the caller has it and the day a bounds
    /// check becomes possible is the day it is wanted.
    ///
    /// Idempotent per base address, which is what makes it safe to call on
    /// every step.
    ///
    /// # Errors
    ///
    /// An unresolvable base pointer, a failing IPC exchange, or an exhausted
    /// `RankData` slab.
    pub fn register_buffer(&mut self, buf: *mut c_void, _buf_bytes: usize) -> crate::error::Result<()> {
        let self_base = base_ptr(buf)?;
        if self.buffers.contains_key(&(self_base as usize)) {
            return Ok(());
        }
        let peer_bases = self.exchange_pointers(self_base)?;
        let slot = self.write_rank_data(&[peer_bases])?;
        self.buffers.insert(self_base as usize, slot);
        Ok(())
    }

    /// upstream `vllm::CustomAllreduce::check_rank_data_capacity` +
    /// the `cudaMemcpy` into `d_rank_data_base_`, for a run of `rows`
    /// consecutive slots.
    ///
    /// One function because both callers wanted exactly this and the C++ had
    /// it twice — once inside `impl_->register_buffer` and once open-coded at
    /// `custom_all_reduce.cu:565-575`, where the wrapper reached past its
    /// own abstraction to advance `impl_->d_rank_data_base_` by hand.
    ///
    /// Returns the device address of the FIRST slot written.
    fn write_rank_data(&mut self, rows: &[Vec<*mut c_void>]) -> crate::error::Result<*mut c_void> {
        let n = rows.len();
        if self.rank_data_next + n > self.rank_data_slots {
            // upstream threw "Rank data buffer is overflowed by X"; here it
            // is the shared exhaustion error, which names the same two
            // numbers by naming the want.
            return Err(Error::exhausted(
                "custom_all_reduce rank_data slots",
                self.rank_data_next + n,
            ));
        }
        // `RankData` is `void* ptrs[8]` (`vllm_custom_all_reduce.cuh:62-64`)
        // regardless of world size; the tail past `world_size` is padding the
        // kernel never reads, and is zeroed here rather than left undefined.
        let mut flat: Vec<*mut c_void> = vec![std::ptr::null_mut(); n * 8];
        for (i, row) in rows.iter().enumerate() {
            for (r, ptr) in row.iter().enumerate() {
                flat[i * 8 + r] = *ptr;
            }
        }
        let first = (self.rank_data as usize + self.rank_data_next * RANK_DATA_BYTES) as *mut c_void;
        // SAFETY: the capacity check above proves `n` slots fit, and `flat`
        // is exactly `n * RANK_DATA_BYTES` bytes of initialised host memory.
        check_rt(
            unsafe {
                cudaMemcpy(
                    first,
                    flat.as_ptr().cast::<c_void>(),
                    n * RANK_DATA_BYTES,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(rank_data)",
        )?;
        self.rank_data_next += n;
        Ok(first)
    }

    /// upstream `vllm::CustomAllreduce::get_graph_buffer_ipc_meta`.
    ///
    /// Returns the concatenated 64-byte handles of every unregistered graph
    /// buffer's BASE allocation, and each buffer's byte offset within it.
    fn graph_buffer_ipc_meta(&self) -> crate::error::Result<(Vec<u8>, Vec<i64>)> {
        let n = self.graph_unreg_buffers.len();
        let mut handles = vec![0u8; n * 64];
        let mut offsets = vec![0i64; n];
        for (i, &ptr) in self.graph_unreg_buffers.iter().enumerate() {
            let base = base_ptr(ptr)?;
            offsets[i] = (ptr as usize as i64) - (base as usize as i64);
            let mut handle = cudaIpcMemHandle_t { reserved: [0; 64] };
            // SAFETY: `base` is a base allocation of this process.
            check_rt(
                unsafe { cudaIpcGetMemHandle(&mut handle, base) },
                "cudaIpcGetMemHandle(graph)",
            )?;
            handles[i * 64..(i + 1) * 64].copy_from_slice(&from_handle(&handle));
        }
        Ok((handles, offsets))
    }

    /// `custom_all_reduce.cu:543-601` — `register_graph_buffers`.
    ///
    /// Registers, in one collective, every buffer an all-reduce met on a
    /// CAPTURING stream and found unregistered. It is called once after a
    /// capture closes.
    ///
    /// **It is a no-op in this tree today**, and will be until the launcher
    /// crosses: [`CustomAllReduce::graph_unreg_buffers`] is fed by the vllm
    /// host launcher and by nothing else, and a decline must not feed it —
    /// see the field's own note. It is ported whole anyway because it is the
    /// half of the graph path that has no device text in it, and porting it
    /// later would mean reconstructing this collective from a deleted file.
    ///
    /// # Errors
    ///
    /// A failing IPC exchange or an exhausted `RankData` slab.
    pub fn register_graph_buffers(&mut self) -> crate::error::Result<()> {
        let n = self.graph_unreg_buffers.len();
        if n == 0 {
            return Ok(());
        }
        let world = self.world_size as usize;

        if self.same_process {
            // `:552-577`. Gather every rank's raw pointers, buffer-minor.
            let mut send = Vec::with_capacity(n * 8);
            for &ptr in &self.graph_unreg_buffers {
                send.extend_from_slice(&(ptr as usize as u64).to_ne_bytes());
            }
            let gathered = self.allgather(&send);
            let mut rows: Vec<Vec<*mut c_void>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(world);
                for r in 0..world {
                    // `:568-570`: rank-major outer, buffer-minor inner.
                    let idx = (r * n + i) * 8;
                    let mut word = [0u8; 8];
                    word.copy_from_slice(&gathered[idx..idx + 8]);
                    row.push(u64::from_ne_bytes(word) as usize as *mut c_void);
                }
                rows.push(row);
            }
            self.write_rank_data(&rows)?;
            self.graph_unreg_buffers.clear();
            return Ok(());
        }

        // `:579-600`, then upstream's `register_graph_buffers`.
        let (self_handles, self_offsets) = self.graph_buffer_ipc_meta()?;
        let all_handles = self.allgather(&self_handles);
        let mut offset_bytes = Vec::with_capacity(n * 8);
        for off in &self_offsets {
            offset_bytes.extend_from_slice(&off.to_ne_bytes());
        }
        let all_offsets = self.allgather(&offset_bytes);

        let handle_bytes = n * 64;
        let mut rows: Vec<Vec<*mut c_void>> = vec![Vec::with_capacity(world); n];
        for r in 0..world {
            for i in 0..n {
                if r == self.rank as usize {
                    rows[i].push(self.graph_unreg_buffers[i]);
                    continue;
                }
                let at = r * handle_bytes + i * 64;
                let mut key = [0u8; 64];
                key.copy_from_slice(&all_handles[at..at + 64]);
                let peer = self.open_ipc_handle(key)?;
                let mut word = [0u8; 8];
                let off_at = (r * n + i) * 8;
                word.copy_from_slice(&all_offsets[off_at..off_at + 8]);
                let offset = i64::from_ne_bytes(word);
                rows[i].push((peer as usize).wrapping_add(offset as usize) as *mut c_void);
            }
        }
        self.write_rank_data(&rows)?;
        self.graph_unreg_buffers.clear();
        Ok(())
    }

    /// `custom_all_reduce.cu:464-486` — `can_handle`, which returned `bool`.
    ///
    /// The `bool` is a [`Decline`] here, because every one of the eight
    /// `return false`s meant something different and the caller could not
    /// tell them apart. It is still a QUERY, not a refusal: the header's
    /// `:88-92` says *"above the threshold the kernel falls off NCCL on
    /// bandwidth, so we short-circuit and return false — caller should fall
    /// back to ncclAllReduce"*. A `Decline` from here is the caller's cue to
    /// use the collective, not an error.
    ///
    /// # Errors
    ///
    /// The [`Decline`] that the corresponding `return false` stood for.
    pub fn can_handle(
        &self,
        input: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> std::result::Result<(), Decline> {
        // `:467`.
        if input.is_null() {
            return Err(Decline::NullInput);
        }
        // `:469-471`. The 16-byte multiple is the kernel's vector width.
        if bytes == 0 || bytes > self.max_bytes || bytes % 16 != 0 {
            return Err(Decline::Bytes {
                bytes,
                max_bytes: self.max_bytes,
            });
        }
        // `:473`.
        if self.world_size > 2 && !self.fully_connected {
            return Err(Decline::NotFullyConnected {
                world_size: self.world_size,
            });
        }
        // `:475-479`. During capture the pointer query is meaningless --
        // the address will be replayed, not dereferenced now -- so the
        // registration check is deferred to `register_graph_buffers` and
        // capture answers YES immediately.
        let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
        // SAFETY: `stream` is a `cudaStream_t` from the caller (null is the
        // legal default stream) and the out-parameter is live.
        if unsafe { cudaStreamIsCapturing(stream as cudaStream_t, &mut status) }
            != cudaError::cudaSuccess
        {
            return Err(Decline::CaptureUnknown);
        }
        if status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive {
            return Ok(());
        }
        // `:481-483`. A throwing `get_base_ptr` was caught and turned into
        // `false`; here the error is simply not distinguished from "not
        // registered", which is what the C++ meant by catching it.
        let Ok(base) = base_ptr(input) else {
            return Err(Decline::Unregistered);
        };
        if !self.buffers.contains_key(&(base as usize)) {
            return Err(Decline::Unregistered);
        }
        // `:485`. The measured crossover with NCCL, which is the only reason
        // this class is optional at all. Below it the P2P kernel wins on
        // latency; above it NCCL wins on bandwidth, and the wider the group
        // the sooner that happens.
        let crossover = if self.world_size <= 2 {
            self.max_bytes
        } else if self.world_size <= 4 {
            1 << 20
        } else {
            256 << 10
        };
        if bytes < crossover {
            Ok(())
        } else {
            Err(Decline::AboveCrossover {
                bytes,
                crossover,
                world_size: self.world_size,
            })
        }
    }

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
            return Err(Decline::FusionTokens {
                tokens,
                max_tokens: fusion.max_tokens,
            });
        }
        // `:492`.
        if hidden != fusion.hidden {
            return Err(Decline::FusionHidden {
                hidden,
                want: fusion.hidden,
            });
        }
        // `:493`. The fusion plane is built for TP=2 only (`:308`), so this
        // is unreachable from a constructed `Fusion` -- kept because the C++
        // kept it and because it is the invariant, not a consequence.
        if self.world_size != 2 {
            return Err(Decline::FusionWorldSize {
                world_size: self.world_size,
            });
        }
        // `:494`. The kernel's vector width in bf16 elements.
        if hidden % 8 != 0 {
            return Err(Decline::FusionHiddenNotOctet { hidden });
        }
        Ok(())
    }

    /// `custom_all_reduce.cu:603-621` — the plain bf16 in-place all-reduce.
    ///
    /// `count` is an ELEMENT count, not bytes (`custom_all_reduce.hpp:100`).
    ///
    /// # The one line that has not crossed
    ///
    /// ```text
    /// impl_->allreduce<__nv_bfloat16>(stream, in, out, (int)count, 36, 512);
    /// ```
    ///
    /// `block_limit = 36` (`:612`) and `threads = 512` (`:613`) are the
    /// tuning constants, carried as [`MAX_BLOCKS`] and [`ALL_REDUCE_THREADS`]
    /// so they survive the file. `36` is not a choice here: it is
    /// `vllm_custom_all_reduce.cuh:46`'s `kMaxBlocks`, and it is the first
    /// dimension of the `Signal` counters, so a larger grid would index off
    /// the end of a 3456-byte struct.
    ///
    /// `(int)count` at `:618` is a silent narrowing. It cannot bite today —
    /// `can_handle` refuses above `max_bytes`, which defaults to 8 MiB —
    /// but `all_reduce_bf16` never called `can_handle`, so nothing enforced
    /// it. **When the launcher crosses, that narrowing becomes a refusal.**
    ///
    /// Nothing is appended to `graph_unreg_buffers` on the capture path, and
    /// that is deliberate: see the field's note.
    #[must_use]
    pub fn all_reduce_bf16(
        &mut self,
        _input: *const c_void,
        _output: *mut c_void,
        count: usize,
        _stream: *mut c_void,
    ) -> AllReduce {
        AllReduce::Declined(Decline::NoDeviceText {
            what: "vllm::CustomAllreduce::allreduce<__nv_bfloat16>",
            header: "flashinfer/comm/vllm_custom_all_reduce.cuh",
            name_expression: format!(
                "vllm::CustomAllreduce::allreduce<__nv_bfloat16>(stream, in, out, \
                 {count}, {MAX_BLOCKS}, {ALL_REDUCE_THREADS})"
            ),
        })
    }

    /// `custom_all_reduce.cu:623-662` — the fused all-reduce + residual add
    /// + RMSNorm.
    ///
    /// The C++ threw when `can_fuse_residual_rmsnorm` said no (`:633-635`);
    /// that throw is the [`Decline`] the query returned, unchanged, which is
    /// the whole point of the query returning one.
    ///
    /// # What has not crossed
    ///
    /// The launcher call at `:658-659`. Everything up to it — every field of
    /// [`FusionParams`], and the four runtime values that select the
    /// instantiation — is computed here, so the refusal names the exact
    /// template point rather than the family.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_residual_rmsnorm_bf16(
        &mut self,
        input: *const c_void,
        residual_inout: *mut c_void,
        rms_gamma: *const c_void,
        norm_out: *mut c_void,
        tokens: i32,
        hidden: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> AllReduce {
        if let Err(decline) = self.can_fuse_residual_rmsnorm(tokens, hidden) {
            return AllReduce::Declined(decline);
        }
        let Some(fusion) = self.fusion.as_ref() else {
            return AllReduce::Declined(Decline::NoFusionWorkspace);
        };
        // `:637-661`, field for field and in the file's order.
        let params = FusionParams {
            nranks: self.world_size,
            rank: self.rank,
            size: tokens * hidden,
            hidden_dim: hidden,
            workspace: fusion.workspace_dev,
            allreduce_in: input,
            allreduce_out: std::ptr::null_mut(),
            residual_in: residual_inout.cast_const(),
            residual_out: residual_inout,
            norm_out,
            quant_out: std::ptr::null_mut(),
            scale_out: std::ptr::null_mut(),
            rms_gamma,
            rms_eps: eps,
            scale_factor: std::ptr::null(),
            use_oneshot: true,
            layout: "SWIZZLED_128x4",
            stream,
            pattern: FusionPattern::ARResidualRMSNorm,
            trigger_completion_at_end: false,
            launch_with_pdl: false,
            use_fp32_acc: true,
        };
        match params.instantiation() {
            Err(decline) => AllReduce::Declined(decline),
            Ok(point) => AllReduce::Declined(Decline::NoDeviceText {
                what: "flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_launcher",
                header: "flashinfer/comm/trtllm_allreduce_fusion.cuh",
                name_expression: point.name_expression(),
            }),
        }
    }
}

/// `custom_all_reduce.cu:403-427` — the destructor.
///
/// Order matters and is upstream's: peer mappings close before the memory
/// they were opened against is freed.
impl Drop for CustomAllReduce {
    fn drop(&mut self) {
        // Every peer mapping this object ever opened, by construction --
        // signal peers, registered buffers, fusion peers and graph buffers
        // all went through `open_ipc_handle`. The C++ walked `signal_peers_`
        // only (`:410-419`) and leaked the rest.
        for (_, ptr) in self.ipc_handles.drain() {
            if !ptr.is_null() {
                // SAFETY: opened by `cudaIpcOpenMemHandle`, closed once.
                ignore_in_drop(unsafe { cudaIpcCloseMemHandle(ptr) });
            }
        }
        let mut owned = vec![self.signal_self, self.rank_data];
        if let Some(fusion) = self.fusion.as_ref() {
            owned.extend_from_slice(&fusion.buffers);
            owned.push(fusion.workspace_dev);
            owned.push(fusion.flag_dev);
        }
        for ptr in owned {
            if !ptr.is_null() {
                // SAFETY: each came from `cudaMalloc` in this object and is
                // freed once.
                ignore_in_drop(unsafe { cudaFree(ptr) });
            }
        }
    }
}

// ── the ABI forms ────────────────────────────────────────────────────────
//
// `custom_all_reduce.hpp:143-201` declared four free functions so the shim
// could name them without a C++ type in the signature. They survive as the
// same four shapes, `car` still an opaque handle, because `table::gemm`'s
// two rows spell their first operand `KernelParam::CustomAllReduce`, which
// `kernels/src/lib.rs:1082-1132` spells `*mut c_void` -- and the owner's
// constraint is that the model compiler must not be able to tell whether a
// symbol is cuBLAS or a JIT'd kernel. It equally must not be able to tell
// whether it is a Rust struct.

/// Reborrow an opaque `car` handle.
///
/// # Safety
///
/// `car` must be null or a pointer to a live [`CustomAllReduce`] owned by
/// the caller, not aliased for the duration of the call.
unsafe fn reborrow<'a>(car: *mut c_void) -> Option<&'a mut CustomAllReduce> {
    if car.is_null() {
        return None;
    }
    // SAFETY: the caller's contract.
    Some(unsafe { &mut *car.cast::<CustomAllReduce>() })
}

/// `custom_all_reduce.hpp:164-180` — the free form of
/// [`CustomAllReduce::all_reduce_bf16`].
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle; `input` and `output`
/// address at least `count` bf16 elements on the device.
#[must_use]
pub unsafe fn all_reduce_bf16(
    car: *mut c_void,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
    stream: *mut c_void,
) -> AllReduce {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return AllReduce::Declined(Decline::NoInstance);
    };
    car.all_reduce_bf16(input, output, count, stream)
}

/// `custom_all_reduce.hpp:186-201` — the free form of
/// [`CustomAllReduce::all_reduce_residual_rmsnorm_bf16`].
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle; `input`, `residual_inout`
/// and `norm_out` address at least `tokens * hidden` bf16 elements, and
/// `rms_gamma` at least `hidden`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn all_reduce_residual_rmsnorm_bf16(
    car: *mut c_void,
    input: *const c_void,
    residual_inout: *mut c_void,
    rms_gamma: *const c_void,
    norm_out: *mut c_void,
    tokens: c_int,
    hidden: c_int,
    eps: f32,
    stream: *mut c_void,
) -> AllReduce {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return AllReduce::Declined(Decline::NoInstance);
    };
    car.all_reduce_residual_rmsnorm_bf16(
        input,
        residual_inout,
        rms_gamma,
        norm_out,
        tokens,
        hidden,
        eps,
        stream,
    )
}

/// `custom_all_reduce.hpp:150-158` — the free forms of the two lifecycle
/// calls, for a caller holding the handle rather than the struct.
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle.
///
/// # Errors
///
/// [`Error::Invalid`] when `car` is null; otherwise whatever
/// [`CustomAllReduce::register_buffer`] refuses.
pub unsafe fn register_buffer(
    car: *mut c_void,
    buf: *mut c_void,
    buf_bytes: usize,
) -> crate::error::Result<()> {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return Err(Error::invalid("custom_all_reduce", "null handle"));
    };
    car.register_buffer(buf, buf_bytes)
}

/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle.
///
/// # Errors
///
/// [`Error::Invalid`] when `car` is null; otherwise whatever
/// [`CustomAllReduce::register_graph_buffers`] refuses.
pub unsafe fn register_graph_buffers(car: *mut c_void) -> crate::error::Result<()> {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return Err(Error::invalid("custom_all_reduce", "null handle"));
    };
    car.register_graph_buffers()
}

#[cfg(test)]
mod tests {
    use super::{
        AOT_POINTS_AFTER_PRUNING, Decline, FusionPattern, INSTANTIATED, LAMPORT_COMM_CAP,
        RANK_DATA_BYTES, REACHED, SIGNAL_BYTES, UPSTREAM_POINTS, align_up, resolve,
    };

    /// Every constant this file carries is host arithmetic over an upstream
    /// struct layout, and none of it touches a device. That is the whole
    /// reason these are tests and not comments.
    #[test]
    fn the_signal_slab_is_the_upstream_struct() {
        // `vllm_custom_all_reduce.cuh:52-60`: `self_counter[36][8]` at 128B
        // alignment plus `peer_counter[2][36][8]`, `FlagType = uint32_t`.
        assert_eq!(SIGNAL_BYTES, 36 * 8 * 4 + 2 * 36 * 8 * 4);
        // `:62-64`: `struct __align__(16) RankData { void* ptrs[8]; }`.
        assert_eq!(RANK_DATA_BYTES, 8 * std::mem::size_of::<*mut u8>());
    }

    #[test]
    fn the_lamport_cap_is_the_largest_aligned_count_a_signed_word_holds() {
        // The flag block's word 3 is `uint32_t` but read as a signed size
        // downstream; `custom_all_reduce.cu:329-333` capped it here.
        assert_eq!(LAMPORT_COMM_CAP, (1usize << 31) - (1 << 21));
        assert_eq!(align_up(LAMPORT_COMM_CAP, 1 << 21), LAMPORT_COMM_CAP);
    }

    #[test]
    fn the_cross_product_is_the_number_kernels_def_measured() {
        assert_eq!(UPSTREAM_POINTS, 240);
        // `4 nranks x 1 pattern x 2 fp32_acc x 3 leaves`.
        assert_eq!(AOT_POINTS_AFTER_PRUNING, 24);
        assert_eq!(INSTANTIATED.len(), 1);
        assert_eq!(INSTANTIATED[0], FusionPattern::ARResidualRMSNorm);
    }

    #[test]
    fn the_one_reached_point_resolves() {
        let got = resolve(
            REACHED.nranks,
            REACHED.pattern,
            REACHED.fp32_acc,
            true,
            false,
        )
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
}
