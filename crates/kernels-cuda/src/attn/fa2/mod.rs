//! The FlashInfer FA2 lattice: 56 roots over one `attn/fa2.cuh`, and the four
//! launches that fire them.
//!
//! # 56 roots and one carried file
//!
//! `attn/fa2.cuh` is ONE file. What made it 56 compiles is the lattice —
//! `(head_dim, GQA group)` for decode, `(head_dim, CTA_TILE_Q, NUM_MMA_KV)`
//! for prefill — and in the routine world that lattice is 56 [`Root`]s over
//! the same text, differing in the NAME a diagnostic and a cache key carry.
//! The row lists are gone: a routine names its own instantiation, so the 460
//! rows the unit world stated became the two `arms` tables below, which are
//! derived from the same axes the geometry is.
//!
//! Every root is `.upstream()` — the FA2 headers resolve against
//! `kernels/flashinfer` as well as the library set — and carries
//! `--device-as-default-execution-space`, which is load-bearing:
//! `page.cuh`'s and `fastdiv.cuh`'s guarded `__host__` constructors are
//! `#ifndef __CUDACC_RTC__`, and everything else in the closure was written
//! for `nvcc`, which defaults an unannotated function to `__host__
//! __device__` where NVRTC's JIT mode defaults it to `__host__` and refuses.
//!
//! # A `FAMILY` under `attn`'s namespace, and six routines in it
//!
//! Six trace symbols DO name this lattice — `attn::dispatch_attention_flashinfer_*`
//! and `attn::attention_flashinfer_prefill` — so the [`ROUTINES`] table at the
//! bottom is the derived half of their [`crate::sigs`] rows, and
//! [`crate::not_yet_crossed`] states them no longer. The namespace is `attn`
//! rather than `fa2` because that is what a trace says; a second `Family` with
//! that namespace is how a file states its own symbols without editing
//! `attn`'s table, and [`crate::sigs`]' own test refuses a symbol two
//! families both claim.
//!
//! The four `pub fn`s that take a params BLOCK whole — [`decode`],
//! [`prefill`] and their capturing forms — are NOT routines and cannot be: a
//! `#[repr(C)]` mirror of a `__grid_constant__` struct has no `Arg` impl,
//! because a trace statement cannot state 224 bytes of filled kernel
//! parameters. They are what the routines launch through.
//!
//! # What stays in `driver-cuda`
//!
//! The plan CACHES, and the widening of a fire's operands into device
//! addresses. A cache owns `Vec`s and is re-planned once per fire; an arm
//! destructures one into a [`DecodePlan`] / [`PrefillPlan`], which is `Copy`,
//! fixed for the fire, and the whole of what a launch reads out of a cache.
//! The params FILLING and the arm cascades are here, beside the mirrors they
//! fill and the `DecodeArm`/`PrefillArm` they answer with.
//!
//! # The `attention_flashinfer_common.cuh` citations, and the file they named
//!
//! Doc comments here and in [`params`], [`plan`] and `kernels/attn/fa2.cuh`
//! cite `attention_flashinfer_common.cuh:NNN` — the FA2 archive's shared
//! body, the C++ this port was written against, and the line numbers that say
//! WHERE each piece came from. **That file is no longer in the tree.** It was
//! `kernels-cuda/spec/`, kept there after its six `.cu` includers went so that
//! the citations would still land, and it has been deleted with the rest of
//! `spec/`.
//!
//! The citations therefore resolve to nothing, exactly as [`plan`]'s
//! `kernels.def` ones do, and the rule that made those acceptable is the rule
//! here: **a citation may dangle where the content it points at is written
//! out beside it.** That is how this port was written — the plan caches, the
//! params filling and the arm cascades below each state their own semantics,
//! quote the sentence they took from upstream, and use the line number as
//! provenance rather than as the answer. Whoever needs the original reads it
//! out of git history; it was this repository's own file, the shared body of
//! the four `attn/attention_flashinfer_hd{64,128,256,512}.cu` translation
//! units the archive crate compiled, and not a copy of anything upstream
//! ships under that name.

/// FA2's host arithmetic: occupancy, tiling and the KV width lattice.
///
/// `src/fa2.rs` until the dissolution, where it sat as a top-level module
/// beside `x/fa2.rs`'s lattice and `src/fa2/params.rs`'s structs -- one
/// family split across three levels of the tree for no reason a reader
/// could recover. Its symbols were always `attn::`.
pub mod geometry;
/// The four `#[repr(C)]` param structs the launches fill.
pub mod params;
/// Reading a plan cache, and preparing a launch out of one.
///
/// `driver-cuda`'s `fire/flashinfer_fa2_dispatch.rs` until §6.3. It fills the
/// mirrors in [`params`], which are pinned against measured struct layouts —
/// and a filler on the other side of the crate boundary could not be reached
/// by those assertions, which is the whole reason it came down.
pub mod dispatch;
/// The plan caches, the two planner factories, and the plan's H2D.
///
/// `driver-cuda`'s `fire/flashinfer_fa2.rs` until §6.3, and before that
/// `csrc/attn/attention_flashinfer.cu` — a file that carried a `.cu`
/// extension for linkage rather than content: `__global__` 0, `__device__` 0,
/// and one launch. It is host arithmetic over [`geometry`] and
/// [`crate::attn::plan`], so it belongs where both of those are.
pub mod plan;

use core::ffi::c_void;
use core::mem::size_of;

use crate::attn::fa2::params::{
    Buffers, DecodeParams, DecodePlan, DecodeScoreParams, DevicePtr, Partials, PrefillPagedParams,
    PrefillPlan, PrefillScoreParams, make_decode_params, make_prefill_params,
};
use crate::attn::fa2::geometry::{DecodeGeometry, Device, KvWidth, PrefillGeometry};
use crate::jit::{ArgValue, Ctx, Cuda, Family, Launch, Root, Routine};
use crate::routine;
use crate::jit::abi::{bf16, unpack_aggregate};
use kernels::routine::{Arg, Env};
use kernels::{Refusal, Ty};

/// `kernels.def`'s `PIE_ATTN_HEAD_DIM` list, in its order.
///
/// Also upstream's `head_dim_supports_cascade_merge` set, by shared origin
/// rather than by construction — two facts that happen to match, and whoever
/// changes one says which.
pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// `kernels.def`'s `PIE_ATTN_DECODE_GQA` list, in its order.
///
/// `utils.cuh:164-183`'s `DISPATCH_GQA_GROUP_SIZE` has an `else`; 5, 6 and 7
/// route to the prefill path instead.
pub const DECODE_GQA: &[u32] = &[1, 2, 3, 4, 8];

// ── The axes ────────────────────────────────────────────────────────────────

/// Which of `dispatch_decode`'s five branches a fire took.
///
/// The discriminant is the index into [`DecodeRoot::arms`], which is what
/// makes an arm a lookup rather than a match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeArm {
    /// `AttnVariantFull` — full attention, no window, no soft cap.
    Full = 0,
    /// `AttnVariantSoftcap` — a soft cap, windowed or not.
    Softcap = 1,
    /// `AttnVariant` — the sliding-window default.
    Window = 2,
    /// `AttnScoreCaptureFull` over `DecodeScoreParams`.
    CaptureFull = 3,
    /// `AttnScoreCapture` over `DecodeScoreParams`.
    CaptureWindow = 4,
}

/// Which of the ten prefill branches a fire took.
///
/// As [`DecodeArm`]: the discriminant indexes [`PrefillRoot::arms`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillArm {
    /// `kCausal`, `AttnVariantFullSoftcap`.
    CausalFullSoftcap = 0,
    /// `kNone`, `AttnVariantFullSoftcap`.
    NoneFullSoftcap = 1,
    /// `kCausal`, `AttnVariantFull`.
    CausalFull = 2,
    /// `kNone`, `AttnVariantFull`.
    NoneFull = 3,
    /// `kCausal`, `AttnVariantSoftcap` — the windowed soft-cap variant.
    CausalSoftcap = 4,
    /// `kCausal`, `AttnVariant`.
    CausalWindow = 5,
    /// `kCausal`, `AttnScoreCapturePrefill` over `PrefillScoreParams`.
    CausalCapture = 6,
    /// `kNone`, `AttnScoreCapturePrefill` over `PrefillScoreParams`.
    NoneCapture = 7,
    /// `kCustom`, `AttnVariantCustomSoftcap`.
    CustomSoftcap = 8,
    /// `kCustom`, `AttnVariantCustom`.
    Custom = 9,
}

// ── The lattice ─────────────────────────────────────────────────────────────

/// One decode lattice point: its root, and the five arms' instantiations.
#[derive(Debug)]
pub struct DecodeRoot {
    /// One of [`HEAD_DIMS`].
    pub head_dim: u32,
    /// One of [`DECODE_GQA`].
    pub group_size: u32,
    /// The compile this point names.
    pub root: Root,
    /// The template-ids NVRTC is handed, indexed by [`DecodeArm`].
    pub arms: [&'static str; 5],
}

/// One prefill lattice point: its root, and the ten arms' instantiations.
#[derive(Debug)]
pub struct PrefillRoot {
    /// One of [`HEAD_DIMS`].
    pub head_dim: u32,
    /// The planner's `CTA_TILE_Q` — one of `{16, 32, 64, 128}`.
    pub cta_tile_q: u32,
    /// `prefill.cuh:4280`'s `DISPATCH_NUM_MMA_KV`, which this lattice makes a
    /// planning output rather than a runtime switch.
    pub num_mma_kv: u32,
    /// The compile this point names.
    pub root: Root,
    /// The template-ids NVRTC is handed, indexed by [`PrefillArm`].
    pub arms: [&'static str; 10],
}

/// One decode instantiation, spelled as NVRTC is handed it.
macro_rules! decode_inst {
    (
        $ns:literal, $tile:literal, $vec:literal, $bdx:literal, $bdy:literal, $bdz:literal,
        $variant:literal, $params:literal
    ) => {
        concat!(
            "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, ",
            stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
            stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
            "::pie::attn::fa2::", $variant, ", ",
            "::pie::attn::fa2::", $params, ">",
        )
    };
}

/// One decode lattice point, from its six template constants.
///
/// The six are `DecodeGeometry::derive`'s answer for this `(head_dim, GQA
/// group)` on an Ampere-or-newer part at `KvWidth::BF16`, written out rather
/// than computed because a `const fn` cannot build the strings. That they
/// agree is [`tests::decode_literals_match_the_derivation`]'s job.
macro_rules! decode_root {
    (
        hd = $hd:literal, gqa = $g:literal,
        stages = $ns:literal, tile = $tile:literal, vec = $vec:literal,
        bdx = $bdx:literal, bdy = $bdy:literal, bdz = $bdz:literal $(,)?
    ) => {
        DecodeRoot {
            head_dim: $hd,
            group_size: $g,
            root: Root::variant(
                concat!("attn/fa2_decode_hd", stringify!($hd), "_g", stringify!($g)),
                "attn/fa2.cuh",
            ),
            arms: [
                decode_inst!($ns, $tile, $vec, $bdx, $bdy, $bdz, "VariantFull", "DecodeParams"),
                decode_inst!(
                    $ns,
                    $tile,
                    $vec,
                    $bdx,
                    $bdy,
                    $bdz,
                    "VariantWindowSoftcap",
                    "DecodeParams"
                ),
                decode_inst!($ns, $tile, $vec, $bdx, $bdy, $bdz, "VariantWindow", "DecodeParams"),
                decode_inst!(
                    $ns,
                    $tile,
                    $vec,
                    $bdx,
                    $bdy,
                    $bdz,
                    "CaptureFull",
                    "DecodeCaptureParams"
                ),
                decode_inst!(
                    $ns,
                    $tile,
                    $vec,
                    $bdx,
                    $bdy,
                    $bdz,
                    "CaptureWindow",
                    "DecodeCaptureParams"
                ),
            ],
        }
    };
}

/// One prefill instantiation, spelled as NVRTC is handed it.
macro_rules! prefill_inst {
    (
        $q:literal, $mmaq:literal, $kv:literal, $dqk:literal, $dvo:literal,
        $wq:literal, $wkv:literal, $mask:literal, $variant:literal, $params:literal
    ) => {
        concat!(
            "::flashinfer::BatchPrefillWithPagedKVCacheKernel<",
            "::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::",
            $mask,
            ", ",
            stringify!($q),
            ", ",
            stringify!($mmaq),
            ", ",
            stringify!($kv),
            ", ",
            stringify!($dqk),
            ", ",
            stringify!($dvo),
            ", ",
            stringify!($wq),
            ", ",
            stringify!($wkv),
            ", ",
            "::pie::attn::fa2::",
            $variant,
            ">, ",
            "::pie::attn::fa2::",
            $params,
            ">",
        )
    };
}

/// One prefill lattice point, from its eight `KernelTraits` arguments.
///
/// [`decode_root!`]'s argument, with
/// [`tests::prefill_literals_match_the_derivation`] as the check.
macro_rules! prefill_root {
    (
        hd = $hd:literal, q = $q:literal, kv = $kv:literal,
        mma_q = $mmaq:literal, d_qk = $dqk:literal, d_vo = $dvo:literal,
        warps_q = $wq:literal, warps_kv = $wkv:literal $(,)?
    ) => {
        PrefillRoot {
            head_dim: $hd,
            cta_tile_q: $q,
            num_mma_kv: $kv,
            root: Root::variant(
                concat!(
                    "attn/fa2_prefill_hd",
                    stringify!($hd),
                    "_q",
                    stringify!($q),
                    "_kv",
                    stringify!($kv),
                ),
                "attn/fa2.cuh",
            ),
            arms: [
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCausal",
                    "VariantFullSoftcap",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kNone",
                    "VariantFullSoftcap",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCausal",
                    "VariantFull",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kNone",
                    "VariantFull",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCausal",
                    "VariantWindowSoftcap",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCausal",
                    "VariantWindow",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCausal",
                    "CapturePrefill",
                    "PrefillCaptureParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kNone",
                    "CapturePrefill",
                    "PrefillCaptureParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCustom",
                    "VariantCustomSoftcap",
                    "PrefillParams"
                ),
                prefill_inst!(
                    $q,
                    $mmaq,
                    $kv,
                    $dqk,
                    $dvo,
                    $wq,
                    $wkv,
                    "kCustom",
                    "VariantCustom",
                    "PrefillParams"
                ),
            ],
        }
    };
}

/// The twenty decode roots — `{64, 128, 256, 512} x {1, 2, 3, 4, 8}`.
///
/// `tile_size_per_bdx = 4` at GQA 1 is `decode.cuh:770`'s special case: with
/// one query head per KV head there is nothing to spread across `bdy`, so the
/// tile goes into the KV axis instead.
pub static DECODE: [DecodeRoot; 20] = [
    decode_root!(hd = 64, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 8, bdy = 1, bdz = 16),
    decode_root!(hd = 64, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 2, bdz = 8),
    // `128 / (8*3) = 5`, so the block is 8x3x5 = **120 threads**, not 128:
    // `bdz` is `num_threads / (bdx * bdy)` by INTEGER division and the
    // remainder is simply not launched.
    decode_root!(hd = 64, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 3, bdz = 5),
    decode_root!(hd = 64, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 4, bdz = 4),
    decode_root!(hd = 64, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 8, bdz = 2),
    decode_root!(hd = 128, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 16, bdy = 1, bdz = 8),
    decode_root!(hd = 128, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 2, bdz = 4),
    decode_root!(hd = 128, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 3, bdz = 2),
    // qwen3 and qwen2's usual shape, and the point the pre-port NVRTC probe
    // was taken at.
    decode_root!(hd = 128, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 4, bdz = 2),
    decode_root!(hd = 128, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 8, bdz = 1),
    decode_root!(hd = 256, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 32, bdy = 1, bdz = 4),
    decode_root!(hd = 256, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 2, bdz = 2),
    decode_root!(hd = 256, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 3, bdz = 1),
    decode_root!(hd = 256, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 4, bdz = 1),
    // The one decode point where `num_threads` exceeds 128: `bdx*bdy = 256`,
    // and `decode.cuh:768` takes the max of that and 128.
    decode_root!(hd = 256, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 8, bdz = 1),
    // **69,632 B of dynamic shared memory** — over the 48 KB default cap, so
    // this point only launches after `cuFuncSetAttribute` raises it. `Launch`
    // asks for it off the byte count; see `jit::launch::issue`.
    decode_root!(hd = 512, gqa = 1, stages = 2, tile = 4, vec = 16, bdx = 32, bdy = 1, bdz = 4),
    decode_root!(hd = 512, gqa = 2, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 2, bdz = 2),
    decode_root!(hd = 512, gqa = 3, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 3, bdz = 1),
    decode_root!(hd = 512, gqa = 4, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 4, bdz = 1),
    decode_root!(hd = 512, gqa = 8, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 8, bdz = 1),
];

/// The thirty-six prefill roots.
///
/// Not a full cross product, and the gaps are upstream's `IsInvalid()` rather
/// than omissions — each is noted where it falls.
pub static PREFILL: [PrefillRoot; 36] = [
    prefill_root!(
        hd = 64,
        q = 16,
        kv = 8,
        mma_q = 1,
        d_qk = 4,
        d_vo = 4,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 64,
        q = 16,
        kv = 4,
        mma_q = 1,
        d_qk = 4,
        d_vo = 4,
        warps_q = 1,
        warps_kv = 4
    ),
    // `NUM_MMA_KV = 1` is absent at head dim 64 for every tile: `NUM_MMA_D_VO`
    // is 4 there, and `prefill.cuh:221-232`'s `NUM_MMA_D_VO == 4 &&
    // NUM_MMA_KV % 2 == 1` clause rejects every odd tile count.
    prefill_root!(
        hd = 64,
        q = 16,
        kv = 2,
        mma_q = 1,
        d_qk = 4,
        d_vo = 4,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 64,
        q = 64,
        kv = 8,
        mma_q = 1,
        d_qk = 4,
        d_vo = 4,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 64,
        q = 64,
        kv = 4,
        mma_q = 1,
        d_qk = 4,
        d_vo = 4,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 64,
        q = 64,
        kv = 2,
        mma_q = 1,
        d_qk = 4,
        d_vo = 4,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 64,
        q = 128,
        kv = 8,
        mma_q = 2,
        d_qk = 4,
        d_vo = 4,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 64,
        q = 128,
        kv = 4,
        mma_q = 2,
        d_qk = 4,
        d_vo = 4,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 64,
        q = 128,
        kv = 2,
        mma_q = 2,
        d_qk = 4,
        d_vo = 4,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 128,
        q = 16,
        kv = 8,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 128,
        q = 16,
        kv = 4,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 128,
        q = 16,
        kv = 2,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 128,
        q = 16,
        kv = 1,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 128,
        q = 64,
        kv = 8,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 128,
        q = 64,
        kv = 4,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 128,
        q = 64,
        kv = 2,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 128,
        q = 64,
        kv = 1,
        mma_q = 1,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    // `NUM_MMA_KV = 8` is absent here: `2 * (8*8 + 2*4*8) = 256`, which is
    // `IsInvalid()`'s register clause exactly at the bound.
    prefill_root!(
        hd = 128,
        q = 128,
        kv = 4,
        mma_q = 2,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 128,
        q = 128,
        kv = 2,
        mma_q = 2,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 128,
        q = 128,
        kv = 1,
        mma_q = 2,
        d_qk = 8,
        d_vo = 8,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 256,
        q = 16,
        kv = 8,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 256,
        q = 16,
        kv = 4,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 256,
        q = 16,
        kv = 2,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 256,
        q = 16,
        kv = 1,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 256,
        q = 64,
        kv = 8,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 256,
        q = 64,
        kv = 4,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 256,
        q = 64,
        kv = 2,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 4,
        warps_kv = 1
    ),
    // **`CTA_TILE_Q = 128` has no valid point at head dim 256.** `NUM_MMA_Q`
    // is 2 and `NUM_MMA_D_VO_TILE` is 16, so `IsInvalid()`'s register clause
    // is `2 * (128 + 8 * NUM_MMA_KV) >= 256` — true for every `NUM_MMA_KV`
    // including zero. Nothing is missing here; nothing can exist.
    prefill_root!(
        hd = 256,
        q = 64,
        kv = 1,
        mma_q = 1,
        d_qk = 16,
        d_vo = 16,
        warps_q = 4,
        warps_kv = 1
    ),
    prefill_root!(
        hd = 512,
        q = 16,
        kv = 8,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 512,
        q = 16,
        kv = 4,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 512,
        q = 16,
        kv = 2,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 1,
        warps_kv = 4
    ),
    prefill_root!(
        hd = 512,
        q = 16,
        kv = 1,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 1,
        warps_kv = 4
    ),
    // `kBf16VOSplit` (`prefill.cuh:4191`): 16-bit KV, head dim >= 512 and
    // `CTA_TILE_Q == 32` pins `NUM_MMA_Q = 1` and both warp counts to 2,
    // which is why this row of four does not follow `get_num_warps_q`.
    prefill_root!(
        hd = 512,
        q = 32,
        kv = 8,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 2,
        warps_kv = 2
    ),
    prefill_root!(
        hd = 512,
        q = 32,
        kv = 4,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 2,
        warps_kv = 2
    ),
    prefill_root!(
        hd = 512,
        q = 32,
        kv = 2,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 2,
        warps_kv = 2
    ),
    prefill_root!(
        hd = 512,
        q = 32,
        kv = 1,
        mma_q = 1,
        d_qk = 32,
        d_vo = 32,
        warps_q = 2,
        warps_kv = 2
    ),
];

/// The decode root for one `(head_dim, GQA group)`, if the lattice holds one.
#[must_use]
pub fn decode_root(head_dim: u32, group_size: u32) -> Option<&'static DecodeRoot> {
    DECODE.iter().find(|p| p.head_dim == head_dim && p.group_size == group_size)
}

/// The prefill root for one `(head_dim, CTA_TILE_Q, NUM_MMA_KV)`.
#[must_use]
pub fn prefill_root(
    head_dim: u32,
    cta_tile_q: u32,
    num_mma_kv: u32,
) -> Option<&'static PrefillRoot> {
    PREFILL.iter().find(|p| {
        p.head_dim == head_dim && p.cta_tile_q == cta_tile_q && p.num_mma_kv == num_mma_kv
    })
}

/// One decode arm's instantiation, by lattice point.
#[must_use]
pub fn decode_instantiation(
    head_dim: u32,
    group_size: u32,
    arm: DecodeArm,
) -> Option<&'static str> {
    decode_root(head_dim, group_size).map(|p| p.arms[arm as usize])
}

/// One prefill arm's instantiation, by lattice point.
#[must_use]
pub fn prefill_instantiation(
    head_dim: u32,
    cta_tile_q: u32,
    num_mma_kv: u32,
    arm: PrefillArm,
) -> Option<&'static str> {
    prefill_root(head_dim, cta_tile_q, num_mma_kv).map(|p| p.arms[arm as usize])
}

// ── The launches ────────────────────────────────────────────────────────────

/// Where a decode fire lands: the lattice point, the arm and the grid.
///
/// Six numbers rather than a plan, because a plan is `driver-cuda`'s
/// vocabulary and this is the whole of what the launch reads out of one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePoint {
    /// One of [`HEAD_DIMS`].
    pub head_dim: u32,
    /// `num_q_heads / num_kv_heads`, one of [`DECODE_GQA`].
    pub group_size: u32,
    /// Which of the five arms the request selected.
    pub arm: DecodeArm,
    /// `decode.cuh:782` — the grid's x, which is the PLAN's
    /// `padded_batch_size` and is why the descriptor must have landed first.
    pub padded_batch_size: u32,
    /// `decode.cuh:782` — the grid's y.
    pub num_kv_heads: u32,
    /// The device the geometry is derived against.
    pub device: Device,
}

/// Where a prefill fire lands. [`DecodePoint`]'s twin.
///
/// `NUM_MMA_KV` is NOT a field: it is
/// [`PrefillGeometry::num_mma_kv`](crate::attn::fa2::geometry::PrefillGeometry::num_mma_kv),
/// derived here from the shared-memory budget, and naming it twice is a
/// second place for it to be wrong. That derivation is the switch that
/// vanished — `prefill.cuh:4280`'s `DISPATCH_NUM_MMA_KV` snapped a runtime
/// budget down to `{8, 4, 2, 1}` and instantiated all four, because the
/// choice depended on a device query. The query is made once here and the
/// fire names one root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPoint {
    /// One of [`HEAD_DIMS`].
    pub head_dim: u32,
    /// The planner's `PrefillPlanInfo::cta_tile_q`. **Read back from the
    /// plan, never recomputed**: the planner split the batch against this
    /// tile, so a fire that chose its own would index a work list built for a
    /// different one.
    pub cta_tile_q: u32,
    /// Which of the ten arms the request selected.
    pub arm: PrefillArm,
    /// `prefill.cuh:4203` — the grid's x.
    pub padded_batch_size: u32,
    /// `prefill.cuh:4203` — the grid's z. **Three axes, with the KV heads in
    /// z**; decode's grid has two and puts them in y.
    pub num_kv_heads: u32,
    /// The device the geometry is derived against.
    pub device: Device,
}

/// A params block a decode `__global__` takes whole.
///
/// Sealed by having only these two impls: the decode entry points are
/// instantiated over `DecodeParams` and `DecodeCaptureParams` and nothing
/// else, so a third mirror would name an instantiation that does not exist.
/// **The arm and the block must agree** — a capture arm over a plain
/// `DecodeParams` is a struct read past its end — and that agreement is the
/// caller's, exactly as it was when the C++ handed a filled block to a
/// `<<<>>>`.
pub trait DecodeBlock: Copy {}
impl DecodeBlock for DecodeParams {}
impl DecodeBlock for DecodeScoreParams {}

/// A params block a prefill `__global__` takes whole. [`DecodeBlock`]'s twin.
pub trait PrefillBlock: Copy {}
impl PrefillBlock for PrefillPagedParams {}
impl PrefillBlock for PrefillScoreParams {}

/// The params block as the kernel's ONE `__grid_constant__` argument.
///
/// The bytes are BORROWED: `cuLaunchKernelEx` copies them out of the caller's
/// binding during the call, and `params` outlives every launch below because
/// it is a parameter of the function that issues one.
fn block<P>(params: &P) -> ArgValue {
    ArgValue::Bytes { ptr: core::ptr::from_ref(params).cast::<u8>(), len: size_of::<P>() }
}

/// Say which lattice point or geometry refused, once, and refuse.
///
/// [`Refusal`] is `Copy` and carries no message, so the detail goes to the
/// log the way `Ctx::launch`'s does — a refusing kernel is asked for once per
/// layer per token.
fn no_point(what: &'static str, why: &dyn core::fmt::Display) -> Refusal {
    tracing::error!(why = %why, "an FA2 fire found no kernel");
    Refusal::Unstated { what }
}

/// `BatchDecodeWithPagedKVCacheDispatched`'s launch, `decode.cuh:770-783`.
///
/// The grid is `dim3(padded_batch_size, num_kv_heads)`, the block is
/// `dim3(bdx, bdy, bdz)` and the dynamic shared allocation is
/// `decode.cuh:772-775` — all three out of
/// [`DecodeGeometry`](crate::attn::fa2::geometry::DecodeGeometry), at `KvWidth::BF16`.
///
/// **bf16 KV throughout, and that is not a simplification.** The lattice
/// carries one KV width; an FP8 cache is widened by
/// `x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active` before a page
/// reaches FA2, which is why that kernel's callers sit immediately above
/// this one.
///
/// # Errors
///
/// [`Refusal::Unstated`] if the lattice holds no root for `at`'s point or the
/// geometry refused — either way with the detail in the log — and
/// [`Refusal::Device`] if the compile, the load or the launch refused.
///
/// # Safety
///
/// Every device address inside `params` must name memory of the extent the
/// kernel reads or writes, and `ctx`'s stream must outlive the launch. The
/// same assertion the caller made when it handed a filled block to a
/// `<<<>>>`.
pub fn decode<P: DecodeBlock>(ctx: &Ctx, at: DecodePoint, params: &P) -> Result<(), Refusal> {
    let Some(point) = decode_root(at.head_dim, at.group_size) else {
        return Err(no_point(
            "an FA2 decode lattice point",
            &format_args!(
                "no decode root for head_dim {} at GQA group {}",
                at.head_dim, at.group_size
            ),
        ));
    };
    let geometry = DecodeGeometry::derive(at.head_dim, at.group_size, KvWidth::BF16, at.device)
        .map_err(|why| no_point("the FA2 decode geometry", &why))?;
    // SAFETY: the caller's contract -- every device address inside `params`
    // addresses live memory of the extent the kernel reads it as, and the
    // block itself outlives the copy because it is this call's parameter.
    unsafe {
        ctx.launch_at(
            &point.root,
            point.arms[at.arm as usize],
            Launch::grid(
                DecodeGeometry::grid(at.padded_batch_size, at.num_kv_heads),
                geometry.block(),
            )
            .smem(geometry.smem_bytes),
            &[block(params)],
        )
    }
}

/// `BatchPrefillWithPagedKVCacheDispatched`'s launch, `prefill.cuh:4203-4297`.
///
/// The grid is `dim3(padded_batch_size, 1, num_kv_heads)`, the block is
/// `dim3(32, NUM_WARPS_Q, NUM_WARPS_KV)` and the shared allocation is
/// `sizeof(KTraits::SharedStoragePaged)`, re-derived by
/// [`PrefillGeometry::shared_storage_paged`](crate::attn::fa2::geometry::PrefillGeometry::shared_storage_paged)
/// and checked against NVRTC's own `sizeof` at one point (49232 == 49232).
///
/// `use_fp16_qk_reduction` is `false` and cannot usefully be anything else:
/// `prefill.cuh:4208-4210`'s `std::conditional` also requires
/// `is_same_v<DTypeQ, half>`, and this lattice is bf16 throughout, so `true`
/// would be unreachable rather than merely unused.
///
/// # Errors
///
/// As [`decode`].
///
/// # Safety
///
/// As [`decode`].
pub fn prefill<P: PrefillBlock>(ctx: &Ctx, at: PrefillPoint, params: &P) -> Result<(), Refusal> {
    let geometry =
        PrefillGeometry::derive(at.head_dim, at.cta_tile_q, KvWidth::BF16, false, at.device)
            .map_err(|why| no_point("the FA2 prefill geometry", &why))?;
    let Some(point) = prefill_root(at.head_dim, at.cta_tile_q, geometry.num_mma_kv) else {
        return Err(no_point(
            "an FA2 prefill lattice point",
            &format_args!(
                "no prefill root for head_dim {} at CTA_TILE_Q {}, NUM_MMA_KV {}",
                at.head_dim, at.cta_tile_q, geometry.num_mma_kv
            ),
        ));
    };
    // SAFETY: as [`decode`]'s.
    unsafe {
        ctx.launch_at(
            &point.root,
            point.arms[at.arm as usize],
            Launch::grid(
                PrefillGeometry::grid(at.padded_batch_size, at.num_kv_heads),
                geometry.block(),
            )
            .smem(geometry.smem_bytes),
            &[block(params)],
        )
    }
}

// ── Which arm a request selects ─────────────────────────────────────────────
//
// These are cascades over the REQUEST's own flags, not over the geometry, and
// they answer with the enums above, which is why they live beside them. Each
// one's order is upstream's and each order is load-bearing; see the doc.

/// Which decode variant a request selects, `dispatch_decode`
/// (`attention_flashinfer_common.cuh:697-722`).
///
/// **The order is load-bearing.** A windowed layer that also has a soft cap
/// takes the soft-cap arm, because the soft-cap test comes second and the
/// window arm is the fallthrough. Reordering these three `if`s is a silent
/// numerics change.
#[must_use]
pub fn decode_arm(
    full_attention_variant: bool,
    window_left: i32,
    logits_soft_cap: f32,
) -> DecodeArm {
    if full_attention_variant && window_left < 0 && logits_soft_cap <= 0.0 {
        return DecodeArm::Full;
    }
    if logits_soft_cap > 0.0 {
        return DecodeArm::Softcap;
    }
    DecodeArm::Window
}

/// Which capturing decode arm a request selects,
/// `dispatch_decode_capture` (`attention_flashinfer_common.cuh`).
///
/// Two arms, not three: the capture variants are instantiated over
/// `AttnScoreCaptureFull` and `AttnScoreCapture` only. `None` is the C++'s
/// `throw` — a soft cap or a window on a capturing dispatch names an
/// instantiation that was never built, and there is nothing to fall back to.
#[must_use]
pub fn decode_capture_arm(
    full_attention_variant: bool,
    window_left: i32,
    logits_soft_cap: f32,
) -> Option<DecodeArm> {
    if logits_soft_cap > 0.0 || window_left >= 0 {
        return None;
    }
    Some(if full_attention_variant { DecodeArm::CaptureFull } else { DecodeArm::CaptureWindow })
}

/// Which prefill variant a request selects, `prefill`
/// (`attention_flashinfer_common.cuh`).
///
/// **The asymmetry is upstream's and is kept.** The full-attention branch has
/// all four combinations of causal × soft-cap; the windowed branch has only
/// the causal ones, because a bidirectional windowed prefill is not
/// instantiated. A caller that asks for one lands on `CausalWindow`, exactly
/// as the C++ did — which is a numerics difference, not a fault, and is the
/// reason it is written out here rather than folded into a table.
#[must_use]
pub fn prefill_arm(full_attention_variant: bool, causal: bool, logits_soft_cap: f32) -> PrefillArm {
    if full_attention_variant {
        return match (causal, logits_soft_cap > 0.0) {
            (true, true) => PrefillArm::CausalFullSoftcap,
            (false, true) => PrefillArm::NoneFullSoftcap,
            (true, false) => PrefillArm::CausalFull,
            (false, false) => PrefillArm::NoneFull,
        };
    }
    if logits_soft_cap > 0.0 { PrefillArm::CausalSoftcap } else { PrefillArm::CausalWindow }
}

/// Which capturing prefill arm a request selects, `prefill_capture`.
///
/// [`decode_capture_arm`]'s counterpart, with the same `None`: soft cap or
/// window is an instantiation that does not exist.
#[must_use]
pub fn prefill_capture_arm(
    causal: bool,
    window_left: i32,
    logits_soft_cap: f32,
) -> Option<PrefillArm> {
    if logits_soft_cap > 0.0 || window_left >= 0 {
        return None;
    }
    Some(if causal { PrefillArm::CausalCapture } else { PrefillArm::NoneCapture })
}

/// Which custom-mask prefill arm a request selects, `prefill_custom`.
///
/// Two arms and no causal axis: the mask *is* the causality, so a custom
/// dispatch that also set `CAUSAL` would mask twice.
#[must_use]
pub fn prefill_custom_arm(logits_soft_cap: f32) -> PrefillArm {
    if logits_soft_cap > 0.0 { PrefillArm::CustomSoftcap } else { PrefillArm::Custom }
}

// ── The two plan descriptors, as arguments ──────────────────────────────────
//
// # Why these implement `Arg` directly and not through `Abi`
//
// An `Abi` impl states a C++ SPELLING, because its whole job is to make a
// Rust mirror and a `__global__`'s parameter checkable against each other.
// **Neither of these ever reaches a kernel.** They are host aggregates a
// routine reads to FILL the block the kernel does take, so there is no
// declaration to check them against and a `CPP` string here would be a
// spelling nothing spells. `Arg` is the smaller obligation and the honest
// one: a value the erased call path can recover from `ArgValue::Bytes`.
//
// The tags are `Ty::DecodePlanCache` and `Ty::PrefillPlanCache`, which is
// what they mean. `Ty`'s own doc says the C++ types were incomplete --
// `struct DecodePlanCache;` and nothing more -- so the row world could only
// carry a handle; what crosses here is the handle's readable contents.

impl Arg<Cuda> for DecodePlan {
    const TY: Ty = Ty::DecodePlanCache;

    fn unpack(value: &ArgValue, at: usize) -> Result<Self, Refusal> {
        unpack_aggregate::<Self>(value, at, Ty::DecodePlanCache)
    }
}

impl Arg<Cuda> for PrefillPlan {
    const TY: Ty = Ty::PrefillPlanCache;

    fn unpack(value: &ArgValue, at: usize) -> Result<Self, Refusal> {
        unpack_aggregate::<Self>(value, at, Ty::PrefillPlanCache)
    }
}

// ── The six dispatches a trace states ───────────────────────────────────────

/// A device address as the params blocks carry one.
///
/// The blocks hold `u64` rather than pointers -- the host may never
/// dereference one, and the device's pointer is 64-bit regardless of the
/// host's -- so this is the one place the width changes.
fn addr<T>(p: *const T) -> DevicePtr {
    p as usize as u64
}

/// The split-KV fold, `VariableLengthMergeStates`.
///
/// A split fire leaves per-chunk partials in `tmp_v`/`tmp_s` -- the params
/// filling redirected `params.o`/`params.lse` there -- and `o` means nothing
/// until this has run. `prefill.cuh:4350-4352` and `decode.cuh:822-824` fire
/// exactly this, in exactly this position: same stream, immediately after the
/// attention kernel.
fn fold(ctx: &Ctx, split: &Partials) -> Result<(), Refusal> {
    crate::cascade::merge_states_varlen(
        ctx,
        split.tmp_v as usize as *mut bf16,
        split.tmp_s as usize as *mut f32,
        split.indptr as usize as *mut i32,
        split.o as usize as *mut bf16,
        split.lse as usize as *mut f32,
        split.max_seq_len,
        split.seq_len as usize as *mut u32,
        split.num_heads,
        split.head_dim,
    )
}

/// Everything the four paged dispatches read the same way, widened once.
#[allow(clippy::too_many_arguments)]
fn buffers(
    q: *const bf16,
    k_pages: *mut bf16,
    v_pages: *mut bf16,
    o: *mut bf16,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    qo_indptr: *const u32,
    lse: *mut f32,
    int_buffer: *mut c_void,
    float_buffer: *mut c_void,
) -> Buffers {
    Buffers {
        q: addr(q),
        k_pages: addr(k_pages),
        v_pages: addr(v_pages),
        o: addr(o),
        kv_page_indices: addr(kv_page_indices),
        kv_page_indptr: addr(kv_page_indptr),
        kv_last_page_lens: addr(kv_last_page_lens),
        qo_indptr: addr(qo_indptr),
        lse: addr(lse),
        int_buffer: addr(int_buffer),
        float_buffer: addr(float_buffer),
    }
}

/// An empty plan cache, as a refusal.
///
/// `attention_flashinfer.cu:504-508` threw here. A value rather than a panic
/// because an unplanned cache is a caller-ordering mistake, which is
/// recoverable, and not a broken JIT, which is not.
const UNPLANNED_DECODE: Refusal = Refusal::Unstated { what: "a planned FA2 decode cache" };

/// A capturing dispatch asked for with a soft cap or a window.
///
/// `attention_flashinfer.cu:551-560` threw here, in these words: the two
/// capture arms are instantiated over `AttnScoreCapture` and
/// `AttnScoreCaptureFull` only, and neither composes with the soft-cap or
/// sliding-window variants -- there is no such instantiation to name.
const CAPTURE_VARIANT: Refusal =
    Refusal::Unstated { what: "a score capture without a soft cap or a window" };

/// A capturing dispatch with no sink to capture into.
///
/// `attention_flashinfer.cu:546-549` and `:849-856`. A null base, or (prefill)
/// a zero window, would make the kernel write every row it was asked to
/// observe to nothing.
const CAPTURE_SINK: Refusal = Refusal::Absent { what: "the score sink" };

/// `attn::dispatch_attention_flashinfer_decode`.
///
/// `dispatch_attention_flashinfer_decode_bf16`
/// (`attention_flashinfer.cu:490-522`) and everything it called: fill the
/// block, pick the arm, launch, and -- when the plan split KV -- fold.
///
/// **bf16 KV, and that is not a simplification.** The lattice carries one KV
/// width; a quantised cache is widened by
/// `x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active` before a page
/// reaches FA2, and `k_pages`/`v_pages` here are the layer's bf16 mirrors.
/// That prelude is the ARM's, because it takes a `&KvLayer` and a layer view
/// is not an argument a trace can state.
///
/// # Errors
///
/// [`Refusal::Unstated`] for an unplanned cache or a lattice point that does
/// not exist, and [`Refusal::Device`] if the compile, the load or the launch
/// refused.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_flashinfer_decode(
    ctx: &Ctx,
    q: *const bf16,
    o: *mut bf16,
    k_pages: Env<*mut bf16>,
    v_pages: Env<*mut bf16>,
    kv_page_indices: Env<*const u32>,
    kv_page_indptr: Env<*const u32>,
    kv_last_page_lens: Env<*const u32>,
    lse: Env<*mut f32>,
    int_buffer: Env<*mut c_void>,
    float_buffer: Env<*mut c_void>,
    plan: Env<DecodePlan>,
    window_left: i32,
    logits_soft_cap: Env<f32>,
    sm_scale: Env<f32>,
    broadcast_q: Env<bool>,
) -> Result<(), Refusal> {
    let plan = plan.into_inner();
    if !plan.valid {
        return Err(UNPLANNED_DECODE);
    }
    let bufs = buffers(
        q,
        *k_pages,
        *v_pages,
        o,
        *kv_page_indices,
        *kv_page_indptr,
        *kv_last_page_lens,
        // Decode has one query row per request, so there is no QO indptr.
        core::ptr::null(),
        *lse,
        *int_buffer,
        *float_buffer,
    );
    let arm = decode_arm(plan.full_attention_variant, window_left, *logits_soft_cap);
    let (params, split) =
        make_decode_params(&plan, &bufs, window_left, *logits_soft_cap, *sm_scale, *broadcast_q);
    decode(ctx, decode_at(&plan, arm, params.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// `attn::dispatch_attention_flashinfer_decode_capture`.
///
/// [`dispatch_attention_flashinfer_decode`] writing the pre-softmax logits to
/// a ragged sink as it goes, `attention_flashinfer.cu:537-607`. The block is
/// [`DecodeScoreParams`] rather than [`DecodeParams`] — `PieScoreParams`
/// derives from `BatchDecodeParams` — so every field the plain filler writes
/// is written by the same call, and the difference is two pointers and the
/// instantiation.
///
/// **`logits_soft_cap` is forced to zero** in the block, matching the C++,
/// which passed `0.f` after refusing a non-zero one: the capture arms are not
/// instantiated over the soft-cap variant, so a value the kernel could not
/// honour would be a lie in the params rather than an error.
///
/// The post-kernels (`attn::attn_score_normalize`, `attn::attn_score_fold_heads`)
/// are NOT fired here and were not fired by the C++ either — they belong to
/// `driver-cuda`'s `fire/attn_score.rs`, on this stream, immediately after
/// this returns.
///
/// # Errors
///
/// As [`dispatch_attention_flashinfer_decode`], plus [`Refusal::Absent`] for a
/// null sink and [`Refusal::Unstated`] for a soft cap or a window, neither of
/// which composes with capture.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_flashinfer_decode_capture(
    ctx: &Ctx,
    q: *const bf16,
    o: *mut bf16,
    k_pages: Env<*mut bf16>,
    v_pages: Env<*mut bf16>,
    kv_page_indices: Env<*const u32>,
    kv_page_indptr: Env<*const u32>,
    kv_last_page_lens: Env<*const u32>,
    lse: Env<*mut f32>,
    int_buffer: Env<*mut c_void>,
    float_buffer: Env<*mut c_void>,
    plan: Env<DecodePlan>,
    score_out: Env<*mut f32>,
    score_indptr: Env<*const i32>,
    window_left: i32,
    logits_soft_cap: Env<f32>,
    sm_scale: Env<f32>,
    broadcast_q: Env<bool>,
) -> Result<(), Refusal> {
    let plan = plan.into_inner();
    if !plan.valid {
        return Err(UNPLANNED_DECODE);
    }
    // `:546-549`, before the variant test, and in that order.
    if score_out.is_null() || score_indptr.is_null() {
        return Err(CAPTURE_SINK);
    }
    let Some(arm) = decode_capture_arm(plan.full_attention_variant, window_left, *logits_soft_cap)
    else {
        return Err(CAPTURE_VARIANT);
    };
    let bufs = buffers(
        q,
        *k_pages,
        *v_pages,
        o,
        *kv_page_indices,
        *kv_page_indptr,
        *kv_last_page_lens,
        core::ptr::null(),
        *lse,
        *int_buffer,
        *float_buffer,
    );
    let (base, split) = make_decode_params(&plan, &bufs, window_left, 0.0, *sm_scale, *broadcast_q);
    let params = DecodeScoreParams {
        base,
        score_out: addr(*score_out),
        score_indptr: addr(*score_indptr),
    };
    decode(ctx, decode_at(&plan, arm, params.base.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// `attn::dispatch_attention_flashinfer_prefill_bf16`,
/// `attention_flashinfer.cu:776-836`.
///
/// The one FA2 row whose KV comes in ALREADY bf16: the fire states `k_pages`
/// and `v_pages` rather than a layer view, so there is no dequant prelude here
/// and there was none in the C++ either.
///
/// The arm reads the PLAN's own variant and mask flags, which is what lets one
/// symbol serve a causal decoder layer and a bidirectional ViT: `tower/qwen3_vl`
/// plans with `causal_mask: false` and fires this.
///
/// # Errors
///
/// As [`dispatch_attention_flashinfer_decode`], plus [`Refusal::Unstated`] if
/// the plan names the Hopper route. It cannot today —
/// `fire::flashinfer_fa2::plan_prefill` writes `use_sm90 = false` — and the
/// refusal is kept so that wiring an SM90 family is one conditional and not an
/// audit.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_flashinfer_prefill_bf16(
    ctx: &Ctx,
    q: *const bf16,
    o: *mut bf16,
    k_pages: Env<*mut bf16>,
    v_pages: Env<*mut bf16>,
    qo_indptr: Env<*const u32>,
    kv_page_indices: Env<*const u32>,
    kv_page_indptr: Env<*const u32>,
    kv_last_page_lens: Env<*const u32>,
    lse: Env<*mut f32>,
    int_buffer: Env<*mut c_void>,
    float_buffer: Env<*mut c_void>,
    plan: Env<PrefillPlan>,
    logits_soft_cap: Env<f32>,
    sm_scale: Env<f32>,
) -> Result<(), Refusal> {
    let plan = plan.into_inner();
    prefill_plan_usable(&plan)?;
    let bufs = buffers(
        q,
        *k_pages,
        *v_pages,
        o,
        *kv_page_indices,
        *kv_page_indptr,
        *kv_last_page_lens,
        *qo_indptr,
        *lse,
        *int_buffer,
        *float_buffer,
    );
    let arm = prefill_arm(plan.full_attention_variant, plan.causal_mask, *logits_soft_cap);
    let (params, split) = make_prefill_params(&plan, &bufs, *logits_soft_cap, *sm_scale);
    prefill(ctx, prefill_at(&plan, arm, params.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// `attn::dispatch_attention_flashinfer_prefill_capture_bf16`,
/// `attention_flashinfer.cu:837-934`.
///
/// [`dispatch_attention_flashinfer_prefill_bf16`] with the score sink and the
/// observation window, over [`PrefillScoreParams`]. As the decode capture: the
/// base is filled by the same filler and the soft cap is forced to zero after
/// the refusal.
///
/// `score_window` is the OBSERVATION window in query rows, not the attention
/// one. A zero is a sink with no rows, which the kernel would still index
/// into, so it is refused with the null bases rather than defaulted.
///
/// # Errors
///
/// As [`dispatch_attention_flashinfer_prefill_bf16`], plus the capture
/// refusals of [`dispatch_attention_flashinfer_decode_capture`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_flashinfer_prefill_capture_bf16(
    ctx: &Ctx,
    q: *const bf16,
    o: *mut bf16,
    k_pages: Env<*mut bf16>,
    v_pages: Env<*mut bf16>,
    qo_indptr: Env<*const u32>,
    kv_page_indices: Env<*const u32>,
    kv_page_indptr: Env<*const u32>,
    kv_last_page_lens: Env<*const u32>,
    lse: Env<*mut f32>,
    int_buffer: Env<*mut c_void>,
    float_buffer: Env<*mut c_void>,
    plan: Env<PrefillPlan>,
    score_out: Env<*mut f32>,
    score_indptr: Env<*const i32>,
    score_window: Env<u32>,
    logits_soft_cap: Env<f32>,
    sm_scale: Env<f32>,
) -> Result<(), Refusal> {
    let plan = plan.into_inner();
    // `:849-856`. The sink first, then the variant, then the plan -- the C++'s
    // order, and the window is part of the sink here rather than of the
    // variant.
    if score_out.is_null() || score_indptr.is_null() || *score_window == 0 {
        return Err(CAPTURE_SINK);
    }
    let Some(arm) = prefill_capture_arm(plan.causal_mask, plan.window_left, *logits_soft_cap) else {
        return Err(CAPTURE_VARIANT);
    };
    prefill_plan_usable(&plan)?;
    let bufs = buffers(
        q,
        *k_pages,
        *v_pages,
        o,
        *kv_page_indices,
        *kv_page_indptr,
        *kv_last_page_lens,
        *qo_indptr,
        *lse,
        *int_buffer,
        *float_buffer,
    );
    let (base, split) = make_prefill_params(&plan, &bufs, 0.0, *sm_scale);
    let params = PrefillScoreParams {
        base,
        score_out: addr(*score_out),
        score_indptr: addr(*score_indptr),
        score_window: *score_window,
    };
    prefill(ctx, prefill_at(&plan, arm, params.base.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// `attn::dispatch_attention_flashinfer_prefill_custom`,
/// `attention_flashinfer.cu:1115-1224`.
///
/// The arbitrary-mask prefill: the fire supplies a packed bit per
/// `(qo_row, kv_pos)` and the kernel reads it instead of deriving causality.
/// Like the decode, it takes a quantised layer's bf16 mirrors, so the dequant
/// prelude is the arm's.
///
/// **`window_left` is `-1` here and NOT the plan's** — `:1163` sets it
/// literally, because the mask states the visibility and a window on top of it
/// would mask twice. This is the one place the filler's plan-sourced window is
/// overwritten, and it is overwritten after the call rather than parameterised
/// into it so that the deviation is visible.
///
/// # Errors
///
/// As [`dispatch_attention_flashinfer_prefill_bf16`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_flashinfer_prefill_custom(
    ctx: &Ctx,
    q: *const bf16,
    o: *mut bf16,
    k_pages: Env<*mut bf16>,
    v_pages: Env<*mut bf16>,
    qo_indptr: Env<*const u32>,
    kv_page_indices: Env<*const u32>,
    kv_page_indptr: Env<*const u32>,
    kv_last_page_lens: Env<*const u32>,
    lse: Env<*mut f32>,
    int_buffer: Env<*mut c_void>,
    float_buffer: Env<*mut c_void>,
    plan: Env<PrefillPlan>,
    mask: Env<*const u8>,
    mask_indptr: Env<*const i32>,
    logits_soft_cap: Env<f32>,
    sm_scale: Env<f32>,
) -> Result<(), Refusal> {
    let plan = plan.into_inner();
    let arm = prefill_custom_arm(*logits_soft_cap);
    prefill_plan_usable(&plan)?;
    let bufs = buffers(
        q,
        *k_pages,
        *v_pages,
        o,
        *kv_page_indices,
        *kv_page_indptr,
        *kv_last_page_lens,
        *qo_indptr,
        *lse,
        *int_buffer,
        *float_buffer,
    );
    let (mut params, split) = make_prefill_params(&plan, &bufs, *logits_soft_cap, *sm_scale);
    // `:1150-1155`, `:1163`.
    params.maybe_custom_mask = addr(*mask);
    params.maybe_mask_indptr = addr(*mask_indptr);
    params.window_left = -1;
    prefill(ctx, prefill_at(&plan, arm, params.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// `attn::attention_flashinfer_prefill`, `attention_flashinfer.cu:1077-1113`
/// — the PLANLESS prefill.
///
/// `whole = true`, and the reason is in the name: no plan cache crosses the
/// fire for this symbol, so one is built over the WHOLE batch on the way in
/// and thrown away. **That planning is the arm's**, because a plan is
/// `driver-cuda`'s vocabulary and the planner is its `plan_prefill` — a
/// hundred lines of policy over `crate::attn::plan::prefill`, including the
/// graph-mode demotion retry and the head-dim gate. What arrives here is the
/// same [`PrefillPlan`] every other prefill takes, which is why this body is
/// its sibling's and not a second one.
///
/// `:1063-1067` fixes three flags this path never varies:
/// `enable_cuda_graph = false`, `full_attention_variant = false`,
/// `causal_mask = true`. So [`prefill_arm`] always answers `CausalSoftcap` or
/// `CausalWindow`, which is what the plan the arm hands over states.
///
/// # Errors
///
/// As [`dispatch_attention_flashinfer_prefill_bf16`].
#[allow(clippy::too_many_arguments)]
pub fn attention_flashinfer_prefill(
    ctx: &Ctx,
    q: *const bf16,
    o: *mut bf16,
    k_pages: Env<*mut bf16>,
    v_pages: Env<*mut bf16>,
    qo_indptr: Env<*const u32>,
    kv_page_indices: Env<*const u32>,
    kv_page_indptr: Env<*const u32>,
    kv_last_page_lens: Env<*const u32>,
    lse: Env<*mut f32>,
    int_buffer: Env<*mut c_void>,
    float_buffer: Env<*mut c_void>,
    plan: Env<PrefillPlan>,
    logits_soft_cap: Env<f32>,
    sm_scale: Env<f32>,
) -> Result<(), Refusal> {
    dispatch_attention_flashinfer_prefill_bf16(
        ctx,
        q,
        o,
        k_pages,
        v_pages,
        qo_indptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        lse,
        int_buffer,
        float_buffer,
        plan,
        logits_soft_cap,
        sm_scale,
    )
}

/// The two plan-validity refusals, in the order the C++ made them.
///
/// `:780` tests `valid` and `:783` tests `use_sm90`, and
/// `dispatch_attention_flashinfer_prefill_custom_bf16:1132` tests both in one
/// `if`. Shared by the four prefill routines so that all four make them the
/// same way.
fn prefill_plan_usable(plan: &PrefillPlan) -> Result<(), Refusal> {
    /// `UNPLANNED_DECODE`'s twin.
    const UNPLANNED_PREFILL: Refusal = Refusal::Unstated { what: "a planned FA2 prefill cache" };

    /// The plan was built for the SM90 launcher, which this lattice has not
    /// ported.
    ///
    /// `dispatch_attention_flashinfer_prefill_bf16:783-798` forwarded to a
    /// separate hopper launcher when `cache.use_sm90`. That launcher lived in
    /// the archive's own tree and was never part of the deleted file, so this
    /// is a **routing** gap and not a numerics one: an FA2 symbol fired
    /// against an SM90 plan reads a different `PrefillPlanInfo` layout.
    const SM90_UNPORTED: Refusal = Refusal::Unstated {
        what: "a non-SM90 FA2 prefill plan; the SM90 launcher is not part of this lattice",
    };

    if !plan.valid {
        return Err(UNPLANNED_PREFILL);
    }
    if plan.use_sm90 {
        return Err(SM90_UNPORTED);
    }
    Ok(())
}

/// The lattice point a decode fire lands on.
///
/// Factored out because the plain and capturing forms differ only in the arm,
/// and a second copy of the GQA division is a second place for it to be wrong.
fn decode_at(plan: &DecodePlan, arm: DecodeArm, padded_batch_size: u32) -> DecodePoint {
    DecodePoint {
        head_dim: plan.head_dim as u32,
        group_size: plan.group_size(),
        arm,
        padded_batch_size,
        num_kv_heads: plan.num_kv_heads as u32,
        device: plan.device,
    }
}

/// [`decode_at`]'s twin. `NUM_MMA_KV` is not here, because it is derived from
/// the shared-memory budget by [`PrefillGeometry::derive`] and naming it twice
/// is a second place for it to be wrong.
fn prefill_at(plan: &PrefillPlan, arm: PrefillArm, padded_batch_size: u32) -> PrefillPoint {
    PrefillPoint {
        head_dim: plan.head_dim as u32,
        cta_tile_q: plan.cta_tile_q,
        arm,
        padded_batch_size,
        num_kv_heads: plan.num_kv_heads as u32,
        device: plan.device,
    }
}

/// The six FA2 symbols a trace states, and what it may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated beside one is what no signature carries.
///
/// **`depth_prefix_plan` is on exactly one row in the whole table and it is
/// this one**: `model-ir`'s `trace/` reads it for the union-tail plan
/// swap, and a decode dispatch is the statement that swap is about.
///
/// `attention_flashinfer_prefill` is `whole` because it plans over the whole
/// fire on the way in, so it owes its caller nothing and cannot be handed a
/// row window: the arm walks `qo_indptr` and `kv_page_indptr` on the HOST, and
/// those are R-shaped.
pub static ROUTINES: &[Routine] = &[
    routine!(dispatch_attention_flashinfer_decode, depth_prefix_plan),
    routine!(dispatch_attention_flashinfer_decode_capture),
    routine!(dispatch_attention_flashinfer_prefill_bf16),
    routine!(dispatch_attention_flashinfer_prefill_capture_bf16),
    routine!(dispatch_attention_flashinfer_prefill_custom),
    routine!(attention_flashinfer_prefill, whole),
];

/// The FA2 lattice's six symbols, under the namespace a trace spells them in.
///
/// `attn`, not `fa2`: the namespace is what a trace SAYS, and it says
/// `attn::dispatch_attention_flashinfer_decode`. A second `Family` with that
/// namespace rather than six rows appended to `x::attn::ROUTINES` keeps the
/// declaration beside the bodies; [`crate::sigs`]' `no_symbol_is_declared_twice`
/// is what makes two families under one namespace safe.
pub static FAMILY: Family = crate::family!(ROUTINES);

/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the decode entry point —
/// `decode.cuh:715-718`.
///
/// Upstream multiplies this by the SM count to bound `plan::decode::estimate`'s
/// split. It is a per-CUBIN fact, so asking it COMPILES the point: the answer
/// is not available before the kernel exists.
///
/// `block_threads` is [`DecodeGeometry::num_threads`](crate::attn::fa2::geometry::DecodeGeometry::num_threads)
/// and **not** the product of the block dims. At GQA group 3 those differ
/// (128 against 120) and `decode.cuh:715` passes the former; getting it wrong
/// is a plausible-looking occupancy for a block shape nothing launches.
///
/// # The arm is `Full` and the other four would answer the same
///
/// An arm selects a variant *functor* — a soft cap, a window predicate — and
/// changes neither `num_threads` nor the shared-memory request, which are the
/// only two things the query reads. Probing `Full` therefore answers for all
/// five and avoids compiling a point the fire may not want.
///
/// `None` when there is no such point, the geometry refused, the compile
/// refused or the driver would not say. Every caller answers it with the SM
/// count — one block per SM, the unsplit end of the range, which is
/// conservative rather than wrong.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: Device) -> Option<u32> {
    use cudarc::driver::sys as dr;

    let point = decode_root(head_dim, group_size)?;
    let geometry = DecodeGeometry::derive(head_dim, group_size, KvWidth::BF16, device).ok()?;
    let resolved =
        crate::jit::cache::resolve(&point.root, point.arms[DecodeArm::Full as usize]).ok()?;

    // The >48 KiB opt-in, which the QUERY needs as much as the launch does:
    // `cuOccupancy…` with a dynamic request above the default cap answers for
    // a configuration the function has not been granted. Only head dim 512 at
    // GQA 1 is over it (69,632 B), so this call is not on the common path,
    // and it is unmemoised because it is made once per plan rather than once
    // per launch -- `jit::launch` keeps the launch-path high-water mark.
    if geometry.smem_bytes > 48 * 1024 {
        // SAFETY: `function` came from a module this process keeps loaded.
        unsafe {
            dr::cuFuncSetAttribute(
                resolved.function,
                dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(geometry.smem_bytes).unwrap_or(i32::MAX),
            );
        }
    }

    let mut blocks: core::ffi::c_int = 0;
    // SAFETY: `blocks` is a live out-parameter and `function` came from a
    // module this process keeps loaded.
    let code = unsafe {
        dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &raw mut blocks,
            resolved.function,
            i32::try_from(geometry.num_threads).unwrap_or(i32::MAX),
            usize::try_from(geometry.smem_bytes).unwrap_or(usize::MAX),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return None;
    }
    u32::try_from(blocks).ok().filter(|n| *n > 0)
}

/// The occupancy query, on a build that selected no CUDA runtime.
#[cfg(not(feature = "_cuda"))]
#[must_use]
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: Device) -> Option<u32> {
    let _ = (head_dim, group_size, device);
    None
}

#[cfg(test)]
mod tests {
    use super::{
        DECODE, DECODE_GQA, DecodeArm, FAMILY, HEAD_DIMS, PREFILL, PrefillArm, decode_arm,
        decode_capture_arm, decode_instantiation, decode_root, prefill_arm, prefill_capture_arm,
        prefill_custom_arm, prefill_instantiation, prefill_root,
    };
    use crate::attn::fa2::geometry::{DecodeGeometry, Device, KvWidth, PrefillGeometry};

    /// The cascade's ORDER, which is the part that can be broken silently.
    ///
    /// A windowed layer with a soft cap takes the soft-cap arm, not the window
    /// arm — `attention_flashinfer_common.cuh:697-722` tests the cap second
    /// and falls through to the window. Written as a test because reordering
    /// the three `if`s compiles and changes the kernel.
    #[test]
    fn a_windowed_layer_with_a_soft_cap_takes_the_softcap_arm() {
        assert_eq!(decode_arm(true, -1, 0.0), DecodeArm::Full);
        assert_eq!(decode_arm(false, -1, 0.0), DecodeArm::Window);
        assert_eq!(decode_arm(true, 4096, 0.0), DecodeArm::Window);
        assert_eq!(decode_arm(true, 4096, 30.0), DecodeArm::Softcap);
        assert_eq!(decode_arm(true, -1, 30.0), DecodeArm::Softcap);
    }

    /// Prefill's windowed branch is CAUSAL ONLY, and that is upstream's.
    ///
    /// A bidirectional windowed prefill is not instantiated, so the request
    /// lands on `CausalWindow`. Written down because it is the one place a
    /// caller can ask for something and get something else, and because the
    /// ViT path (`driver-cuda`'s `tower/qwen3_vl`) asks for `causal = false` —
    /// it reaches `NoneFull` only because it also passes
    /// `full_attention_variant = false` with `window_left = -1`... which is
    /// exactly this fallthrough. The assertion below is what keeps that in
    /// view.
    #[test]
    fn a_bidirectional_windowed_prefill_falls_through_to_causal() {
        assert_eq!(prefill_arm(true, true, 0.0), PrefillArm::CausalFull);
        assert_eq!(prefill_arm(true, false, 0.0), PrefillArm::NoneFull);
        assert_eq!(prefill_arm(true, true, 30.0), PrefillArm::CausalFullSoftcap);
        assert_eq!(prefill_arm(true, false, 30.0), PrefillArm::NoneFullSoftcap);
        assert_eq!(prefill_arm(false, true, 30.0), PrefillArm::CausalSoftcap);
        assert_eq!(prefill_arm(false, true, 0.0), PrefillArm::CausalWindow);
        assert_eq!(prefill_arm(false, false, 0.0), PrefillArm::CausalWindow);
    }

    /// The capture arms compose with neither a soft cap nor a window.
    ///
    /// `None` rather than a nearest arm: there is no instantiation, so the
    /// only honest answers are this and a throw, and the C++ threw.
    #[test]
    fn capture_does_not_compose_with_softcap_or_window() {
        assert_eq!(decode_capture_arm(true, -1, 0.0), Some(DecodeArm::CaptureFull));
        assert_eq!(decode_capture_arm(false, -1, 0.0), Some(DecodeArm::CaptureWindow));
        assert_eq!(decode_capture_arm(true, -1, 30.0), None);
        assert_eq!(decode_capture_arm(true, 4096, 0.0), None);
        assert_eq!(prefill_capture_arm(true, -1, 0.0), Some(PrefillArm::CausalCapture));
        assert_eq!(prefill_capture_arm(false, -1, 0.0), Some(PrefillArm::NoneCapture));
        assert_eq!(prefill_capture_arm(true, -1, 30.0), None);
        assert_eq!(prefill_capture_arm(true, 0, 0.0), None);
        assert_eq!(prefill_custom_arm(0.0), PrefillArm::Custom);
        assert_eq!(prefill_custom_arm(30.0), PrefillArm::CustomSoftcap);
    }

    /// The family declares the six symbols a trace states, under `attn`, and
    /// the decode dispatch is the one row in the whole table that carries
    /// `depth_prefix_plan`.
    ///
    /// `model-ir`'s `trace/` reads that column for the union-tail plan
    /// swap. It survived the crossing as `routine!`'s trailing fact; a row
    /// that quietly lost it would lower a different plan and nothing else
    /// would say so.
    #[test]
    fn the_family_declares_the_six_and_keeps_the_depth_prefix_column() {
        let decode = FAMILY
            .routine("attn::dispatch_attention_flashinfer_decode")
            .expect("the decode dispatch is declared");
        assert!(decode.depth_prefix_plan, "the union-tail plan swap reads this column");
        assert!(!decode.whole, "a decode dispatch covers a row range");

        let planless = FAMILY
            .routine("attn::attention_flashinfer_prefill")
            .expect("the planless prefill is declared");
        assert!(planless.whole, "it plans over the whole fire on the way in");
        assert!(!planless.depth_prefix_plan);

        assert_eq!(FAMILY.routines.len(), 6);
        for symbol in [
            "attn::dispatch_attention_flashinfer_decode_capture",
            "attn::dispatch_attention_flashinfer_prefill_bf16",
            "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
            "attn::dispatch_attention_flashinfer_prefill_custom",
        ] {
            let r = FAMILY.routine(symbol).unwrap_or_else(|| panic!("{symbol} is declared"));
            assert!(!r.whole && !r.depth_prefix_plan, "{symbol} states neither fact");
        }
    }

    /// Every decode point's six template constants are the ones
    /// [`DecodeGeometry::derive`] produces for that point.
    #[test]
    fn decode_literals_match_the_derivation() {
        for &head_dim in HEAD_DIMS {
            for &group in DECODE_GQA {
                let point = decode_root(head_dim, group)
                    .unwrap_or_else(|| panic!("no root for hd {head_dim} gqa {group}"));
                let geometry = DecodeGeometry::derive(head_dim, group, KvWidth::BF16, Device::L40S)
                    .unwrap_or_else(|why| panic!("hd {head_dim} gqa {group}: {why}"));
                let wanted = format!(
                    "::flashinfer::BatchDecodeWithPagedKVCacheKernel\
                     <::flashinfer::PosEncodingMode::kNone, {}, {}, {}, {}, {}, {}, ",
                    geometry.num_stages_smem,
                    geometry.tile_size_per_bdx,
                    geometry.vec_size,
                    geometry.bdx,
                    geometry.bdy,
                    geometry.bdz,
                );
                for arm in point.arms {
                    assert!(
                        arm.starts_with(&wanted),
                        "hd {head_dim} gqa {group}: the lattice states\n  {arm}\n\
                         the derivation wants\n  {wanted}",
                    );
                }
            }
        }
    }

    /// Every prefill point's `KernelTraits` arguments are the ones
    /// [`PrefillGeometry::derive`] produces for that point.
    ///
    /// `NUM_MMA_KV` is the point's own rather than the derivation's: the
    /// derivation picks ONE tile count from this box's shared-memory budget
    /// and the lattice carries every count that is valid anywhere, which is
    /// what [`the_derived_num_mma_kv_names_a_root`] checks the other half of.
    #[test]
    fn prefill_literals_match_the_derivation() {
        for point in &PREFILL {
            let geometry = PrefillGeometry::derive(
                point.head_dim,
                point.cta_tile_q,
                KvWidth::BF16,
                true,
                Device::L40S,
            )
            .unwrap_or_else(|why| panic!("{}: {why}", point.root.name));
            let wanted = format!(
                ", {}, {}, {}, {}, {}, {}, {}, ",
                point.cta_tile_q,
                geometry.num_mma_q,
                point.num_mma_kv,
                geometry.num_mma_d_qk,
                geometry.num_mma_d_vo,
                geometry.num_warps_q,
                geometry.num_warps_kv,
            );
            for arm in point.arms {
                assert!(
                    arm.contains(&wanted),
                    "{}: the lattice states\n  {arm}\nthe derivation wants\n  {wanted}",
                    point.root.name,
                );
            }
        }
    }

    /// The `NUM_MMA_KV` the derivation picks on this box names a root that
    /// exists — the fire's own lookup, made ahead of the fire.
    #[test]
    fn the_derived_num_mma_kv_names_a_root() {
        for &head_dim in HEAD_DIMS {
            for &cta_tile_q in &[16u32, 32, 64, 128] {
                let Ok(geometry) = PrefillGeometry::derive(
                    head_dim,
                    cta_tile_q,
                    KvWidth::BF16,
                    true,
                    Device::L40S,
                ) else {
                    continue;
                };
                assert!(
                    prefill_root(head_dim, cta_tile_q, geometry.num_mma_kv).is_some(),
                    "hd {head_dim} q {cta_tile_q} derived NUM_MMA_KV {} and no root holds it",
                    geometry.num_mma_kv,
                );
            }
        }
    }

    /// Fifty-six roots, no name twice, and no instantiation twice.
    ///
    /// The count is the whole point of the pass this file landed in: 56 units
    /// carrying 460 rows became 56 roots carrying none, because a routine
    /// names its own instantiation.
    #[test]
    fn names_and_instantiations_are_unique() {
        let mut names: Vec<&str> = Vec::new();
        let mut instantiations: Vec<&str> = Vec::new();
        for name in DECODE.iter().map(|p| p.root.name).chain(PREFILL.iter().map(|p| p.root.name)) {
            assert!(!names.contains(&name), "{name} is declared twice");
            names.push(name);
        }
        for arm in DECODE.iter().flat_map(|p| p.arms).chain(PREFILL.iter().flat_map(|p| p.arms)) {
            assert!(!instantiations.contains(&arm), "{arm} is stated twice");
            instantiations.push(arm);
        }
        assert_eq!(names.len(), 56);
        assert_eq!(instantiations.len(), 20 * 5 + 36 * 10);
    }

    /// Every instantiation is ABSOLUTELY qualified, and every root asks for
    /// the upstream header set and the execution-space flag.
    #[test]
    fn every_root_states_what_a_compile_of_it_needs() {
        for point in DECODE
            .iter()
            .map(|p| (&p.root, p.arms.as_slice()))
            .chain(PREFILL.iter().map(|p| (&p.root, p.arms.as_slice())))
        {
            let (root, arms) = point;
            assert_eq!(
                root.headers,
                crate::jit::Headers::LibraryAndUpstream,
                "{}: fa2.cuh includes `attn/flashinfer/*`",
                root.name,
            );
            assert_eq!(
                root.options,
                &["--device-as-default-execution-space"],
                "{}: NVRTC's JIT mode defaults an unannotated function to __host__",
                root.name,
            );
            assert_eq!(root.file, "attn/fa2.cuh");
            for arm in arms {
                assert!(arm.starts_with("::flashinfer::"), "{} instantiates {arm}", root.name);
                assert!(
                    !arm.contains("::pie::::"),
                    "{} double-qualified: {arm}",
                    root.name
                );
            }
        }
    }

    /// The template SHAPE, against the strings the unit world handed NVRTC.
    ///
    /// Fifteen strings, character for character, and they cover all fifteen
    /// arms: five decode over one point and ten prefill over another. The
    /// axes are checked against the derivation above and the uniqueness above
    /// that, so what is left for a transcription to get wrong is the template
    /// itself — the qualification, the separators, the argument ORDER — and
    /// that is what one point per template pins.
    #[test]
    fn the_templates_are_what_the_unit_world_handed_nvrtc() {
        let decode = [
            (
                DecodeArm::Full,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie::attn::fa2::VariantFull, ::pie::attn::fa2::DecodeParams>",
            ),
            (
                DecodeArm::Softcap,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie::attn::fa2::VariantWindowSoftcap, ::pie::attn::fa2::DecodeParams>",
            ),
            (
                DecodeArm::Window,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie::attn::fa2::VariantWindow, ::pie::attn::fa2::DecodeParams>",
            ),
            (
                DecodeArm::CaptureFull,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie::attn::fa2::CaptureFull, ::pie::attn::fa2::DecodeCaptureParams>",
            ),
            (
                DecodeArm::CaptureWindow,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie::attn::fa2::CaptureWindow, ::pie::attn::fa2::DecodeCaptureParams>",
            ),
        ];
        for (arm, wanted) in decode {
            assert_eq!(decode_instantiation(64, 1, arm), Some(wanted), "decode hd64 g1 {arm:?}");
        }

        let prefill = [
            (
                PrefillArm::CausalFullSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantFullSoftcap>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::NoneFullSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantFullSoftcap>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalFull,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantFull>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::NoneFull,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantFull>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantWindowSoftcap>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalWindow,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantWindow>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalCapture,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::CapturePrefill>, ::pie::attn::fa2::PrefillCaptureParams>",
            ),
            (
                PrefillArm::NoneCapture,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::CapturePrefill>, ::pie::attn::fa2::PrefillCaptureParams>",
            ),
            (
                PrefillArm::CustomSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantCustomSoftcap>, ::pie::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::Custom,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, 64, 1, 2, 8, 8, 4, 1, ::pie::attn::fa2::VariantCustom>, ::pie::attn::fa2::PrefillParams>",
            ),
        ];
        for (arm, wanted) in prefill {
            assert_eq!(
                prefill_instantiation(128, 64, 2, arm),
                Some(wanted),
                "prefill hd128 q64 kv2 {arm:?}",
            );
        }
    }

    /// A point outside the lattice is `None` rather than a nearest match.
    #[test]
    fn a_point_the_lattice_does_not_hold_is_absent() {
        assert!(decode_root(96, 4).is_none(), "96 is deliberately absent");
        assert!(decode_root(128, 5).is_none(), "5/6/7 route to the prefill path");
        assert!(prefill_root(256, 128, 4).is_none(), "IsInvalid() for every NUM_MMA_KV");
        assert!(decode_root(128, 4).is_some());
    }
}
