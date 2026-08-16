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
use crate::attn::plan::info::{DecodePlanInfo, PrefillPlanInfo};
use crate::jit::{ArgValue, Ctx, Cuda, Family, Launch, Root, Routine};
use crate::routine;
use crate::jit::abi::{bf16, unpack_aggregate};
use kernels::keys;
use kernels::routine::{Arg, Env, In, Out};
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
//
// # `DecodePlan` NO LONGER CROSSES, AND `PrefillPlan` STILL DOES
//
// `.wiki/kilimanjaro.md` §4 plans both away and D1 says a routine takes
// fields rather than a struct. The decode three took their sixteen leaves as
// `Env<keys::Fa2Decode*>` and `decode_plan_of_leaves` rebuilds the aggregate
// inside this file -- so this impl now serves the FOLD and not a parameter,
// and deleting it would take `unpack_aggregate` out of a path the five
// prefill launchers still use. The [`ROUTINES`] ledger carries what the
// prefill half is waiting on, which is the planless arm's local cache and
// not the ceiling.

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

// `UNPLANNED_DECODE` STOOD HERE and is the one leaf the unfold deleted rather
// than named. `attention_flashinfer.cu:504-508` threw on `!cache.valid`; the
// three decode launchers tested it because they held the cache. They no
// longer do, and `bind/table.rs`'s `fa2_decode_leaves` refuses before it
// answers a single leaf -- so a validity flag reaching this file would be an
// answer to a question already asked. `UNPLANNED_PREFILL` below still exists
// because the five prefill launchers still take the aggregate.

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

/// One decode plan's named leaves, back into the aggregate `params.rs`
/// consumes.
///
/// # The offsets are ADDRESSES here, against a zero base
///
/// `offset_ptr(base, off)` is `base.saturating_add(off)` for a non-negative
/// `off` and `base` for a negative one (`params.rs:561`). A zero base makes
/// the first case the identity, so `make_decode_params` writes back exactly
/// the address `bind/table.rs`'s `fa2_decode_leaves` resolved -- which
/// reproduces `offset_ptr` on the other side, so the two agree by
/// construction rather than by review. A null leaf carries `-1`.
///
/// The alternative was forty lines of `DecodeParams` filling copied into this
/// file. `params.rs` stays the one place that knows how a block is laid out.
///
/// `enable_cuda_graph` is not a leaf: the block mask is carved only under
/// `split_kv && enable_cuda_graph`, so a non-null mask says both.
#[allow(clippy::too_many_arguments)]
fn decode_plan_of_leaves(
    request_indices: *const i32,
    kv_tile_indices: *const i32,
    o_indptr: *const i32,
    kv_chunk_size: *const i32,
    block_valid_mask: *const u8,
    tmp_v: *mut f32,
    tmp_s: *mut f32,
    padded_batch: i32,
    split_kv: bool,
    requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    hnd_layout: bool,
    full_attention: bool,
) -> DecodePlan {
    fn at<T>(p: *const T) -> i64 {
        if p.is_null() { -1 } else { p as usize as i64 }
    }
    DecodePlan {
        info: DecodePlanInfo {
            padded_batch_size: i64::from(padded_batch),
            v_offset: at(tmp_v.cast_const()),
            s_offset: at(tmp_s.cast_const()),
            request_indices_offset: at(request_indices),
            kv_tile_indices_offset: at(kv_tile_indices),
            o_indptr_offset: at(o_indptr),
            block_valid_mask_offset: at(block_valid_mask),
            kv_chunk_size_ptr_offset: at(kv_chunk_size),
            enable_cuda_graph: !block_valid_mask.is_null(),
            split_kv,
        },
        device: crate::attn::fa2::plan::fa_device(),
        num_requests: requests,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        int_base_bytes: 0,
        hnd_layout,
        full_attention_variant: full_attention,
        valid: true,
    }
}

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
#[kernels_macros::routine]
pub fn dispatch_attention_flashinfer_decode(
    ctx: &Ctx,
    // `In(0)` STATED, and the same token on all six of this family's
    // launchers. The warrant is `arms/fa2.rs`'s six identical
    // `let q = cx.arg_in(0)?.cast::<bf16>().cast_const();` lines, one per
    // arm; the ROUTINES ledger below carries what changes and what does not.
    //
    // THAT SENTENCE ONCE CONTINUED *"`o` and `lse` stay counted, and that is
    // the ledger's paragraph too -- `Or<T>` is not a region and wrapping it
    // would state nothing."* Both halves were true and the conclusion did not
    // follow: an ATTRIBUTE is not a wrapper, and `#[source(Out(n))]` states
    // the slot without touching the provenance `Or` exists to carry. The two
    // are stated below and this family no longer counts to anything.
    //
    // AND THE ATTRIBUTE IS GONE AGAIN, one Stage 6 round later. `o` and
    // `lse` say the slot in the type -- so all three of this launcher's
    // statement operands are spelled rather than counted, and `q`'s
    // `In<0, _>` is no longer the only one. (`OutSlot` and `Or` carried this
    // between rounds; both are deleted.)
    q: In<0, bf16>,
    // `Out(0)` STATED. The slot is not a guess: `arms/fa2.rs`'s `o_or` is
    // documented `Source::Or(&Out(0), &Attn("o_out"))` and its body reads
    // `cx.arg_out(0)` before falling back, in all six arms; and
    // `model-dsl/src/cuda/base.rs:373` builds this family's two-result
    // statement with `[(shape, BF16), (Tokens x q_heads, F32)]` in that
    // order, so result 0 IS the bf16 attention output.
    //
    // STATING A SLOT IS NOT WRAPPING ONE, and the ledger's *"`Or<T>` IS NOT A
    // REGION AND `o` IS NOT WRAPPED"* stands unamended. `Out<0, _>` would
    // state nothing because `stated_source` reads the LAST path segment; an
    // ATTRIBUTE is read before any wrapper (`kernels-macros/src/lib.rs:216`
    // sets `stated` on the attribute branch first), so the index becomes an
    // answer while `Or` goes on carrying `Provenance::Either`. That
    // provenance is the whole reason `Out<0, _>` is wrong here -- it would
    // make the result REQUIRED on a text that declines to name one.
    //
    // AND `Out<0, bf16>` IS THAT MARK IN A TYPE, which took a
    // repair to become true. `OutSlot` states a slot and claims no
    // rectangle, which is the objection above answered -- but its `Arg` impl
    // omitted `PROV` and took the `Trace` default, so wrapping an `Or` in it
    // did the one thing the `Or` is here to prevent. `148ca6a6a` made the
    // seven position wrappers forward `T::PROV` (`routine.rs:824-902`), and
    // this spelling now says both halves at once: `OutSlot` says WHICH
    // result, `Or` says the statement MAY decline to place it. The ledger
    // below carries the arity arithmetic; the short version is that the band
    // does not move, because `Either` is what this parameter already
    // reported and is what it still reports.
    o: Out<0, bf16>,
    // THE FOUR PAGED FACTS, SPELLED. All six launchers in this file carry the
    // same four, this is the only place the argument is written, and the
    // ROUTINES ledger below carries what the conversion does and does not
    // move.
    //
    // Each was `Env<*mut bf16>` or `Env<*const u32>` and derived the SAME
    // `Source` from the parameter's NAME, through `fact_of`
    // (`kernels-macros/src/lib.rs:640-643`). That the source does not move is
    // the point and also the whole risk, because a change with no observable
    // effect is one nobody can check. What moves is where the claim lives: a
    // table keyed on a string answers a rename by going quiet, and
    // `Env<keys::KvKeys>` cannot be unbound by one.
    //
    // CHECKED AGAINST THE ARM, PER PARAMETER, rather than read off the name.
    // `arms/fa2.rs:226-229` witnesses this launcher and the other five read
    // identically at `:277`, `:328`, `:379`, `:441` and `:629`:
    // `Env(layer.k_bf16_pages.cast::<bf16>())` and
    // `Env(plan_of.kv_page_indices)`. Layer view and plan -- environment on
    // both, with no `cx.arg_in(n)` anywhere near them. `s3-norm` found two
    // `positions` parameters that were statement OPERANDS wearing a fact's
    // name, and converting one of those by name type-checks, binds, and
    // reads the wrong buffer in silence. The check is the work here; the
    // edit is the easy half.
    //
    // `*mut u8` AT THE KEY AND `bf16` IN THE BODY, which is the convention
    // and not an oversight. `keys.rs:483-505`: the plane's element type is
    // the caller's business, because one cache is bf16 under one model and
    // an fp8 pair under another. FA2's is bf16 for a reason this file states
    // twice -- the launcher doc above, and `buffers`' `*mut bf16` -- so the
    // cast below is where an opaque plane meets a dispatch that exists at
    // one width only.
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    // THE FIFTH PAGED ARRAY, AND IT SAID `None` FOR AS LONG AS THE FACT
    // WAS ONLY REACHABLE BY NAME. Two drafts of this comment are worth
    // keeping because each was wrong in a different way and the second is
    // the instructive one.
    //
    // Draft one: *"there is no key to spell"*. False. `fact!` takes its
    // source as `$src:expr` (`keys.rs:189`) and four keys have been passing
    // `Source::Named(<keys::WeightBias as keys::Fact>::KEY)` through it all along
    // (`keys.rs:580-589`). Draft two, correcting it: *"A key is one line
    // away; what is not available is finding it FROM THE NAME."* True, and
    // the line was written -- `keys::KvLastPageLens` (`keys.rs:615`).
    //
    // WHAT THE CONVERSION ACTUALLY FIXES IS A FALSE COLUMN ENTRY, not a
    // notation. `fact_of` had no arm for this name, so `derive_all` fell to
    // `Shape::Env => None` -- and `None` is a REFUSAL, "no source names
    // this". There is a source and every arm binds it:
    // `arms/fa2.rs:231` passes `Env(plan_of.kv_last_page_lens)`, and
    // `Facts::plan` fills that field from `a.kv_last_page_lens_d`
    // (`Fire::plan`), which is the `AttnCtx` field name the source's
    // string quotes. Six columns in this file were saying nothing about a
    // pointer the binder holds by a name that matches character for
    // character.
    //
    // ARITY IS UNTOUCHED AND THAT IS CHECKABLE RATHER THAN ASSUMED.
    // `Env<T>::PROV` is `Provenance::Env` (`routine.rs:355`) and `fact!`
    // forces `PROV = Env` on every key, so the `(Ty, Provenance)` pair
    // `impl_kernel_fn!` records is bit-identical before and after; `TY`
    // forwards to `*const u32` on both sides. What changes is `Derived`:
    // `source: None, stated: false` becomes
    // `Source::Named(<keys::KvLastPageLens as keys::Fact>::KEY), stated: true`.
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // THE LSE IS ENVIRONMENT ON THIS SYMBOL, AND THE TEXT THAT WANTED IT AS A
    // RESULT HAS A SYMBOL OF ITS OWN.
    //
    // This was `Out<1, f32>` and before that `#[source(Out(1))]`, and the
    // argument that put the mark there is kept below because it was right
    // about the slot and the slot is where it went -- onto
    // [`dispatch_attention_flashinfer_decode_lse`], not onto this launcher.
    //
    // # What `Or` was carrying, and which half of it this parameter used
    //
    // `Or<T>` plants exactly one fact: `const PROV: Provenance = Either`
    // (`kernels/src/routine.rs:522`), forwarding everything else. `Either` on
    // a `Binds::Writes` parameter is read twice. `arity_problem`
    // (`model-ir/src/kernels.rs:303`) counts it into `opt_writes` and so
    // raises §6.2's CEILING to `writes + opt_writes`; and
    // `accepts_an_unstated_result` (`:227`) reports that the symbol will
    // accept a value-producing region's buffer for a statement that declares
    // no result at all, which `model-compiler/src/lower/walk.rs:472` reads
    // before it substitutes.
    //
    // ONLY THE CEILING WAS EVER IN PLAY HERE, and only because ONE SYMBOL WAS
    // TWO ARITIES. `base.rs:259`'s `attention_flashinfer_decode` declares one
    // result -- or none, inside a value region -- and `base.rs:373`'s
    // `attention_flashinfer_decode_lse` declares two, and both named this
    // string. That is D2 (`.wiki/kilimanjaro3.md` §3.8) exactly: *an optional
    // argument means two functions*, and the two texts differ in ARITY rather
    // than in body. They are two symbols now and the band here is `[0, 1]`,
    // with no second result left to admit.
    //
    // `o` KEEPS ITS `Or` AND THIS DOES NOT, which is the asymmetry to hold on
    // to. `o` uses the OTHER reader: `attn_at` (`base.rs:1121`) declares no
    // result inside a value region, `llama_like/forward/mod.rs:1638` states
    // this dispatch inside `dsl::guarded_value`, and without `Either` on `o`
    // the lowerer hands the guard's `Val` to nobody. `lse` was never in a
    // region-hosted statement that declined it -- the texts that want it
    // declare it, and the texts that do not have no second buffer at all.
    //
    // # Why `Env`, and why no key
    //
    // Every arm of the one-result spelling passed `AttnCtx::lse_out_d`
    // (`arms/fa2.rs`'s `lse_slab`): the per-fire scratch slab
    // `fire/launch.rs:3417` publishes and `:3087` slices per split. A
    // one-result statement has no `arg_out(1)` to prefer over it, so the
    // address comes from the FIRE and `Env` is the word for that.
    //
    // NO `keys::` FACT IS MINTED FOR IT. `kernels/src/keys.rs` §1 has none for
    // `lse_out_d`, and adding one would be a name for a deficiency rather
    // than a fix (F7) -- the slab is not a published plane, it is this
    // family's scratch, and `score_out` on the custom launcher is spelled
    // bare for the same reason. `int_buffer` and `float_buffer` USED TO BE
    // ON THAT LIST AND ARE NOT ANY MORE: they were never a deficiency, only
    // an unnamed answer, and the block below is where that is argued.
    //
    // # The witness argument, kept, because it is what the split rests on
    //
    // `attn::dispatch_attention_flashinfer_decode` and
    // `attn::attention_flashinfer_prefill` are the only two of the six with a
    // two-result builder (`base.rs:373` and `:514`) placing the F32
    // `[Tokens, q_heads]` LSE second, and they are exactly the two that got a
    // `_lse` twin. The other four were stated `Out(1)` BY ELIMINATION over
    // the provenance-eligible set -- a statement operand can only land on a
    // `Trace` or `Either` parameter, every other parameter here is `Env<_>`,
    // so `q`, `o` and `lse` were the whole set -- and elimination over an
    // empty set is what the four turned out to be: no text declares their
    // second result, so `Env` is not a demotion, it is the reading.
    //
    // AND THE ARM WAS THE THING THE MARK COULD NOT CHECK. That paragraph is
    // at `arms/fa2.rs`'s planless prefill: `Out(1)` was honoured by two arms
    // of six, and the one symbol whose second spelling really did want its
    // own buffer was writing to the slab. A mark on a parameter says where a
    // value would come FROM; it cannot say that anyone went and got it. Two
    // symbols can, because the arm that binds the `_lse` one has no fallback
    // to fall back to.
    lse: Env<keys::AttnLseOut>,
    // ── THE WORKSPACE, NAMED, AND WHY IT IS THE DECODE CARVE ──────────────
    //
    // `Env<*mut c_void>` was not a blocked fact, it was an UNNAMED one. The
    // type says *"the driver hands over something"*, a column cannot read
    // that, so an arm had to carry it -- while `Cx::attn_workspace`
    // (`bind/cx.rs:208`) had been answering the whole time.
    // `keys::AttnWorkspaceInt` / `keys::AttnWorkspaceFloat` (`keys.rs` §1)
    // are the words for what it answers, and `operand()` binds both off
    // `f.cx.attn_workspace()` (`bind/table.rs:1005-1022`).
    //
    // WHICH CARVE IS THE ONE THING TO GET RIGHT HERE. `AttnCtx` holds TWO:
    // `workspace` and `prefill_workspace` (`bind/mod.rs:1151` and `:1161`),
    // kept apart because *"a FlashInfer plan writes its schedule into the
    // workspace it was raised against, so a decode plan and a prefill plan
    // sharing one is one clobbering the other"*. `Facts::attn_workspace`
    // (`bind/facts.rs:305`) answers `AttnCtx::workspace` -- the DECODE one --
    // and its doc says it *"does not choose between them"*, so these two keys
    // mean the decode carve and nothing else.
    //
    // WHICH MAKES THE SPELLING A PER-LAUNCHER CHECK AND NOT A RENAME. TWO of
    // this file's eight may say it, and they are the planless prefill pair --
    // whose arm plans against `workspace` *"as the entry point it replaces
    // did"* (`arms/fa2.rs`'s planless body). The six unfolded launchers name
    // neither carve: `fa2_decode_leaves` and `fa2_prefill_leaves` resolve
    // against `Cx::attn_workspace` and `Cx::attn_prefill_workspace`
    // respectively and answer ADDRESSES, so no base reaches a signature and
    // no launcher can be handed the wrong one.
    // THE ONE AGGREGATE LEFT IN THIS SIGNATURE, AND THE SIGNATURE §4
    // SKETCHES FLAT. Twenty leaves, all `Copy`, all resolvable off `Cx`:
    // §4's nine (four int-workspace pointers plus the mask, two float,
    // `padded_batch_size`, `split_kv`) and the seven `make_decode_params`
    // reads that its sketch has no line for. Unfolded, this launcher takes 28
    // arguments and `impl_kernel_fn!` stops at 24. The [`ROUTINES`] ledger
    // carries the count per launcher, the empirical proof of the ceiling, and
    // the correction to §0's licence that came out of checking it.
    // §4'S PLAN, FLAT. `bind/table.rs`'s `fa2_decode_leaves` resolves each of
    // these off the cache the fire raised and `decode_plan_of_leaves` folds
    // them back into what `params.rs` consumes, so nothing here recomputes a
    // layout. THE WORKSPACE PAIR WENT WITH THE AGGREGATE: an address the
    // driver already offset needs no base, and `int_base_bytes` was only ever
    // part of that sum.
    request_indices: Env<keys::Fa2DecodeRequestIndices>,
    kv_tile_indices: Env<keys::Fa2DecodeKvTileIndices>,
    o_indptr: Env<keys::Fa2DecodeOIndptr>,
    kv_chunk_size: Env<keys::Fa2DecodeKvChunkSize>,
    block_valid_mask: Env<keys::Fa2DecodeBlockValidMask>,
    tmp_v: Env<keys::Fa2DecodeTmpV>,
    tmp_s: Env<keys::Fa2DecodeTmpS>,
    padded_batch: Env<keys::Fa2DecodePaddedBatch>,
    split_kv: Env<keys::Fa2DecodeSplitKv>,
    requests: Env<keys::Fa2DecodeRequests>,
    num_q_heads: Env<keys::Fa2DecodeNumQHeads>,
    num_kv_heads: Env<keys::Fa2DecodeNumKvHeads>,
    head_dim: Env<keys::Fa2DecodeHeadDim>,
    page_size: Env<keys::Fa2DecodePageSize>,
    hnd_layout: Env<keys::Fa2DecodeHndLayout>,
    full_attention: Env<keys::Fa2DecodeFullAttention>,
    window_left: Env<keys::WindowLeft>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    // THE SOFTMAX SCALE, SPELLED. All eight launchers in this file carry it,
    // this is the only place the argument is written, and it moved for the
    // same reason `kv_last_page_lens` did four parameters up: `Env<f32>`
    // derives `Shape::Env => None`, and `None` is the claim that NO source
    // names the value. One does. `keys::SmScale` = "attn.sm_scale"
    // (`keys.rs` §1) and `operand()` answers it off `f.sm_scale()`
    // (`bind/table.rs`), which is `AttnCtx::sm_scale` -- the exact field
    // every arm in `arms/fa2.rs` passed by hand as `Env(a.sm_scale)`.
    //
    // CHECKED PER ARM RATHER THAN BY NAME, as the paged four were: all six
    // hand arms read `a.sm_scale` off the same `AttnCtx` borrow they read
    // `a.logits_soft_cap` off, and `logits_soft_cap` STAYS `Env<f32>` one
    // line above because `keys.rs` has no fact for it and minting one would
    // be F7's *"a deficiency gets a fix, not a name"* -- the cap is a
    // per-fire dispatch input the binder holds no query for.
    //
    // `f32` AT THE KEY AND `f32` IN THE BODY, so the only edit inside is a
    // second star: `*sm_scale` became `**sm_scale` at each of the five
    // `make_*_params` calls, `Env` deref then `SmScale` deref. Arity is
    // untouched -- `fact!` forces `PROV = Env` and `TY = f32`.
    sm_scale: Env<keys::SmScale>,
    #[lit(false)]
    broadcast_q: Env<bool>,
) -> Result<(), Refusal> {
    let plan = decode_plan_of_leaves(
        **request_indices,
        **kv_tile_indices,
        **o_indptr,
        **kv_chunk_size,
        **block_valid_mask,
        **tmp_v,
        **tmp_s,
        **padded_batch,
        **split_kv,
        **requests,
        **num_q_heads,
        **num_kv_heads,
        **head_dim,
        **page_size,
        **hnd_layout,
        **full_attention,
    );
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        **kv_page_indices,
        **kv_page_indptr,
        **kv_last_page_lens,
        // Decode has one query row per request, so there is no QO indptr.
        core::ptr::null(),
        **lse,
        // The zero base `decode_plan_of_leaves` documents: every workspace
        // address is already in the leaves.
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    let arm = decode_arm(plan.full_attention_variant, **window_left, **logits_soft_cap);
    let (params, split) =
        make_decode_params(&plan, &bufs, **window_left, **logits_soft_cap, **sm_scale, *broadcast_q);
    decode(ctx, decode_at(&plan, arm, params.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// `attn::dispatch_attention_flashinfer_decode_lse` — the SAME kernel as
/// [`dispatch_attention_flashinfer_decode`] over a statement that declares
/// both of its results.
///
/// # D2, and the one symbol that really was two functions
///
/// `.wiki/kilimanjaro3.md` §3.8: *an optional argument means two functions*.
/// One string, `attn::dispatch_attention_flashinfer_decode`, was being
/// stated at two arities — `base.rs:259`'s `attention_flashinfer_decode`
/// declares one result (or none inside a value region, `attn_at` at
/// `base.rs:1121`) and `base.rs:373`'s `attention_flashinfer_decode_lse`
/// declares two. The rule fits because the two texts differ in ARITY and in
/// nothing else: the plan, the lattice point, the params block and the split
/// fold are all the same, which is why this is a forward and not a body.
///
/// **What it buys is that the ARM has no choice to make.** `arms/fa2.rs`'s
/// `lse_slab` fallback served one symbol for both texts by asking
/// `cx.arg_out(1)` and taking `AttnCtx::lse_out_d` when the answer was no —
/// and the same shape on the planless prefill wrote a declared result to the
/// fire's scratch slab for as long as nobody opened the arm (written up at
/// `arms/fa2.rs`'s `fa2_prefill_capture_arm`). Here both results are
/// `Out<N, _>` with no `Or`, so a text that forgot one is refused at LOAD by
/// `arity_problem`'s floor rather than bound to the wrong address at fire
/// time.
///
/// `depth_prefix_plan` on the row, exactly as its twin: gpt-oss states this
/// spelling (`gpt_oss/forward/mod.rs:188`, through `attention_for_lse`) and
/// took the union-tail plan swap before the split. A split that quietly
/// dropped it would lower a different plan for one family only.
///
/// # Errors
///
/// [`dispatch_attention_flashinfer_decode`]'s, unchanged — this adds no
/// refusal of its own, because a missing result is not reachable here.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn dispatch_attention_flashinfer_decode_lse(
    ctx: &Ctx,
    q: In<0, bf16>,
    // REQUIRED, where the twin's is `Out<0, bf16>`, and the difference is
    // the whole of the split. `Or`'s `Provenance::Either` is what
    // `accepts_an_unstated_result` (`model-ir/src/kernels.rs:227`) reads to
    // let a value-producing region hand its buffer to a statement declaring
    // nothing; the twin needs that because `attn_at` declines to name a
    // result inside `llama_like/forward/mod.rs:1638`'s `dsl::guarded_value`.
    // The `_lse` text declares two results unconditionally and is stated
    // outside any region (`gpt_oss/forward/mod.rs:188`), so the clause is
    // never reached and `Trace` is the honest provenance.
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // `Out(1)`, REQUIRED, and this is the parameter the split exists for. On
    // the twin it is `Env<*mut f32>` — the fire's scratch slab — because no
    // one-result text has a second buffer to point at. Here every text has
    // one: `attention_sink_rescale` reads it two statements downstream
    // (`gpt_oss/forward/mod.rs:189`).
    lse: Out<1, f32>,
    // The decode carve, keyed: this launcher's arm passes `a.workspace`. The
    // argument is at `dispatch_attention_flashinfer_decode`'s pair.
    // §4'S PLAN, FLAT. `bind/table.rs`'s `fa2_decode_leaves` resolves each of
    // these off the cache the fire raised and `decode_plan_of_leaves` folds
    // them back into what `params.rs` consumes, so nothing here recomputes a
    // layout. THE WORKSPACE PAIR WENT WITH THE AGGREGATE: an address the
    // driver already offset needs no base, and `int_base_bytes` was only ever
    // part of that sum.
    request_indices: Env<keys::Fa2DecodeRequestIndices>,
    kv_tile_indices: Env<keys::Fa2DecodeKvTileIndices>,
    o_indptr: Env<keys::Fa2DecodeOIndptr>,
    kv_chunk_size: Env<keys::Fa2DecodeKvChunkSize>,
    block_valid_mask: Env<keys::Fa2DecodeBlockValidMask>,
    tmp_v: Env<keys::Fa2DecodeTmpV>,
    tmp_s: Env<keys::Fa2DecodeTmpS>,
    padded_batch: Env<keys::Fa2DecodePaddedBatch>,
    split_kv: Env<keys::Fa2DecodeSplitKv>,
    requests: Env<keys::Fa2DecodeRequests>,
    num_q_heads: Env<keys::Fa2DecodeNumQHeads>,
    num_kv_heads: Env<keys::Fa2DecodeNumKvHeads>,
    head_dim: Env<keys::Fa2DecodeHeadDim>,
    page_size: Env<keys::Fa2DecodePageSize>,
    hnd_layout: Env<keys::Fa2DecodeHndLayout>,
    full_attention: Env<keys::Fa2DecodeFullAttention>,
    window_left: Env<keys::WindowLeft>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
    #[lit(false)]
    broadcast_q: Env<bool>,
) -> Result<(), Refusal> {
    dispatch_attention_flashinfer_decode(
        ctx,
        // `q` and the five paged facts FORWARD WHOLE — both ends spell them
        // identically, so the compiler checks the slot agrees
        // (`kernels-cuda/src/rope.rs:62`'s rule). `o` and `lse` are the two
        // that cannot. Both spellings say REQUIRED now: the arm the twin was
        // made for -- a statement declaring no result inside a value region
        // -- states its destination on the OP (`Op::dest`), so neither
        // signature has to be optional to serve it.
        q,
        o,
        k_pages,
        v_pages,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        <keys::AttnLseOut as keys::Fact>::env(lse.ptr),
        request_indices,
        kv_tile_indices,
        o_indptr,
        kv_chunk_size,
        block_valid_mask,
        tmp_v,
        tmp_s,
        padded_batch,
        split_kv,
        requests,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        hnd_layout,
        full_attention,
        window_left,
        logits_soft_cap,
        sm_scale,
        broadcast_q,
    )
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
#[kernels_macros::routine]
pub fn dispatch_attention_flashinfer_decode_capture(
    ctx: &Ctx,
    q: In<0, bf16>,
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // `Env`, as [`dispatch_attention_flashinfer_decode`]'s. The capture
    // dispatch has ONE spelling (`base.rs:1011`, `-> Option<Val>`), so no
    // text ever declared its second result and `Or`'s `Provenance::Either`
    // was widening a ceiling nothing reached.
    lse: Env<keys::AttnLseOut>,
    // The decode carve, keyed: this launcher's arm passes `a.workspace`. The
    // argument is at `dispatch_attention_flashinfer_decode`'s pair.
    // §4'S PLAN, FLAT. `bind/table.rs`'s `fa2_decode_leaves` resolves each of
    // these off the cache the fire raised and `decode_plan_of_leaves` folds
    // them back into what `params.rs` consumes, so nothing here recomputes a
    // layout. THE WORKSPACE PAIR WENT WITH THE AGGREGATE: an address the
    // driver already offset needs no base, and `int_base_bytes` was only ever
    // part of that sum.
    request_indices: Env<keys::Fa2DecodeRequestIndices>,
    kv_tile_indices: Env<keys::Fa2DecodeKvTileIndices>,
    o_indptr: Env<keys::Fa2DecodeOIndptr>,
    kv_chunk_size: Env<keys::Fa2DecodeKvChunkSize>,
    block_valid_mask: Env<keys::Fa2DecodeBlockValidMask>,
    tmp_v: Env<keys::Fa2DecodeTmpV>,
    tmp_s: Env<keys::Fa2DecodeTmpS>,
    padded_batch: Env<keys::Fa2DecodePaddedBatch>,
    split_kv: Env<keys::Fa2DecodeSplitKv>,
    requests: Env<keys::Fa2DecodeRequests>,
    num_q_heads: Env<keys::Fa2DecodeNumQHeads>,
    num_kv_heads: Env<keys::Fa2DecodeNumKvHeads>,
    head_dim: Env<keys::Fa2DecodeHeadDim>,
    page_size: Env<keys::Fa2DecodePageSize>,
    hnd_layout: Env<keys::Fa2DecodeHndLayout>,
    full_attention: Env<keys::Fa2DecodeFullAttention>,
    score_out: Env<*mut f32>,
    score_indptr: Env<*const i32>,
    window_left: Env<keys::WindowLeft>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
    #[lit(false)]
    broadcast_q: Env<bool>,
) -> Result<(), Refusal> {
    let plan = decode_plan_of_leaves(
        **request_indices,
        **kv_tile_indices,
        **o_indptr,
        **kv_chunk_size,
        **block_valid_mask,
        **tmp_v,
        **tmp_s,
        **padded_batch,
        **split_kv,
        **requests,
        **num_q_heads,
        **num_kv_heads,
        **head_dim,
        **page_size,
        **hnd_layout,
        **full_attention,
    );
    // `:546-549`, before the variant test, and in that order.
    if score_out.is_null() || score_indptr.is_null() {
        return Err(CAPTURE_SINK);
    }
    let Some(arm) = decode_capture_arm(plan.full_attention_variant, **window_left, **logits_soft_cap)
    else {
        return Err(CAPTURE_VARIANT);
    };
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        **kv_page_indices,
        **kv_page_indptr,
        **kv_last_page_lens,
        core::ptr::null(),
        **lse,
        // The zero base `decode_plan_of_leaves` documents: every workspace
        // address is already in the leaves.
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    let (base, split) = make_decode_params(&plan, &bufs, **window_left, 0.0, **sm_scale, *broadcast_q);
    let params = DecodeScoreParams {
        base,
        score_out: addr(*score_out),
        score_indptr: addr(*score_indptr),
    };
    decode(ctx, decode_at(&plan, arm, params.base.padded_batch_size), &params)?;
    if plan.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// One prefill plan's named leaves, back into the aggregate `params.rs`
/// consumes.
///
/// [`decode_plan_of_leaves`]' twin, and its doc is this one's: the offsets are
/// ADDRESSES here, against a zero base, so `make_prefill_params` writes back
/// exactly what `bind/table.rs`'s `fa2_prefill_leaves` resolved. A null leaf
/// carries `-1`.
///
/// `enable_cuda_graph` is not a leaf: the block mask is carved only under
/// `split_kv && enable_cuda_graph`, so a non-null mask says both. Neither is
/// `valid` nor `use_sm90` — both are refusals the driver makes before it
/// answers a single leaf, and [`prefill_plan_usable`] restates them for the
/// planless pair, which still hands over the aggregate.
#[allow(clippy::too_many_arguments)]
fn prefill_plan_of_leaves(
    request_indices: *const i32,
    qo_tile_indices: *const i32,
    kv_tile_indices: *const i32,
    merge_indptr: *const i32,
    o_indptr: *const i32,
    kv_chunk_size: *const i32,
    block_valid_mask: *const u8,
    tmp_v: *mut f32,
    tmp_s: *mut f32,
    padded_batch: i32,
    split_kv: bool,
    total_rows: i32,
    cta_tile_q: u32,
    requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    window_left: i32,
    hnd_layout: bool,
    full_attention: bool,
    causal_mask: bool,
) -> PrefillPlan {
    fn at<T>(p: *const T) -> i64 {
        if p.is_null() { -1 } else { p as usize as i64 }
    }
    PrefillPlan {
        info: PrefillPlanInfo {
            padded_batch_size: i64::from(padded_batch),
            total_num_rows: i64::from(total_rows),
            // The DEVICE row count, which this driver does not fill on either
            // side; `make_prefill_params` writes the field 0 and the fold
            // reads what the kernel used.
            total_num_rows_offset: -1,
            cta_tile_q: i64::from(cta_tile_q),
            request_indices_offset: at(request_indices),
            qo_tile_indices_offset: at(qo_tile_indices),
            kv_tile_indices_offset: at(kv_tile_indices),
            merge_indptr_offset: at(merge_indptr),
            o_indptr_offset: at(o_indptr),
            kv_chunk_size_ptr_offset: at(kv_chunk_size),
            v_offset: at(tmp_v.cast_const()),
            s_offset: at(tmp_s.cast_const()),
            block_valid_mask_offset: at(block_valid_mask),
            enable_cuda_graph: !block_valid_mask.is_null(),
            split_kv,
        },
        device: crate::attn::fa2::plan::fa_device(),
        num_requests: requests,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        cta_tile_q,
        window_left,
        hnd_layout,
        full_attention_variant: full_attention,
        causal_mask,
        use_sm90: false,
        valid: true,
    }
}

/// The paged prefill, over a plan and the buffers it addresses.
///
/// The body [`dispatch_attention_flashinfer_prefill_bf16`] had before its plan
/// was unfolded, kept as a function because [`attention_flashinfer_prefill`]
/// forwards into it and the planless pair still takes the aggregate. Its
/// `bufs` carry the workspace bases the offsets are relative to, which on the
/// unfolded path are null and on the planless path are the decode carve.
fn prefill_paged(
    ctx: &Ctx,
    bufs: &Buffers,
    plan: &PrefillPlan,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> Result<(), Refusal> {
    prefill_plan_usable(plan)?;
    let arm = prefill_arm(plan.full_attention_variant, plan.causal_mask, logits_soft_cap);
    let (params, split) = make_prefill_params(plan, bufs, logits_soft_cap, sm_scale);
    prefill(ctx, prefill_at(plan, arm, params.padded_batch_size), &params)?;
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
#[kernels_macros::routine]
pub fn dispatch_attention_flashinfer_prefill_bf16(
    ctx: &Ctx,
    q: In<0, bf16>,
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    // THE SIXTH PAGED ARRAY, AND THE LAST OF THEM TO GET A WORD. The five
    // around it were converted a stage ago; this one stayed `Env<*const u32>`
    // because `keys.rs` §1 had no fact for the QUERY CSR, only for the KV
    // ones. `keys::QoIndptr` = "plan.qo_indptr" closed that, and it is the
    // single most-read plan field in `driver-cuda/src/bind/arms/` -- thirteen
    // hand reads of `plan_of.qo_indptr` across the family, all of them the
    // same `AttnCtx::qo_indptr_d` that `Facts::plan` copies in
    // (`bind/facts.rs`).
    //
    // WHAT THE CONVERSION FIXES IS A FALSE COLUMN ENTRY, exactly as with
    // `kv_last_page_lens` above: `None` said no source named it while every
    // arm bound it. It reads `Source::Named("plan.qo_indptr"), stated: true`
    // now, and `**qo_indptr` in the bodies is the one mechanical consequence.
    //
    // IT DOES NOT UNBLOCK ANY ROW IN THIS FILE, and that is worth stating so
    // the next reader does not go looking: all six FA2 arms are held by the
    // boxed `DecodePlanCache`/`PrefillPlanCache` a `Cx` may not hand over
    // (`bind/mod.rs:1638` -- `Route::Driver` for exactly that reason), so a
    // complete column changes nothing about who binds them. The column is
    // still worth completing: it is what the row would derive the day the
    // plan cache stops being a driver-owned mutable.
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // `Env`, as [`dispatch_attention_flashinfer_decode`]'s. One spelling
    // (`base.rs:274`), one declared result; the LSE has only ever come from
    // the fire's slab.
    lse: Env<keys::AttnLseOut>,
    // ── THE PLAN, LEAF BY LEAF, AND THE BASES THAT WENT WITH IT ──────────
    //
    // `int_buffer`/`float_buffer` stood here as the file's last bare
    // `Env<*mut c_void>` pair, pinned `is_none()` because THIS PARAMETER TOOK
    // BOTH CARVES: `fa2_prefill_arm` reached it with `a.prefill_workspace`
    // and [`attention_flashinfer_prefill`] forwarded `a.workspace`. The
    // unfold answers that by deleting the question -- a base with nothing to
    // offset -- and each caller resolves its own leaves against its own
    // carve. `bind/table.rs`'s `fa2_prefill_leaves` reads the prefill one.
    request_indices: Env<keys::Fa2PrefillRequestIndices>,
    qo_tile_indices: Env<keys::Fa2PrefillQoTileIndices>,
    kv_tile_indices: Env<keys::Fa2PrefillKvTileIndices>,
    merge_indptr: Env<keys::Fa2PrefillMergeIndptr>,
    o_indptr: Env<keys::Fa2PrefillOIndptr>,
    kv_chunk_size: Env<keys::Fa2PrefillKvChunkSize>,
    block_valid_mask: Env<keys::Fa2PrefillBlockValidMask>,
    tmp_v: Env<keys::Fa2PrefillTmpV>,
    tmp_s: Env<keys::Fa2PrefillTmpS>,
    padded_batch: Env<keys::Fa2PrefillPaddedBatch>,
    split_kv: Env<keys::Fa2PrefillSplitKv>,
    total_rows: Env<keys::Fa2PrefillTotalRows>,
    cta_tile_q: Env<keys::Fa2PrefillCtaTileQ>,
    requests: Env<keys::Fa2PrefillRequests>,
    num_q_heads: Env<keys::Fa2PrefillNumQHeads>,
    num_kv_heads: Env<keys::Fa2PrefillNumKvHeads>,
    head_dim: Env<keys::Fa2PrefillHeadDim>,
    page_size: Env<keys::Fa2PrefillPageSize>,
    window_left: Env<keys::Fa2PrefillWindowLeft>,
    hnd_layout: Env<keys::Fa2PrefillHndLayout>,
    full_attention: Env<keys::Fa2PrefillFullAttention>,
    causal_mask: Env<keys::Fa2PrefillCausalMask>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
) -> Result<(), Refusal> {
    let plan = prefill_plan_of_leaves(
        **request_indices,
        **qo_tile_indices,
        **kv_tile_indices,
        **merge_indptr,
        **o_indptr,
        **kv_chunk_size,
        **block_valid_mask,
        **tmp_v,
        **tmp_s,
        **padded_batch,
        **split_kv,
        **total_rows,
        **cta_tile_q,
        **requests,
        **num_q_heads,
        **num_kv_heads,
        **head_dim,
        **page_size,
        **window_left,
        **hnd_layout,
        **full_attention,
        **causal_mask,
    );
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        **kv_page_indices,
        **kv_page_indptr,
        **kv_last_page_lens,
        **qo_indptr,
        **lse,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    prefill_paged(ctx, &bufs, &plan, **logits_soft_cap, **sm_scale)
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
#[kernels_macros::routine]
pub fn dispatch_attention_flashinfer_prefill_capture_bf16(
    ctx: &Ctx,
    q: In<0, bf16>,
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // `Env`, as [`dispatch_attention_flashinfer_decode`]'s. One spelling
    // (`base.rs:1011`), one declared result.
    lse: Env<keys::AttnLseOut>,
    // THE PREFILL CARVE'S TWO KEYS STOOD HERE, and the unfold spends them:
    // `bind/table.rs`'s `fa2_prefill_leaves` reads
    // `Cx::attn_prefill_workspace` -- the same carve, one hop earlier -- and
    // hands over addresses, so a base parameter would have nothing to offset.
    // The arm still passes `a.prefill_workspace` to the upload, which is what
    // keeps the row armed.
    request_indices: Env<keys::Fa2PrefillRequestIndices>,
    qo_tile_indices: Env<keys::Fa2PrefillQoTileIndices>,
    kv_tile_indices: Env<keys::Fa2PrefillKvTileIndices>,
    merge_indptr: Env<keys::Fa2PrefillMergeIndptr>,
    o_indptr: Env<keys::Fa2PrefillOIndptr>,
    kv_chunk_size: Env<keys::Fa2PrefillKvChunkSize>,
    block_valid_mask: Env<keys::Fa2PrefillBlockValidMask>,
    tmp_v: Env<keys::Fa2PrefillTmpV>,
    tmp_s: Env<keys::Fa2PrefillTmpS>,
    padded_batch: Env<keys::Fa2PrefillPaddedBatch>,
    split_kv: Env<keys::Fa2PrefillSplitKv>,
    total_rows: Env<keys::Fa2PrefillTotalRows>,
    cta_tile_q: Env<keys::Fa2PrefillCtaTileQ>,
    requests: Env<keys::Fa2PrefillRequests>,
    num_q_heads: Env<keys::Fa2PrefillNumQHeads>,
    num_kv_heads: Env<keys::Fa2PrefillNumKvHeads>,
    head_dim: Env<keys::Fa2PrefillHeadDim>,
    page_size: Env<keys::Fa2PrefillPageSize>,
    // The PLAN's window, which this launcher reads twice: the capture arm
    // exists only at `-1`, and the block carries it to the kernel.
    window_left: Env<keys::Fa2PrefillWindowLeft>,
    hnd_layout: Env<keys::Fa2PrefillHndLayout>,
    full_attention: Env<keys::Fa2PrefillFullAttention>,
    causal_mask: Env<keys::Fa2PrefillCausalMask>,
    score_out: Env<*mut f32>,
    score_indptr: Env<*const i32>,
    score_window: Env<u32>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
) -> Result<(), Refusal> {
    let plan = prefill_plan_of_leaves(
        **request_indices,
        **qo_tile_indices,
        **kv_tile_indices,
        **merge_indptr,
        **o_indptr,
        **kv_chunk_size,
        **block_valid_mask,
        **tmp_v,
        **tmp_s,
        **padded_batch,
        **split_kv,
        **total_rows,
        **cta_tile_q,
        **requests,
        **num_q_heads,
        **num_kv_heads,
        **head_dim,
        **page_size,
        **window_left,
        **hnd_layout,
        **full_attention,
        **causal_mask,
    );
    // `:849-856`. The sink first, then the variant, then the plan -- the C++'s
    // order, and the window is part of the sink here rather than of the
    // variant.
    if score_out.is_null() || score_indptr.is_null() || *score_window == 0 {
        return Err(CAPTURE_SINK);
    }
    let Some(arm) = prefill_capture_arm(plan.causal_mask, plan.window_left, **logits_soft_cap) else {
        return Err(CAPTURE_VARIANT);
    };
    prefill_plan_usable(&plan)?;
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        **kv_page_indices,
        **kv_page_indptr,
        **kv_last_page_lens,
        **qo_indptr,
        **lse,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    let (base, split) = make_prefill_params(&plan, &bufs, 0.0, **sm_scale);
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
#[kernels_macros::routine]
pub fn dispatch_attention_flashinfer_prefill_custom(
    ctx: &Ctx,
    q: In<0, bf16>,
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // `Env`, as [`dispatch_attention_flashinfer_decode`]'s. One spelling
    // (`base.rs:1051`), one declared result.
    lse: Env<keys::AttnLseOut>,
    // The prefill carve's two keys stood here; see
    // `dispatch_attention_flashinfer_prefill_capture_bf16` for where they went.
    request_indices: Env<keys::Fa2PrefillRequestIndices>,
    qo_tile_indices: Env<keys::Fa2PrefillQoTileIndices>,
    kv_tile_indices: Env<keys::Fa2PrefillKvTileIndices>,
    merge_indptr: Env<keys::Fa2PrefillMergeIndptr>,
    o_indptr: Env<keys::Fa2PrefillOIndptr>,
    kv_chunk_size: Env<keys::Fa2PrefillKvChunkSize>,
    block_valid_mask: Env<keys::Fa2PrefillBlockValidMask>,
    tmp_v: Env<keys::Fa2PrefillTmpV>,
    tmp_s: Env<keys::Fa2PrefillTmpS>,
    padded_batch: Env<keys::Fa2PrefillPaddedBatch>,
    split_kv: Env<keys::Fa2PrefillSplitKv>,
    total_rows: Env<keys::Fa2PrefillTotalRows>,
    cta_tile_q: Env<keys::Fa2PrefillCtaTileQ>,
    requests: Env<keys::Fa2PrefillRequests>,
    num_q_heads: Env<keys::Fa2PrefillNumQHeads>,
    num_kv_heads: Env<keys::Fa2PrefillNumKvHeads>,
    head_dim: Env<keys::Fa2PrefillHeadDim>,
    page_size: Env<keys::Fa2PrefillPageSize>,
    // Taken and then OVERWRITTEN with `-1` in the block below: the mask is
    // the causality here, and a plan planned under a window still names the
    // work list this launcher walks.
    window_left: Env<keys::Fa2PrefillWindowLeft>,
    hnd_layout: Env<keys::Fa2PrefillHndLayout>,
    full_attention: Env<keys::Fa2PrefillFullAttention>,
    causal_mask: Env<keys::Fa2PrefillCausalMask>,
    mask: Env<*const u8>,
    mask_indptr: Env<*const i32>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
) -> Result<(), Refusal> {
    let plan = prefill_plan_of_leaves(
        **request_indices,
        **qo_tile_indices,
        **kv_tile_indices,
        **merge_indptr,
        **o_indptr,
        **kv_chunk_size,
        **block_valid_mask,
        **tmp_v,
        **tmp_s,
        **padded_batch,
        **split_kv,
        **total_rows,
        **cta_tile_q,
        **requests,
        **num_q_heads,
        **num_kv_heads,
        **head_dim,
        **page_size,
        **window_left,
        **hnd_layout,
        **full_attention,
        **causal_mask,
    );
    let arm = prefill_custom_arm(**logits_soft_cap);
    prefill_plan_usable(&plan)?;
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        **kv_page_indices,
        **kv_page_indptr,
        **kv_last_page_lens,
        **qo_indptr,
        **lse,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    let (mut params, split) = make_prefill_params(&plan, &bufs, **logits_soft_cap, **sm_scale);
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
/// same [`PrefillPlan`] the planned three no longer take, which is why this
/// body is [`prefill_paged`]'s caller and not a second one.
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
#[kernels_macros::routine]
pub fn attention_flashinfer_prefill(
    ctx: &Ctx,
    q: In<0, bf16>,
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // `Env`, as [`dispatch_attention_flashinfer_decode`]'s -- and this is
    // the symbol whose SECOND spelling was the defect written up at
    // `arms/fa2.rs`'s planless prefill. `attention_flashinfer_prefill_lse`
    // (`base.rs:514`) names [`attention_flashinfer_prefill_lse`] now, so the
    // text that declares an LSE and the text that does not no longer share
    // an arm that has to guess which one it got.
    lse: Env<keys::AttnLseOut>,
    // The decode carve, keyed, on a PREFILL launcher on purpose: the planless
    // arm plans and uploads against `a.workspace` and not
    // `a.prefill_workspace`, *"as the entry point it replaces did"*. These
    // two are the BASES the aggregate's offsets are relative to, which is why
    // this pair survived the unfold that deleted every other workspace
    // parameter in this file: the plan below is a plan and not a run of
    // resolved addresses.
    int_buffer: Env<keys::AttnWorkspaceInt>,
    float_buffer: Env<keys::AttnWorkspaceFloat>,
    // STILL THE AGGREGATE, AND IT CANNOT BECOME LEAVES HERE. The arm builds
    // this plan from the host CSR mirrors on the way in; nothing published it
    // before the fire, so `operand()` has nothing to read and there is no
    // column to answer. `.wiki/kilimanjaro3.md` §3.3 keeps `Cx` query-only,
    // so a binder that planned would be the other half of the same mistake.
    plan: Env<PrefillPlan>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
) -> Result<(), Refusal> {
    // NOT A FORWARD ANY MORE, and the carve argument that stood here is what
    // the change spends. The planned launcher took two workspace bases and a
    // plan, and one parameter carried two carves depending on which caller
    // reached it; it now takes resolved leaves, so this body calls the shared
    // paged prefill directly with ITS carve and its own aggregate.
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        **kv_page_indices,
        **kv_page_indptr,
        **kv_last_page_lens,
        **qo_indptr,
        **lse,
        **int_buffer,
        **float_buffer,
    );
    prefill_paged(ctx, &bufs, &plan.into_inner(), **logits_soft_cap, **sm_scale)
}

/// `attn::attention_flashinfer_prefill_lse` — the planless prefill over a
/// statement that declares both of its results.
///
/// [`dispatch_attention_flashinfer_decode_lse`]'s argument, on the other
/// symbol that carried two arities. `attention_flashinfer_prefill_planless`
/// (`base.rs:294`, through `attn_at`) declares one result and
/// `attention_flashinfer_prefill_lse` (`base.rs:514`) declares two, and both
/// named `attn::attention_flashinfer_prefill`.
///
/// **This one had already cost something.** `arms/fa2.rs`'s planless arm
/// passed `AttnCtx::lse_out_d` without consulting `arg_out(1)`, so
/// `model/src/deepseek_v4/forward/mod.rs:146` — which traces the two-result
/// spelling and feeds the LSE to `lse_log2_to_ln`, `combine_attn_outputs` and
/// `attn_sink_correction` — was reading a buffer the launch never wrote. That
/// was repaired in the arm; this is the repair made STRUCTURAL, because the
/// arm bound to this symbol has no fallback available to it.
///
/// `whole = true`, as its twin: no plan cache crosses the fire for either
/// spelling, so the arm plans over the whole batch on the way in.
///
/// # Errors
///
/// [`dispatch_attention_flashinfer_prefill_bf16`]'s.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn attention_flashinfer_prefill_lse(
    ctx: &Ctx,
    q: In<0, bf16>,
    // REQUIRED, where [`attention_flashinfer_prefill`]'s is
    // `Out<0, bf16>`. Same reason as the decode pair's: the one-result
    // spelling goes through `attn_at`, which declares nothing inside
    // `llama_like/forward/mod.rs:1638`'s `dsl::guarded_value` and needs
    // `Provenance::Either` for `model-compiler/src/lower/walk.rs:472` to hand
    // it the region's buffer. `attention_flashinfer_prefill_lse` declares two
    // results unconditionally and `deepseek_v4/forward/mod.rs:146` states it
    // outside any region.
    o: Out<0, bf16>,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    lse: Out<1, f32>,
    // The decode carve, keyed, as [`attention_flashinfer_prefill`]'s: one
    // arm, one `Form` match, one `a.workspace`.
    int_buffer: Env<keys::AttnWorkspaceInt>,
    float_buffer: Env<keys::AttnWorkspaceFloat>,
    plan: Env<PrefillPlan>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    sm_scale: Env<keys::SmScale>,
) -> Result<(), Refusal> {
    attention_flashinfer_prefill(
        ctx,
        q,
        o,
        k_pages,
        v_pages,
        qo_indptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        <keys::AttnLseOut as keys::Fact>::env(lse.ptr),
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
    /// The prefill half of what `UNPLANNED_DECODE` was.
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
///
/// # The derived column, and why it will not cross
///
/// All six now carry one. `.wiki/kilimanjaro.md` §8 puts fa2 LAST and this is
/// what last looks like: the column is a LEDGER here, not a binding path, and
/// §8 said D6 would ship *"deriving `operands` and reading it nowhere"* — for
/// this family that is the finished state rather than a stage.
///
/// What it derives is real. `q` is `In(0)`, `o` and `lse` are `Out(0)` and
/// `Out(1)` and both are nullable — which the `Or<*mut _>` wrappers were
/// already saying and nothing was reading. `k_pages` and `v_pages` reach
/// `KvKeys`/`KvValues` through two aliases added to `kernels-macros`'
/// `fact_of` for exactly these two parameters, and `kv_page_indices` and
/// `kv_page_indptr` were already spelled canonically.
///
/// # `q` STATES `In(0)` NOW, AND `o` AND `lse` STILL COUNT
///
/// `.wiki/kilimanjaro2.md` E2 — *"counting is the last derivation that
/// guesses"* — reaches a family with no shape parameters at all, so what it
/// buys here is exactly one thing and it is worth naming precisely: the
/// index stops depending on where the parameter SITS. Position derived
/// `In(0)` for `q` because it is the only `*const bf16` in a list whose
/// other pointers are all `Env` or `Or`, and `classify` reads an `Env<*mut
/// bf16>` as `Shape::Env` rather than as a pointer — so the count was right
/// and was right for a reason no reader could see from the signature. All
/// six arms read `cx.arg_in(0)`, so the token states what the arms already
/// agreed on.
///
/// **`Or<T>` IS NOT A REGION AND `o` IS NOT WRAPPED.** `Or<Out<0, *mut
/// bf16>>` would compile and would state nothing: `stated_source` matches on
/// the LAST path segment of a parameter's type (`kernels-macros/src/lib.rs:335`),
/// which is `Or`, and `Or` states no slot — so the nesting would read as a
/// claim while the index went on being counted underneath it. The provenance
/// is the reason the two cannot simply be merged: `o` is `Provenance::Either`
/// because this same symbol serves a text that declares the result (gemma-4)
/// and one that does not (llama_like), where `Out<0, _>` is
/// `Provenance::Trace` and would make the result required. That is
/// `routine.rs:110`'s third case, not a shade of the other two.
///
/// # D2 SPLIT THE TWELVE INTO SIX `Or`s, SIX `Env`s AND FOUR NEW SLOTS
///
/// **Read this before the section below, which is the state it amends.**
/// Kilimanjaro III Stage 6 deletes `Or<T>` from operand position, and the
/// audit that did it found the twelve marks were carrying two different
/// facts under one spelling.
///
/// `Or<T>` plants one const: `Provenance::Either` (`kernels/src/routine.rs:522`).
/// On a `Binds::Writes` parameter that is read in two places, and this family
/// had one parameter in each:
///
/// * **`o` — `accepts_an_unstated_result`** (`model-ir/src/kernels.rs:227`),
///   which `model-compiler/src/lower/walk.rs:472` consults before handing a
///   value-producing region's buffer to a statement that declares no result.
///   `attn_at` (`base.rs:1121`) declares NONE inside a value region and
///   `llama_like/forward/mod.rs:1638` states five of these six symbols inside
///   `dsl::guarded_value`. Take `Or` off `o` and 56 guard regions produce a
///   value nobody writes — the defect `model/tests/arena_soundness.rs`'s
///   `UNWRITTEN` records 54 instances of. **All six `o`s keep their `Or`,
///   and that is not a deferral, it is the answer.**
/// * **`lse` — `arity_problem`'s ceiling** (`:303`) and nothing else. It was
///   `Either` because ONE STRING WAS TWO ARITIES:
///   `attention_flashinfer_decode` declares one result and
///   `attention_flashinfer_decode_lse` (`base.rs:373`) declares two, and
///   `attention_flashinfer_prefill_planless` and
///   `attention_flashinfer_prefill_lse` (`:514`) do the same on the planless
///   symbol. D2 (`.wiki/kilimanjaro3.md` §3.8) — *an optional argument means
///   two functions* — is the rule for exactly that, because the two texts
///   differ in ARITY and not in body.
///
/// So the two two-arity strings became four:
/// [`dispatch_attention_flashinfer_decode_lse`] and
/// [`attention_flashinfer_prefill_lse`] take `o: Out<0, bf16>` and
/// `lse: Out<1, f32>` REQUIRED and forward to their twins, and all six
/// original `lse`s became `Env<*mut f32>` — the fire's `lse_out_d` scratch
/// slab, which is what every one-result arm was passing anyway.
///
/// ## The arithmetic after the split
///
/// The six originals now give `writes = 0, opt_writes = 1` (`o` alone), a
/// band of `[0, 1]`, and every one-result-or-none text fits it. The four
/// texts that declare two results now name symbols with `writes = 2,
/// opt_writes = 0` and a band of `[2, 2]` — a floor, where before there was
/// none, so a text that stopped declaring its LSE is refused at LOAD instead
/// of silently writing it to the fire's slab.
///
/// **The ceiling FELL from 2 to 1 on the six, and that is the point.** A band
/// that admitted both arities is what let `arms/fa2.rs`'s planless prefill
/// write deepseek-v4's declared LSE into scratch for as long as nobody opened
/// the arm: every count agreed, and no count has ever been able to see an
/// address.
///
/// `STATED_SLOTS` does not move. Six `Out(1)` marks left and six arrived —
/// three on each new symbol, less the `q`/`o` pair each already had — so
/// `bind/table.rs`'s census reads the same number for a different reason,
/// which is the sort of coincidence worth writing down rather than
/// discovering.
///
/// # THE TWELVE WERE `OutSlot<N, Or<T>>`, AND GETTING THERE TOOK A REPAIR
///
/// kilimanjaro2 Stage 6 deletes `#[source(...)]`, so every slot attribute in
/// the tree owes a type-level spelling, and `819fdc34b` built the pair these
/// twelve were waiting for: [`kernels::routine::OutSlot`] states result `N`
/// and claims NO rectangle, which is exactly the objection the section above
/// makes to `Out<0, _>`. The conversion still could not be made when it
/// landed, and the section below is the argument that found out why. It is
/// kept because it is the reasoning that produced `148ca6a6a`, and it is
/// marked as the state BEFORE that commit so nobody reads it as live.
///
/// ## What was wrong, and what fixed it
///
/// Three of the four things a conversion has to preserve survived
/// `Out<0, bf16>` from the day the type existed:
/// `stated_source` reads the last path segment and sees `OutSlot`, so the
/// index states correctly; `classify` peels `OutSlot` and then `Or` and
/// keeps `nullable: true` (`kernels-macros/src/lib.rs:287-296`); and the
/// runtime `Or` value is still inside the wrapper, so the arm's fallback is
/// untouched. The fourth was PROVENANCE, and it is the one this family is
/// about. `OutSlot`'s `Arg` impl declared `TY` and `SPELLING` and **omitted
/// `PROV`**, taking the trait default of `Provenance::Trace` — so wrapping
/// an `Or` did precisely what `Out<0, _>` was rejected for above: it made
/// the result REQUIRED.
///
/// **`148ca6a6a` made all seven position wrappers forward `T::PROV`**
/// (`routine.rs:824-902`). `Or<T>` keeps its `Either` through the wrap and
/// the twelve are converted.
///
/// ## The arithmetic, which is the part worth keeping
///
/// `arity_problem` (`model-ir/src/kernels.rs:271`) ends on
/// `if declared < writes || declared > writes + opt_writes`, and
/// `ARITY_EXCEPTIONS` is `&[]` — nothing is spared. These six give
/// `writes = 0, opt_writes = 2`, so the band is `[0, 2]` and every statement
/// this family has fits it. Converted WITHOUT forwarding they would have
/// given `writes = 2, opt_writes = 0` and a band of `[2, 2]`, and `attn_at`
/// (`model-dsl/src/cuda/base.rs:1138`) — the builder behind
/// `attention_flashinfer_decode` and the plan-free prefill — declares
/// `shape.map(...)`, ONE result, or ZERO inside a value region. Both would
/// have failed `declared < writes` on the ordinary decode path. Only the two
/// `_lse` builders (`base.rs:373`, `:514`) declare two and would have
/// survived.
///
/// **WITH forwarding the band does not move at all**, which is a stronger
/// result than the one the repair was argued for. The general safety
/// argument is that forwarding shifts a nullable out of `writes` and into
/// `opt_writes`, dropping the lower bound while the upper bound is a SUM and
/// stays put, so an admitted band can only widen. These twelve are the
/// degenerate case of it: they were `Or<_>` and therefore already `Either`
/// before the conversion, and `OutSlot<N, Or<_>>` forwards that same
/// `Either`, so `writes` and `opt_writes` are UNCHANGED and the band is the
/// same `[0, 2]` it was. Nothing about what loads moves.
///
/// `arity_problem`'s own doc names this family as the reason the mechanism
/// exists: *"a [`Provenance::Either`] argument is one the statement MAY
/// place — FA2's `o`, which is the trace's result on a text that names one
/// and the guard's arena landing on a text that does not"*.
///
/// ## Nothing renumbered, and the check is cheap to state
///
/// `stated_source` SETS `*outs = N + 1` where the attribute left the counter
/// alone (`gemm` found this on `gemv_bf16`'s `bias`, which moved
/// `In(0)` -> `In(1)`), so a conversion can move a later BARE pointer. **Not
/// one of these six signatures has a bare pointer.** Every non-wrapped
/// pointer in all six is `Env<_>`, and `derive_all` sends `Shape::Env` to
/// `fact_of(&name)` without touching either counter
/// (`kernels-macros/src/lib.rs:238`). The counter now reaches 2 and is read
/// by nothing. No `<FN>_DERIVED` entry moves; `stated` was already `true` on
/// all twelve, because an attribute states just as loudly as a wrapper.
///
/// ## One thing the conversion buys beyond retiring an attribute
///
/// [`attention_flashinfer_prefill`] forwards `o` and `lse` WHOLE into the
/// shared [`prefill_paged`] body, the way it already forwarded `q`. That
/// forward used to pass two `Or<_>`s that agreed about nothing but their
/// pointee type; now both ends spell the slot, so the compiler checks that
/// the caller's result 0 is the callee's result 0. The comment at `q` quotes
/// rope's rule — *"forward the whole region where the slot agrees; build one
/// where the caller is inventing the operand"* — and it now covers all three
/// operands rather than one.
///
/// # ALL TWELVE SLOTS ARE STATED, AND THAT IS THE `alias()` GATE MOVING
///
/// Every `In(_)`/`Out(_)` these six derive is now written down: `q` by its
/// `In<0, _>` region and `o`/`lse` by their `OutSlot<0, _>`/`OutSlot<1, _>`
/// wrappers. The counter reaches nothing here any more, so all six leave
/// `bind/table.rs`'s `COUNTED_SLOT_SYMBOLS` list and twelve entries join
/// `STATED_SLOTS`.
///
/// **No derived value changed and no binding could have.** The marks
/// state exactly what the counter had already reached -- `o` was `Out(0)`
/// and `lse` was `Out(1)` before, because they are the only two parameters
/// the counter sees -- and `alias()`, the correction `Derived::stated`
/// switches off, opens `let Source::Slot(Kind::In, n) = d.source else { return *d };`
/// (`bind/table.rs:1202`). It has never touched an `Out`. So the risk this
/// gate exists to manage -- *"turning off the correction AND supplying the
/// wrong answer"* -- is not reachable from an `Out`-only conversion. A family
/// whose counted slots were `In`s would owe a harder argument than this one.
///
/// The evidence for the indices themselves is at the two parameters, and it
/// is uneven on purpose: four of the six are witnessed by a statement builder
/// or an arm that reads `arg_out(1)`, and the two capture/custom prefills are
/// argued BY ELIMINATION over the provenance-eligible parameters. That
/// argument is written out at `lse` so it can be disagreed with.
///
/// The other half of E1 does not apply to this family, and the absence is a
/// finding rather than an omission: **not one of these six takes a shape
/// parameter.** There is no `#[source(Rows)]`, no `#[source(OutWidth(0))]`
/// to delete, because a FlashInfer dispatch is told its rectangle by the
/// PLAN — `DecodePlan`/`PrefillPlan` carry `padded_batch_size`,
/// `num_qo_heads` and the CSR the kernel walks — and a plan is an `Env`
/// aggregate. So `q.rows` and `q.width` arrive and go unread, which costs
/// two `i32`s in a `Copy` value and is the price of the address and the
/// extent travelling together everywhere else.
///
/// A DRAFT OF THAT SENTENCE ALSO LISTED *"no `Width<slot::_>`"*, and the
/// notation it named no longer exists: `b57cf25ea` deleted
/// `kernels::routine::Rows`, `Width<S>` and `mod slot` outright, so there is
/// no wrapper spelling left for a shape parameter to have. `Source::Named(<keys::Rows as keys::Fact>::KEY)`,
/// `Source::InWidth` and `Source::OutWidth` are untouched — the variants
/// remain, only the way of writing them into a signature went — which is why
/// the two `#[source(...)]` spellings above are the ones worth naming. The
/// six launchers are unaffected either way; they never had one.
///
/// What blocks it is not a vocabulary gap and cannot be closed by marking:
///
/// * `bind/table.rs`'s `operand()` answers NO `Kv*` variant. Not one. The
///   pool's geometry is the driver's own allocation and `Facts` carries none
///   of it, so the four pointers that DO derive correctly resolve to nothing.
/// * `plan` is `Env<DecodePlan>`, a by-value aggregate. `Facts` is `Copy` and
///   small on purpose; a plan is neither.
/// * `int_buffer` and `float_buffer` are FlashInfer workspace addresses
///   carved by `attention_workspace.rs`. THIS BULLET IS RETIRED. *"Not
///   operands of any statement"* is still true and was never the question: a
///   fact does not have to be stated to be answered. `Cx::attn_workspace`
///   answers the decode carve and `Cx::attn_prefill_workspace` the other, and
///   the six unfolded launchers name neither -- both carves are resolved
///   inside `fa2_decode_leaves`/`fa2_prefill_leaves`, which fold
///   `int_base_bytes` in and hand out addresses. The planless pair still asks
///   for the DECODE carve by name, because that is the one its arm plans and
///   uploads into.
///
/// So the six one-line upload arms in `bind/arms/fa2.rs` stay. They are not
/// technical debt and this column is not a step towards deleting them: it is
/// the first machine-checked statement of what a fa2 statement carries, sat
/// next to a hand arm that binds what it cannot.
///
/// # THE FOUR PAGED FACTS ARE SPELLED IN TYPES, AND THIS FAMILY IS 24 OF 54
///
/// Stage 5's whole remaining gate was four parameter names —
/// `k_pages`/`v_pages`/`kv_page_indices`/`kv_page_indptr`, 54 sites, every
/// one of them in `attn/` — and this file held 24 of them, four on each of
/// the six launchers. All 24 now read `Env<keys::KvKeys>`,
/// `Env<keys::KvValues>`, `Env<keys::KvPageIndices>` and
/// `Env<keys::KvPageIndptr>`. The argument is at
/// [`dispatch_attention_flashinfer_decode`]'s parameters; what belongs here
/// is what the conversion moved.
///
/// ## Nothing derived moved, and that is the claim to be suspicious of
///
/// `fact_of` already answered these four names with these four sources
/// (`kernels-macros/src/lib.rs:640-643`), so every entry of every column is
/// byte-identical before and after. A refactor whose success condition is
/// "no observable change" is one that cannot be tested, which is why the
/// per-parameter check against `bind/arms/fa2.rs` was the work and the
/// substitution was not: the failure this conversion can produce is not a
/// wrong `Source`, it is a RIGHT source on a parameter that was never a fact.
/// `s3-norm` found two of those among nineteen `positions`.
///
/// ## Arity is untouched, by the same rule that made the last round safe
///
/// `arity_problem` matches `(_, Provenance::Env) => {}` first
/// (`model-ir/src/kernels.rs:271-320`), and `fact!` sets
/// `PROV = Provenance::Env` unconditionally (`keys.rs:216`), so these
/// parameters were invisible to the read/write counts before and are
/// invisible now. Neither `reads`, `opt_reads`, `writes` nor `opt_writes`
/// moves for any of the six. This is the opposite situation from the
/// `OutSlot` round below, where the provenance was the whole question.
///
/// ## `Ty` changed on eight of them and nothing reads it
///
/// `k_pages`/`v_pages` declared `Ty::PtrBf16` and now declare `*mut u8`'s.
/// That is a real change to `args` and it is inert for a reason worth
/// naming rather than assuming: `arity_problem` discards an `Env` before it
/// reads a `Ty`, `abi_admits` (`bind/table.rs`) is reached only for a
/// parameter the binder BINDS, and `operand()` answers no `Kv*` variant at
/// all — the bullet list above says so. So the declared type here is
/// documentation, which is exactly the condition under which `KvPageIndices`
/// spent months declaring `i32` for a `u32` table. A key nobody can reach is
/// a key nobody proof-reads.
///
/// ## The bodies cast back to bf16, and that is not the key being wrong
///
/// `keys::KvKeys` is `*mut u8` because the cache's element type varies by
/// model (`keys.rs:483-505`). FA2 exists at one width — a quantised layer is
/// widened into `k_bf16_pages` by the arm's prelude before a page reaches
/// here — so `buffers` keeps `*mut bf16` and the five call sites cast. Ten
/// casts, all `*mut u8` -> `*mut bf16`, none of them touching `const`.
///
/// ## `kv_last_page_lens` IS the fifth array, and it converted one pass later
///
/// **STATE BEFORE `keys::KvLastPageLens` EXISTED, kept because it is the
/// reasoning that produced the key.** All six now read
/// `Env<keys::KvLastPageLens>` and their arms bind
/// `keys::KvLastPageLens::env(plan_of.kv_last_page_lens)`; `xqa.rs`'s
/// attribute is retired. Arity is untouched, because `fact!` forces
/// `PROV = Env` on every key and `Env<*const u32>` already was one -- the
/// `(Ty, Provenance)` pair `impl_kernel_fn!` records is bit-identical. What
/// moved is the column: `source: None, stated: false` became
/// `Source::Named(<keys::KvLastPageLens as keys::Fact>::KEY), stated: true`, on six pointers
/// whose arms were already binding them from a field of that name.
///
/// It sits between the converted four in all six signatures. `fact_of` had
/// no arm for the name, so it derived `None`, and
/// `kernels-macros/src/lib.rs:691-695` explains that a non-unit variant is
/// not reachable from a name at all: the fact is `Attn("kv_last_page_lens_d")`
/// and `fact_of` builds one `Ident`.
///
/// **THE GAP IS IN THE NAME LOOKUP AND NOT IN `keys.rs`, WHICH MATTERS
/// BECAUSE THE NAME LOOKUP IS BEING DELETED.** `fact!` takes `$src:expr`
/// (`keys.rs:189`) and `WeightBias => Source::Named(<keys::WeightBias as keys::Fact>::KEY)` plus
/// three siblings already pass a `&'static str` through it
/// (`keys.rs:580-589`), so `KvLastPageLens = "kv.last_page_lens" =>
/// Source::Named(<keys::KvLastPageLens as keys::Fact>::KEY) => *const u32` was available whenever
/// someone wanted it. It is `keys.rs:615` now, and it retired these six and
/// `xqa.rs`'s `#[source(Attn("kv_last_page_lens_d"))]`.
///
/// **THE MIGRATION COPIED THE OLD MECHANISM'S LIMIT INTO THE NEW ONE'S
/// DOCUMENTATION.** Three places said the key could not be written because
/// the fact carries a `&'static str`. `fact!` never cared; the `Ident`
/// interpolation was `fact_of`'s, and `fact_of` is deleted. Written up at
/// the key, and the general form is worth more than this instance: a
/// migration inherits the constraints of the thing it replaces unless
/// somebody tests them against the replacement.
///
/// # Kilimanjaro §4, done on decode: the plan is sixteen named leaves
///
/// `.wiki/kilimanjaro.md` §4 sketches these eight with the plan UNFOLDED --
/// nine `Env<keys::X>` leaves (`request_indices`, `kv_tile_indices`,
/// `o_indptr`, `kv_chunk_size_ptr`, `block_valid_mask`, `tmp_v`, `tmp_s`,
/// `padded_batch_size`, `split_kv`) where `plan: Env<DecodePlan>` and its two
/// workspace pointers stood -- and licences it with §0. Both halves were
/// checked against the code rather than assumed: the licence is real in a
/// corrected form, the sketch is seven leaves short, and the ceiling that
/// blocked it has since risen to 36.
///
/// **The three DECODE launchers are unfolded.** Sixteen
/// `Env<keys::Fa2Decode*>` at 8..24, `plan`/`int_buffer`/`float_buffer` gone,
/// `bind/table.rs`'s `fa2_decode_leaves` resolving every one off the cache
/// and `decode_plan_of_leaves` folding them back into what `params.rs`
/// consumes. The five PREFILL launchers still take the aggregate; see the
/// bottom of this section for what is left there.
///
/// ## §0's licence is bucket-wide, not model-wide
///
/// §4 reads `plan/decode.rs:170` as *"with the flag on, `split_kv` is always
/// true, so `padded_batch_size` is `max_grid_size / gdy` -- head geometry
/// alone"*. `estimate` returns BEFORE that line whenever
/// `batch_size * gdy >= max_grid_size` (`plan/decode.rs:95-107`), with
/// `split_kv: false`; `plan_impl` then takes the other branch and
/// `padded_batch_size` is `req.batch_size`. Every int offset but the first is
/// `int_alloc.alloc(padded * 4, ..)` carved from it, so on that branch the
/// addresses move with the fire's batch.
///
/// Prefill is further from §0 and in the same direction. With the graph flag
/// on, `cta_tile_q` is `fa2_determine_cta_tile_q((total_num_rows -
/// batch_size + 1) * gqa, ..)`, `padded_batch_size` is
/// `max(max_batch_size_if_split, total_num_tiles_q)`
/// (`plan/prefill.rs:166-173`, `:252`), `o_indptr` is sized `batch_size + 1`
/// and `merge_indptr` `total_num_rows + 1` (`:328`, `:389`). Not one of those
/// is a constant of the model.
///
/// **What they are is constant per REPLAY BUCKET, which is the invariant the
/// change needs and the one §0 should have claimed.** `fire/launch.rs:194`
/// keys a capture on `BucketKey::new(requests, rows, class, model_id)`, and
/// every offset above is a function of exactly (requests, rows, model
/// geometry) -- so a replay never meets a plan carved at a different padding.
/// And the exposure does not move either way: `make_decode_params`
/// (`params.rs:632-644`) already resolves `base + offset` host-side and the
/// captured node already bakes the result, so putting the same arithmetic in
/// `operand()` swaps one host site for another that runs on the same fires.
///
/// ## The arity, measured, against a ceiling that then rose
///
/// Measured, not estimated: a temporary `routine!(probe_arity_25)` over a
/// 25-argument `fn` failed at `jit/mod.rs:82` with *"the trait bound
/// `..: KernelFn<..>` is not satisfied"*. **A `#[routine]` fn alone compiles
/// at any arity -- the wall is the `routine!` line**, which is where the
/// table entry needs the impl. `impl_kernel_fn!` is stamped through 36 now,
/// which clears the whole table below with one to spare.
///
/// The leaf count is taken from the CONSUMER, not from §4's sketch, because
/// the sketch is seven short on decode. `make_decode_params` (`params.rs:630`)
/// reads `info`'s ten fields and also `num_requests`, `num_q_heads`,
/// `num_kv_heads`, `head_dim`, `page_size`, `hnd_layout` and
/// `int_base_bytes`; `decode_arm` reads `full_attention_variant`; `decode_at`
/// reads `device`. Of those, `int_base_bytes` is absorbed into the resolved
/// pointer, `valid` and `enable_cuda_graph` become a driver-side refusal and
/// a null, and `device` is `ffa2::fa_device()` in the body -- the other seven
/// have to be parameters, on top of §4's nine. `make_prefill_params` adds
/// `qo_tile_indices`, `merge_indptr` and `total_num_rows`, and reads
/// `window_left` and `cta_tile_q` FROM THE PLAN (`params.rs:764`, `:2265`),
/// which §4 has no line for at all.
///
/// | launcher | was | unfolded |
/// |---|---|---|
/// | `dispatch_attention_flashinfer_decode` | 15 | **28** — done |
/// | `dispatch_attention_flashinfer_decode_lse` | 15 | **28** — done |
/// | `dispatch_attention_flashinfer_decode_capture` | 17 | **30** — done |
/// | `dispatch_attention_flashinfer_prefill_bf16` | 14 | **33** — done |
/// | `dispatch_attention_flashinfer_prefill_capture_bf16` | 17 | **36** — done |
/// | `dispatch_attention_flashinfer_prefill_custom` | 16 | **35** — done |
/// | `attention_flashinfer_prefill` | 14 | 14 — planless |
/// | `attention_flashinfer_prefill_lse` | 14 | 14 — planless |
///
/// The capture twin lands ON the ceiling. A thirty-seventh parameter on it
/// needs another `impl_kernel_fn!` row before it needs anything else.
///
/// ## What the planless pair is waiting on, which is nothing a key can give
///
/// Not the ceiling and not the licence -- both are settled. The blocker is
/// structural: [`attention_flashinfer_prefill`]'s arm BUILDS a
/// `PrefillPlanCache` from the HOST mirrors of the CSR
/// (`AttnCtx::qo_indptr_h`, `::kv_page_indptr_h`) on the way in and drops it
/// on the way out, so no cache crosses the fire and `operand()` has nothing
/// to read. `.wiki/kilimanjaro3.md` §3.3 keeps `Cx` query-only, so the binder
/// may not plan on its behalf either. The pair also plans into the DECODE
/// carve, so answering it off `fa2_prefill_leaves` would resolve against
/// storage the decode plan staged.
///
/// What did change is the forward. Both used to call
/// [`dispatch_attention_flashinfer_prefill_bf16`] whole; that launcher's plan
/// is now twenty-two leaves, so they call [`prefill_paged`] directly and keep
/// their own `Env<PrefillPlan>`. Decode has no planless twin, which is why it
/// went first.
///
/// ## The decode pair's column is complete, and the arm still stands
///
/// Every entry answers on `dispatch_attention_flashinfer_decode` and its
/// `_lse` twin -- `lse` at 7 through `keys::AttnLseOut`, `window_left` at 24
/// through `keys::WindowLeft`, `logits_soft_cap` at 25 through
/// `keys::AttnLogitsSoftCap`, `broadcast_q` at 27 as a `Source::Lit`, and the
/// sixteen leaves between. The pin block below walks the whole column rather
/// than naming indices, so a parameter added without a source stops the
/// build. The capture twin is two short: `score_out` and `score_indptr` are
/// `AttnCtx` fields no `Fire` accessor returns.
///
/// §4 predicted the residue would be the upload alone. It is four things:
/// the upload (`fire/launch.rs:223` -- a replay runs no arm), the
/// `spec.aux`/`per_head_dim` refusal (a `Source` cannot assert a slot is
/// EMPTY), the dequant prelude (a second launch), and `lse_slab`'s null
/// refusal, which `keys::AttnLseOut` deliberately does not make.
pub static ROUTINES: &[Routine] = &[
    routine!(
        dispatch_attention_flashinfer_decode,
        depth_prefix_plan
    ),
    // D2's two halves, added when one symbol at two arities became two
    // symbols at one each. `depth_prefix_plan` and `whole` are copied from
    // the twin above and below rather than defaulted: gpt-oss took the
    // union-tail plan swap through the decode string and deepseek-v4 took
    // `whole` through the planless one, and a defaulted column here would
    // change the lowering of one family and nothing would say so.
    routine!(
        dispatch_attention_flashinfer_decode_lse,
        depth_prefix_plan
    ),
    routine!(
        dispatch_attention_flashinfer_decode_capture
    ),
    routine!(
        dispatch_attention_flashinfer_prefill_bf16
    ),
    routine!(
        dispatch_attention_flashinfer_prefill_capture_bf16
    ),
    routine!(
        dispatch_attention_flashinfer_prefill_custom
    ),
    routine!(
        attention_flashinfer_prefill,
        whole
    ),
    routine!(
        attention_flashinfer_prefill_lse,
        whole
    ),
];

/// The FA2 lattice's eight symbols, under the namespace a trace spells them in.
///
/// `attn`, not `fa2`: the namespace is what a trace SAYS, and it says
/// `attn::dispatch_attention_flashinfer_decode`. A second `Family` with that
/// namespace rather than six rows appended to `x::attn::ROUTINES` keeps the
/// declaration beside the bodies; [`crate::sigs`]' `no_symbol_is_declared_twice`
/// is what makes two families under one namespace safe.
pub static FAMILY: Family = crate::family!(ROUTINES);

/// D2's four columns, pinned: the two new symbols' slots and the two old
/// ones' provenance.
///
/// **This session cannot run a test, so every claim about the split is a
/// `const` assertion over the derived column.** What a split can get wrong is
/// not that it fails to compile — a wrong split compiles perfectly and runs a
/// different kernel over different memory. Three things can move under it and
/// all three are here.
///
/// **THE SLOTS MUST NOT SHIFT.** `stated_source`
/// (`kernels-macros/src/lib.rs:271`) SETS the counter to `N + 1` at a stated
/// index rather than merely consuming one, so removing a stated wrapper can
/// renumber a later bare pointer. Nothing in these signatures is bare — every
/// non-region pointer is `Env<_>`, which `derive_all` sends to the key
/// without touching either counter — so `q` at `In(0)` and `o` at `Out(0)`
/// are expected to be exactly where they were on the twins, and identical on
/// the new symbols. An assertion that they are is the difference between
/// believing that and knowing it.
///
/// **`nullable` MOVES ON `lse` AND MUST NOT ON `o`.** `Or` sets it
/// (`classify`, `:175`); `Env` and a bare `Out` do not. The pairs below read
/// `o` nullable / `lse` not on the twins — because `o` still carries
/// `Provenance::Either` for `accepts_an_unstated_result` and `lse` no longer
/// carries it for anything — and NEITHER nullable on the `_lse` symbols,
/// where both results are the statement's.
///
/// **`len()` CATCHES A PARAMETER LEAVING** rather than moving, which an index
/// assertion cannot: `ctx: &Ctx` is not in the derived column, so the decode
/// pair is 15 rows and the planless pair is 14. (That last clause read *"14
/// and 14"* until Stage 3 counted them against the assertions three lines
/// below it, which had been right the whole time.)
const _: () = {
    // The decode pair. Same launcher, same plan, same lattice point; the
    // difference is entirely in these four columns.
    let d = <dispatch_attention_flashinfer_decode as kernels::Derivation>::DERIVED;
    // 15 → 28. The aggregate and the workspace pair left; sixteen named
    // leaves arrived at 8..24. `impl_kernel_fn!` is stamped through 36.
    assert!(d.len() == 28);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `o` KEEPS `Or`. `attn_at` declares no result inside
    // `llama_like/forward/mod.rs:1638`'s `dsl::guarded_value`, and this flag
    // going false is that guard's value losing its writer.
        // `nullable` WAS TRUE HERE AND IS FALSE NOW, AND THIS PIN IS HOW YOU KNOW.
    // `Or<_>` left this parameter with Kilimanjaro III's last step: the fact
    // it carried -- `Provenance::Either`, which `walk.rs` read to decide that
    // a statement declaring no result should be handed its value region's
    // buffer -- is stated on the OP now (`Op::dest`), by the builder that
    // minted that buffer. A launcher's type no longer decides a lowering.
    assert!(!d[1].nullable);
    // `lse` WAS `d[7]` WITH `Source::Slot(Out, 1)` AND `nullable`, then
    // `Env<*mut f32>` with no source at all. It is `keys::AttnLseOut` now,
    // answered off `Fire::lse_out` -- which does NOT null-check, so a fire
    // with no lse destination still binds and the launcher's own test
    // decides.
    assert!(kernels::source_is_named(
        &d[7].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    assert!(!d[7].nullable);

    let d = <dispatch_attention_flashinfer_decode_lse as kernels::Derivation>::DERIVED;
    assert!(d.len() == 28);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // THE OPERAND SLOTS DID NOT SHIFT. Both results sit where the twin's
    // marks put them, which is what makes the forward at the bottom of the
    // body a forward rather than a re-ordering.
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[7].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(!d[1].nullable);
    assert!(!d[7].nullable);

    // The planless prefill pair. One `Env` more than the decode pair before
    // `lse` -- `qo_indptr` -- so the LSE sits at 7 rather than 6, and that
    // one index is why these are written out instead of looped.
    let d = <attention_flashinfer_prefill as kernels::Derivation>::DERIVED;
    assert!(d.len() == 14);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
        // `nullable` WAS TRUE HERE AND IS FALSE NOW, AND THIS PIN IS HOW YOU KNOW.
    // `Or<_>` left this parameter with Kilimanjaro III's last step: the fact
    // it carried -- `Provenance::Either`, which `walk.rs` read to decide that
    // a statement declaring no result should be handed its value region's
    // buffer -- is stated on the OP now (`Op::dest`), by the builder that
    // minted that buffer. A launcher's type no longer decides a lowering.
    assert!(!d[1].nullable);
    assert!(kernels::source_is_named(
        &d[8].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    assert!(!d[8].nullable);

    let d = <attention_flashinfer_prefill_lse as kernels::Derivation>::DERIVED;
    assert!(d.len() == 14);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(!d[1].nullable);
    assert!(!d[8].nullable);

    // THE FOUR THAT DID NOT SPLIT, AND THE CLAIM THAT REPLACED THEIR OLD ONE.
    //
    // This block asserted `nullable` on each `o` and said why: *"a pass that
    // mandatory-ised `o` on any of these stops the tree compiling here rather
    // than un-writing a guard."* It did exactly that, to the pass that
    // mandatory-ised them -- and the reason it was guarding is gone.
    //
    // `o` was optional so that a region-hosted text declaring no result could
    // still be handed the guard's buffer: `walk.rs` saw `outputs: []`, asked
    // whether the LAUNCHER carried a `Provenance::Either` parameter, and
    // substituted. The buffer is on the statement now (`Op::dest`, stamped by
    // `TraceBuilder::push` from the region that minted it), so the substitution
    // is a lookup and `o` can say what it has always done, which is write.
    //
    // The assertion that survives is the one that was load-bearing: `o` is
    // still result 0. A pass that renumbered it would move a write to the
    // wrong buffer, and that is what this stops now.
    assert!(matches!(
        <dispatch_attention_flashinfer_decode_capture as kernels::Derivation>::DERIVED[1].source,
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));
    assert!(matches!(
        <dispatch_attention_flashinfer_prefill_bf16 as kernels::Derivation>::DERIVED[1].source,
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));
    assert!(matches!(
        <dispatch_attention_flashinfer_prefill_capture_bf16 as kernels::Derivation>::DERIVED[1].source,
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));
    assert!(matches!(
        <dispatch_attention_flashinfer_prefill_custom as kernels::Derivation>::DERIVED[1].source,
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));

    // ── STAGE 3: THE TWO FACTS THAT GOT WORDS, AND THE ONE THAT DID NOT ──
    //
    // `keys::QoIndptr` and `keys::SmScale` replaced `Env<*const u32>` and
    // `Env<f32>` on all eight launchers. Both were `None` columns over
    // pointers every arm bound, and the failure mode of fixing that is
    // silent: `Env<T>` is `Provenance::Env` for every `T`, so a key on the
    // WRONG parameter type-checks, binds, and reads another buffer. These
    // assertions say WHICH parameter carries WHICH key, by index.
    //
    // `source_is_named` (`kernels/src/lib.rs:902`) compares the `&'static
    // str` inside `Source::Named`, so a key whose string is edited without
    // its binder arm being edited stops the build here.
    let d = <dispatch_attention_flashinfer_decode as kernels::Derivation>::DERIVED;
    // 13 → 26. Sixteen leaves landed between `lse` and the tail scalars.
    assert!(kernels::source_is_named(
        &d[26].source,
        <kernels::keys::SmScale as kernels::keys::Fact>::KEY
    ));
    // AND THE NEIGHBOUR, WHICH IS A DIFFERENT KEY AND NOT THE SAME ONE.
    // `logits_soft_cap` sits one index below `sm_scale` and is read off the
    // same `AttnCtx` borrow, which is exactly the shape of a mistaken sweep.
    // `keys::AttnLogitsSoftCap` is total -- `0` means none -- where
    // `keys::SmScale` refuses; binding either through the other's arm would
    // compile.
    assert!(kernels::source_is_named(
        &d[25].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));

    let d = <attention_flashinfer_prefill as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[4].source,
        <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[13].source,
        <kernels::keys::SmScale as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[12].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));
    // THE FORWARD IS INDEX-FOR-INDEX. `attention_flashinfer_prefill_lse`
    // hands its whole parameter list to the twin above, so a key landing on
    // a different index on either side is a re-ordering wearing a forward.
    let d = <attention_flashinfer_prefill_lse as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[4].source,
        <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[13].source,
        <kernels::keys::SmScale as kernels::keys::Fact>::KEY
    ));
    let d = <dispatch_attention_flashinfer_prefill_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[4].source,
        <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY
    ));
    // 13 → 32, for the decode pair's reason: the leaves landed between `lse`
    // and the tail scalars.
    assert!(kernels::source_is_named(
        &d[32].source,
        <kernels::keys::SmScale as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[31].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));
    let d = <dispatch_attention_flashinfer_prefill_capture_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[4].source,
        <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[35].source,
        <kernels::keys::SmScale as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[34].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));
    let d = <dispatch_attention_flashinfer_prefill_custom as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[4].source,
        <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[34].source,
        <kernels::keys::SmScale as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[33].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));

    // ── STAGE 3, SECOND PASS: THE WORKSPACE, AND THE CARVE IT IS NOT ─────
    //
    // `keys::AttnWorkspaceInt` / `...Float` landed on FIVE of these eight and
    // deliberately not on the other three, so this is the one place in the
    // file where the interesting assertion is the NEGATIVE one. The rule the
    // indices below encode: a launcher may name the workspace exactly when
    // its arm passes `AttnCtx::workspace`, because that is the field
    // `Facts::attn_workspace` reads. `fa2_prefill_arm`,
    // `fa2_prefill_capture_arm` and `fa2_prefill_custom_arm` pass
    // `prefill_workspace` -- a SECOND carve that exists because two plans
    // sharing one workspace clobber each other's schedule
    // (`bind/mod.rs:1155-1161`) -- so a key on those three would bind the
    // decode carve to a prefill plan. That is not a naming slip, it is the
    // bug the field split was made to prevent, and it would be invisible
    // until a fire read someone else's schedule.
    // THE DECODE THREE NO LONGER TAKE IT, AND THAT IS THE UNFOLD. Their
    // `int_buffer`/`float_buffer` were bases for offsets the driver now adds
    // itself: `bind/table.rs`'s `fa2_decode_leaves` resolves
    // `int_buffer + int_base_bytes + off` once and answers the address. A
    // parameter that reappeared here would be a base with nothing to offset.
    //
    // INT BEFORE FLOAT WAS THE ASSERTION THAT STOOD HERE, on two same-typed
    // adjacent indices. Its successor is the leaf run below: sixteen
    // same-shaped `Env<keys::Fa2Decode*>` at 8..24, where a swap type-checks
    // and reads a work list one array over.
    // THE PLANLESS PAIR, AT NINE AND TEN because `qo_indptr` sits ahead of
    // them -- the same one-index shift the `lse` assertions above turn on.
    // These two are prefill launchers naming the DECODE carve, which is
    // correct and reads wrong: their arm plans, uploads and launches against
    // `a.workspace`, as the entry point it replaces did.
    let d = <attention_flashinfer_prefill as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[9].source,
        <kernels::keys::AttnWorkspaceInt as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[10].source,
        <kernels::keys::AttnWorkspaceFloat as kernels::keys::Fact>::KEY
    ));
    let d = <attention_flashinfer_prefill_lse as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[9].source,
        <kernels::keys::AttnWorkspaceInt as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[10].source,
        <kernels::keys::AttnWorkspaceFloat as kernels::keys::Fact>::KEY
    ));
    // ── AND THE OTHER CARVE, WHICH NOTHING NAMES ANY MORE ────────────────
    //
    // `keys::AttnPrefillWorkspaceInt`/`...Float` stood at nine and ten on
    // `_capture` and `_custom`, and the entry worth keeping is what happened
    // before that: the block asserted `is_none()` on all THREE planned
    // launchers, the two keys landed, and this file stopped compiling -- the
    // assertion catching a deliberate change, which is the only way to find
    // out it would have caught a mistake.
    //
    // BOTH ROUNDS ARE SPENT NOW. The carve is still the thing to get right
    // and it is got right one hop earlier: `bind/table.rs`'s
    // `fa2_prefill_leaves` reads `Cx::attn_prefill_workspace` and answers
    // ADDRESSES, so no prefill launcher takes a base and none can be handed
    // the decode one. What replaces these four assertions is the leaf run
    // below, at the indices the bases and the aggregate vacated.

    // ── AND THE PAIR THAT WAS WORDLESS BECAUSE IT WAS TWO FACTS ──────────
    //
    // `dispatch_attention_flashinfer_prefill_bf16`'s `int_buffer` and
    // `float_buffer` were pinned `is_none()` here, with the note that the
    // parameter took `AttnCtx::prefill_workspace` from `fa2_prefill_arm` and
    // `AttnCtx::workspace` from [`attention_flashinfer_prefill`]'s forward,
    // so either key would be right for one caller and a silent cross-carve
    // read for the other -- and that *"WHAT WOULD ACTUALLY FIX IT is not a
    // third key: it is the forward."*
    //
    // THAT IS WHAT THE UNFOLD DID. The forward is gone: the planless pair
    // calls `prefill_paged` with its own carve's bases and its own aggregate,
    // and this launcher takes neither a base nor a plan. One parameter with
    // two meanings became no parameter at all.

    // ── THE EIGHT AGGREGATES, PINNED UNNAMED ────────────────────────────
    //
    // ── THE TWO AGGREGATES LEFT, PINNED UNNAMED ─────────────────────────
    //
    // Six of the eight crossed; these two are the planless pair, and the
    // `is_none()` on `plan` is the accurate reading rather than a gap. The
    // arm BUILDS this plan from the host CSR mirrors on the way in, so
    // nothing published it before the fire and `operand()` has nothing to
    // read -- `.wiki/kilimanjaro3.md` §3.3 keeps `Cx` query-only, so the
    // binder cannot plan on its way past.
    //
    // WHAT THESE ASSERTIONS BUY IS THE ORDERING, WHICH IS THE ONE THING A
    // TWENTY-LEAF UNFOLD CAN GET SILENTLY WRONG. `Env<T>` is
    // `Provenance::Env` for every `T`, so `keys::Fa2PrefillKvTileIndices` on
    // the parameter that meant `request_indices` type-checks, binds and reads
    // a work list one array over. That is what the two runs below are for,
    // and the assertion that fails first says which launcher's run slipped.
    //
    // The two families differ by one at the head because `qo_indptr` sits at
    // four on every prefill signature and nothing corresponds to it on
    // decode; the prefill run is six leaves longer, which is what a schedule
    // that tiles QO rows has and a decode schedule does not.
    let d = <dispatch_attention_flashinfer_decode_capture as kernels::Derivation>::DERIVED;
    assert!(d.len() == 30);
    let d = <dispatch_attention_flashinfer_prefill_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 33);
    let d = <dispatch_attention_flashinfer_prefill_capture_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 36);
    let d = <dispatch_attention_flashinfer_prefill_custom as kernels::Derivation>::DERIVED;
    assert!(d.len() == 35);
    let d = <attention_flashinfer_prefill as kernels::Derivation>::DERIVED;
    assert!(d.len() == 14);
    assert!(d[11].source.is_none());
    let d = <attention_flashinfer_prefill_lse as kernels::Derivation>::DERIVED;
    assert!(d.len() == 14);
    assert!(d[11].source.is_none());

    // ── AND THE THREE THAT DID, LEAF BY LEAF ────────────────────────────
    //
    // Sixteen `Env<keys::Fa2Decode*>` at 8..24 on all three decode
    // launchers, in one order. `Env<T>` is `Provenance::Env` for every `T`,
    // so a key on the wrong parameter type-checks, binds, and reads the
    // neighbouring array -- these assertions are the only thing that says
    // WHICH index carries WHICH fact.
    //
    // The three run identically because `_lse` forwards its whole list to
    // the plain launcher and `_capture` differs only after 24.
    macro_rules! decode_leaves {
        ($sym:ty) => {{
            let d = <$sym as kernels::Derivation>::DERIVED;
            assert!(kernels::source_is_named(
                &d[8].source,
                <kernels::keys::Fa2DecodeRequestIndices as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[9].source,
                <kernels::keys::Fa2DecodeKvTileIndices as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[10].source,
                <kernels::keys::Fa2DecodeOIndptr as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[11].source,
                <kernels::keys::Fa2DecodeKvChunkSize as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[12].source,
                <kernels::keys::Fa2DecodeBlockValidMask as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[13].source,
                <kernels::keys::Fa2DecodeTmpV as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[14].source,
                <kernels::keys::Fa2DecodeTmpS as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[15].source,
                <kernels::keys::Fa2DecodePaddedBatch as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[16].source,
                <kernels::keys::Fa2DecodeSplitKv as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[17].source,
                <kernels::keys::Fa2DecodeRequests as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[18].source,
                <kernels::keys::Fa2DecodeNumQHeads as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[19].source,
                <kernels::keys::Fa2DecodeNumKvHeads as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[20].source,
                <kernels::keys::Fa2DecodeHeadDim as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[21].source,
                <kernels::keys::Fa2DecodePageSize as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[22].source,
                <kernels::keys::Fa2DecodeHndLayout as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[23].source,
                <kernels::keys::Fa2DecodeFullAttention as kernels::keys::Fact>::KEY
            ));
        }};
    }
    decode_leaves!(dispatch_attention_flashinfer_decode);
    decode_leaves!(dispatch_attention_flashinfer_decode_lse);
    decode_leaves!(dispatch_attention_flashinfer_decode_capture);

    // ── AND THE PLANNED PREFILL THREE, LEAF BY LEAF ─────────────────────
    //
    // Twenty-two `Env<keys::Fa2Prefill*>` at 9..31 on all three, in one
    // order, one index later than decode's run because `qo_indptr` sits at
    // four here. The hazard is decode's and worse: four `*const i32` work
    // lists run consecutively at 9..15, so a swapped pair type-checks, binds,
    // and schedules a kernel against the wrong array.
    //
    // `window_left` at 27 is `keys::Fa2PrefillWindowLeft` and NOT
    // `keys::WindowLeft`: this is the window the split was sized against,
    // read back off the plan, where the other is the statement's own. On the
    // planned three they are the same number today and the keys are not the
    // same fact -- a statement may state a window a plan was not built for,
    // and the leaf is what the schedule actually indexes.
    macro_rules! prefill_leaves {
        ($sym:ty) => {{
            let d = <$sym as kernels::Derivation>::DERIVED;
            assert!(kernels::source_is_named(
                &d[9].source,
                <kernels::keys::Fa2PrefillRequestIndices as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[10].source,
                <kernels::keys::Fa2PrefillQoTileIndices as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[11].source,
                <kernels::keys::Fa2PrefillKvTileIndices as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[12].source,
                <kernels::keys::Fa2PrefillMergeIndptr as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[13].source,
                <kernels::keys::Fa2PrefillOIndptr as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[14].source,
                <kernels::keys::Fa2PrefillKvChunkSize as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[15].source,
                <kernels::keys::Fa2PrefillBlockValidMask as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[16].source,
                <kernels::keys::Fa2PrefillTmpV as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[17].source,
                <kernels::keys::Fa2PrefillTmpS as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[18].source,
                <kernels::keys::Fa2PrefillPaddedBatch as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[19].source,
                <kernels::keys::Fa2PrefillSplitKv as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[20].source,
                <kernels::keys::Fa2PrefillTotalRows as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[21].source,
                <kernels::keys::Fa2PrefillCtaTileQ as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[22].source,
                <kernels::keys::Fa2PrefillRequests as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[23].source,
                <kernels::keys::Fa2PrefillNumQHeads as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[24].source,
                <kernels::keys::Fa2PrefillNumKvHeads as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[25].source,
                <kernels::keys::Fa2PrefillHeadDim as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[26].source,
                <kernels::keys::Fa2PrefillPageSize as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[27].source,
                <kernels::keys::Fa2PrefillWindowLeft as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[28].source,
                <kernels::keys::Fa2PrefillHndLayout as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[29].source,
                <kernels::keys::Fa2PrefillFullAttention as kernels::keys::Fact>::KEY
            ));
            assert!(kernels::source_is_named(
                &d[30].source,
                <kernels::keys::Fa2PrefillCausalMask as kernels::keys::Fact>::KEY
            ));
        }};
    }
    prefill_leaves!(dispatch_attention_flashinfer_prefill_bf16);
    prefill_leaves!(dispatch_attention_flashinfer_prefill_capture_bf16);
    prefill_leaves!(dispatch_attention_flashinfer_prefill_custom);

    // THE PREFILL HEAD AND TAIL. `lse` is `Env` on all three: no text of any
    // of these symbols declares a second result.
    let d = <dispatch_attention_flashinfer_prefill_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[8].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
    // The capture twin is three short and the custom two, and they are their
    // own: `score_out`/`score_indptr`/`score_window` at 31..34 and the custom
    // mask pair at 31..33 are `AttnCtx` fields no `Fire` accessor returns.
    let c = <dispatch_attention_flashinfer_prefill_capture_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &c[8].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    let mut i = 0;
    while i < c.len() {
        assert!(c[i].source.is_some() || i == 31 || i == 32 || i == 33);
        i += 1;
    }
    assert!(c[31].source.is_none());
    assert!(c[32].source.is_none());
    assert!(c[33].source.is_none());
    let m = <dispatch_attention_flashinfer_prefill_custom as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &m[8].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    let mut i = 0;
    while i < m.len() {
        assert!(m[i].source.is_some() || i == 31 || i == 32);
        i += 1;
    }
    assert!(m[31].source.is_none());
    assert!(m[32].source.is_none());

    // AND THE TAIL, WHICH IS NOW COMPLETE ON ALL THREE.
    //
    // `keys::WindowLeft` reads `window_of(spec, attn, layer)`, the same call
    // `attn_plan` (`bind/mod.rs:1387`) switches the decode plan on at `== -1`,
    // so launcher and selector agree by construction. The slot spelling
    // `Param<0, i32>` reaches the same statement field but cannot carry that
    // value: `LaunchSpec::params` is `Vec<u32>`, so an unbounded window
    // arrives as `0xFFFF_FFFF` and `as_declared` converts `U32 -> I32` only
    // under `n <= i32::MAX as u32`.
    //
    // `broadcast_q` is a `Source::Lit` and not a key: every call site in the
    // tree passes `false`, and a key would name a constant as though it were
    // a fact about the fire.
    let d = <dispatch_attention_flashinfer_decode as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(
        &d[7].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[24].source,
        <kernels::keys::WindowLeft as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[25].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));
    assert!(matches!(d[27].source, Some(kernels::Source::Lit(kernels::Lit::Bool(false)))));
    assert!(d[27].stated);

    // The capture twin, whose two extra parameters at 24/25 push the tail
    // to 26..29.
    let c = <dispatch_attention_flashinfer_decode_capture as kernels::Derivation>::DERIVED;
    assert!(c.len() == 30);
    assert!(kernels::source_is_named(
        &c[7].source,
        <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &c[26].source,
        <kernels::keys::WindowLeft as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &c[27].source,
        <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY
    ));
    assert!(matches!(c[29].source, Some(kernels::Source::Lit(kernels::Lit::Bool(false)))));

    // EVERY COLUMN ON THE DECODE THREE IS ANSWERED. What keeps the arm is
    // `fire/launch.rs:223` and nothing in this list: a replay runs no arm, so
    // the H2D that refreshes the plan descriptor has to happen somewhere
    // `Bound::derived` -- which goes straight to `table::dispatch` -- does
    // not.
    let e = <dispatch_attention_flashinfer_decode_lse as kernels::Derivation>::DERIVED;
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        assert!(e[i].source.is_some());
        i += 1;
    }
    // THE CAPTURE TWIN IS TWO SHORT, AND THEY ARE ITS OWN TWO. `score_out`
    // at 24 and `score_indptr` at 25 are the `AttnCtx` fields the capture
    // spelling publishes into; no `Fire` accessor returns either, so there is
    // nothing for a key to be answered off -- F7, a deficiency gets a fix.
    let mut i = 0;
    while i < c.len() {
        assert!(c[i].source.is_some() || i == 24 || i == 25);
        i += 1;
    }
    assert!(c[24].source.is_none());
    assert!(c[25].source.is_none());
};

/// `#[lit(..)]`'s four variants, one parameter each.
///
/// `broadcast_q` above exercises `Bool` on a live row; the other three have no
/// site in this family yet, and a mark with no consumer is how a macro arm
/// rots. These are the consumers.
#[allow(dead_code)]
mod lit_pins {
    use super::{Ctx, Env, Refusal};

    #[kernels_macros::routine]
    pub fn all_four(
        _ctx: &Ctx,
        #[lit(null)] _absent: Env<*mut f32>,
        #[lit(true)] _flag: bool,
        #[lit(-1)] _count: i32,
        #[lit(1.702)] _alpha: f32,
    ) -> Result<(), Refusal> {
        Ok(())
    }

    const _: () = {
        let d = <all_four as kernels::Derivation>::DERIVED;
        assert!(d.len() == 4);
        assert!(matches!(d[0].source, Some(kernels::Source::Lit(kernels::Lit::Null))));
        assert!(matches!(d[1].source, Some(kernels::Source::Lit(kernels::Lit::Bool(true)))));
        assert!(matches!(d[2].source, Some(kernels::Source::Lit(kernels::Lit::I32(-1)))));
        assert!(matches!(d[3].source, Some(kernels::Source::Lit(kernels::Lit::F32(x))) if x == 1.702));
        // A LITERAL DOES NOT CONSUME AN OPERAND SLOT. `_absent` is a `*mut f32`
        // and would have counted as `Out(0)` on position alone; the attribute
        // runs before the counters, so a row that states one keeps its numbering.
        assert!(d[0].stated);
    };
}

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

    /// The family declares the eight symbols a trace states, under `attn`,
    /// and the two decode dispatches are the only rows in the whole table
    /// that carry `depth_prefix_plan`.
    ///
    /// `model-ir`'s `trace/` reads that column for the union-tail plan
    /// swap. It survived the crossing as `routine!`'s trailing fact; a row
    /// that quietly lost it would lower a different plan and nothing else
    /// would say so — which is why the `_lse` row added by D2 carries it
    /// too. gpt-oss states the `_lse` decode spelling
    /// (`gpt_oss/forward/mod.rs:188`) and was swapping before the split.
    #[test]
    fn the_family_declares_the_eight_and_keeps_the_depth_prefix_column() {
        for symbol in [
            "attn::dispatch_attention_flashinfer_decode",
            "attn::dispatch_attention_flashinfer_decode_lse",
        ] {
            let decode = FAMILY.routine(symbol).unwrap_or_else(|| panic!("{symbol} is declared"));
            assert!(decode.depth_prefix_plan, "the union-tail plan swap reads this column");
            assert!(!decode.whole, "a decode dispatch covers a row range");
        }

        for symbol in
            ["attn::attention_flashinfer_prefill", "attn::attention_flashinfer_prefill_lse"]
        {
            let planless = FAMILY.routine(symbol).unwrap_or_else(|| panic!("{symbol} is declared"));
            assert!(planless.whole, "it plans over the whole fire on the way in");
            assert!(!planless.depth_prefix_plan);
        }

        assert_eq!(FAMILY.routines.len(), 8);
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
