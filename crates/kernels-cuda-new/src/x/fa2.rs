//! The FlashInfer FA2 lattice: 56 roots over one `attn/fa2.cuh`, and the four
//! launches that fire them.
//!
//! # 56 roots and one `include_str!`
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
//! `csrc/src/attn/flashinfer` as well as the library set — and carries
//! `--device-as-default-execution-space`, which is load-bearing:
//! `page.cuh`'s and `fastdiv.cuh`'s guarded `__host__` constructors are
//! `#ifndef __CUDACC_RTC__`, and everything else in the closure was written
//! for `nvcc`, which defaults an unannotated function to `__host__
//! __device__` where NVRTC's JIT mode defaults it to `__host__` and refuses.
//!
//! # No `FAMILY`, and no line in `lib.rs`
//!
//! No trace statement names an FA2 kernel. `driver-cuda` reaches these by
//! path from its own plan caches, exactly as it does `x::driver_internal`'s
//! launchers, so there is nothing for `call()` to resolve and no `ROUTINES`
//! table to hold a `Routine` nobody looks up. The four `pub fn`s below also
//! take a params BLOCK — a `#[repr(C)]` mirror with no `Arg` impl — which a
//! `Routine` could not carry even if a trace wanted one.
//!
//! # What stays in `driver-cuda`
//!
//! The plan caches, the params filling and the arm cascades. A params block
//! is filled from a plan descriptor and a workspace, which are the driver's
//! vocabulary; the arm a request selects is a cascade over the request's own
//! flags. What crossed is what this file is: which root, which instantiation,
//! and the rectangle.

use core::mem::size_of;

use crate::fa2::params::{DecodeParams, DecodeScoreParams, PrefillPagedParams, PrefillScoreParams};
use crate::fa2::{DecodeGeometry, Device, KvWidth, PrefillGeometry};
use crate::jit::{ArgValue, Ctx, Launch, Root};
use kernels::Refusal;

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

/// `attn/fa2.cuh` — `PagedTraits`, the six variant aliases and the two params
/// structs live in one header, and every root below is that header.
const TEXT: &str = include_str!("../../csrc/src/attn/fa2.cuh");

/// The path a diagnostic names, relative to `csrc/src`.
const FILE: &str = "attn/fa2.cuh";

/// `--device-as-default-execution-space`, and it is load-bearing — see the
/// module header.
const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

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
            "::pie_cuda_driver::kernels::attn::fa2::", $variant, ", ",
            "::pie_cuda_driver::kernels::attn::fa2::", $params, ">",
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
            root: Root::new(
                concat!("attn/fa2_decode_hd", stringify!($hd), "_g", stringify!($g)),
                TEXT,
                FILE,
            )
            .options(OPTIONS)
            .upstream(),
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
            "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::",
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
            "::pie_cuda_driver::kernels::attn::fa2::",
            $variant,
            ">, ",
            "::pie_cuda_driver::kernels::attn::fa2::",
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
            root: Root::new(
                concat!(
                    "attn/fa2_prefill_hd",
                    stringify!($hd),
                    "_q",
                    stringify!($q),
                    "_kv",
                    stringify!($kv),
                ),
                TEXT,
                FILE,
            )
            .options(OPTIONS)
            .upstream(),
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
/// [`PrefillGeometry::num_mma_kv`](crate::fa2::PrefillGeometry::num_mma_kv),
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
/// [`DecodeGeometry`](crate::fa2::DecodeGeometry), at `KvWidth::BF16`.
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
        ctx.launch(
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
/// [`PrefillGeometry::shared_storage_paged`](crate::fa2::PrefillGeometry::shared_storage_paged)
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
        ctx.launch(
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

/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the decode entry point —
/// `decode.cuh:715-718`.
///
/// Upstream multiplies this by the SM count to bound `plan::decode::estimate`'s
/// split. It is a per-CUBIN fact, so asking it COMPILES the point: the answer
/// is not available before the kernel exists.
///
/// `block_threads` is [`DecodeGeometry::num_threads`](crate::fa2::DecodeGeometry::num_threads)
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
        DECODE, DECODE_GQA, DecodeArm, HEAD_DIMS, PREFILL, PrefillArm, decode_instantiation,
        decode_root, prefill_instantiation, prefill_root,
    };
    use crate::fa2::{DecodeGeometry, Device, KvWidth, PrefillGeometry};

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
                    !arm.contains("::pie_cuda_driver::kernels::::"),
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
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie_cuda_driver::kernels::attn::fa2::VariantFull, ::pie_cuda_driver::kernels::attn::fa2::DecodeParams>",
            ),
            (
                DecodeArm::Softcap,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie_cuda_driver::kernels::attn::fa2::VariantWindowSoftcap, ::pie_cuda_driver::kernels::attn::fa2::DecodeParams>",
            ),
            (
                DecodeArm::Window,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie_cuda_driver::kernels::attn::fa2::VariantWindow, ::pie_cuda_driver::kernels::attn::fa2::DecodeParams>",
            ),
            (
                DecodeArm::CaptureFull,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie_cuda_driver::kernels::attn::fa2::CaptureFull, ::pie_cuda_driver::kernels::attn::fa2::DecodeCaptureParams>",
            ),
            (
                DecodeArm::CaptureWindow,
                "::flashinfer::BatchDecodeWithPagedKVCacheKernel<::flashinfer::PosEncodingMode::kNone, 2, 4, 8, 8, 1, 16, ::pie_cuda_driver::kernels::attn::fa2::CaptureWindow, ::pie_cuda_driver::kernels::attn::fa2::DecodeCaptureParams>",
            ),
        ];
        for (arm, wanted) in decode {
            assert_eq!(decode_instantiation(64, 1, arm), Some(wanted), "decode hd64 g1 {arm:?}");
        }

        let prefill = [
            (
                PrefillArm::CausalFullSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantFullSoftcap>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::NoneFullSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantFullSoftcap>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalFull,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantFull>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::NoneFull,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantFull>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantWindowSoftcap>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalWindow,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantWindow>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::CausalCapture,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::CapturePrefill>, ::pie_cuda_driver::kernels::attn::fa2::PrefillCaptureParams>",
            ),
            (
                PrefillArm::NoneCapture,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::CapturePrefill>, ::pie_cuda_driver::kernels::attn::fa2::PrefillCaptureParams>",
            ),
            (
                PrefillArm::CustomSoftcap,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantCustomSoftcap>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
            ),
            (
                PrefillArm::Custom,
                "::flashinfer::BatchPrefillWithPagedKVCacheKernel<::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, 64, 1, 2, 8, 8, 4, 1, ::pie_cuda_driver::kernels::attn::fa2::VariantCustom>, ::pie_cuda_driver::kernels::attn::fa2::PrefillParams>",
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
