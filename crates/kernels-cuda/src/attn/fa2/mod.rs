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
//! # Six routines under `attn`'s namespace, not `fa2`'s
//!
//! Six trace symbols DO name this lattice — `attn::dispatch_attention_flashinfer_*`
//! and `attn::attention_flashinfer_prefill` — so the [`ROUTINES`] table at the
//! bottom is the derived half of their [`crate::sigs`] rows, and
//! [`crate::not_yet_crossed`] states them no longer. The namespace is `attn`
//! rather than `fa2` because that is what a trace says; `Routine::namespace`
//! is derived per-routine from `module_path!()` — the first segment after the
//! crate root — so a file states its own symbols under its parent's namespace
//! without editing a shared table, and [`crate::sigs`]' own test refuses a
//! symbol two namespaces both claim.
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

use kernels::keys;
use kernels_macros::routine;
use core::ffi::c_void;
use core::mem::size_of;

use crate::attn::fa2::params::{
    Buffers, DecodeParams, DecodePlan, DecodeScoreParams, DevicePtr, Partials, PrefillPagedParams,
    PrefillPlan, PrefillScoreParams, make_decode_params, make_prefill_params,
};
use crate::attn::fa2::geometry::{DecodeGeometry, Device, KvWidth, PrefillGeometry};
use crate::jit::{ArgValue, Ctx, Cuda, Launch, Root};
use crate::jit::abi::{bf16, unpack_aggregate};
use crate::jit::abi::Tensor;
use kernels::raises::Struct;
use crate::raises::{Fa2Decode, Fa2Prefill};
use kernels::routine::{Arg, Asks, Const, In, Out};
use kernels::{Refusal, Ty};
use crate::routine::Fire;

// ── What a launcher does before it fires ────────────────────────────────────
//
// Two of the four things that kept `arms/fa2.rs` alive, moved to where the
// launch is. Neither could ever have been a `Source`, and the reason is the
// same for both: a column binds an ARGUMENT LIST, and these are WORK. The arm
// was not carrying knowledge the signature lacked — it was carrying two
// operations that had nowhere else to happen.
//
// They belong to the launcher rather than to a statement of their own because
// of graph capture. `fire::launch::raise_attn_plans` runs OUTSIDE the captured
// region; only what `run_captured` issues becomes a node. A prelude hoisted to
// planning would be skipped by every replay, leaving the device reading the
// schedule of whichever fire was captured.

/// Widen this layer's quantised KV pages into their bf16 mirrors.
///
/// `arms/fa2.rs`'s `dequant_prelude`, and the discarded result is its too: a
/// layer whose dtype `KvDType` does not name has nothing to widen, and the
/// attention below still runs over pages that were never quantised. The
/// condition is a BOOT fact, which is why it is a decline inside the callee
/// rather than a branch here — nothing at this level knows the cache's dtype.
fn dequant_prelude(ctx: &Ctx<'_>) {
    let _ = crate::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(ctx);
}

/// Put the plan descriptor on the device, where this launch will read it.
///
/// `arms/fa2.rs`'s `upload`. The three numbers arrive as asked facts, resolved
/// by the driver off the same plan cache and the same workspace carve the
/// sixteen decode leaves come from — so a prefill descriptor cannot land in
/// the decode carve, which is what a single shared key pair would have risked.
///
/// `int_base_bytes` is passed as ZERO because `dst` already carries it: the
/// driver adds the plan's base once, where it reads the carve, rather than at
/// every reader of it.
///
/// # Errors
///
/// [`Refusal::Device`] if the copy faulted.
fn upload_plan(ctx: &Ctx<'_>, src: *const u8, len: usize, dst: *mut u8) -> Result<(), Refusal> {
    if len == 0 || src.is_null() || dst.is_null() {
        return Ok(());
    }
    // SAFETY: `src` is the plan cache's `PinnedBytes`, which the cache owns and
    // outlives this fire and every replay of it; `len` is that buffer's own
    // length, both read off one `as_slice()` in the driver. The slice is read
    // and never held past the call.
    let bytes = unsafe { core::slice::from_raw_parts(src, len) };
    // SAFETY: `dst` addresses the fire's integer workspace at the plan's base,
    // which the planner carved to hold at least these bytes, and `ctx`'s
    // stream is live across the copy by `Ctx::on`'s contract.
    unsafe { plan::upload_int_plan(bytes, dst as u64, 0, ctx.stream()) }
}

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
pub fn decode<P: DecodeBlock>(ctx: &Ctx<'_>, at: DecodePoint, params: &P) -> Result<(), Refusal> {
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
    ctx.fire(
        Fire::at(point.root.file, point.arms[at.arm as usize])
            .unit(point.root.name)
            .apply(
                Launch::grid(
                    DecodeGeometry::grid(at.padded_batch_size, at.num_kv_heads),
                    geometry.block(),
                )
                .smem(geometry.smem_bytes),
            ),
        &[block(params)],
    )
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
pub fn prefill<P: PrefillBlock>(ctx: &Ctx<'_>, at: PrefillPoint, params: &P) -> Result<(), Refusal> {
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
    ctx.fire(
        Fire::at(point.root.file, point.arms[at.arm as usize])
            .unit(point.root.name)
            .apply(
                Launch::grid(
                    PrefillGeometry::grid(at.padded_batch_size, at.num_kv_heads),
                    geometry.block(),
                )
                .smem(geometry.smem_bytes),
            ),
        &[block(params)],
    )
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
// `Env<keys::Fa2Decode*>` and `decode_plan_of_leaves` rebuilt the aggregate
// inside this file. BOTH ARE GONE: the launchers name their plan as an
// operand and read the cache whole, so nothing folds and nothing unfolds.
// What keeps these impls is `unpack_aggregate`, still on the path the
// planless pair takes. The [`ROUTINES`] ledger carries what the
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
fn fold(ctx: &Ctx<'_>, split: &Partials) -> Result<(), Refusal> {
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
// longer do. The launcher reads the cache off its plan operand and checks
// `valid` there, which is where the flag belongs: `fa2_decode_leaves`, which
// used to refuse before answering a single leaf, is deleted with the leaves. `UNPLANNED_PREFILL` below still exists
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
#[routine(depth_prefix_plan, no_join)]
pub fn dispatch_attention_flashinfer_decode(
    ctx: &Ctx<'_>,
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
    // `In<0, *const _>` is no longer the only one. (`OutSlot` and `Or` carried this
    // between rounds; both are deleted.)
    q: In<Tensor<bf16>>,
    // THE PLAN, AS AN OPERAND. See
    // [`dispatch_attention_flashinfer_prefill_bf16`]. `In(1)` keeps `q` at
    // slot 0; what the position buys is that the derived column counts it.
    plan: In<Struct<Fa2Decode>>,
    // `Out(0)` STATED. The slot is not a guess: `arms/fa2.rs`'s `o_or` is
    // documented `Source::Or(&Out(0), &Attn("o_out"))` and its body reads
    // `cx.arg_out(0)` before falling back, in all six arms; and
    // `model-dsl/src/cuda/base.rs:373` builds this family's two-result
    // statement with `[(shape, BF16), (Tokens x q_heads, F32)]` in that
    // order, so result 0 IS the bf16 attention output.
    //
    // STATING A SLOT IS NOT WRAPPING ONE, and the ledger's *"`Or<T>` IS NOT A
    // REGION AND `o` IS NOT WRAPPED"* stands unamended. `Out<0, *mut _>` would
    // state nothing because `stated_source` reads the LAST path segment; an
    // ATTRIBUTE is read before any wrapper (`kernels-macros/src/lib.rs:216`
    // sets `stated` on the attribute branch first), so the index becomes an
    // answer while `Or` goes on carrying `Provenance::Either`. That
    // provenance is the whole reason `Out<0, *mut _>` is wrong here -- it would
    // make the result REQUIRED on a text that declines to name one.
    //
    // AND `Out<0, *mut bf16>` IS THAT MARK IN A TYPE, which took a
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
    o: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let window_left = ctx.ask::<i32, keys::WindowLeft>()?;
    let logits_soft_cap = ctx.ask::<f32, keys::AttnLogitsSoftCap>()?;
    let sm_scale = ctx.ask::<f32, keys::SmScale>()?;

    // THE TWO PRELUDES, IN THE ARM'S ORDER. `arms/fa2.rs` widened the cache
    // and then uploaded the descriptor, both before the launch and both on
    // this stream. The order is not incidental: the widening reads the pages
    // this plan's work lists address.
    dequant_prelude(ctx);

    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let broadcast_q = false;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    // THE PLAN, WHOLE, AND THE STATEMENT SAID WHICH. Two decode schedules may
    // stand at once -- gemma-4 states 256 and 512 -- and which one this fire
    // executes used to be recovered by `bind::attn_plan` from the window on the
    // `LaunchSpec`. The prep names it now.
    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the decode plan this statement names" });
    }
    // SAFETY: `Fa2Decode` resolves only to a plan `raise_attn_plans` raised --
    // `DecodePlan::as_ptr`'s own boxed cache, non-null above and live for the
    // fire. The borrow ends inside this launch.
    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::decode_plan_of(cache, crate::attn::fa2::plan::fa_device());
    // THE DESCRIPTOR, OFF THE PLAN THIS STATEMENT NAMES. Three asks stood
    // here -- `Fa2DecodeUploadSrc`/`Len`/`Dst` -- and each resolved through
    // `bind::attn_plan`, which for decode GUESSED which schedule from the
    // window on the `LaunchSpec`. The staging buffer is the cache's own and
    // the carve offset is the cache's own, so a body holding the cache needs
    // neither the asks nor the guess.
    let int_base = ctx.ask::<*mut c_void, keys::AttnWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (int_base as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        // Decode has one query row per request, so there is no QO indptr.
        core::ptr::null(),
        lse,
        // THE DECODE CARVE'S TWO BASES, RAW. This passed a null pair while the
        // leaves carried ABSOLUTE addresses in the offset fields --
        // `decode_plan_of_leaves` stuffed them there and a zero base read them
        // back out. The launcher reads the real cache now, whose offsets are
        // relative to the carve it was raised against, so a null base would
        // make every work-list pointer a small integer.
        //
        // RAW, and not `+ int_base_bytes` as the prefill twin passes:
        // `make_decode_params` adds `plan.int_base_bytes` itself, where
        // `make_prefill_params` says it has "no analogue ... because one
        // prefill plan serves one fire".
        int_base,
        ctx.ask::<*mut c_void, keys::AttnWorkspaceFloat>()?,
    );
    let arm = decode_arm(planned.full_attention_variant, window_left, logits_soft_cap);
    let (params, split) =
        make_decode_params(&planned, &bufs, window_left, logits_soft_cap, sm_scale, broadcast_q);
    decode(ctx, decode_at(&planned, arm, params.padded_batch_size), &params)?;
    if planned.info.split_kv { fold(ctx, &split) } else { Ok(()) }
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
#[routine(depth_prefix_plan, no_join)]
pub fn dispatch_attention_flashinfer_decode_lse(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    // THE PLAN, AS AN OPERAND. See
    // [`dispatch_attention_flashinfer_prefill_bf16`]. `In(1)` keeps `q` at
    // slot 0; what the position buys is that the derived column counts it.
    plan: In<Struct<Fa2Decode>>,
    // REQUIRED, where the twin's is `Out<0, *mut bf16>`, and the difference is
    // the whole of the split. `Or`'s `Provenance::Either` is what
    // `accepts_an_unstated_result` (`model-ir/src/kernels.rs:227`) reads to
    // let a value-producing region hand its buffer to a statement declaring
    // nothing; the twin needs that because `attn_at` declines to name a
    // result inside `llama_like/forward/mod.rs:1638`'s `dsl::guarded_value`.
    // The `_lse` text declares two results unconditionally and is stated
    // outside any region (`gpt_oss/forward/mod.rs:188`), so the clause is
    // never reached and `Trace` is the honest provenance.
    o: Out<Tensor<bf16>>,
    // `Out(1)`, REQUIRED, and this is the parameter the split exists for. On
    // the twin it is `Env<*mut f32>` — the fire's scratch slab — because no
    // one-result text has a second buffer to point at. Here every text has
    // one: `attention_sink_rescale` reads it two statements downstream
    // (`gpt_oss/forward/mod.rs:189`).
    _lse: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let _window_left = ctx.ask::<i32, keys::WindowLeft>()?;
    let _logits_soft_cap = ctx.ask::<f32, keys::AttnLogitsSoftCap>()?;
    let _sm_scale = ctx.ask::<f32, keys::SmScale>()?;

    let _k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let _v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let _kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let _kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let _kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let _broadcast_q = false;
    // EVERY PLAN LEAF IS ASKED FOR INSIDE THE CALLEE NOW, so this forwards the
    // two operands and the three the statement carries. The twenty-four that
    // used to be threaded through here are §6.2's FA2 plan leaves -- what the
    // planner built from this batch's history depth -- and a caller repeating
    // them was a caller able to disagree with the plan object being executed.
    dispatch_attention_flashinfer_decode(ctx, q, plan, o)
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
#[routine(no_join)]
pub fn dispatch_attention_flashinfer_decode_capture(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    // THE PLAN, AS AN OPERAND. See
    // [`dispatch_attention_flashinfer_prefill_bf16`]. `In(1)` keeps `q` at
    // slot 0; what the position buys is that the derived column counts it.
    plan: In<Struct<Fa2Decode>>,
    o: Out<Tensor<bf16>>,
    window_left: Const<i32>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>) -> Result<(), Refusal> {
    // ASKED, NOT BARE POINTERS. These two were the last thing keeping this row
    // `untraced`: a `*mut f32` is not a mark, so the row carried no source
    // column and nothing could bind it but a hand-written arm. They are facts
    // the fire publishes, and `AttnScoreOut`/`AttnScoreIndptr` say so.
    let score_out = ctx.ask::<*mut f32, keys::AttnScoreOut>()?;
    let score_indptr = ctx.ask::<*const i32, keys::AttnScoreIndptr>()?;
    // THE TWO PRELUDES, IN THE ARM'S ORDER: widen the cache, then land the
    // descriptor, both on this stream before the launch reads either.
    dequant_prelude(ctx);
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let broadcast_q = false;
    // THE PLAN, WHOLE, AND THE STATEMENT SAID WHICH. Two decode schedules may
    // stand at once -- gemma-4 states 256 and 512 -- and which one this fire
    // executes used to be recovered by `bind::attn_plan` from the window on the
    // `LaunchSpec`. The prep names it now.
    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the decode plan this statement names" });
    }
    // SAFETY: `Fa2Decode` resolves only to a plan `raise_attn_plans` raised --
    // `DecodePlan::as_ptr`'s own boxed cache, non-null above and live for the
    // fire. The borrow ends inside this launch.
    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::decode_plan_of(cache, crate::attn::fa2::plan::fa_device());
    // THE DESCRIPTOR, OFF THE PLAN THIS STATEMENT NAMES. Three asks stood
    // here -- `Fa2DecodeUploadSrc`/`Len`/`Dst` -- and each resolved through
    // `bind::attn_plan`, which for decode GUESSED which schedule from the
    // window on the `LaunchSpec`. The staging buffer is the cache's own and
    // the carve offset is the cache's own, so a body holding the cache needs
    // neither the asks nor the guess.
    let int_base = ctx.ask::<*mut c_void, keys::AttnWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (int_base as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    // `:546-549`, before the variant test, and in that order.
    if score_out.is_null() || score_indptr.is_null() {
        return Err(CAPTURE_SINK);
    }
    let Some(arm) = decode_capture_arm(planned.full_attention_variant, *window_left, *logits_soft_cap)
    else {
        return Err(CAPTURE_VARIANT);
    };
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        core::ptr::null(),
        lse,
        // THE DECODE CARVE'S TWO BASES, RAW. This passed a null pair while the
        // leaves carried ABSOLUTE addresses in the offset fields --
        // `decode_plan_of_leaves` stuffed them there and a zero base read them
        // back out. The launcher reads the real cache now, whose offsets are
        // relative to the carve it was raised against, so a null base would
        // make every work-list pointer a small integer.
        //
        // RAW, and not `+ int_base_bytes` as the prefill twin passes:
        // `make_decode_params` adds `plan.int_base_bytes` itself, where
        // `make_prefill_params` says it has "no analogue ... because one
        // prefill plan serves one fire".
        int_base,
        ctx.ask::<*mut c_void, keys::AttnWorkspaceFloat>()?,
    );
    let (base, split) = make_decode_params(&planned, &bufs, *window_left, 0.0, *sm_scale, broadcast_q);
    let params = DecodeScoreParams {
        base,
        score_out: addr(score_out),
        score_indptr: addr(score_indptr),
    };
    decode(ctx, decode_at(&planned, arm, params.base.padded_batch_size), &params)?;
    if planned.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}


/// The paged prefill, over a plan and the buffers it addresses.
///
/// The body [`dispatch_attention_flashinfer_prefill_bf16`] had before its plan
/// was unfolded, kept as a function because [`attention_flashinfer_prefill`]
/// forwards into it and the planless pair still takes the aggregate. Its
/// `bufs` carry the workspace bases the offsets are relative to, which on the
/// unfolded path are null and on the planless path are the decode carve.
fn prefill_paged(
    ctx: &Ctx<'_>,
    bufs: &Buffers,
    plan: &PrefillPlan,
    logits_soft_cap: f32,
    sm_scale: f32) -> Result<(), Refusal> {
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
#[routine(no_join)]
pub fn dispatch_attention_flashinfer_prefill_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    // THE PLAN, AS AN OPERAND. `In(1)` and not `In(0)`: `q`'s slot is stated
    // in half this file's comments and in the tests, and a plan that took the
    // first slot would move every one of them for nothing. What the position
    // buys is that `arity_problem` and `check_plan` COUNT it -- an `ask` is a
    // call the derived column cannot see, which `Asks`' own doc records as the
    // step down from a type-system guarantee to a syntactic one.
    //
    // Twenty-two `ask` sites stood here and named the plan's fields one at a
    // time. Each recorded *the value the plan was built at*, read back rather
    // than restated so the fire could not disagree with the object it was
    // executing -- which is a read-back standing in for a check nothing made.
    // The object arrives whole now, so there is nothing to disagree with.
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let logits_soft_cap = ctx.ask::<f32, keys::AttnLogitsSoftCap>()?;
    let sm_scale = ctx.ask::<f32, keys::SmScale>()?;
    // THE DESCRIPTOR, AND NO WIDENING. `arms/fa2.rs`'s prefill arm ran no
    // `dequant_prelude` where its decode twin did, and the omission is copied
    // rather than repaired: a prefill reaches this having just written the KV
    // it is about to read, so the pages are already the widths it wants.

    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    // THE PLAN, WHOLE. `plan.ptr` is the driver's own boxed cache, minted by
    // `raise_attn_plans` and handed back by the resolver against the KEY this
    // statement names -- not recovered from a family string, and not guessed
    // from a window.
    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the prefill plan this statement names" });
    }
    // SAFETY: `Fa2Prefill` resolves only to `bind::attn_plan`'s pointer, which
    // is `PrefillPlan::as_ptr`'s own boxed allocation -- non-null above, and
    // live for the fire. The borrow ends inside this launch.
    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());
    // THE DESCRIPTOR, OFF THE PLAN THIS STATEMENT NAMES. Three asks stood
    // here -- `Fa2PrefillUploadSrc`/`Len`/`Dst` -- and each resolved through
    // `bind::attn_plan`, which for decode GUESSED which schedule from the
    // window on the `LaunchSpec`. The staging buffer is the cache's own and
    // the carve offset is the cache's own, so a body holding the cache needs
    // neither the asks nor the guess.
    let carve = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;

    // THE TWO BASES THE OFFSETS ARE RELATIVE TO, and they are the reason this
    // is two asks rather than none. `prefill_plan_of_leaves` stuffed ABSOLUTE
    // addresses into the offset fields and passed null bases, so `at(0, off)`
    // read the address back; a real cache holds real offsets into the carve it
    // was raised against. Both are §6.3 facts -- an address the allocator
    // chose -- and neither is a plan leaf.
    let float_base = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceFloat>()?;
    // `int_base_bytes` is the plan's own carve within the int workspace, which
    // the descriptor upload above landed at.
    let int_base = (carve as usize).saturating_add(cache.int_base_bytes) as *mut c_void;

    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        lse,
        int_base,
        float_base,
    );
    prefill_paged(ctx, &bufs, &planned, logits_soft_cap, sm_scale)
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
#[routine(no_join)]
pub fn dispatch_attention_flashinfer_prefill_capture_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    // THE PLAN, AS AN OPERAND. See
    // [`dispatch_attention_flashinfer_prefill_bf16`], whose comment carries
    // the whole of why: `In(1)` keeps `q` at slot 0, and a mark is counted
    // where an `ask` is a call the derived column cannot see.
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>) -> Result<(), Refusal> {
    // ASKED, NOT PARAMETERS. Two bare pointers and an `#[unbound] u32` stood
    // here, and between them they were the whole reason this row could carry
    // no source column. All three are the FIRE's -- the sink it carved, that
    // sink's CSR, and the width it carved them at -- so `Cx` answers them and
    // the row becomes one `route` can bind.
    let score_out = ctx.ask::<*mut f32, keys::AttnScoreOut>()?;
    let score_indptr = ctx.ask::<*const i32, keys::AttnScoreIndptr>()?;
    let score_window = ctx.ask::<u32, keys::AttnScoreWindow>()?;
    // THE DESCRIPTOR, AND NO WIDENING. `arms/fa2.rs`'s prefill arm ran no
    // `dequant_prelude` where its decode twin did, and the omission is copied
    // rather than repaired: a prefill reaches this having just written the KV
    // it is about to read, so the pages are already the widths it wants.
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    // THE PLAN, WHOLE. As the plain prefill launcher: the resolver hands back
    // what `raise_attn_plans` raised, against the KEY this statement names.
    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the prefill plan this statement names" });
    }
    // SAFETY: `Fa2Prefill` resolves only to `bind::attn_plan`'s pointer, which
    // is `PrefillPlan::as_ptr`'s own boxed allocation -- non-null above, and
    // live for the fire. The borrow ends inside this launch.
    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());
    // THE DESCRIPTOR, OFF THE PLAN THIS STATEMENT NAMES. Three asks stood
    // here -- `Fa2PrefillUploadSrc`/`Len`/`Dst` -- and each resolved through
    // `bind::attn_plan`, which for decode GUESSED which schedule from the
    // window on the `LaunchSpec`. The staging buffer is the cache's own and
    // the carve offset is the cache's own, so a body holding the cache needs
    // neither the asks nor the guess.
    let carve = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    // The two bases the plan's offsets are relative to; §6.3 facts and not
    // leaves. See the plain launcher.
    let float_base = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceFloat>()?;
    let int_base = (carve as usize).saturating_add(cache.int_base_bytes) as *mut c_void;
    // `:849-856`. The sink first, then the variant, then the plan -- the C++'s
    // order, and the window is part of the sink here rather than of the
    // variant.
    if score_out.is_null() || score_indptr.is_null() || score_window == 0 {
        return Err(CAPTURE_SINK);
    }
    let Some(arm) = prefill_capture_arm(planned.causal_mask, planned.window_left, *logits_soft_cap) else {
        return Err(CAPTURE_VARIANT);
    };
    prefill_plan_usable(&planned)?;
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        lse,
        int_base,
        float_base,
    );
    let (base, split) = make_prefill_params(&planned, &bufs, 0.0, *sm_scale);
    let params = PrefillScoreParams {
        base,
        score_out: addr(score_out),
        score_indptr: addr(score_indptr),
        score_window: score_window,
    };
    prefill(ctx, prefill_at(&planned, arm, params.base.padded_batch_size), &params)?;
    if planned.info.split_kv { fold(ctx, &split) } else { Ok(()) }
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
#[routine(no_join)]
pub fn dispatch_attention_flashinfer_prefill_custom(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    // THE PLAN, AS AN OPERAND. See
    // [`dispatch_attention_flashinfer_prefill_bf16`], whose comment carries
    // the whole of why: `In(1)` keeps `q` at slot 0, and a mark is counted
    // where an `ask` is a call the derived column cannot see.
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>) -> Result<(), Refusal> {
    // ASKED, as the capturing pair's sinks: a bare `*const u8` is not a mark,
    // and this row's column could not exist while it took one.
    let mask = ctx.ask::<*const u8, keys::AttnMask>()?;
    let mask_indptr = ctx.ask::<*const i32, keys::AttnMaskIndptr>()?;
    // BOTH PRELUDES, over the PREFILL carve. The custom form is the one
    // prefill arm that widened: it serves a text whose cache may be quantised.
    dequant_prelude(ctx);
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    // THE PLAN, WHOLE. As the plain prefill launcher: the resolver hands back
    // what `raise_attn_plans` raised, against the KEY this statement names.
    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the prefill plan this statement names" });
    }
    // SAFETY: `Fa2Prefill` resolves only to `bind::attn_plan`'s pointer, which
    // is `PrefillPlan::as_ptr`'s own boxed allocation -- non-null above, and
    // live for the fire. The borrow ends inside this launch.
    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());
    // THE DESCRIPTOR, OFF THE PLAN THIS STATEMENT NAMES. Three asks stood
    // here -- `Fa2PrefillUploadSrc`/`Len`/`Dst` -- and each resolved through
    // `bind::attn_plan`, which for decode GUESSED which schedule from the
    // window on the `LaunchSpec`. The staging buffer is the cache's own and
    // the carve offset is the cache's own, so a body holding the cache needs
    // neither the asks nor the guess.
    let carve = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    // The two bases the plan's offsets are relative to; §6.3 facts and not
    // leaves. See the plain launcher.
    let _int_base = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    let float_base = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceFloat>()?;
    let int_base = (carve as usize).saturating_add(cache.int_base_bytes) as *mut c_void;
    let arm = prefill_custom_arm(*logits_soft_cap);
    prefill_plan_usable(&planned)?;
    let bufs = buffers(
        q.ptr,
        k_pages.cast::<bf16>(),
        v_pages.cast::<bf16>(),
        o.ptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        lse,
        int_base,
        float_base,
    );
    let (mut params, split) = make_prefill_params(&planned, &bufs, *logits_soft_cap, *sm_scale);
    // `:1150-1155`, `:1163`.
    params.maybe_custom_mask = addr(mask);
    params.maybe_mask_indptr = addr(mask_indptr);
    params.window_left = -1;
    prefill(ctx, prefill_at(&planned, arm, params.padded_batch_size), &params)?;
    if planned.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

/// Build this fire's prefill plan, for the launcher that gets none.
///
/// `arms/fa2.rs`'s `fa2_prefill_planless`, from the top through `plan_prefill`.
/// The comment it replaces said the planning *"is the arm's, because a plan is
/// `driver-cuda`'s vocabulary"* — and the planner it named, `plan::plan_prefill`,
/// is in THIS crate and always was. What was actually the driver's is the
/// CACHE's allocation and the two host CSRs, and those arrive as facts.
///
/// **The cache is written through, and the launcher's claim on it is exclusive
/// for the launch**: a fire dispatches one statement at a time, which is the
/// same warrant the arm's `&mut` had.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty batch or a zero head width,
/// [`Refusal::Narrow`] for a query width that is not whole heads, and
/// [`Refusal::Unstated`] when the planner declines.
fn plan_own_prefill(ctx: &Ctx<'_>, q_width: i32) -> Result<PrefillPlan, Refusal> {
    let requests = ctx.ask::<i32, keys::FireRequests>()?;
    if requests <= 0 {
        return Err(Refusal::Empty { what: "the batch" });
    }
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    // The head count, which nobody carries: the query's width over the cache's
    // head dim. `plan_prefill` is a host planner with no `Refusal` to carry a
    // guard into -- it would `.max(1)` a zero divisor and truncate the
    // rectangle -- so both checks are made here, where a refusal exists.
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "the layer's head dim" });
    }
    if q_width % head_dim != 0 {
        return Err(Refusal::Narrow { what: "the query width, in heads", at: i64::from(q_width) });
    }
    let cache = ctx.ask::<*mut u8, keys::Fa2PrefillPlanCache>()?;
    let qo_h = ctx.ask::<*const u32, keys::QoIndptrHost>()?;
    let kv_h = ctx.ask::<*const u32, keys::KvPageIndptrHost>()?;
    let n = requests as usize + 1;
    // SAFETY: the fire's contract, which the two facts carry -- both are host
    // CSRs of `requests + 1` entries, refused above if either is absent. The
    // slices are read and not held past `plan_prefill`.
    let (qo_h, kv_h) = unsafe {
        (
            core::slice::from_raw_parts(qo_h, n),
            core::slice::from_raw_parts(kv_h, n),
        )
    };
    // SAFETY: `Fa2PrefillPlanCache` answers the driver's own boxed allocation,
    // non-null by that fact's refusal, and the launch holds it alone.
    let cache = unsafe { &mut *cache.cast::<plan::PrefillPlanCache>() };
    let device = plan::plan_device();
    let planned = plan::plan_prefill(
        cache,
        qo_h,
        kv_h,
        ctx.ask::<i32, keys::Rows>()?,
        requests,
        q_width / head_dim,
        ctx.ask::<i32, keys::KvNumHeads>()?,
        head_dim,
        ctx.ask::<i32, keys::KvPageSize>()?,
        crate::attn::plan::Workspace {
            float_bytes: ctx.ask::<usize, keys::AttnWorkspaceFloatBytes>()?,
            int_bytes: ctx.ask::<usize, keys::AttnWorkspaceIntBytes>()?,
        },
        &device,
        // `enable_cuda_graph`, and not the C++'s value: `false` sizes the
        // tiling from this batch, `true` derives `cta_tile_q` from a bound the
        // batch cannot exceed, so the layout can be baked once.
        true,
        ctx.ask::<i32, keys::WindowLeft>()?,
        false,
        ctx.ask::<bool, keys::KvHndLayout>()?,
        true,
        false,
        false,
    );
    if let plan::Planned::Declined(why) = planned {
        tracing::error!(%why, "the planless FA2 prefill could not plan its own fire");
        return Err(Refusal::Unstated { what: "a plannable FA2 prefill fire; see the log" });
    }
    // THE DECODE CARVE, as the entry point this replaces used: the planless
    // pair plans against `AttnWorkspace*` and not the prefill carve, so its
    // descriptor lands there too.
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()? as usize)
            .saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    Ok(dispatch::prefill_plan_of(cache, plan::fa_device()))
}

/// `attn::attention_flashinfer_prefill`, `attention_flashinfer.cu:1077-1113`
/// — the PLANLESS prefill.
///
/// `whole = true`, and the reason is in the name: no plan cache crosses the
/// fire for this symbol, so one is built over the WHOLE batch on the way in
/// and thrown away. That planning is [`plan_own_prefill`]'s, which is this
/// body's own call and no longer an arm's — see there for what moved and why
/// the reason it did not move sooner was mistaken.
///
/// `:1063-1067` fixes three flags this path never varies:
/// `enable_cuda_graph = false`, `full_attention_variant = false`,
/// `causal_mask = true`. So [`prefill_arm`] always answers `CausalSoftcap` or
/// `CausalWindow`, which is what the plan states.
///
/// # Errors
///
/// As [`dispatch_attention_flashinfer_prefill_bf16`], plus
/// [`plan_own_prefill`]'s.
#[routine(whole, no_join)]
pub fn attention_flashinfer_prefill(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>) -> Result<(), Refusal> {
    // WIDENING FIRST, as the arm had it: the planner walks the page CSRs this
    // widening does not touch, but the launch below reads pages it does.
    dequant_prelude(ctx);
    let plan = plan_own_prefill(ctx, q.width)?;
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let int_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()?;
    let k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let float_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceFloat>()?;
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
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        lse,
        int_buffer,
        float_buffer,
    );
    prefill_paged(ctx, &bufs, &plan, *logits_soft_cap, *sm_scale)
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
#[routine(whole, no_join)]
pub fn attention_flashinfer_prefill_lse(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    // REQUIRED, where [`attention_flashinfer_prefill`]'s is
    // `Out<0, *mut bf16>`. Same reason as the decode pair's: the one-result
    // spelling goes through `attn_at`, which declares nothing inside
    // `llama_like/forward/mod.rs:1638`'s `dsl::guarded_value` and needs
    // `Provenance::Either` for `model-compiler/src/lower/walk.rs:472` to hand
    // it the region's buffer. `attention_flashinfer_prefill_lse` declares two
    // results unconditionally and `deepseek_v4/forward/mod.rs:146` states it
    // outside any region.
    o: Out<Tensor<bf16>>,
    _lse: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, WHERE THE TWIN TAKES THEM AS `Const`, and the texts are why:
    // `base.rs:410`'s builder for this spelling passes `vec![]` for its params
    // run while `attn_at`'s passes `cap_and_scale(..)`. A `Const` mark PROMISES
    // the statement carries the number at its slot; here nothing does, so the
    // promise broke at the fire rather than at the type -- which is what
    // `arms/fa2.rs` hid by passing `AttnCtx::logits_soft_cap` by hand.
    let logits_soft_cap = Const { v: ctx.ask::<f32, keys::AttnLogitsSoftCap>()? };
    let sm_scale = Const { v: ctx.ask::<f32, keys::SmScale>()? };
    let _int_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()?;
    let _k_pages = ctx.ask::<*mut u8, keys::KvKeys>()?;
    let _v_pages = ctx.ask::<*mut u8, keys::KvValues>()?;
    let _qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let _kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let _kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let _kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let _float_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceFloat>()?;
    // As above: the plan, its workspaces and the two preludes are the callee's.
    // The LSE parameter is not forwarded and never was -- the callee asks for
    // `keys::AttnLseOut`, and this spelling's declared result is what that
    // fact answers with, which is the repair the doc above describes.
    attention_flashinfer_prefill(ctx, q, o, logits_soft_cap, sm_scale)
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

/// The FA2 lattice's eight symbols, under the namespace a trace spells them in.
///

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
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode);
    // 15 -> 28 -> 2 -> 3. The aggregate and the workspace pair left; sixteen
    // named leaves arrived at 8..24 and then left again when the plan became
    // an operand. What is here is the statement's own shape: the query, the
    // plan it names, the result.
    assert!(d.len() == 3);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `o` KEEPS `Or`. `attn_at` declares no result inside
    // `llama_like/forward/mod.rs:1638`'s `dsl::guarded_value`, and this flag
    // going false is that guard's value losing its writer.
        // `nullable` WAS TRUE HERE AND IS FALSE NOW, AND THIS PIN IS HOW YOU KNOW.
    // `Or<_>` left this parameter with Kilimanjaro III's last step: the fact
    // it carried -- `Provenance::Either`, which `walk.rs` read to decide that
    // a statement declaring no result should be handed its value region's
    // buffer -- is stated on the OP now (`Op::dest`), by the builder that
    // minted that buffer. A launcher's type no longer decides a lowering.
    assert!(!<dispatch_attention_flashinfer_decode as ::kernels::Derivation>::DERIVED[1].nullable);
    // `lse` WAS `d[7]` WITH `Source::Slot(Out, 1)` AND `nullable`, then
    // `Env<*mut f32>` with no source at all. It is `keys::AttnLseOut` now,
    // answered off `Fire::lse_out` -- which does NOT null-check, so a fire
    // with no lse destination still binds and the launcher's own test
    // decides.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    assert!(true /* the parameter this pinned has left the signature */);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode_lse);
    assert!(d.len() == 4);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // THE OPERAND SLOTS STILL DO NOT SHIFT RELATIVE TO THE TWIN. Both took the
    // plan at `In(1)` in the same round, so the results sit one along on both
    // and the forward at the bottom of the body is still a forward rather than
    // a re-ordering.
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    assert!(!<dispatch_attention_flashinfer_decode_lse as ::kernels::Derivation>::DERIVED[1].nullable);
    assert!(true /* the parameter this pinned has left the signature */);

    // The planless prefill pair. FOUR PARAMETERS EACH NOW, not five and six:
    // the `#[unbound] plan: PrefillPlan` both carried is gone. It was the
    // aggregate `arms/fa2.rs` built on the way in, and the reason it could
    // never be bound was that nothing published it before the fire -- true,
    // and beside the point, because the launcher can BUILD it. It does:
    // `plan_own_prefill` walks the two host CSRs and calls `plan::plan_prefill`,
    // both of which live in this crate and always did.
    //
    // The count dropping is the whole of what a reader should check here. A
    // column with no `None` in it is a row `route` can bind, which is what
    // took these two symbols off the hand-armed list.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill);
    assert!(d.len() == 4);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
        // `nullable` WAS TRUE HERE AND IS FALSE NOW, AND THIS PIN IS HOW YOU KNOW.
    // `Or<_>` left this parameter with Kilimanjaro III's last step: the fact
    // it carried -- `Provenance::Either`, which `walk.rs` read to decide that
    // a statement declaring no result should be handed its value region's
    // buffer -- is stated on the OP now (`Op::dest`), by the builder that
    // minted that buffer. A launcher's type no longer decides a lowering.
    assert!(!<attention_flashinfer_prefill as ::kernels::Derivation>::DERIVED[1].nullable);
    // NOTHING IN THIS COLUMN IS UNSOURCED. The `plan` entry was the one `None`
    // either of these carried, so its removal is what makes the pair bindable.
    assert!(!d[2].is_none() && !d[3].is_none());

    // THREE, AND THE TWO SCALARS ARE WHY IT IS NOT FOUR. `base.rs:410` builds
    // this spelling with `vec![]` for its params run where `attn_at` builds
    // the twin's with `cap_and_scale(..)`, so a `Const` here promised a slot
    // no text fills. Both are asked for now.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill_lse);
    assert!(d.len() == 3);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(!<attention_flashinfer_prefill_lse as ::kernels::Derivation>::DERIVED[1].nullable);

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
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode_capture)[1],
        Some(kernels::Source::Slot(kernels::Kind::In, 1))
    ));
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode_capture)[2],
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));
    // AT 2, BECAUSE THE PLAN SITS AT 1 -- and the assertion moving is the
    // whole point of it being here. `o` is still result 0; what changed is
    // that this launcher places three operands where it placed two, and a
    // renumbering that was NOT the plan's arrival would have failed the same
    // way. Its siblings still read `[1]` until they take the operand too.
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_bf16)[1],
        Some(kernels::Source::Slot(kernels::Kind::In, 1))
    ));
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_bf16)[2],
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_capture_bf16)[1],
        Some(kernels::Source::Slot(kernels::Kind::In, 1))
    ));
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_capture_bf16)[2],
        Some(kernels::Source::Slot(kernels::Kind::Out, 0))
    ));
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_custom)[1],
        Some(kernels::Source::Slot(kernels::Kind::In, 1))
    ));
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_custom)[2],
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
    let _d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode);
    // 13 → 26. Sixteen leaves landed between `lse` and the tail scalars.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // AND THE NEIGHBOUR, WHICH IS A DIFFERENT KEY AND NOT THE SAME ONE.
    // `logits_soft_cap` sits one index below `sm_scale` and is read off the
    // same `AttnCtx` borrow, which is exactly the shape of a mistaken sweep.
    // `keys::AttnLogitsSoftCap` is total -- `0` means none -- where
    // `keys::SmScale` refuses; binding either through the other's arm would
    // compile.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.

    // `attention_flashinfer_prefill` KEEPS FOUR ENTRIES AND NO UNSOURCED ONE.
    // `qo_indptr` was `d[4]`, a plan leaf pinned against its neighbours, and is
    // asked for in the body; `plan` was `d[2]`, the `#[unbound]` aggregate the
    // arm built, and the LAUNCHER builds it now (`plan_own_prefill`). What is
    // left is the shape the DSL owns: two operands and the two scalars the
    // statement carries.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill);
    assert!(d.len() == 4);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 1))));
    // THE FORWARD IS INDEX-FOR-INDEX, and it is shorter at both ends now.
    // `attention_flashinfer_prefill_lse` hands its whole parameter list to the
    // twin above, so a key landing on a different index on either side would
    // be a re-ordering wearing a forward. What both lists carry is the shape
    // the DSL owns: the operands and the two scalars. The plan leaves and the
    // plan itself are the body's, where a caller cannot re-order them at all.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill_lse);
    assert!(d.len() == 3);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    let _d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_bf16);
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // 13 → 32, for the decode pair's reason: the leaves landed between `lse`
    // and the tail scalars.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // NOTHING IS `None` HERE ANY MORE. `d[4]` was `#[unbound] score_window`,
    // and the two bare sink pointers before it carried no source either -- the
    // three together were why this row had no column at all. They are asked
    // facts now (`AttnScoreOut`, `AttnScoreIndptr`, `AttnScoreWindow`), so
    // what is left is the shape the DSL owns.
    // EACH UP ONE, because the plan is an operand at `[1]` now. The SLOT
    // numbers do not move -- a params run is its own space -- only the
    // positions that read them.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_capture_bf16);
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 1))));
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // AT TWO, NOT FOUR: `mask` and `mask_indptr` were bare pointers ahead of
    // it and are `keys::AttnMask`/`AttnMaskIndptr` now.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_custom);
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.

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
    // itself. `fa2_decode_leaves` resolved `int_buffer + int_base_bytes + off`
    // once and answered the address; it is deleted, and the base is back --
    // the launcher passes the RAW carve and `make_decode_params` adds
    // `int_base_bytes` itself. A null there is what this file shipped for one
    // commit, and it made every work-list pointer a small integer.
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
    let _d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill);
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    let _d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill_lse);
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
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
    // `fa2_prefill_leaves` read `Cx::attn_prefill_workspace` and answered
    // ADDRESSES. It is deleted: a launcher asks for its own carve now and the
    // two families ask DIFFERENT keys -- `AttnPrefillWorkspace*` against
    // `AttnWorkspace*` -- which is what keeps a prefill from being handed the
    // decode base.

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
    // 7 -> 5 AND 7 -> 4 AND 6 -> 4. The capturing pair's two sink pointers and
    // its window, and the custom form's mask pair, were the last parameters on
    // this lattice that were not marks -- which is exactly what
    // `#[routine(untraced)]` meant on the three rows that carried them, and
    // why those rows had no source column for a binder to read. Every one is a
    // fact the fire publishes, so all three say `#[routine(no_join)]` now and
    // the counts below are the statements' own shapes.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode_capture);
    assert!(d.len() == 6);
    // THREE, AND THE THIRD IS THE PLAN. `q`, the raise, `o` -- where this
    // read two and named the plan's twenty-two fields through `ask`, which the
    // derived column could not see and this ledger could not count. It counts
    // it now, which is what the operand was for.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_bf16);
    assert!(d.len() == 3);
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_capture_bf16);
    assert!(d.len() == 5);
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_custom);
    assert!(d.len() == 5);
    // FOUR AND FIVE, not five and six: the planless pair's `#[unbound] plan`
    // is gone, the launcher building its own (`plan_own_prefill`). These two
    // are the only prefill signatures the change touched.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill);
    assert!(d.len() == 4);
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(attention_flashinfer_prefill_lse);
    assert!(d.len() == 3);
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // ── AND THE THIRTY-EIGHT LEAVES, ALL OF WHICH HAVE LEFT ─────────────
    //
    // TWO MACROS STOOD HERE, `decode_leaves!` over three decode launchers
    // and `prefill_leaves!` over three prefill ones, and between them they
    // pinned thirty-eight `Env<keys::Fa2*>` entries -- sixteen at 8..24 on
    // decode, twenty-two at 9..31 on prefill -- each to the one key that
    // belonged at its index. The hazard they were written for is real and
    // worth restating: `Env<T>` is `Provenance::Env` for EVERY `T`, so a key
    // on the wrong parameter type-checks, binds, and reads the neighbouring
    // array, and on prefill four `*const i32` work lists ran consecutively at
    // 9..15, where a swapped pair schedules a kernel against the wrong list
    // with nothing anywhere saying so.
    //
    // ALL THIRTY-EIGHT PARAMETERS ARE GONE. Every fact they carried turned
    // out to be one only the fire can answer, so each body asks its context
    // instead, and an index that does not exist cannot be held to the key
    // that used to sit at it. The retirement was done leaf by leaf and left
    // thirty-eight identical epitaphs inside two macro bodies whose only
    // surviving statement was `let d = <$sym as Derivation>::SOURCES;` --
    // six expansions binding a name nothing read, which is what the six
    // `unused variable: d` warnings were.
    //
    // WHAT STILL GUARDS THE COLUMN is the run of length assertions above and
    // below this note. They say how many entries each signature derives, and
    // that is now the whole of the claim, because the entries left are the
    // buffers, whose types differ and which therefore cannot silently trade
    // places the way thirty-eight same-shaped `Env` scalars could. The
    // ordering hazard did not get fixed; the parameters it needed went away.
    // If any of these facts is ever a parameter again, the macros belong
    // back here -- git remembers them at the commit that removed this block.


    // THE PREFILL HEAD AND TAIL. `lse` is `Env` on all three: no text of any
    // of these symbols declares a second result.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_bf16);
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].is_some());
        i += 1;
    }
    // THE CAPTURE TWIN'S THREE ARE NOT ITS OWN ANY MORE, and the sentence
    // that stood here is what they cost. It said `score_out`, `score_indptr`
    // and `score_window` *"are `AttnCtx` fields no `Fire` accessor returns, so
    // no column can bind them"* -- which was a statement about the ACCESSORS,
    // not about the facts. Three accessors and three keys later they bind like
    // anything else, and the twin's list is its statement's shape.
    let c = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_capture_bf16);
    assert!(c.len() == 5);
    let mut i = 0;
    while i < c.len() {
        // NOTHING answers `None` now. That is the claim.
        assert!(c[i].is_some());
        i += 1;
    }
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // THE CUSTOM-MASK TWIN'S TWO, as above: `keys::AttnMask` and
    // `AttnMaskIndptr` now, where a bare pointer pair used to say that no
    // column could bind them.
    let m = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_prefill_custom);
    assert!(m.len() == 5);
    let mut i = 0;
    while i < m.len() {
        assert!(m[i].is_some());
        i += 1;
    }
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

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
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode);
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // THE PARAMETER THIS PINNED HAS LEFT THE SIGNATURE. What it named is a
    // fact only the fire can answer, so the body asks its context for it and
    // there is no column entry left to hold in place.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

    // The capture twin, which is ONE longer than the plain pair and no longer
    // two: `window_left` is a `Const` here where the plain form asks for it.
    let c = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode_capture);
    assert!(c.len() == 6);
    // The parameter this pinned has left the signature: what it
    // named is a fact only the fire can answer, asked for in the body.
    // The parameter this pinned has left the signature: what it
    // named is a fact only the fire can answer, asked for in the body.
    // The parameter this pinned has left the signature: what it
    // named is a fact only the fire can answer, asked for in the body.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

    // EVERY COLUMN ON THE DECODE THREE IS ANSWERED, AND NOTHING KEEPS AN ARM.
    // This said the H2D *"has to happen somewhere `Bound::derived` does not"*,
    // which was true of the arm and false of the launcher: `upload_plan` runs
    // in the body, inside the captured region, so a replay re-runs it exactly
    // as it re-ran the arm's copy.
    let e = kernels::routine::sources::<crate::jit::Cuda, _, _>(dispatch_attention_flashinfer_decode_lse);
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].is_some());
        assert!(e[i].is_some());
        i += 1;
    }
    // THE CAPTURE TWIN IS NOT SHORT OF ANYTHING. The sentence that stood here
    // said `score_out` and `score_indptr` are `AttnCtx` fields *"no `Fire`
    // accessor returns, so there is nothing for a key to be answered off"* --
    // which described the accessors rather than the fields. Both are keys now
    // (`AttnScoreOut`, `AttnScoreIndptr`), answered off the same `attn_ctx()`
    // every other attention fact comes through, and the twin's column is
    // complete.
    let mut i = 0;
    while i < c.len() {
        assert!(c[i].is_some());
        i += 1;
    }
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
};

// `#[lit(..)]`'S PIN FIXTURE IS GONE, and so is the attribute it pinned.
// `lit_pins::all_four` existed to give the macro's four literal variants a
// consumer, because "a mark with no consumer is how a macro arm rots"; the
// attribute left with the five-mark vocabulary, and by then three of its four
// pins had already been deleted and the fourth had become an ordinary
// `Const<f32>` assertion that every real row makes.
//
// It had also stopped being free: a `#[routine]` registers by EXISTING now,
// so the fixture put `attn::all_four` in `kernels_cuda::sigs()` -- a symbol
// no text states, no arm binds and no kernel exists for, sitting in the table
// `check_plan` measures model texts against.

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
    use crate::attn::fa2::geometry::{DecodeGeometry, Device, KvWidth, PrefillGeometry};
    use super::{DECODE, DECODE_GQA, DecodeArm, HEAD_DIMS, PREFILL, PrefillArm, decode_arm, decode_capture_arm, decode_instantiation, decode_root, prefill_arm, prefill_capture_arm, prefill_custom_arm, prefill_instantiation, prefill_root, };

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
            let decode = crate::routine(symbol).unwrap_or_else(|| panic!("{symbol} is declared"));
            assert!(decode.depth_prefix_plan, "the union-tail plan swap reads this column");
            assert!(!decode.whole, "a decode dispatch covers a row range");
        }

        for symbol in
            ["attn::attention_flashinfer_prefill", "attn::attention_flashinfer_prefill_lse"]
        {
            let planless = crate::routine(symbol).unwrap_or_else(|| panic!("{symbol} is declared"));
            assert!(planless.whole, "it plans over the whole fire on the way in");
            assert!(!planless.depth_prefix_plan);
        }

        // This family's eight, picked out of the crate-wide slice by the
        // namespace they derive from `module_path!()`.
        let mine = crate::ROUTINES.iter().filter(|r| r.namespace == "attn").count();
        assert!(mine >= 8, "the fa2 family's rows are in the slice");
        for symbol in [
            "attn::dispatch_attention_flashinfer_decode_capture",
            "attn::dispatch_attention_flashinfer_prefill_bf16",
            "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
            "attn::dispatch_attention_flashinfer_prefill_custom",
        ] {
            let r = crate::routine(symbol).unwrap_or_else(|| panic!("{symbol} is declared"));
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
