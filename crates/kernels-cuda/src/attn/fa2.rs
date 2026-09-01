//! The fa2 plane: FlashInfer's decode/prefill kernels, fired one parameter
//! block at a time. The instantiation a fire resolves is *derived*, not
//! tabulated — [`decode_symbol`]/[`prefill_symbol`] spell the template
//! arguments from the same [`DecodeGeometry`]/[`PrefillGeometry`] that size
//! the launch, so the name NVRTC lowers and the smem/block geometry it is
//! fired at cannot drift apart. Selection lives here, below the entries
//! (decision #13) — a dispatch arm never sees a lattice point.

use crate::error::Error;

use crate::attn::fa2_abi::{DecodeParams, Partials, PrefillPagedParams};
use crate::attn::plan::Device;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse, symbol};

pub const FILE: &str = "attn/attention.cuh";

/// The decode variants an instantiation can be stamped with. The capture
/// arms are unreached until a graph-capture consumer exists; they keep
/// their spelling here so that consumer names an arm, not a string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeArm {
    Full,
    Softcap,
    Window,
    CaptureFull,
    CaptureWindow,
}

/// The prefill variants, same bargain as [`DecodeArm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillArm {
    CausalFullSoftcap,
    NoneFullSoftcap,
    CausalFull,
    NoneFull,
    CausalSoftcap,
    CausalWindow,
    CausalCapture,
    NoneCapture,
    CustomSoftcap,
    Custom,
}

/// One fa2 head width outside the stamped lattice, refused before NVRTC
/// ever sees a name for it. The geometry derivations bound most shapes on
/// their own; this is the belt over those braces.
fn instantiated(op: &'static str, head_dim: u32) -> Result<(), Error> {
    if crate::attn::plan::head_dim_instantiated(head_dim) {
        return Ok(());
    }
    Err(refuse(
        op,
        format!("no fa2 unit is stamped at head width {head_dim}; the lattice holds 64/128/256/512"),
    ))
}

/// The decode instantiation, spelled from the derived geometry. The old
/// plane carried these as a 20-root static table; every template argument
/// below restates a field [`DecodeGeometry::derive`] computes, so the name
/// NVRTC lowers and the launch shape are one derivation, not two.
///
/// `NUM_STAGES_SMEM` is stamped from `g.num_stages_smem` — the table
/// stamped an unconditional 2 while the smem budget was sized at 1 stage on
/// cc < 8, a latent drift this derivation closes.
fn decode_symbol(
    op: &'static str,
    g: &DecodeGeometry,
    arm: DecodeArm,
) -> Result<&'static str, Error> {
    instantiated(op, g.head_dim)?;
    let (variant, params) = match arm {
        DecodeArm::Full => ("VariantFull", "DecodeParams"),
        DecodeArm::Softcap => ("VariantWindowSoftcap", "DecodeParams"),
        DecodeArm::Window => ("VariantWindow", "DecodeParams"),
        DecodeArm::CaptureFull => ("CaptureFull", "DecodeCaptureParams"),
        DecodeArm::CaptureWindow => ("CaptureWindow", "DecodeCaptureParams"),
    };
    Ok(symbol(&format!(
        "::flashinfer::BatchDecodeWithPagedKVCacheKernel<\
         ::flashinfer::PosEncodingMode::kNone, \
         {ns}, {tile}, {vec}, {bdx}, {bdy}, {bdz}, \
         ::pie::attn::fa2::{variant}, ::pie::attn::fa2::{params}>",
        ns = g.num_stages_smem,
        tile = g.tile_size_per_bdx,
        vec = g.vec_size,
        bdx = g.bdx,
        bdy = g.bdy,
        bdz = g.bdz,
    )))
}

/// The prefill instantiation, spelled from the derived geometry — the old
/// 36-root static table, restated as the derivation it always was.
fn prefill_symbol(
    op: &'static str,
    g: &PrefillGeometry,
    arm: PrefillArm,
) -> Result<&'static str, Error> {
    instantiated(op, g.head_dim)?;
    let (mask, variant, params) = match arm {
        PrefillArm::CausalFullSoftcap => ("kCausal", "VariantFullSoftcap", "PrefillParams"),
        PrefillArm::NoneFullSoftcap => ("kNone", "VariantFullSoftcap", "PrefillParams"),
        PrefillArm::CausalFull => ("kCausal", "VariantFull", "PrefillParams"),
        PrefillArm::NoneFull => ("kNone", "VariantFull", "PrefillParams"),
        PrefillArm::CausalSoftcap => ("kCausal", "VariantWindowSoftcap", "PrefillParams"),
        PrefillArm::CausalWindow => ("kCausal", "VariantWindow", "PrefillParams"),
        PrefillArm::CausalCapture => ("kCausal", "CapturePrefill", "PrefillCaptureParams"),
        PrefillArm::NoneCapture => ("kNone", "CapturePrefill", "PrefillCaptureParams"),
        PrefillArm::CustomSoftcap => ("kCustom", "VariantCustomSoftcap", "PrefillParams"),
        PrefillArm::Custom => ("kCustom", "VariantCustom", "PrefillParams"),
    };
    Ok(symbol(&format!(
        "::flashinfer::BatchPrefillWithPagedKVCacheKernel<\
         ::pie::attn::fa2::PagedTraits<::flashinfer::MaskMode::{mask}, \
         {q}, {mmaq}, {kv}, {dqk}, {dvo}, {wq}, {wkv}, \
         ::pie::attn::fa2::{variant}>, ::pie::attn::fa2::{params}>",
        q = g.cta_tile_q,
        mmaq = g.num_mma_q,
        kv = g.num_mma_kv,
        dqk = g.num_mma_d_qk,
        dvo = g.num_mma_d_vo,
        wq = g.num_warps_q,
        wkv = g.num_warps_kv,
    )))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePoint {
    pub head_dim: u32,
    pub group_size: u32,
    pub arm: DecodeArm,
    pub padded_batch_size: u32,
    pub num_kv_heads: u32,
    pub device: Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPoint {
    pub head_dim: u32,
    pub cta_tile_q: u32,
    pub arm: PrefillArm,
    pub padded_batch_size: u32,
    pub num_kv_heads: u32,
    pub device: Device,
}

/// The whole parameter block as the launch's one argument. The bytes are
/// copied into the pinned slots before `ctx.fire` returns, so the borrow
/// only has to outlive the call.
pub(crate) fn block<P>(params: &P) -> ArgValue {
    ArgValue::Bytes {
        ptr: core::ptr::from_ref(params).cast::<u8>(),
        len: core::mem::size_of::<P>(),
    }
}

pub(crate) fn decode(
    ctx: &Ctx,
    op: &'static str,
    at: DecodePoint,
    params: &DecodeParams,
) -> Result<(), Error> {
    let geometry =
        DecodeGeometry::derive(op, at.head_dim, at.group_size, KvWidth::BF16, &at.device)?;

    ctx.fire(
        op,
        Fire::at(FILE, decode_symbol(op, &geometry, at.arm)?).apply(
            Launch::grid(
                DecodeGeometry::grid(at.padded_batch_size, at.num_kv_heads),
                geometry.block(),
            )
            .smem(geometry.smem_bytes),
        ),
        &[block(params)],
    )
}

pub(crate) fn prefill(
    ctx: &Ctx,
    op: &'static str,
    at: PrefillPoint,
    params: &PrefillPagedParams,
) -> Result<(), Error> {
    let geometry = PrefillGeometry::derive(
        op,
        at.head_dim,
        at.cta_tile_q,
        KvWidth::BF16,
        false,
        &at.device,
    )?;
    ctx.fire(
        op,
        Fire::at(FILE, prefill_symbol(op, &geometry, at.arm)?).apply(
            Launch::grid(
                PrefillGeometry::grid(at.padded_batch_size, at.num_kv_heads),
                geometry.block(),
            )
            .smem(geometry.smem_bytes),
        ),
        &[block(params)],
    )
}

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
    if logits_soft_cap > 0.0 {
        PrefillArm::CausalSoftcap
    } else {
        PrefillArm::CausalWindow
    }
}

#[must_use]
pub fn prefill_custom_arm(logits_soft_cap: f32) -> PrefillArm {
    if logits_soft_cap > 0.0 {
        PrefillArm::CustomSoftcap
    } else {
        PrefillArm::Custom
    }
}

// ── the cascade merge that folds split-kv partials ──────────────────────────

const NUM_THREADS: u32 = 128;

const NUM_SMEM_STAGES: u32 = 4;

#[must_use]
const fn merge_geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

#[must_use]
const fn merge_smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = merge_geometry(head_dim) else {
        return None;
    };
    Some(NUM_SMEM_STAGES * bdy * head_dim * 2 + NUM_THREADS * 4)
}

/// The merge instantiation, spelled from [`merge_geometry`] — the same
/// `<vec, bdx, bdy, stages>` tuple that shapes the launch.
fn merge_varlen_inst(head_dim: u32) -> Option<&'static str> {
    let (vec_size, bdx, bdy) = merge_geometry(head_dim)?;
    Some(symbol(&format!(
        "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
         {vec_size}, {bdx}, {bdy}, {NUM_SMEM_STAGES}, \
         ::pie::attn::merge_lse::DTypeIn, ::pie::attn::merge_lse::DTypeO, \
         ::pie::attn::merge_lse::IdType>",
    )))
}

fn no_merge_row(op: &'static str) -> Error {
    refuse(
        op,
        "no cascade merge is stamped at this head width -- 64, 128, 256 and 512 are here",
    )
}

fn grid_blocks(per_sm: u32, max_seq_len: u32, num_heads: u32, num_sms: u32) -> u32 {
    let work_bound = max_seq_len
        .saturating_mul(num_heads)
        .div_ceil(num_sms)
        .max(1);
    per_sm.min(work_bound).saturating_mul(num_sms).max(num_sms)
}

#[cfg(feature = "_cuda")]
fn merge_blocks_per_sm(instantiation: &'static str, smem: u32) -> u32 {
    use cudarc::driver::sys as dr;

    let Some(root) = crate::jit::Root::of(FILE) else {
        return 1;
    };
    let Ok(resolved) = crate::jit::cache::resolve(&root, instantiation) else {
        return 1;
    };
    let mut blocks: core::ffi::c_int = 0;

    let code = unsafe {
        dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &raw mut blocks,
            resolved.function,
            i32::try_from(NUM_THREADS).unwrap_or(i32::MAX),
            usize::try_from(smem).unwrap_or(usize::MAX),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return 1;
    }
    u32::try_from(blocks).unwrap_or(1).max(1)
}

#[cfg(not(feature = "_cuda"))]
fn merge_blocks_per_sm(_instantiation: &'static str, _smem: u32) -> u32 {
    1
}

/// Folds a split schedule's partial planes into the final output. `Ok` on a
/// non-split plan is a bug in the caller, not here — the entries call this
/// only under `info.split_kv`.
pub(crate) fn fold(ctx: &Ctx, op: &'static str, split: &Partials) -> Result<(), Error> {
    let head_dim = split.head_dim;
    let (_, bdx, bdy) = merge_geometry(head_dim).ok_or_else(|| no_merge_row(op))?;
    let smem = merge_smem_bytes(head_dim).ok_or_else(|| no_merge_row(op))?;
    let instantiation = merge_varlen_inst(head_dim).ok_or_else(|| no_merge_row(op))?;
    for (ptr, which) in [
        (split.tmp_v, "the partial value plane"),
        (split.tmp_s, "the partial state plane"),
        (split.indptr, "the merge indptr"),
        (split.o, "the folded output"),
    ] {
        if ptr == 0 {
            return Err(refuse(op, format!("{which} the fold reads is null")));
        }
    }

    let num_sms = ctx.multiprocessors().unwrap_or(1).max(1);
    let blocks = grid_blocks(
        merge_blocks_per_sm(instantiation, smem),
        split.max_seq_len,
        split.num_heads,
        num_sms,
    );

    ctx.fire(
        op,
        Fire::at(FILE, instantiation)
            .apply(Launch::grid([blocks, 1, 1], [bdx, bdy, 1]).smem(smem)),
        &[
            ArgValue::Ptr(split.tmp_v),
            ArgValue::Ptr(split.tmp_s),
            ArgValue::Ptr(split.indptr),
            ArgValue::Ptr(split.o),
            ArgValue::Ptr(split.lse),
            split.max_seq_len.arg(),
            ArgValue::Ptr(split.seq_len),
            split.num_heads.arg(),
            // **THE FOLD IS WHERE FA2 TOUCHES THE PLANE**, so it is the one
            // launch of this family that takes the seat. Under a split
            // schedule the attention itself writes only the partial planes
            // in the plan's workspace; `v_merged`/`s_merged` are the fire's
            // own output and log-sum-exp rectangles, and a region handed
            // those UNSLICED needs `win[1]` to find its rows. A null seat is
            // row zero and the whole extent, which is every fire that is not
            // a body's.
            ctx.stage(),
        ],
    )
}

// ── occupancy probes the engine sizes plans with ────────────────────────────

/// How many decode blocks one SM holds at this lattice point — the
/// occupancy fact behind [`decode_max_grid_size`]. Resolves (and so may
/// compile) the instantiation; host work for the prepare phase, never for
/// an entry.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: &Device) -> Option<u32> {
    use cudarc::driver::sys as dr;

    const OP: &str = "attention.plan_decode";
    let geometry = DecodeGeometry::derive(OP, head_dim, group_size, KvWidth::BF16, device).ok()?;
    let entrypoint = decode_symbol(OP, &geometry, DecodeArm::Full).ok()?;
    let root = crate::jit::Root::of(FILE)?;
    let resolved = crate::jit::cache::resolve(&root, entrypoint).ok()?;

    if geometry.smem_bytes > 48 * 1024 {
        unsafe {
            dr::cuFuncSetAttribute(
                resolved.function,
                dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(geometry.smem_bytes).unwrap_or(i32::MAX),
            );
        }
    }

    let mut blocks: core::ffi::c_int = 0;

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

#[cfg(not(feature = "_cuda"))]
#[must_use]
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: &Device) -> Option<u32> {
    let _ = (head_dim, group_size, device);
    None
}

/// The `max_grid_size` fact `plan_decode` takes as an argument: occupancy
/// times SM count, floored at the SM count when the probe cannot answer.
#[must_use]
pub fn decode_max_grid_size(
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    device: &Device,
) -> u32 {
    let floor = device.num_sm.max(1);
    if !crate::attn::plan::head_dim_instantiated(head_dim) {
        return floor;
    }
    let group = if num_kv_heads > 0 {
        (num_q_heads / num_kv_heads).max(1)
    } else {
        1
    };
    match decode_blocks_per_sm(head_dim, group, device) {
        Some(per_sm) => per_sm.max(1).saturating_mul(floor),
        None => floor,
    }
}

// ─── the launch geometry, derived host-side (was fa2/geometry.rs) ──────────

// The fa2 launch geometry, derived host-side exactly as the device text
// derives it: block shapes, tile widths, and the shared-memory budget per
// instantiation. Every constant here restates a formula in `attn/attention.cuh`
// (line references kept from the transcription), so a disagreement is a
// wrong launch, not a style choice.


/// The kv element width in bytes; the lattice is stamped at bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvWidth(pub u32);

impl KvWidth {
    pub const BF16: Self = Self(2);

    pub const POINTER: u32 = 8;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeGeometry {
    pub num_stages_smem: u32,
    pub tile_size_per_bdx: u32,
    pub vec_size: u32,
    pub bdx: u32,
    pub bdy: u32,
    pub bdz: u32,
    pub num_threads: u32,
    pub smem_bytes: u32,
    pub head_dim: u32,
}

impl DecodeGeometry {
    pub fn derive(
        op: &'static str,
        head_dim: u32,
        group_size: u32,
        kv: KvWidth,
        dev: &Device,
    ) -> Result<Self, Error> {
        if head_dim == 0 {
            return Err(refuse(op, "fa2 decode head_dim is zero (decode.cuh:762)"));
        }
        let a = 16 / kv.0;
        let b = head_dim / 32;
        let vec_size = if a > b { a } else { b };
        if vec_size == 0 {
            return Err(refuse(op, "fa2 decode head_dim is zero (decode.cuh:762)"));
        }
        let bdx = head_dim / vec_size;
        if bdx > 32 {
            return Err(refuse(
                op,
                format!("fa2 decode head_dim {head_dim} needs bdx > 32 (decode.cuh:765)"),
            ));
        }
        // 12 joined the lattice with qwen4 (24 query heads over 2 kv).
        if !matches!(group_size, 1 | 2 | 3 | 4 | 8 | 12) {
            return Err(refuse(
                op,
                format!(
                    "fa2 decode GQA group {group_size} is outside DISPATCH_GQA_GROUP_SIZE \
                     (utils.cuh:164)"
                ),
            ));
        }
        let bdy = group_size;
        let lanes = bdx * bdy;
        let num_threads = if lanes > 128 { lanes } else { 128 };
        let bdz = num_threads / lanes;
        let tile_size_per_bdx = if group_size == 1 {
            if kv.0 == 1 { 2 } else { 4 }
        } else {
            1
        };
        let num_stages_smem = if dev.cc_major >= 8 { 2 } else { 1 };
        let staged = 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * kv.0;
        let offsets = tile_size_per_bdx * num_threads * KvWidth::POINTER;
        let exchange = 2 * bdy * bdz * 4;
        let tail = if offsets > exchange {
            offsets
        } else {
            exchange
        };
        Ok(Self {
            num_stages_smem,
            tile_size_per_bdx,
            vec_size,
            bdx,
            bdy,
            bdz,
            num_threads,
            smem_bytes: staged + tail,
            head_dim,
        })
    }

    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        [self.bdx, self.bdy, self.bdz]
    }

    #[must_use]
    pub const fn grid(padded_batch_size: u32, num_kv_heads: u32) -> [u32; 3] {
        [padded_batch_size, num_kv_heads, 1]
    }
}

#[allow(clippy::if_same_then_else)]
const fn num_warps_q(cta_tile_q: u32) -> u32 {
    if cta_tile_q == 32 {
        1
    } else if cta_tile_q > 16 {
        4
    } else {
        1
    }
}

const fn num_warps_kv(cta_tile_q: u32) -> u32 {
    4 / num_warps_q(cta_tile_q)
}

#[allow(clippy::if_same_then_else)]
const fn num_mma_q(cta_tile_q: u32) -> u32 {
    if cta_tile_q == 32 {
        2
    } else if cta_tile_q > 64 {
        2
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillGeometry {
    pub cta_tile_q: u32,
    pub num_mma_q: u32,
    pub num_mma_kv: u32,
    pub num_mma_d_qk: u32,
    pub num_mma_d_vo: u32,
    pub num_warps_q: u32,
    pub num_warps_kv: u32,
    pub cta_tile_kv: u32,
    pub smem_bytes: u32,
    pub head_dim: u32,
}

impl PrefillGeometry {
    pub fn derive(
        op: &'static str,
        head_dim: u32,
        cta_tile_q: u32,
        kv: KvWidth,
        use_fp16_qk_reduction: bool,
        dev: &Device,
    ) -> Result<Self, Error> {
        if !matches!(cta_tile_q, 16 | 32 | 64 | 128) {
            return Err(refuse(
                op,
                format!(
                    "fa2 prefill cta_tile_q {cta_tile_q} is outside DISPATCH_CTA_TILE_Q \
                     (utils.cuh:135)"
                ),
            ));
        }
        let q_width = 2u32;

        let vo_split_layout = kv.0 == 2 && head_dim >= 512 && cta_tile_q == 32;
        let (num_mma_q, num_warps_q_, num_warps_kv_) = if vo_split_layout {
            (1, 2, 2)
        } else {
            (
                num_mma_q(cta_tile_q),
                num_warps_q(cta_tile_q),
                num_warps_kv(cta_tile_q),
            )
        };

        let num_mma_d_qk = head_dim / 16;
        let num_mma_d_vo = head_dim / 16;

        let use_repack = kv.0 == 1 && head_dim != 64 && head_dim <= 256 && cta_tile_q > 16;
        let kv_shared = num_mma_d_vo > 16
            && num_mma_d_vo.is_multiple_of(num_warps_kv_)
            && (kv.0 == 2 || cta_tile_q > 16);
        let vo_split_dispatch = num_mma_d_vo > 16 && num_mma_d_vo.is_multiple_of(num_warps_kv_);

        let per_mma_kv = (if kv_shared {
            head_dim * 16 * num_warps_kv_ * kv.0
        } else {
            (head_dim + head_dim) * 16 * num_warps_kv_ * kv.0
        }) + (if use_repack {
            head_dim * 16 * num_warps_kv_ * q_width
        } else {
            0
        }) + (if vo_split_dispatch {
            cta_tile_q * num_warps_kv_ * 16 * q_width
        } else {
            0
        });

        let vo_split_fixed = if vo_split_dispatch {
            num_warps_kv_ * cta_tile_q * 8 + 2048
        } else {
            0
        };
        let shared_rope_freq = 0;
        let fixed_smem = cta_tile_q * head_dim * q_width + vo_split_fixed + shared_rope_freq;

        let min_valid_mma_kv = if kv.0 == 1 && num_warps_q_ > 2 {
            num_warps_q_ / 2
        } else {
            1
        };
        let ctas_per_sm = if dev.max_smem_per_sm >= 2 * (fixed_smem + min_valid_mma_kv * per_mma_kv)
        {
            2
        } else {
            1
        };
        let per_block = {
            let a = dev.max_smem_per_sm / ctas_per_sm;
            if a < dev.max_smem_per_block_optin {
                a
            } else {
                dev.max_smem_per_block_optin
            }
        };
        let _ = use_fp16_qk_reduction;
        let max_mma_kv_reg = 8 / num_mma_q;
        if per_block <= fixed_smem || (per_block - fixed_smem) < per_mma_kv {
            return Err(refuse(
                op,
                format!(
                    "the fa2 prefill kv tile does not fit shared memory: {} bytes needed, \
                     {per_block} per block (prefill.cuh:4270)",
                    fixed_smem + per_mma_kv
                ),
            ));
        }
        let max_mma_kv_smem = (per_block - fixed_smem) / per_mma_kv;
        let budget = if max_mma_kv_smem < max_mma_kv_reg {
            max_mma_kv_smem
        } else {
            max_mma_kv_reg
        };
        let num_mma_kv = if budget >= 8 {
            8
        } else if budget >= 4 {
            4
        } else if budget >= 2 {
            2
        } else {
            1
        };

        let num_mma_d_vo_tile = if num_mma_d_vo > 16 { 16 } else { num_mma_d_vo };
        let num_mma_d_vo_per_warp = if vo_split_dispatch {
            num_mma_d_vo / num_warps_kv_
        } else {
            num_mma_d_vo
        };
        let reg_frags = if vo_split_dispatch {
            num_mma_d_vo_per_warp
        } else {
            num_mma_d_vo_tile
        };
        let invalid = (if head_dim >= 512 {
            cta_tile_q > 32
        } else {
            cta_tile_q == 32
        }) || num_mma_d_vo < 4
            || (num_mma_d_vo == 4 && num_mma_kv % 2 == 1)
            || num_mma_q * (8 * reg_frags + 2 * 4 * num_mma_kv) >= 256;
        if invalid {
            return Err(refuse(
                op,
                "no fa2 prefill trait instantiation exists at this tile shape",
            ));
        }

        let cta_tile_kv = num_mma_kv * num_warps_kv_ * 16;
        let smem_bytes = Self::shared_storage_paged(
            cta_tile_q,
            cta_tile_kv,
            head_dim,
            num_warps_kv_,
            kv,
            q_width,
        );
        if smem_bytes > dev.max_smem_per_block_optin {
            return Err(refuse(
                op,
                format!(
                    "the fa2 prefill shared storage needs {smem_bytes} bytes; the device \
                     opts in to {}",
                    dev.max_smem_per_block_optin
                ),
            ));
        }
        Ok(Self {
            cta_tile_q,
            num_mma_q,
            num_mma_kv,
            num_mma_d_qk,
            num_mma_d_vo,
            num_warps_q: num_warps_q_,
            num_warps_kv: num_warps_kv_,
            cta_tile_kv,
            smem_bytes,
            head_dim,
        })
    }

    /// `sizeof(SharedStorage)` for the paged prefill traits, restated.
    #[must_use]
    pub const fn shared_storage_paged(
        cta_tile_q: u32,
        cta_tile_kv: u32,
        head_dim: u32,
        num_warps_kv: u32,
        kv: KvWidth,
        q_width: u32,
    ) -> u32 {
        const fn align16(n: u32) -> u32 {
            n.div_ceil(16) * 16
        }

        let kv_share_shape = head_dim / 16 > 16 && (head_dim / 16).is_multiple_of(num_warps_kv);
        let vo_split = kv_share_shape;
        let v_share_active = kv_share_shape && (kv.0 == 2 || cta_tile_q > 16);

        let mut a = 0;
        a = align16(a) + cta_tile_q * head_dim * q_width;
        a = align16(a) + cta_tile_kv * head_dim * kv.0;
        a = align16(a)
            + if v_share_active {
                kv.0
            } else {
                cta_tile_kv * head_dim * kv.0
            };
        let a = align16(a);

        let sync_o_elems = if num_warps_kv == 1 || vo_split {
            1
        } else {
            num_warps_kv * cta_tile_q * if head_dim > 256 { 256 } else { head_dim }
        };
        let sync_md_elems = if num_warps_kv == 1 {
            1
        } else {
            num_warps_kv * cta_tile_q
        };
        let mut b = 0;
        b = align16(b) + sync_o_elems * 4;
        b = align16(b) + sync_md_elems * 8;
        let b = align16(b);

        let c = align16(cta_tile_q * head_dim * q_width);

        let mut off = if a > b { a } else { b };
        if c > off {
            off = c;
        }

        off = align16(off) + 1;
        off = align16(off) + 1;
        off = align16(off) + q_width;
        off = align16(off)
            + if vo_split {
                cta_tile_q * cta_tile_kv * q_width
            } else {
                q_width
            };
        off = align16(off)
            + if vo_split {
                num_warps_kv * cta_tile_q * 8
            } else {
                8
            };
        align16(off)
    }

    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        [32, self.num_warps_q, self.num_warps_kv]
    }

    #[must_use]
    pub const fn grid(padded_batch_size: u32, num_kv_heads: u32) -> [u32; 3] {
        [padded_batch_size, 1, num_kv_heads]
    }
}
