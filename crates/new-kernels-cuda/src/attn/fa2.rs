//! The fa2 lattice: every FlashInfer decode/prefill instantiation this
//! plane is stamped for, the arm tables that pick a variant, and the
//! launchers that fire one parameter block at the geometry
//! [`geometry`] derives. Selection lives here, below the entries
//! (decision #13) — a dispatch arm never sees a lattice point.

pub mod geometry;

pub mod params;

use new_kernels::KernelError;

use crate::attn::fa2::geometry::{DecodeGeometry, KvWidth, PrefillGeometry};
use crate::attn::fa2::params::{DecodeParams, Partials, PrefillPagedParams};
use crate::attn::plan::Device;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse};

pub const FILE: &str = "attn/fa2.cuh";

const MERGE_FILE: &str = "cascade/merge_states.cuh";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeArm {
    Full = 0,
    Softcap = 1,
    Window = 2,
    CaptureFull = 3,
    CaptureWindow = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillArm {
    CausalFullSoftcap = 0,
    NoneFullSoftcap = 1,
    CausalFull = 2,
    NoneFull = 3,
    CausalSoftcap = 4,
    CausalWindow = 5,
    CausalCapture = 6,
    NoneCapture = 7,
    CustomSoftcap = 8,
    Custom = 9,
}

#[derive(Debug)]
pub struct DecodeRoot {
    pub head_dim: u32,
    pub group_size: u32,
    pub arms: [&'static str; 5],
}

#[derive(Debug)]
pub struct PrefillRoot {
    pub head_dim: u32,
    pub cta_tile_q: u32,
    pub num_mma_kv: u32,
    pub arms: [&'static str; 10],
}

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

macro_rules! decode_root {
    (
        hd = $hd:literal, gqa = $g:literal,
        stages = $ns:literal, tile = $tile:literal, vec = $vec:literal,
        bdx = $bdx:literal, bdy = $bdy:literal, bdz = $bdz:literal $(,)?
    ) => {
        DecodeRoot {
            head_dim: $hd,
            group_size: $g,
            arms: [
                decode_inst!(
                    $ns,
                    $tile,
                    $vec,
                    $bdx,
                    $bdy,
                    $bdz,
                    "VariantFull",
                    "DecodeParams"
                ),
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
                decode_inst!(
                    $ns,
                    $tile,
                    $vec,
                    $bdx,
                    $bdy,
                    $bdz,
                    "VariantWindow",
                    "DecodeParams"
                ),
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

pub static DECODE: [DecodeRoot; 20] = [
    decode_root!(
        hd = 64,
        gqa = 1,
        stages = 2,
        tile = 4,
        vec = 8,
        bdx = 8,
        bdy = 1,
        bdz = 16
    ),
    decode_root!(
        hd = 64,
        gqa = 2,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 8,
        bdy = 2,
        bdz = 8
    ),
    decode_root!(
        hd = 64,
        gqa = 3,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 8,
        bdy = 3,
        bdz = 5
    ),
    decode_root!(
        hd = 64,
        gqa = 4,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 8,
        bdy = 4,
        bdz = 4
    ),
    decode_root!(
        hd = 64,
        gqa = 8,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 8,
        bdy = 8,
        bdz = 2
    ),
    decode_root!(
        hd = 128,
        gqa = 1,
        stages = 2,
        tile = 4,
        vec = 8,
        bdx = 16,
        bdy = 1,
        bdz = 8
    ),
    decode_root!(
        hd = 128,
        gqa = 2,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 16,
        bdy = 2,
        bdz = 4
    ),
    decode_root!(
        hd = 128,
        gqa = 3,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 16,
        bdy = 3,
        bdz = 2
    ),
    decode_root!(
        hd = 128,
        gqa = 4,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 16,
        bdy = 4,
        bdz = 2
    ),
    decode_root!(
        hd = 128,
        gqa = 8,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 16,
        bdy = 8,
        bdz = 1
    ),
    decode_root!(
        hd = 256,
        gqa = 1,
        stages = 2,
        tile = 4,
        vec = 8,
        bdx = 32,
        bdy = 1,
        bdz = 4
    ),
    decode_root!(
        hd = 256,
        gqa = 2,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 32,
        bdy = 2,
        bdz = 2
    ),
    decode_root!(
        hd = 256,
        gqa = 3,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 32,
        bdy = 3,
        bdz = 1
    ),
    decode_root!(
        hd = 256,
        gqa = 4,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 32,
        bdy = 4,
        bdz = 1
    ),
    decode_root!(
        hd = 256,
        gqa = 8,
        stages = 2,
        tile = 1,
        vec = 8,
        bdx = 32,
        bdy = 8,
        bdz = 1
    ),
    decode_root!(
        hd = 512,
        gqa = 1,
        stages = 2,
        tile = 4,
        vec = 16,
        bdx = 32,
        bdy = 1,
        bdz = 4
    ),
    decode_root!(
        hd = 512,
        gqa = 2,
        stages = 2,
        tile = 1,
        vec = 16,
        bdx = 32,
        bdy = 2,
        bdz = 2
    ),
    decode_root!(
        hd = 512,
        gqa = 3,
        stages = 2,
        tile = 1,
        vec = 16,
        bdx = 32,
        bdy = 3,
        bdz = 1
    ),
    decode_root!(
        hd = 512,
        gqa = 4,
        stages = 2,
        tile = 1,
        vec = 16,
        bdx = 32,
        bdy = 4,
        bdz = 1
    ),
    decode_root!(
        hd = 512,
        gqa = 8,
        stages = 2,
        tile = 1,
        vec = 16,
        bdx = 32,
        bdy = 8,
        bdz = 1
    ),
];

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

#[must_use]
pub fn decode_root(head_dim: u32, group_size: u32) -> Option<&'static DecodeRoot> {
    DECODE
        .iter()
        .find(|p| p.head_dim == head_dim && p.group_size == group_size)
}

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
) -> Result<(), KernelError> {
    let Some(point) = decode_root(at.head_dim, at.group_size) else {
        return Err(refuse(
            op,
            format!(
                "no fa2 decode lattice point at head width {} x GQA group {}",
                at.head_dim, at.group_size
            ),
        ));
    };
    let geometry =
        DecodeGeometry::derive(op, at.head_dim, at.group_size, KvWidth::BF16, &at.device)?;

    ctx.fire(
        op,
        Fire::at(FILE, point.arms[at.arm as usize]).apply(
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
) -> Result<(), KernelError> {
    let geometry = PrefillGeometry::derive(
        op,
        at.head_dim,
        at.cta_tile_q,
        KvWidth::BF16,
        false,
        &at.device,
    )?;
    let Some(point) = prefill_root(at.head_dim, at.cta_tile_q, geometry.num_mma_kv) else {
        return Err(refuse(
            op,
            format!(
                "no fa2 prefill lattice point at head width {} x CTA_TILE_Q {} x NUM_MMA_KV {}",
                at.head_dim, at.cta_tile_q, geometry.num_mma_kv
            ),
        ));
    };

    ctx.fire(
        op,
        Fire::at(FILE, point.arms[at.arm as usize]).apply(
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

const fn merge_varlen_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
             8, 8, 16, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        128 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
             8, 16, 8, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        256 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
             8, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        512 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
             16, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        _ => None,
    }
}

fn no_merge_row(op: &'static str) -> KernelError {
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

    let Some(root) = crate::jit::Root::of(MERGE_FILE) else {
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
pub(crate) fn fold(ctx: &Ctx, op: &'static str, split: &Partials) -> Result<(), KernelError> {
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
        Fire::at(MERGE_FILE, instantiation)
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
        ],
    )
}

// ── occupancy probes the driver sizes plans with ────────────────────────────

/// How many decode blocks one SM holds at this lattice point — the
/// occupancy fact behind [`decode_max_grid_size`]. Resolves (and so may
/// compile) the instantiation; host work for the prepare phase, never for
/// an entry.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: &Device) -> Option<u32> {
    use cudarc::driver::sys as dr;

    let point = decode_root(head_dim, group_size)?;
    let geometry = DecodeGeometry::derive(
        "attention.plan_decode",
        head_dim,
        group_size,
        KvWidth::BF16,
        device,
    )
    .ok()?;
    let root = crate::jit::Root::of(FILE)?;
    let resolved = crate::jit::cache::resolve(&root, point.arms[DecodeArm::Full as usize]).ok()?;

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
