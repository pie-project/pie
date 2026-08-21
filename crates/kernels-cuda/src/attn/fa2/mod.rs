
pub mod geometry;

pub mod params;

pub mod dispatch;

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

fn dequant_prelude(ctx: &Ctx<'_>) {
    let _ = crate::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(ctx);
}

fn upload_plan(ctx: &Ctx<'_>, src: *const u8, len: usize, dst: *mut u8) -> Result<(), Refusal> {
    if len == 0 || src.is_null() || dst.is_null() {
        return Ok(());
    }

    let bytes = unsafe { core::slice::from_raw_parts(src, len) };

    unsafe { plan::upload_int_plan(bytes, dst as u64, 0, ctx.stream()) }
}

pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

pub const DECODE_GQA: &[u32] = &[1, 2, 3, 4, 8];

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
    pub root: Root,
    pub arms: [&'static str; 5],
}

#[derive(Debug)]
pub struct PrefillRoot {

    pub head_dim: u32,
    pub cta_tile_q: u32,
    pub num_mma_kv: u32,
    pub root: Root,
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

pub static DECODE: [DecodeRoot; 20] = [
    decode_root!(hd = 64, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 8, bdy = 1, bdz = 16),
    decode_root!(hd = 64, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 2, bdz = 8),
    decode_root!(hd = 64, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 3, bdz = 5),
    decode_root!(hd = 64, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 4, bdz = 4),
    decode_root!(hd = 64, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 8, bdz = 2),
    decode_root!(hd = 128, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 16, bdy = 1, bdz = 8),
    decode_root!(hd = 128, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 2, bdz = 4),
    decode_root!(hd = 128, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 3, bdz = 2),
    decode_root!(hd = 128, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 4, bdz = 2),
    decode_root!(hd = 128, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 8, bdz = 1),
    decode_root!(hd = 256, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 32, bdy = 1, bdz = 4),
    decode_root!(hd = 256, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 2, bdz = 2),
    decode_root!(hd = 256, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 3, bdz = 1),
    decode_root!(hd = 256, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 4, bdz = 1),
    decode_root!(hd = 256, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 8, bdz = 1),
    decode_root!(hd = 512, gqa = 1, stages = 2, tile = 4, vec = 16, bdx = 32, bdy = 1, bdz = 4),
    decode_root!(hd = 512, gqa = 2, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 2, bdz = 2),
    decode_root!(hd = 512, gqa = 3, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 3, bdz = 1),
    decode_root!(hd = 512, gqa = 4, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 4, bdz = 1),
    decode_root!(hd = 512, gqa = 8, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 8, bdz = 1),
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
    DECODE.iter().find(|p| p.head_dim == head_dim && p.group_size == group_size)
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

#[must_use]
pub fn decode_instantiation(
    head_dim: u32,
    group_size: u32,
    arm: DecodeArm,
) -> Option<&'static str> {
    decode_root(head_dim, group_size).map(|p| p.arms[arm as usize])
}

#[must_use]
pub fn prefill_instantiation(
    head_dim: u32,
    cta_tile_q: u32,
    num_mma_kv: u32,
    arm: PrefillArm,
) -> Option<&'static str> {
    prefill_root(head_dim, cta_tile_q, num_mma_kv).map(|p| p.arms[arm as usize])
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

pub trait DecodeBlock: Copy {}
impl DecodeBlock for DecodeParams {}
impl DecodeBlock for DecodeScoreParams {}

pub trait PrefillBlock: Copy {}
impl PrefillBlock for PrefillPagedParams {}
impl PrefillBlock for PrefillScoreParams {}

fn block<P>(params: &P) -> ArgValue {
    ArgValue::Bytes { ptr: core::ptr::from_ref(params).cast::<u8>(), len: size_of::<P>() }
}

fn no_point(what: &'static str, why: &dyn core::fmt::Display) -> Refusal {
    tracing::error!(why = %why, "an FA2 fire found no kernel");
    Refusal::Unstated { what }
}

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

#[must_use]
pub fn prefill_custom_arm(logits_soft_cap: f32) -> PrefillArm {
    if logits_soft_cap > 0.0 { PrefillArm::CustomSoftcap } else { PrefillArm::Custom }
}

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

fn addr<T>(p: *const T) -> DevicePtr {
    p as usize as u64
}

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

const CAPTURE_VARIANT: Refusal =
    Refusal::Unstated { what: "a score capture without a soft cap or a window" };

const CAPTURE_SINK: Refusal = Refusal::Absent { what: "the score sink" };

#[routine(depth_prefix_plan, no_join)]
pub fn dispatch_attention_flashinfer_decode(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Decode>>,
    o: Out<Tensor<bf16>>,
    window_left: Const<i32>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let window_left = *window_left;
    let logits_soft_cap = *logits_soft_cap;
    let sm_scale = *sm_scale;

    dequant_prelude(ctx);

    let k_pages = kvc.keys;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let broadcast_q = false;
    let v_pages = kvc.values;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;

    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the decode plan this statement names" });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::decode_plan_of(cache, crate::attn::fa2::plan::fa_device());

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
        core::ptr::null(),
        lse,
        int_base,
        ctx.ask::<*mut c_void, keys::AttnWorkspaceFloat>()?,
    );
    let arm = decode_arm(planned.full_attention_variant, window_left, logits_soft_cap);
    let (params, split) =
        make_decode_params(&planned, &bufs, window_left, logits_soft_cap, sm_scale, broadcast_q);
    decode(ctx, decode_at(&planned, arm, params.padded_batch_size), &params)?;
    if planned.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

#[routine(depth_prefix_plan, no_join)]
pub fn dispatch_attention_flashinfer_decode_lse(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Decode>>,
    o: Out<Tensor<bf16>>,
    _lse: Out<Tensor<f32>>,
    _window_left: Const<i32>,
    _logits_soft_cap: Const<f32>,
    _sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let _window_left = *_window_left;
    let _logits_soft_cap = *_logits_soft_cap;
    let _sm_scale = *_sm_scale;

    let _k_pages = kvc.keys;
    let _v_pages = kvc.values;
    let _kv_page_indices = kvc.page_indices as *const u32;
    let _kv_page_indptr = kvc.page_indptr as *const u32;
    let _kv_last_page_lens = kvc.last_page_lens as *const u32;
    let _broadcast_q = false;

    dispatch_attention_flashinfer_decode(ctx, q, plan, o)
}

#[routine(no_join)]
pub fn dispatch_attention_flashinfer_decode_capture(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Decode>>,
    o: Out<Tensor<bf16>>,
    window_left: Const<i32>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let score_out = ctx.ask::<*mut f32, keys::AttnScoreOut>()?;
    let score_indptr = ctx.ask::<*const i32, keys::AttnScoreIndptr>()?;

    dequant_prelude(ctx);
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let broadcast_q = false;

    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the decode plan this statement names" });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::decode_plan_of(cache, crate::attn::fa2::plan::fa_device());

    let int_base = ctx.ask::<*mut c_void, keys::AttnWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (int_base as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;

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

#[routine(no_join)]
pub fn dispatch_attention_flashinfer_prefill_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    qo_indptr: In<Tensor<i32>>,
    kvc: In<Struct<KvCache>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let logits_soft_cap = *logits_soft_cap;
    let sm_scale = *sm_scale;

    let qo_indptr = qo_indptr.ptr as *const u32;
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;

    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the prefill plan this statement names" });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());

    let carve = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;

    let float_base = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceFloat>()?;

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

#[routine(no_join)]
pub fn dispatch_attention_flashinfer_prefill_capture_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    score_window: Const<u32>,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let score_out = ctx.ask::<*mut f32, keys::AttnScoreOut>()?;
    let score_indptr = ctx.ask::<*const i32, keys::AttnScoreIndptr>()?;
    let score_window = *score_window;

    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;

    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the prefill plan this statement names" });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());

    let carve = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;

    let float_base = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceFloat>()?;
    let int_base = (carve as usize).saturating_add(cache.int_base_bytes) as *mut c_void;

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

#[routine(no_join)]
pub fn dispatch_attention_flashinfer_prefill_custom(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    maskv: In<Struct<AttnMask>>,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>) -> Result<(), Refusal> {
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
    }
    let maskv = unsafe { &*maskv.ptr };
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let mask = maskv.mask;
    let mask_indptr = maskv.indptr;

    dequant_prelude(ctx);
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;

    if plan.ptr.is_null() {
        return Err(Refusal::Null { what: "the prefill plan this statement names" });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());

    let carve = ctx.ask::<*mut c_void, keys::AttnPrefillWorkspaceInt>()?;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;

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

    params.maybe_custom_mask = addr(mask);
    params.maybe_mask_indptr = addr(mask_indptr);
    params.window_left = -1;
    prefill(ctx, prefill_at(&planned, arm, params.padded_batch_size), &params)?;
    if planned.info.split_kv { fold(ctx, &split) } else { Ok(()) }
}

fn plan_own_prefill(ctx: &Ctx<'_>, q_width: i32,
    requests: i32,
    head_dim: i32,
    rows: i32,
    kv_num_heads: i32,
    kvc: &crate::views::PagedKvView,
    attn_workspace_float_bytes: usize,
    attn_workspace_int_bytes: usize,
    window_left: i32) -> Result<PrefillPlan, Refusal> {
    let requests = requests;
    if requests <= 0 {
        return Err(Refusal::Empty { what: "the batch" });
    }
    let head_dim = head_dim;

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

    let (qo_h, kv_h) = unsafe {
        (
            core::slice::from_raw_parts(qo_h, n),
            core::slice::from_raw_parts(kv_h, n),
        )
    };

    let cache = unsafe { &mut *cache.cast::<plan::PrefillPlanCache>() };
    let device = plan::plan_device();
    let planned = plan::plan_prefill(
        cache,
        qo_h,
        kv_h,
        rows,
        requests,
        q_width / head_dim,
        kv_num_heads,
        head_dim,
        (kvc.page_size),
        crate::attn::plan::Workspace {
            float_bytes: attn_workspace_float_bytes,
            int_bytes: attn_workspace_int_bytes,
        },
        &device,
        true,
        window_left,
        false,
        (kvc.layout as bool),
        true,
        false,
        false,
    );
    if let plan::Planned::Declined(why) = planned {
        tracing::error!(%why, "the planless FA2 prefill could not plan its own fire");
        return Err(Refusal::Unstated { what: "a plannable FA2 prefill fire; see the log" });
    }

    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()? as usize)
            .saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    Ok(dispatch::prefill_plan_of(cache, plan::fa_device()))
}

#[routine(whole, no_join)]
pub fn attention_flashinfer_prefill(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    dequant_prelude(ctx);
    let plan = plan_own_prefill(ctx, q.width)?;
    let lse = ctx.ask::<*mut f32, keys::AttnLseOut>()?;
    let int_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()?;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let float_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceFloat>()?;

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

#[routine(whole, no_join)]
pub fn attention_flashinfer_prefill_lse(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    _lse: Out<Tensor<f32>>,
    attn_logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };

    let logits_soft_cap = Const { v: (*attn_logits_soft_cap) };
    let sm_scale = Const { v: (*sm_scale) };
    let _int_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceInt>()?;
    let _k_pages = kvc.keys;
    let _v_pages = kvc.values;
    let _qo_indptr = qo_indptr.ptr as *const u32;
    let _kv_page_indices = kvc.page_indices as *const u32;
    let _kv_page_indptr = kvc.page_indptr as *const u32;
    let _kv_last_page_lens = kvc.last_page_lens as *const u32;
    let _float_buffer = ctx.ask::<*mut core::ffi::c_void, keys::AttnWorkspaceFloat>()?;

    attention_flashinfer_prefill(ctx, q, o, logits_soft_cap, sm_scale)
}

fn prefill_plan_usable(plan: &PrefillPlan) -> Result<(), Refusal> {

    const UNPLANNED_PREFILL: Refusal = Refusal::Unstated { what: "a planned FA2 prefill cache" };

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

#[cfg(feature = "_cuda")]
#[must_use]
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: Device) -> Option<u32> {
    use cudarc::driver::sys as dr;

    let point = decode_root(head_dim, group_size)?;
    let geometry = DecodeGeometry::derive(head_dim, group_size, KvWidth::BF16, device).ok()?;
    let resolved =
        crate::jit::cache::resolve(&point.root, point.arms[DecodeArm::Full as usize]).ok()?;

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
pub fn decode_blocks_per_sm(head_dim: u32, group_size: u32, device: Device) -> Option<u32> {
    let _ = (head_dim, group_size, device);
    None
}
