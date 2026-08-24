pub mod geometry;

pub mod params;

pub mod dispatch;

pub mod plan;

use core::ffi::c_void;
use core::mem::size_of;
use core::ptr::NonNull;
use kernels::Bind;

use crate::attn::fa2::geometry::{DecodeGeometry, Device, KvWidth, PrefillGeometry};
use crate::attn::fa2::params::{
    Buffers, DecodeParams, DecodePlan, DevicePtr, Partials, PrefillPagedParams, PrefillPlan,
    make_decode_params, make_prefill_params,
};
use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, unpack_aggregate};
use crate::jit::{ArgValue, Ctx, Cuda, Launch, Root};
use crate::raises::{Fa2Decode, Fa2Prefill};
use crate::views::{AttnMask, KvCache, KvPageIndptrHost, QoIndptrHost};
use kernels::plane::Fire;
use kernels::plane::{Arg, Const, In, Out};
use kernels::raises::Struct;
use kernels::{Refusal, Ty};

fn dequant_prelude(
    ctx: &Ctx<'_>,
    kvc: *const crate::views::PagedKvView,
    num_kv_heads: i32,
    head_dim: i32,
) {
    let _ = crate::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
        ctx,
        In {
            ptr: kvc,
            rows: 0,
            width: 0,
        },
        Const { v: num_kv_heads },
        Const { v: head_dim },
    );
}

fn upload_plan(ctx: &Ctx<'_>, src: *const u8, len: usize, dst: *mut u8) -> Result<(), Refusal> {
    if len == 0 || src.is_null() || dst.is_null() {
        return Ok(());
    }

    let bytes = unsafe { core::slice::from_raw_parts(src, len) };

    unsafe { plan::upload_int_plan(bytes, dst as u64, 0, ctx.stream()) }
}

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

fn block<P>(params: &P) -> ArgValue {
    ArgValue::Bytes {
        ptr: core::ptr::from_ref(params).cast::<u8>(),
        len: size_of::<P>(),
    }
}

fn no_point(what: &'static str, why: &dyn core::fmt::Display) -> Refusal {
    tracing::error!(why = %why, "an FA2 fire found no kernel");
    Refusal::Unstated { what }
}

pub fn decode(ctx: &Ctx<'_>, at: DecodePoint, params: &DecodeParams) -> Result<(), Refusal> {
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

pub fn prefill(
    ctx: &Ctx<'_>,
    at: PrefillPoint,
    params: &PrefillPagedParams,
) -> Result<(), Refusal> {
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

const NUM_THREADS: u32 = 128;
const NUM_SMEM_STAGES: u32 = 4;
const NO_ROW: Refusal = Refusal::Unstated {
    what: "a cascade merge at this head dim -- 64, 128, 256 and 512 are here",
};

#[must_use]
const fn geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

#[must_use]
const fn smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = geometry(head_dim) else {
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

fn null_check(is_null: bool, which: &'static str) -> Result<(), Refusal> {
    if is_null {
        Err(Refusal::Null { what: which })
    } else {
        Ok(())
    }
}

fn grid_blocks(per_sm: u32, max_seq_len: u32, num_heads: u32, num_sms: u32) -> u32 {
    let work_bound = max_seq_len
        .saturating_mul(num_heads)
        .div_ceil(num_sms)
        .max(1);
    per_sm.min(work_bound).saturating_mul(num_sms).max(num_sms)
}

#[cfg(feature = "_cuda")]
fn blocks_per_sm(instantiation: &str, smem: u32) -> u32 {
    use cudarc::driver::sys as dr;

    let Ok(resolved) = crate::jit::cache::resolve(
        &crate::jit::Root::new("cascade/merge_states.cuh"),
        instantiation,
    ) else {
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
fn blocks_per_sm(_instantiation: &str, _smem: u32) -> u32 {
    1
}

#[allow(clippy::too_many_arguments)]
fn merge_states_varlen(
    ctx: &Ctx<'_>,
    v: *mut bf16,
    s: *mut f32,
    indptr: *mut i32,
    v_merged: *mut bf16,
    s_merged: *mut f32,
    max_seq_len: u32,
    seq_len: *mut u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), Refusal> {
    let (_, bdx, bdy) = geometry(head_dim).ok_or(NO_ROW)?;
    let smem = smem_bytes(head_dim).ok_or(NO_ROW)?;
    let instantiation = merge_varlen_inst(head_dim).ok_or(NO_ROW)?;
    null_check(v.is_null(), "v")?;
    null_check(s.is_null(), "s")?;
    null_check(indptr.is_null(), "indptr")?;
    null_check(v_merged.is_null(), "v_merged")?;

    let num_sms = ctx.multiprocessors()?.max(1);
    let blocks = grid_blocks(
        blocks_per_sm(instantiation, smem),
        max_seq_len,
        num_heads,
        num_sms,
    );

    ctx.fire(
        Fire::at("cascade/merge_states.cuh", instantiation)
            .apply(Launch::grid([blocks, 1, 1], [bdx, bdy, 1]).smem(smem)),
        &[
            v.arg(),
            s.arg(),
            indptr.arg(),
            v_merged.arg(),
            NonNull::new(s_merged).arg(),
            max_seq_len.arg(),
            NonNull::new(seq_len).arg(),
            num_heads.arg(),
        ],
    )
}

fn fold(ctx: &Ctx<'_>, split: &Partials) -> Result<(), Refusal> {
    merge_states_varlen(
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

#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_flashinfer_decode(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Decode>>,
    o: Out<Tensor<bf16>>,
    window_left: Const<i32>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc_ptr = kvc.ptr;
    let kvc = unsafe { &*kvc_ptr };

    let window_left = *window_left;
    let logits_soft_cap = *logits_soft_cap;
    let sm_scale = *sm_scale;

    let k_pages = kvc.keys;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let lse = lse.map_or(core::ptr::null_mut(), |l| l.ptr);
    let broadcast_q = false;
    let v_pages = kvc.values;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;

    if plan.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the decode plan this statement names",
        });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::decode_plan_of(cache, crate::attn::fa2::plan::fa_device());

    dequant_prelude(ctx, kvc_ptr, cache.num_kv_heads, cache.head_dim);

    let int_base = cache.int_workspace;
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
        cache.float_workspace,
    );
    let arm = decode_arm(planned.full_attention_variant, window_left, logits_soft_cap);
    let (params, split) = make_decode_params(
        &planned,
        &bufs,
        window_left,
        logits_soft_cap,
        sm_scale,
        broadcast_q,
    );
    decode(
        ctx,
        decode_at(&planned, arm, params.padded_batch_size),
        &params,
    )?;
    if planned.info.split_kv {
        fold(ctx, &split)
    } else {
        Ok(())
    }
}

fn prefill_paged(
    ctx: &Ctx<'_>,
    bufs: &Buffers,
    plan: &PrefillPlan,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> Result<(), Refusal> {
    prefill_plan_usable(plan)?;
    let arm = prefill_arm(
        plan.full_attention_variant,
        plan.causal_mask,
        logits_soft_cap,
    );
    let (params, split) = make_prefill_params(plan, bufs, logits_soft_cap, sm_scale);
    prefill(
        ctx,
        prefill_at(plan, arm, params.padded_batch_size),
        &params,
    )?;
    if plan.info.split_kv {
        fold(ctx, &split)
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_attention_flashinfer_prefill_custom(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<Fa2Prefill>>,
    o: Out<Tensor<bf16>>,
    window_left: Const<i32>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    maskv: In<Struct<AttnMask>>,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc_ptr = kvc.ptr;
    let kvc = unsafe { &*kvc_ptr };

    let mask = maskv.mask;
    let mask_indptr = maskv.indptr;

    let lse = lse.map_or(core::ptr::null_mut(), |l| l.ptr);
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;

    if plan.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the prefill plan this statement names",
        });
    }

    let cache = unsafe { &*plan.ptr };
    let planned = dispatch::prefill_plan_of(cache, crate::attn::fa2::plan::fa_device());

    dequant_prelude(ctx, kvc_ptr, cache.num_kv_heads, cache.head_dim);

    let carve = cache.int_workspace;
    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (carve as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;

    let float_base = cache.float_workspace;
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
    params.window_left = *window_left;
    prefill(
        ctx,
        prefill_at(&planned, arm, params.padded_batch_size),
        &params,
    )?;
    if planned.info.split_kv {
        fold(ctx, &split)
    } else {
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "eleven independent plan inputs, one caller; a struct would only rename them"
)]
fn plan_own_prefill(
    ctx: &Ctx<'_>,
    q_width: i32,
    requests: i32,
    head_dim: i32,
    rows: i32,
    kv_num_heads: i32,
    kvc: &crate::views::PagedKvView,
    cache: &mut plan::PrefillPlanCache,
    qo_h: *const u32,
    kv_h: *const u32,
    window_left: i32,
) -> Result<PrefillPlan, Refusal> {
    if requests <= 0 {
        return Err(Refusal::Empty { what: "the batch" });
    }

    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the layer's head dim",
        });
    }
    if q_width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the query width, in heads",
            at: i64::from(q_width),
        });
    }
    if qo_h.is_null() || kv_h.is_null() {
        return Err(Refusal::Null {
            what: "the host indptr pair the planless prefill plans from",
        });
    }
    let n = requests as usize + 1;

    let (qo_h, kv_h) = unsafe {
        (
            core::slice::from_raw_parts(qo_h, n),
            core::slice::from_raw_parts(kv_h, n),
        )
    };

    let device = plan::plan_device();
    let workspace = crate::attn::plan::Workspace {
        float_bytes: cache.float_workspace_bytes,
        int_bytes: cache.int_workspace_bytes,
    };
    let planned = plan::plan_prefill(
        cache,
        qo_h,
        kv_h,
        rows,
        requests,
        q_width / head_dim,
        kv_num_heads,
        head_dim,
        kvc.page_size,
        workspace,
        &device,
        true,
        window_left,
        false,
        kvc.layout != 0,
        true,
        false,
        false,
    );
    if let plan::Planned::Declined(why) = planned {
        tracing::error!(%why, "the planless FA2 prefill could not plan its own fire");
        return Err(Refusal::Unstated {
            what: "a plannable FA2 prefill fire; see the log",
        });
    }

    upload_plan(
        ctx,
        cache.int_upload.as_slice().as_ptr(),
        cache.int_upload.as_slice().len(),
        (cache.int_workspace as usize).saturating_add(cache.int_base_bytes) as *mut u8,
    )?;
    Ok(dispatch::prefill_plan_of(cache, plan::fa_device()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_flashinfer_prefill(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    logits_soft_cap: Const<f32>,
    sm_scale: Const<f32>,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>,
    head_dim: Const<i32>,
    plan_cache: In<Struct<Fa2Prefill>>,
    qo_indptr_host: In<Struct<QoIndptrHost>>,
    kv_page_indptr_host: In<Struct<KvPageIndptrHost>>,
    kv_num_heads: Const<i32>,
    window_left: Const<i32>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc_ptr = kvc.ptr;
    let kvc = unsafe { &*kvc_ptr };
    if plan_cache.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the prefill plan cache this statement names",
        });
    }

    let cache = unsafe { &mut *plan_cache.ptr.cast_mut() };

    dequant_prelude(ctx, kvc_ptr, *kv_num_heads, *head_dim);
    let plan = plan_own_prefill(
        ctx,
        q.width,
        qo_indptr.rows,
        *head_dim,
        q.rows,
        *kv_num_heads,
        kvc,
        cache,
        qo_indptr_host.ptr,
        kv_page_indptr_host.ptr,
        *window_left,
    )?;
    let lse = lse.map_or(core::ptr::null_mut(), |l| l.ptr);
    let int_buffer = cache.int_workspace;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let float_buffer = cache.float_workspace;

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

fn prefill_plan_usable(plan: &PrefillPlan) -> Result<(), Refusal> {
    const UNPLANNED_PREFILL: Refusal = Refusal::Unstated {
        what: "a planned FA2 prefill cache",
    };

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
